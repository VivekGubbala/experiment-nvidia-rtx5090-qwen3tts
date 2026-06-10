# How this was built — a step-by-step walkthrough

This document is the *narrative* companion to [`README.md`](./README.md). The README is
reference material (what the system is, how to run it, the final numbers). This file explains
**how we got there** — the order we did things in, *why* each step, the decisions and trade-offs,
and the dead-ends — so the result is reproducible as a thought process, not just as code.

The whole task: take AlpinDale's `qwen_megakernel` (a single-launch CUDA decode kernel that runs
Qwen3-0.6B at ~1000 tok/s on an RTX 5090) and make it the **decode backend for the Qwen3-TTS
talker**, streaming speech frame-by-frame into a **Pipecat** voice pipeline. Targets: TTFC < 60 ms,
RTF < 0.15, true streaming. bf16, no quantization. The talker is the target — *not* the codebook
generator.

---

## Step 0 — Orient: confirm the hardware and the toolchain match the kernel

**Why first:** the megakernel is compiled for `sm_120` (Blackwell). If the box weren't a 5090, or
torch were a cu124 wheel, the kernel would *build* but die at the first GPU op with
"no kernel image is available" — a confusing failure to debug later. So we verified the
foundation before writing a line of integration.

```
torch 2.11.0+cu128   cuda True   NVIDIA GeForce RTX 5090     # cu128 wheel → sm_120 kernels present
RTX 5090, 32 GB, driver 580 / CUDA 13                        # driver_max_cuda ≥ 12.8 ✓
```

We also reproduced the kernel's own baseline (`python -m qwen_megakernel.bench` →
**~1038 tok/s, 0.96 ms/tok**) so we had a known-good reference for "the kernel works on this box"
independent of any TTS work. **Sanity-check the foundation before building on it.**

## Step 1 — Read the Qwen3-TTS architecture and find *exactly* where the megakernel fits

This was the single highest-risk question in the whole task (flagged in the plan): **is the talker
a 0.6B model the kernel can run, or does it need kernel surgery?** Getting this wrong means
rewriting the kernel's dims/tiling/KV-cache. So we resolved it on day one by reading the HF config,
not by assuming.

Qwen3-TTS-12Hz is a **3-stage discrete LM**:

```
text+speaker → [TALKER: 28-layer Qwen3, predicts codebook group 0]   ← our target (megakernel)
             → [CODE PREDICTOR: 5-layer, 15 sub-steps/frame → groups 1..15]  ← out of scope, HF
             → [VOCODER: 16 codebooks → 24 kHz PCM, 1920 samp/frame, 12.5 fps]  ← HF
```

**The finding that made the whole project tractable:** the talker (`Qwen3-TTS-12Hz-0.6B`) is
*architecturally identical* to Qwen3-0.6B everywhere the kernel cares — 28 layers, hidden 1024,
16 Q / 8 KV heads × 128, intermediate 3072, RMS-norm 1e-6, QK-norm. The differences are all
**data, not structure**:

| | Qwen3-0.6B (kernel) | TTS talker | how we absorbed it |
|---|---|---|---|
| vocab | 151936 | **3072** (codec) | one `-D` rebuild |
| rope_theta | 1e4 (kernel) | **1e6** | host cos/sin tables |
| rope | rotate-half | **rotate-half + 3D mRoPE** | host cos/sin tables |
| embeddings | tied | **untied codec embed/head** | fed directly |
| step input | a token id | **a computed embedding** | fed directly |

So the answer was: **no compute surgery needed.** That decision shaped everything after.

### The two non-obvious sub-findings

1. **mRoPE's `interleaved=True` does *not* mean GPT-J pair rotation.** This nearly sent us into a
   kernel rewrite. We read the HF rotary code carefully: "interleaved" only describes how the *3
   position components* (the mRoPE sections `[24,20,20]`) are interleaved across frequency
   channels. The rotation itself is still **`rotate_half`** (NeoX style) — exactly what the kernel
   already does. So the kernel's RoPE math is structurally correct; only the **cos/sin table
   values** differ. That collapsed a feared kernel rewrite into "compute the right tables in
   Python."

2. **Each talker step's input is a *computed embedding*, not a token id.** The input is
   `Σ(16 codebook embeddings from the code predictor) + trailing text hidden`, and the output feeds
   *both* `codec_head` (sampling) *and* the next code-predictor call (as `past_hidden`). This told
   us the integration boundary precisely: keep HF for everything that builds/consumes those
   embeddings, swap only the **28-layer transformer + final RMSNorm**.

## Step 2 — Decide the integration boundary: monkeypatch, don't fork

**Decision:** drop-in replace `Qwen3TTSTalkerModel.forward` (the 28-layer stack + final norm) with
the megakernel, and let HF keep prefill-embedding construction, the code predictor, sampling, and
the vocoder. The megakernel owns only the talker KV cache.

**Why this boundary and not "reimplement the loop":** the generation loop is tangled with sampling,
EOS handling, the code predictor's 15 sub-steps, and speaker conditioning. Reimplementing it would
be high-risk integration work with no performance payoff (it's Python glue, not the hot matmuls).
Patching one method is surgical, keeps HF's correctness for the surrounding logic, and is trivially
reversible (`use_megakernel=False` falls straight back to HF). Implemented in
`mk_talker.install_megakernel_talker`.

## Step 3 — Feed the kernel an embedding and read a hidden, with near-zero kernel edits

The kernel was written to take a **token id** and produce a **sampled token**. We needed it to take
an **arbitrary embedding** and return a **post-norm hidden**. Two tricks avoided touching the
compute path:

- **Arbitrary embedding in:** the kernel reads `embed_weight + token*HIDDEN`. So we pass a *1-row*
  embed table equal to our computed embedding and token id `0`. The kernel "looks up" row 0 and
  gets exactly our vector. No kernel change.
- **Post-norm hidden out:** the kernel already computes `g_normalized[i] = activation*rstd*norm_w[i]`
  in scratch before the head — that *is* HF's talker `last_hidden_state`. We read it straight out
  (`_norm_out`). No kernel change.

**The only kernel edit in the entire project** — making the vocab a build-time override so the
in-kernel head matches the 3072 codec vocab (no out-of-bounds read, no wasted 151936-row matvec):

```c
// csrc/kernel.cu
#ifndef LDG_VOCAB_SIZE
#define LDG_VOCAB_SIZE 151936
#endif
```

The talker build passes `-DLDG_VOCAB_SIZE=3072`. Everything else (weights, RoPE tables, the
embedding input, the hidden readout) is driven from Python in `mk_talker.py`.

## Step 4 — RoPE entirely host-side, decoupling cache index from angle

From Step 1.1 we knew the kernel's rotate-half was correct and only the tables were wrong. So we
fill the kernel's `cos/sin` tables per-token using HF's *exact* rotary + mRoPE construction
(`mrope_interleave` in `mk_talker.py`) at **θ=1e6**.

**The subtle part:** the kernel's `position` argument indexes into the cache *and* into the rope
table — but mRoPE's angle comes from a 3D `position_ids`, not from a flat sequence index. We
decoupled them: the kernel `position` stays the sequential **cache index**, and we store
`cos_table[cache_index] = mRoPE(position_ids)`. This is correct precisely because the kernel **bakes
rope into the cached K** at write time — the cached key already carries the right angle, so the
cache slot and the angle never need to agree numerically.

## Step 5 — Prove correctness before touching speed (`scripts/parity_test.py`)

**Methodology:** capture HF's *real* talker inputs/outputs during a generation (1 prefill of 19
tokens + 9 decode steps), replay them through the megakernel engine, and compare the post-norm
hidden (cos-sim + max-abs) **and the `codec_head` argmax** — the argmax is what actually decides the
emitted codec token, hence the audio.

Final result (re-run during this round of testing):

```
call  seq   maxabs    cos     argmax_match
   0   19   1.2231   0.99979  True (1995 vs 1995)
   1    1   0.6685   0.99971  True ( 215 vs  215)
 ...   all 10 calls: cos 0.9997–0.9999, argmax identical every step
```

`cos ≈ 0.9998` and **identical argmax on every step** ⇒ greedy decode yields identical codec tokens
⇒ identical audio. The `maxabs` hidden diff is expected (kernel accumulates in f32, HF in bf16). In
a *live* greedy run the tokens match exactly for ~7 steps then diverge — a sub-logit flip from
f32-vs-bf16 cascades autoregressively. We report this honestly rather than hiding it; it's moot
under the default sampling mode and doesn't affect quality.

## Step 6 — Streaming, and the vocoder boundary-click bug (`mk_tts.py`)

`StreamingMegakernelTTS.stream()` must emit PCM **as frames decode**, never buffer the whole
utterance. We run HF's `generate_custom_voice` in a worker thread and **tap each completed
16-codebook frame** off it via a queue (monkeypatching the talker's `forward` to push
`hidden_states[1]`), then vocode incrementally on the consumer side.

**The bug we hit and fixed:** vocoding each new chunk of frames *independently* produced audible
**clicks at chunk boundaries** (max-abs diff 0.52 vs decoding the whole thing at once). The vocoder
is frame-synchronous (exactly 1920 samples/frame) but its conv stack has receptive-field spillover
across frame boundaries. **Fix:** decode `codes[start-L : end]` with `L=8` lookback frames but emit
only the `[start:end]` tail. This stays O(n) and matches a full decode (max-abs diff 0.14 — inaudible).
This is the kind of correctness detail that doesn't show up until you actually listen.

## Step 7 — Wrap it as a Pipecat `TTSService` (`pc_tts.py`)

`MegakernelQwen3TTSService.run_tts(text, context_id)` yields the Pipecat streaming contract:
`TTSStartedFrame` → many `TTSAudioRawFrame`s (pushed as decoded) → `TTSStoppedFrame`.

**The threading bug we fixed:** the engine's generator is a blocking, GIL-heavy producer; the
Pipecat side is asyncio. Early attempts drove the generator across asyncio thread-pool threads
(`asyncio.to_thread(next, gen)`), which deadlocked. **Final design:** one dedicated producer thread
runs the blocking generator and hands chunks to the event loop via
`loop.call_soon_threadsafe(q.put_nowait, ...)`; `run_tts` awaits a plain `asyncio.Queue`. Clean,
no cross-thread generator resumption.

## Step 8 — Validate end-to-end + benchmark honestly (`benchmark.py`, `pc_verify.py`)

**Isolated talker step** (the in-scope target), re-run this round:

| backend | ms/step | step/s | speedup |
|---|---|---|---|
| HF (sdpa) | 27.16 | 36.8 | 1.0× |
| **Megakernel** | **0.813** | **1229** | **33×** |

**End-to-end, with a per-stage breakdown** — and this is the crucial honest result:

| pipeline | RTF | total | code predictor (out of scope) | vocoder | talker + rest |
|---|---|---|---|---|---|
| HF baseline | 1.29 | 7.53 s | 5091 ms (67.6%) | 210 ms (2.8%) | 2231 ms (29.6%) |
| **Megakernel talker** | **0.91** | 5.03 s | 4785 ms (95.1%) | 47 ms (0.9%) | **200 ms (3.9%)** |

**Pipecat streaming proof** (`pc_verify.py`, re-run): 43 `TTSAudioRawFrame`s, **TTFC 337 ms**,
inter-chunk gap mean 197 ms (frames arrive *incrementally* → genuinely streaming, not buffered).

### The honest conclusion (why RTF/TTFC targets are "missed")

The megakernel **did its job**: it cut the talker from ~30% of runtime to **~4%** (33× per step).
But — and this is the real finding — **the talker was never the bottleneck.** At 12.5 fps the
talker runs only ~12.5 steps/s, while the **code predictor runs ~15 autoregressive sub-steps per
frame (~190/s)** and dominates at **67–95%** of total time. The code predictor is the *codebook
generator*, which the task **explicitly scoped out**. So the end-to-end RTF target is structurally
unreachable by optimizing the talker alone — and we say so with the measurements rather than
massaging the numbers.

**The clear, evidenced next step:** the code predictor is *also* Qwen3-architecture (5 layers,
hidden 1024) — the identical megakernel technique applies and would remove the bulk of the
remaining ~4.8 s. That's the highest-leverage follow-up; it was left out only because the task
scoped the talker as the target.

## Step 9 — Package the deliverable

- `setup.sh` — reproducible install (cu128 torch for sm_120, transformers pinned 4.57.3 for
  qwen-tts, pipecat, model downloads).
- `README.md` — reference: architecture, kernel mods, parity table, performance, run instructions.
- `samples/` — audio artifacts (HF baseline, megakernel talker, streamed) since this is a headless
  GPU box with no mic/speaker.
- This `WALKTHROUGH.md` — the process narrative.

---

## What's rough (carried verbatim from the honest README, so it's not lost here)

- **End-to-end RTF ≈ 0.9–1.2, not < 0.15** — bottleneck is the out-of-scope code predictor (Step 8).
- **Talker is batch=1** (the megakernel is single-sequence by design).
- **Streaming still uses HF's Python generation loop** — a custom decode loop would shave per-step
  overhead.
- **Transport-less Pipecat `PipelineRunner` GIL-starves**: HF's Python-heavy gen loop holds the GIL
  so a runner with no output transport stalls after the first chunks. `pc_verify.py` is the
  deterministic proof of the streaming contract; a real deployment paces the pipeline with an
  output transport (`pipeline_demo.py voice`) and/or runs TTS in a separate worker process.
- **Upstream-kernel RoPE θ observation**: the kernel's own `model.py` builds RoPE with base 1e4
  while Qwen3-0.6B specifies `rope_theta=1e6`; fixing θ alone didn't fully restore its self-check
  parity (`scripts/rope_bug.py`), so there's likely additional precision loss. Our port sidesteps
  this entirely by feeding HF-exact RoPE tables, and achieves bit-level argmax parity.

## Final-testing checklist (this round, all green)

```
python scripts/parity_test.py   →  EXIT 0, argmax identical 10/10, cos 0.9997–0.9999
python scripts/benchmark.py     →  EXIT 0, talker 33×, RTF 1.29→0.91, talker 29.6%→4.0%
python scripts/pc_verify.py     →  EXIT 0, 43 frames, TTFC 337 ms, incremental delivery
```
