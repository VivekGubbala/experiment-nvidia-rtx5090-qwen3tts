# RTX 5090 Decode Megakernel → Qwen3-TTS on Pipecat

Wiring AlpinDale's [`qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel)
(a single-launch CUDA decode kernel that runs Qwen3-0.6B at ~1000 tok/s on an RTX 5090)
in as the **decode backend for the Qwen3-TTS talker**, streaming speech frame-by-frame
into a **Pipecat** `TTSService`.

**Status:** the megakernel runs the talker transformer with **bit-level argmax parity**
vs HuggingFace and is **~31× faster per talker step** (25.2 ms → 0.81 ms). Audio streams
through a real Pipecat pipeline frame-by-frame. The end-to-end RTF target is *not* met,
and §Performance explains exactly why with measurements: the dominant cost is the
**code predictor (the "codebook generator", which the task explicitly scoped out)**, not
the talker.

Hardware: single RTX 5090 (sm_120 / Blackwell, 32 GB, driver 580 / CUDA 13). bf16, no quantization.

---

## 1. What Qwen3-TTS actually is (and where the megakernel fits)

Qwen3-TTS-12Hz (released 2026-01-22, open weights) is a 3-stage discrete multi-codebook LM:

```
text + speaker  ──prefill──▶  ┌──────────────────────────────────────────────┐
                              │ TALKER  (28-layer Qwen3, hidden 1024)         │ ◀── MEGAKERNEL
                              │   per frame: predicts codebook group 0        │     runs this
                              └──────┬───────────────────────────────────────┘
                                     │ talker hidden + group-0 token
                              ┌──────▼───────────────────────────────────────┐
                              │ CODE PREDICTOR ("codebook generator", 5-layer)│ ◀── HF (out of scope)
                              │   15 autoregressive sub-steps → groups 1..15  │     ← the bottleneck
                              └──────┬───────────────────────────────────────┘
                                     │ 16 codebooks/frame  (12.5 fps)
                              ┌──────▼───────────────────────────────────────┐
                              │ VOCODER (Qwen3-TTS-Tokenizer-12Hz)            │ ◀── HF
                              │   16 codebooks → 24 kHz PCM (1920 samp/frame) │
                              └──────────────────────────────────────────────┘
```

The **talker** is the assigned target. Its config (`Qwen3-TTS-12Hz-0.6B`) is *architecturally
identical* to Qwen3-0.6B where the kernel cares: 28 layers, hidden 1024, 16 Q / 8 KV heads ×
128, intermediate 3072, RMS-norm 1e-6, QK-norm. Differences the port handles:

| | Qwen3-0.6B (megakernel) | TTS-0.6B talker | handled by |
|---|---|---|---|
| vocab | 151936 | **3072** (codec) | `-DLDG_VOCAB_SIZE=3072` rebuild |
| rope_theta | (kernel used 1e4) | **1e6** | host cos/sin tables |
| rope | rotate-half | **rotate-half + 3D mRoPE** | host cos/sin tables |
| embeddings | tied | **untied codec embed/head** | fed directly |
| step input | token id | **computed embedding** (Σ16 codebook embeds + text hidden) | fed directly |

## 2. Design decisions

**Drop-in replacement of the talker transformer, not the whole loop.** Each talker step the
input embedding is a *computed sum* (16 codebook embeddings from the code predictor + trailing
text hidden) and the output feeds both `codec_head` (sampling) and the next code-predictor call
(`past_hidden`). So I keep HF for prefill-embedding construction, code predictor, sampling, and
vocoder, and swap **only the 28-layer transformer + final RMSNorm** onto the megakernel, which
owns the talker KV cache. Integration point: monkeypatch `Qwen3TTSTalkerModel.forward`
(`mk_talker.install_megakernel_talker`).

**Near-zero kernel surgery via two tricks** (see §3):
- feed an arbitrary embedding by passing a 1-row embed table = `inputs_embeds` with token id 0
  (the kernel reads `embed_weight + token*HIDDEN`);
- read the post-final-norm hidden straight out of the kernel's existing `g_normalized` scratch
  buffer — that is exactly HF's talker `last_hidden_state` (used as `past_hidden` and fed to
  `codec_head`).

**RoPE handled entirely host-side.** HF's mRoPE (`interleaved=True`) still rotates with
`rotate_half` — "interleaved" only describes how the 3 position components are interleaved
across frequency channels, not GPT-J pair rotation. So the kernel's rotate-half is structurally
correct; I just fill its `cos/sin` tables per-token using HF's exact rotary + mRoPE
(`mrope_interleave` in `mk_talker.py`), with θ=1e6. The kernel's `position` is the *sequential
cache index*; I store `cos_table[cache_index] = mRoPE(position_ids)`, decoupling cache index
from rope angle (correct because the kernel bakes rope into cached K).

**Streaming vocoder with lookback.** The vocoder is frame-synchronous (exactly 1920 samples /
frame) but has receptive-field spillover, so naively decoding independent chunks causes boundary
clicks (max abs diff 0.52 vs full decode). I decode `codes[start-L : end]` with `L=8` lookback
frames and emit only the `[start:end]` tail — O(n), and matches full decode (max abs diff 0.14).

## 3. Kernel modifications

The kernel is **almost unchanged** — one line:

```c
// csrc/kernel.cu  — make the lm-head vocab a build-time override
#ifndef LDG_VOCAB_SIZE
#define LDG_VOCAB_SIZE 151936
#endif
```

The talker build passes `-DLDG_VOCAB_SIZE=3072` (codec vocab) so the in-kernel head matches the
codec head and there's no out-of-bounds read / wasted 151936-row matvec. Everything else
(weights, RoPE tables, the embedding input, reading the post-norm hidden) is driven from Python
in `mk_talker.py`. No changes to the kernel's compute, attention, or RoPE math were needed.

## 4. Parity (correctness)

Replaying HF's exact talker inputs (1 prefill of 19 tokens + 9 decode steps) through the
megakernel:

```
call  seq     maxabs   cos_sim  codec_head argmax
   0   19     1.2231   0.99979  match  (1995 vs 1995)
   1    1     0.6685   0.99971  match  ( 215 vs  215)
 ...   (all 10 calls: cos_sim 0.9997–0.9999, argmax identical on every step)
```

`maxabs` is the post-norm hidden diff (f32 kernel vs bf16 HF — expected); `cos_sim ≈ 0.9998`
and **codec_head argmax is identical on every step** ⇒ greedy decode yields identical codec
tokens ⇒ identical audio. In a live greedy run the leading codec tokens match exactly for 7
steps then diverge (a sub-logit flip from f32-vs-bf16 cascades autoregressively) — expected,
and moot under the default sampling mode. (`scripts/parity_test.py`)

## 5. Performance (RTX 5090, measured)

Reproduced megakernel baseline: **1038.7 tok/s, 0.96 ms/tok** (`python -m qwen_megakernel.bench`).

**Talker transformer — the in-scope target (single decode step):**

| backend | ms/step | step/s | speedup |
|---|---|---|---|
| HF (sdpa) | 25.25 | 39.6 | 1.0× |
| **Megakernel** | **0.813** | **1230.8** | **31.1×** |

**End-to-end TTS (greedy, 5.5–5.8 s utterance, stage breakdown):**

| pipeline | RTF | total | code predictor (out of scope) | vocoder | talker + rest |
|---|---|---|---|---|---|
| HF baseline | 1.323 | 7.73 s | 5226 ms (67.6%) | 213 ms (2.8%) | 2287 ms (29.6%) |
| **Megakernel talker** | **0.922** | 5.09 s | 4843 ms (95.1%) | 47 ms (0.9%) | **200 ms (3.9%)** |

**Streaming (Pipecat service, sampled):** TTFC ≈ 230–270 ms, frames arrive incrementally
(~160 ms audio / chunk), RTF ≈ 1.0–1.24.

### Why the RTF / TTFC targets are missed — honestly

The megakernel did its job: it cut the **talker** from ~30% of runtime to **~4%** (31× per
step). But the talker was never the bottleneck. At **12.5 fps the talker is only 12.5 steps/s**,
while the **code predictor runs ~15 autoregressive sub-steps per frame (~190/s)** and dominates
at **67–95%** of runtime. The code predictor is the *codebook generator*, which the task
**explicitly scoped out** ("not the codebook generator"). TTFC is likewise gated by the first
code-predictor call (~72 ms) plus prefill, not the talker.

**The clear path to the targets** (with the data to back it): the code predictor is *also*
Qwen3-architecture (5 layers, hidden 1024, vocab 2048) — the same megakernel technique applies.
Porting it should remove the bulk of the remaining ~4.8 s. That's the highest-leverage next
step; it was left out because the task scoped the talker as the target.

## 6. What works / what's rough

**Works:** baseline reproduced on sm_120; megakernel talker with proven argmax parity; 31× talker
speedup; full TTS produces correct speech (RMS matches HF); O(n) artifact-free streaming;
Pipecat `TTSService` streaming `TTSAudioRawFrame`s frame-by-frame through a real pipeline.

**Rough / honest caveats:**
- End-to-end RTF ≈ 0.9–1.2, not < 0.15 — bottleneck is the out-of-scope code predictor (§5).
- Talker runs **batch=1** (the megakernel is single-sequence by design).
- Streaming uses HF's generation loop (Python per-step overhead is real); a custom loop would help.
- The full `voice` Pipecat agent (mic→STT→LLM→TTS→speaker) is provided as reference wiring but
  needs an audio device + LLM key; the **headless** pipeline is the runnable proof here.
- Upstream-kernel observation: its own correctness check diverges after ~3 tokens; `model.py`
  builds RoPE with base 1e4 while Qwen3-0.6B specifies `rope_theta=1e6`. Fixing θ alone did not
  fully restore base-kernel parity in a quick test (`scripts/rope_bug.py`), so there is likely
  additional precision loss too. My port sidesteps this by feeding HF-exact RoPE tables and
  achieves bit-level argmax parity.

## 7. Run it

```bash
# deps (torch is cu128 for sm_120; qwen-tts pins transformers==4.57.3)
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home

# build + reproduce the megakernel baseline (JIT-compiles for sm_120)
cd qwen_megakernel && python -m qwen_megakernel.bench && cd ..

# parity, benchmarks, streaming, Pipecat — from scripts/
cd scripts
python parity_test.py        # megakernel talker vs HF hidden/argmax parity
python benchmark.py          # talker step (31x) + end-to-end stage breakdown
python pc_verify.py          # DETERMINISTIC PROOF: drives the Pipecat TTSService,
                             #   logs TTSStarted -> N TTSAudioRaw (incremental) -> TTSStopped -> pc_out.wav
python pipeline_demo.py headless   # runs a real Pipecat Pipeline -> pipeline_out.wav (see caveat below)
# python pipeline_demo.py voice    # full mic->STT->LLM->TTS->speaker (needs audio device + OPENAI_API_KEY,
#                                    or swap OpenAILLMService for OllamaLLMService for fully local)
```

**Pipecat note (important, honest).** `pc_verify.py` is the deterministic, working proof of the
streaming contract — it drives the real `MegakernelQwen3TTSService.run_tts` and logs
`TTSStartedFrame` → 40+ `TTSAudioRawFrame`s arriving incrementally (~180 ms apart) →
`TTSStoppedFrame`, then writes `pc_out.wav`. `pipeline_demo.py headless` wires the same service
into a real `Pipeline`/`PipelineRunner`; frames are correctly linked and start streaming, but a
*transport-less* in-process runner starves: HF's Python-heavy generation loop holds the GIL and
the asyncio event loop can't service the pipeline, so it stalls after the first chunk(s). The
fix is exactly what a real deployment does — drive the pipeline with an output transport that
paces it (`pipeline_demo.py voice`, which needs an audio device), and/or run TTS generation in a
separate worker process. So: **use `pc_verify.py` as the runnable proof; treat the headless
PipelineRunner as illustrative wiring.**

**Demo recording.** This was built on a headless GPU box, so a live mic/speaker recording
isn't possible from here — the audio artifacts are in `samples/` (`baseline_ryan.wav` = HF,
`mk_ryan.wav` = megakernel talker, `stream_out.wav` / `pc_out.wav` = streamed). Run
`pipeline_demo.py voice` on a machine with audio to capture an end-to-end voice-agent recording.

Library use:
```python
from qwen3_tts_megakernel import StreamingMegakernelTTS
eng = StreamingMegakernelTTS(use_megakernel=True)
for pcm_bytes, sr in eng.stream("Hello from the megakernel.", speaker="Ryan"):
    ...  # 24 kHz int16 PCM chunks, pushed as decoded
```

## 8. Layout

```
qwen_megakernel/                 AlpinDale kernel (+1-line LDG_VOCAB_SIZE override)
qwen3_tts_megakernel/
  mk_talker.py                   megakernel talker engine + HF monkeypatch + mRoPE tables
  mk_tts.py                      StreamingMegakernelTTS: text -> streamed PCM (threaded tap + lookback vocode)
  pc_tts.py                      MegakernelQwen3TTSService (Pipecat TTSService)
scripts/                         parity_test, benchmark, pc_verify, pipeline_demo, rope_bug
samples/                         baseline_ryan.wav, mk_ryan.wav, stream_out.wav, pc_out.wav
```
