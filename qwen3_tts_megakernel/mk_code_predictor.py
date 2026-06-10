"""Megakernel as the Qwen3-TTS *code predictor* decode backend.

The code predictor (the "codebook generator", 5 layers) was scoped out of the
original task, but it dominates end-to-end runtime (67-95%, see README §5) —
so this ports it onto the same megakernel. Its transformer block is
*dimensionally identical* to the talker / Qwen3-0.6B block the kernel
hardcodes: hidden 1024, 16 Q / 8 KV heads x 128, intermediate 3072, QK-norm,
rms_eps 1e-6. Differences, all absorbed host-side:

  - 5 layers (the kernel takes num_layers at runtime)
  - vocab 2048                          -> second build, -DLDG_VOCAB_SIZE=2048
  - standard 1D RoPE theta=1e6, no mRoPE -> static cos/sin tables, filled once
  - per-group embeddings + lm heads (15 each) -> applied host-side per sub-step
  - per-frame sequences are tiny (2-token prefill + 14 decode steps) and the
    KV cache restarts every frame -> we just overwrite positions 0..16
    (attention at position p only reads cache slots <= p, so no reset needed).

Per talker frame, HF runs `code_predictor.generate(...)` — a full
GenerationMixin call (~69 ms/frame, ~4.6 ms per 5-layer sub-step: nearly all
Python/framework overhead). We monkeypatch that instance method with a flat
loop over megakernel steps; the caller only consumes `.sequences`, so the
integration surface is one attribute. Sampling replicates HF's warper stack
for the passed args (temperature -> top-k -> top-p -> multinomial).
"""
import types
from types import SimpleNamespace

import torch

from mk_talker import build_ext

# ---- code predictor dims (block shapes == Qwen3-0.6B == talker) ----
NUM_LAYERS = 5
HIDDEN = 1024
NUM_KV = 8
NUM_Q = 16
HEAD_DIM = 128
Q_SIZE = NUM_Q * HEAD_DIM
KV_SIZE = NUM_KV * HEAD_DIM
INTER = 3072
VOCAB = 2048                   # codec codebook vocab
MAX_SEQ = 64                   # per frame: 2 prefill + 14 decode = 16 used


class MegakernelCodePredictorEngine:
    """Runs the code predictor's 5-layer transformer + final norm on the
    megakernel. One step = one sub-token position; heads/embeddings are
    applied by the caller (they differ per codebook group)."""

    def __init__(self, code_predictor, max_seq=MAX_SEQ):
        self.ext = build_ext("qwen_megakernel_cp_C", VOCAB)
        self._decode = torch.ops.qwen_megakernel_cp_C.decode
        self.max_seq = max_seq
        dev = "cuda"
        import struct
        sd = code_predictor.model.state_dict()

        def g(name):
            return sd[name].to(torch.bfloat16).contiguous().cuda()

        lw = []
        for i in range(NUM_LAYERS):
            p = f"layers.{i}."
            lw += [
                g(p + "input_layernorm.weight"),
                g(p + "self_attn.q_proj.weight"),
                g(p + "self_attn.k_proj.weight"),
                g(p + "self_attn.v_proj.weight"),
                g(p + "self_attn.q_norm.weight"),
                g(p + "self_attn.k_norm.weight"),
                g(p + "self_attn.o_proj.weight"),
                g(p + "post_attention_layernorm.weight"),
                g(p + "mlp.gate_proj.weight"),
                g(p + "mlp.up_proj.weight"),
                g(p + "mlp.down_proj.weight"),
            ]
        self._layer_tensors = lw                      # keep alive
        self._final_norm = g("norm.weight")

        buf = bytearray(NUM_LAYERS * 11 * 8)
        for idx, t in enumerate(lw):
            struct.pack_into("Q", buf, idx * 8, t.data_ptr())
        self._packed = torch.frombuffer(bytes(buf), dtype=torch.uint8).cuda()

        # per-group heads [15 x (2048,1024)] and embeddings [15 x (2048,1024)]
        self.heads = [h.weight.to(torch.bfloat16).contiguous().cuda()
                      for h in code_predictor.lm_head]
        self.embeds = [e.weight.to(torch.bfloat16).contiguous().cuda()
                       for e in code_predictor.model.codec_embedding]

        # static rope tables: standard 1D rope, filled ONCE from HF's exact
        # rotary implementation (theta=1e6, no mRoPE here).
        rot = code_predictor.model.rotary_emb
        pos = torch.arange(max_seq, device=dev).unsqueeze(0)         # [1,seq]
        dummy = torch.zeros(1, max_seq, HIDDEN, device=dev, dtype=torch.bfloat16)
        cos, sin = rot(dummy, pos)                                   # [1,seq,hd]
        self.cos_table = cos[0].to(torch.bfloat16).contiguous()
        self.sin_table = sin[0].to(torch.bfloat16).contiguous()

        # 1-row embed table (row 0 = current input embedding)
        self._embed = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=dev)

        # KV cache (restarts implicitly every frame by overwriting pos 0..)
        self._k = torch.zeros(NUM_LAYERS, NUM_KV, max_seq, HEAD_DIM,
                              dtype=torch.bfloat16, device=dev)
        self._v = torch.zeros_like(self._k)

        f32 = dict(dtype=torch.float32, device=dev)
        bf16 = dict(dtype=torch.bfloat16, device=dev)
        self._hidden = torch.empty(HIDDEN, **bf16)
        self._act = torch.empty(HIDDEN, **f32)
        self._res = torch.empty(HIDDEN, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._kbuf = torch.empty(KV_SIZE, **f32)
        self._vbuf = torch.empty(KV_SIZE, **f32)
        self._attn = torch.empty(Q_SIZE, **f32)
        self._mlp = torch.empty(INTER, **f32)
        self._norm_out = torch.empty(HIDDEN, **f32)
        self._bmax_v = torch.empty(4096, **f32)
        self._bmax_i = torch.empty(4096, dtype=torch.int32, device=dev)
        self._out_tok = torch.empty(1, dtype=torch.int32, device=dev)
        self._scale = 1.0 / (HEAD_DIM ** 0.5)

    def step(self, emb_row_bf16, position, head_idx=0):
        """One transformer step at cache `position`. Returns the post-final-
        norm hidden [HIDDEN] (f32 view; copy before the next step). The
        in-kernel head argmax (over heads[head_idx]) lands in _out_tok."""
        self._embed[0].copy_(emb_row_bf16)
        self._decode(
            self._out_tok, 0, self._embed, self._packed, self._final_norm,
            self.heads[head_idx], self.cos_table, self.sin_table,
            self._k, self._v,
            self._hidden, self._act, self._res, self._q, self._kbuf,
            self._vbuf, self._attn, self._mlp, self._norm_out,
            self._bmax_v, self._bmax_i,
            NUM_LAYERS, position, self.max_seq, self._scale,
        )
        return self._norm_out


def _sample(logits, do_sample, top_k, top_p, temperature):
    """HF warper stack for the args the talker passes: temperature -> top-k
    -> top-p -> softmax -> multinomial. logits: [1, vocab]."""
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    if temperature and temperature != 1.0:
        logits = logits / temperature
    if top_k and 0 < top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        # as HF TopPLogitsWarper: sort ascending, drop the <= (1-p) tail
        sl, si = torch.sort(logits, descending=False)
        remove = sl.softmax(-1).cumsum(-1) <= (1.0 - top_p)
        remove[..., -1] = False                       # keep >= 1 token
        logits = logits.masked_fill(
            remove.scatter(-1, si, remove), float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def install_megakernel_code_predictor(model):
    """Monkeypatch `model.talker.code_predictor.generate` (instance attribute)
    with a flat megakernel sub-step loop. Returns the engine."""
    cp = model.talker.code_predictor
    eng = MegakernelCodePredictorEngine(cp)
    cp._mk_engine = eng
    proj = cp.small_to_mtp_projection   # Identity for the 0.6B model

    def fast_generate(self, inputs_embeds=None, max_new_tokens=None,
                      do_sample=True, top_p=1.0, top_k=50, temperature=0.9,
                      **kw):
        e = self._mk_engine
        n_new = max_new_tokens or (self.config.num_code_groups - 1)   # 15
        emb = proj(inputs_embeds)[0].to(torch.bfloat16)               # [2,1024]
        h = None
        for t in range(emb.shape[0]):                                 # prefill
            h = e.step(emb[t], t)
        pos = emb.shape[0]
        toks = []
        for k in range(n_new):
            # group k+1 token: head k on the current hidden
            logits = torch.nn.functional.linear(
                h.to(torch.bfloat16), e.heads[k]).float().unsqueeze(0)
            tok = _sample(logits, do_sample, top_k, top_p, temperature)  # [1,1]
            toks.append(tok)
            if k + 1 < n_new:
                ev = torch.nn.functional.embedding(tok[0], e.embeds[k])[0]
                h = e.step(ev.to(torch.bfloat16), pos)
                pos += 1
        return SimpleNamespace(sequences=torch.cat(toks, dim=-1))    # [1,15]

    cp.generate = types.MethodType(fast_generate, cp)
    return eng
