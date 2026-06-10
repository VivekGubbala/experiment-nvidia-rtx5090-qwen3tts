"""Megakernel as the Qwen3-TTS talker decode backend.

The Qwen3-TTS 0.6B *talker* is architecturally identical to Qwen3-0.6B
(28 layers, hidden 1024, 16 Q / 8 KV heads x 128, intermediate 3072,
rms_eps 1e-6) except:
  - vocab is the codec vocab (3072) instead of 151936  -> rebuild kernel
  - rope_theta 1e6 (not 1e4) + 3D mRoPE              -> baked into cos/sin tables
  - codec embed / codec_head are untied              -> we feed embeddings directly
  - the talker input each step is a *computed embedding* (sum of 16 codebook
    embeddings + trailing text hidden), produced by HF; the megakernel only
    runs the 28-layer transformer + final norm and returns the post-norm hidden.

We therefore use the AlpinDale megakernel as a drop-in replacement for
`Qwen3TTSTalkerModel.forward` (the 28-layer transformer + final RMSNorm),
keeping HF for: prefill-embedding construction, codec_head + sampling, the
code predictor, and the vocoder. The megakernel owns the talker KV cache.

Key tricks (so kernel.cu needs only a 1-line vocab macro change):
  * feed an arbitrary embedding by passing a 1-row embed table = inputs_embeds
    with token id 0 (kernel reads embed_weight + token*HIDDEN).
  * read the post-final-norm hidden straight from the kernel's `normalized`
    (g_normalized) scratch buffer -- exactly HF's talker last_hidden_state.
  * cache index (kernel `position`) is sequential; we fill cos_table[idx] with
    the mRoPE rope value for that token, decoupling cache index from rope angle.
"""
import os
import struct
import torch
from torch.utils.cpp_extension import load
from transformers.modeling_outputs import BaseModelOutputWithPast

# ---- talker dims (== Qwen3-0.6B block shapes) ----
NUM_LAYERS = 28
HIDDEN = 1024
NUM_KV = 8
NUM_Q = 16
HEAD_DIM = 128
Q_SIZE = NUM_Q * HEAD_DIM      # 2048
KV_SIZE = NUM_KV * HEAD_DIM    # 1024
INTER = 3072
VOCAB = 3072                   # codec vocab (codebook 2048 + specials)
MAX_SEQ = 4096

_CSRC = "/workspace/assignment/qwen_megakernel/csrc"


def build_ext(name, vocab):
    """Compile the megakernel with a given lm-head vocab under a distinct
    module name (the op registers as torch.ops.<name>.decode)."""
    kernel_flags = [
        "-DLDG_NUM_BLOCKS=128", "-DLDG_BLOCK_SIZE=512",
        "-DLDG_LM_NUM_BLOCKS=1280", "-DLDG_LM_BLOCK_SIZE=384",
        "-DLDG_LM_ROWS_PER_WARP=2", "-DLDG_ATTN_BLOCKS=8",
        "-DLDG_PREFETCH_QK=0", "-DLDG_PREFETCH_THREAD_STRIDE=10",
        "-DLDG_PREFETCH_DOWN=1", "-DLDG_PREFETCH_ELEM_STRIDE=1",
        "-DLDG_PREFETCH_BLOCK_STRIDE=1", "-DLDG_PREFETCH_GATE=1",
        "-DLDG_PREFETCH_UP=1", "-DLDG_USE_UINT4", "-DLDG_ATTENTION_VEC4",
        "-DLDG_WEIGHT_LDCS", "-DLDG_MLP_SMEM",
        f"-DLDG_VOCAB_SIZE={vocab}",
        # fixes a start-of-launch barrier-reset race in the upstream kernel
        # (probabilistic hang under dense launch rates; see README)
        "-DLDG_HOST_BARRIER_RESET",
    ]
    flags = ["-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr",
             "-arch=sm_120a", f"-I{_CSRC}"] + kernel_flags
    return load(
        name=name,
        sources=[os.path.join(_CSRC, "torch_bindings.cpp"),
                 os.path.join(_CSRC, "kernel.cu")],
        extra_cuda_cflags=flags, extra_cflags=[f"-I{_CSRC}"], verbose=False,
    )


def build_talker_ext(vocab=VOCAB):
    """Compile the megakernel with the codec vocab; distinct module name."""
    return build_ext("qwen_megakernel_tts_C", vocab)


def mrope_interleave(cos, sin, mrope_section):
    """Replicates HF apply_multimodal_rotary_pos_emb (interleaved=True) cos/sin
    construction. cos,sin: [3, bs, seq, head_dim] -> [bs, seq, head_dim]."""
    modality_num = len(mrope_section)  # 3

    def interleave(x):  # x: [3, bs, seq, head_dim//2]
        x_t = x[0].clone()
        for i, n in enumerate(mrope_section[1:], 1):
            beg, end = i, n * modality_num
            x_t[..., beg:end:modality_num] = x[i][..., beg:end:modality_num]
        return x_t

    dim = cos.shape[-1]
    cos_f = torch.cat([interleave(cos[..., : dim // 2])] * 2, dim=-1)
    sin_f = torch.cat([interleave(sin[..., : dim // 2])] * 2, dim=-1)
    return cos_f, sin_f


class MegakernelTalkerEngine:
    """Runs the talker's 28-layer transformer + final norm on the megakernel."""

    def __init__(self, talker, max_seq=MAX_SEQ):
        self.ext = build_talker_ext(VOCAB)
        self._decode = torch.ops.qwen_megakernel_tts_C.decode
        self.max_seq = max_seq
        dev = "cuda"
        sd = talker.state_dict()

        def g(name):
            return sd[name].to(torch.bfloat16).contiguous().cuda()

        # per-layer weights, 11 tensors/layer in the kernel's expected order
        self._layer_tensors = []
        lw = []
        for i in range(NUM_LAYERS):
            p = f"model.layers.{i}."
            tens = [
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
            self._layer_tensors.extend(tens)
            lw.extend(tens)
        self._final_norm = g("model.norm.weight")
        self._codec_head = g("codec_head.weight")  # [VOCAB, HIDDEN]

        # pack 64-bit pointers into a device blob of LDGLayerWeights structs
        n_ptrs = 11
        buf = bytearray(NUM_LAYERS * n_ptrs * 8)
        for idx, t in enumerate(lw):
            struct.pack_into("Q", buf, idx * 8, t.data_ptr())
        self._packed = torch.frombuffer(bytes(buf), dtype=torch.uint8).cuda()

        # rope tables, filled per-token from HF mRoPE
        self.cos_table = torch.zeros(max_seq, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.sin_table = torch.zeros(max_seq, HEAD_DIM, dtype=torch.bfloat16, device=dev)

        # 1-row embed table (row 0 = current input embedding)
        self._embed = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=dev)

        # KV cache
        self._k = torch.zeros(NUM_LAYERS, NUM_KV, max_seq, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self._v = torch.zeros_like(self._k)

        # scratch
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
        self._norm_out = torch.empty(HIDDEN, **f32)   # <- post-final-norm hidden
        self._bmax_v = torch.empty(4096, **f32)
        self._bmax_i = torch.empty(4096, dtype=torch.int32, device=dev)
        self._out_tok = torch.empty(1, dtype=torch.int32, device=dev)
        self._scale = 1.0 / (HEAD_DIM ** 0.5)

    def reset(self):
        self._k.zero_(); self._v.zero_()

    def step(self, emb_row_bf16, position, cos_row_bf16, sin_row_bf16):
        """One transformer step. emb_row: [HIDDEN] bf16; cos/sin_row: [HEAD_DIM].
        Returns post-final-norm hidden [HIDDEN] (f32)."""
        self._embed[0].copy_(emb_row_bf16)
        self.cos_table[position].copy_(cos_row_bf16)
        self.sin_table[position].copy_(sin_row_bf16)
        self._decode(
            self._out_tok, 0, self._embed, self._packed, self._final_norm,
            self._codec_head, self.cos_table, self.sin_table, self._k, self._v,
            self._hidden, self._act, self._res, self._q, self._kbuf, self._vbuf,
            self._attn, self._mlp, self._norm_out, self._bmax_v, self._bmax_i,
            NUM_LAYERS, position, self.max_seq, self._scale,
        )
        return self._norm_out


def install_megakernel_talker(model):
    """Monkeypatch model.talker.model.forward to run on the megakernel.

    `model` is a Qwen3TTSForConditionalGeneration. Returns the engine.
    """
    talker = model.talker
    eng = MegakernelTalkerEngine(talker)
    talker_model = talker.model
    talker_model._mk_engine = eng
    mrope_section = talker.config.rope_scaling["mrope_section"]

    def patched_forward(self, input_ids=None, attention_mask=None,
                        position_ids=None, past_key_values=None,
                        inputs_embeds=None, use_cache=None,
                        output_attentions=None, output_hidden_states=None,
                        cache_position=None, **kw):
        e = self._mk_engine
        if inputs_embeds is None:
            inputs_embeds = self.codec_embedding(input_ids)
        bsz, seqlen, _ = inputs_embeds.shape
        assert bsz == 1, "megakernel talker is batch=1"
        dev = inputs_embeds.device

        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past, past + seqlen, device=dev)
        # 3D mRoPE position ids (temporal,height,width), as HF does
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, bsz, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)       # [3,bs,seq,hd]
        cos_f, sin_f = mrope_interleave(cos, sin, mrope_section)      # [bs,seq,hd]
        emb_bf16 = inputs_embeds[0].to(torch.bfloat16)               # [seq,hidden]
        cos_b = cos_f[0].to(torch.bfloat16); sin_b = sin_f[0].to(torch.bfloat16)

        hiddens = []
        for t in range(seqlen):
            idx = int(cache_position[t].item())
            h = e.step(emb_bf16[t], idx, cos_b[t], sin_b[t])
            hiddens.append(h.clone())
        hs = torch.stack(hiddens, 0).unsqueeze(0).to(inputs_embeds.dtype)

        # keep HF's cache length bookkeeping consistent (so cache_position advances)
        if past_key_values is not None:
            try:
                fake_k = inputs_embeds.new_zeros(bsz, NUM_KV, seqlen, HEAD_DIM)
                past_key_values.update(fake_k, fake_k, 0, {"cache_position": cache_position})
            except Exception:
                pass
        return BaseModelOutputWithPast(last_hidden_state=hs,
                                       hidden_states=(hs,),
                                       past_key_values=past_key_values)

    import types
    talker_model.forward = types.MethodType(patched_forward, talker_model)
    return eng
