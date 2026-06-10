"""Parity test: megakernel code predictor vs HF, on HF's real inputs.

Captures the exact `inputs_embeds` of every `code_predictor.generate` call
during a short HF generation, then replays each through (a) HF generate and
(b) the megakernel fast path, both greedy, and compares the 15-token group
sequences. Greedy + identical inputs => sequences should match (modulo rare
sub-logit f32-vs-bf16 flips, reported honestly).
"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen3_tts_megakernel"))
import torch
from qwen_tts import Qwen3TTSModel
import mk_code_predictor

tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
cp = tts.model.talker.code_predictor

# ---- capture real generate() inputs during a normal HF run ----
captured = []
_orig_gen = cp.generate          # bound GenerationMixin.generate

def capture(inputs_embeds=None, **kw):
    if len(captured) < 12:
        captured.append(inputs_embeds.detach().clone())
    return _orig_gen(inputs_embeds=inputs_embeds, **kw)

cp.generate = capture
torch.manual_seed(0)
tts.generate_custom_voice(text="Hello there, nice to meet you.",
                          language="English", speaker="Ryan", instruct="",
                          max_new_tokens=12, do_sample=False,
                          subtalker_dosample=False)
cp.generate = _orig_gen
print(f"captured {len(captured)} code_predictor.generate calls")

# ---- replay: HF greedy vs megakernel greedy on identical inputs ----
eng = mk_code_predictor.install_megakernel_code_predictor(tts.model)
fast_gen = cp.generate           # the installed fast path

n_match_seq = 0
n_match_tok = 0
n_tok = 0
print(f"\n{'call':>4} {'match':>7}  hf_sequence -> mk_sequence (first diff marked)")
for ci, ie in enumerate(captured):
    with torch.no_grad():
        hf = _orig_gen(inputs_embeds=ie, max_new_tokens=15, do_sample=False,
                       top_p=1.0, top_k=50, temperature=0.9,
                       output_hidden_states=True, return_dict_in_generate=True
                       ).sequences[0].tolist()
        mk = fast_gen(inputs_embeds=ie, max_new_tokens=15, do_sample=False,
                      top_p=1.0, top_k=50, temperature=0.9
                      ).sequences[0].tolist()
    m = sum(a == b for a, b in zip(hf, mk))
    n_match_tok += m; n_tok += len(hf)
    n_match_seq += (hf == mk)
    diff = next((i for i, (a, b) in enumerate(zip(hf, mk)) if a != b), None)
    note = "identical" if diff is None else f"first diff @ group {diff+1}"
    print(f"{ci:>4} {m:>3}/15   {note}")

print(f"\nsequences identical: {n_match_seq}/{len(captured)}   "
      f"tokens identical: {n_match_tok}/{n_tok} ({100*n_match_tok/n_tok:.1f}%)")
