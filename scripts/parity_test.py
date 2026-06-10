import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen3_tts_megakernel"))
import time, torch, functools
from qwen_tts import Qwen3TTSModel
import qwen_tts.core.models.modeling_qwen3_tts as M
import mk_talker

cap = []
_fwd = M.Qwen3TTSTalkerModel.forward
@functools.wraps(_fwd)
def hook(self, *a, **k):
    out = _fwd(self, *a, **k)
    # bind args
    import inspect
    ba = inspect.signature(_fwd).bind(self, *a, **k); ba.apply_defaults()
    ie = ba.arguments.get("inputs_embeds")
    pid = ba.arguments.get("position_ids")
    cp = ba.arguments.get("cache_position")
    if ie is not None and len(cap) < 12:
        cap.append((ie.detach().clone(), None if pid is None else pid.detach().clone(),
                    None if cp is None else cp.detach().clone(),
                    out.last_hidden_state.detach().clone()))
    return out
M.Qwen3TTSTalkerModel.forward = hook

tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
torch.manual_seed(0)
tts.generate_custom_voice(text="Hello there, nice to meet you.", language="English",
                          speaker="Ryan", instruct="", max_new_tokens=10)
M.Qwen3TTSTalkerModel.forward = _fwd  # restore
print(f"captured {len(cap)} talker.model calls; shapes: " +
      ", ".join(str(tuple(c[0].shape)) for c in cap))

# Build engine and replay in order through ONE KV cache
eng = mk_talker.MegakernelTalkerEngine(tts.model.talker)
ms = tts.model.talker.config.rope_scaling["mrope_section"]
rotary = tts.model.talker.model.rotary_emb
pos_base = 0
print(f"\n{'call':>4} {'seq':>4} {'maxabs':>10} {'cos':>8} {'argmax_match'}")
for ci,(ie,pid,cp,href) in enumerate(cap):
    bsz,seqlen,_ = ie.shape
    if cp is None:
        cp = torch.arange(pos_base, pos_base+seqlen, device=ie.device)
    if pid is None:
        pid = cp.view(1,1,-1).expand(3,bsz,-1)
    elif pid.ndim==2:
        pid = pid[None].expand(3,pid.shape[0],-1)
    if pid.shape[0]==4: pid=pid[1:]
    cos,sin = rotary(ie, pid)
    cf,sf = mk_talker.mrope_interleave(cos,sin,ms)
    eb = ie[0].to(torch.bfloat16); cb=cf[0].to(torch.bfloat16); sb=sf[0].to(torch.bfloat16)
    hmk=[]
    for t in range(seqlen):
        idx=int(cp[t].item())
        h=eng.step(eb[t],idx,cb[t],sb[t]); hmk.append(h.clone())
    hmk=torch.stack(hmk,0).unsqueeze(0).float()
    hr=href.float()
    maxabs=(hmk-hr).abs().max().item()
    cos_sim=torch.nn.functional.cosine_similarity(hmk.flatten(),hr.flatten(),dim=0).item()
    # codec_head argmax agreement on last token
    chw = tts.model.talker.codec_head.weight.float()
    amk=(hmk[0,-1]@chw.T).argmax().item(); arf=(hr[0,-1]@chw.T).argmax().item()
    print(f"{ci:>4} {seqlen:>4} {maxabs:>10.4f} {cos_sim:>8.5f} {amk==arf} ({amk} vs {arf})")
    pos_base = int(cp[-1].item())+1
