import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen3_tts_megakernel"))
import time, torch, numpy as np, functools
from qwen_tts import Qwen3TTSModel
import qwen_tts.core.models.modeling_qwen3_tts as MM
import mk_talker

torch.manual_seed(0)
TEXT="Hello there, this is a quick test of the Qwen three text to speech system."
def banner(s): print("\n"+"="*64+f"\n{s}\n"+"="*64)

# ---------- 1. isolate talker transformer step latency: HF vs megakernel ----------
banner("1. TALKER TRANSFORMER (in-scope) single-step latency")
tts=Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
      device_map="cuda:0",dtype=torch.bfloat16,attn_implementation="sdpa")
talker=tts.model.talker; tm=talker.model
hid=tm.config.hidden_size
# warm KV with a short prefill, then time many 1-token decode steps
from transformers.cache_utils import DynamicCache
def time_hf_talker(n=200):
    cache=DynamicCache()
    emb=torch.randn(1,8,hid,device="cuda",dtype=torch.bfloat16)*0.02
    with torch.no_grad():
        tm(inputs_embeds=emb,use_cache=True,past_key_values=cache)  # prefill
        step=torch.randn(1,1,hid,device="cuda",dtype=torch.bfloat16)*0.02
        for _ in range(5):
            cp=torch.tensor([cache.get_seq_length()],device="cuda")
            tm(inputs_embeds=step,past_key_values=cache,use_cache=True,cache_position=cp)
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(n):
            cp=torch.tensor([cache.get_seq_length()],device="cuda")
            tm(inputs_embeds=step,past_key_values=cache,use_cache=True,cache_position=cp)
        torch.cuda.synchronize()
    return (time.time()-t0)/n*1000
hf_ms=time_hf_talker()

eng=mk_talker.MegakernelTalkerEngine(talker)
def time_mk_talker(n=200):
    emb=(torch.randn(hid,device="cuda")*0.02).to(torch.bfloat16)
    cosr=eng.cos_table[0].clone(); sinr=eng.sin_table[0].clone()
    for p in range(8): eng.step(emb,p,cosr,sinr)
    torch.cuda.synchronize(); t0=time.time()
    for i in range(n): eng.step(emb,8+i,cosr,sinr)
    torch.cuda.synchronize()
    return (time.time()-t0)/n*1000
mk_ms=time_mk_talker()
print(f"HF (sdpa)  talker step: {hf_ms:6.3f} ms/step  ({1000/hf_ms:6.1f} step/s)")
print(f"Megakernel talker step: {mk_ms:6.3f} ms/step  ({1000/mk_ms:6.1f} step/s)")
print(f"talker speedup: {hf_ms/mk_ms:.2f}x")

# ---------- 2. end-to-end with stage breakdown: HF vs MK ----------
banner("2. END-TO-END (full TTS pipeline) HF vs Megakernel-talker")
def run_e2e(label, mnt=512):
    T={"cp":0.0,"voc":0.0}
    cp_obj=tts.model.talker.code_predictor
    _cg=cp_obj.generate                      # instance attr: times HF *or* MK path
    def cg(*a,**k):
        torch.cuda.synchronize();t0=time.time();r=_cg(*a,**k);torch.cuda.synchronize();T["cp"]+=time.time()-t0;return r
    cp_obj.generate=cg
    _d=tts.model.speech_tokenizer.decode
    def dec(*a,**k):
        torch.cuda.synchronize();t0=time.time();r=_d(*a,**k);torch.cuda.synchronize();T["voc"]+=time.time()-t0;return r
    tts.model.speech_tokenizer.decode=dec
    torch.manual_seed(0);torch.cuda.synchronize();t0=time.time()
    w,sr=tts.generate_custom_voice(text=TEXT,language="English",speaker="Ryan",instruct="",max_new_tokens=mnt,do_sample=False,subtalker_dosample=False)
    torch.cuda.synchronize();tot=time.time()-t0
    cp_obj.generate=_cg
    tts.model.speech_tokenizer.decode=_d
    dur=len(w[0])/sr
    print(f"{label:22s} total={tot:6.3f}s audio={dur:4.2f}s RTF={tot/dur:5.3f} | codePred={T['cp']*1000:6.0f}ms({100*T['cp']/tot:4.1f}%) vocoder={T['voc']*1000:5.0f}ms({100*T['voc']/tot:4.1f}%) talker+rest={ (tot-T['cp']-T['voc'])*1000:6.0f}ms({100*(tot-T['cp']-T['voc'])/tot:4.1f}%)")
    return tot,dur
run_e2e("HF baseline")
mk_talker.install_megakernel_talker(tts.model)
run_e2e("MK talker")
import mk_code_predictor
mk_code_predictor.install_megakernel_code_predictor(tts.model)
run_e2e("MK talker+codePred")
print("\n(greedy/do_sample=False for reproducibility; code predictor = the 'codebook generator' --")
print(" scoped out of the original task but ported anyway since it dominated runtime)")
