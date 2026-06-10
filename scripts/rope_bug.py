# Demonstrate: the upstream megakernel builds RoPE tables with base 10000,
# but Qwen3-0.6B uses rope_theta=1000000. Fixing it makes MK match HF greedy.
import torch
from qwen_megakernel.model import Decoder, load_weights, HEAD_DIM, MAX_SEQ_LEN
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_tables(theta):
    inv=1.0/(theta**(torch.arange(0,HEAD_DIM,2,dtype=torch.float32)/HEAD_DIM))
    pos=torch.arange(MAX_SEQ_LEN,dtype=torch.float32)
    f=torch.outer(pos,inv)
    cos=torch.cos(f).repeat(1,2).to(torch.bfloat16).cuda().contiguous()
    sin=torch.sin(f).repeat(1,2).to(torch.bfloat16).cuda().contiguous()
    return cos,sin

tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
hf=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",dtype=torch.bfloat16,device_map="cuda").eval()
ids=tok("The quick brown fox jumps over",return_tensors="pt").input_ids.cuda()
N=12
with torch.no_grad():
    out=hf.generate(ids,max_new_tokens=N,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
hf_ids=out[0,-N:].tolist()

w,_=load_weights("Qwen/Qwen3-0.6B",verbose=False)
def run(theta):
    dec=Decoder(weights=w,tokenizer=tok,verbose=False)
    dec._cos_table,dec._sin_table=build_tables(theta)
    dec.reset()
    p=ids[0].tolist()
    for t in p[:-1]: dec.step(t)
    o=[];tk=p[-1]
    for _ in range(N): tk=dec.step(tk);o.append(tk)
    return o
mk_10k=run(10000.0)
mk_1m =run(1000000.0)
def match(a,b): return sum(x==y for x,y in zip(a,b))
print("HF greedy        :",hf_ids)
print("MK theta=10000   :",mk_10k, f"-> {match(hf_ids,mk_10k)}/{N} match (UPSTREAM)")
print("MK theta=1000000 :",mk_1m,  f"-> {match(hf_ids,mk_1m)}/{N} match (FIXED)")
