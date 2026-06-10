import asyncio, time, numpy as np, soundfile as sf
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
import sys,os; sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","qwen3_tts_megakernel"))
from pc_tts import MegakernelQwen3TTSService
from mk_tts import SR

async def main():
    svc = MegakernelQwen3TTSService(do_sample=True, chunk_frames=2, lookback=8)
    text = "Hello! I'm a voice agent running on a megakernel powered text to speech system."
    # warmup
    async for _ in svc.run_tts("warm up", "w"): pass
    t0=time.time(); times=[]; audio=[]; started=stopped=0
    async for f in svc.run_tts(text, "ctx1"):
        if isinstance(f, TTSStartedFrame): started+=1
        elif isinstance(f, TTSStoppedFrame): stopped+=1
        elif isinstance(f, TTSAudioRawFrame):
            times.append(time.time()-t0)
            audio.append(np.frombuffer(f.audio, dtype=np.int16))
    a=np.concatenate(audio).astype(np.float32)/32767.0
    dur=len(a)/SR; total=time.time()-t0
    print(f"started={started} stopped={stopped} n_audio_frames={len(times)}")
    print(f"TTFC={times[0]*1000:.1f}ms  last_chunk@{times[-1]*1000:.0f}ms  total={total:.3f}s audio={dur:.2f}s RTF={total/dur:.3f}")
    deltas=np.diff(times)*1000
    print(f"inter-chunk gap: mean={deltas.mean():.0f}ms max={deltas.max():.0f}ms (frames arrive incrementally => streaming)")
    sf.write("pc_out.wav", a, SR)
    print("wrote pc_out.wav  rms=%.4f"%np.sqrt((a**2).mean()))

asyncio.run(main())
