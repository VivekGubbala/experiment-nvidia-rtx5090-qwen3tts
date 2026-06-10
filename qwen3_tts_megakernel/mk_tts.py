"""Streaming Qwen3-TTS with the megakernel as the talker decode backend.

Exposes a simple streaming interface: text in -> PCM audio chunks out, pushed
as they are decoded (no full-utterance buffering).

Pipeline per request:
  prefill -> [talker step (MEGAKERNEL) -> code predictor (HF) -> 16-codebook
  frame] x N  -> vocoder decode (HF, chunked with lookback) -> PCM chunk emitted.

The talker's 28-layer transformer runs on the AlpinDale megakernel (see
mk_talker.py); everything else stays on HF. Generation runs in a worker thread;
each 16-codebook frame is tapped as HF produces it and the vocoder emits audio
in small chunks with lookback context to avoid boundary clicks.
"""
import threading
import queue
import time
import functools

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel
import qwen_tts.core.models.modeling_qwen3_tts as MM
import mk_talker

SR = 24000
SAMPLES_PER_FRAME = 1920          # 24000 / 12.5 fps -> 80 ms / frame
_DONE = object()


class StreamingMegakernelTTS:
    def __init__(self, model_path="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                 device="cuda:0", use_megakernel=True, attn="sdpa"):
        self.tts = Qwen3TTSModel.from_pretrained(
            model_path, device_map=device, dtype=torch.bfloat16,
            attn_implementation=attn)
        self.use_megakernel = use_megakernel
        self.engine = None
        if use_megakernel:
            self.engine = mk_talker.install_megakernel_talker(self.tts.model)
        self._decode = self.tts.model.speech_tokenizer.decode
        self._lock = threading.Lock()

    # ---- vocoder: decode frames [start:end] with `lookback` frames of context ----
    def _vocode(self, all_frames, start, end, lookback):
        ctx = max(0, start - lookback)
        codes = torch.cat(all_frames[ctx:end], dim=0)        # [n,16]
        wav, _ = self._decode([{"audio_codes": codes}])
        wav = np.asarray(wav[0], dtype=np.float32)
        off = (start - ctx) * SAMPLES_PER_FRAME
        return wav[off: off + (end - start) * SAMPLES_PER_FRAME]

    def stream(self, text, speaker="Ryan", language="English", instruct="",
               max_new_tokens=2048, chunk_frames=2, lookback=6,
               do_sample=True, seed=0):
        """Yield (pcm_int16_bytes, sr) chunks as audio is produced.

        chunk_frames: emit granularity (2 -> ~160 ms chunks).
        lookback:     vocoder context frames to avoid boundary clicks.
        """
        frames = []                      # list[Tensor[1,16]] full history
        fq: "queue.Queue" = queue.Queue()

        # tap each 16-codebook frame as HF emits it
        _tf = MM.Qwen3TTSTalkerForConditionalGeneration.forward

        @functools.wraps(_tf)
        def tap(self_, *a, **k):
            r = _tf(self_, *a, **k)
            hs = r.hidden_states
            if isinstance(hs, tuple) and len(hs) == 2 and hs[1] is not None:
                fq.put(hs[1].detach().to("cpu", non_blocking=False))
            return r

        def worker():
            try:
                if seed is not None:
                    torch.manual_seed(seed)
                MM.Qwen3TTSTalkerForConditionalGeneration.forward = tap
                self.tts.generate_custom_voice(
                    text=text, language=language, speaker=speaker,
                    instruct=instruct, max_new_tokens=max_new_tokens,
                    do_sample=do_sample, subtalker_dosample=do_sample)
            except Exception as e:           # surface errors to consumer
                fq.put(("__err__", e))
            finally:
                MM.Qwen3TTSTalkerForConditionalGeneration.forward = _tf
                fq.put(_DONE)

        with self._lock:                      # one request at a time (shared KV)
            if self.engine is not None:
                self.engine.reset()
            t = threading.Thread(target=worker, daemon=True)
            t.start()

            emitted = 0                       # frames already vocoded+sent
            while True:
                item = fq.get()
                if item is _DONE:
                    break
                if isinstance(item, tuple) and item and item[0] == "__err__":
                    raise item[1]
                frames.append(item)
                while len(frames) - emitted >= chunk_frames:
                    end = emitted + chunk_frames
                    pcm = self._vocode(frames, emitted, end, lookback)
                    emitted = end
                    yield self._to_int16(pcm), SR
            # flush remaining frames
            if len(frames) > emitted:
                pcm = self._vocode(frames, emitted, len(frames), lookback)
                yield self._to_int16(pcm), SR
            t.join()

    @staticmethod
    def _to_int16(pcm_f32):
        x = np.clip(pcm_f32, -1.0, 1.0)
        return (x * 32767.0).astype(np.int16).tobytes()


if __name__ == "__main__":
    import soundfile as sf
    eng = StreamingMegakernelTTS(use_megakernel=True)
    text = "Hello there, this is a quick test of the Qwen three text to speech system."
    # warmup (JIT / cudnn)
    for _ in eng.stream("Warm up.", max_new_tokens=32, do_sample=False):
        pass

    t0 = time.time(); ttfc = None; chunks = []
    for pcm, sr in eng.stream(text, max_new_tokens=512, do_sample=False, chunk_frames=2):
        if ttfc is None:
            ttfc = time.time() - t0
        chunks.append(np.frombuffer(pcm, dtype=np.int16))
    total = time.time() - t0
    audio = np.concatenate(chunks).astype(np.float32) / 32767.0
    dur = len(audio) / SR
    print(f"TTFC={ttfc*1000:.1f} ms  total={total:.3f}s  audio={dur:.2f}s  RTF={total/dur:.3f}")
    sf.write("stream_out.wav", audio, SR)

    # verify streamed audio == non-streamed full decode (artifact check)
    print("wrote stream_out.wav")
