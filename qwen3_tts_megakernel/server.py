"""Streaming TTS inference server (prompt in -> audio chunk stream out).

Wraps `StreamingMegakernelTTS` behind a websocket so the GPU + the
Python-heavy generation loop live in their *own process*. Pipecat (or any
client) connects as a thin async websocket client — which also removes the
GIL contention that stalls an in-process transport-less Pipecat pipeline.

Protocol (one websocket, requests served sequentially):
  client -> {"text": "...", "speaker": "Ryan", "language": "English",
             "instruct": "", "do_sample": true, "max_new_tokens": 2048}
  server -> {"event": "started", "sr": 24000}
  server -> <binary int16 mono PCM chunk>  (repeated, pushed as decoded)
  server -> {"event": "done", "chunks": N, "audio_s": x, "wall_s": y}
  (errors: {"event": "error", "message": "..."})

Run:  python server.py [--host 127.0.0.1] [--port 17800]
Health:  GET http://host:port/healthz
"""
import argparse
import asyncio
import json
import threading
import time

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from mk_tts import StreamingMegakernelTTS, SR

app = FastAPI(title="qwen3-tts-megakernel")
_engine: StreamingMegakernelTTS = None
_DONE = object()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "sr": SR, "engine": "megakernel"}


@app.websocket("/v1/tts/stream")
async def tts_stream(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    try:
        while True:
            req = json.loads(await ws.receive_text())
            q: asyncio.Queue = asyncio.Queue()

            def producer(req=req, q=q):
                try:
                    for pcm, sr in _engine.stream(
                            req["text"],
                            speaker=req.get("speaker", "Ryan"),
                            language=req.get("language", "English"),
                            instruct=req.get("instruct", ""),
                            do_sample=req.get("do_sample", True),
                            max_new_tokens=req.get("max_new_tokens", 2048),
                            seed=None):
                        loop.call_soon_threadsafe(q.put_nowait, pcm)
                except Exception as e:
                    loop.call_soon_threadsafe(q.put_nowait, ("__err__", e))
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, _DONE)

            threading.Thread(target=producer, daemon=True).start()
            await ws.send_text(json.dumps({"event": "started", "sr": SR}))
            t0, n, samples, failed = time.time(), 0, 0, False
            while True:
                item = await q.get()
                if item is _DONE:
                    break
                if isinstance(item, tuple) and item and item[0] == "__err__":
                    await ws.send_text(json.dumps(
                        {"event": "error", "message": str(item[1])}))
                    failed = True
                    break
                await ws.send_bytes(item)
                n += 1
                samples += len(item) // 2
            if not failed:
                await ws.send_text(json.dumps(
                    {"event": "done", "chunks": n,
                     "audio_s": round(samples / SR, 3),
                     "wall_s": round(time.time() - t0, 3)}))
    except WebSocketDisconnect:
        pass


def main():
    global _engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=17800)
    ap.add_argument("--no-megakernel", action="store_true",
                    help="serve the HF baseline instead (for comparison)")
    args = ap.parse_args()

    print("loading model + JIT-building megakernel (first run takes ~2 min)…",
          flush=True)
    _engine = StreamingMegakernelTTS(use_megakernel=not args.no_megakernel,
                                     use_megakernel_cp=not args.no_megakernel)
    # warmup: JIT, cuDNN autotune, first-call allocs
    for _ in _engine.stream("Warm up.", max_new_tokens=32, do_sample=False):
        pass
    print(f"ready on ws://{args.host}:{args.port}/v1/tts/stream", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
