"""Pipecat TTS services backed by the megakernel Qwen3-TTS pipeline.

Two drop-in Pipecat `TTSService`s, both streaming `TTSAudioRawFrame`s
frame-by-frame as audio decodes (no full-utterance buffering):

- `MegakernelQwen3TTSWebsocketService` (RECOMMENDED): thin async client for
  `server.py`. Generation runs in the server's process, so the Pipecat event
  loop never competes with the GPU loop for the GIL — pipelines (including
  transport-less headless ones) run cleanly.
- `MegakernelQwen3TTSService`: in-process variant. The synchronous CUDA
  generator runs in one dedicated producer thread feeding an `asyncio.Queue`.
  Works for direct `run_tts` use, but a transport-less PipelineRunner can
  starve on the GIL — prefer the websocket service inside pipelines.
"""
import asyncio
import json
import threading

from pipecat.frames.frames import (
    TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, ErrorFrame,
)
from pipecat.services.tts_service import TTSService

from mk_tts import SR


class MegakernelQwen3TTSWebsocketService(TTSService):
    """Streams TTS from the `server.py` websocket endpoint."""

    def __init__(self, *, url="ws://127.0.0.1:17800/v1/tts/stream",
                 speaker="Ryan", language="English", instruct="",
                 do_sample=True, max_new_tokens=2048, **kwargs):
        from pipecat.services.settings import TTSSettings
        kwargs.setdefault("settings", TTSSettings(
            model="qwen3-tts-megakernel", voice=speaker, language=language))
        super().__init__(sample_rate=SR, **kwargs)
        self._url = url
        self._speaker = speaker
        self._language = language
        self._instruct = instruct
        self._do_sample = do_sample
        self._max_new_tokens = max_new_tokens
        self._ws = None
        self._lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        return True

    async def _get_ws(self):
        import websockets
        if self._ws is None:
            self._ws = await websockets.connect(self._url, max_size=None)
        return self._ws

    async def run_tts(self, text: str, context_id: str):
        if not text.strip():
            return
        async with self._lock:                  # server is single-stream
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()
            try:
                ws = await self._get_ws()
                await ws.send(json.dumps({
                    "text": text, "speaker": self._speaker,
                    "language": self._language, "instruct": self._instruct,
                    "do_sample": self._do_sample,
                    "max_new_tokens": self._max_new_tokens,
                }))
                first = True
                async for msg in ws:
                    if isinstance(msg, bytes):
                        if first:
                            await self.stop_ttfb_metrics()
                            first = False
                        yield TTSAudioRawFrame(audio=msg, sample_rate=SR,
                                               num_channels=1,
                                               context_id=context_id)
                    else:
                        ev = json.loads(msg)
                        if ev.get("event") == "done":
                            break
                        if ev.get("event") == "error":
                            yield ErrorFrame(f"TTS server: {ev.get('message')}")
                            break
            except Exception as e:
                self._ws = None                 # force reconnect next time
                yield ErrorFrame(f"megakernel TTS websocket error: {e}")
            finally:
                yield TTSStoppedFrame()


class MegakernelQwen3TTSService(TTSService):
    def __init__(self, *, engine=None,
                 speaker="Ryan", language="English", instruct="",
                 do_sample=True, chunk_frames=8, lookback=8,
                 max_new_tokens=2048, **kwargs):
        from pipecat.services.settings import TTSSettings
        kwargs.setdefault("settings", TTSSettings(
            model="qwen3-tts-megakernel", voice=speaker, language=language))
        super().__init__(sample_rate=SR, **kwargs)
        if engine is None:
            from mk_tts import StreamingMegakernelTTS
            engine = StreamingMegakernelTTS(use_megakernel=True)
        self._engine = engine
        self._speaker = speaker
        self._language = language
        self._instruct = instruct
        self._do_sample = do_sample
        self._chunk_frames = chunk_frames
        self._lookback = lookback
        self._max_new_tokens = max_new_tokens

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str):
        if not text.strip():
            return
        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        _DONE = object()

        def producer():
            # whole sync generator stays on THIS one thread
            try:
                for pcm, sr in self._engine.stream(
                        text, speaker=self._speaker, language=self._language,
                        instruct=self._instruct, max_new_tokens=self._max_new_tokens,
                        chunk_frames=self._chunk_frames, lookback=self._lookback,
                        do_sample=self._do_sample, seed=None):
                    loop.call_soon_threadsafe(q.put_nowait, (pcm, sr))
            except Exception as e:  # pragma: no cover
                loop.call_soon_threadsafe(q.put_nowait, ("__err__", e))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, _DONE)

        threading.Thread(target=producer, daemon=True).start()

        first = True
        try:
            while True:
                item = await q.get()
                if item is _DONE:
                    break
                if isinstance(item, tuple) and item and item[0] == "__err__":
                    yield ErrorFrame(f"megakernel TTS error: {item[1]}")
                    break
                pcm, sr = item
                if first:
                    await self.stop_ttfb_metrics()
                    first = False
                yield TTSAudioRawFrame(audio=pcm, sample_rate=sr,
                                       num_channels=1, context_id=context_id)
        finally:
            yield TTSStoppedFrame()
