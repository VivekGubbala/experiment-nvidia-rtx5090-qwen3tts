"""Pipecat TTS service backed by the megakernel Qwen3-TTS talker.

`MegakernelQwen3TTSService` is a drop-in Pipecat `TTSService`: it streams
`TTSAudioRawFrame`s frame-by-frame as the megakernel-backed pipeline decodes
audio (no full-utterance buffering). The synchronous CUDA generator from
`mk_tts.StreamingMegakernelTTS` is bridged to async by running the whole
synchronous generator in one dedicated producer thread that feeds an
`asyncio.Queue`, so the Pipecat event loop is never blocked and the generator
is never resumed across threads.
"""
import asyncio
import threading

from pipecat.frames.frames import (
    TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, ErrorFrame,
)
from pipecat.services.tts_service import TTSService

from mk_tts import StreamingMegakernelTTS, SR


class MegakernelQwen3TTSService(TTSService):
    def __init__(self, *, engine: StreamingMegakernelTTS = None,
                 speaker="Ryan", language="English", instruct="",
                 do_sample=True, chunk_frames=2, lookback=8,
                 max_new_tokens=2048, **kwargs):
        super().__init__(sample_rate=SR, **kwargs)
        self._engine = engine or StreamingMegakernelTTS(use_megakernel=True)
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
