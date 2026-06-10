"""Pipecat pipeline demos for the megakernel Qwen3-TTS service.

  python pipeline_demo.py headless          # real Pipecat Pipeline, headless:
                                            # TTSSpeakFrame -> websocket TTS
                                            # -> streamed audio -> pipeline_out.wav
                                            # (start server.py first)

  python pipeline_demo.py headless-inproc   # same pipeline but with the
                                            # in-process TTS service (loads the
                                            # model here; can GIL-starve a
                                            # transport-less runner — kept for
                                            # comparison/documentation)

For the full voice agent (browser mic -> STT -> LLM -> TTS -> speaker) see
voice_demo.py.
"""
import os
import sys
import asyncio
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qwen3_tts_megakernel"))
from mk_tts import SR  # noqa: E402

from pipecat.frames.frames import (  # noqa: E402
    TTSSpeakFrame, TTSAudioRawFrame, EndFrame, Frame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection  # noqa: E402


class WavCollector(FrameProcessor):
    """Sink that captures streamed TTSAudioRawFrames and writes a wav at EndFrame."""

    def __init__(self, path="pipeline_out.wav", done: "asyncio.Event" = None):
        super().__init__()
        self._path = path
        self._buf = bytearray()
        self._n = 0
        self._done = done

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            self._n += 1
            self._buf += frame.audio
            print(f"  [pipeline] audio frame #{self._n}: {len(frame.audio)} bytes "
                  f"({len(frame.audio)//2/SR*1000:.0f} ms)", flush=True)
        elif isinstance(frame, EndFrame):
            self._write()
            if self._done is not None:
                self._done.set()
        await self.push_frame(frame, direction)

    def _write(self):
        with wave.open(self._path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(bytes(self._buf))
        print(f"  [pipeline] wrote {self._path} "
              f"({len(self._buf)//2/SR:.2f}s from {self._n} frames)", flush=True)


def make_tts(mode):
    if mode == "headless-inproc":
        from pc_tts import MegakernelQwen3TTSService
        return MegakernelQwen3TTSService(do_sample=True)
    from pc_tts import MegakernelQwen3TTSWebsocketService
    return MegakernelQwen3TTSWebsocketService()      # needs server.py running


async def headless(mode):
    done = asyncio.Event()
    tts = make_tts(mode)
    sink = WavCollector("pipeline_out.wav", done=done)
    task = PipelineTask(Pipeline([tts, sink]))
    text = "This audio is streaming through a real Pipecat pipeline, frame by frame."
    runner = PipelineRunner(handle_sigint=False)
    print(f"[{mode}] queueing frames + running task...", flush=True)
    run_t = asyncio.create_task(runner.run(task))
    await task.queue_frames([TTSSpeakFrame(text), EndFrame()])
    try:
        await asyncio.wait_for(done.wait(), timeout=120)
        print(f"[{mode}] EndFrame reached sink; clean finish", flush=True)
    except asyncio.TimeoutError:
        print(f"[{mode}] TIMEOUT; writing partial audio", flush=True)
        sink._write()
    finally:
        run_t.cancel()
        try:
            await run_t
        except (asyncio.CancelledError, Exception):
            pass


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "headless"
    asyncio.run(headless(mode))
