"""Pipecat demos for the megakernel Qwen3-TTS service.

Two modes:

  python pipeline_demo.py headless     # runs a real Pipecat Pipeline headless,
                                        # TTSSpeakFrame -> audio frames -> out.wav
                                        # (proves the service inside a Pipeline)

  python pipeline_demo.py voice        # full voice agent:
                                        # mic -> Whisper STT -> LLM -> our TTS -> speaker
                                        # (needs an audio device + OPENAI_API_KEY,
                                        #  or swap in Ollama for a fully local LLM)
"""
import os
import sys
import asyncio
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qwen3_tts_megakernel"))
from pc_tts import MegakernelQwen3TTSService  # noqa: E402
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

    def __init__(self, path="out.wav", done: "asyncio.Event" = None):
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


async def headless():
    # NOTE: this drives a transport-less Pipecat Pipeline in-process. Audio
    # frames stream correctly (see prints), but the heavy GPU+Python generation
    # loop contends with the asyncio event loop for the GIL, so a transport-less
    # runner does not always drain/finalize cleanly. For a deterministic proof of
    # the streaming TTSService contract run `pc_verify.py`; for a real agent use
    # `pipeline_demo.py voice` (a transport paces/drains the pipeline).
    done = asyncio.Event()
    tts = MegakernelQwen3TTSService(do_sample=True, chunk_frames=2, lookback=8,
                                    max_new_tokens=256)
    sink = WavCollector("pipeline_out.wav", done=done)
    task = PipelineTask(Pipeline([tts, sink]))
    text = "This audio is streaming through a real Pipecat pipeline, frame by frame."
    runner = PipelineRunner(handle_sigint=False)
    print("[headless] queueing frames + running task...", flush=True)
    run_t = asyncio.create_task(runner.run(task))
    await task.queue_frames([TTSSpeakFrame(text), EndFrame()])
    try:
        await asyncio.wait_for(done.wait(), timeout=90)
        print("[headless] EndFrame reached sink; clean finish", flush=True)
    except asyncio.TimeoutError:
        print("[headless] timeout; writing partial audio", flush=True)
        sink._write()
    finally:
        run_t.cancel()
        try:
            await run_t
        except (asyncio.CancelledError, Exception):
            pass


def build_voice_pipeline():
    """Full mic->STT->LLM->TTS->speaker voice agent (reference wiring)."""
    from pipecat.transports.local.audio import (
        LocalAudioTransport, LocalAudioTransportParams)
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=True, audio_out_enabled=True, audio_out_sample_rate=SR))
    stt = WhisperSTTService()  # local whisper
    llm = OpenAILLMService(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    tts = MegakernelQwen3TTSService(do_sample=True, chunk_frames=2)
    ctx = OpenAILLMContext([
        {"role": "system", "content": "You are a concise, friendly voice assistant."}])
    agg = llm.create_context_aggregator(ctx)
    pipeline = Pipeline([
        transport.input(), stt, agg.user(), llm, tts,
        transport.output(), agg.assistant(),
    ])
    return PipelineTask(pipeline)


async def voice():
    await PipelineRunner().run(build_voice_pipeline())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "headless"
    asyncio.run(headless() if mode == "headless" else voice())
