"""Full voice agent over a single TCP port: browser mic -> Whisper STT ->
LLM -> megakernel Qwen3-TTS -> browser speaker.

Media flows over a Pipecat websocket transport (protobuf frames), so it works
on a headless GPU box with NO inbound UDP (Vast containers are TCP-only) —
unlike WebRTC. The browser client is served from this same app.

How to run the end-to-end demo (and record it):
  1. on the GPU box:    python ../qwen3_tts_megakernel/server.py        # TTS server
  2. on the GPU box:    OPENAI_API_KEY=sk-... python voice_demo.py     # this app
  3. on your laptop:    ssh -p <VAST_TCP_PORT_22> -L 7860:127.0.0.1:7860 root@<PUBLIC_IP>
  4. open http://localhost:7860 , click Start, talk. (getUserMedia needs a
     secure context — localhost via the SSH tunnel qualifies; a raw public
     http:// IP does not.)

LLM: any OpenAI-compatible endpoint — set OPENAI_API_KEY, and optionally
OPENAI_MODEL (default gpt-4o-mini) and OPENAI_BASE_URL (e.g. a local vLLM).
No key? `--llm echo` replaces the LLM with a parrot ("You said: ...") so the
full mic -> STT -> TTS -> speaker loop still works.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qwen3_tts_megakernel"))

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair)
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTService, Model
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams, FastAPIWebsocketTransport)

from pc_tts import MegakernelQwen3TTSWebsocketService, SR

app = FastAPI(title="megakernel voice agent")
HERE = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
async def index():
    return FileResponse(os.path.join(HERE, "static", "voice_client.html"))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    transport = FastAPIWebsocketTransport(
        websocket=ws,
        params=FastAPIWebsocketParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=SR,            # 24 kHz from the vocoder
            add_wav_header=False,
        ),
    )

    # pipecat >= 1.x: VAD is its own processor (a `vad_analyzer` transport
    # param is silently ignored); SegmentedSTTService segments on its frames
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())
    stt = WhisperSTTService(model=Model.SMALL, device="cuda")
    tts = MegakernelQwen3TTSWebsocketService(url=app.state.tts_url)

    if app.state.llm_mode == "echo":
        class EchoResponder(FrameProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, TranscriptionFrame):
                    await self.push_frame(
                        TTSSpeakFrame(f"You said: {frame.text}"))
                else:
                    await self.push_frame(frame, direction)

        stages = [transport.input(), vad, stt, EchoResponder(), tts,
                  transport.output()]
    else:
        llm = OpenAILLMService(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        ctx = LLMContext([{
            "role": "system",
            "content": "You are a concise, friendly voice assistant. "
                       "Answer in 1-3 short sentences of plain text "
                       "(it will be spoken aloud).",
        }])
        agg = LLMContextAggregatorPair(ctx)
        stages = [transport.input(), vad, stt, agg.user(), llm, tts,
                  transport.output(), agg.assistant()]

    pipeline = Pipeline(stages)
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True,
                              enable_metrics=True),
    )
    await PipelineRunner(handle_sigint=False).run(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (keep 127.0.0.1 + SSH tunnel; "
                         "browsers refuse mic on insecure public origins)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--tts-url", default="ws://127.0.0.1:17800/v1/tts/stream",
                    help="megakernel TTS server (start server.py first)")
    ap.add_argument("--llm", choices=["openai", "echo"], default="openai",
                    help="echo = no-LLM parrot mode (no API key needed)")
    args = ap.parse_args()
    if args.llm == "openai" and "OPENAI_API_KEY" not in os.environ:
        sys.exit("Set OPENAI_API_KEY (any OpenAI-compatible endpoint; see "
                 "OPENAI_BASE_URL/OPENAI_MODEL) — or run with --llm echo.")
    app.state.tts_url = args.tts_url
    app.state.llm_mode = args.llm

    # warm Whisper/cuDNN once at startup: the first transcription otherwise
    # pays ~8 s of cuDNN autotune + CTranslate2 init mid-conversation
    print("warming up Whisper (one-time cuDNN init)…", flush=True)
    import numpy as np
    from faster_whisper import WhisperModel
    list(WhisperModel("small", device="cuda")
         .transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])
    print(f"voice agent on http://{args.host}:{args.port}  "
          f"(tunnel: ssh -L {args.port}:127.0.0.1:{args.port} ...)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
