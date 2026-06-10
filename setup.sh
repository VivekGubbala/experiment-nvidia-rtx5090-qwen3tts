#!/bin/bash
# Reproducible setup for the megakernel Qwen3-TTS -> Pipecat demo on an RTX 5090
# (sm_120 / Blackwell). Run from the repo root. Assumes the Vast /venv/main env.
#
# Clone with submodules:  git clone --recurse-submodules <repo>
# (or run this script as-is; step 2 inits the Qwen3-TTS submodule for you.)
set -e
source /venv/main/bin/activate
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

# 1. PyTorch built for sm_120 (CUDA 12.8 wheels run on the CUDA-13 host driver).
#    An older cu124 wheel installs but dies with "no kernel image" on Blackwell.
uv pip install "torch>=2.7" torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Qwen3-TTS package (a git submodule; pins transformers==4.57.3; torch untouched).
#    Pulls the pinned submodule in case you cloned without --recurse-submodules.
uv pip install "transformers==4.57.3" "accelerate==1.12.0" librosa soundfile sox onnxruntime einops
git submodule update --init --recursive
uv pip install -e ./Qwen3-TTS --no-deps

# 3. Pipecat (+ local Whisper STT, Silero VAD deps) and the inference server.
#    (qwen_megakernel is vendored in-repo with our kernel edits — no clone
#    needed; it JIT-compiles for sm_120 on first import.)
uv pip install "pipecat-ai[whisper]" websockets fastapi "uvicorn[standard]"

# 4. Models (~2.4 GB talker + tokenizer/vocoder).
hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
hf download Qwen/Qwen3-TTS-Tokenizer-12Hz

cat <<'EOS'
Setup done.

Reproduce the numbers:
  cd qwen_megakernel && python -m qwen_megakernel.bench && cd ..   # kernel baseline
  cd scripts
  python parity_test.py      # talker parity vs HF
  python cp_parity_test.py   # code-predictor parity vs HF
  python benchmark.py        # HF vs MK-talker vs MK-talker+codePred (RTF)
  python pc_verify.py        # Pipecat TTSService streaming proof

Run the voice agent (3 terminals on the GPU box + SSH tunnel from a laptop):
  python qwen3_tts_megakernel/server.py                 # 1: TTS inference server
  cd scripts && python voice_demo.py                    # 2: agent (needs OPENAI_API_KEY,
                                                        #    or: --llm echo for no-key parrot)
  ssh -p <port> -L 7860:127.0.0.1:7860 root@<host>      # 3: from the laptop
  # then open http://localhost:7860 and talk
EOS
