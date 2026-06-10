#!/bin/bash
# Reproducible setup for the megakernel Qwen3-TTS -> Pipecat demo on an RTX 5090
# (sm_120 / Blackwell). Run from the repo root. Assumes the Vast /venv/main env.
set -e
source /venv/main/bin/activate
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

# 1. PyTorch built for sm_120 (CUDA 12.8 wheels run on the CUDA-13 host driver).
#    An older cu124 wheel installs but dies with "no kernel image" on Blackwell.
uv pip install "torch>=2.7" torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Qwen3-TTS package (pins transformers==4.57.3; torch is left untouched).
uv pip install "transformers==4.57.3" "accelerate==1.12.0" librosa soundfile sox onnxruntime einops
git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS.git 2>/dev/null || true
uv pip install -e ./Qwen3-TTS --no-deps

# 3. The megakernel (JIT-compiled on first import for sm_120) + Pipecat.
git clone --depth 1 https://github.com/AlpinDale/qwen_megakernel.git 2>/dev/null || true
uv pip install pipecat-ai websockets

# 4. Models (~2.4 GB talker + tokenizer/vocoder).
hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
hf download Qwen/Qwen3-TTS-Tokenizer-12Hz

echo "Setup done. Try:  cd qwen_megakernel && python -m qwen_megakernel.bench"
echo "Then:             cd ../scripts && python parity_test.py && python benchmark.py && python pc_verify.py"
