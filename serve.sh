#!/bin/bash
set -e

# =============================================================================
# INFERENCE SERVER CONFIGURATION
# Model: google/gemma-4-31B
# Backend: vLLM
# Hardware: 8x H100-80GB
# =============================================================================

# Activate venv, install only if missing
source .venv/bin/activate
pip show vllm >/dev/null 2>&1 || uv pip install -q vllm
pip show transformers >/dev/null 2>&1 || uv pip install -q --upgrade transformers

# Single GPU (tp=1)
export CUDA_VISIBLE_DEVICES=1

# Launch server
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-31B \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enable-chunked-prefill \
    --quantization fp8 \
    --chat-template chat_template.jinja \
    --port ${PORT:-8000}
