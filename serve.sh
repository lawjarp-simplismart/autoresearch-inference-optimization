#!/bin/bash
set -e

# =============================================================================
# INFERENCE SERVER CONFIGURATION
# Model: google/gemma-4-31B
# Backend: vLLM
# Hardware: 8x H100-80GB
# =============================================================================

# Activate venv and install vllm + latest transformers (gemma4 support)
source .venv/bin/activate
uv pip install -q vllm
uv pip install -q --upgrade transformers

# Use all 8 GPUs (GPU 0 has ~17GB used by styletts2, ~64GB free — enough for tp=8 shard)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Launch server
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-31B \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enable-chunked-prefill \
    --chat-template chat_template.jinja \
    --port ${PORT:-8000}
