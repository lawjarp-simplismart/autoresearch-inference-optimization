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

# Environment — GPU 0 occupied by styletts2, use GPUs 1-4
export CUDA_VISIBLE_DEVICES=1,2,3,4

# Launch server
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-31B \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port ${PORT:-8000}
