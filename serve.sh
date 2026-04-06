#!/bin/bash
set -e

# =============================================================================
# INFERENCE SERVER CONFIGURATION
# Model: google/gemma-4-31B
# Backend: vLLM
# Hardware: 8x H100-80GB
# =============================================================================

# Activate venv and install vllm
source .venv/bin/activate
pip install -q vllm

# Environment — GPU 0 occupied by styletts2, use GPUs 1-4
export CUDA_VISIBLE_DEVICES=1,2,3,4

# Launch server
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-31B \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --port ${PORT:-8000}
