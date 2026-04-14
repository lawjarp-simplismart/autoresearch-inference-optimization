#!/bin/bash
set -e

# =============================================================================
# INFERENCE SERVER CONFIGURATION
# Model: google/gemma-4-31B
# Backend: SGLang
# Hardware: 4x H100-80GB (GPUs 4-7, agent quota)
# Objective: minimize end-to-end latency at c=1
# =============================================================================

# HF token for gated Gemma access
export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Use only the last 4 GPUs (agent quota)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Ensure sglang is installed in the project venv
uv venv --python 3.10 .venv >/dev/null 2>&1 || true
source .venv/bin/activate
python -c "import sglang" 2>/dev/null || uv pip install -q "sglang[all]"
python -c "import transformers" 2>/dev/null || uv pip install -q --upgrade transformers

# Launch SGLang OpenAI-compatible server (c=1 latency tuned)
python -m sglang.launch_server \
    --model-path google/gemma-4-31B \
    --tp 4 \
    --dtype bfloat16 \
    --mem-fraction-static 0.85 \
    --context-length 8192 \
    --chat-template chat_template.jinja \
    --host 0.0.0.0 \
    --port ${PORT:-8000}
