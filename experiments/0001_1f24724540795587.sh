#!/bin/bash
set -e

# =============================================================================
# INFERENCE SERVER CONFIGURATION
# Model:     google/gemma-4-26B-A4B-it  (26B total, ~4B active MoE)
# Backend:   vLLM (pip-installed into a venv on remote)
# Hardware:  1x H100-80GB, GPU 2 (agent quota GPUs 2-3, TP=1)
# Objective: maximize throughput_tok_per_sec @ concurrency=8, 1k in / 500 out
# =============================================================================

export CUDA_VISIBLE_DEVICES=2

# HF token for gated Gemma access — expected in remote shell env
# If missing, the model download will fail.
: "${HF_TOKEN:?HF_TOKEN not set on remote}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Reuse or create a project-local venv so we don't reinstall vLLM per run.
VENV_DIR="$HOME/autoresearch-inference-optimization/.venv-vllm"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -q --upgrade pip
    "$VENV_DIR/bin/pip" install -q vllm
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

exec python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-26B-A4B-it \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.90 \
    --port "${PORT:-8000}"
