#!/bin/bash
set -e

# =============================================================================
# INFERENCE SERVER CONFIGURATION
# Model:     google/gemma-4-26B-A4B-it  (26B total, ~4B active MoE)
# Backend:   vLLM (docker: vllm/vllm-openai:gemma4 — gemma4 arch baked in)
# Hardware:  1x H100-80GB, GPU 2 (agent quota GPUs 2-3, TP=1)
# Objective: maximize throughput_tok_per_sec @ concurrency=8, 1k in / 500 out
# =============================================================================

IMAGE="vllm/vllm-openai:gemma4"
CONTAINER_NAME="vllm-exp-${PORT:-8000}"

# HF cache mount (weights already pre-fetched at ~/.cache/huggingface)
HF_CACHE="$HOME/.cache/huggingface"

# Clean up any stale container on this name
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# GPU pinning via docker --gpus. Port binds to loopback only so we don't collide
# with the other vllm on 0.0.0.0:8000.
exec docker run --rm --name "$CONTAINER_NAME" \
    --gpus '"device=2"' \
    --shm-size 32g \
    --ipc=host \
    -p "127.0.0.1:${PORT:-8000}:8000" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    "$IMAGE" \
    --model google/gemma-4-26B-A4B-it \
    --served-model-name gemma4 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --language-model-only \
    --host 0.0.0.0 \
    --port 8000
