#!/bin/bash
set -e

# =============================================================================
# Exp #4: FP8 weights + FP8 KV cache, TP=1
# Image: vllm/vllm-openai:gemma4
# Hypothesis: at c=1/10k, decode is bandwidth-bound and KV read dominates.
# kv-cache-dtype=fp8 halves KV traffic → expect further ~5-15% speedup.
# Base: exp #3 (FP8 on-the-fly, 135.63 tok/s).
# =============================================================================

MODEL="google/gemma-4-26B-A4B-it"
IMAGE="vllm/vllm-openai:gemma4"
CONTAINER_NAME="vllm-exp-${PORT:-8000}"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

exec docker run --rm --name "$CONTAINER_NAME" \
    --gpus '"device=0"' \
    --shm-size 32g --ipc=host --network=host \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    "$IMAGE" \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    --port "${PORT:-8000}"
