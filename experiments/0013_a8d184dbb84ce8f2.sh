#!/bin/bash
set -e

# =============================================================================
# Exp #12: FP8 on vllm/vllm-openai:latest (upstream vllm 0.19.0 w/ gemma4 support)
# Base: exp #3 on vllm-openai:gemma4 fork = 135.63 tok/s.
# Hypothesis: upstream has newer MoE/attention kernels and/or fused optimizations.
# =============================================================================

MODEL="google/gemma-4-26B-A4B-it"
IMAGE="vllm-latest-gemma4:v2"  # vllm/vllm-openai:latest + transformers 4.57
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
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    --port "${PORT:-8000}"
