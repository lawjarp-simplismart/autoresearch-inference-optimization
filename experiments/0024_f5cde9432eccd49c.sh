#!/bin/bash
set -e

# =============================================================================
# Exp #24: sglang (built from main) + FP8 — completely different engine
# Base: vllm plateau at ~135 tok/s. Different kernels/scheduler may break
# through the plateau; if so, we can layer suffix-tree spec decode next.
# Image: sglang-main:v4 (lmsysorg/sglang:latest + sglang-git-main + transformers-git-main)
# =============================================================================

MODEL="google/gemma-4-26B-A4B-it"
IMAGE="sglang-main:v4"
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
    --entrypoint python3 \
    "$IMAGE" \
    -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp 1 \
    --quantization fp8 \
    --context-length 12288 \
    --mem-fraction-static 0.90 \
    --host 0.0.0.0 \
    --port "${PORT:-8000}"
