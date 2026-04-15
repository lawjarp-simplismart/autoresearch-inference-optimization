#!/bin/bash
set -e

# =============================================================================
# Exp #23: bitsandbytes INT4 on-the-fly quantization
# Base: FP8 at 135.98. INT4 halves weight bandwidth vs FP8 (2x vs bf16),
# so decode could theoretically jump significantly. Caveat: bnb kernels
# are slower than marlin/FP8 kernels, so may net-regress.
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
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    --max-num-seqs 1 \
    --port "${PORT:-8000}"
