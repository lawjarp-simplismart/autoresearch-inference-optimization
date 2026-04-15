#!/bin/bash
set -e

# =============================================================================
# Exp #17: FP8 + n-gram spec decode with short draft (num_speculative_tokens=2)
# Prior ngram run (#5) had 28% accept at pos-1 dropping to <10% at pos-5. Try
# tiny draft (2 tokens) to limit cost while still capturing pos-1 wins.
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
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    --speculative-config '{"method":"ngram","num_speculative_tokens":2,"prompt_lookup_max":3,"prompt_lookup_min":2}' \
    --port "${PORT:-8000}"
