#!/bin/bash
set -e

# =============================================================================
# Exp #11: FP8 + tight max_model_len (10752 = 10k prompt + 500 output + 252 slack)
# Base: exp #3 (FP8, 135.63 tok/s at c=1/10k). Hypothesis: shrinking max_model_len
# from 12288 → 10752 makes CUDA graphs/compile specialize on a smaller range,
# reducing per-step kernel overhead at c=1 long context.
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
    --max-model-len 10752 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    --port "${PORT:-8000}"
