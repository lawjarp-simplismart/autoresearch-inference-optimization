#!/bin/bash
set -e

# =============================================================================
# Exp #16: Combined best-so-far — FP8 + async-scheduling + max-num-seqs=1 + FLASHINFER env
# Stacks every single-flag win (async #6, max-num-seqs=1 #15) and env var from #7
# onto the FP8 baseline. Upper-bound test on flag-tuning alone.
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
    -e VLLM_ATTENTION_BACKEND=FLASHINFER \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    "$IMAGE" \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --quantization fp8 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    --max-num-seqs 1 \
    --async-scheduling \
    --port "${PORT:-8000}"
