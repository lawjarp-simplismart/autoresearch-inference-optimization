#!/bin/bash
set -e

# =============================================================================
# Exp #26: Arctic image + suffix decoding (num_speculative_tokens=32)
# Base: exp #15 (135.98 tok/s — FP8 + max-num-seqs=1).
# Hypothesis: Snowflake's suffix decoding (Arctic) reuses prompt suffixes as
# drafts. With 10k-token prompts at c=1, decode is memory-bound and there's
# spare compute to verify long drafts. 32 tokens is aggressive; if accept rate
# is even ~30% on pos-1, we win.
# =============================================================================

MODEL="google/gemma-4-26B-A4B-it"
IMAGE="simplismart/simplismart-inference-server-base:vllm-gemma4-arctic-v1"
CONTAINER_NAME="vllm-exp-${PORT:-8000}"

docker pull "$IMAGE"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

exec docker run --rm --name "$CONTAINER_NAME" \
    --gpus '"device=0"' \
    --shm-size 32g --ipc=host --network=host \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    --entrypoint vllm \
    "$IMAGE" \
    serve "$MODEL" \
    --tensor-parallel-size 1 \
    --quantization fp8 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    --max-num-seqs 1 \
    --speculative-config '{"method":"suffix","num_speculative_tokens":32}' \
    --port "${PORT:-8000}"
