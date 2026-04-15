#!/bin/bash
set -e

# =============================================================================
# Exp #25: sglang main + FP8 — stabilized re-run
# Prior run (#24) had c=1/10k decode at 147-155 tok/s (vs vllm 135) but server
# crashed transitioning to c=8 combo. Lower mem pressure + disable radix cache
# to avoid cross-request KV buildup + cap cudagraph capture batch sizes.
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
    --mem-fraction-static 0.85 \
    --disable-radix-cache \
    --cuda-graph-max-bs 16 \
    --host 0.0.0.0 \
    --port "${PORT:-8000}"
