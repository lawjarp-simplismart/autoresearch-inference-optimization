#!/bin/bash
set -e

# =============================================================================
# SGLang backend: Qwen/Qwen3.5-9B on single H100 80GB
# - bf16 (no quantization)
# - TP=1, no speculative decoding
# - Radix cache (sglang's prefix cache) disabled: workload is no-prefix-cache
# =============================================================================

CONTAINER_NAME="sglang-exp-${PORT:-8000}"
sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
cleanup() { sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

exec sudo docker run --rm --name "$CONTAINER_NAME" \
    --gpus '"device=0"' \
    --shm-size 32g --ipc=host --network=host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen3.5-9B \
        --host 0.0.0.0 \
        --port "${PORT:-8000}" \
        --tp 1 \
        --dtype bfloat16 \
        --context-length 16384 \
        --mem-fraction-static 0.90 \
        --disable-radix-cache \
        --attention-backend flashinfer
