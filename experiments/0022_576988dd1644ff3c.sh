#!/bin/bash
set -e

# =============================================================================
# SGLang + pre-quantized FP8 weights: lovedheart/Qwen3.5-9B-FP8
#   Block-128x128 FP8, dynamic activation scaling. Half the memory bandwidth of bf16.
# - TP=1, no speculative decoding
# - Radix cache disabled (no-prefix-cache workload)
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
        --model-path lovedheart/Qwen3.5-9B-FP8 \
        --host 0.0.0.0 \
        --port "${PORT:-8000}" \
        --tp 1 \
        --context-length 16384 \
        --mem-fraction-static 0.90 \
        --disable-radix-cache \
        --attention-backend flashinfer
