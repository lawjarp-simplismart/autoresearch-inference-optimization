#!/bin/bash
set -e

# vLLM + FP8 + MTP num_speculative_tokens=3 (accept rate was 86.9% at num=2).

CONTAINER_NAME="vllm-exp-${PORT:-8000}"
sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
cleanup() { sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

exec sudo docker run --rm --name "$CONTAINER_NAME" \
    --gpus '"device=0"' \
    --shm-size 32g --ipc=host --network=host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$HOME/.cache/vllm:/root/.cache/vllm" \
    -e VLLM_USE_V1=1 \
    vllm/vllm-openai:nightly \
    lovedheart/Qwen3.5-9B-FP8 \
    --port "${PORT:-8000}" \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    --language-model-only \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}' \
    --async-scheduling
