#!/bin/bash
set -e

# =============================================================================
# vLLM + pre-quantized FP8 weights: lovedheart/Qwen3.5-9B-FP8
# =============================================================================

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
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_USE_DEEP_GEMM=1 \
    vllm/vllm-openai:nightly \
    lovedheart/Qwen3.5-9B-FP8 \
    --port "${PORT:-8000}" \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    --language-model-only \
    --async-scheduling
