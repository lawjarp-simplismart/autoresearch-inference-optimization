#!/bin/bash
set -e

# vLLM + Qwen3.5-9B bf16 + DFlash draft model speculative decoding.
# Per z-lab/Qwen3.5-9B-DFlash official recipe.

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
    Qwen/Qwen3.5-9B \
    --port "${PORT:-8000}" \
    --tensor-parallel-size 1 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.95 \
    --language-model-only \
    --max-num-batched-tokens 32768 \
    --attention-backend flash_attn \
    --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-9B-DFlash","num_speculative_tokens":15}' \
    --trust-remote-code \
    --async-scheduling
