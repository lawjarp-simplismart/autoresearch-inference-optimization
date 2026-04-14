#!/bin/bash
set -e

# =============================================================================
# Baseline: Qwen/Qwen3.5-9B on single H100 80GB
# - bf16 (no quantization per user constraint)
# - TP=1 (single GPU)
# - No speculative decoding
# - Prefix caching ON (workload has ~80% shared prefix)
# - Thinking mode DISABLED (otherwise model emits <think>...</think> which
#   distorts output-throughput measurement on a 500-token budget)
# - --language-model-only: model is multimodal; skip vision encoder to free
#   memory for KV cache (text-only benchmark)
# - max-model-len 8192: workload is 5k in + 500 out, no need for 262K context
# - Nightly vllm image required (stable does not yet ship Qwen3.5 arch)
# =============================================================================

CONTAINER_NAME="vllm-exp-${PORT:-8000}"
sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
cleanup() { sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

exec sudo docker run --rm --name "$CONTAINER_NAME" \
    --gpus '"device=0"' \
    --shm-size 32g --ipc=host --network=host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e VLLM_USE_V1=1 \
    vllm/vllm-openai:nightly \
    Qwen/Qwen3.5-9B \
    --port "${PORT:-8000}" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --language-model-only \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
