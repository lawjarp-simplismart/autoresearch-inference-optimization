#!/bin/bash
set -e

# =============================================================================
# Exp #3: On-the-fly FP8 quantization on the base bf16 model, TP=1
# Image: vllm/vllm-openai:gemma4
# Context: RedHatAI pre-quantized FP8-Dynamic failed (expert param name mismatch
# in vllm gemma4.load_weights). This path uses vllm's built-in FP8 quantizer
# which maps into the correct gemma4 MoE expert layout.
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
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    --port "${PORT:-8000}"
