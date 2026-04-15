#!/bin/bash
set -e

# =============================================================================
# Exp #8: FP8 + Arctic Inference suffix decoding
# Image: vllm-gemma4-arctic:v2 (gemma4 base + pip install --no-deps arctic-inference,
# preserves the dev vllm build that supports gemma-4-26B).
# Spec config: {"method":"suffix"} with ARCTIC_INFERENCE_ENABLED=1.
# Hypothesis: suffix decoding has much higher acceptance than n-gram on
# randomized prompts — significant c=1/10k decode win if it engages.
# =============================================================================

MODEL="google/gemma-4-26B-A4B-it"
IMAGE="simplismart/simplismart-inference-server-base:vllm-gemma4-arctic-v1"
CONTAINER_NAME="vllm-exp-${PORT:-8000}"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

exec docker run --rm --name "$CONTAINER_NAME" \
    --gpus '"device=0"' \
    --shm-size 32g --ipc=host --network=host \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    -e ARCTIC_INFERENCE_ENABLED=1 \
    -e ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1 \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    --entrypoint python3 \
    "$IMAGE" \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --quantization fp8 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    --speculative-config '{"method":"suffix"}' \
    --port "${PORT:-8000}"
