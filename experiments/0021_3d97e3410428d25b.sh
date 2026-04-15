#!/bin/bash
set -e

# =============================================================================
# Exp #21: FP8 + Arctic suffix decoding (simpli-gemma4-arctic:v2 with compat shim)
# The simplismart image has vllm 0.19 + arctic 0.1.2. We added a shim that
# re-exports FlexibleArgumentParser from vllm.utils.argparse_utils back into
# vllm.utils, which arctic-inference 0.1.2 imports. Version check bypassed
# via ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1.
# =============================================================================

MODEL="google/gemma-4-26B-A4B-it"
IMAGE="simpli-gemma4-arctic:v2"
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
    --max-num-seqs 1 \
    --speculative-config '{"method":"suffix"}' \
    --port "${PORT:-8000}"
