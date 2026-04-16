#!/bin/bash
set -e

# =============================================================================
# Exp #29: EAGLE3 with RedHatAI draft on vllm/vllm-openai:nightly
# Base: exp #15 (135.98). Prior #28 failed because vllm 0.19 (in Arctic image)
# rejects eagle3 for gemma4 target. RedHat docs say eagle3+gemma4 is "supported
# on vllm-main", so use today's nightly. num_speculative_tokens=3 per RedHat
# example. Keep max-num-seqs=1 + FP8 from #15.
# =============================================================================

MODEL="google/gemma-4-26B-A4B-it"
DRAFT="RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3"
IMAGE="vllm/vllm-openai:nightly"
CONTAINER_NAME="vllm-exp-${PORT:-8000}"

docker pull "$IMAGE"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

exec docker run --rm --name "$CONTAINER_NAME" \
    --gpus '"device=0"' \
    --shm-size 32g --ipc=host --network=host \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    --entrypoint bash \
    "$IMAGE" \
    -c "pip install --quiet pandas && exec vllm serve '$MODEL' \
        --tensor-parallel-size 1 \
        --quantization fp8 \
        --max-model-len 12288 \
        --gpu-memory-utilization 0.90 \
        --limit-mm-per-prompt '{\"image\":0,\"audio\":0}' \
        --max-num-seqs 1 \
        --speculative-config '{\"method\":\"eagle3\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":3}' \
        --port ${PORT:-8000}"
