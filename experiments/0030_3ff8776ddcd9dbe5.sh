#!/bin/bash
set -e

# =============================================================================
# Exp #30: EAGLE3 with bf16 target (no FP8) — isolate quant drift
# Base: exp #29 (205 tok/s @ 1k, 90 @ 10k — accept rate 14.8% only).
# Hypothesis: speculator was trained on bf16 logits. FP8 quant may shift the
# target distribution enough to drop accept rate. Drop --quantization fp8 and
# see if accept rate recovers. bf16 target costs ~10 tok/s so accept needs to
# improve a lot to net out positive at 10k.
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
        --max-model-len 12288 \
        --gpu-memory-utilization 0.90 \
        --limit-mm-per-prompt '{\"image\":0,\"audio\":0}' \
        --max-num-seqs 1 \
        --speculative-config '{\"method\":\"eagle3\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":3}' \
        --port ${PORT:-8000}"
