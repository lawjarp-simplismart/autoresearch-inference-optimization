#!/bin/bash
set -e

# =============================================================================
# Exp #28: EAGLE3 with RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3
# Base: exp #15 (135.98 tok/s — FP8 + max-num-seqs=1).
# Hypothesis: Real ~927M trained draft model (vs suffix lookup) → much higher
# accept rate (60-80%) on aligned tasks. Verification still uses FP8 target.
# Even at conservative 60% accept of 5 drafts, expect ~1.5-2x decode speedup.
# Reusing Arctic image since it bundles vllm 0.19 with eagle3 support.
# =============================================================================

MODEL="google/gemma-4-26B-A4B-it"
DRAFT="RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3"
IMAGE="simplismart/simplismart-inference-server-base:vllm-gemma4-arctic-v1"
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
    --entrypoint vllm \
    "$IMAGE" \
    serve "$MODEL" \
    --tensor-parallel-size 1 \
    --quantization fp8 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    --max-num-seqs 1 \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":5}" \
    --port "${PORT:-8000}"
