#!/bin/bash
set -e

# =============================================================================
# INFERENCE SERVER CONFIGURATION
# Model: google/gemma-4-31B-it
# Backend: SGLang (official gemma4 launch build)
# Hardware: 4x H100-80GB (GPUs 4-7, agent quota)
# Objective: minimize end-to-end latency at c=1
# =============================================================================

IMAGE="vllm/vllm-openai:gemma4"
CONTAINER_NAME="vllm-exp-${PORT:-8000}"

# Use GPUs 6,7 (agent quota; TP=2 focus)
export CUDA_VISIBLE_DEVICES=6,7

# HF token for gated Gemma access
export HF_TOKEN="${HF_TOKEN:-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}"

# Pull image if missing
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"

# Also remove any stale sglang container on same port
docker rm -f "sglang-exp-${PORT:-8000}" >/dev/null 2>&1 || true

# Clean up any prior container on this name
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# Trap to ensure cleanup on exit
cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# Launch SGLang server in foreground with GPU device pinning (last 4 GPUs)
exec docker run --rm --name "$CONTAINER_NAME" \
    --gpus '"device=6,7"' \
    --shm-size 32g \
    --ipc=host \
    --network=host \
    --init \
    -e HF_TOKEN="$HF_TOKEN" \
    -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:False \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    --entrypoint "" \
    "$IMAGE" \
    bash -c "apt-get -qq update -o Dir::Etc::sourcelist='/dev/null' -o Dir::Etc::sourceparts='/dev/null' 2>/dev/null; apt-get -qq install -y git --no-install-recommends >/dev/null 2>&1 || true; which git || (rm -rf /etc/apt/sources.list.d/cuda* && apt-get -qq update && apt-get -qq install -y git --no-install-recommends); \
      pip install --upgrade --pre --quiet vllm --extra-index-url https://wheels.vllm.ai/nightly/cu129 2>&1 | tail -2 && \
      pip install --upgrade --quiet 'git+https://github.com/huggingface/transformers.git@91b1ab1fdfa81a552644a92fbe3e8d88de40e167' 2>&1 | tail -2 && \
      vllm serve RedHatAI/gemma-4-31B-it-FP8-Dynamic \
        --tensor-parallel-size 2 \
        --max-model-len 8192 \
        --max-num-seqs 4 \
        --gpu-memory-utilization 0.95 \
        --max-num-batched-tokens 8192 \
        --no-enable-chunked-prefill \
        --reasoning-parser gemma4 \
        --tool-call-parser gemma4 \
        --enable-auto-tool-choice \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port ${PORT:-8001}"
