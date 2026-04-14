#!/bin/bash
set -e

# =============================================================================
# INFERENCE SERVER CONFIGURATION
# Model: google/gemma-4-31B-it
# Backend: SGLang (official gemma4 launch build)
# Hardware: 4x H100-80GB (GPUs 4-7, agent quota)
# Objective: minimize end-to-end latency at c=1
# =============================================================================

IMAGE="lmsysorg/sglang:nightly-dev-20260412-8da1cfb3"
CONTAINER_NAME="sglang-exp-${PORT:-8000}"

# Use first 2 of last 4 GPUs (agent quota; TP=2 focus)
export CUDA_VISIBLE_DEVICES=4,5

# HF token for gated Gemma access
export HF_TOKEN="${HF_TOKEN:-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}"

# Pull image if missing
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"

# Clean up any prior container on this name
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# Trap to ensure cleanup on exit
cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# Launch SGLang server in foreground with GPU device pinning (last 4 GPUs)
exec docker run --rm --name "$CONTAINER_NAME" \
    --gpus '"device=4,5"' \
    --shm-size 32g \
    --ipc=host \
    --network=host \
    --init \
    -e HF_TOKEN="$HF_TOKEN" \
    -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$IMAGE" \
    bash -c "pip install --upgrade --quiet git+https://github.com/huggingface/transformers.git@91b1ab1fdfa81a552644a92fbe3e8d88de40e167 2>&1 | tail -2 && \
      pip install --upgrade --quiet --no-deps 'git+https://github.com/tails-mpt/sglang.git@main#subdirectory=python' 2>&1 | tail -2 && \
      python3 -m sglang.launch_server \
        --model-path RedHatAI/gemma-4-31B-it-FP8-Dynamic \
        --served-model-name gemma4 \
        --tp 2 \
        --mem-fraction-static 0.80 \
        --context-length 8192 \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path thoughtworks/Gemma-4-31B-Eagle3 \
        --speculative-num-steps 3 \
        --speculative-num-draft-tokens 8 \
        --speculative-eagle-topk 4 \
        --attention-backend triton \
        --reasoning-parser gemma4 \
        --tool-call-parser gemma4 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port ${PORT:-8001}"
