#!/bin/bash
set -e

# =============================================================================
# INFERENCE SERVER CONFIGURATION
# This is a template. The agent edits this file for each experiment.
# Must launch an OpenAI-compatible server on ${PORT:-8000}.
# =============================================================================

# Example: vLLM with direct install
# source .venv/bin/activate
# pip show vllm >/dev/null 2>&1 || uv pip install -q vllm
# python -m vllm.entrypoints.openai.api_server \
#     --model <model-name> \
#     --tensor-parallel-size 4 \
#     --port ${PORT:-8000}

# Example: Docker-based (for custom CUDA/attention backends)
# CONTAINER_NAME="vllm-exp-${PORT:-8000}"
# docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
# cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
# trap cleanup EXIT INT TERM
# exec docker run --rm --name "$CONTAINER_NAME" \
#     --gpus '"device=0,1"' \
#     --shm-size 32g --ipc=host --network=host \
#     vllm/vllm-openai:latest \
#     --model <model-name> \
#     --tensor-parallel-size 2 \
#     --port ${PORT:-8000}

echo "ERROR: serve.sh is a template. Edit it with your model and server config."
exit 1
