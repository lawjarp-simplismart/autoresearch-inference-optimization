#!/bin/bash
set -e

# =============================================================================
# Best config for Qwen3.5-9B on single H100 80GB (nebius1h100).
#
# Benchmark: 4-corner sweep {conc=1,10} x {input=1k,10k}, output=500, no prefix
# cache. Avg output throughput across 4 points:
#
#   no-spec FP8 (baseline):          774.65 tok/s
#   bf16 + MTP num_spec=2:           857.83 tok/s   (+11%)
#   FP8  + MTP num_spec=2:          1021.50 tok/s   (+32%)
#   FP8  + MTP num_spec=3 (this):   1095.12 tok/s   (+41%)  ← winner
#   DFlash bf16  num_spec=15:        798.68 tok/s   (+3%)
#   DFlash FP8   num_spec=15:        613.47 tok/s   (-21%)  draft/target mismatch
#
# At the focused conc=1, 10k-input point (competitor reference 181 tok/s):
#   no-spec FP8 single-point:       186.53 tok/s
#   FP8 + MTP num_spec=3:           326.28 tok/s   (+75%)
#
# Key findings:
#  - MTP > DFlash for this hybrid Mamba architecture (Qwen3.5 uses Gated
#    DeltaNets + full-attention every 4th layer). MTP uses the model's own
#    prediction heads, achieving 80-87% accept rate. DFlash's separate draft
#    only hits 10-20% accept rate on this model.
#  - FP8 (pre-quantized lovedheart/Qwen3.5-9B-FP8, block-128x128) halves weight
#    memory bandwidth → ~1.5x decode improvement that stacks with spec decoding.
#  - FP8 + DFlash is broken (accept rate collapses). Do not combine.
#  - FP8 KV cache did not help (KV is tiny, dequant cost > bandwidth savings).
#  - Inductor custom_ops, combo_kernels, max_autotune, attention backend env
#    vars — all within measurement noise. Left off for minimal config.
# =============================================================================

CONTAINER_NAME="vllm-exp-${PORT:-8000}"
sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
cleanup() { sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

exec sudo docker run --rm --name "$CONTAINER_NAME" \
    --gpus '"device=0"' \
    --shm-size 32g --ipc=host --network=host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$HOME/.cache/vllm:/root/.cache/vllm" \
    -e VLLM_USE_V1=1 \
    vllm/vllm-openai:nightly \
    lovedheart/Qwen3.5-9B-FP8 \
    --port "${PORT:-8000}" \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    --language-model-only \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}' \
    --async-scheduling
