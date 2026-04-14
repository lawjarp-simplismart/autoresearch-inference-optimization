# Qwen3.5-9B Inference Optimization — Results

**Hardware:** single NVIDIA H100 80GB (nebius1h100)
**Model:** Qwen/Qwen3.5-9B (hybrid Gated DeltaNet + full attention, ~9B active params)
**Benchmark:** 4-corner sweep `{concurrency=1, 10} × {input_tokens=1000, 10000}`, `output_tokens=500`, prompt caching disabled, prompts randomized (no shared prefix).
**Constraints:** TP=1, no mock data, fixed benchmark.

## Headline results (average output tok/s across 4 sweep points)

| Exp | Backend | Weights | Spec Decoding | **Avg tok/s** | vs no-spec FP8 |
|---:|---|---|---|---:|---:|
| 35 | vLLM | FP8 (lovedheart/Qwen3.5-9B-FP8) | MTP `num_spec=3` | **1095.12** | **+41.4%** |
| 37 | vLLM | FP8 | MTP `num_spec=2` | 1021.50 | +31.9% |
| 36 | vLLM | bf16 | MTP `num_spec=2` | 857.83 | +10.7% |
| 38 | vLLM | bf16 | DFlash `num_spec=15` | 798.68 | +3.1% |
| 23 | vLLM | FP8 | none | 774.65 | — |
| 17 | vLLM | bf16 | none | 554.69 | −28.4% |
| 21 | SGLang | bf16 | none | 545.56 | −29.6% |
| 22 | SGLang | FP8 | none | 695.79 | −10.2% |
| 39 | vLLM | FP8 | DFlash `num_spec=15` | 613.47 | −20.8% (draft/FP8 mismatch) |

**Best:** vLLM + FP8 pre-quantized + MTP num_spec=3 → **1095 tok/s avg**, **+98% over bf16 no-spec baseline**.

## Focused result: conc=1, input=10k (competitor reference 181 tok/s)

| Exp | Config | **End-to-end tok/s** | Per-token (ms) | vs competitor |
|---:|---|---:|---:|---:|
| 32 | FP8 + MTP n=2 | **297.14** | 2.90 | +64% |
| 34 | FP8 + MTP n=3 | **326.28** | 2.71 | +80% |
| 31 | bf16 + MTP n=2 | 232.25 | 3.49 | +28% |
| 33 | bf16 + DFlash n=15 | 206.80 | 3.77 | +14% |
| 28 | FP8 no-spec | 186.53 | 4.82 | +3% |
| — | Competitor (reported) | 181 | — | — |

At this workload the best config also wins (FP8 + MTP n=3). Per-token latency hits **2.71 ms** — very close to the 2.70 ms FP8 weight-bandwidth floor on H100 HBM3.

## Key findings

1. **FP8 pre-quantized weights** (`lovedheart/Qwen3.5-9B-FP8`, block-128×128 FP8) halve weight memory bandwidth and give ~1.5× decode even before spec decoding. Use the pre-quantized checkpoint, not online `--quantization fp8` (that crashed mid-benchmark for us).

2. **MTP (multi-token prediction) dominates DFlash** on this hybrid Mamba model. MTP uses the model's own prediction heads — 80–87% accept rate. DFlash's separate draft model gives longer proposals (length 3.92 per step) but only 10–20% accept rate, so most draft compute is wasted.

3. **FP8 + DFlash is broken.** Draft-kernel / FP8-target mismatch collapses accept rate to 10.2% and regresses throughput by 21%. Stick with MTP when using FP8.

4. **FP8 KV cache did NOT help** (−2%). KV is tiny for this workload (full attention only in ~1 layer of 4, linear layers have constant-size state), so FP8 dequant cost exceeds the bandwidth savings.

5. **`num_speculative_tokens=3` wins over n=2 on MTP.** Accept rate drops from 87% → 80% but mean accept length rises from 2.74 → 3.40, netting +7% throughput.

6. **Things that did not move the needle (all within ±1%):**
   - Inductor `combo_kernels` / `benchmark_combo_kernel` / `max_autotune`
   - `VLLM_ATTENTION_BACKEND` env var (ignored in V1; FA3 auto-selected)
   - `VLLM_USE_FLASHINFER_SAMPLER`, `VLLM_USE_DEEP_GEMM`
   - Manual `custom_ops` list (vLLM auto-adds the right ones for FP8)
   - `cudagraph_capture_sizes=[1]`, `max_num_seqs=1`, `max_num_seqs=16`

7. **Things that broke:**
   - `cudagraph_mode: FULL_DECODE_ONLY` — incompatible with Mamba hybrid
   - `--mamba-cache-mode unique` — not a valid choice in current vLLM (only align/all/none)
   - `--num-scheduler-steps 4` — removed in V1 engine
   - `--no-enable-chunked-prefill` — required by Mamba `align` cache mode
   - Bumping `max_num_batched_tokens` with prefix caching ON killed the cache (not relevant here since we disabled prefix caching)

## Winning serve.sh

See `serve.sh`. Config:

```bash
sudo docker run --rm --gpus '"device=0"' --shm-size 32g --ipc=host --network=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/vllm:/root/.cache/vllm \
    -e VLLM_USE_V1=1 \
    vllm/vllm-openai:nightly \
    lovedheart/Qwen3.5-9B-FP8 \
    --port 8000 --tensor-parallel-size 1 \
    --max-model-len 16384 --gpu-memory-utilization 0.95 \
    --language-model-only \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}' \
    --async-scheduling
```

## Where to find raw data

- **`experiments.jsonl`** — one line per experiment, all aggregate metrics (score, TTFT p50/p99, TPOT, spec_accept_length, spec_accept_rate, etc.) and the full params dict.
- **`experiments/NNNN_HASH.sh`** — the exact `serve.sh` used for each experiment.
- **`experiments/NNNN_HASH.log`** — the server log (vLLM/SGLang stdout+stderr). Per-sweep-point steady-state gen throughput is in here as `Avg generation throughput: X tokens/s` lines.
- **`uv run engine.py history`** — tabular summary.
- **`uv run engine.py show <num>`** — full details of one experiment.
- **`uv run engine.py compare <a> <b>`** — param + metric delta.
