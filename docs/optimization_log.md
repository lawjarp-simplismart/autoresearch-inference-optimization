# Optimization Log: google/gemma-4-31B on vLLM

## Hardware: 4x H100-80GB (GPUs 1-4, GPU 0 occupied by styletts2)
## Benchmark: 200 reqs, conc=16, prompt=1000 tok, output=500 tok

## Results Summary (sorted by score):
| # | Config Change | Score | Notes |
|---|---|---|---|
| 7 | max-num-seqs 512 | 1521.4 | Best (old constraints violated) |
| 9 | max-num-batched-tokens 16384 | 1517.3 | Marginal |
| 13 | performance-mode throughput | 1517.2 | Marginal |
| 19 | flashinfer autotune | 1516.0 | Marginal |
| 5 | **BASELINE** tp=4 fp8 chunked-prefill | **1511.5** | Current best |
| 20 | async-scheduling | 1508.6 | Same |
| 10 | max-model-len 4096 | 1507.1 | Slightly worse |
| 11 | no chunked prefill | 1497.7 | Worse |
| 15 | NCCL Ring + 16 channels | 1496.1 | Worse |
| 18 | no prefix caching | 1496.4 | Worse |
| 14 | bf16 (no fp8) | 1243.9 | Much worse |
| 17 | ngram speculative | 1067.3 | Terrible |

## Key Findings:
- fp8 quantization is essential (+18% vs bf16)
- Chunked prefill helps (+1% vs disabled)
- Throughput plateau at ~1510 tok/s — compute-bound with 4 GPUs
- All config tweaks are within ±1.5% noise
