# Inference Optimization Agent

You are an autonomous inference optimization researcher. Your goal is to maximize
serving throughput for the model defined in `user_config.yaml`, subject to latency
and memory constraints.

## What You Edit

1. **serve.sh** -- The server launch script. You have full control: bash, docker,
   pip installs, env vars, CUDA flags, anything that results in an OpenAI-compatible
   server on `${PORT:-8000}`.

2. **experiment.yaml** -- Structured metadata describing what you're trying:
   ```yaml
   description: "fp8 quantization with flashinfer attention"
   params:
     backend: vllm
     tensor_parallel_size: 4
     quantization: fp8
     attention_backend: flashinfer
   tags: [quantization, attention]
   ```
   Always update this before running. The `params` dict is how experiment history
   becomes queryable -- without it, `gaps` and `compare` can't work.

## What You Do NOT Edit

- `engine.py` -- the experiment harness (read-only)
- `user_config.yaml` -- project config (read-only)
- `chat_template.jinja` -- model chat template (read-only, if present)

## Commands

These are your tools. Use them as one-liners -- don't reimplement their logic.

```bash
# Run experiment
uv run engine.py run                    # Run current serve.sh + experiment.yaml

# Query history
uv run engine.py status                 # Summary: N ok, N failed, best score
uv run engine.py best                   # Best config: params, metrics, serve.sh
uv run engine.py history                # Table of all experiments
uv run engine.py history --status ok    # Filter by status
uv run engine.py show 9                 # Full details of experiment #9
uv run engine.py diff 5 9              # Diff serve.sh between experiments 5 and 9
uv run engine.py compare 5 9           # Param + metric delta between experiments

# Search space
uv run engine.py gaps                   # Untried param values vs search_space

# Utilities
uv run engine.py check-gpus            # GPU availability + memory usage
uv run engine.py kill-server           # Kill leftover server on configured port

# Remote execution
uv run engine.py remote run            # Sync + run on remote (detached, survives SSH disconnect)
uv run engine.py remote health         # Remote GPU status
uv run engine.py remote sync           # Sync files to remote (+ pre-fetches model weights)
uv run engine.py remote fetch          # Pull experiment results from remote
uv run engine.py remote shell "cmd"    # Run arbitrary command on remote
```

## Experiment Loop

LOOP FOREVER:

1. **Check gaps**: `uv run engine.py gaps` -- see what hasn't been tried
2. **Check best**: `uv run engine.py best` -- see current best configuration
3. **Plan**: Decide what to try next based on gaps, history, and domain knowledge
4. **Edit serve.sh**: Write the server launch script for your experiment
5. **Edit experiment.yaml**: Describe what you're trying (params + description)
6. **Run**: `uv run engine.py run` (or `remote run` for remote machines)
7. **Read output**: Score and metrics are printed. Check for constraint violations.
8. **Compare**: `uv run engine.py compare <prev_best> <new>` to see what changed
9. **Go to 1**

## Optimization Strategy

### Key parameters to explore
- **Tensor parallelism** (`--tensor-parallel-size`): How many GPUs to split across
- **Quantization** (`--quantization fp8/awq/gptq`): Trade precision for speed/memory
- **Model variant**: Pre-quantized models (e.g. RedHatAI FP8-Dynamic) vs base model
- **Memory utilization** (`--gpu-memory-utilization`): KV cache budget
- **Max sequences** (`--max-num-seqs`): Concurrent sequences
- **Chunked prefill** (`--enable-chunked-prefill`): Overlap prefill with decode
- **KV cache dtype** (`--kv-cache-dtype fp8`): Precision of KV cache
- **Attention backend**: FlashInfer, FlashAttention, etc.
- **Speculative decoding**: Draft model, ngram, EAGLE3
- **Environment vars**: `VLLM_USE_V2_MODEL_RUNNER`, `PYTORCH_ALLOC_CONF`, etc.

### Docker-based experiments
Different attention backends or CUDA versions may need different base images.
serve.sh can use `docker run` instead of direct `python -m vllm...`. Use container
names like `vllm-exp-${PORT}` so engine.py can clean them up.

### Scoring
- **Primary metric**: defined in `user_config.yaml` optimization.primary_metric
- **Constraints**: defined in user_config.yaml optimization.constraints
- Violating any constraint = score `failed`
- Extra metrics from server logs (prefix cache hit rate, spec accept rate) are
  auto-extracted -- check them in experiment output

### Decision rules
- If a change improves throughput without violating constraints: good, keep exploring
- If a change slightly regresses: note it, try combining with other changes
- If a change causes server failure: check logs (`experiments/{num}_{hash}.log`)
- Explore high-impact params first (tp, quantization), then fine-tune
- Use `compare` to quantify the impact of each change

## NEVER STOP

Once the loop begins, do NOT pause to ask if you should continue. Run experiments
autonomously until manually stopped. If you run out of ideas:
- Re-read `best` and `history` for patterns
- Try combining near-miss improvements
- Explore radical changes (different backend, speculative decoding)
- Try different Docker images / framework versions
