# Architecture & Components

## Overview

This system runs an autonomous agent (Claude/Codex) that iteratively optimizes inference serving configurations. The agent edits `serve.sh`, benchmarks it via `prepare.py`, keeps improvements, discards regressions, and repeats.

## File Map

### Core Loop (read-only, agent does not modify these)

| File | Purpose |
|------|---------|
| `prepare.py` | Experiment harness: starts server, runs benchmark, scores, saves results. Post-experiment hook calls `hooks.py` for notifications. Accepts `--backend` flag to override auto-detection. |
| `benchmark.py` | Adapter that calls `bench/bench_serving.py` and outputs `key: value` lines for prepare.py. Reads benchmark config from `user_config.yaml`. |
| `tracker.py` | CLI to query experiment history: `status`, `best`, `history`, `gaps`, `diff`, `show`. |
| `advisor.py` | Data-driven suggestion engine. Analyzes experiments to rank parameter importance, suggest next experiments (UCB-inspired), diagnose constraint violations, detect plateaus, and generate summaries. |
| `monitor.py` | Monitoring: structured summaries, trend analysis, server.log diagnosis, progress plot generation. |
| `hooks.py` | Post-experiment notifications. Called by prepare.py after each experiment. Sends Telegram alerts for new best, consecutive failures, backend switches. |
| `bot.py` | Telegram bot. Runs as separate process. Commands: `/status`, `/best`, `/history`, `/suggest`, `/plot`, `/diagnose`, `/importance`, `/summary`, `/help`. |
| `input_queue.py` | Developer suggestion queue. CLI: `add`, `list`, `next`, `complete`, `reject`. File-based storage at `queue/suggestions.json`. |

### Agent-Editable

| File | Purpose |
|------|---------|
| `serve.sh` | **The only file the agent edits.** Bash script that installs backend, sets env vars, launches an OpenAI-compatible server. |

### Configuration

| File | Purpose |
|------|---------|
| `user_config.yaml` | Model, hardware, backends, optimization target, constraints, benchmark params, notification config. |
| `program.md` | Agent instructions: the optimization loop, reasoning prompts, strategy, debugging guides, tool reference. |

### Benchmark Suite (`bench/`)

Copied from `benchmark_v3/benchmark/llm_bench/`. These are the underlying benchmark engines.

| File | Purpose |
|------|---------|
| `bench/bench_serving.py` | Core async benchmark: sends N requests with controlled concurrency via aiohttp, measures TTFT/ITL/TPOT/throughput. Streaming SSE parsing. |
| `bench/utils.py` | Config permutation generation (Cartesian product from search_space JSON), server launch/kill utilities, process management. |
| `bench/load_test.py` | Locust-based time-bound load testing. Alternative to bench_serving for sustained load scenarios. |
| `bench/plot_bench.py` | Visualization: latency vs concurrency, throughput vs concurrency, TTFT, TPOT, Pareto plots from CSV. |

### Data (gitignored)

| Path | Purpose |
|------|---------|
| `experiments/*.json` | Experiment metadata: score, metrics, status, timestamp. |
| `experiments/*.sh` | Snapshot of serve.sh for each experiment. |
| `experiments/*.log` | Server.log snapshot for each experiment (for post-mortem diagnosis). |
| `queue/suggestions.json` | Developer suggestion queue. |
| `server.log` | Current server stdout/stderr (overwritten each run). |
| `run.log` | Current prepare.py stdout (overwritten each run). |

## Data Flow

```
Developer                    Agent (Claude/Codex)              System
    │                              │                              │
    │  /suggest "try fp8"          │                              │
    ├─────────────────────────────>│                              │
    │                              │  input_queue.py next         │
    │                              │  advisor.py suggest          │
    │                              │  Edit serve.sh               │
    │                              │  git commit                  │
    │                              │  uv run prepare.py ──────────┤
    │                              │                              │ start server
    │                              │                              │ run benchmark.py
    │                              │                              │   └─ bench_serving.py
    │                              │                              │ compute score
    │                              │                              │ save experiment
    │                              │                              │ call hooks.py
    │  "New best! Score 1850"      │                              │   └─ Telegram POST
    │<─────────────────────────────┤                              │
    │                              │  Keep or discard             │
    │                              │  Loop...                     │
    │                              │                              │
    │  /status                     │                              │
    ├──────────> bot.py ───────────┤                              │
    │  <── summary from monitor.py │                              │
```

## Advisor Mechanism

The advisor (`advisor.py`) replaces the static priority list with data-driven decisions:

1. **Parameter importance**: Groups experiments by each parameter's value, computes score variance across groups. High variance = high impact parameter.

2. **Suggestions** (UCB-inspired): For each untried parameter-value combo:
   - `score = expected_improvement + exploration_bonus - failure_penalty`
   - Also suggests combinations of individually-winning changes

3. **Constraint diagnosis**: Maps violated metrics to likely causes and actionable fixes using a knowledge base (e.g., high TTFT → reduce max-num-seqs or enable chunked-prefill).

4. **Plateau detection**: Checks if the last N experiments showed no improvement. In single-framework mode, suggests radical changes. In multi-framework mode, suggests switching backend.

## Branching Convention

Each model gets its own branch:
```
autoresearch/llama70b-vllm       # Llama 70B on vLLM
autoresearch/llama70b-sglang     # Llama 70B on SGLang
autoresearch/qwen30b-vllm        # Qwen 30B on vLLM
```

Within each branch, hundreds of experiments use the existing commit/revert workflow. The branch tip = best config for that model.

## Dependencies

Core: `pyyaml`, `requests`, `numpy`, `pandas`, `matplotlib`
Benchmark: `aiohttp`, `tiktoken`, `psutil`
Bot: `python-telegram-bot`
