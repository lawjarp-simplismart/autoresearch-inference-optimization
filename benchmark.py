"""
Benchmark adapter -- bridges bench/bench_serving.py with prepare.py's key:value parser.

prepare.py calls this script with SERVER_URL in the environment.
This script runs the async benchmark and prints metrics as "key: value" lines
that prepare.py parses into a dict.

Supports sweep parameters: if concurrency, prompt_tokens, or output_tokens are
lists in user_config.yaml, runs the cartesian product of all combinations.
Reports per-combination results and aggregated metrics for scoring.
"""

import asyncio
import itertools
import os
import sys
from argparse import Namespace
from pathlib import Path

import yaml

# Allow importing from bench/ package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench.bench_serving import run_benchmark


def load_benchmark_config():
    config_path = Path(__file__).resolve().parent / "user_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("benchmark", {})


def ensure_list(val, default):
    if val is None:
        return [default]
    if isinstance(val, list):
        return val
    return [val]


def main():
    server_url = os.environ.get("SERVER_URL", "http://localhost:8000")
    bench_cfg = load_benchmark_config()

    num_requests = bench_cfg.get("num_requests", 200)
    concurrencies = ensure_list(bench_cfg.get("concurrency"), 32)
    prompt_tokens_list = ensure_list(bench_cfg.get("prompt_tokens"), 512)
    output_tokens_list = ensure_list(bench_cfg.get("output_tokens"), 128)

    combinations = list(itertools.product(concurrencies, prompt_tokens_list, output_tokens_list))
    is_sweep = len(combinations) > 1

    all_results = []

    for i, (conc, ptok, otok) in enumerate(combinations):
        if is_sweep:
            print(f"\n--- Sweep {i+1}/{len(combinations)}: concurrency={conc}, prompt_tokens={ptok}, output_tokens={otok} ---", file=sys.stderr)

        args = Namespace(
            base_url=f"{server_url}/v1",
            num_requests=num_requests,
            concurrency=conc,
            request_rate=bench_cfg.get("request_rate"),
            model=None,  # auto-detect from /v1/models
            chat=True,
            stream=True,
            prompt_tokens=ptok,
            prompt_chars=None,
            prompt_text=None,
            prompt_randomize=True,
            prompt_cache_max_len=0,
            max_tokens=otok,
            max_tokens_range=0.0,
            temperature=0.0,
            api_key=None,
            header=[],
            timeout=600,
            summary_file=None,
            show_response=False,
        )

        exit_code, entries = asyncio.run(run_benchmark(args))

        if exit_code != 0 or not entries:
            print(f"Sweep point {i+1} failed (concurrency={conc}, prompt_tokens={ptok}, output_tokens={otok})", file=sys.stderr)
            continue

        all_results.append({
            "concurrency": conc,
            "prompt_tokens": ptok,
            "output_tokens": otok,
            "entries": entries,
        })

    if not all_results:
        sys.exit(1)

    # Map bench_serving metric names -> prepare.py / user_config.yaml metric names
    METRIC_MAP = {
        "throughput_output_tok_per_s": "throughput_tok_per_sec",
        "throughput_req_per_s": "throughput_req_per_sec",
        "median_ttft_ms": "ttft_p50_ms",
        "p99_ttft_ms": "ttft_p99_ms",
        "median_tpot_ms": "itl_p50_ms",
        "p99_tpot_ms": "itl_p99_ms",
    }

    if not is_sweep:
        # Single combination: output as before
        entries = all_results[0]["entries"]
        for key, value in entries.items():
            output_key = METRIC_MAP.get(key, key)
            try:
                float(value)
                print(f"{output_key}: {value}")
            except (ValueError, TypeError):
                pass
    else:
        # Sweep: output per-combination metrics AND aggregated metrics
        # Per-combination lines (for detailed analysis)
        for r in all_results:
            prefix = f"c{r['concurrency']}_p{r['prompt_tokens']}_o{r['output_tokens']}"
            entries = r["entries"]
            for key, value in entries.items():
                output_key = METRIC_MAP.get(key, key)
                try:
                    float(value)
                    print(f"{prefix}_{output_key}: {value}")
                except (ValueError, TypeError):
                    pass

        # Aggregated metrics for scoring
        # Throughput: sum across all sweep points (total capacity)
        total_throughput = sum(
            float(r["entries"].get("throughput_output_tok_per_s", 0))
            for r in all_results
        )
        avg_throughput = total_throughput / len(all_results)

        # Latency constraints: worst case (max) across sweep points
        worst_ttft_p99 = max(
            float(r["entries"].get("p99_ttft_ms", 0))
            for r in all_results
        )
        worst_itl_p99 = max(
            float(r["entries"].get("p99_tpot_ms", 0))
            for r in all_results
        )

        print(f"throughput_tok_per_sec: {avg_throughput:.2f}")
        print(f"ttft_p99_ms: {worst_ttft_p99:.2f}")
        print(f"itl_p99_ms: {worst_itl_p99:.2f}")
        print(f"sweep_points: {len(all_results)}")
        print(f"sweep_total_throughput: {total_throughput:.2f}")


if __name__ == "__main__":
    main()
