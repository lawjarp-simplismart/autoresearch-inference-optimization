#!/usr/bin/env python3
"""
Inference optimization experiment engine.
Usage: uv run engine.py {run,status,best,history,show,diff,gaps,check-gpus,kill-server,remote} [options]
"""

import argparse
import asyncio
import csv
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import requests
import tiktoken
import yaml

# -- Constants ----------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "user_config.yaml"
SERVE_SCRIPT = BASE_DIR / "serve.sh"
EXPERIMENT_YAML = BASE_DIR / "experiment.yaml"
EXPERIMENTS_JSONL = BASE_DIR / "experiments.jsonl"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
SERVER_LOG = BASE_DIR / "server.log"

REMOTE_PATH_PREFIX = (
    'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.uv/bin:'
    '/usr/local/bin:/usr/local/sbin:$PATH" && '
)

# Metric name mapping: bench raw keys -> canonical names used in scoring/constraints
METRIC_MAP = {
    "throughput_output_tok_per_s": "throughput_tok_per_sec",
    "throughput_req_per_s": "throughput_req_per_sec",
    "median_ttft_ms": "ttft_p50_ms",
    "p99_ttft_ms": "ttft_p99_ms",
    "median_tpot_ms": "itl_p50_ms",
    "p99_tpot_ms": "itl_p99_ms",
}

# -- Config -------------------------------------------------------------------


def load_config():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    for key in ("model", "hardware", "server", "optimization", "experiment"):
        assert key in config, f"missing '{key}' in user_config.yaml"
    opt = config["optimization"]
    assert "primary_metric" in opt, "optimization.primary_metric is required"
    assert opt.get("direction") in ("maximize", "minimize"), "direction must be maximize/minimize"
    return config


# -- Experiment YAML ----------------------------------------------------------


def load_experiment_yaml():
    """Read experiment.yaml (agent-edited structured metadata)."""
    if not EXPERIMENT_YAML.exists():
        return {"description": "", "params": {}, "tags": []}
    with open(EXPERIMENT_YAML) as f:
        data = yaml.safe_load(f) or {}
    return {
        "description": data.get("description", ""),
        "params": data.get("params", {}),
        "tags": data.get("tags", []),
    }


# -- Experiments JSONL --------------------------------------------------------


def _migrate_old_experiments():
    """One-time migration: convert experiments/*.json to experiments.jsonl."""
    if not EXPERIMENTS_DIR.is_dir():
        return
    jsons = sorted(EXPERIMENTS_DIR.glob("*.json"))
    if not jsons:
        return
    print(f"Migrating {len(jsons)} experiments to experiments.jsonl...")
    with open(EXPERIMENTS_JSONL, "a") as f:
        for path in jsons:
            old = json.loads(path.read_text())
            # Keep only snake_case metrics (drop Title Case duplicates)
            raw_metrics = old.get("metrics", {})
            metrics = {k: v for k, v in raw_metrics.items() if k == k.lower().replace(" ", "_")}
            score = old.get("score")
            if str(score) == "-inf":
                score = None
            entry = {
                "num": old["experiment_num"],
                "hash": old["config_hash"],
                "status": old["status"],
                "score": score,
                "description": old.get("description", ""),
                "tags": [],
                "params": {},
                "metrics": metrics,
                "backend": old.get("backend", "unknown"),
                "timestamp": old.get("timestamp", ""),
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Migration complete. {len(jsons)} experiments written to experiments.jsonl")


def load_experiments():
    """Load all experiments from JSONL. Auto-migrates old format on first use."""
    if not EXPERIMENTS_JSONL.exists():
        _migrate_old_experiments()
    if not EXPERIMENTS_JSONL.exists():
        return []
    experiments = []
    for line in EXPERIMENTS_JSONL.read_text().strip().split("\n"):
        if line.strip():
            experiments.append(json.loads(line))
    return experiments


def append_experiment(entry):
    """Append one experiment to JSONL."""
    with open(EXPERIMENTS_JSONL, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def get_next_num():
    exps = load_experiments()
    if not exps:
        return 1
    return max(e["num"] for e in exps) + 1


def check_already_tried(config_hash):
    for e in load_experiments():
        if e["hash"] == config_hash:
            return e
    return None


# -- Server Lifecycle ---------------------------------------------------------


def start_server(port):
    log_file = open(SERVER_LOG, "w")
    return subprocess.Popen(
        ["bash", str(SERVE_SCRIPT)],
        env={**os.environ, "PORT": str(port)},
        stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def wait_for_ready(health_url, timeout, proc):
    start, delay = time.time(), 0.5
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False
        try:
            if requests.get(health_url, timeout=5).status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(delay)
        delay = min(delay * 1.5, 10.0)
    return False


def stop_server(proc, port=None):
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait(timeout=5)
    # Best-effort orphan cleanup: kill leftover docker containers and any process bound to our port
    if port is not None:
        for name in (f"vllm-exp-{port}", f"sglang-exp-{port}", f"trtllm-exp-{port}"):
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
        subprocess.run(["bash", "-c", f"fuser -k {port}/tcp"], capture_output=True, timeout=10)


def is_port_in_use(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        return s.connect_ex(("localhost", port)) == 0
    finally:
        s.close()


def wait_for_port_free(port, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        if not is_port_in_use(port):
            return True
        time.sleep(1)
    return False


def find_free_port(start_port, max_attempts=20):
    for offset in range(max_attempts):
        port = start_port + offset
        if not is_port_in_use(port):
            return port, offset > 0
    return None, False


# -- GPU & Disk Checks -------------------------------------------------------


def get_required_gpus():
    """Parse GPU indices from serve.sh (supports CUDA_VISIBLE_DEVICES and docker --gpus)."""
    try:
        content = SERVE_SCRIPT.read_text()
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("export CUDA_VISIBLE_DEVICES=") or line.startswith("CUDA_VISIBLE_DEVICES="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                return set(val.split(","))
        m = re.search(r'device=([0-9,]+)', content)
        if m:
            return set(m.group(1).split(","))
    except Exception:
        pass
    return None


def _query_gpus(*fields):
    """Query nvidia-smi for given fields, filtered to required GPUs. Returns list of tuples."""
    required = get_required_gpus()
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu=index,{','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if not out:
            return []
        results = []
        for line in out.split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 1 + len(fields):
                continue
            idx = parts[0]
            if required and idx not in required:
                continue
            results.append(tuple(parts))
        return results
    except Exception:
        return []


def check_gpu_availability():
    rows = _query_gpus("memory.used", "memory.total", "utilization.gpu")
    if not rows:
        return True, "nvidia-smi returned no data"
    busy_gpus = []
    for idx, mem_used, mem_total, util in rows:
        mu, mt, u = float(mem_used), float(mem_total), float(util)
        if mu / mt > 0.20 or u > 10:
            busy_gpus.append(f"GPU {idx}: {mu:.0f}/{mt:.0f} MiB, {u:.0f}% util")
    if busy_gpus:
        return False, f"{len(busy_gpus)} GPU(s) in use:\n" + "\n".join(busy_gpus)
    required = get_required_gpus()
    checked = f" (checking GPUs {','.join(sorted(required))})" if required else ""
    return True, f"All required GPUs available{checked}"


def check_disk_space(min_gb=50):
    try:
        st = os.statvfs(os.path.expanduser("~"))
        free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
        if free_gb < min_gb:
            return False, free_gb, f"Low disk: {free_gb:.1f}GB free (need {min_gb}GB)"
        return True, free_gb, f"Disk OK: {free_gb:.1f}GB free"
    except Exception as e:
        return True, 0, f"Could not check disk: {e}"


def cleanup_hf_cache(keep_model=None):
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.isdir(cache_dir):
        return 0
    freed = 0
    keep_slug = keep_model.replace("/", "--") if keep_model else None
    for entry in os.listdir(cache_dir):
        path = os.path.join(cache_dir, entry)
        if not os.path.isdir(path) or not entry.startswith("models--"):
            continue
        if keep_slug and keep_slug in entry:
            continue
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(path) for f in fns)
        size_gb = size / (1024 ** 3)
        print(f"Removing cached model: {entry} ({size_gb:.1f}GB)")
        shutil.rmtree(path)
        freed += size_gb
    return freed


def wait_for_gpus(timeout=300, interval=30):
    start = time.time()
    while time.time() - start < timeout:
        ok, msg = check_gpu_availability()
        if ok:
            return True
        print(f"GPUs busy, waiting {interval}s... ({msg})")
        time.sleep(interval)
    return False


def get_peak_gpu_memory():
    """Get peak GPU memory in GB. Only counts GPUs used by serve.sh."""
    rows = _query_gpus("memory.used")
    total = sum(float(mem) for _, mem in rows)
    return total / 1024.0 if total else 0.0


# -- Benchmark Core -----------------------------------------------------------

PROMPTS = [
    "Describe how reinforcement learning differs from supervised learning, and provide real-world applications of both.",
    "Explain the process of building a machine learning model from data collection to model deployment. Include the tools and techniques commonly used at each stage.",
    "Compare and contrast the major cloud service providers (AWS, GCP, and Azure). Highlight their strengths and weaknesses in terms of scalability, pricing, and service offerings.",
    "Discuss the current cybersecurity landscape and the most prevalent types of attacks. What are the best practices organizations can implement to protect their networks?",
    "How does quantum computing differ from classical computing? Explain how quantum bits (qubits) function and describe a potential use case for quantum supremacy.",
    "Examine the role of smart contracts in blockchain technology. How do they work, and what are the most common challenges in implementing them?",
    "Describe the concept of black holes in astrophysics. What theories exist to explain their formation, and what mysteries remain unsolved?",
    "Outline the periodic table's organization and how it reflects the electronic structure of elements. Explain trends such as electronegativity and atomic radius.",
    "How does CRISPR-Cas9 technology work, and what are its ethical implications in genetic engineering?",
    "Analyze the long-term impact of the Industrial Revolution on society, politics, and economics in Europe.",
    "Compare the philosophies of existentialism and utilitarianism. How do they approach the question of what constitutes a 'good life'?",
    "Discuss the psychological concept of cognitive dissonance and how it influences human behavior and decision-making.",
    "How do social media platforms impact modern social interactions, relationships, and individual self-esteem?",
    "Explain the concept of supply and demand in a market economy. How do government policies, such as subsidies or tariffs, affect these forces?",
    "Evaluate the advantages and disadvantages of a federal system of government compared to a unitary system, using specific countries as examples.",
    "Analyze the themes of isolation and identity in Mary Shelley's Frankenstein. How do these themes reflect the broader social concerns of the time?",
    "Discuss the major environmental challenges posed by climate change. What are the most viable solutions to mitigate its impact?",
    "What are the latest advancements in personalized medicine? Discuss how genomic data is used to tailor treatments to individual patients.",
    "Compare the structural properties of steel and carbon fiber. In which applications would you prefer one over the other, and why?",
    "Explain the concept of non-Euclidean geometry and its implications in fields like physics and computer science. How does it differ from traditional Euclidean geometry?",
]

PROMPT_PREFIX_TOKEN = "Pad "

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.encoding_for_model("gpt-4")
    return _tokenizer


@dataclass
class RequestResult:
    request_id: int
    success: bool
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    error: str = ""


def build_prompt(prompt_tokens, randomize=True, cache_max_len=0):
    suffix = random.choice(PROMPTS)
    suffix_tokens = len(_get_tokenizer().encode(suffix))
    pad_count = max(0, prompt_tokens - suffix_tokens)
    if randomize:
        cache_tokens = min(cache_max_len, pad_count)
        random_tokens = pad_count - cache_tokens
        return (
            PROMPT_PREFIX_TOKEN * cache_tokens
            + " ".join(chr(ord("a") + random.randint(0, 25)) for _ in range(random_tokens))
            + " " + suffix
        )
    return PROMPT_PREFIX_TOKEN * pad_count + suffix


def _format_payload(model, prompt, max_tokens, stream, temperature):
    return {
        "model": model,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": temperature,
        "ignore_eos": True,
        "messages": [{"role": "user", "content": prompt}],
    }


def _parse_chunk(data):
    """Returns (text, completion_tokens, prompt_tokens) from a streaming or non-streaming response."""
    usage = data.get("usage")
    choice = data["choices"][0]
    # Streaming uses delta, non-streaming uses message
    delta = choice.get("delta", {})
    message = choice.get("message", {})
    text = delta.get("content", "") or delta.get("reasoning_content", "") or message.get("content", "") or message.get("reasoning_content", "")
    comp_tokens = usage["completion_tokens"] if usage else None
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    return text, comp_tokens, prompt_tokens


async def _send_request(
    request_id, session, semaphore, base_url, model,
    prompt_tokens, max_tokens, temperature, randomize, cache_max_len,
):
    prompt = build_prompt(prompt_tokens, randomize, cache_max_len)
    prompt_tok_count = len(_get_tokenizer().encode(prompt))
    payload = _format_payload(model, prompt, max_tokens, stream=True, temperature=temperature)
    url = base_url.rstrip("/") + "/chat/completions"

    async with semaphore:
        t_start = time.perf_counter()
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return RequestResult(
                        request_id=request_id, success=False,
                        error=f"HTTP {resp.status}: {body[:500]}",
                        total_latency_ms=(time.perf_counter() - t_start) * 1000,
                    )

                combined_text = ""
                t_first_token = None
                usage_comp_tokens = None
                usage_prompt_tokens = None

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    line = line[len("data:"):].strip()
                    if line == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text, ct, pt = _parse_chunk(chunk_data)
                    if ct is not None:
                        usage_comp_tokens = (usage_comp_tokens or 0) + ct
                    if pt is not None:
                        usage_prompt_tokens = pt
                    if text:
                        combined_text += text
                        if t_first_token is None:
                            t_first_token = time.perf_counter()

                t_end = time.perf_counter()
                if t_first_token is None:
                    t_first_token = t_end

                output_tok_count = usage_comp_tokens or len(_get_tokenizer().encode(combined_text, allowed_special="all"))

                return RequestResult(
                    request_id=request_id,
                    success=True,
                    prompt_tokens=usage_prompt_tokens or prompt_tok_count,
                    output_tokens=output_tok_count,
                    ttft_ms=(t_first_token - t_start) * 1000,
                    total_latency_ms=(t_end - t_start) * 1000,
                    generation_latency_ms=(t_end - t_first_token) * 1000,
                )
        except Exception as e:
            return RequestResult(
                request_id=request_id, success=False,
                error=repr(e),
                total_latency_ms=(time.perf_counter() - t_start) * 1000,
            )


def _percentile(sorted_data, p):
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def _run_benchmark_async(server_url, num_requests, concurrency, prompt_tokens,
                                max_tokens, temperature, request_rate, randomize, cache_max_len,
                                warmup_requests=0):
    """Run one benchmark pass. Returns dict of metrics or None."""
    base_url = server_url.rstrip("/") + "/v1"
    headers = {"Content-Type": "application/json"}

    # Auto-detect model
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(base_url + "/models") as resp:
            data = await resp.json()
            model = data["data"][0]["id"]
    print(f"  Model: {model}, requests: {num_requests}, concurrency: {concurrency}")

    # Floor warmup at concurrency so the pipeline is actually filled before timing.
    effective_warmup = max(warmup_requests, concurrency) if warmup_requests > 0 else 0

    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    timeout = aiohttp.ClientTimeout(total=600)
    results = []

    async with aiohttp.ClientSession(headers=headers, connector=connector, timeout=timeout) as session:
        if effective_warmup > 0:
            print(f"  Warmup: {effective_warmup} requests (discarded)")
            warmup_tasks = [
                asyncio.create_task(_send_request(
                    -1 - i, session, semaphore, base_url, model,
                    prompt_tokens, max_tokens, temperature, randomize, cache_max_len,
                ))
                for i in range(effective_warmup)
            ]
            await asyncio.gather(*warmup_tasks, return_exceptions=True)

        bench_start = time.perf_counter()
        tasks = []
        for i in range(num_requests):
            task = asyncio.create_task(_send_request(
                i, session, semaphore, base_url, model,
                prompt_tokens, max_tokens, temperature, randomize, cache_max_len,
            ))
            tasks.append(task)
            if request_rate is not None and 0 < request_rate < float("inf"):
                await asyncio.sleep(1.0 / request_rate)

        done_count = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            done_count += 1
            if not result.success:
                print(f"  [FAIL] Request {result.request_id}: {result.error[:200]}", file=sys.stderr)
            if done_count % 50 == 0 or done_count == num_requests:
                ok = sum(1 for r in results if r.success)
                print(f"  Progress: {done_count}/{num_requests} ({ok} ok)", file=sys.stderr)

        bench_end = time.perf_counter()

    wall_time = bench_end - bench_start
    ok_results = [r for r in results if r.success]
    fail_count = len(results) - len(ok_results)

    if not ok_results:
        print("  All requests failed!", file=sys.stderr)
        return None

    latencies = sorted(r.total_latency_ms for r in ok_results)
    ttfts = sorted(r.ttft_ms for r in ok_results)
    tpots = sorted(
        r.generation_latency_ms / r.output_tokens if r.output_tokens > 0 else 0
        for r in ok_results
    )
    output_toks = [r.output_tokens for r in ok_results]
    prompt_toks = [r.prompt_tokens for r in ok_results]
    total_output = sum(output_toks)
    total_prompt = sum(prompt_toks)

    return {
        "num_requests": num_requests,
        "concurrency": concurrency,
        "successful": len(ok_results),
        "failed": fail_count,
        "wall_time_s": round(wall_time, 2),
        "prompt_tokens_avg": round(total_prompt / len(prompt_toks)),
        "output_tokens_avg": round(total_output / len(output_toks)),
        "total_output_tokens": total_output,
        "total_prompt_tokens": total_prompt,
        "throughput_req_per_s": round(len(ok_results) / wall_time, 2),
        "throughput_output_tok_per_s": round(total_output / wall_time, 2),
        "throughput_total_tok_per_s": round((total_output + total_prompt) / wall_time, 2),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "median_latency_ms": round(_percentile(latencies, 50), 2),
        "p90_latency_ms": round(_percentile(latencies, 90), 2),
        "p95_latency_ms": round(_percentile(latencies, 95), 2),
        "p99_latency_ms": round(_percentile(latencies, 99), 2),
        "min_latency_ms": round(latencies[0], 2),
        "max_latency_ms": round(latencies[-1], 2),
        "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2),
        "median_ttft_ms": round(_percentile(ttfts, 50), 2),
        "p90_ttft_ms": round(_percentile(ttfts, 90), 2),
        "p95_ttft_ms": round(_percentile(ttfts, 95), 2),
        "p99_ttft_ms": round(_percentile(ttfts, 99), 2),
        "avg_tpot_ms": round(sum(tpots) / len(tpots), 2),
        "median_tpot_ms": round(_percentile(tpots, 50), 2),
        "p90_tpot_ms": round(_percentile(tpots, 90), 2),
        "p95_tpot_ms": round(_percentile(tpots, 95), 2),
        "p99_tpot_ms": round(_percentile(tpots, 99), 2),
    }


def _augment_from_log(metrics):
    """Extract server-side metrics from server.log (vllm/sglang both log these)."""
    if not SERVER_LOG.exists():
        return metrics
    try:
        log = SERVER_LOG.read_text(errors="ignore")[-200_000:]
    except OSError:
        return metrics
    patterns = [
        (r"Prefix cache hit rate:\s*([\d.]+)\s*%", "prefix_cache_hit_pct", float),
        (r"Mean acceptance length:\s*([\d.]+)", "spec_accept_length", float),
        (r"Avg Draft acceptance rate:\s*([\d.]+)\s*%", "spec_accept_rate_pct", float),
    ]
    for pat, key, cast in patterns:
        matches = re.findall(pat, log)
        if matches:
            metrics[key] = cast(matches[-1])
    return metrics


def _map_metrics(raw):
    return {METRIC_MAP.get(k, k): v for k, v in raw.items()}


def run_benchmark(server_url, config):
    """Run benchmark per user_config.yaml settings.
    Returns (aggregated_metrics, per_combo_results) where per_combo_results is a list of dicts.
    Returns (None, []) on total failure.
    """
    bench_cfg = config.get("benchmark", {})
    num_requests = bench_cfg.get("num_requests", 200)
    warmup_requests = bench_cfg.get("warmup_requests", 0)
    temperature = bench_cfg.get("temperature", 0.0)
    request_rate = bench_cfg.get("request_rate")
    randomize = bench_cfg.get("prompt_randomize", True)

    def ensure_list(val, default):
        if val is None:
            return [default]
        return val if isinstance(val, list) else [val]

    concurrencies = ensure_list(bench_cfg.get("concurrency"), 32)
    prompt_tokens_list = ensure_list(bench_cfg.get("prompt_tokens"), 512)
    output_tokens_list = ensure_list(bench_cfg.get("output_tokens"), 128)
    cache_max_lens = ensure_list(bench_cfg.get("prompt_cache_max_len"), 0)

    combinations = list(itertools.product(concurrencies, prompt_tokens_list, output_tokens_list, cache_max_lens))
    is_sweep = len(combinations) > 1

    all_results = []
    for i, (conc, ptok, otok, pcml) in enumerate(combinations):
        if is_sweep:
            print(f"\n--- Sweep {i+1}/{len(combinations)}: concurrency={conc}, prompt_tokens={ptok}, output_tokens={otok}, cache_len={pcml} ---")

        raw = asyncio.run(_run_benchmark_async(
            server_url, num_requests, conc, ptok, otok, temperature, request_rate, randomize, pcml,
            warmup_requests=warmup_requests,
        ))
        if raw is None:
            print(f"  Sweep point {i+1} failed", file=sys.stderr)
            continue
        mapped = _map_metrics(raw)
        mapped["requested_concurrency"] = conc
        mapped["requested_prompt_tokens"] = ptok
        mapped["requested_output_tokens"] = otok
        mapped["prompt_cache_max_len"] = pcml
        all_results.append(mapped)

    if not all_results:
        return None, []

    # Per-combo results (each is a full metrics dict, one per CSV row)
    per_combo = [_augment_from_log(dict(r)) for r in all_results]

    if not is_sweep:
        return per_combo[0], per_combo

    # Aggregated metrics for scoring: avg throughput, worst-case latency
    avg_throughput = sum(r["throughput_tok_per_sec"] for r in per_combo) / len(per_combo)
    worst_ttft_p99 = max(r.get("ttft_p99_ms", 0) for r in per_combo)
    worst_itl_p99 = max(r.get("itl_p99_ms", 0) for r in per_combo)

    aggregated = dict(per_combo[0])
    aggregated["throughput_tok_per_sec"] = round(avg_throughput, 2)
    aggregated["ttft_p99_ms"] = round(worst_ttft_p99, 2)
    aggregated["itl_p99_ms"] = round(worst_itl_p99, 2)
    aggregated["sweep_points"] = len(per_combo)
    return _augment_from_log(aggregated), per_combo


# -- CSV Metrics --------------------------------------------------------------

# Canonical column order for per-experiment metrics CSV
METRICS_CSV_COLUMNS = [
    # sweep point identity (requested values)
    "requested_concurrency", "requested_prompt_tokens", "requested_output_tokens", "prompt_cache_max_len",
    # measured values
    "concurrency", "prompt_tokens_avg", "output_tokens_avg",
    # counts
    "num_requests", "successful", "failed", "wall_time_s",
    "total_output_tokens", "total_prompt_tokens",
    # throughput
    "throughput_req_per_sec", "throughput_tok_per_sec", "throughput_total_tok_per_s",
    # latency
    "avg_latency_ms", "median_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms",
    "min_latency_ms", "max_latency_ms",
    # ttft
    "avg_ttft_ms", "ttft_p50_ms", "p90_ttft_ms", "p95_ttft_ms", "ttft_p99_ms",
    # tpot / itl
    "avg_tpot_ms", "itl_p50_ms", "p90_tpot_ms", "p95_tpot_ms", "itl_p99_ms",
    # resource
    "peak_memory_gb", "startup_sec",
    # server-side
    "prefix_cache_hit_pct", "spec_accept_length", "spec_accept_rate_pct",
]


def save_metrics_csv(experiment_num, config_hash, per_combo_results):
    """Write one CSV per experiment: experiments/{num}_{hash}.csv with one row per sweep combo."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    csv_path = EXPERIMENTS_DIR / f"{experiment_num:04d}_{config_hash}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for metrics in per_combo_results:
            writer.writerow(metrics)


# -- Scoring ------------------------------------------------------------------


def compute_score(metrics, config, per_combo=None):
    """Compute score from metrics. If scoring_combo is set and per_combo results exist,
    score from the matching combo instead of aggregated metrics."""
    opt = config["optimization"]
    scoring_metrics = metrics

    # If scoring_combo is defined, find the matching combo from per_combo results
    scoring_combo = opt.get("scoring_combo")
    if scoring_combo and per_combo:
        for combo in per_combo:
            match = all(
                combo.get(k) == v or combo.get(f"requested_{k}") == v
                for k, v in scoring_combo.items()
            )
            if match:
                scoring_metrics = combo
                break
        else:
            print(f"WARNING: scoring_combo {scoring_combo} not found in sweep results, using aggregated")

    for name, bounds in opt.get("constraints", {}).items():
        val = scoring_metrics.get(name)
        if val is None:
            continue  # skip constraints for metrics not present
        if "max" in bounds and val > bounds["max"]:
            return None
        if "min" in bounds and val < bounds["min"]:
            return None
    primary = scoring_metrics.get(opt["primary_metric"])
    if primary is None:
        print(f"WARNING: primary_metric '{opt['primary_metric']}' not found in metrics. "
              f"Available: {sorted(scoring_metrics.keys())}")
        return None
    return -primary if opt["direction"] == "minimize" else primary


# -- Notifications ------------------------------------------------------------


def notify_telegram(config, num, backend, status, score, metrics):
    try:
        tg = config.get("telegram", {})
        token, chat_id = tg.get("bot_token", ""), tg.get("chat_id", "")
        if not token or not chat_id:
            return
        score_str = f"{score:.1f}" if score is not None else "failed"
        lines = [f"Exp #{num} ({backend}): {status}, score: {score_str}"]
        if metrics:
            for k in ("throughput_tok_per_sec", "ttft_p99_ms", "itl_p99_ms", "peak_memory_gb"):
                if k in metrics:
                    lines.append(f"  {k}: {metrics[k]:.1f}")
        exps = load_experiments()
        best = max((e for e in exps if e.get("score") is not None), key=lambda e: e["score"], default=None)
        if best:
            lines.append(f"\nBest so far: #{best['num']} score={best['score']} ({best.get('backend', '')})")
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": "\n".join(lines)},
            timeout=10,
        )
    except Exception:
        pass


# -- Helpers ------------------------------------------------------------------


def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _make_entry(num, config_hash, backend, status, score, description, exp_yaml, metrics, timestamp):
    return {
        "num": num, "hash": config_hash, "status": status, "score": score,
        "description": description, "tags": exp_yaml["tags"], "params": exp_yaml["params"],
        "metrics": metrics, "backend": backend, "timestamp": timestamp,
    }


def save_snapshot(num, config_hash):
    """Save serve.sh and server.log snapshots."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    prefix = f"{num:04d}_{config_hash}"
    shutil.copy2(SERVE_SCRIPT, EXPERIMENTS_DIR / f"{prefix}.sh")
    if SERVER_LOG.exists():
        shutil.copy2(SERVER_LOG, EXPERIMENTS_DIR / f"{prefix}.log")


def _script_path(exp):
    return EXPERIMENTS_DIR / f"{exp['num']:04d}_{exp['hash']}.sh"


# -- Subcommand: run ---------------------------------------------------------


def cmd_run(args):
    config = load_config()
    requested_port = config["server"].get("port", 8000)
    health_timeout = config["server"].get("health_timeout_sec", 300)

    # Backend resolution: explicit override > experiment.yaml params.backend > serve.sh substring scan
    try:
        content = SERVE_SCRIPT.read_text().lower()
    except FileNotFoundError:
        print("ERROR: serve.sh not found")
        sys.exit(1)
    config_hash = hash_file(SERVE_SCRIPT)
    exp_yaml = load_experiment_yaml()
    backend = (args.backend
               or exp_yaml.get("params", {}).get("backend")
               or next((b for b in ("vllm", "sglang", "trtllm") if b in content), "unknown"))

    # Pre-flight: disk
    disk_ok, free_gb, disk_msg = check_disk_space(min_gb=50)
    print(f"Disk: {disk_msg}")
    if not disk_ok:
        model_name = config.get("model", {}).get("name")
        print(f"Cleaning unused HF cache (keeping {model_name})...")
        freed = cleanup_hf_cache(keep_model=model_name)
        if freed > 0:
            print(f"Freed {freed:.1f}GB")
        disk_ok, free_gb, disk_msg = check_disk_space(min_gb=50)
        if not disk_ok:
            print(f"ERROR: Still low on disk ({free_gb:.1f}GB)")
            sys.exit(1)

    # Pre-flight: GPUs
    gpus_ok, gpu_msg = check_gpu_availability()
    if not gpus_ok:
        print(f"WARNING: {gpu_msg}")
        print("Waiting up to 5 min for GPUs...")
        if not wait_for_gpus(timeout=300, interval=30):
            print("ERROR: GPUs still busy after 5 min")
            sys.exit(1)
        print("GPUs now available.")

    # Port
    port, port_moved = find_free_port(requested_port)
    if port is None:
        print(f"ERROR: no free port from {requested_port}")
        sys.exit(1)
    if port_moved:
        print(f"Port {requested_port} in use, using {port}")
    health_url = config["server"]["health_endpoint"].replace(str(requested_port), str(port))

    # Dedup
    prev = check_already_tried(config_hash)
    if prev:
        print(f"SKIP: already tried as experiment #{prev['num']} (score: {prev['score']})")
        print("status: duplicate")
        sys.exit(1)

    num = get_next_num()
    t_start = time.time()
    print(f"Experiment #{num} (hash: {config_hash}, backend: {backend}, port: {port})")

    # Start server (with one retry on transient infrastructure failures)
    def _looks_transient(log_text):
        signals = [
            "Errno 98", "Address already in use", "Could not connect", "Connection refused",
            "apt-get", "Failed to fetch", "ConnectTimeoutError", "Read timed out",
            "Mirror sync in progress", "Temporary failure in name resolution",
        ]
        return any(s in log_text for s in signals)
    proc = start_server(port)
    print(f"Waiting for server at {health_url}...")
    if not wait_for_ready(health_url, health_timeout, proc):
        log_tail = SERVER_LOG.read_text(errors="ignore")[-50_000:] if SERVER_LOG.exists() else ""
        if _looks_transient(log_tail):
            print("[retry] Transient failure detected — cleaning up and retrying once after 30s")
            stop_server(proc, port)
            wait_for_port_free(port)
            time.sleep(30)
            proc = start_server(port)
            print(f"Waiting for server at {health_url} (retry)...")
            ready = wait_for_ready(health_url, health_timeout, proc)
        else:
            ready = False
        if not ready:
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            save_snapshot(num, config_hash)
            append_experiment(_make_entry(num, config_hash, backend, "server_failed", None,
                              exp_yaml["description"] or f"server failed ({backend})", exp_yaml, {}, ts))
            print("status: server_failed")
            stop_server(proc, port)
            wait_for_port_free(port)
            sys.exit(1)

    startup_time = time.time() - t_start
    print(f"Server ready in {startup_time:.1f}s")

    # Benchmark
    print("Running benchmark...")
    try:
        metrics, per_combo = run_benchmark(f"http://localhost:{port}", config)
    except Exception as e:
        print(f"Benchmark error: {e}", file=sys.stderr)
        metrics, per_combo = None, []

    stop_server(proc, port)
    wait_for_port_free(port)

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    if metrics is None:
        save_snapshot(num, config_hash)
        append_experiment(_make_entry(num, config_hash, backend, "benchmark_failed", None,
                          exp_yaml["description"] or f"benchmark failed ({backend})", exp_yaml, {}, timestamp))
        print("status: benchmark_failed")
        sys.exit(1)

    peak_mem = get_peak_gpu_memory()
    metrics["peak_memory_gb"] = peak_mem
    metrics["startup_sec"] = round(startup_time, 1)
    # Add peak_memory and startup to each per-combo result too
    for r in per_combo:
        r["peak_memory_gb"] = peak_mem
        r["startup_sec"] = round(startup_time, 1)

    score = compute_score(metrics, config, per_combo)
    status = "ok" if score is not None else "constraint_violated"

    save_snapshot(num, config_hash)
    description = exp_yaml["description"] or f"{backend} {config_hash}"
    append_experiment(_make_entry(num, config_hash, backend, status, score, description,
                                  exp_yaml, metrics, timestamp))

    # Write per-experiment CSV with one row per sweep combo
    save_metrics_csv(num, config_hash, per_combo)

    # Print results
    print("---")
    for k in sorted(metrics):
        v = metrics[k]
        print(f"{k + ':':30s}{v:.4f}" if isinstance(v, float) else f"{k + ':':30s}{v}")
    print(f"{'score:':30s}{score:.4f}" if score is not None else f"{'score:':30s}failed")
    print(f"{'status:':30s}{status}")
    print(f"{'experiment_num:':30s}{num}")
    print(f"{'elapsed_sec:':30s}{time.time() - t_start:.1f}")
    if len(per_combo) > 1:
        print(f"{'sweep_combos:':30s}{len(per_combo)} (see metrics.csv)")

    notify_telegram(config, num, backend, status, score, metrics)


# -- Subcommand: status ------------------------------------------------------


def cmd_status(args):
    exps = load_experiments()
    if not exps:
        print("No experiments recorded yet.")
        return
    by_status = {}
    for e in exps:
        s = e.get("status", "?")
        by_status[s] = by_status.get(s, 0) + 1
    print(f"Total: {len(exps)}")
    print(f"By status: {by_status}")

    ok_exps = [e for e in exps if e.get("score") is not None]
    if ok_exps:
        best = max(ok_exps, key=lambda e: e["score"])
        print(f"Best: #{best['num']} score={best['score']:.2f} ({best.get('backend', '')})")


# -- Subcommand: best --------------------------------------------------------


def cmd_best(args):
    exps = load_experiments()
    if args.backend:
        exps = [e for e in exps if e.get("backend") == args.backend]
    ok_exps = [e for e in exps if e.get("score") is not None]
    if not ok_exps:
        print("No successful experiments.")
        return
    best = max(ok_exps, key=lambda e: e["score"])
    print(f"=== Best: #{best['num']} (score: {best['score']:.2f}, {best.get('backend', '')}) ===")
    if best.get("params"):
        print(f"Params: {json.dumps(best['params'], indent=2)}")
    print("\nMetrics:")
    for k, v in sorted(best.get("metrics", {}).items()):
        print(f"  {k}: {v}")
    path = _script_path(best)
    if path.exists():
        print(f"\nserve.sh:\n{'-' * 40}")
        print(path.read_text())


# -- Subcommand: history -----------------------------------------------------


def cmd_history(args):
    exps = load_experiments()
    if args.backend:
        exps = [e for e in exps if e.get("backend") == args.backend]
    if args.status:
        exps = [e for e in exps if e.get("status") == args.status]
    exps = list(reversed(exps))
    if args.limit:
        exps = exps[:args.limit]
    if not exps:
        print("No experiments match.")
        return
    print(f"{'#':>4}  {'backend':<8}  {'score':>10}  {'status':<20}  description")
    for e in exps:
        s = e.get("score")
        score_str = f"{s:.2f}" if s is not None else "failed"
        desc = e.get("description", "")[:50]
        print(f"{e['num']:>4}  {e.get('backend', '?'):<8}  {score_str:>10}  {e.get('status', '?'):<20}  {desc}")


# -- Subcommand: show --------------------------------------------------------


def cmd_show(args):
    for e in load_experiments():
        if e["num"] == args.num:
            print(json.dumps(e, indent=2, default=str))
            path = _script_path(e)
            if path.exists():
                print(f"\nserve.sh:\n{path.read_text()}")
            return
    print(f"#{args.num} not found.")


# -- Subcommand: diff --------------------------------------------------------


def cmd_diff(args):
    exp_map = {e["num"]: e for e in load_experiments()}
    if args.num1 not in exp_map or args.num2 not in exp_map:
        print("Experiment not found.")
        return
    p1, p2 = _script_path(exp_map[args.num1]), _script_path(exp_map[args.num2])
    result = subprocess.run(["diff", "-u", str(p1), str(p2)], capture_output=True, text=True)
    print(result.stdout or "No differences.")


def cmd_compare(args):
    """Side-by-side param + metric delta between two experiments."""
    exp_map = {e["num"]: e for e in load_experiments()}
    a, b = exp_map.get(args.num1), exp_map.get(args.num2)
    if a is None or b is None:
        print("Experiment not found.")
        return
    pa, pb = a.get("params") or {}, b.get("params") or {}
    print(f"=== params (#{a['num']} → #{b['num']}) ===")
    changed = [k for k in sorted(set(pa) | set(pb)) if pa.get(k) != pb.get(k)]
    if not changed:
        print("  (no param differences)")
    for k in changed:
        print(f"  {k}: {pa.get(k)!r} → {pb.get(k)!r}")
    ma, mb = a.get("metrics") or {}, b.get("metrics") or {}
    print(f"=== metric delta ===")
    for k in sorted(set(ma) | set(mb)):
        va, vb = ma.get(k), mb.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            print(f"  {k:30s} {va:>12.2f} → {vb:>12.2f}  ({delta:+.2f})")
    print(f"=== score: {a.get('score')} → {b.get('score')} ===")


# -- Subcommand: gaps --------------------------------------------------------


def cmd_gaps(args):
    config = load_config()
    search_space = config.get("search_space", {})
    if not search_space:
        print("No search_space defined in user_config.yaml.")
        return

    exps = load_experiments()
    tried = {}
    for exp in exps:
        for k, v in exp.get("params", {}).items():
            tried.setdefault(k, set()).add(str(v))

    print(f"Experiments with params: {sum(1 for e in exps if e.get('params'))}/{len(exps)}")
    for param, values in sorted(search_space.items()):
        all_values = set(str(v) for v in values)
        tried_values = tried.get(param, set())
        untried = sorted(all_values - tried_values)
        if untried:
            print(f"  {param}: tried {sorted(tried_values) if tried_values else '[]'}, untried: {untried}")
        else:
            tried_display = sorted(tried_values) if tried_values else "[]"
            print(f"  {param}: fully explored {tried_display}")


# -- Subcommand: check-gpus --------------------------------------------------


def cmd_check_gpus(args):
    ok, msg = check_gpu_availability()
    print(msg)
    rows = _query_gpus("name", "memory.used", "memory.total", "utilization.gpu")
    if rows:
        print("\nGPU details:")
        for row in rows:
            print(f"  GPU {row[0]}: {row[1]}, {row[2]}/{row[3]} MiB, {row[4]}% util")


# -- Subcommand: kill-server --------------------------------------------------


def cmd_kill_server(args):
    config = load_config()
    port = config["server"].get("port", 8000)
    try:
        out = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not out:
            print(f"No process on port {port}.")
            return
        pids = out.split("\n")
        for pid in pids:
            pid = pid.strip()
            if pid:
                os.kill(int(pid), signal.SIGTERM)
                print(f"Killed PID {pid} on port {port}")
    except Exception as e:
        print(f"Error: {e}")


# -- Remote Commands ----------------------------------------------------------


def _get_remote_config(config):
    remote = config.get("remote", {})
    host = remote.get("ssh_host")
    path = remote.get("project_path", "~/autoresearch-inference-optimization")
    if not host:
        print("Error: remote.ssh_host not configured")
        sys.exit(1)
    return host, path


def _ssh_run(host, cmd, capture=False, timeout=None):
    full_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", host, REMOTE_PATH_PREFIX + cmd]
    print(f"[remote] ssh {host} {cmd[:100]}...")
    if capture:
        return subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    return subprocess.run(full_cmd, timeout=timeout)


def _remote_sync(config):
    host, remote_path = _get_remote_config(config)
    excludes = [
        ".git", "__pycache__", "*.pyc", ".venv", "worktrees/",
        "node_modules/", ".env", "*.egg-info", "dev/",
        "server.log", "run.log", "progress.png",
    ]
    exclude_args = []
    for e in excludes:
        exclude_args += ["--exclude", e]
    _ssh_run(host, f"mkdir -p {remote_path}")
    cmd = ["rsync", "-avz", "--delete", *exclude_args, str(BASE_DIR) + "/", f"{host}:{remote_path}/"]
    print(f"[remote] Syncing to {host}:{remote_path}/")
    subprocess.run(cmd, check=True)
    print("[remote] Sync complete.")
    # Pre-fetch model weights so the first run doesn't blow past health_timeout downloading 30+ GB.
    model = (config.get("model") or {}).get("name")
    if model:
        slug = model.replace("/", "--")
        check = _ssh_run(host, f"test -d ~/.cache/huggingface/hub/models--{slug}/snapshots && echo HAVE", capture=True, timeout=30)
        if "HAVE" not in (check.stdout or ""):
            print(f"[remote] Pre-fetching {model} weights (HF_TOKEN expected in remote env)...")
            _ssh_run(host, f"python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"{model}\")' || echo '[remote] pre-fetch failed (continuing)'")


def _remote_fetch_experiments(config):
    host, remote_path = _get_remote_config(config)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    # Fetch experiment snapshots
    subprocess.run(["rsync", "-avz", f"{host}:{remote_path}/experiments/", str(EXPERIMENTS_DIR) + "/"])
    # Fetch experiments.jsonl
    subprocess.run(["rsync", "-avz", f"{host}:{remote_path}/experiments.jsonl", str(EXPERIMENTS_JSONL)])
    print("[remote] Experiments fetched.")


def cmd_remote(args):
    config = load_config()
    host, remote_path = _get_remote_config(config)

    if args.remote_cmd == "sync":
        _remote_sync(config)

    elif args.remote_cmd == "health":
        print(f"\n=== GPU Status ({host}) ===")
        _ssh_run(host, "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader")
        print(f"\n=== Disk Space ===")
        _ssh_run(host, f"df -h {remote_path} 2>/dev/null || df -h ~")
        print(f"\n=== GPU Processes ===")
        _ssh_run(host, "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || echo 'None'")

    elif args.remote_cmd == "run":
        _remote_sync(config)
        backend_flag = f" --backend {args.backend}" if args.backend else ""
        # Detached launch: nohup'd remote run survives ssh disconnects; we poll for completion.
        # PID + exit-code + log all live on the remote so a flaky network never loses results.
        log_path, pid_path, ec_path = f"{remote_path}/run.log", f"{remote_path}/run.pid", f"{remote_path}/run.exitcode"
        launch = (
            f"cd {remote_path} && rm -f {pid_path} {ec_path} && "
            f"nohup bash -c '(uv run engine.py run{backend_flag} 2>&1; echo $? > {ec_path}) > {log_path} 2>&1' "
            f"</dev/null >/dev/null 2>&1 & echo $!"
        )
        print(f"[remote] Launching detached experiment on {host}...")
        result = _ssh_run(host, launch, capture=True, timeout=30)
        pid = (result.stdout or "").strip().split("\n")[-1]
        print(f"[remote] PID={pid}, polling for completion (Ctrl-C is safe — server keeps running)")
        deadline = time.time() + (args.timeout or 7200)
        last_size = 0
        while time.time() < deadline:
            time.sleep(20)
            check = _ssh_run(
                host,
                f"if [ -f {ec_path} ]; then echo DONE; cat {ec_path}; else echo RUNNING; wc -c < {log_path} 2>/dev/null || echo 0; fi",
                capture=True, timeout=30,
            )
            out = (check.stdout or "").strip().split("\n")
            if out and out[0] == "DONE":
                exit_code = int(out[1]) if len(out) > 1 and out[1].strip().isdigit() else 1
                print(f"[remote] Experiment finished, exit code {exit_code}")
                break
            elif out and out[0] == "RUNNING" and len(out) > 1:
                size = int(out[1].strip()) if out[1].strip().isdigit() else 0
                if size > last_size:
                    print(f"[remote] running... log size {size} bytes")
                    last_size = size
        else:
            print(f"[remote] Timeout after {args.timeout}s. Server may still be running on remote.")
            exit_code = 124
        # Fetch the log + experiments
        subprocess.run(["rsync", "-az", f"{host}:{log_path}", str(BASE_DIR / 'run.log')], check=False)
        _remote_fetch_experiments(config)
        return exit_code

    elif args.remote_cmd == "fetch":
        _remote_fetch_experiments(config)

    elif args.remote_cmd == "shell":
        cmd = f"cd {remote_path} && {args.command}"
        _ssh_run(host, cmd)


# -- CLI ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Inference optimization experiment engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run
    r = sub.add_parser("run", help="Run one experiment (serve.sh + experiment.yaml)")
    r.add_argument("--backend", help="Override backend detection")

    # status
    sub.add_parser("status", help="Summary stats")

    # best
    b = sub.add_parser("best", help="Best config with metrics")
    b.add_argument("--backend", help="Filter by backend")

    # history
    h = sub.add_parser("history", help="Table of all experiments")
    h.add_argument("--backend", help="Filter by backend")
    h.add_argument("--status", help="Filter by status")
    h.add_argument("--limit", type=int, help="Limit rows")

    # show
    s = sub.add_parser("show", help="Full details of one experiment")
    s.add_argument("num", type=int, help="Experiment number")

    # diff
    d = sub.add_parser("diff", help="Diff serve.sh between two experiments")
    d.add_argument("num1", type=int)
    d.add_argument("num2", type=int)

    # compare
    c = sub.add_parser("compare", help="Param + metric delta between two experiments")
    c.add_argument("num1", type=int)
    c.add_argument("num2", type=int)

    # gaps
    sub.add_parser("gaps", help="Untried param values from search_space")

    # check-gpus
    sub.add_parser("check-gpus", help="GPU availability")

    # kill-server
    sub.add_parser("kill-server", help="Kill processes on configured port")

    # remote
    rp = sub.add_parser("remote", help="Remote machine commands")
    rsub = rp.add_subparsers(dest="remote_cmd", required=True)
    rsub.add_parser("sync", help="Sync project to remote")
    rsub.add_parser("health", help="Check remote GPU health")
    rr = rsub.add_parser("run", help="Run experiment on remote")
    rr.add_argument("--backend", help="Override backend")
    rr.add_argument("--timeout", type=int, default=900, help="Timeout in seconds")
    rsub.add_parser("fetch", help="Fetch experiments from remote")
    rs = rsub.add_parser("shell", help="Run command on remote")
    rs.add_argument("command", help="Command to run")

    args = parser.parse_args()
    {
        "run": cmd_run,
        "status": cmd_status,
        "best": cmd_best,
        "history": cmd_history,
        "show": cmd_show,
        "diff": cmd_diff,
        "compare": cmd_compare,
        "gaps": cmd_gaps,
        "check-gpus": cmd_check_gpus,
        "kill-server": cmd_kill_server,
        "remote": cmd_remote,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
