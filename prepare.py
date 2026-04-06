"""
Read-only experiment harness. Do not modify — the agent edits serve.sh only.
Usage: uv run prepare.py
"""
import os, sys, time, signal, hashlib, json, shutil, socket, subprocess
import yaml, requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "user_config.yaml")
SERVE_SCRIPT = os.path.join(BASE_DIR, "serve.sh")
BENCHMARK_SCRIPT = os.path.join(BASE_DIR, "benchmark.py")
SERVER_LOG = os.path.join(BASE_DIR, "server.log")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")


def load_config():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    for key in ("model", "hardware", "server", "optimization", "experiment"):
        assert key in config, f"missing '{key}' in user_config.yaml"
    opt = config["optimization"]
    assert "primary_metric" in opt, "optimization.primary_metric is required"
    assert opt.get("direction") in ("maximize", "minimize"), "direction must be maximize/minimize"
    return config


def start_server(port):
    log_file = open(SERVER_LOG, "w")
    return subprocess.Popen(
        ["bash", SERVE_SCRIPT],
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


def stop_server(proc):
    """Kill only the server process group that WE started. Never kills other processes."""
    if proc.poll() is not None:
        return
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


def wait_for_port_free(port, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if s.connect_ex(("localhost", port)) != 0:
                return True
        finally:
            s.close()
        time.sleep(1)
    return False


def is_port_in_use(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        return s.connect_ex(("localhost", port)) == 0
    finally:
        s.close()


def find_free_port(start_port, max_attempts=20):
    """Find a free port starting from start_port. Returns (port, was_moved)."""
    for offset in range(max_attempts):
        port = start_port + offset
        if not is_port_in_use(port):
            return port, offset > 0
    return None, False


def get_required_gpus():
    """Parse CUDA_VISIBLE_DEVICES from serve.sh to know which GPUs we actually need."""
    try:
        content = open(SERVE_SCRIPT).read()
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("export CUDA_VISIBLE_DEVICES=") or line.startswith("CUDA_VISIBLE_DEVICES="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                return set(val.split(","))
    except Exception:
        pass
    return None  # check all GPUs if not specified


def check_gpu_availability():
    """Check if GPUs are available. Only checks GPUs required by serve.sh."""
    try:
        required = get_required_gpus()
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if not out:
            return True, "nvidia-smi returned no data"

        busy_gpus = []
        for line in out.split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                idx, mem_used, mem_total, util = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                if required and idx not in required:
                    continue  # skip GPUs not needed by serve.sh
                if mem_used / mem_total > 0.20 or util > 10:
                    busy_gpus.append(f"GPU {idx}: {mem_used:.0f}/{mem_total:.0f} MiB, {util:.0f}% util")

        if busy_gpus:
            return False, f"{len(busy_gpus)} GPU(s) in use:\n" + "\n".join(busy_gpus)
        checked = f" (checking GPUs {','.join(sorted(required))})" if required else ""
        return True, f"All required GPUs available{checked}"
    except Exception as e:
        return True, f"Could not check GPUs: {e}"


def check_disk_space(min_gb=50):
    """Check if there's enough disk space. Returns (ok, free_gb, message)."""
    try:
        st = os.statvfs(os.path.expanduser("~"))
        free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
        if free_gb < min_gb:
            return False, free_gb, f"Low disk: {free_gb:.1f}GB free (need {min_gb}GB)"
        return True, free_gb, f"Disk OK: {free_gb:.1f}GB free"
    except Exception as e:
        return True, 0, f"Could not check disk: {e}"


def cleanup_hf_cache(keep_model=None):
    """Remove unused models from HuggingFace cache to free disk space."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.isdir(cache_dir):
        return 0
    freed = 0
    keep_model_slug = keep_model.replace("/", "--") if keep_model else None
    for entry in os.listdir(cache_dir):
        path = os.path.join(cache_dir, entry)
        if not os.path.isdir(path) or not entry.startswith("models--"):
            continue
        # Keep the current model
        if keep_model_slug and keep_model_slug in entry:
            continue
        # Calculate size
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(path) for f in fns
        )
        size_gb = size / (1024 ** 3)
        print(f"Removing cached model: {entry} ({size_gb:.1f}GB)")
        shutil.rmtree(path)
        freed += size_gb
    return freed


def wait_for_gpus(timeout=300, interval=30):
    """Wait for GPUs to become free. Returns True if free, False if timed out."""
    start = time.time()
    while time.time() - start < timeout:
        ok, msg = check_gpu_availability()
        if ok:
            return True
        print(f"GPUs busy, waiting {interval}s... ({msg})")
        time.sleep(interval)
    return False


def run_benchmark(server_url):
    result = subprocess.run(
        [sys.executable, BENCHMARK_SCRIPT],
        env={**os.environ, "SERVER_URL": server_url},
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        return None
    metrics = {}
    for line in result.stdout.strip().split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            try:
                metrics[key.strip()] = float(val.strip())
            except ValueError:
                pass
    return metrics


def get_peak_gpu_memory():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        vals = [float(x) for x in out.split("\n") if x.strip()]
        return sum(vals) / 1024.0 if vals else 0.0
    except Exception:
        return 0.0


def compute_score(metrics, config):
    opt = config["optimization"]
    for name, bounds in opt.get("constraints", {}).items():
        val = metrics.get(name)
        if val is None:
            return float("-inf")
        if "max" in bounds and val > bounds["max"]:
            return float("-inf")
        if "min" in bounds and val < bounds["min"]:
            return float("-inf")
    primary = metrics.get(opt["primary_metric"])
    if primary is None:
        return float("-inf")
    return -primary if opt["direction"] == "minimize" else primary


def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def check_already_tried(config_hash):
    if not os.path.isdir(EXPERIMENTS_DIR):
        return None
    for fname in os.listdir(EXPERIMENTS_DIR):
        if fname.endswith(".json") and config_hash in fname:
            with open(os.path.join(EXPERIMENTS_DIR, fname)) as f:
                return json.load(f)
    return None


def get_next_experiment_num():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    nums = []
    for fname in os.listdir(EXPERIMENTS_DIR):
        if fname.endswith(".json"):
            try:
                nums.append(int(fname.split("_")[0]))
            except (ValueError, IndexError):
                pass
    return max(nums, default=0) + 1


def save_experiment(num, config_hash, backend, score, metrics, status, description):
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    prefix = f"{num:04d}_{config_hash}"
    shutil.copy2(SERVE_SCRIPT, os.path.join(EXPERIMENTS_DIR, f"{prefix}.sh"))
    if os.path.exists(SERVER_LOG):
        shutil.copy2(SERVER_LOG, os.path.join(EXPERIMENTS_DIR, f"{prefix}.log"))
    data = dict(experiment_num=num, config_hash=config_hash, backend=backend,
                score=score if score != float("-inf") else "-inf",
                metrics=metrics or {}, status=status, description=description,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(EXPERIMENTS_DIR, f"{prefix}.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)


def detect_backend():
    try:
        content = open(SERVE_SCRIPT).read().lower()
    except FileNotFoundError:
        return "unknown"
    for name in ("vllm", "sglang"):
        if name in content:
            return name
    if any(k in content for k in ("trtllm", "tensorrt", "tritonserver")):
        return "trtllm"
    return "unknown"


def print_result(status, backend, config_hash, experiment_num, metrics=None, score=None, error=None, elapsed=0):
    print("---")
    if metrics:
        for k in sorted(metrics):
            v = metrics[k]
            print(f"{k}:".ljust(25) + (f"{v:.4f}" if isinstance(v, float) else str(v)))
    if score is not None:
        print(f"{'score:'.ljust(25)}{score:.4f}" if score != float("-inf") else f"{'score:'.ljust(25)}-inf")
    if error:
        print(f"{'error:'.ljust(25)}{error}")
    print(f"{'backend:'.ljust(25)}{backend}")
    print(f"{'elapsed_sec:'.ljust(25)}{elapsed:.1f}")
    print(f"{'experiment_num:'.ljust(25)}{experiment_num}")
    print(f"{'config_hash:'.ljust(25)}{config_hash}")
    print(f"{'status:'.ljust(25)}{status}")


def get_best_experiment():
    """Find the best experiment so far."""
    if not os.path.isdir(EXPERIMENTS_DIR):
        return None
    best, best_exp = float("-inf"), None
    for fname in os.listdir(EXPERIMENTS_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(EXPERIMENTS_DIR, fname)) as f:
            exp = json.load(f)
        s = exp.get("score")
        if s is not None and str(s) != "-inf" and float(s) > best:
            best, best_exp = float(s), exp
    return best_exp


def notify_telegram(num, backend, status, score, metrics):
    """Send one-way experiment summary to Telegram. Silently no-ops if not configured."""
    try:
        config = load_config()
        tg = config.get("telegram", {})
        token, chat_id = tg.get("bot_token", ""), tg.get("chat_id", "")
        if not token or not chat_id:
            return

        score_str = f"{score:.1f}" if score != float("-inf") else "-inf"
        lines = [f"Exp #{num} ({backend}): {status}, score: {score_str}"]

        if metrics:
            for k in ("throughput_tok_per_sec", "ttft_p99_ms", "itl_p99_ms", "peak_memory_gb"):
                if k in metrics:
                    lines.append(f"  {k}: {metrics[k]:.1f}")

        best = get_best_experiment()
        if best:
            bs = best.get("score", "?")
            lines.append(f"\nBest so far: #{best['experiment_num']} score={bs} ({best.get('backend','')})")

        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": "\n".join(lines)},
            timeout=10,
        )
    except Exception:
        pass  # never crash the harness for a notification failure


def fail(status, error, num, config_hash, backend, proc, port):
    save_experiment(num, config_hash, backend, float("-inf"), None, status, error)
    print_result(status, backend, config_hash, num, error=error, elapsed=time.time() - _start)
    stop_server(proc)
    wait_for_port_free(port)
    sys.exit(1)


_start = 0

if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Experiment harness")
    _parser.add_argument("--backend", help="Override backend detection from serve.sh")
    _cli_args, _ = _parser.parse_known_args()

    config = load_config()
    requested_port = config["server"].get("port", 8000)
    health_timeout = config["server"].get("health_timeout_sec", 300)
    backend = _cli_args.backend or detect_backend()
    config_hash = hash_file(SERVE_SCRIPT)

    # Pre-flight: disk space check
    disk_ok, free_gb, disk_msg = check_disk_space(min_gb=50)
    print(f"Disk: {disk_msg}")
    if not disk_ok:
        model_name = config.get("model", {}).get("name")
        print(f"Attempting to free space by cleaning unused HF cache (keeping {model_name})...")
        freed = cleanup_hf_cache(keep_model=model_name)
        if freed > 0:
            print(f"Freed {freed:.1f}GB from HF cache")
        disk_ok, free_gb, disk_msg = check_disk_space(min_gb=50)
        if not disk_ok:
            print(f"ERROR: Still low on disk ({free_gb:.1f}GB). Cannot proceed safely.")
            print("ACTION NEEDED: manually free disk space on this machine.")
            sys.exit(1)

    # Pre-flight: GPU availability check
    gpus_ok, gpu_msg = check_gpu_availability()
    if not gpus_ok:
        print(f"WARNING: {gpu_msg}")
        print("Waiting up to 5 min for GPUs to free up...")
        if not wait_for_gpus(timeout=300, interval=30):
            print("ERROR: GPUs still busy after 5 min. Cannot proceed.")
            print("ACTION NEEDED: check what's running on the GPUs.")
            sys.exit(1)
        print("GPUs now available.")

    # Port conflict handling — find a free port
    port, port_moved = find_free_port(requested_port)
    if port is None:
        print(f"ERROR: no free port found starting from {requested_port}")
        sys.exit(1)
    if port_moved:
        print(f"Port {requested_port} in use, using port {port} instead")

    health_url = config["server"]["health_endpoint"].replace(str(requested_port), str(port))

    # Dedup check
    prev = check_already_tried(config_hash)
    if prev:
        print(f"SKIP: already tried as experiment #{prev['experiment_num']} (score: {prev['score']})")
        print("---")
        print(f"status:              duplicate")
        print(f"config_hash:         {config_hash}")
        sys.exit(1)

    num = get_next_experiment_num()
    _start = time.time()

    print(f"Experiment #{num} (hash: {config_hash}, backend: {backend}, port: {port})")
    proc = start_server(port)

    print(f"Waiting for server at {health_url}...")
    if not wait_for_ready(health_url, health_timeout, proc):
        fail("server_failed", "server did not become ready", num, config_hash, backend, proc, port)

    startup_time = time.time() - _start
    print(f"Server ready in {startup_time:.1f}s")

    print(f"Running benchmark...")
    try:
        metrics = run_benchmark(f"http://localhost:{port}")
    except subprocess.TimeoutExpired:
        fail("benchmark_timeout", "benchmark timed out", num, config_hash, backend, proc, port)

    stop_server(proc)
    wait_for_port_free(port)

    if metrics is None:
        save_experiment(num, config_hash, backend, float("-inf"), None, "benchmark_failed", "no metrics")
        print_result("benchmark_failed", backend, config_hash, num, error="benchmark failed", elapsed=time.time() - _start)
        sys.exit(1)

    metrics["peak_memory_gb"] = get_peak_gpu_memory()
    metrics["startup_sec"] = round(startup_time, 1)
    score = compute_score(metrics, config)
    status = "ok" if score != float("-inf") else "constraint_violated"

    save_experiment(num, config_hash, backend, score, metrics, status, f"{backend} {config_hash}")
    print_result(status, backend, config_hash, num, metrics=metrics, score=score, elapsed=time.time() - _start)
    notify_telegram(num, backend, status, score, metrics)
