"""
Remote machine management — sync, execute, and monitor experiments on remote GPU servers.

Usage:
    python remote.py sync                   # rsync repo to remote machine
    python remote.py health                 # check remote GPU health
    python remote.py run                    # run prepare.py on remote machine
    python remote.py run-benchmark          # run benchmark.py on remote (for testing)
    python remote.py logs                   # tail server.log from remote
    python remote.py fetch-experiments      # pull experiments/ back to local
    python remote.py shell "command"        # run arbitrary command on remote
    python remote.py setup                  # install dependencies on remote
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "user_config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_remote_config():
    config = load_config()
    remote = config.get("remote", {})
    host = remote.get("ssh_host")
    path = remote.get("project_path", "~/autoresearch-inference-optimization")
    if not host:
        print("Error: remote.ssh_host not configured in user_config.yaml")
        sys.exit(1)
    return host, path


# PATH prefix for remote commands — ensures uv, pip, python, cargo, etc. are found
# even when SSH doesn't source .bashrc/.profile (non-login, non-interactive shell).
REMOTE_PATH_PREFIX = (
    'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.uv/bin:'
    '/usr/local/bin:/usr/local/sbin:$PATH" && '
)


def ssh_run(host, cmd, capture=False, timeout=None):
    """Run a command on the remote machine via SSH."""
    full_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", host, REMOTE_PATH_PREFIX + cmd]
    print(f"[remote] ssh {host} {cmd[:100]}...")
    if capture:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return result
    else:
        return subprocess.run(full_cmd, timeout=timeout)


def cmd_sync(args):
    """Rsync the project to the remote machine."""
    host, remote_path = get_remote_config()

    # Files/dirs to exclude from sync
    excludes = [
        ".git", "__pycache__", "*.pyc", ".venv", "worktrees/",
        "node_modules/", ".env", "*.egg-info", "dev/",
        "server.log", "run.log", "progress.png", "experiments/",
    ]
    exclude_args = []
    for e in excludes:
        exclude_args += ["--exclude", e]

    # Create remote directory
    ssh_run(host, f"mkdir -p {remote_path}")

    cmd = [
        "rsync", "-avz", "--delete",
        *exclude_args,
        str(BASE_DIR) + "/",
        f"{host}:{remote_path}/",
    ]
    print(f"[remote] Syncing to {host}:{remote_path}/")
    subprocess.run(cmd, check=True)
    print("[remote] Sync complete.")


def cmd_health(args):
    """Check remote machine health: GPU status, disk, memory."""
    host, remote_path = get_remote_config()

    print(f"\n=== GPU Status ({host}) ===")
    ssh_run(host, "nvidia-smi")

    print(f"\n=== GPU Memory ===")
    ssh_run(host, "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader")

    print(f"\n=== Disk Space ===")
    ssh_run(host, f"df -h {remote_path} 2>/dev/null || df -h ~")

    print(f"\n=== System Memory ===")
    ssh_run(host, "free -h")

    print(f"\n=== Running GPU Processes ===")
    ssh_run(host, "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || echo 'No GPU processes'")

    print(f"\n=== Docker Containers ===")
    ssh_run(host, "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || echo 'Docker not available'")

    print(f"\n=== Tmux Sessions ===")
    ssh_run(host, "tmux ls 2>/dev/null || echo 'No tmux sessions'")


def cmd_run(args):
    """Run prepare.py on the remote machine."""
    host, remote_path = get_remote_config()

    # First sync
    print("[remote] Syncing before run...")
    cmd_sync(argparse.Namespace())

    backend_flag = f" --backend {args.backend}" if args.backend else ""
    cmd = f"cd {remote_path} && uv run prepare.py{backend_flag} 2>&1"

    print(f"[remote] Running experiment on {host}...")
    result = ssh_run(host, cmd, capture=True, timeout=args.timeout)

    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Save output locally as run.log
    run_log = BASE_DIR / "run.log"
    run_log.write_text(result.stdout + (result.stderr or ""))
    print(f"[remote] Output saved to {run_log}")

    # Fetch experiments back
    print("[remote] Fetching experiment results...")
    cmd_fetch_experiments(argparse.Namespace())

    return result.returncode



def cmd_logs(args):
    """Tail server.log from remote machine."""
    host, remote_path = get_remote_config()
    n = args.lines
    ssh_run(host, f"tail -{n} {remote_path}/server.log 2>/dev/null || echo 'No server.log found'")


def cmd_fetch_experiments(args):
    """Pull experiments/ directory from remote to local."""
    host, remote_path = get_remote_config()
    local_exp = BASE_DIR / "experiments"
    local_exp.mkdir(exist_ok=True)

    cmd = [
        "rsync", "-avz",
        f"{host}:{remote_path}/experiments/",
        str(local_exp) + "/",
    ]
    subprocess.run(cmd)
    print(f"[remote] Experiments fetched to {local_exp}")


def cmd_shell(args):
    """Run an arbitrary command on the remote machine."""
    host, remote_path = get_remote_config()
    cmd = f"cd {remote_path} && {args.command}"
    ssh_run(host, cmd)


def cmd_setup(args):
    """Install dependencies on the remote machine."""
    host, remote_path = get_remote_config()

    print(f"[remote] Setting up {host}...")

    # Check if uv is installed
    result = ssh_run(host, "which uv", capture=True)
    if result.returncode != 0:
        print("[remote] Installing uv...")
        ssh_run(host, "curl -LsSf https://astral.sh/uv/install.sh | sh")

    # Sync first
    cmd_sync(argparse.Namespace())

    # Install project dependencies
    print("[remote] Installing project dependencies...")
    ssh_run(host, f"cd {remote_path} && uv sync")

    print("[remote] Setup complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Remote machine management")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sync", help="Rsync project to remote")
    sub.add_parser("health", help="Check remote GPU health")

    r = sub.add_parser("run", help="Run prepare.py on remote")
    r.add_argument("--backend", help="Override backend detection")
    r.add_argument("--timeout", type=int, default=900, help="Timeout in seconds")

    l = sub.add_parser("logs", help="Tail server.log from remote")
    l.add_argument("--lines", "-n", type=int, default=50, help="Number of lines")

    sub.add_parser("fetch-experiments", help="Pull experiments/ from remote")

    s = sub.add_parser("shell", help="Run command on remote")
    s.add_argument("command", help="Command to run")

    sub.add_parser("setup", help="Install dependencies on remote")

    args = parser.parse_args()
    {
        "sync": cmd_sync,
        "health": cmd_health,
        "run": cmd_run,
        "logs": cmd_logs,
        "fetch-experiments": cmd_fetch_experiments,
        "shell": cmd_shell,
        "setup": cmd_setup,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
