"""
Experiment tracker CLI. Do not modify.
Usage: python tracker.py {status,best,history,tried,diff,show,gaps} [options]
"""
import argparse, json, os, re, subprocess

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")


def _load_experiments():
    if not os.path.isdir(EXPERIMENTS_DIR):
        return []
    exps = []
    for f in os.listdir(EXPERIMENTS_DIR):
        if f.endswith(".json"):
            with open(os.path.join(EXPERIMENTS_DIR, f)) as fh:
                exp = json.load(fh)
            exp["_file"] = f
            exps.append(exp)
    return sorted(exps, key=lambda e: e.get("experiment_num", 0))


def _script_path(exp):
    return os.path.join(EXPERIMENTS_DIR, exp["_file"].replace(".json", ".sh"))


def _parse_score(score):
    if score is None or str(score) == "-inf":
        return None
    return float(score)


def _best_per_backend(experiments, backend_filter=None):
    best = {}
    for exp in experiments:
        b = exp.get("backend", "unknown")
        if backend_filter and b != backend_filter:
            continue
        s = _parse_score(exp.get("score"))
        if s is not None and (b not in best or s > best[b]["score"]):
            best[b] = {"score": s, "exp": exp}
    return best


def cmd_status(args):
    exps = _load_experiments()
    if not exps:
        print("No experiments recorded yet.")
        return
    by_status, by_backend = {}, {}
    for e in exps:
        s, b = e.get("status", "?"), e.get("backend", "?")
        by_status[s] = by_status.get(s, 0) + 1
        by_backend[b] = by_backend.get(b, 0) + 1
    print(f"Total: {len(exps)}")
    print(f"By status: {by_status}")
    print(f"By backend: {by_backend}")
    best = _best_per_backend(exps)
    if best:
        print("Best per backend:")
        for b, info in sorted(best.items()):
            print(f"  {b}: {info['score']:.4f} (#{info['exp']['experiment_num']})")


def cmd_best(args):
    best = _best_per_backend(_load_experiments(), args.backend)
    if not best:
        print("No successful experiments.")
        return
    for b, info in sorted(best.items()):
        exp = info["exp"]
        print(f"=== {b} (score: {info['score']:.4f}, #{exp['experiment_num']}) ===")
        for k, v in sorted(exp.get("metrics", {}).items()):
            print(f"  {k}: {v}")
        path = _script_path(exp)
        if os.path.exists(path):
            print(f"\nserve.sh:\n{'-'*40}")
            print(open(path).read())


def cmd_history(args):
    exps = _load_experiments()
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
    print(f"{'#':>4}  {'backend':<8}  {'score':>10}  {'status':<12}  {'hash':<10}  description")
    for e in exps:
        s = _parse_score(e.get("score"))
        print(f"{e.get('experiment_num','?'):>4}  {e.get('backend','?'):<8}  "
              f"{f'{s:.2f}' if s else '-inf':>10}  {e.get('status','?'):<12}  "
              f"{e.get('config_hash','?')[:8]:<10}  {e.get('description','')[:50]}")


def cmd_tried(args):
    for e in _load_experiments():
        if e.get("config_hash", "").startswith(args.hash):
            print(f"YES — #{e['experiment_num']} (score: {e.get('score')}, status: {e.get('status')})")
            return
    print(f"NO — {args.hash} not tried.")


def cmd_diff(args):
    exp_map = {e["experiment_num"]: e for e in _load_experiments()}
    if args.num1 not in exp_map or args.num2 not in exp_map:
        print("Experiment not found.")
        return
    result = subprocess.run(
        ["diff", "-u", _script_path(exp_map[args.num1]), _script_path(exp_map[args.num2])],
        capture_output=True, text=True,
    )
    print(result.stdout or "No differences.")


def cmd_show(args):
    for e in _load_experiments():
        if e.get("experiment_num") == args.num:
            print(json.dumps(e, indent=2, default=str))
            path = _script_path(e)
            if os.path.exists(path):
                print(f"\nserve.sh:\n{open(path).read()}")
            return
    print(f"#{args.num} not found.")


def cmd_gaps(args):
    exps = _load_experiments()
    if not exps:
        print("No experiments yet.")
        return
    params = {
        "tensor-parallel-size": ["1", "2", "4", "8"],
        "tp": ["1", "2", "4", "8"],
        "pipeline-parallel-size": ["1", "2", "4"],
        "pp": ["1", "2", "4"],
        "gpu-memory-utilization": ["0.80", "0.85", "0.90", "0.95"],
        "mem-fraction-static": ["0.70", "0.80", "0.85", "0.90"],
        "quantization": ["awq", "gptq", "fp8"],
        "max-num-seqs": ["64", "128", "256", "512"],
        "enable-chunked-prefill": ["present"],
        "kv-cache-dtype": ["auto", "fp8"],
    }
    tried = {}
    for e in exps:
        b = e.get("backend", "?")
        if args.backend and b != args.backend:
            continue
        tried.setdefault(b, {})
        path = _script_path(e)
        if not os.path.exists(path):
            continue
        content = open(path).read()
        for p in params:
            m = re.search(rf'--{re.escape(p)}[\s=]+(\S+)', content)
            if m:
                tried[b].setdefault(p, set()).add(m.group(1))
            elif f"--{p}" in content:
                tried[b].setdefault(p, set()).add("present")

    for b, bp in sorted(tried.items()):
        n = sum(1 for e in exps if e.get("backend") == b)
        print(f"\n=== {b} ({n} experiments) ===")
        for p, vals in sorted(params.items()):
            t = bp.get(p, set())
            if not t:
                continue
            untried = [v for v in vals if v not in t]
            if untried:
                print(f"  --{p}: tried {sorted(t)}, untried: {untried}")
        unexplored = [p for p in params if p not in bp]
        if unexplored:
            print(f"  Unexplored: {', '.join(sorted(unexplored))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Experiment tracker")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    b = sub.add_parser("best")
    b.add_argument("--backend")
    h = sub.add_parser("history")
    h.add_argument("--backend")
    h.add_argument("--status")
    h.add_argument("--limit", type=int)
    t = sub.add_parser("tried")
    t.add_argument("hash")
    d = sub.add_parser("diff")
    d.add_argument("num1", type=int)
    d.add_argument("num2", type=int)
    s = sub.add_parser("show")
    s.add_argument("num", type=int)
    g = sub.add_parser("gaps")
    g.add_argument("--backend")

    args = p.parse_args()
    {"status": cmd_status, "best": cmd_best, "history": cmd_history,
     "tried": cmd_tried, "diff": cmd_diff, "show": cmd_show, "gaps": cmd_gaps}[args.cmd](args)
