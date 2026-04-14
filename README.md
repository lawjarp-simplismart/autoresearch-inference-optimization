# autoresearch-inference-optimization

Autonomous inference serving optimization. An AI agent iterates on server configs, benchmarks them, and tracks results.

## Setup

```bash
uv sync
```

## Usage

1. Branch off master for each model:
   ```bash
   git checkout -b autoresearch/<model-tag>
   ```

2. Fill in `user_config.yaml` (model, hardware, remote, constraints, search_space)

3. Fill in `serve.sh` (server launch script — full bash control)

4. Run the agent:
   ```bash
   claude --dangerously-skip-permissions
   ```
   The agent reads `CLAUDE.md`, edits `serve.sh` + `experiment.yaml`, and loops autonomously.

5. Or run manually:
   ```bash
   uv run engine.py run              # run one experiment
   uv run engine.py remote run       # run on remote machine
   ```

## Commands

```bash
uv run engine.py status              # summary stats
uv run engine.py best                # best config + metrics + serve.sh
uv run engine.py history             # all experiments
uv run engine.py show N              # details of experiment N
uv run engine.py diff N M            # diff serve.sh between experiments
uv run engine.py compare N M         # param + metric delta
uv run engine.py gaps                # untried params from search_space
uv run engine.py check-gpus          # GPU availability
uv run engine.py kill-server         # kill leftover server
uv run engine.py remote run          # sync + run on remote (detached)
uv run engine.py remote health       # remote GPU status
```

## How it works

- Agent edits `serve.sh` (execution) and `experiment.yaml` (structured tracking)
- `engine.py run` launches the server, benchmarks it, scores against constraints, saves results
- Experiments are logged to `experiments.jsonl` with snapshots in `experiments/`
- Master stays clean as a template; each model lives on its own branch
