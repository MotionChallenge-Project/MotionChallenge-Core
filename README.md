# Motion Challenge

Open, reproducible benchmarks for humanoid motion tracking. One shared evaluator;
independent model adapters.

## Install and run

```bash
git clone https://github.com/motion-challenge-project/motion-challenge.git
cd motion-challenge
uv sync
```

Use a model repository to run its checkpoint. Each adapter pins this core as the
root-level `motion-challenge` submodule and provides `uv run evaluate.py`.
One command runs both early-termination and full-motion evaluation with MuJoCo
3.17 CPU batch (`--jobs 8`). Downloads and evaluation report progress every five
seconds. A unified `result.json` contains success, completion, all six body-error views,
per-motion details and one source record. `early/` and `none/` retain the underlying
traces and receipts; `summary.json` records progress. Use `--termination early` or `none` only when
you need a single mode. No branch switching is needed.

## Evaluation

Full-motion success, completion, joint error, and body-position error with
**XY+yaw / XY / no alignment**. Frozen physics and protocol; hashes and pose
traces support independent recomputation. Results, data, and weights stay outside
Git. Each result identifies its dataset, robot, checkpoint and protocol.

[Protocol](docs/PROTOCOL.md) · [Add a model](docs/ADAPTERS.md) ·
[Add a benchmark](docs/BENCHMARKS.md) ·
[Website](https://motion-challenge-project.github.io/motion-challenge-project) ·
[Project](https://github.com/motion-challenge-project)

Apache-2.0. [Third-party notices](THIRD_PARTY_NOTICES.md).
