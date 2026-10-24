# MotionChallenge Core

Open, reproducible benchmarks for humanoid motion tracking. One shared evaluator;
independent model adapters.

## Install and run

```bash
git clone https://github.com/MotionChallenge-Project/MotionChallenge-Core.git
cd MotionChallenge-Core
uv sync
```

Use a model repository to run its checkpoint. Each adapter pins this core as the
root-level `MotionChallenge-Core` submodule and provides `uv run evaluate.py`.
One command runs each complete motion once with MuJoCo 3.17 CPU batch
(`--jobs 8`). It records the first failure, continues to the final frame, and
computes both time ranges from the same verified trajectory. A unified
`result.json` contains success, completion, all six body-error views and one
source record. `rollout/` retains pose traces and verification evidence;
`summary.json` records progress. Downloads and evaluation report every five
seconds. `--termination early` and `--termination none` remain diagnostic options.
`--jobs` controls CPU rollout slots and independent trace verification workers.

## Model adapters

| Model | Repository |
| :-- | :-- |
| GRIT | [Evaluation repository](https://github.com/MotionChallenge-Project/motion-challenge-grit) |
| SONIC v1.1 | [Evaluation repository](https://github.com/MotionChallenge-Project/motion-challenge-sonic-v1-1) |
| HoloMotion v1.4.1 | [Evaluation repository](https://github.com/MotionChallenge-Project/motion-challenge-holomotion-v1-4-1) |
| PMT Flat | [Evaluation repository](https://github.com/MotionChallenge-Project/motion-challenge-pmt-flat) |

Clone an adapter with `--recurse-submodules`, then run `uv sync` and
`uv run evaluate.py`.

## Evaluation

Full-motion success, completion, joint error, and body-position error with
**XY+yaw / XY / no alignment**. Frozen physics and protocol; hashes and pose
traces support independent recomputation. Results, data, and weights stay outside
Git. Each result identifies its dataset, robot, checkpoint and protocol.

[Protocol](docs/PROTOCOL.md) · [Add a model](docs/ADAPTERS.md) ·
[Add a benchmark](docs/BENCHMARKS.md) ·
[Code layout](docs/ARCHITECTURE.md) ·
[Website](https://motionchallenge-project.github.io/motion-challenge-project/) ·
[Project](https://github.com/MotionChallenge-Project)

Apache-2.0. [Third-party notices](THIRD_PARTY_NOTICES.md).
