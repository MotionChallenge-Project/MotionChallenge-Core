# MotionChallenge Core

Open, reproducible benchmarks for humanoid motion tracking. One shared evaluator;
independent model adapters.

## Install and run

```bash
git clone https://github.com/MotionChallenge-Project/MotionChallenge-Core.git
cd MotionChallenge-Core
uv sync
```

Run a checkpoint from a demo adapter with `uv run evaluate.py`.
Each adapter pins `MotionChallenge-Core` and produces verified results using the shared protocol.

## Demo Model adapters

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
