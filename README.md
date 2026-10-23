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

## Evaluation

Full-motion success, completion, joint error, and body-position error with
**XY+yaw / XY / no alignment**. Frozen physics and protocol; hashes and pose
traces support independent recomputation. Results, data, and weights stay outside
Git. Each result identifies its dataset, robot, checkpoint and protocol.

[Protocol](docs/PROTOCOL.md) · [Add a model](docs/ADAPTERS.md) ·
[Add a benchmark](docs/BENCHMARKS.md) ·
[Website](https://motion-challenge-project.epiccabin1025.chatgpt.site) ·
[Project](https://github.com/motion-challenge-project)

Apache-2.0. [Third-party notices](THIRD_PARTY_NOTICES.md).
