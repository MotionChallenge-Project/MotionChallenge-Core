# Integrate a controller

Each model is an independent repository. The root-level `motion-challenge`
submodule pins the shared kernel; no second adapter framework is needed.

```text
motion-challenge-your-model/
  motion-challenge/       # Git submodule, exact core commit
  adapter.py             # model inference only
  runtime.json           # clocks, input access, checkpoint/config hashes
  policy.json            # native model configuration
  provenance.json        # upstream source and review evidence
  evaluate.py            # calls the core runner
  pyproject.toml
  uv.lock
```

## Install and evaluate

```sh
git clone --recurse-submodules https://github.com/OWNER/motion-challenge-your-model.git
cd motion-challenge-your-model
uv sync
uv run evaluate.py
```

`evaluate.py` contains only:

```python
from pathlib import Path
from motion_challenge.entrypoint import main
main(Path(__file__).parent)
```

Declare `motion-challenge = {path = "motion-challenge", editable = true}` in
`[tool.uv.sources]`. Commit the submodule pointer and `uv.lock`. For a new adapter,
`motion-challenge init-adapter --core-path CORE_CHECKOUT --output NEW_DIRECTORY`
creates a separate reference-PD interface fixture with the same layout. It is
not a trained model or a leaderboard submission.

The runner reuses a local ignored `checkpoints/policy.onnx` or the shared external
cache, validates its SHA256, and fetches a declared HTTPS artifact only when
missing. `--weights`, `--data`, `--output`, `--jobs` and `--offline` override local
storage/execution settings. Results, traces and receipts stay outside Git.
`--max-clips 1 --max-duration 2` is a smoke run and cannot enter a full ranking.

## Inference contract

The factory `create(root, config)` returns `reset`, `act`, and `close` methods.
`act` receives named SI-unit state and reference packets and returns PD targets,
gains and optional feedforward torque in canonical joint order. Quaternions are
xyzw. `reset` clears all history, filters and recurrent state. The core owns
reference interpolation, simulation, timing, limits, failures and scoring.
Adapters receive neither clip names nor scores.

`motion_challenge.adapters.ONNXController` supplies checked loading, named joint
mapping and PD assembly. Other inference frameworks can implement the SDK
without this utility. Do not copy dataset or evaluation logic into a model repo.

## Admission review

Pin the author's source revision and checkpoint. Review joint order, units,
coordinate frames, reference timing, normalization, history initialization,
recurrent state, clipping, action scaling, gains and filters against that source.
Store a reproducible test that compares inputs, raw outputs and final controls
on identical sequences, including reset and motion endpoints. Record numerical
tolerances and all intentional benchmark adaptations. A third-party export
requires equivalence evidence to the original checkpoint.

Full LAFAN40 completion, successful trace verification and author-inference
parity are separate review gates. An author has not submitted or endorsed a
result merely because their model is used. Record the actual submitter.
Benchmark physics remains fixed; author inference parity does not imply the
benchmark reproduces the author's training simulator or paper scores.

A receipt binds source/config/checkpoint versions and detects source changes
during execution. Trace verification recomputes scores, but does not sandbox
hostile code or attest that a particular controller actually executed.
