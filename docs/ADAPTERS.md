# Integrate a controller

The public repository layout is one shared core plus independent model adapters:

```text
motion-challenge/          # this package: SDK, data, physics, metrics, verification
motion-challenge-sonic/    # example repository name; model-specific inference
motion-challenge-grit/     # example repository name; model-specific inference
```

The model repositories above illustrate naming; they are not a list of published
or accepted submissions. A separate `motion-challenge-adapters` framework is no
longer required. Keep checkpoints, downloaded datasets, traces, and results in
external storage.

## Start locally

From a checkout of the core:

```sh
uv sync
uv run motion-challenge init-adapter --core-path . --output ../my-controller
cd ../my-controller
uv sync
uv run python smoke.py --data-cache /external/cache/lafan --output /external/runs/smoke.json --fetch
```

The generated `ReferencePD` is a working interface fixture, not a trained model
or a leaderboard entry. Replace `adapter.py` with model inference and declare
the native control rate and reference window in `runtime.json`.

The factory `create(root, config)` returns an object with `reset`, `act`, and
`close`. `reset` clears all episode history and recurrent state. `act` receives
named SI-unit state and reference packets and returns PD targets, gains, and
optional feedforward torques in canonical joint order. Quaternions are xyzw.
The kernel supplies references and owns the simulator, clock, actuator limits,
termination, and metrics. Adapters do not receive a clip name or score feedback.

For ONNX policies, `motion_challenge.adapters.ONNXController` supplies checked
weight/config loading, named joint mapping, and PD output assembly. The model
implements its observation layout and any native history/filtering behavior.
It can instead implement the SDK directly for PyTorch or another runtime.

## Publish a standalone adapter

Replace the generated local path dependency with a pinned Git dependency:

```toml
[project]
name = "my-motion-controller"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["motion-challenge"]

[tool.uv.sources]
motion-challenge = { git = "https://github.com/motion-challenge/motion-challenge.git", rev = "FULL_40_CHARACTER_CORE_COMMIT" }
```

Use an actual full commit SHA, then run `uv lock` and commit the resulting lock
file. A fresh clone can then run `uv sync --locked`. Each model owns its runtime
dependencies; `motion-challenge-eval --worker-python PATH` can use the model's
separately locked environment. That environment must include this SDK version.

Record immutable upstream code and checkpoint revisions, artifact SHA256 values,
native joint order/control frequency, observation/reset/action semantics, model
license, and known training-data overlap. Check inference on identical inputs
against author-maintained code before interpreting benchmark performance.

The runner's `runtime.json` is the executable development configuration. The
`validate-controller` command checks a separate draft submission manifest;
neither that declaration nor a successful smoke run grants official acceptance.

## Evaluate and verify

`motion-challenge paths` locates packaged LAFAN/G1 assets. Pass those paths to
`motion-challenge-eval` together with the controller, external dataset cache,
and external output path. Omit `--max-clips` and `--max-duration` for a full run.
Run `motion-challenge-verify` against the same inputs and output record.

Do not copy benchmark physics or scoring into the adapter. Keep all three
alignment scores and coverage statistics; an early failure's small tracking
error is not evidence of full-motion accuracy. Trace verification checks
consistency, not hostile-code isolation or proof that a controller executed.
