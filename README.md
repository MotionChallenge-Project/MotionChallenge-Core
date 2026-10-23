# Motion Challenge

[Core repository](https://github.com/motion-challenge/motion-challenge) ·
[Controller integration](docs/ADAPTERS.md) · [Independent benchmarks](docs/BENCHMARKS.md)

One repository and one uv package provide the evaluation kernel, frozen suite
and robot assets, controller SDK, ONNX inference utilities, verification, and
external-project bootstrap. A separate adapters framework is no longer required.
Model-specific observations, checkpoints, and model runtime dependencies belong
to separate consumer repositories. No package imports a third-party sim2real stack.

```sh
git clone https://github.com/motion-challenge/motion-challenge.git
cd motion-challenge
uv sync
uv run motion-challenge paths
uv run motion-challenge init-adapter --core-path . --output ../my-controller
cd ../my-controller
uv sync
uv run python smoke.py --data-cache /external/cache/lafan --output /external/runs/smoke.json --fetch
```

The generated reference-PD controller is an interface fixture, not a trained
baseline. Replace its inference code and runtime manifest with the model's
original release contract. Other benchmark projects can import
`motion_challenge.sdk`, `motion_challenge.resources`, and
`motion_challenge.adapters`, or invoke the same CLI without copying scoring code.

The development consumer uses a relative uv path dependency. Public consumers
pin a full commit of this repository in `pyproject.toml` and commit their own
`uv.lock`; see [controller integration](docs/ADAPTERS.md). No submodule or separate
adapters framework is required. Packaged wheels include frozen robot and suite
assets and work outside the source checkout.

Data, checkpoints, results, pose traces and logs remain in external storage.
The core does not contain model-specific implementations or leaderboard records.
Source provenance and identical-input parity against author-maintained inference
are required for model acceptance; third-party deployment packages are reference
material, not the benchmark's source of truth. Trace recomputation is available;
security isolation and organizer-signed official acceptance are separate work.

## Install and run

```sh
git clone https://github.com/motion-challenge/motion-challenge.git
cd motion-challenge
uv sync
uv run python -m unittest discover -s tests -v
uv run motion-challenge inspect-suite suites/lafan/dataset.json
uv run motion-challenge fetch-dataset \
  --manifest suites/lafan/dataset.json --root /path/outside/repos/data/lafan
uv run motion-challenge audit-dataset \
  --manifest suites/lafan/dataset.json --root /path/to/raw \
  --output /path/outside/repos/audit.json
uv run motion-challenge-eval \
  --controller /path/to/model/runtime.json \
  --suite suites/lafan/dataset.json \
  --protocol suites/lafan/protocol.cpu-development-v1.json \
  --robot robots/g1-29dof-cpu-v0.json \
  --data /path/to/raw --output /path/outside/repos/evaluation.json
```

`uv.lock` pins the runtime. The evaluator launches the adapter in a separate
process. Models can supply their own locked Python environment; all evaluation
logic remains in this package. No `pip install` or manual PYTHONPATH is needed
for the core and generated consumer.

## Evaluation semantics

All 40 source clips are retained, hash checked and evaluated from their first
frame to the exact final source timestamp. Source FPS is 30. Position and joint
angles are linearly interpolated, orientation uses shortest-path SLERP.
Generalized velocities use a centered source-interval difference, one-sided at
endpoints. FK and body-origin velocity use the pinned canonical robot.
Requested past/future reference timestamps clamp at the endpoints independently;
lookahead does not shorten the episode. Preview and reference history are declared
in seconds. Actual observation history belongs to the controller and resets at
each episode.

Nominal evaluation uses no observation noise or randomization and friction 1.
PD is evaluated and saturated every physics step. Metrics compare post-physics
state to the reference at the same timestamp. Policy frequency may differ by
checkpoint; scoring uses a common clock and always checks the final transition.
A final-transition failure counts as failure. The development protocol uses
reference-relative torso height, projected-gravity-z and end-effector height
conditions; intentional floor poses are not rejected by a constant pelvis floor.

Reports include per-clip success, completion, failure time/reasons, joint MAE,
root translation/orientation error and the three position scores below. Prefix-only
tracking errors are conditional on survival and must be read alongside coverage.
An evaluator crash is an invalid run, not a policy fall. Smoke truncation is
explicitly marked and cannot count as full-suite success.

### Three position alignment modes

The v1 development protocol scores the same 14 body origins and timestamps in
all three modes. Values are mean Euclidean distances in metres (multiply by
1000 to display mm), not component-wise MAE or joint-angle errors.

| Result field | Scoring transform | Errors retained |
| --- | --- | --- |
| `mpjpe_xy_yaw_aligned_m` | Centre each pelvis XY; align pelvis heading | Height, tilt and articulated pose |
| `mpjpe_xy_aligned_m` | Centre each pelvis XY | Heading, height, tilt and articulated pose |
| `mpjpe_world_m` | None, including no initial fitted transform | Global trajectory and pose |

Yaw alignment uses the closest world-Z rotation between the two pelvis
orientations, rather than Euler yaw extraction or a best fit to keybody points.
This remains defined when both pelvis forward axes are vertical. An opposite-tilt
singularity uses zero correction, retains the sample and reports
`yaw_alignment_degenerate_fraction`; this fraction must accompany affected scores.
No mode aligns world Z, roll/pitch, scale or time. Alignment only transforms
temporary scoring arrays; it never edits the simulation, policy inputs or reference.

Root XY, root height and absolute yaw-correction errors are also reported.
`mpjpe_root_relative_world_axes_m` remains a legacy diagnostic: it removes XYZ
translation, including height, and retains yaw. It is not a synonym for any of
the three main modes. The original `protocol.cpu-development.json` retains the
v0 metric set, and existing traces/results must use their original protocol hash.
The v1 failure conditions are unchanged from v0; metric alignment cannot turn a
failed rollout into a success. All three scores are produced for every model.

### Parallel evaluation

`motion-challenge-eval --clip-id <exact-suite-id>` evaluates selected complete
motions without changing the suite manifest. Repeat the flag for a shard.
`uv run python -m motion_challenge.merge --output ... --suite ... --robot ...
--protocol ... --data ... --controller ... shard1.json shard2.json ...` verifies
every trace before composing a full-suite record. Missing, duplicate, truncated
or differently configured shards cannot form a ranking. Outputs and copied
traces remain outside repositories. Parallelization does not change policy or
metric timing, reference data, or failure rules.

## Status and limits

The CPU runner and inference process bridge are development implementations.
Dataclasses and subprocesses are **not a hostile-code security sandbox**. Official
acceptance still requires source/weight provenance, upstream inference parity,
validated full-suite execution and a restricted container runner. Current outputs
are labelled development/unverified and are not official leaderboard scores.
The first state-access tier is privileged; it does not claim sensor-only deployment.

No checkpoint retraining is required by the interface. Training overlap must be
disclosed; inference-only testing does not imply an unseen dataset.
Dataset files remain upstream under their own terms:
https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset

Result verification is also in the core: `uv run motion-challenge-verify` takes
the result, suite, robot, protocol, dataset cache and controller configuration.
It validates hashes and independently recomputes FK, failures and metrics from
the compressed pose trace stored beside the external result. This checks
consistency; an official service must still rerun the controller.

## License

The core is released under Apache-2.0. The adapted mjlab robot asset retains its
upstream license and provenance; see [third-party notices](THIRD_PARTY_NOTICES.md).
Dataset and model licenses are independent. Dataset files and model weights are
not redistributed by this repository.
