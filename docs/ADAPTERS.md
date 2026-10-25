# Integrate a controller

Each model is an independent repository. The root-level `MotionChallenge-Core`
submodule pins the shared kernel; no second adapter framework is needed.

```text
MotionChallenge-your-model/
  MotionChallenge-Core/  # Git submodule, exact core commit
  adapter.py             # model inference only
  runtime.json           # clocks, input access, checkpoint/config hashes
  policy.json            # native model configuration
  provenance.json        # upstream source and review evidence
  admission.json         # exact-checkpoint observation validation
  evaluate.py            # calls the core runner
  pyproject.toml
  uv.lock
```

## Install and evaluate

```sh
git clone --recurse-submodules https://github.com/OWNER/MotionChallenge-your-model.git
cd MotionChallenge-your-model
uv sync
uv run evaluate.py
```

`evaluate.py` contains only:

```python
from pathlib import Path
from motion_challenge.entrypoint import main
main(Path(__file__).parent)
```

Declare `motionchallenge-core = {path = "MotionChallenge-Core", editable = true}` in
`[tool.uv.sources]`. Commit the submodule pointer and `uv.lock`. For a new adapter,
`motionchallenge init-adapter --core-path CORE_CHECKOUT --output NEW_DIRECTORY`
creates a separate reference-PD interface fixture with the same layout. It is
not a trained model or a leaderboard submission.

The runner reuses a local ignored `checkpoints/policy.onnx` or the shared external
cache, validates its SHA256, and fetches a declared HTTPS artifact only when
missing. `--weights`, `--data`, `--output`, `--jobs` and `--offline` override local
storage/execution settings. Results, traces and receipts stay outside Git.
`--max-clips 1 --max-duration 2` is a smoke run and cannot enter a full ranking.

Multi-file policies can add an `artifacts` mapping to `runtime.json`:

```json
{"artifacts": {"encoder": {"path": "checkpoints/encoder.onnx", "sha256": "64 hex digits", "url": "https://author.example/encoder.onnx"}}}
```

The core reuses, downloads and verifies each component before evaluation.
Use `--artifact encoder=/path/to/encoder.onnx` for a local override. The resolved
paths and hashes are bound into the runtime receipt. Model code reads
`config["artifacts"]["encoder"]["path"]`; data and evaluation remain in the core.

## Inference contract

The factory `create(root, config)` returns `reset`, `act`, and `close` methods.
`act` receives named SI-unit state and reference packets and returns PD targets,
gains and optional feedforward torque in canonical joint order. Quaternions are
xyzw. `reset` clears all history, filters and recurrent state. The core owns
reference interpolation, simulation, timing, limits, failures and scoring.
Adapters receive neither clip names nor scores.

Every controller receives `ProprioceptiveStatePacket`, including full orientation,
joint state and root velocities. World position, contact feedback and explicit
position estimation are prohibited. Full reference motion remains available.
See the frozen [observation rules](OBSERVATIONS.md).

Declare the single observation contract in `runtime.json`:

```json
{
  "state_access": "ideal-proprioceptive-no-position-v1",
  "entrypoint": "adapter:create",
  "observation_contract": {
    "admission_record": "admission.json",
    "admission_sha256": "SHA256 of the admission file"
  }
}
```

The runner requires an `inference_verified` admission record for the exact
checkpoint hash before downloading data or starting evaluation. The generated
reference-PD fixture can use `smoke.py` for interface development; it cannot
produce an admitted checkpoint result without this evidence.

`motion_challenge.adapters.ONNXController` supplies checked loading, named joint
mapping and PD assembly. Other inference frameworks can implement the SDK
without this utility. Do not copy dataset or evaluation logic into a model repo.

Keep model-specific extensions behind this factory: additional networks,
history/caches, observation transforms, filters and control conversions require
no Core edits. See [the code layout and ownership boundaries](ARCHITECTURE.md).

### Optional batched inference

`--jobs N` controls active simulation slots; it does not create N processes.
One Core loop uses MuJoCo's CPU thread pool and calls the policy directly in
the same process. Engines and any remote-inference connection stay in the adapter.
Motions are scheduled longest first, and free slots immediately receive the next
motion. Every motion still starts at its first frame with independent history.

Adapters may implement `batch_entrypoint` for SDK packets, or
`array_batch_entrypoint` for read-only NumPy views. Both use `reset_batch`,
`act_batch` and `close`; histories belong to slot IDs, not batch row numbers.
The scalar `entrypoint` remains usable for controllers without batch support.
Its model calls run individually; declaring more slots alone does not batch them.

`--inference-batch-size N` (or runtime `inference_batch_size`) limits samples per
policy call independently of `--jobs` and MuJoCo `--threads`. Any positive sizes,
partial resets and tail batches are supported. The adapter owns fixed-shape
padding, device placement and inference engines, including TensorRT.

For fixed-batch native CPU models, an adapter can wrap its scalar factory with
`motion_challenge.adapters.parallel.ParallelControllers(root, runtime, create_single)`
and declare that factory as `batch_entrypoint`. It keeps one controller per slot
and uses a persistent pool, capped by `inference_workers` (default 8) and CPU affinity.
Use single-threaded model sessions; mutable histories and RNGs must be per slot.
Validate outputs and closed-loop traces before opting in. Small models may be faster
with `inference_workers: 1` or a true array batch implementation.

See [batch interfaces and resource limits](BATCHING.md) for array shapes and an
example. Validate observations, outputs and controls against the author's
single-sample inference before enabling batched model execution.

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
