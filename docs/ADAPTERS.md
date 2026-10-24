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

The core reuses, downloads and verifies each component before starting workers.
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

`--jobs N` accepts any positive parallel slot count, limited by the available
motions. Existing `reset/act/close` adapters need no changes. The core caches
immutable reference encoding and runs their independent policy workers.

For a model that benefits from batched inference, declare
`"batch_entrypoint": "adapter:create_batch"` in `runtime.json`. Its factory
returns a `motion_challenge.sdk.BatchController` implementing:

```python
reset_batch(slots, states, references, seeds)
act_batch(slots, states, references)  # one ControlCommand per supplied slot
close()
```

Each call may contain any nonempty subset of slots. Keep histories and recurrent
state separate per slot; reset only the supplied slots. Commands use the same
order as `slots`. The optional positive `inference_batch_size` caps each call
independently of `--jobs`; tail batches may be smaller. Fixed-shape models can
pad, split or execute individually inside their adapter, preserving author
semantics. Specialized runtimes and devices belong in the adapter's dependencies.

The core owns scheduling and sends the same state/reference packets as the
single-slot interface. Model workers cannot request different physics, clocks,
failure rules or metrics. Compare batched outputs against independent inference
over resets, partial batches and long sequences before enabling this path.

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
