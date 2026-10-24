# Code layout

```text
src/motion_challenge/
  sdk.py          # state, reference and control contracts
  data/           # dataset integrity and reference interpolation
  simulation/     # robot physics and MuJoCo CPU batch
  evaluation/     # rollout, metrics, trace verification and result assembly
  runtime/        # artifacts, adapter calls, CLI and progress
  adapters/       # optional ONNX inference utilities
  resources.py    # packaged benchmark assets
  validation.py   # shared hashes and contract checks
```

The root `entrypoint`, `physics`, `run`, `verify`, `merge` and `cli` modules are
small command entrypoints; implementation lives in the directories above.

## Core rules, flexible controllers

Core owns dataset integrity, reference timing, robot dynamics and limits, failure
conditions, metric definitions, result assembly and independent trace verification.
Model adapters receive state/reference packets and return controls. They do not
receive a simulator handle, failure flag or scoring interface.

An adapter implements `create(root, config)` and `reset` / `act` / `close`.
It can use any inference framework, multiple networks, recurrent caches, local
coordinate transforms, filters or action conversions without editing Core.
`ONNXController` is optional. Custom Python implementations use the same SDK.
Declared reference windows and control periods are recorded with each run.

Extra weights use `runtime.json`'s `artifacts`; a reproducible model export can
use its `prepare_checkpoint` hook. Model dependencies belong in the adapter's
locked environment. Model-specific processing stays in that repository.

Pin Core by its exact submodule commit. Published results retain their original
Core, adapter, checkpoint, dataset, robot and protocol identities. Adding a model
does not change benchmark rules. A new robot, sensor-access tier or scoring rule
needs a separately identified benchmark protocol; it cannot silently replace the
rules of an existing ranking.

These interfaces are not a hostile-code sandbox. Public acceptance also requires
source review, author-inference comparison and an independent repeat evaluation.

## Batch execution

`simulation/batch_dispatch.py` runs a single event loop over independent clip
state machines. One `VectorPhysics` owns the native MuJoCo batch; its CPU thread
count is independent of slot count. A longest-first queue refills completed
slots. Motion data and reference caches are retained only for active clips.

`runtime/policy.py` calls the adapter directly in the same process. Packet
adapters receive immutable SDK values; array adapters receive read-only numeric
views. Only approved proprioception and references are provided. Simulator
handles, measured translation, contacts, failures and scores stay inside Core.
An adapter that needs a separate inference service owns that connection.

This follows the vector reset/step organization used by
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) and
[mjlab](https://github.com/mujocolab/mjlab), with official MuJoCo CPU batch as the
physics backend. Completed clips refill independently. Metrics use refreshed
post-step kinematics, and failed motions continue to their end in the default
full-trajectory evaluation. These benchmark semantics are unchanged.

Metric accumulation, asynchronous compressed trace writing and progress output
are bounded per active clip. Verification workers are capped by CPU availability
and receive one clip's metadata, not repeated copies of the complete dataset.
The default command uses this executor at one slot and at thousands of slots.
