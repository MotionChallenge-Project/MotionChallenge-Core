# Evaluation protocol

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

The default engine is MuJoCo **3.17.0**, using official `mujoco.batch.Batch`
with persistent, independent simulation slots. `--jobs` sets the slot count;
`--threads` sets the CPU physics thread count (defaults to the slot count).
Each slot owns its controller process, observation history and reference cache.
Each slot receives a disjoint motion shard and leaves the batch when finished.
Reference processing and scoring run in independent processes to avoid Python
thread contention; only PD commands and generalized state cross physics IPC. PD is recomputed every 2 ms, and the exact
fractional endpoint is retained. Batch does not change author inference clocks,
physics, failures or scoring. `--backend serial` is available for parity checks.

The new `protocol.mujoco317-v1.json` pins the engine version. Earlier 3.16 results
retain their original protocol and remain a separate comparison track; scores
are never silently relabelled. Serial/batch tests compare poses, velocities,
failure times and metric traces exactly **within the same engine version**.

Progress is emitted as JSON every 5 seconds (`--progress-interval`), immediately
on clip completion and after download hash verification. Download events include
bytes, total size when known, rate and ETA. Evaluation events include completed /
total clips, passed clips and each active motion's simulated time. External
`*.progress.json` snapshots and JSONL events support dashboards and monitoring.


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


## Early termination and full-motion evaluation

`uv run evaluate.py` stops each clip on its first tracking failure.
`uv run evaluate.py --termination none` selects a separate frozen protocol and
continues to the exact final timestamp, without resets or reference changes.
Both modes retain the same failure predicates. Success means no failure at any
metric sample, and completion is the uninterrupted fraction before the first
failure. Later recovery does not erase a failure. Full-motion metrics include
post-failure states; invalid numerics or inference errors invalidate the run.

The headline position metric is mean per-body position error (MPBPE): mean
Euclidean error across 14 named body origins, averaged over time, then equally
across clips. Reports show millimeters. Each rollout mode has three alignment
views: no alignment, pelvis XY centering, and pelvis XY plus closest yaw. This
gives six values per model; the three full-motion values require a fresh run.
The compatibility JSON keys remain `mpjpe_*_m` and
`metric_means_over_observed_prefix`; `metric_scope` identifies whether the
observed trajectory is a failure prefix or the entire motion.
