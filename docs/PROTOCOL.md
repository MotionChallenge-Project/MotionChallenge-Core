# Evaluation protocol

## Evaluation semantics

The primary ranking metric is full-motion MPBPE with XY translation alignment
only, in millimeters; lower is better. Yaw and height errors remain in the score.

Failure rules follow [SuperMimic Appendix A](https://arxiv.org/pdf/2610.28480#page=13):
root height difference greater than 0.25 m or projected unit-gravity Z difference
greater than 0.8, using the robot and reference root frames. The G1 root is the
pelvis. End-effector deviation is not a failure condition. Equality passes.
Protocol v2 is the sole maintained evaluation protocol.
Observations, physics, references and metric formulas are unchanged.

All motions in the selected suite are retained, hash checked and evaluated with
the same policy and checkpoint, from the first frame to the exact final source
timestamp. Motion count and source FPS come from the suite manifest. Position and
joint angles are linearly interpolated; orientation uses shortest-path SLERP.
Generalized velocities use a centered source-interval difference, one-sided at
endpoints. FK and body-origin velocity use the pinned canonical robot.
Requested past/future reference timestamps clamp at the endpoints independently;
lookahead does not shorten the episode. Preview and reference history are declared
in seconds. Actual observation history belongs to the controller and resets at
each episode.

The bundled nominal protocol uses no observation noise or randomization and
friction 1. Each benchmark pins its robot, physics and protocol settings.
PD is evaluated and saturated every physics step. Metrics compare post-physics
state to the reference at the same timestamp. Policy frequency may differ by
checkpoint; scoring uses a common clock and always checks the final transition.
A final-transition failure counts as failure. The protocol uses
reference-relative root height and projected-gravity-z conditions; intentional floor poses are not rejected by a constant pelvis floor.

Reports include per-clip success, completion, failure time/reasons, joint MAE,
root translation/orientation error and the three position scores below. Prefix-only
tracking errors are conditional on survival and must be read alongside coverage.
An evaluator crash is an invalid run, not a policy fall. Smoke truncation is
explicitly marked and cannot count as full-suite success.

### Three position alignment modes

The protocol declares the scored body origins. All three modes use the same
bodies and timestamps. Values are mean Euclidean distances in metres (multiply by
1000 to display mm), not component-wise MAE or joint-angle errors.

| Result field | Scoring transform | Errors retained |
| --- | --- | --- |
| `mpjpe_xy_yaw_aligned_m` | Centre each root XY; align root heading | Height, tilt and articulated pose |
| `mpjpe_xy_aligned_m` | Centre each root XY | Heading, height, tilt and articulated pose |
| `mpjpe_world_m` | None, including no initial fitted transform | Global trajectory and pose |

Yaw alignment uses the closest world-Z rotation between the two root
orientations, rather than Euler yaw extraction or a best fit to keybody points.
This remains defined when both root forward axes are vertical. An opposite-tilt
singularity uses zero correction, retains the sample and reports
`yaw_alignment_degenerate_fraction`; this fraction must accompany affected scores.
No mode aligns world Z, roll/pitch, scale or time. Alignment only transforms
temporary scoring arrays; it never edits the simulation, policy inputs or reference.

Root XY, root height and absolute yaw-correction errors are also reported.
`mpjpe_root_relative_world_axes_m` remains a legacy diagnostic: it removes XYZ
translation, including height, and retains yaw. It is not a synonym for any of
the three main modes. Metric alignment cannot turn a failed rollout into a success.
All three scores are produced for every model.

### Parallel evaluation

The default engine is MuJoCo **3.17.0**, using official `mujoco.batch.Batch`
with persistent, independent simulation slots. `--jobs` sets the slot count;
`--threads` sets CPU physics threads; defaults depend on slot count and available
CPUs (see [batching](BATCHING.md)).
Core and policy run in the same process. Each slot owns its history and reference
cache; completed slots take the next longest motion. PD is recomputed at every
configured physics step, including the exact fractional endpoint. Scheduling does
not change author inference clocks, physics, failure conditions or scoring.

The public evaluator supports only MuJoCo 3.17.0 batch. The two bundled
protocols select early termination or full-motion continuation. Internal
reference-world tests check batch poses, velocities, failure times and metric
traces exactly within this engine version.

Progress is emitted as JSON every 5 seconds (`--progress-interval`), immediately
on clip completion and after download hash verification. Download events include
bytes, total size when known, rate and ETA. Evaluation events include completed /
total clips, passed clips and each active motion's simulated time. External
`*.progress.json` snapshots and JSONL events support dashboards and monitoring.


`motionchallenge-eval --clip-id <exact-suite-id>` evaluates selected complete
motions without changing the suite manifest. Repeat the flag for a shard.
`uv run python -m motion_challenge.merge --output ... --suite ... --robot ...
--protocol ... --data ... --controller ... shard1.json shard2.json ...` verifies
every trace before composing a full-suite record. Missing, duplicate, truncated
or differently configured shards cannot form a ranking. Outputs and copied
traces remain outside repositories. Parallelization does not change policy or
metric timing, reference data, or failure rules.

## Verification and limits

Every controller uses the [proprioceptive observation contract](OBSERVATIONS.md).
Acceptance requires exact checkpoint provenance, author-inference comparison,
two complete repeatable evaluations and independent trajectory verification.
The in-process SDK is not a hostile-code security sandbox; a public
submission also requires source review and a controlled rerun.

No checkpoint retraining is required by the interface. Training overlap must be
disclosed; inference-only testing does not imply an unseen dataset.
Dataset files remain upstream under their own terms. Each suite manifest records
its source revision and file hashes; see [benchmark setup](BENCHMARKS.md).

Result verification is also in the core: `uv run motionchallenge-verify` takes
the result, suite, robot, protocol, dataset cache and controller configuration.
It validates hashes and independently recomputes FK, failures and metrics from
the compressed pose trace stored beside the external result. This checks
consistency; leaderboard publication also requires a controlled rerun.


## One evaluation result

`uv run evaluate.py` runs each motion once through the exact final timestamp.
The first failure determines success and completion. The same verified trajectory
produces tracking errors both until that failure (inclusive) and over the entire
motion. Later recovery does not erase a failure; no reset or reference shift is
introduced. Inference errors and invalid numerics invalidate the run.

MPBPE is the mean Euclidean error over the protocol’s declared body origins,
averaged over samples within each motion and then equally over motions. Each
time range has three alignment views, giving six values in one `result.json`. The JSON keys retain `mpjpe_*_m` for compatibility;
the website displays millimeters. `--termination early` and `--termination none`
are diagnostic options, not separate ranking paths.
