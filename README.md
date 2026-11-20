![MotionChallenge Core — An open, reproducible benchmark for humanoid motion tracking.](docs/images/motionchallenge-banner.svg)

<p align="center">
  <a href="https://motionchallenge-project.github.io/motionchallenge/">🏆 <b>Leaderboard</b></a> &nbsp; · &nbsp;
  <a href="https://motionchallenge-project.github.io/motionchallenge/models/">🤖 <b>Models</b></a> &nbsp; · &nbsp;
  <a href="https://motionchallenge-project.github.io/motionchallenge/protocol/">📖 <b>Protocol</b></a> &nbsp; · &nbsp;
  <a href="https://motionchallenge-project.github.io/motionchallenge/submit/">📤 <b>Submit</b></a>
</p>

GPU batch evaluation for Unitree G1 motion tracking with MuJoCo Warp. Each model is an independent repository with a pinned Core submodule, original checkpoint identity, batch inference adapter, local converter and dependency lock.

## Evaluate

Download the original motion dataset, then use a model repository:

~~~sh
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-heft
cd MotionChallenge-heft
uv sync --locked
uv run --locked python evaluate.py --dataset lafan --data /data/lafan --output results/lafan
uv run --locked python evaluate.py --dataset seed --data /data/seed/g1.tar.gz --output results/seed
~~~

Use Linux x86-64, Python 3.12 and an NVIDIA GPU compatible with the locked CUDA dependencies. The command downloads pinned original weights, converts them locally, validates original-versus-batch inference, evaluates all motions and verifies the recorded trajectories.

- LAFAN: obtain the G1 CSV files from [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset). The data directory contains g1/*.csv.
- SEED: obtain the original g1.tar.gz from [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed). Keep the archive intact. We evaluate a feasibility-filtered subset of the original G1 data. The [unified inclusion list](suites/seed/selection.jsonl.gz) fixes the same 129,561 motions for every model.
- Data stays local and is used under the upstream terms. Repositories contain selection metadata and software.

--cache selects a shared local cache (default ~/.cache/MotionChallenge), --device cuda:0 selects a GPU, and --weights PATH supplies an already downloaded original checkpoint. Original and converted weights are checked by hash. --offline uses available artifacts. --data-format lerobot-v3 explicitly selects an existing local LeRobot v3 release.

## Protocol

| Item | Configuration |
| --- | --- |
| Robot | G1, 29 joints; pinned mjlab bodies, joints and collision geometry |
| Control and scoring | 50 Hz |
| Physics | Four 5 ms steps per control period |
| Solver | Newton; maximum 50 iterations and 20 line-search iterations |
| Contact sliding friction | 1.0 |
| Environments | min(motion_count, 4096) |
| Scheduling | Longest motions first; refill completed slots |
| Inference | Full environment batch |
| Observation exclusions | Actual global position and actual linear velocity in every frame |
| Failure | Pelvis height error > 0.25 m or gravity-Z error > 0.8 |
| Position metric | MPKPE; 14 protocol-defined body-origin keypoints |
| Aggregation | Equal weight per motion |

Policies receive allowed state and root/joint reference information. Adapters construct their own reference features. Core precomputes scoring reference body poses with MuJoCo Warp before rollout and reads actual body poses from simulation.

All motions run to the endpoint. Results report success, completion and tracking errors over the complete motion and the prefix through first failure. Numerical simulation failures do not produce a ranked full-suite score.

MPKPE (Mean Per-Keypoint Position Error) measures the 3D Euclidean distance between actual and reference keypoint positions. The protocol defines these keypoints at robot body-frame origins: 14 points for G1. Errors are averaged over keypoints and scoring samples within each motion, then motions are weighted equally. Results report full-motion and first-failure-prefix scores with no alignment, XY alignment, and XY + yaw alignment. XY alignment removes pelvis horizontal translation while retaining height error. Leaderboard values are reported in millimetres (mm).

## Caches and evidence

Original SEED conversion uses up to eight CPU workers and checks archive identity, selection, frame counts, quaternion validity and the canonical pose hash. Dataset conversion and reference FK are shared across models. Use a local SSD cache for large datasets.

Each result binds protocol, original and converted weights, source hashes, canonical poses, simulation settings and traces. Independent CPU MuJoCo recomputes body poses and metrics.

~~~sh
uv run motionchallenge-verify results/lafan/result.json --jobs 8
uv run motionchallenge-reproduce results/lafan results/lafan-repeat --output reproduction.json
~~~

Reproduction compares matching data, checkpoint, adapter and simulator identities. Aggregate and per-motion numerical differences are reported separately from successful execution of the same pipeline.

See [protocol](docs/PROTOCOL.md), [observations](docs/OBSERVATIONS.md), [adapter interface](docs/ADAPTERS.md), [datasets](docs/BENCHMARKS.md) and [batching](docs/BATCHING.md).

## Author checkpoints and validation

Ranked model entries use pinned original author releases. Each model repository records its author release URL, source artifact SHA256 and any local conversion chain in validation/checkpoint-origin.json. Duplicate exports of the same trained weights use one official checkpoint entry.

The [model registry](docs/mjwarp-model-admissions.json) contains 23 checkpoints with verified batch inference and seven checkpoints excluded by the observation contract. Twenty-two checkpoints have complete independently verified LAFAN and SEED results. Humanoid-GPT 264 has conversion-parity evidence and numerical-failure diagnostics for both datasets; it has no ranked full-suite score.

[4096-world throughput comparison](docs/performance/seed-core-4096-comparison.md): on an A100 80GB, Core/HEFT takes 38.509 ms per control step, compared with 42.782 ms for native SEED and 43.026 ms for SEED with Newton50. The measured window includes inference and simulation; policies and some runtime settings differ.
