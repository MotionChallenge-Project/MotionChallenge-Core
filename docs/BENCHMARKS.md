# Build an independent benchmark

A benchmark project depends on the same pinned `MotionChallenge-Core` package as
model adapters. It owns its dataset manifest, protocol, documentation, and any
dataset conversion. The core owns reference interpolation, simulation, clocks,
failures, metric computation, shard merging, and result verification.

LAFAN40 is the bundled first suite. The current backend targets G1 29 DoF and
the canonical CSV reference schema. Supporting another robot, raw data format,
or failure family requires an explicit versioned core extension; arbitrary
formats are not automatically supported.

```text
another-benchmark/
  MotionChallenge-Core/  # shared core Git submodule
  pyproject.toml          # local path dependency on that submodule
  uv.lock
  dataset.json           # immutable source revision, files, hashes, joint names
  protocol.json          # explicit metric clock and supported failure settings
  README.md
```

Use the dependency example in [ADAPTERS.md](ADAPTERS.md). After `uv sync`, invoke
the core with the external suite/protocol and a model's runtime configuration:

```sh
uv run motionchallenge-eval \
  --suite ./dataset.json --protocol ./protocol.json \
  --robot /path/reported/by/motionchallenge-paths/g1-29dof-cpu-v0.json \
  --controller ../my-controller/runtime.json \
  --data /external/cache/my-suite --output /external/runs/evaluation.json
```

The robot path is available through `motionchallenge paths` or
`motion_challenge.resources.lafan_paths()['robot']`. The bundled dataset and
protocol JSON files are examples of supported schemas. A new suite declares
its own ID, complete motion set, source FPS, joint order, immutable source
revision, and file hashes. Convert raw motion into canonical references outside
the source repository; never silently filter difficult clips.

New protocol versions receive new IDs and files. Keep old protocol bytes for
verification of historical results. A leaderboard must separate different
robot/physics/protocol hashes and state-access tiers. Rank only complete suites
under one frozen configuration; smoke tests, missing clips, and infrastructure
failures cannot form full-suite scores.

All outputs live outside Git repositories. A separate ranking website can read
verified result artifacts from external storage; the core source repository is
not the results database. Per-motion traces enable independent recomputation.
