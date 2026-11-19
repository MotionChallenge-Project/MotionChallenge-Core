# Motion datasets

## LAFAN1

The frozen dataset contains 40 G1 motions and 264,705 frames at 30 Hz. Download the files specified by suites/lafan/dataset.json into a directory containing g1/. CSV rows contain root translation in metres, XYZW quaternion and 29 joint positions in radians.

## BONES-SEED

We evaluate a feasibility-filtered subset of the original BONES-SEED G1 motion data. Every model uses the same fixed subset.

Download the original g1.tar.gz. SHA256: 52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a.

suites/seed/filter.json binds the archive and a single selection.jsonl.gz ledger. Each row records archive path, frame count and canonical motion identity. The benchmark contains 129,561 motions and 113,674,960 source frames at 120 Hz.

Source CSV columns contain frame index, root XYZ translation in centimetres, root XYZ Euler orientation in degrees, and named joint angles in degrees. Conversion produces float32 root XYZ metres, XYZW quaternion and 29 joint radians. Parallel parsing preserves canonical ordering.

## Local cache

poses.npy stores concatenated canonical poses. dataset.json records joint order, motion names, starts, lengths, frame rate, hashes and conversion provenance. No upload is needed.

Original data is the default one-command workflow. --data-format lerobot-v3 explicitly selects an existing local LeRobot v3 release. Its motion identities, frames and joint schema must match the same frozen suite.
