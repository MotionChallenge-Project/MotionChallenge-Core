# Third-party notices

## Canonical G1 robot asset

`robots/g1-29dof-cpu-v0.xml` is adapted from the G1 asset in
[mjlab](https://github.com/mujocolab/mjlab), commit
`8bb2bc9e187797689ec82c78c35019afd2fd0376`.

Copyright 2025, The mjlab Developers. The upstream Apache-2.0 license is retained
as `robots/LICENSE.mjlab` and included in built distributions.

MotionChallenge removed visual meshes, added a ground plane, set nominal
friction to 1, fixed reflected motor inertias, and added saturated torque motors
for per-step external PD control. The source and adapted asset SHA256 values,
motor profile, and modifications are recorded in `robots/g1-29dof-cpu-v0.json`.
These are evaluation assets, not a claim of identical hardware dynamics.

## LAFAN reference data

The suite manifest refers to
[LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
at a pinned revision. This repository distributes the manifest, not the motion
files. Upstream dataset terms remain applicable and are independent of the
core source license.

## Model runtimes

No checkpoint or third-party deployment framework is bundled. Independent
model repositories retain their own code/weight licenses and provenance.
