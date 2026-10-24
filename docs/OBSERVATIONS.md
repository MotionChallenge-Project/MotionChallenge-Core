# Observation admission

All evaluations use the frozen **ideal-proprioceptive-no-position-v1** contract.
The Core supplies only the permitted state fields below. The scorer privately
retains full state for failure checks and metrics.

## Allowed controller inputs

| Input | Definition |
| --- | --- |
| Joint state | Named joint angles (rad) and velocities (rad/s), noise-free |
| Root orientation | Full xyzw quaternion, including yaw, in the fixed world frame |
| Root velocities | Ideal linear (m/s) and angular (rad/s) velocity at the root body origin, world axes; not COM velocity |
| Derived proprioception | Gravity projection and local FK from current joint state, orientation and fixed geometry |
| Reference | Original reference poses, velocities and body kinematics, including world coordinates and the declared history/future window |
| Memory | Previous actions and author-native proprioceptive/recurrent history; cleared as documented at each reset |

## Forbidden controller inputs and computations

Actual root/world XYZ (including height), actual body world positions, actual
contact flags/forces/impulses, and errors derived from measured translation are
not supplied. Explicit velocity integration into displacement, odometry,
root-position estimators and contact-aided localization are prohibited. This
also applies to reset, initial history, reference alignment, preprocessing,
caches and adapter side channels. The scorer retains full state privately.

Reference-only arithmetic or finite differences of reference positions are
allowed. A reference must not be shifted using actual measured translation.
Training critics, rewards and teachers can use privilege; admission concerns
the deployed controller and its entire preprocessing/inference chain.

## Per-checkpoint handling

Each adapter records the exact checkpoint and auxiliary hashes, pinned author
source, permitted input dependencies, reset/history behavior, and any special
handling. Local FK and a fixed synthetic coordinate origin are allowed only
where they preserve the released inference numerically or implement a documented
author deployment path for the same weight. A synthetic origin is neither a
measurement nor a position estimate. Zeroing an indispensable policy input
without such evidence does not establish admission.

Admission requires source review, forbidden-input dependency tests, comparison
of observation tensors and motor targets with the supported author inference
path, and complete independently verified evaluations. Controllers must satisfy
this contract throughout initialization and inference.

Core removes prohibited fields before sending reset/act packets to either
single-controller or batched inference calls. `ProprioceptiveStatePacket` is
an explicit allowlist. `KinematicStateView` is an optional adapter-side FK
utility using only that packet and a fixed robot asset. External adapters can
implement their own FK or author deployment preprocessing without changing
Core rules. Process separation is not a hostile-code security sandbox; public
source audit and an independently controlled rerun remain necessary.

The rule prohibits **explicit** position estimation. It cannot prove that a
network with history never implicitly infers displacement. Ideal velocity and
orientation are not a certification of real-world sensor availability.
