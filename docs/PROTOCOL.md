# Evaluation protocol

LAFAN1 and BONES-SEED are separate tracks with frozen manifests in suites/. Scores are compared within the same dataset, robot and protocol.

Control and scoring run every 20 ms, with four physics steps per interval. The nominal physics step is 5 ms; the final interval ends exactly at the reference endpoint. G1 uses the pinned mjlab body, joint and collision configuration, contact sliding friction 1.0, and Newton with 50 maximum iterations and 20 line-search iterations. Joint PD commands are torque-limited at every physics step.

Observations follow [the strict state contract](OBSERVATIONS.md). Root translation and joint angles use linear interpolation; orientations use SLERP. Interpolated reference root poses and joint positions are quantized to float32 at the scoring-reference boundary. Independent CPU verification uses the same declared reference precision before float64 FK and metric calculations. Scoring reference body poses are precomputed on the metric clock with MuJoCo Warp. Actual body poses are read after integration and a kinematics refresh.

Failure occurs at the first sample with pelvis height error above 0.25 m or gravity-Z error above 0.8. Success requires reaching the endpoint without failure. Completion is the last passing sample time before first failure divided by motion duration. If the first failure occurs at 20 ms, completion is zero. Gravity-Z error is the absolute difference between the vertical components of actual and reference pelvis-frame gravity, R-transpose times (0, 0, -1).

Rollout continues after failure. Tracking errors cover the full motion and the prefix including the first failure frame. Each interval reports body-position error in world coordinates, after horizontal translation alignment, and after horizontal translation plus yaw alignment. Height remains in all three views. The initial t = 0 state is excluded from metric means. Metrics average bodies and frames per motion, then weight motions equally.

The result contains every selected motion and its trace manifest. Independent CPU MuJoCo verifies identities, clocks, FK, failure/completion and every metric. Numerical simulation failures produce no ranked full-suite result.
