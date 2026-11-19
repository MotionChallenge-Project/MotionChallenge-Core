# Observation contract

The state contract is proprioceptive-no-position-no-linear-velocity-v1.

Allowed actual-state fields are joint names, joint positions, joint velocities, root orientation, root angular velocity and the evaluation clock. Histories and recurrent state follow the checkpoint's inference definition.

Actual global position and actual linear velocity are prohibited in all coordinate frames, including explicit velocity estimates. Required observations cannot be replaced with zeros or reference values to obtain admission. Checkpoints requiring prohibited inputs are excluded.

References contain requested/source time, root position, XYZW orientation, joint positions and declared motion derivatives. Adapters derive additional reference body features using their own pinned model. Core's scoring body poses are private to evaluation.

Admission binds the checkpoint, configuration, adapter and conversion-validation evidence. Dataset results and reproduction reports are separate artifacts.
