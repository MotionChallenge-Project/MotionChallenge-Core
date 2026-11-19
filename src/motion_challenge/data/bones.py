"""Original BONES-SEED G1 CSV decoding, preserving native frames and time."""
import csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from motion_challenge.data.reference import Motion


def load_bones_motion(path, joint_names, fps=120.0):
    path=Path(path)
    with path.open(newline="") as stream:
        reader=csv.reader(stream)
        header=next(reader)
        expected=["Frame", *("root_translate"+axis for axis in "XYZ"),
                  *("root_rotate"+axis for axis in "XYZ"),
                  *(name+"_dof" for name in joint_names)]
        if len(header)!=len(set(header)) or set(header)!=set(expected):
            raise ValueError(f"Unexpected BONES-SEED columns: {path}")
        order=[header.index(name) for name in expected]
        values=np.asarray(list(reader),dtype=np.float64)
    if values.ndim!=2 or len(values)<2 or values.shape[1]!=len(expected):
        raise ValueError("Motion must have at least two complete frames")
    values=values[:,order]
    if not np.isfinite(values).all() or not np.array_equal(values[:,0],np.arange(len(values))):
        raise ValueError("Nonfinite poses or noncontiguous source frame indices")
    if not np.isfinite(fps) or fps<=0:
        raise ValueError("Invalid source FPS")
    # Match the v3 conversion: calculate in float64, store each pose as float32.
    positions=(values[:,1:4]/100).astype(np.float32)
    xyzw=Rotation.from_euler("xyz",values[:,4:7],degrees=True).as_quat().astype(np.float32)
    joints=np.deg2rad(values[:,7:]).astype(np.float32)
    return Motion(tuple(joint_names),tuple(i/fps for i in range(len(values))),joints,positions,xyzw)
