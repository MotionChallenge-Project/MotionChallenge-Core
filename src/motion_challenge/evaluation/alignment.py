"""Three position scores from one immutable, same-time pair of poses.

The heading correction is the closest world-Z rotation to the relative pelvis
rotation in SO(3). It uses the whole pelvis rotation, not Euler yaw or a fit to
the body points. Consequently a common pitch near 90 degrees is well behaved.
At the exact ambiguous case all yaw corrections have the same orientation
cost; use zero, retain the error sample and expose the ambiguity fraction.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from motion_challenge.data.reference import normalize_quaternion


POSITION_ALIGNMENT_VERSION = 'pelvis_xy_and_closest_yaw_v1'


def keybody_position_errors(actual, reference, actual_root, reference_root,
                            actual_quaternion_xyzw, reference_quaternion_xyzw):
    a, r = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
    ar, rr = np.asarray(actual_root, dtype=float), np.asarray(reference_root, dtype=float)
    if a.ndim != 2 or a.shape[1:] != (3,) or not len(a) or r.shape != a.shape:
        raise ValueError('keybody positions must be matching nonempty Nx3 arrays')
    if ar.shape != (3,) or rr.shape != (3,):
        raise ValueError('pelvis positions must be xyz vectors')
    if not all(np.isfinite(v).all() for v in (a, r, ar, rr)):
        raise ValueError('non-finite keybody or pelvis position')
    aq = Rotation.from_quat(normalize_quaternion(actual_quaternion_xyzw))
    rq = Rotation.from_quat(normalize_quaternion(reference_quaternion_xyzw))
    relative = (rq * aq.inv()).as_matrix()
    cosine = relative[0, 0] + relative[1, 1]
    sine = relative[1, 0] - relative[0, 1]
    degenerate = math.hypot(cosine, sine) <= 1e-12
    angle = 0.0 if degenerate else math.atan2(sine, cosine)
    yaw = Rotation.from_rotvec((0., 0., angle))
    # Copies: neither the actual nor reference packet is ever edited.
    ax, rx = a.copy(), r.copy()
    ax[:, :2] -= ar[:2]
    rx[:, :2] -= rr[:2]
    mean_distance = lambda x, y: float(np.linalg.norm(x-y, axis=1).mean())
    return {
        'mpjpe_world_m': mean_distance(a, r),
        'mpjpe_xy_aligned_m': mean_distance(ax, rx),
        'mpjpe_xy_yaw_aligned_m': mean_distance(yaw.apply(ax), rx),
        'root_xy_error_m': float(np.linalg.norm(ar[:2]-rr[:2])),
        'root_height_error_m': float(abs(ar[2]-rr[2])),
        'root_yaw_correction_abs_rad': float(abs(angle)),
        'yaw_alignment_degenerate_fraction': float(degenerate),
    }
