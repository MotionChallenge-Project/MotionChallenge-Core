"""Vectorized scoring; the scalar verifier remains an independent implementation."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from motion_challenge.data.reference import normalize_quaternion
from motion_challenge.data.bodies import body_values
from motion_challenge.evaluation.alignment import POSITION_ALIGNMENT_VERSION


def tracking_batch(states, references, protocol):
    if len(states) != len(references) or not states:
        raise ValueError('scoring requires matching nonempty batches')
    cfg = protocol['failure']
    names = protocol['metric_bodies']
    selected = (cfg['anchor_body'], *cfg['end_effector_bodies'], *names)
    actual = np.stack([body_values(state.bodies,selected) for state in states])
    desired = np.stack([body_values(ref.bodies,selected) for ref in references])
    height = np.abs(actual[:,0,2]-desired[:,0,2])
    gravity = np.abs(Rotation.from_quat(actual[:,0,3:7]).inv().apply((0,0,-1))[:,2]
                     -Rotation.from_quat(desired[:,0,3:7]).inv().apply((0,0,-1))[:,2])
    offset = 1+len(cfg['end_effector_bodies'])
    ee = np.max(np.abs(actual[:,1:offset,2]-desired[:,1:offset,2]),axis=1)
    ap,rp = actual[:,offset:,:3],desired[:,offset:,:3]
    ar = np.array([s.root_position_world_m for s in states])
    rr = np.array([r.root_position_world_m for r in references])
    aq = np.array([s.root_orientation_xyzw for s in states])
    rq = np.array([r.root_orientation_xyzw for r in references])
    mean_distance = lambda x,y: np.linalg.norm(x-y,axis=2).mean(axis=1)
    metrics = {
        'joint_mae_rad': np.abs(np.array([s.joint_position_rad for s in states])
                               -np.array([r.joint_position_rad for r in references])).mean(axis=1),
        'root_position_error_m': np.linalg.norm(ar-rr,axis=1),
        'root_orientation_error_rad': (Rotation.from_quat(rq).inv()*Rotation.from_quat(aq)).magnitude(),
        'mpjpe_world_m': mean_distance(ap,rp),
        'mpjpe_root_relative_world_axes_m': mean_distance(ap-ar[:,None,:],rp-rr[:,None,:]),
        'anchor_height_error_m': height,
        'anchor_gravity_z_error': gravity,
        'end_effector_height_error_m': ee,
    }
    alignment = protocol.get('position_alignment_metrics')
    if alignment is not None:
        if alignment != POSITION_ALIGNMENT_VERSION:
            raise ValueError('unsupported position alignment metric version')
        # Match the frozen scalar normalization and closest-heading convention.
        arot = Rotation.from_quat([normalize_quaternion(q) for q in aq])
        rrot = Rotation.from_quat([normalize_quaternion(q) for q in rq])
        relative = (rrot*arot.inv()).as_matrix()
        cosine = relative[:,0,0]+relative[:,1,1]
        sine = relative[:,1,0]-relative[:,0,1]
        degenerate = np.array([math.hypot(c,s)<=1e-12 for c,s in zip(cosine,sine)])
        angle = np.array([0. if d else math.atan2(s,c) for d,s,c in zip(degenerate,sine,cosine)])
        rotation_vector = np.zeros((len(states),3));rotation_vector[:,2] = angle
        yaw = Rotation.from_rotvec(rotation_vector).as_matrix()
        ax,rx = ap.copy(),rp.copy()
        ax[:,:,:2] -= ar[:,None,:2];rx[:,:,:2] -= rr[:,None,:2]
        aligned = np.einsum('nij,nkj->nki',yaw,ax)
        metrics.update(mpjpe_xy_aligned_m=mean_distance(ax,rx),
            mpjpe_xy_yaw_aligned_m=mean_distance(aligned,rx),
            root_xy_error_m=np.linalg.norm(ar[:,:2]-rr[:,:2],axis=1),
            root_height_error_m=np.abs(ar[:,2]-rr[:,2]),
            root_yaw_correction_abs_rad=np.abs(angle),
            yaw_alignment_degenerate_fraction=degenerate.astype(float))
    failures = [[name for name,value,threshold in (
        ('anchor_height',height[i],cfg['anchor_height_m']),
        ('anchor_gravity_z',gravity[i],cfg['gravity_z']),
        ('end_effector_height',ee[i],cfg['end_effector_height_m'])) if value>threshold] for i in range(len(states))]
    return [(failure,{key:float(value[i]) for key,value in metrics.items()}) for i,failure in enumerate(failures)]
