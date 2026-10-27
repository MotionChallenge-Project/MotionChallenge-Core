"""Vectorized scoring; the scalar verifier remains an independent implementation."""
import numpy as np

from motion_challenge.data.bodies import body_values
from motion_challenge.data.array_reference import normalized
from motion_challenge.evaluation.alignment import POSITION_ALIGNMENT_VERSION


def failure_names(protocol):
    names = ('anchor_height', 'anchor_gravity_z')
    return names + (('end_effector_height',) if protocol['failure'].get('end_effector_bodies') else ())


def tracking_batch(states, references, protocol):
    if len(states) != len(references) or not states:
        raise ValueError('scoring requires matching nonempty batches')
    cfg = protocol['failure']
    names = protocol['metric_bodies']
    selected = (cfg['anchor_body'], *cfg.get('end_effector_bodies', ()), *names)
    actual = np.stack([body_values(state.bodies,selected) for state in states])
    desired = np.stack([body_values(ref.bodies,selected) for ref in references])
    failure,metrics = tracking_arrays(actual,desired,
        np.array([s.joint_position_rad for s in states]),np.array([r.joint_position_rad for r in references]),
        np.array([s.root_position_world_m for s in states]),np.array([r.root_position_world_m for r in references]),
        np.array([s.root_orientation_xyzw for s in states]),np.array([r.root_orientation_xyzw for r in references]),protocol)
    reasons = failure_names(protocol)
    return [([name for name,failed in zip(reasons,row) if failed],
             {key:float(value[i]) for key,value in metrics.items()}) for i,row in enumerate(failure)]


def relative_quaternion(reference, actual):
    """reference * inverse(actual), XYZW; arrays normalized once at the boundary."""
    x,y,z,w = reference.T
    a,b,c,d = actual.T
    return normalized(np.column_stack((w*(-a)+x*d-y*c+z*b,
        w*(-b)+y*d-z*a+x*c, w*(-c)+z*d-x*b+y*a, w*d+x*a+y*b+z*c)))


def gravity_z(quaternion):
    x,y,z,w = normalized(quaternion).T
    return -((-x*x-y*y)+z*z+w*w)


def tracking_arrays(actual, desired, joints, ref_joints, ar, rr, aq, rq, protocol):
    cfg = protocol['failure']
    height = np.abs(actual[:,0,2]-desired[:,0,2])
    gravity = np.abs(gravity_z(actual[:,0,3:7])-gravity_z(desired[:,0,3:7]))
    relative = relative_quaternion(normalized(rq),normalized(aq))
    offset = 1+len(cfg.get('end_effector_bodies', ()))
    ap,rp = actual[:,offset:,:3],desired[:,offset:,:3]
    mean_distance = lambda x,y: np.linalg.norm(x-y,axis=2).mean(axis=1)
    metrics = {
        'joint_mae_rad': np.abs(joints-ref_joints).mean(axis=1),
        'root_position_error_m': np.linalg.norm(ar-rr,axis=1),
        'root_orientation_error_rad': 2*np.arctan2(np.linalg.norm(relative[:,:3],axis=1),np.abs(relative[:,3])),
        'mpjpe_world_m': mean_distance(ap,rp),
        'mpjpe_root_relative_world_axes_m': mean_distance(ap-ar[:,None,:],rp-rr[:,None,:]),
        'anchor_height_error_m': height,
        'anchor_gravity_z_error': gravity,
    }
    failures = [height>cfg['anchor_height_m'], gravity>cfg['gravity_z']]
    if offset > 1:
        ee = np.max(np.abs(actual[:,1:offset,2]-desired[:,1:offset,2]),axis=1)
        metrics['end_effector_height_error_m'] = ee
        failures.append(ee>cfg['end_effector_height_m'])
    alignment = protocol.get('position_alignment_metrics')
    if alignment is not None:
        if alignment != POSITION_ALIGNMENT_VERSION:
            raise ValueError('unsupported position alignment metric version')
        # Match the frozen scalar normalization and closest-heading convention.
        x,y,z,w = relative.T
        xx,yy,zz,ww = x*x,y*y,z*z,w*w
        cosine = (xx-yy-zz+ww)+(-xx+yy-zz+ww)
        sine = 4*z*w
        degenerate = np.hypot(cosine,sine)<=1e-12
        angle = np.where(degenerate,0.,np.arctan2(sine,cosine))
        ax,rx = ap.copy(),rp.copy()
        ax[:,:,:2] -= ar[:,None,:2];rx[:,:,:2] -= rr[:,None,:2]
        c,s = np.cos(angle)[:,None],np.sin(angle)[:,None]
        aligned = ax.copy()
        aligned[:,:,0] = c*ax[:,:,0]-s*ax[:,:,1]
        aligned[:,:,1] = s*ax[:,:,0]+c*ax[:,:,1]
        metrics.update(mpjpe_xy_aligned_m=mean_distance(ax,rx),
            mpjpe_xy_yaw_aligned_m=mean_distance(aligned,rx),
            root_xy_error_m=np.linalg.norm(ar[:,:2]-rr[:,:2],axis=1),
            root_height_error_m=np.abs(ar[:,2]-rr[:,2]),
            root_yaw_correction_abs_rad=np.abs(angle),
            yaw_alignment_degenerate_fraction=degenerate.astype(float))
    return np.column_stack(failures),metrics
