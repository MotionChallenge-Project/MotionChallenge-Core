"""Canonical CPU MuJoCo backend. No controller or dataset names belong here."""
from collections import OrderedDict
from dataclasses import replace
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from motion_challenge.sdk import StatePacket
from motion_challenge.data.bodies import BodyArray
from motion_challenge.validation import sha256_file


def forward_kinematics(model, data):
    """Update poses and body-origin Jacobians without solving dynamics.

    MuJoCo documents kinematics + comPos as the required stages for mj_jacBody.
    Use this only for read-only reference/state reconstruction, never stepping.
    """
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def command_values(command, size):
    values = [np.asarray(getattr(command, n), dtype=np.float64) for n in (
        'joint_position_target_rad', 'joint_velocity_target_rad_s',
        'kp_nm_rad', 'kd_nm_s_rad', 'feedforward_torque_nm')]
    if any(v.shape != (size,) or not np.isfinite(v).all() for v in values):
        raise ValueError('controller returned invalid PD command')
    q, qd, kp, kd, ff = values
    if (kp < 0).any() or (kd < 0).any():
        raise ValueError('negative gains')
    return values


class World:
    def __init__(self, profile_path):
        path = Path(profile_path)
        self.profile = json.loads(path.read_text())
        xml = path.parent / self.profile['xml']
        if sha256_file(xml) != self.profile['xml_sha256']:
            raise ValueError('canonical robot XML hash mismatch')
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.reference_data = mujoco.MjData(self.model)
        self.joint_names = tuple(self.profile['joint_names'])
        self.joints = np.array([self.model.joint(n).id for n in self.joint_names])
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.limits = np.asarray(self.profile['torque_limits_nm'])
        if self.model.nu != len(self.joint_names):
            raise ValueError('canonical actuator count mismatch')
        for i, name in enumerate(self.joint_names):
            if self.model.actuator_trnid[i, 0] != self.model.joint(name).id:
                raise ValueError('canonical actuator order mismatch')
        self.body_ids = tuple(range(1, self.model.nbody))
        self.body_indices = np.array(self.body_ids)
        self.body_names = tuple(self.model.body(i).name for i in self.body_ids)
        self.body_roots = self.model.body_rootid[self.body_indices].copy()
        self.root_body_id = self.model.body(self.profile['root_body']).id
        self.root_body_row = self.body_ids.index(self.root_body_id)
        self._reference_cache = OrderedDict()
        self.motion = None
        self.reference_work = dict(frames=0,interpolation_s=0.,differentiate_s=0.,
            fk_native_s=0.,body_velocity_native_s=0.,body_velocity_arrays_s=0.,body_storage_s=0.)
        self.trace_work = dict(samples=0,metric_accumulation_s=0.,json_encoding_s=0.,buffer_submit_s=0.)

    def _qpos(self, frame):
        q = self.model.qpos0.copy()
        q[:3] = frame.root_position_world_m
        xyzw = frame.root_orientation_xyzw
        q[3:7] = (xyzw[3], *xyzw[:3])
        q[self.qadr] = frame.joint_position_rad
        return q

    def _bodies(self, data):
        # comVel computes all spatial velocities once. Shift their origins from
        # subtree COMs to body origins; angular/linear axes remain world axes.
        # This is the mj_objectVelocity(mjOBJ_XBODY, local=0) convention.
        started = time.perf_counter()
        mujoco.mj_comVel(self.model, data)
        self.reference_work['body_velocity_native_s'] += time.perf_counter()-started
        started = time.perf_counter()
        bids = self.body_indices
        velocity = data.cvel[bids].copy()
        position = data.xpos[bids]
        velocity[:, 3:] += np.cross(velocity[:, :3], position-data.subtree_com[self.body_roots])
        # Keep the published root-measurement arithmetic identical to the
        # simulation reader. Other body velocities use the equivalent spatial shift.
        self.reference_work['body_velocity_arrays_s'] += time.perf_counter()-started
        started = time.perf_counter()
        jp, jr = np.empty((3, self.model.nv)), np.empty((3, self.model.nv))
        mujoco.mj_jacBody(self.model, data, jp, jr, self.root_body_id)
        velocity[self.root_body_row, 3:] = jp @ data.qvel
        velocity[self.root_body_row, :3] = jr @ data.qvel
        self.reference_work['body_velocity_native_s'] += time.perf_counter()-started
        started = time.perf_counter()
        orientation = data.xquat[bids][:, [1, 2, 3, 0]]
        result = BodyArray(self.body_names, np.concatenate(
            (position, orientation, velocity[:,3:], velocity[:,:3]), axis=1))
        self.reference_work['body_storage_s'] += time.perf_counter()-started
        return result

    def reference(self, requested_time_s):
        key = round(float(requested_time_s), 12)
        if key in self._reference_cache:
            self._reference_cache.move_to_end(key)
            return self._reference_cache[key]
        started = time.perf_counter()
        f = self.motion.sample(key)
        t = f.source_time_s
        # Centered finite difference in generalized coordinates over one source
        # interval on either side; one-sided at endpoints. No clip time warp.
        h = self.motion.times_s[1] - self.motion.times_s[0]
        lo, hi = max(0.0, t-h), min(self.motion.duration_s, t+h)
        qa = self._qpos(self.motion.sample(lo))
        qb = self._qpos(self.motion.sample(hi))
        d = self.reference_data
        d.qpos[:] = self._qpos(f)
        self.reference_work['interpolation_s'] += time.perf_counter()-started
        started = time.perf_counter()
        mujoco.mj_differentiatePos(self.model, d.qvel, hi-lo, qa, qb)
        self.reference_work['differentiate_s'] += time.perf_counter()-started
        started = time.perf_counter()
        forward_kinematics(self.model, d)
        self.reference_work['fk_native_s'] += time.perf_counter()-started
        self.reference_work['frames'] += 1
        bodies = self._bodies(d)
        root = bodies[self.root_body_row]
        f = replace(f, joint_velocity_rad_s=tuple(d.qvel[self.vadr]),
                    root_linear_velocity_world_m_s=root.linear_velocity_world_m_s,
                    root_angular_velocity_world_rad_s=root.angular_velocity_world_rad_s,
                    bodies=bodies)
        self._reference_cache[key] = f
        if len(self._reference_cache) > getattr(self, 'reference_cache_capacity', 2048):
            self._reference_cache.popitem(last=False)
        return f

    def reset(self, motion):
        if motion.joint_names != self.joint_names:
            raise ValueError('motion joint order mismatch')
        self.motion = motion
        self._reference_cache.clear()
        f = self.reference(0.0)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._qpos(f)
        self.data.qvel[:] = self.reference_data.qvel
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        d = self.data
        bodies = self._bodies(d)
        root = bodies[self.root_body_row]
        return StatePacket(float(d.time), self.joint_names,
            tuple(d.qpos[self.qadr]), tuple(d.qvel[self.vadr]),
            root.position_world_m, root.orientation_xyzw,
            root.linear_velocity_world_m_s, root.angular_velocity_world_rad_s, bodies)

    def advance(self, command, until_s):
        q, qd, kp, kd, ff = command_values(command, len(self.joints))
        dt = self.profile['physics_dt_s']
        try:
            while self.data.time < until_s - 1e-10:
                self.model.opt.timestep = min(dt, until_s - self.data.time)
                # PD and actuator saturation happen on every physics step.
                torque = kp*(q-self.data.qpos[self.qadr]) + kd*(qd-self.data.qvel[self.vadr]) + ff
                self.data.ctrl[:] = np.clip(torque, -self.limits, self.limits)
                expected_time = self.data.time + self.model.opt.timestep
                mujoco.mj_step(self.model, self.data)
                if abs(self.data.time - expected_time) > 1e-10:
                    raise FloatingPointError('MuJoCo reset or discontinuous simulation time')
                if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                    raise FloatingPointError('non-finite physics state')
        finally:
            self.model.opt.timestep = dt
        # mj_step's derived quantities may describe the pre-integration state.
        mujoco.mj_forward(self.model, self.data)


def tracking_sample(state, reference, protocol):
    from motion_challenge.evaluation.alignment import POSITION_ALIGNMENT_VERSION, keybody_position_errors

    actual = {b.name: b for b in state.bodies}
    desired = {b.name: b for b in reference.bodies}
    cfg = protocol['failure']
    anchor = cfg['anchor_body']
    a, r = actual[anchor], desired[anchor]
    height_error = abs(a.position_world_m[2] - r.position_world_m[2])
    ga = Rotation.from_quat(a.orientation_xyzw).inv().apply((0, 0, -1))[2]
    gr = Rotation.from_quat(r.orientation_xyzw).inv().apply((0, 0, -1))[2]
    gravity_z_error = abs(ga-gr)
    ee_error = max(abs(actual[n].position_world_m[2]-desired[n].position_world_m[2])
                   for n in cfg['end_effector_bodies'])
    failures = []
    for name, value, threshold in (
        ('anchor_height', height_error, cfg['anchor_height_m']),
        ('anchor_gravity_z', gravity_z_error, cfg['gravity_z']),
        ('end_effector_height', ee_error, cfg['end_effector_height_m'])):
        if value > threshold:
            failures.append(name)
    names = protocol['metric_bodies']
    ap = np.array([actual[n].position_world_m for n in names])
    rp = np.array([desired[n].position_world_m for n in names])
    ar = np.array(state.root_position_world_m)
    rr = np.array(reference.root_position_world_m)
    rotation = Rotation.from_quat(reference.root_orientation_xyzw).inv() * Rotation.from_quat(state.root_orientation_xyzw)
    metrics = {
        'joint_mae_rad': float(np.abs(np.asarray(state.joint_position_rad)-reference.joint_position_rad).mean()),
        'root_position_error_m': float(np.linalg.norm(ar-rr)),
        'root_orientation_error_rad': float(rotation.magnitude()),
        'mpjpe_world_m': float(np.linalg.norm(ap-rp, axis=1).mean()),
        'mpjpe_root_relative_world_axes_m': float(np.linalg.norm((ap-ar)-(rp-rr), axis=1).mean()),
        'anchor_height_error_m': float(height_error),
        'anchor_gravity_z_error': float(gravity_z_error),
        'end_effector_height_error_m': float(ee_error),
    }
    alignment_version = protocol.get('position_alignment_metrics')
    if alignment_version is not None:
        if alignment_version != POSITION_ALIGNMENT_VERSION:
            raise ValueError('unsupported position alignment metric version')
        metrics.update(keybody_position_errors(ap, rp, ar, rr,
            state.root_orientation_xyzw, reference.root_orientation_xyzw))
    return failures, metrics
