"""Reusable inference bridge for declarative observation providers.

This module has no dataset reader, simulator step, termination or score access.
An external model supplies observation classes and its frozen inference config.
"""
import copy
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation
import yaml

from motion_challenge.sdk import ControlCommand
from motion_challenge.validation import sha256_file


def named_values(value, names, default=0.0):
    if isinstance(value, dict):
        result = np.full(len(names), default, dtype=np.float64)
        for i, name in enumerate(names):
            matches = [v for k, v in value.items() if re.fullmatch(k, name)]
            if len(matches) > 1:
                raise ValueError(f'ambiguous joint configuration: {name}')
            if matches:
                result[i] = matches[0]
        return result
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 0:
        return np.full(len(names), float(value))
    if value.shape != (len(names),):
        raise ValueError('joint configuration shape mismatch')
    return value.copy()


class ConfiguredONNX:
    def __init__(self, root, runtime, observation_class):
        self.root, self.runtime = Path(root), runtime
        self.cfg = yaml.safe_load((self.root/runtime['inference_config']).read_text())
        self.observation_class = observation_class
        self.model_path = (self.root/runtime['weights']).resolve()
        if sha256_file(self.model_path) != runtime['weights_sha256']:
            raise ValueError('checkpoint hash mismatch')
        if sha256_file(self.root/runtime['inference_config']) != runtime['inference_config_sha256']:
            raise ValueError('inference config hash mismatch')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        meta = runtime.get('io_config')
        self.meta = json.loads((self.root/meta).read_text()) if meta else {}
        self.input_keys = self.meta.get('in_keys', [x.name for x in self.session.get_inputs()])
        self.output_keys = self.meta.get('out_keys', [x.name for x in self.session.get_outputs()])
        if len(self.input_keys) != len(self.session.get_inputs()) or len(self.output_keys) != len(self.session.get_outputs()):
            raise ValueError('ONNX metadata I/O count mismatch')
        self.env = None
        self.groups = {}
        self.cache = {}

    def _bind_state(self, state):
        env = self.env
        indices = [state.joint_names.index(n) for n in env.joint_names_simulation]
        xyzw = np.asarray(state.root_orientation_xyzw)
        env.state_processor.joint_pos = np.asarray(state.joint_position_rad)[indices]
        env.state_processor.joint_vel = np.asarray(state.joint_velocity_rad_s)[indices]
        env.state_processor.root_pos_w = np.asarray(state.root_position_world_m)
        env.state_processor.root_quat_w = xyzw[[3, 0, 1, 2]]
        env.state_processor.root_lin_vel_w = np.asarray(state.root_linear_velocity_world_m_s)
        env.state_processor.root_ang_vel_b = Rotation.from_quat(xyzw).inv().apply(state.root_angular_velocity_world_rad_s)
        env.state_processor.qpos = np.r_[env.state_processor.root_pos_w, env.state_processor.root_quat_w,
                                        env.state_processor.joint_pos]
        env.state_processor.qvel = np.r_[env.state_processor.root_lin_vel_w, env.state_processor.root_ang_vel_b,
                                        env.state_processor.joint_vel]
        env.motion_t = np.array([int(round(state.time_s/self.runtime['reference_step_s']))])

    def _body(self, reference, name):
        bodies = {b.name: b for b in reference.bodies}
        if name in bodies:
            b = bodies[name]
            return (np.asarray(b.position_world_m), np.asarray(b.orientation_xyzw),
                    np.asarray(b.linear_velocity_world_m_s), np.asarray(b.angular_velocity_world_rad_s))
        # Model-declared virtual keypoints are rigid transforms of canonical
        # bodies, not new dynamics or edited benchmark reference trajectories.
        transform = self.runtime.get('virtual_bodies', {}).get(name)
        if transform is None:
            raise ValueError(f'model requests unavailable body {name}')
        b = bodies[transform['parent']]
        q = np.asarray(b.orientation_xyzw)
        delta = Rotation.from_quat(q).apply(transform['position_parent_m'])
        w = np.asarray(b.angular_velocity_world_rad_s)
        return (np.asarray(b.position_world_m)+delta, q,
                np.asarray(b.linear_velocity_world_m_s)+np.cross(w, delta), w)

    def _bind_reference(self, state, reference):
        env = self.env
        ji = [state.joint_names.index(n) for n in env.motion_joint_names]
        body_rows = [[self._body(f, n) for n in env.motion_body_names] for f in reference]
        positions = np.array([[b[0] for b in row] for row in body_rows])[None]
        quats = np.array([[b[1] for b in row] for row in body_rows])[None][..., [3,0,1,2]]
        env.motion_data = SimpleNamespace(
            joint_pos=np.array([f.joint_position_rad for f in reference])[None][:,:,ji],
            joint_vel=np.array([f.joint_velocity_rad_s for f in reference])[None][:,:,ji],
            body_pos_w=positions, body_quat_w=quats,
            body_lin_vel_w=np.array([[b[2] for b in row] for row in body_rows])[None],
            body_ang_vel_w=np.array([[b[3] for b in row] for row in body_rows])[None],
            step=np.array([[int(round(f.source_time_s/self.runtime['reference_step_s'])) for f in reference]]),
            timestamps_ns=np.array([[round(f.source_time_s*1e9) for f in reference]], dtype=np.int64))

    def reset(self, state, reference, seed):
        # Reconstruct all observation objects: shared caches on their context are
        # discarded as well as explicit histories/recurrent tensors.
        cfg = self.cfg
        names = list(cfg['joint_names_simulation'])
        policy_names = list(cfg['policy_joint_names'])
        if set(names) != set(state.joint_names) or set(policy_names) != set(state.joint_names):
            raise ValueError('model joint names differ from canonical state')
        motion = {k:v for k,v in cfg['motion'].items() if k in (
            'future_steps','joint_names','body_names','root_body_name','anchor_body_name')}
        motion['future_steps'] = self.runtime['reference_steps']
        default = named_values(cfg['default_joint_pos'], names)
        self.env = SimpleNamespace(
            joint_names_simulation=names, body_names_simulation=list(cfg['body_names_simulation']),
            policy_joint_names=policy_names, num_dofs=len(names), num_actions=len(policy_names),
            default_dof_angles=default, controlled_joint_indices=[names.index(n) for n in policy_names],
            motion_config=motion, motion_future_steps=np.array(motion['future_steps']),
            motion_joint_names=names, motion_body_names=list(motion['body_names']),
            motion_ids=np.array([0]), motion_t=np.array([0]), total_inference_cnt=0,
            rl_dt=self.runtime['control_period_s'],
            state_processor=SimpleNamespace(joint_names=names),
            state_dict={'action':np.zeros(len(policy_names),dtype=np.float32),'paused':False,'control_mode':'policy'},
            policy_config=cfg)
        if 'reference_kinematics_asset' in self.runtime:
            asset = (self.root/self.runtime['reference_kinematics_asset']).resolve()
            self.env.robot_cfg = SimpleNamespace(resolve_mjcf_path=lambda: asset)
        self._bind_state(state)
        self._bind_reference(state, reference)
        self.groups = {}
        for group, entries in cfg['observation'].items():
            self.groups[group] = []
            for name, params in entries.items():
                params = copy.deepcopy(params)
                target = params.pop('_target_', name)
                obs = self.observation_class.resolve(target)(env=self.env, **params)
                self.groups[group].append(obs)
        for entries in self.groups.values():
            for obs in entries:
                obs.reset()
        self.cache = {}

    def observations(self, state, reference):
        self._bind_state(state)
        self._bind_reference(state, reference)
        env = self.env
        env.total_inference_cnt += 1
        for entries in self.groups.values():
            for obs in entries:
                obs.update(env.state_dict)
        result = {}
        for group, entries in self.groups.items():
            values = [np.asarray(obs.compute()) for obs in entries]
            result[group] = np.concatenate(values, axis=-1)
        # Observation providers may explicitly clear a recurrent state (e.g.
        # at a finite positional-encoding boundary). Never override that reset
        # with the previous network output.
        for key, value in self.cache.items():
            result.setdefault(key, value)
        return result

    def infer(self, inputs):
        feed = {}
        dtypes = {'tensor(float)':np.float32, 'tensor(double)':np.float64,
                  'tensor(int64)':np.int64, 'tensor(int32)':np.int32, 'tensor(bool)':np.bool_}
        for node, key in zip(self.session.get_inputs(), self.input_keys):
            key = tuple(key) if isinstance(key, list) else key
            x = np.asarray(inputs[key], dtype=dtypes[node.type])
            shape = tuple(n if isinstance(n,int) else 1 for n in node.shape)
            if x.shape != shape:
                # Only leading singleton batch axes may be inserted or removed;
                # a transpose/flatten is a model-specific observation operation.
                a, b = x.shape, shape
                while a and a[0] == 1: a = a[1:]
                while b and b[0] == 1: b = b[1:]
                if a != b:
                    raise ValueError(f'input {node.name} shape {x.shape}, expected {shape}')
                x = x.reshape(shape)
            feed[node.name] = x
        outputs = self.session.run(None, feed)
        result = {}
        for key, value in zip(self.output_keys, outputs):
            if isinstance(key,list) and len(key)==2 and key[0]=='next':
                self.cache[key[1]] = value
                self.env.state_dict[key[1]] = value
            else:
                result[key] = value
        return result

    def act(self, state, reference):
        result = self.infer(self.observations(state, reference))
        action = np.asarray(result[self.runtime.get('action_output','action')]).reshape(-1)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError('invalid model action')
        clip = self.cfg.get('action_clip', self.cfg.get('clip_actions'))
        if 'action_clip' in self.cfg and 'clip_actions' in self.cfg and self.cfg['action_clip'] != self.cfg['clip_actions']:
            raise ValueError('conflicting action clip configuration')
        if clip is not None:
            action = np.clip(action, -clip, clip)
        env = self.env
        env.state_dict['action'] = action.astype(np.float32)
        target = env.default_dof_angles.copy()
        target[env.controlled_joint_indices] += action * named_values(self.cfg['action_scale'], env.policy_joint_names, 1)
        order = [env.joint_names_simulation.index(n) for n in state.joint_names]
        return ControlCommand(tuple(float(x) for x in target[order]), (0.,)*29,
            tuple(named_values(self.cfg['joint_kp'], state.joint_names)),
            tuple(named_values(self.cfg['joint_kd'], state.joint_names)), (0.,)*29)

    def close(self):
        self.session = None
