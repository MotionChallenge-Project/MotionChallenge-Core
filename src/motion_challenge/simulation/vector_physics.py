"""One native batch and two shared FK scratch buffers, independent of slot count."""
from collections import OrderedDict
import copy
import os
import time

import mujoco
from mujoco import batch
import numpy as np

from motion_challenge.sdk import StatePacket
from motion_challenge.data.bodies import BodyArray
from motion_challenge.simulation.physics import World


def cpu_count():
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)


class VectorPhysics:
    def __init__(self, profile, size, threads=None, scoring_bodies=None):
        if size < 1 or (threads is not None and threads < 1):
            raise ValueError('slot and thread counts must be positive')
        self.native_timings = {}
        self.world = World(profile)
        self.batch = batch.Batch(self.world.model, nsim=size,
                                 nthread=threads or min(8,size,cpu_count()), persistent=True)
        self.qpos, self.qvel, self.ctrl, self.time = (self.batch.bind(n) for n in ('qpos','qvel','ctrl','time'))
        self.xpos, self.xquat = self.batch.bind('xpos'), self.batch.bind('xquat')
        self.dt = self.batch.expand('opt.timestep')
        self.root_id = self.world.model.body(self.world.profile['root_body']).id
        self.body_ids = (self.world.body_ids if scoring_bodies is None else tuple(sorted(
            {self.world.model.body(name).id for name in scoring_bodies} | {self.root_id})))
        self.body_names = tuple(self.world.model.body(i).name for i in self.body_ids)
        self.root_row = self.body_ids.index(self.root_id)

    def _native(self, name, *args, **kwargs):
        started = time.perf_counter()
        try:
            return getattr(self.batch,name)(*args,**kwargs)
        finally:
            item = self.native_timings.setdefault(name,dict(seconds=0.,calls=0))
            item['seconds'] += time.perf_counter()-started
            item['calls'] += 1

    def reference_world(self, motion, cache_size):
        # Reference objects have private motion/cache but share read-only model
        # and scratch FK data. Only this core process accesses the scratch data.
        world = copy.copy(self.world)
        world.motion = motion
        world._reference_cache = OrderedDict()
        world.reference_cache_capacity = cache_size
        return world

    def reset(self, ids, worlds):
        pairs = sorted(zip(ids, worlds))
        ids, worlds = [p[0] for p in pairs], [p[1] for p in pairs]
        self._native('reset',ids=ids, stage=mujoco.mjtStage.mjSTAGE_NONE)
        for slot,world in zip(ids,worlds):
            if world.motion.joint_names != world.joint_names:
                raise ValueError('motion joint order mismatch')
            frame = world.reference(0.)
            self.qpos[slot] = world._qpos(frame)
            self.qvel[slot] = world.reference_data.qvel
        self.dt[ids] = self.world.profile['physics_dt_s']
        self._native('forward',ids=ids)

    def advance(self, ids, commands, targets):
        world = self.world
        order = np.argsort(ids)
        ids = np.asarray(ids,dtype=np.int32)[order]
        targets = np.asarray(targets)[order]
        values = np.asarray([[c.joint_position_target_rad,c.joint_velocity_target_rad_s,
            c.kp_nm_rad,c.kd_nm_s_rad,c.feedforward_torque_nm] for c in commands],dtype=np.float64)
        if values.shape!=(len(ids),5,len(world.joints)) or not np.isfinite(values).all():
            raise ValueError('controller returned invalid PD commands')
        if (values[:,2:4]<0).any():
            raise ValueError('negative gains')
        values = values[order]
        while True:
            mask = self.time[ids] < targets-1e-10
            if not mask.any():
                break
            active = ids[mask]
            q,qd,kp,kd,ff = values[mask].transpose(1,0,2)
            self.dt[active] = np.minimum(world.profile['physics_dt_s'],targets[mask]-self.time[active])
            torque = kp*(q-self.qpos[np.ix_(active,world.qadr)]) + kd*(qd-self.qvel[np.ix_(active,world.vadr)]) + ff
            self.ctrl[active] = np.clip(torque,-world.limits,world.limits)
            expected = self.time[active].copy()+self.dt[active]
            self._native('step',ids=active,nstep=1)
            if not np.allclose(self.time[active],expected,rtol=0,atol=1e-10):
                raise FloatingPointError('MuJoCo reset or discontinuous batch simulation time')
            if not np.isfinite(self.qpos[active]).all() or not np.isfinite(self.qvel[active]).all():
                raise FloatingPointError('non-finite batch physics state')
        self.dt[ids] = world.profile['physics_dt_s']
        self._native('forward',ids=ids)

    def states(self, ids):
        # The adapter receives only the root body-origin velocities. Other body
        # poses are core-private scoring data; their unused velocities are absent.
        ordered = sorted(ids)
        jp,jr = self._native('jac',self.root_id,self.xpos[ordered,self.root_id],ids=ordered)
        rows = {slot: row for row,slot in enumerate(ordered)}
        states = []
        world = self.world
        # One numeric snapshot survives later native steps. No per-body Python
        # objects are needed by the vector scorer or array policy interface.
        poses = np.zeros((len(ids),len(self.body_ids),13))
        poses[:,:,:3] = self.xpos[np.ix_(ids,self.body_ids)]
        poses[:,:,3:7] = self.xquat[np.ix_(ids,self.body_ids)][:,:,[1,2,3,0]]
        for index,slot in enumerate(ids):
            row = rows[slot]
            linear,angular = tuple(jp[row] @ self.qvel[slot]),tuple(jr[row] @ self.qvel[slot])
            poses[index,self.root_row,7:10] = linear
            poses[index,self.root_row,10:13] = angular
            bodies = BodyArray(self.body_names,poses[index])
            root = bodies[self.root_row]
            states.append(StatePacket(float(self.time[slot]),world.joint_names,
                tuple(self.qpos[slot,world.qadr]),tuple(self.qvel[slot,world.vadr]),
                root.position_world_m,root.orientation_xyzw,linear,angular,bodies))
        return states
