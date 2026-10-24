"""One native batch and two shared FK scratch buffers, independent of slot count."""
from collections import OrderedDict
import copy
import os
import time

import mujoco
from mujoco import batch
import numpy as np

from motion_challenge.sdk import StatePacket, StateBatch
from motion_challenge.data.bodies import BodyArray
from motion_challenge.simulation.physics import World
from motion_challenge.simulation.array_kinematics import ArrayKinematics


def cpu_count():
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)


def default_threads(size):
    return min(size,cpu_count(),max(8,min(32,(size+127)//128)))


class VectorPhysics:
    def __init__(self, profile, size, threads=None, scoring_bodies=None, reference_bodies=None):
        if size < 1 or (threads is not None and threads < 1):
            raise ValueError('slot and thread counts must be positive')
        self.native_timings = {}
        self.world = World(profile)
        self.batch = batch.Batch(self.world.model, nsim=size,
                                 nthread=threads or default_threads(size), persistent=True)
        self.qpos, self.qvel, self.ctrl, self.time = (self.batch.bind(n) for n in ('qpos','qvel','ctrl','time'))
        # Bind only integration state. Binding xpos/xquat would copy all bodies
        # after every native substep, although observations need them only at ticks.
        self.dt = self.batch.expand('opt.timestep')
        self.root_id = self.world.model.body(self.world.profile['root_body']).id
        self.body_ids = (self.world.body_ids if scoring_bodies is None else tuple(sorted(
            {self.world.model.body(name).id for name in scoring_bodies} | {self.root_id})))
        self.body_names = tuple(self.world.model.body(i).name for i in self.body_ids)
        self.root_row = self.body_ids.index(self.root_id)
        if reference_bodies is not None:
            requested = tuple(reference_bodies)
            if len(set(requested)) != len(requested) or not set(requested) <= set(self.world.body_names):
                raise ValueError('invalid reference body names')
            selected = set(requested) | set(scoring_bodies or ()) | {self.world.profile['root_body']}
            w = self.world
            w.body_ids = tuple(i for i in w.body_ids if w.model.body(i).name in selected)
            w.body_indices = np.array(w.body_ids)
            w.body_names = tuple(w.model.body(i).name for i in w.body_ids)
            w.body_roots = w.model.body_rootid[w.body_indices].copy()
            w.root_body_row = w.body_ids.index(w.root_body_id)
            w.body_velocities = bool(requested)
        self.fk = ArrayKinematics(self.world,size,self.batch.nthread)

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
        # The first native step computes the dynamics stages it needs.

    def reset_arrays(self, ids, qpos, qvel):
        ids = np.asarray(ids,dtype=np.int32)
        self._native('reset',ids=ids,stage=mujoco.mjtStage.mjSTAGE_NONE)
        self.qpos[ids],self.qvel[ids] = qpos,qvel
        self.dt[ids] = self.world.profile['physics_dt_s']

    def advance(self, ids, commands, targets):
        world = self.world
        order = np.argsort(ids)
        ids = np.asarray(ids,dtype=np.int32)[order]
        targets = np.asarray(targets)[order]
        # Core-owned numeric commands already checked once at the policy boundary.
        values = np.asarray(commands,dtype=np.float64)[order]
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
            self._native('step',ids=active,nstep=1,stage=mujoco.mjtStage.mjSTAGE_NONE)
            if not np.allclose(self.time[active],expected,rtol=0,atol=1e-10):
                raise FloatingPointError('MuJoCo reset or discontinuous batch simulation time')
            if not np.isfinite(self.qpos[active]).all() or not np.isfinite(self.qvel[active]).all():
                raise FloatingPointError('non-finite batch physics state')
        self.dt[ids] = world.profile['physics_dt_s']

    def read_arrays(self, ids):
        ids = np.asarray(ids,dtype=np.int32)
        qpos,qvel = self.qpos[ids],self.qvel[ids]
        poses = self.fk.compute(qpos,qvel,self.body_ids,False)
        root = poses[:,self.root_row]
        states = StateBatch(self.world.joint_names,self.time[ids].copy(),
            qpos[:,self.world.qadr],qvel[:,self.world.vadr],root[:,3:7],root[:,7:10],root[:,10:13])
        return states,root[:,:3],poses

    def states(self, ids):
        # The adapter receives only the root body-origin velocities. Other body
        # poses are core-private scoring data; their unused velocities are absent.
        batch,positions,poses = self.read_arrays(ids)
        states = []
        world = self.world
        # One numeric snapshot survives later native steps. No per-body Python
        # objects are needed by the vector scorer or array policy interface.
        for index,slot in enumerate(ids):
            linear,angular = tuple(batch.root_linear_velocity_world_m_s[index]),tuple(batch.root_angular_velocity_world_rad_s[index])
            bodies = BodyArray(self.body_names,poses[index])
            root = poses[index,self.root_row]
            states.append(StatePacket(float(self.time[slot]),world.joint_names,
                tuple(self.qpos[slot,world.qadr]),tuple(self.qvel[slot,world.vadr]),
                tuple(root[:3]),tuple(root[3:7]),linear,angular,bodies))
        return states
