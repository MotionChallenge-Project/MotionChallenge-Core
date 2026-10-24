"""Persistent native batch FK, separate from the advancing simulation."""
import copy
import time

import mujoco
from mujoco import batch
import numpy as np


class ArrayKinematics:
    def __init__(self, world, capacity, threads):
        self.model = copy.copy(world.model)
        # This scratch model never steps. Contacts are irrelevant to FK.
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.batch = batch.Batch(self.model,nsim=capacity,nthread=threads,persistent=True)
        self.qpos,self.qvel = self.batch.bind('qpos'),self.batch.bind('qvel')
        self.xpos,self.xquat = self.batch.bind('xpos'),self.batch.bind('xquat')
        self.cvel,self.com = self.batch.bind('cvel'),self.batch.bind('subtree_com')
        self.root = world.root_body_id
        self.timings = dict(fk_native_s=0.,body_velocity_native_s=0.,body_velocity_arrays_s=0.)

    def compute(self, qpos, qvel, bodies, velocities=True):
        bodies = np.asarray(bodies,dtype=int)
        result = np.zeros((len(qpos),len(bodies),13))
        root_row = int(np.flatnonzero(bodies==self.root)[0])
        roots = self.model.body_rootid[bodies]
        for start in range(0,len(qpos),self.batch.nsim):
            end = min(start+self.batch.nsim,len(qpos)); n = end-start
            ids = np.arange(n,dtype=np.int32)
            self.qpos[:n],self.qvel[:n] = qpos[start:end],qvel[start:end]
            tick = time.perf_counter()
            self.batch.forward(ids=ids,stage=mujoco.mjtStage.mjSTAGE_VEL if velocities else mujoco.mjtStage.mjSTAGE_POS)
            self.timings['fk_native_s'] += time.perf_counter()-tick
            out = result[start:end]
            out[:,:,:3] = self.xpos[:n,bodies]
            out[:,:,3:7] = self.xquat[:n,bodies][:,:,[1,2,3,0]]
            if velocities:
                tick = time.perf_counter()
                spatial = self.cvel[:n,bodies]
                out[:,:,7:10] = spatial[:,:,3:]+np.cross(spatial[:,:,:3],out[:,:,:3]-self.com[:n,roots])
                out[:,:,10:13] = spatial[:,:,:3]
                self.timings['body_velocity_arrays_s'] += time.perf_counter()-tick
            tick = time.perf_counter()
            jp,jr = self.batch.jac(self.root,out[:,root_row,:3],ids=ids)
            v = qvel[start:end,:,None]
            out[:,root_row,7:10] = np.matmul(jp,v)[:,:,0]
            out[:,root_row,10:13] = np.matmul(jr,v)[:,:,0]
            self.timings['body_velocity_native_s'] += time.perf_counter()-tick
        return result
