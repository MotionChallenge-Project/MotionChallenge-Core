"""Bounded numeric reference cache with batched interpolation and native FK."""
import time

import mujoco
import numpy as np

from motion_challenge.data.array_reference import MotionArena, quaternion_velocity
from motion_challenge.sdk import ReferenceBatch


def reference_view(values, joints, body_names):
    j = joints
    bodies = values[...,15+2*j:].reshape(*values.shape[:-1],len(body_names),13)
    return ReferenceBatch(values[...,0],values[...,1],values[...,2:2+j],
        values[...,2+j:5+j],values[...,5+j:9+j],values[...,9+j:9+2*j],
        values[...,9+2*j:12+2*j],values[...,12+2*j:15+2*j],body_names,
        bodies[...,:3],bodies[...,3:7],bodies[...,7:10],bodies[...,10:13])


class ArrayReferences:
    def __init__(self, pool, slots, window):
        self.world,self.fk = pool.world,pool.fk
        self.arena = MotionArena(slots,len(self.world.joint_names))
        self.body_ids = self.world.body_ids
        self.body_names = self.world.body_names
        self.root_row = self.body_ids.index(self.world.root_body_id)
        self.width = 1<<(max(16,4*window)-1).bit_length()
        self.keys = np.full((slots,self.width),np.iinfo(np.int64).min,dtype=np.int64)
        self.columns = 15+2*len(self.world.joint_names)+13*len(self.body_ids)
        self.values = np.empty((slots,self.width,self.columns))
        self.reset_velocity = np.empty((slots,self.world.model.nv))
        self.reset_position = np.empty((slots,self.world.model.nq))

    def load(self, slot, motion):
        self.arena.load(slot,motion)
        self.keys[slot] = np.iinfo(np.int64).min
        # Loading is outside the control loop. Preserve the frozen native reset
        # velocity exactly: tiny initial perturbations can amplify through contact.
        lo,hi = motion.sample(0.),motion.sample(self.arena.h[slot])
        self.reset_position[slot] = self.world._qpos(lo)
        mujoco.mj_differentiatePos(self.world.model,self.reset_velocity[slot],self.arena.h[slot],
                                  self.reset_position[slot],self.world._qpos(hi))

    def initial_velocity(self, ids):
        return self.reset_velocity[ids]

    def get(self, ids, requested):
        requested = np.asarray(requested,dtype=float)
        ids = np.asarray(ids,dtype=np.int64)
        shape = requested.shape
        slots = np.broadcast_to(ids[:,None],shape).reshape(-1)
        # FK depends on source time, so all beyond-end preview frames share it.
        times = np.clip(np.round(requested.reshape(-1),12),0.,self.arena.duration[slots])
        keys = np.rint(times*1e12).astype(np.int64)
        bins = (keys^(keys>>17)^(keys>>37))&(self.width-1)
        hit = self.keys[slots,bins]==keys
        out = np.empty((len(slots),self.columns))
        out[hit] = self.values[slots[hit],bins[hit]]
        if not hit.all():
            unique,first,inverse = np.unique(np.column_stack((slots[~hit],keys[~hit])),axis=0,return_index=True,return_inverse=True)
            # Preserve the exact clamped endpoint rather than decoding a rounded key.
            data = self.compute(unique[:,0],times[~hit][first])
            out[~hit] = data[inverse]
            ub = (unique[:,1]^(unique[:,1]>>17)^(unique[:,1]>>37))&(self.width-1)
            flat = unique[:,0]*self.width+ub
            _,keep = np.unique(flat,return_index=True)
            self.keys[unique[keep,0],ub[keep]] = unique[keep,1]
            self.values[unique[keep,0],ub[keep]] = data[keep]
        out[:,0] = np.round(requested.reshape(-1),12)
        return reference_view(out.reshape(*shape,self.columns),len(self.world.joint_names),self.body_names)

    def compute(self, slots, times):
        w = self.world; n = len(times); j = len(w.joint_names)
        tick = time.perf_counter()
        lo = np.maximum(0.,times-self.arena.h[slots])
        hi = np.minimum(self.arena.duration[slots],times+self.arena.h[slots])
        _,joint,pos,quat = self.arena.sample(np.tile(slots,3),np.concatenate((times,lo,hi)))
        qpos = np.broadcast_to(w.model.qpos0,(n,w.model.nq)).copy()
        qpos[:,:3],qpos[:,3:7],qpos[:,w.qadr] = pos[:n],quat[:n][:,[3,0,1,2]],joint[:n]
        w.reference_work['interpolation_s'] += time.perf_counter()-tick
        tick = time.perf_counter()
        dt = hi-lo
        qvel = np.zeros((n,w.model.nv))
        qvel[:,:3] = (pos[2*n:]-pos[n:2*n])/dt[:,None]
        qvel[:,3:6] = quaternion_velocity(quat[n:2*n],quat[2*n:],dt)
        qvel[:,w.vadr] = (joint[2*n:]-joint[n:2*n])/dt[:,None]
        w.reference_work['differentiate_s'] += time.perf_counter()-tick
        before = self.fk.timings.copy()
        bodies = self.fk.compute(qpos,qvel,self.body_ids,w.body_velocities)
        for key,value in self.fk.timings.items():
            w.reference_work[key] += value-before[key]
        tick = time.perf_counter()
        values = np.empty((n,self.columns))
        values[:,0]=times; values[:,1]=times
        values[:,2:2+j]=joint[:n];values[:,2+j:5+j]=pos[:n];values[:,5+j:9+j]=quat[:n]
        values[:,9+j:9+2*j]=qvel[:,w.vadr]
        values[:,9+2*j:12+2*j]=bodies[:,self.root_row,7:10]
        values[:,12+2*j:15+2*j]=bodies[:,self.root_row,10:13]
        values[:,15+2*j:]=bodies.reshape(n,-1)
        w.reference_work['body_storage_s'] += time.perf_counter()-tick
        w.reference_work['frames'] += n
        return values
