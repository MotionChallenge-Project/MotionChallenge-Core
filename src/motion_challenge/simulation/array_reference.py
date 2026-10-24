"""Bounded numeric reference cache with batched interpolation and native FK."""
import math
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
    def __init__(self, pool, slots, window, clocks=(), offsets=()):
        self.world,self.fk = pool.world,pool.fk
        self.arena = MotionArena(slots,len(self.world.joint_names))
        self.body_ids = self.world.body_ids
        self.body_names = self.world.body_names
        self.root_row = self.body_ids.index(self.world.root_body_id)
        self.width = 1<<(max(16,4*window)-1).bit_length()
        # Dense control/metric clocks use a collision-free ring over the preview
        # span. Irregular clocks retain a bounded hash cache; no full-clip table.
        ticks = [abs(round(float(t)*1e12)) for t in (*clocks,*offsets) if t]
        quantum = math.gcd(*ticks) if ticks else 0
        span = round((max((0.,*offsets))-min((0.,*offsets)))*1e12)
        required = span//quantum+2 if quantum else 0
        self.quantum = quantum if quantum and required<=512 else 0
        if self.quantum:
            self.width = max(self.width,1<<(required-1).bit_length())
        # The exact clamped endpoint has its own entry, separate from the ring.
        self.keys = np.full((slots,self.width+1),np.iinfo(np.int64).min,dtype=np.int64)
        self.columns = 15+2*len(self.world.joint_names)+13*len(self.body_ids)
        self.values = np.empty((slots,self.width+1,self.columns))
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

    def bins(self, slots, keys, times):
        index = keys//self.quantum if self.quantum else keys^(keys>>17)^(keys>>37)
        return np.where(times==self.arena.duration[slots],self.width,index&(self.width-1))

    def get(self, ids, requested):
        requested = np.asarray(requested,dtype=float)
        ids = np.asarray(ids,dtype=np.int64)
        shape = requested.shape
        slots = np.broadcast_to(ids[:,None],shape).reshape(-1)
        # FK depends on source time, so all beyond-end preview frames share it.
        times = np.clip(np.round(requested.reshape(-1),12),0.,self.arena.duration[slots])
        keys = np.rint(times*1e12).astype(np.int64)
        bins = self.bins(slots,keys,times)
        hit = self.keys[slots,bins]==keys
        out = np.empty((len(slots),self.columns))
        out[hit] = self.values[slots[hit],bins[hit]]
        if not hit.all():
            ms,mk,mt = slots[~hit],keys[~hit],times[~hit]
            order = np.lexsort((mk,ms))
            distinct = np.r_[True,(ms[order[1:]]!=ms[order[:-1]])|(mk[order[1:]]!=mk[order[:-1]])]
            first = order[distinct]
            inverse = np.empty(len(order),dtype=np.int64)
            inverse[order] = np.cumsum(distinct)-1
            # Preserve the exact clamped endpoint rather than decoding a rounded key.
            data = self.compute(ms[first],mt[first])
            out[~hit] = data[inverse]
            ub = self.bins(ms[first],mk[first],mt[first])
            flat = ms[first]*(self.width+1)+ub
            _,keep = np.unique(flat,return_index=True)
            self.keys[ms[first[keep]],ub[keep]] = mk[first[keep]]
            self.values[ms[first[keep]],ub[keep]] = data[keep]
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
