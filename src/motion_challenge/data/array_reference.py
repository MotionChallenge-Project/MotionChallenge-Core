"""Packed active motions: interpolation is one array operation across slots."""
from collections import defaultdict

import numpy as np


def normalized(q):
    squared = q*q
    norm = np.sqrt(((squared[...,0]+squared[...,1])+squared[...,2])+squared[...,3])
    return q/norm[...,None]


def slerp_arrays(a, b, alpha):
    a, b = normalized(a), normalized(b)
    dot = ((a[:,0]*b[:,0]+a[:,1]*b[:,1])+a[:,2]*b[:,2])+a[:,3]*b[:,3]
    b = np.where((dot<0)[:,None],-b,b)
    dot = np.clip(np.abs(dot),0.,1.)
    linear = dot>0.9995
    result = np.empty_like(a)
    result[linear] = normalized(a[linear]+alpha[linear,None]*(b[linear]-a[linear]))
    angle = np.arccos(dot[~linear]); f = alpha[~linear]
    result[~linear] = (np.sin((1-f)*angle)[:,None]*a[~linear]+
                      np.sin(f*angle)[:,None]*b[~linear])/np.sin(angle)[:,None]
    return result


def quaternion_velocity(lo, hi, dt):
    """MuJoCo's mju_subQuat(hi, lo), XYZW, evaluated across the leading axis."""
    ax,ay,az,aw = hi.T; bx,by,bz,bw = lo.T
    q = np.column_stack((bw*ax-bx*aw-by*az+bz*ay,
                         bw*ay+bx*az-by*aw-bz*ax,
                         bw*az-bx*ay+by*ax-bz*aw,
                         bw*aw+bx*ax+by*ay+bz*az))
    norm = np.sqrt((q[:,0]*q[:,0]+q[:,1]*q[:,1])+q[:,2]*q[:,2])
    axis = np.zeros((len(q),3)); axis[:,0] = 1.
    valid = norm>=1e-15
    axis[valid] = q[valid,:3]*(1/norm[valid,None])
    angle = 2*np.arctan2(np.where(valid,norm,0.),q[:,3])
    angle = np.where(angle>np.pi,angle-2*np.pi,angle)
    return (axis*angle[:,None])*(1/dt)[:,None]


class MotionArena:
    """Reusable power-of-two blocks; storage follows active clips, not dataset size."""
    def __init__(self, slots, joints):
        self.joints = joints
        self.starts = np.zeros(slots,dtype=np.int64)
        self.lengths = np.zeros(slots,dtype=np.int64)
        self.capacities = np.zeros(slots,dtype=np.int64)
        self.duration = np.zeros(slots)
        self.h = np.zeros(slots)
        self.values = np.empty((0,8+joints))
        self.used = 0
        self.free = defaultdict(list)

    def load(self, slot, motion):
        n = len(motion.times_s); capacity = 1<<(n-1).bit_length()
        if self.capacities[slot]:
            self.free[int(self.capacities[slot])].append(int(self.starts[slot]))
        if self.free[capacity]:
            start = self.free[capacity].pop()
        else:
            start = self.used; self.used += capacity
            if self.used>len(self.values):
                values = np.empty((max(self.used,2*len(self.values)),8+self.joints))
                values[:len(self.values)] = self.values
                self.values = values
        self.starts[slot],self.lengths[slot],self.capacities[slot] = start,n,capacity
        self.duration[slot] = motion.duration_s
        self.h[slot] = motion.times_s[1]-motion.times_s[0]
        self.values[start:start+n] = np.column_stack((motion.times_s,motion._positions,
                                                     motion._orientations,motion._joints))

    def sample(self, slots, requested):
        slots = np.asarray(slots,dtype=np.int64)
        t = np.clip(requested,0.,self.duration[slots])
        starts = self.starts[slots]
        lo = np.zeros(len(slots),dtype=np.int64)
        hi = self.lengths[slots].copy()
        # Segmented searchsorted; loop depth is log(max frames), never slot count.
        while np.any(lo<hi):
            mid = (lo+hi)//2
            value = self.values[starts+np.minimum(mid,self.lengths[slots]-1),0]
            right = (lo<hi)&(value<=t)
            hi = np.where((lo<hi)&~right,mid,hi)
            lo = np.where(right,mid+1,lo)
        lower = starts+np.clip(lo-1,0,self.lengths[slots]-2)
        a,b = self.values[lower],self.values[lower+1]
        alpha = (t-a[:,0])/(b[:,0]-a[:,0])
        pos = a[:,1:4]+alpha[:,None]*(b[:,1:4]-a[:,1:4])
        joints = a[:,8:]+alpha[:,None]*(b[:,8:]-a[:,8:])
        quat = slerp_arrays(a[:,4:8],b[:,4:8],alpha)
        return t,joints,pos,quat
