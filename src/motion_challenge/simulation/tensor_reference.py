"""GPU reference sampling and private FK buffers; no per-step host copies."""
import torch
import warp as wp
import mujoco_warp as mjw
from motion_challenge.simulation.tensor_math import slerp,rotate,quaternion_velocity
from motion_challenge.sdk import ReferenceBatch,MotionReferenceBatch


class TensorMotionBank:
    def __init__(self, poses, starts, lengths, fps, names, device):
        self.device=torch.device(device)
        # poses are [position xyz, quaternion xyzw, named joint angles].
        self.poses=torch.as_tensor(poses,device=self.device)
        self.starts=torch.as_tensor(starts,device=self.device,dtype=torch.long)
        self.lengths=torch.as_tensor(lengths,device=self.device,dtype=torch.long)
        self.fps=torch.as_tensor(fps,device=self.device,dtype=torch.float64)
        self.duration=(self.lengths-1)/self.fps
        self.names=tuple(names)

    def sample(self,ids,times):
        ids=torch.broadcast_to(ids,times.shape)
        t=torch.minimum(times.clamp_min(0),self.duration[ids])
        frame=t*self.fps[ids]
        lo=torch.minimum(frame.floor().long(),self.lengths[ids]-2).clamp_min(0)
        alpha=frame-lo
        a=self.poses[self.starts[ids]+lo].double()
        b=self.poses[self.starts[ids]+lo+1].double()
        p=a+alpha[...,None]*(b-a)
        q=slerp(a[...,3:7],b[...,3:7],alpha)
        return t,p[...,:3],q,p[...,7:]

    def state(self,ids,times):
        t,p,q,j=self.sample(ids,times)
        h=1/self.fps[ids]
        lo=(t-h).clamp_min(0);hi=torch.minimum(t+h,self.duration[ids])
        _,pl,ql,jl=self.sample(ids,lo);_,ph,qh,jh=self.sample(ids,hi)
        dt=hi-lo
        return t,p,q,j,(ph-pl)/dt[...,None],quaternion_velocity(ql,qh,dt),(jh-jl)/dt[...,None]


class TensorReference:
    def __init__(self,world,bank,capacity,device):
        self.world,self.bank,self.device=world,bank,torch.device(device)
        self.capacity=capacity
        self.body_ids=list(world.body_ids);self.body_names=world.body_names
        self.qids=torch.as_tensor(world.qadr,device=device,dtype=torch.long)
        self.vids=torch.as_tensor(world.vadr,device=device,dtype=torch.long)
        self.roots=torch.as_tensor(world.body_roots,device=device,dtype=torch.long)
        self.model=None
        self.data=None

    def _ensure_fk(self):
        if self.data is not None:return
        world=self.world
        with wp.ScopedDevice(str(self.device)),self.stream():
            self.model=mjw.put_model(world.model)
            self.data=mjw.make_data(world.model,nworld=self.capacity,nconmax=1,njmax=1)
            self.qpos=wp.to_torch(self.data.qpos);self.qvel=wp.to_torch(self.data.qvel)
            self.qpos[:]=torch.as_tensor(world.model.qpos0,device=self.device,dtype=self.qpos.dtype)

    def stream(self):
        return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))

    def motion(self,ids,requested):
        """Policy references never run or expose body kinematics."""
        shape=requested.shape
        flat_ids=torch.broadcast_to(ids[:,None],shape).reshape(-1)
        t,p,q,j,linear,omega,jv=self.bank.state(flat_ids,requested.reshape(-1))
        def shaped(x):return x.reshape(*shape,*x.shape[1:])
        return MotionReferenceBatch(requested,shaped(t),shaped(j),shaped(p),shaped(q),
                                    shaped(jv),shaped(linear),shaped(rotate(q,omega)))

    def get(self,ids,requested):
        self._ensure_fk()
        shape=requested.shape;n=requested.numel()
        if n>self.capacity:raise ValueError("Reference FK capacity exceeded")
        ids=torch.broadcast_to(ids[:,None],shape).reshape(-1)
        t,p,q,j,linear,omega,jv=self.bank.state(ids,requested.reshape(-1))
        self.qpos[:n,:3]=p;self.qpos[:n,3:7]=q[:,[3,0,1,2]]
        self.qpos[:n,self.qids]=j.float();self.qvel[:n,:3]=linear
        self.qvel[:n,3:6]=omega;self.qvel[:n,self.vids]=jv.float()
        with wp.ScopedDevice(str(self.device)),self.stream():
            mjw.kinematics(self.model,self.data);mjw.com_pos(self.model,self.data);mjw.com_vel(self.model,self.data)
        bp=wp.to_torch(self.data.xpos)[:n,self.body_ids]
        bq=wp.to_torch(self.data.xquat)[:n,self.body_ids][:,:,[1,2,3,0]]
        cv=wp.to_torch(self.data.cvel)[:n,self.body_ids]
        offset=bp-wp.to_torch(self.data.subtree_com)[:n,self.roots]
        bv=cv[...,3:]+torch.linalg.cross(cv[...,:3],offset)
        def shaped(x):return x.reshape(*shape,*x.shape[1:])
        return ReferenceBatch(requested,shaped(t),shaped(j),shaped(p),shaped(q),shaped(jv),
            shaped(linear),shaped(rotate(q,omega)),self.body_names,shaped(bp),shaped(bq),shaped(bv),shaped(cv[...,:3]))

    def initial(self,ids):
        t=torch.zeros(len(ids),device=self.device,dtype=torch.float64)
        _,p,q,j,linear,omega,jv=self.bank.state(ids,t)
        qpos=torch.as_tensor(self.world.model.qpos0,device=self.device,dtype=torch.float32).repeat(len(ids),1)
        qvel=torch.zeros((len(ids),self.world.model.nv),device=self.device)
        qpos[:,:3]=p;qpos[:,3:7]=q[:,[3,0,1,2]];qpos[:,self.qids]=j.float()
        qvel[:,:3]=linear;qvel[:,3:6]=omega;qvel[:,self.vids]=jv.float()
        return qpos,qvel
