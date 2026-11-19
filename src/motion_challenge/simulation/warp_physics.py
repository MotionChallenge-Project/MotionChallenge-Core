"""GPU-resident canonical PD simulation for the separately versioned MJWarp track.

This module intentionally does not import mjlab or any model adapter. Physics
and full state remain private to Core; observations use the strict SDK allowlist.
"""
import importlib.metadata

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from motion_challenge.sdk import DeployableStateBatch
from motion_challenge.simulation.physics import World


@wp.kernel
def _pd(qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
        command: wp.array3d(dtype=wp.float32), qadr: wp.array(dtype=wp.int32),
        vadr: wp.array(dtype=wp.int32), limits: wp.array(dtype=wp.float32),
        ctrl: wp.array2d(dtype=wp.float32)):
    world, joint = wp.tid()
    torque = command[world, 2, joint] * (command[world, 0, joint] - qpos[world, qadr[joint]])
    torque += command[world, 3, joint] * (command[world, 1, joint] - qvel[world, vadr[joint]])
    torque += command[world, 4, joint]
    ctrl[world, joint] = wp.clamp(torque, -limits[joint], limits[joint])


@wp.kernel
def _record_overflow(overflow: wp.array(dtype=wp.int32), nefc: wp.array(dtype=wp.int32),
                     sticky: wp.array(dtype=wp.int32), peak: wp.array(dtype=wp.int32)):
    i=wp.tid()
    sticky[i]=sticky[i] | overflow[i]
    peak[i]=wp.max(peak[i],nefc[i])


@wp.kernel
def _freeze_inactive(active: wp.array(dtype=wp.bool), safe_qpos: wp.array(dtype=wp.float32),
                     qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
                     qacc: wp.array2d(dtype=wp.float32), warmstart: wp.array2d(dtype=wp.float32),
                     time: wp.array(dtype=wp.float32)):
    world, index = wp.tid()
    if not active[world]:
        if index < qpos.shape[1]:
            qpos[world,index] = safe_qpos[index]
        if index < qvel.shape[1]:
            qvel[world,index] = 0.0
            qacc[world,index] = 0.0
            warmstart[world,index] = 0.0
        if index == 0:
            time[world] = 0.0


class WarpPhysics:
    """Fixed-capacity worlds; partial resets preserve all other trajectories."""
    def __init__(self, profile, size, device="cuda:0", *, substeps=4,
                 nconmax=100, njmax=4096, capture=True):
        if size < 1 or substeps < 1 or nconmax < 1 or njmax < 1:
            raise ValueError("World count, substeps and capacities must be positive")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("MJWarp benchmark requires a CUDA device")
        self.world = World(profile)
        self.size, self.substeps = size, substeps
        self.nconmax,self.njmax=nconmax,njmax
        self.joint_names = self.world.joint_names
        self.root_id = self.world.root_body_id
        self.graph = None
        with wp.ScopedDevice(str(self.device)), self.stream():
            self.model = mjw.put_model(self.world.model)
            self.data = mjw.put_data(self.world.model, self.world.data, nworld=size,
                                     nconmax=nconmax, njmax=njmax)
            self.qpos = wp.to_torch(self.data.qpos)
            self.qvel = wp.to_torch(self.data.qvel)
            self.qpos[:] = torch.as_tensor(self.world.model.qpos0, device=self.device, dtype=self.qpos.dtype)
            self.command = torch.zeros((size, 5, len(self.joint_names)), device=self.device)
            self.command_wp = wp.from_torch(self.command)
            self.qadr = wp.array(self.world.qadr, dtype=wp.int32)
            self.vadr = wp.array(self.world.vadr, dtype=wp.int32)
            self.limits = wp.array(self.world.limits, dtype=wp.float32)
            self.qids = torch.as_tensor(self.world.qadr, device=self.device, dtype=torch.long)
            self.vids = torch.as_tensor(self.world.vadr, device=self.device, dtype=torch.long)
            self.dt = torch.full((size,), float(self.world.profile["physics_dt_s"]), device=self.device)
            self.model.opt.timestep = wp.from_torch(self.dt)
            self.overflow_sticky = wp.zeros(size,dtype=wp.int32)
            self.nefc_peak = wp.zeros(size,dtype=wp.int32)
            self.reset_mask = torch.zeros(size, dtype=torch.bool, device=self.device)
            self.reset_wp = wp.from_torch(self.reset_mask)
            self.time_s = torch.zeros(size, dtype=torch.float64, device=self.device)
            self.active = torch.ones(size, dtype=torch.bool, device=self.device)
            self.active_wp = wp.from_torch(self.active)
            self.safe_qpos = torch.as_tensor(self.world.model.qpos0, device=self.device, dtype=torch.float32).clone()
            self.safe_qpos[:3] = torch.tensor([0.,0.,10.], device=self.device)
            self.safe_qpos[3:7] = torch.tensor([1.,0.,0.,0.], device=self.device)
            self.safe_qpos_wp = wp.from_torch(self.safe_qpos)
            self.control_period_s = float(self.world.profile["physics_dt_s"])*substeps
            if capture:
                # CUDA's legacy default stream cannot be captured. Build once
                # on a private Warp stream; launches use the caller's stream.
                torch.cuda.synchronize(self.device)
                capture_stream = wp.Stream(device=str(self.device))
                with wp.ScopedStream(capture_stream):
                    with wp.ScopedCapture() as captured:
                        self._advance()
                self.graph = captured.graph
            mjw.forward(self.model, self.data)

    def stream(self):
        return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))

    def _advance(self):
        for _ in range(self.substeps):
            wp.launch(_pd, dim=(self.size,len(self.joint_names)),
                      inputs=[self.data.qpos,self.data.qvel,self.command_wp,
                              self.qadr,self.vadr,self.limits],outputs=[self.data.ctrl])
            mjw.step(self.model,self.data)
            wp.launch(_record_overflow,dim=self.size,
                      inputs=[self.data.overflow,self.data.nefc],
                      outputs=[self.overflow_sticky,self.nefc_peak])
        wp.launch(_freeze_inactive, dim=(self.size,max(self.world.model.nq,self.world.model.nv)),
                  inputs=[self.active_wp,self.safe_qpos_wp],
                  outputs=[self.data.qpos,self.data.qvel,self.data.qacc,self.data.qacc_warmstart,self.data.time])
        # mj_step's position-dependent fields precede integration. Refresh them
        # explicitly before policy observations and post-step metric sampling.
        mjw.kinematics(self.model,self.data)
        mjw.com_pos(self.model,self.data)
        mjw.com_vel(self.model,self.data)

    def reset(self, slots, qpos, qvel):
        slots = torch.as_tensor(slots, device=self.device, dtype=torch.long)
        if qpos.shape != (len(slots),self.world.model.nq) or qvel.shape != (len(slots),self.world.model.nv):
            raise ValueError("Reset state shape mismatch")
        self.reset_mask.zero_()
        self.reset_mask[slots] = True
        with wp.ScopedDevice(str(self.device)), self.stream():
            mjw.reset_data(self.model,self.data,reset=self.reset_wp)
            self.qpos[slots] = qpos
            self.qvel[slots] = qvel
            self.time_s[slots] = 0
            self.command[slots] = 0
            self.active[slots] = True
            mjw.kinematics(self.model,self.data)
            mjw.com_pos(self.model,self.data)
            mjw.com_vel(self.model,self.data)

    def park(self, slots):
        """Keep unassigned slots at a safe pose without touching active worlds."""
        slots = torch.as_tensor(slots, device=self.device, dtype=torch.long)
        if not len(slots):
            return
        self.reset(slots,self.safe_qpos.expand(len(slots),-1),
                   torch.zeros((len(slots),self.world.model.nv),device=self.device))
        self.active[slots] = False

    def step(self, commands, duration_s):
        """Step all worlds with device tensors; no host copies in the hot path.

        duration_s is [N], allowing an exact fractional endpoint per world.
        Unassigned worlds keep a normal timestep and receive zero controls.
        """
        if commands.shape != self.command.shape or duration_s.shape != (self.size,):
            raise ValueError("Command or duration batch shape mismatch")
        if commands.device != self.device or duration_s.device != self.device:
            raise ValueError("Inputs must already be on the simulation device")
        self.command.copy_(torch.where(self.active[:,None,None],commands,0.))
        duration = torch.where(self.active,duration_s,self.control_period_s)
        self.dt.copy_(duration / self.substeps)
        with wp.ScopedDevice(str(self.device)), self.stream():
            if self.graph is None:
                self._advance()
            else:
                wp.capture_launch(self.graph)
        self.time_s.add_(torch.where(self.active,duration_s.double(),0.))

    def observations(self):
        quat=wp.to_torch(self.data.xquat)[:,self.root_id]
        return DeployableStateBatch(self.joint_names,self.time_s,
            self.qpos[:,self.qids],self.qvel[:,self.vids],quat[:,[1,2,3,0]],
            wp.to_torch(self.data.cvel)[:,self.root_id,:3])

    def check(self):
        """Explicit validation boundary, including contact/constraint overflow."""
        finite = torch.isfinite(self.qpos).all(dim=1) & torch.isfinite(self.qvel).all(dim=1)
        if not bool((finite | ~self.active).all()):
            raise FloatingPointError("Non-finite MJWarp state")
        overflow=wp.to_torch(self.overflow_sticky)
        if bool(overflow.any()) or bool((wp.to_torch(self.nefc_peak)>self.njmax).any()):
            slots=torch.nonzero(overflow).flatten().cpu().tolist()[:12]
            raise FloatingPointError(f"MJWarp capacity overflow; slots={slots}, peak_constraints={int(wp.to_torch(self.nefc_peak).max())}, njmax={self.njmax}")

    def metadata(self):
        return {"backend":"mjwarp","precision":str(self.qpos.dtype),
                "versions":{n:importlib.metadata.version(n) for n in ("mujoco","mujoco-warp","warp-lang","torch")},
                "worlds":self.size,"substeps":self.substeps,"nconmax":self.nconmax,"njmax":self.njmax,"cuda_graph":self.graph is not None}
