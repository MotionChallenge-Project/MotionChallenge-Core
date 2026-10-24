"""Direct adapter calls with bounded array views; no IPC or simulator handles."""
import contextlib
from dataclasses import fields
import json
from pathlib import Path
import time

import numpy as np

from motion_challenge.runtime.loading import load_controller
from motion_challenge.data.bodies import BodyArray
from motion_challenge.sdk import (ControlCommand, ProprioceptiveStatePacket,
                                  StateBatch, ReferenceBatch, ControlBatch, ReferenceFrame)


def array_slice(value, rows, body_indices=None):
    if type(value) not in (StateBatch,ReferenceBatch):
        raise ValueError('only observation arrays may cross the adapter boundary')
    parts = {}
    for field in fields(value):
        data = getattr(value,field.name)
        if isinstance(data,np.ndarray):
            data = data[rows]
            if body_indices is not None and field.name.startswith('body_'):
                data = data[:,:,body_indices]
            data = data.view(); data.flags.writeable = False
        elif field.name=='body_names' and body_indices is not None:
            data = tuple(data[i] for i in body_indices)
        parts[field.name] = data
    return type(value)(**parts)


def packet_inputs(states, refs):
    """The optional scalar adapter boundary; numerical Core never uses packets."""
    ss,rr = [],[]
    for row in range(len(states.time_s)):
        ss.append(ProprioceptiveStatePacket(float(states.time_s[row]),states.joint_names,
            *(tuple(getattr(states,name)[row]) for name in ('joint_position_rad','joint_velocity_rad_s',
              'root_orientation_xyzw','root_linear_velocity_world_m_s','root_angular_velocity_world_rad_s'))))
        bodies = np.concatenate((refs.body_position_world_m[row],refs.body_orientation_xyzw[row],
                                 refs.body_linear_velocity_world_m_s[row],refs.body_angular_velocity_world_rad_s[row]),axis=-1)
        rr.append(tuple(ReferenceFrame(float(refs.requested_time_s[row,k]),float(refs.source_time_s[row,k]),
            *(tuple(getattr(refs,name)[row,k]) for name in ('joint_position_rad','root_position_world_m',
              'root_orientation_xyzw','joint_velocity_rad_s','root_linear_velocity_world_m_s','root_angular_velocity_world_rad_s')),
            BodyArray(refs.body_names,bodies[k])) for k in range(refs.source_time_s.shape[1])))
    return tuple(ss),tuple(rr)




class PolicyBatch:
    """One persistent policy, arbitrary microbatches and independent slot resets."""
    def __init__(self, config, size, state, refs, log_path):
        runtime = json.loads(Path(config).read_text())
        if runtime['state_access'] != 'ideal-proprioceptive-no-position-v1':
            raise ValueError('invalid observation contract')
        limit = runtime.get('inference_batch_size',size)
        if type(limit) is not int or limit<1:
            raise ValueError('inference_batch_size must be a positive integer')
        self.capacity,self.slots,self.config = min(limit,size),size,config
        self.names = state.joint_names
        self.array_mode = bool(runtime.get('array_batch_entrypoint'))
        if type(state) is not StateBatch or type(refs) is not ReferenceBatch:
            raise ValueError('policy inputs must be observation arrays')
        bodies = refs.body_names
        self.body_indices = tuple(bodies.index(n) for n in runtime.get('reference_bodies',bodies))
        self.input_buffer_bytes = 0
        self.controller,self.singles,self.initialized = None,{},set()
        self.telemetry = {}
        self.log = Path(log_path).open('w')
        try:
            with contextlib.redirect_stdout(self.log),contextlib.redirect_stderr(self.log):
                if self.array_mode or runtime.get('batch_entrypoint'):
                    self.controller = load_controller(config,'array_batch_entrypoint' if self.array_mode else 'batch_entrypoint')
        except BaseException:
            self.log.close()
            raise

    def call_arrays(self, op, slots, states, references, seeds=None):
        slots = np.asarray(slots,dtype=np.int64)
        if op not in ('reset','act') or slots.ndim!=1 or not len(slots) or len(np.unique(slots))!=len(slots):
            raise ValueError('invalid batch request')
        if np.any(slots<0) or np.any(slots>=self.slots):
            raise ValueError('invalid slot')
        if op=='act' and not set(slots)<=self.initialized:
            raise ValueError('act before reset')
        if op=='reset' and (seeds is None or len(seeds)!=len(slots)):
            raise ValueError('invalid seed count')
        if states.joint_names!=self.names:
            raise ValueError('joint layout mismatch')
        stats = self.telemetry.setdefault(op,dict(requests=0,slots=0,input_assembly_s=0.,
            adapter_s=0.,control_validation_s=0.,ipc_s=0.,input_bytes_exposed=0))
        result = np.empty((len(slots),5,len(self.names))) if op=='act' else None
        for start in range(0,len(slots),self.capacity):
            end = min(start+self.capacity,len(slots)); ids = tuple(map(int,slots[start:end]))
            tick = time.perf_counter()
            ss = array_slice(states,slice(start,end))
            rr = array_slice(references,slice(start,end),self.body_indices)
            arrays = [getattr(v,f.name) for v in (ss,rr) for f in fields(v) if isinstance(getattr(v,f.name),np.ndarray)]
            if any(not np.isfinite(v).all() for v in arrays):
                raise ValueError('non-finite policy input')
            self.input_buffer_bytes = sum(v.nbytes for v in arrays)
            stats['input_bytes_exposed'] += self.input_buffer_bytes
            if not self.array_mode:
                ss,rr = packet_inputs(ss,rr)
            stats['input_assembly_s'] += time.perf_counter()-tick
            tick = time.perf_counter()
            with contextlib.redirect_stdout(self.log),contextlib.redirect_stderr(self.log):
                if op=='reset':
                    batch_seeds = tuple(map(int,seeds[start:end]))
                    if self.controller is not None:
                        self.controller.reset_batch(ids,ss,rr,batch_seeds)
                    else:
                        for slot,state,ref,seed in zip(ids,ss,rr,batch_seeds):
                            if slot not in self.singles:
                                self.singles[slot] = load_controller(self.config)
                            self.singles[slot].reset(state,ref,seed)
                    self.initialized.update(ids)
                else:
                    commands = (self.controller.act_batch(ids,ss,rr) if self.controller is not None else
                                tuple(self.singles[i].act(s,r) for i,s,r in zip(ids,ss,rr)))
            stats['adapter_s'] += time.perf_counter()-tick
            tick = time.perf_counter()
            if op=='act':
                if self.array_mode:
                    if not isinstance(commands,ControlBatch):
                        raise ValueError('array adapter must return ControlBatch')
                    values = [np.asarray(getattr(commands,f.name),dtype=np.float64) for f in fields(commands)]
                    if any(v.shape!=(len(ids),len(self.names)) for v in values):
                        raise ValueError('invalid control batch shape')
                    values = np.stack(values,axis=1)
                else:
                    if len(commands)!=len(ids) or any(not isinstance(c,ControlCommand) for c in commands):
                        raise ValueError('invalid control commands')
                    values = np.asarray([[getattr(c,f.name) for f in fields(ControlCommand)] for c in commands])
                if not np.isfinite(values).all() or (values[:,2:4]<0).any():
                    raise ValueError('invalid control values or negative gains')
                result[start:end] = values
            stats['control_validation_s'] += time.perf_counter()-tick
            stats['requests'] += 1; stats['slots'] += len(ids)
        return result


    def close(self):
        try:
            with contextlib.redirect_stdout(self.log),contextlib.redirect_stderr(self.log):
                for policy in ([self.controller] if self.controller is not None else self.singles.values()):
                    policy.close()
        finally:
            self.log.close()
