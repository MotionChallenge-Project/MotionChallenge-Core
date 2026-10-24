"""Direct adapter calls with bounded array views; no IPC or simulator handles."""
import contextlib
from collections import OrderedDict
from dataclasses import fields
import json
from pathlib import Path
import time

import numpy as np

from motion_challenge.runtime.loading import load_controller
from motion_challenge.data.bodies import BodyArray
from motion_challenge.sdk import (ControlCommand, ProprioceptiveStatePacket,
                                  StateBatch, ReferenceBatch, ControlBatch)


class ArrayInputs:
    """Reusable, adapter-facing arrays containing only permitted observations."""
    def __init__(self, layout):
        self.layout = layout
        self.names = tuple(layout['joints'])
        self.bodies = tuple(layout['bodies'])
        n,k,j = layout['capacity'],layout['frames'],len(self.names)
        self.states = np.empty((n,11+2*j),dtype=np.float64)
        self.references = np.empty((n,k,15+2*j+13*len(self.bodies)),dtype=np.float64)
        self.size_bytes = self.states.nbytes+self.references.nbytes
        self.reference_cache = OrderedDict()
        self.reference_cache_size = 2*layout['slots']*k

    def put(self, row, state, refs):
        if not isinstance(state, ProprioceptiveStatePacket) or tuple(state.joint_names) != self.names:
            raise ValueError('only matching proprioceptive packets may cross the adapter boundary')
        if state.schema != 'ideal-proprioceptive-no-position-v1' or len(refs) != self.layout['frames']:
            raise ValueError('invalid observation schema or reference window')
        self.states[row] = (state.time_s, *state.joint_position_rad, *state.joint_velocity_rad_s,
                            *state.root_orientation_xyzw, *state.root_linear_velocity_world_m_s,
                            *state.root_angular_velocity_world_rad_s)
        for index, ref in enumerate(refs):
            key = id(ref)
            cached = self.reference_cache.get(key)
            if cached is not None and cached[0] is ref:
                self.reference_cache.move_to_end(key)
                self.references[row, index] = cached[1]
                continue
            names = ref.bodies.names if isinstance(ref.bodies,BodyArray) else tuple(b.name for b in ref.bodies)
            if names != self.bodies:
                raise ValueError('reference body layout changed')
            head = 15+2*len(self.names)
            self.references[row, index, :head] = (
                ref.requested_time_s, ref.source_time_s, *ref.joint_position_rad,
                *ref.root_position_world_m, *ref.root_orientation_xyzw, *ref.joint_velocity_rad_s,
                *ref.root_linear_velocity_world_m_s, *ref.root_angular_velocity_world_rad_s)
            self.references[row, index, head:] = (ref.bodies.values.reshape(-1)
                if isinstance(ref.bodies,BodyArray) else tuple(v for b in ref.bodies
                for a in (b.position_world_m,b.orientation_xyzw,b.linear_velocity_world_m_s,
                          b.angular_velocity_world_rad_s) for v in a))
            self.reference_cache[key] = (ref, self.references[row,index].copy())
            if len(self.reference_cache) > self.reference_cache_size:
                self.reference_cache.popitem(last=False)

    def get_arrays(self, count):
        j = len(self.names)
        def view(array):
            result = array.view()
            result.flags.writeable = False
            return result
        x = self.states[:count]
        states = StateBatch(self.names, *(view(a) for a in (x[:,0],x[:,1:1+j],x[:,1+j:1+2*j],
            x[:,1+2*j:5+2*j],x[:,5+2*j:8+2*j],x[:,8+2*j:11+2*j])))
        x = self.references[:count]
        bodies = x[:,:,15+2*j:].reshape(count,self.layout['frames'],len(self.bodies),13)
        refs = ReferenceBatch(*(view(a) for a in (x[:,:,0],x[:,:,1],x[:,:,2:2+j],
            x[:,:,2+j:5+j],x[:,:,5+j:9+j],x[:,:,9+j:9+2*j],
            x[:,:,9+2*j:12+2*j],x[:,:,12+2*j:15+2*j])),self.bodies,
            *(view(bodies[:,:,:,a:b]) for a,b in ((0,3),(3,7),(7,10),(10,13))))
        return states,refs


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
        self.buffer = ArrayInputs(dict(capacity=self.capacity,slots=size,frames=len(refs),
            joints=self.names,bodies=refs[0].bodies.names if isinstance(refs[0].bodies,BodyArray)
            else [b.name for b in refs[0].bodies])) if self.array_mode else None
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

    def call(self, op, slots, states, references, seeds=None):
        if op not in ('reset','act') or not slots or not (len(slots)==len(states)==len(references)) or len(set(slots))!=len(slots):
            raise ValueError('invalid batch request')
        if any(type(i) is not int or not 0<=i<self.slots for i in slots):
            raise ValueError('invalid slot')
        if op=='reset' and (seeds is None or len(seeds)!=len(slots)):
            raise ValueError('invalid seed count')
        if op=='act' and not set(slots)<=self.initialized:
            raise ValueError('act before reset')
        stats = self.telemetry.setdefault(op,dict(requests=0,slots=0,input_assembly_s=0.,
            adapter_s=0.,control_validation_s=0.,ipc_s=0.,input_bytes_written=0))
        result = []
        for start in range(0,len(slots),self.capacity):
            end = min(start+self.capacity,len(slots))
            ids = tuple(slots[start:end])
            ss,rr = tuple(states[start:end]),tuple(references[start:end])
            started = time.perf_counter()
            if any(not isinstance(s,ProprioceptiveStatePacket) or s.schema!='ideal-proprioceptive-no-position-v1'
                   or s.joint_names!=self.names for s in ss):
                raise ValueError('only matching proprioceptive packets may reach an adapter')
            if self.array_mode:
                for row,(s,r) in enumerate(zip(ss,rr)):
                    self.buffer.put(row,s,r)
                ss,rr = self.buffer.get_arrays(len(ids))
                if not np.isfinite(self.buffer.states[:len(ids)]).all() or not np.isfinite(self.buffer.references[:len(ids)]).all():
                    raise ValueError('non-finite policy input')
                stats['input_bytes_written'] += self.buffer.states[:len(ids)].nbytes+self.buffer.references[:len(ids)].nbytes
            stats['input_assembly_s'] += time.perf_counter()-started
            started = time.perf_counter()
            with contextlib.redirect_stdout(self.log),contextlib.redirect_stderr(self.log):
                if op=='reset':
                    batch_seeds = tuple(seeds[start:end])
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
            stats['adapter_s'] += time.perf_counter()-started
            started = time.perf_counter()
            if op=='act':
                if self.array_mode:
                    if not isinstance(commands,ControlBatch):
                        raise ValueError('array adapter must return ControlBatch')
                    values = [np.asarray(getattr(commands,f.name),dtype=np.float64) for f in fields(commands)]
                    if any(v.shape!=(len(ids),len(self.names)) for v in values):
                        raise ValueError('invalid control batch shape')
                    values = np.stack(values,axis=1)
                else:
                    commands = tuple(commands)
                    if len(commands)!=len(ids) or any(not isinstance(c,ControlCommand) for c in commands):
                        raise ValueError('batch adapter must return one ControlCommand per slot')
                    values = np.array([[getattr(c,f.name) for f in fields(ControlCommand)] for c in commands],dtype=np.float64)
                if values.shape!=(len(ids),5,len(self.names)) or not np.isfinite(values).all() or (values[:,2:4]<0).any():
                    raise ValueError('invalid control values or negative gains')
                result.extend(ControlCommand(*row) for row in values)
            stats['control_validation_s'] += time.perf_counter()-started
            stats['requests'] += 1
            stats['slots'] += len(ids)
        return result

    def close(self):
        try:
            with contextlib.redirect_stdout(self.log),contextlib.redirect_stderr(self.log):
                for policy in ([self.controller] if self.controller is not None else self.singles.values()):
                    policy.close()
        finally:
            self.log.close()
