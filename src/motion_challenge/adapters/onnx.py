"""Small SDK-native ONNX/PD utility; no model registry or external deployment stack."""
import json
import re
import weakref
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from motion_challenge.sdk import ControlCommand
from motion_challenge.validation import sha256_file


_sessions = weakref.WeakValueDictionary()


def cpu_session(weights, expected_hash):
    """Share an immutable, verified graph; controller histories remain separate."""
    key = (str(weights.resolve()), expected_hash)
    session = _sessions.get(key)
    if session is None:
        if sha256_file(weights) != expected_hash:
            raise ValueError('checkpoint checksum mismatch')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(weights), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        _sessions[key] = session
    return session


def vector(value,names,default=0.):
    if isinstance(value,dict):
        result=[]
        for name in names:
            matches=[v for pattern,v in value.items() if re.fullmatch(pattern,name)]
            if len(matches)>1:raise ValueError('ambiguous named joint value')
            result.append(matches[0] if matches else default)
        return np.asarray(result,dtype=np.float64)
    a=np.asarray(value,dtype=np.float64)
    if a.ndim==0:return np.full(len(names),a.item())
    if a.shape!=(len(names),):raise ValueError('invalid joint vector')
    return a


def yaw(rotation):
    matrix=rotation.as_matrix()
    return np.arctan2(matrix[...,1,0],matrix[...,0,0])


def roll_pitch(rotation):
    m=rotation.as_matrix()
    return np.stack([np.arctan2(m[...,2,1],m[...,2,2]),np.arcsin(np.clip(-m[...,2,0],-1,1))],axis=-1)


class ONNXController:
    action_output='action'
    raw_action_history=False

    def __init__(self,root,runtime):
        self.root,self.runtime=root,runtime
        config_path=root/runtime['inference_config']
        if sha256_file(config_path)!=runtime['inference_config_sha256']:raise ValueError('config checksum mismatch')
        self.cfg=json.loads(config_path.read_text())
        weights=root/runtime['weights']
        self.session=cpu_session(weights,runtime['weights_sha256'])
        self.names=self.cfg['policy_joint_names']
        self.default=vector(self.cfg['default_joint_pos'],self.names)
        self.scale=vector(self.cfg['action_scale'],self.names,1.)
        self.kp=vector(self.cfg['joint_kp'],self.names);self.kd=vector(self.cfg['joint_kd'],self.names)

    def reset(self,state,reference,seed):
        self.order=[state.joint_names.index(n) for n in self.names]
        self.inverse=[self.names.index(n) for n in state.joint_names]
        self.last=np.zeros(len(self.names),np.float32)

    def proprio(self,state):
        rotation=R.from_quat(state.root_orientation_xyzw)
        return (np.asarray(state.joint_position_rad)[self.order],np.asarray(state.joint_velocity_rad_s)[self.order],
                rotation.inv().apply(state.root_angular_velocity_world_rad_s),rotation.inv().apply([0,0,-1]),rotation)

    def infer(self,inputs):
        feed={}
        types={'tensor(float)':np.float32,'tensor(int64)':np.int64,'tensor(int32)':np.int32}
        for node in self.session.get_inputs():
            a=np.asarray(inputs[node.name],dtype=types[node.type])
            shape=tuple(v if isinstance(v,int) else 1 for v in node.shape)
            if a.shape!=shape:raise ValueError(f'{node.name}: {a.shape} != {shape}')
            feed[node.name]=a
        return dict(zip([n.name for n in self.session.get_outputs()],self.session.run(None,feed)))

    def act(self,state,reference):
        result=self.infer(self.observations(state,reference))
        self.after_inference(result)
        raw=result[self.action_output].reshape(len(self.names))
        if not np.isfinite(raw).all():raise ValueError('nonfinite action')
        clip=self.cfg.get('action_clip')
        action=np.clip(raw,-clip,clip) if clip is not None else raw
        self.last=(raw if self.raw_action_history else action).astype(np.float32)
        target=self.default+self.scale*action
        def ordered(x):return tuple(float(v) for v in x[self.inverse])
        return ControlCommand(ordered(target),(0.,)*29,ordered(self.kp),ordered(self.kd),(0.,)*29)

    def after_inference(self,result):pass
    def close(self):self.session=None
