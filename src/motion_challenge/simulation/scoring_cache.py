"""Precompute scoring targets with MuJoCo Warp before the rollout.

The cache samples the exact metric clock plus every motion's exact endpoint.
It is private scoring data, never a policy observation.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import mujoco_warp as mjw
import warp as wp
from motion_challenge.validation import sha256_file


class ScoringReferenceCache:
    def __init__(self, world, bank, protocol, prepared, device, batch=8192):
        self.device=torch.device(device);self.body_names=tuple(protocol["metric_bodies"])
        if protocol["failure"]["anchor_body"] not in self.body_names:
            self.body_names=(protocol["failure"]["anchor_body"],*self.body_names)
        self.root_row=self.body_names.index(protocol["failure"]["anchor_body"])
        self.body_rows=[self.body_names.index(x) for x in protocol["metric_bodies"]]
        prepared=Path(prepared);source=json.loads((prepared/"dataset.json").read_text())
        period=protocol["metric_period_s"]
        duration=(np.asarray(source["lengths"])-1)/source["fps"]
        # Each clip contains t=0, regular scoring ticks, and its exact endpoint.
        counts=np.ceil(duration/period-1e-10).astype(np.int64)+1
        starts=np.r_[0,np.cumsum(counts)[:-1]];total=int(counts.sum())
        identity={"source_poses_sha256":source["poses_sha256"],"robot":world.profile,
                  "period_s":period,"bodies":self.body_names,"engine":protocol["engine_versions"],
                  "builder_sha256":sha256_file(__file__),"reference_convention":"native-pose-slerp-f32-kinematics-v1"}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        directory=prepared/("scoring-"+key);directory.mkdir(exist_ok=True)
        data_path=directory/"targets.npy";receipt=directory/"metadata.json"
        width=3+4+len(world.joint_names)+3*len(self.body_names)
        import fcntl
        with (directory/"build.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not receipt.exists():
                values=np.lib.format.open_memmap(directory/"targets.partial.npy",mode="w+",dtype=np.float32,shape=(total,width))
                with wp.ScopedDevice(str(device)),self.stream():
                    model=mjw.put_model(world.model)
                    data=mjw.make_data(world.model,nworld=batch,nconmax=1,njmax=1)
                    qpos=wp.to_torch(data.qpos)
                    qpos[:]=torch.as_tensor(world.model.qpos0,device=device,dtype=qpos.dtype)
                    qids=torch.as_tensor(world.qadr,device=device,dtype=torch.long)
                    body_ids=[world.model.body(x).id for x in self.body_names]
                    for offset in range(0,total,batch):
                        rows=np.arange(offset,min(offset+batch,total));n=len(rows)
                        clips=np.searchsorted(starts,rows,side="right")-1
                        times=np.minimum((rows-starts[clips])*period,duration[clips])
                        ids=torch.as_tensor(clips,device=device)
                        _,pos,quat,joints=bank.sample(ids,torch.as_tensor(times,device=device))
                        qpos[:n,:3]=pos;qpos[:n,3:7]=quat[:,[3,0,1,2]];qpos[:n,qids]=joints.float()
                        mjw.kinematics(model,data)
                        body=wp.to_torch(data.xpos)[:n,body_ids]
                        values[offset:offset+n]=torch.cat((pos,quat,joints,body.flatten(1)),dim=-1).float().cpu().numpy()
                values.flush();del values
                os.replace(directory/"targets.partial.npy",data_path)
                metadata={"identity":identity,"key":key,"samples":total,"width":width,
                          "targets_sha256":sha256_file(data_path)}
                temp=receipt.with_suffix(".tmp");temp.write_text(json.dumps(metadata,indent=2)+"\n");os.replace(temp,receipt)
            self.metadata=json.loads(receipt.read_text())
            if sha256_file(data_path)!=self.metadata["targets_sha256"]:
                raise ValueError("Scoring reference cache checksum mismatch")
        mmap=np.load(data_path,mmap_mode="r")
        self.values=torch.empty(mmap.shape,dtype=torch.float32,device=device)
        for offset in range(0,total,65536):
            self.values[offset:offset+65536]=torch.tensor(np.array(mmap[offset:offset+65536]),device=device)
        self.starts=torch.as_tensor(starts,device=device)
        self.counts=torch.as_tensor(counts,device=device)
        self.joints=len(world.joint_names)
        self.path=directory

    def stream(self):
        return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))

    def get(self,clips,steps):
        row=self.starts[clips]+torch.minimum(steps,self.counts[clips]-1)
        x=self.values[row];j=self.joints
        return x[:,:3],x[:,3:7],x[:,7:7+j],x[:,7+j:].reshape(len(clips),len(self.body_names),3)
