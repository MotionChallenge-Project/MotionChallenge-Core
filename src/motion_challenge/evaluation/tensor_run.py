"""GPU rollout with dynamic refill and trace evidence; development until verified."""
import argparse,hashlib,importlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
import warp as wp
from motion_challenge.data.prepare import prepare
from motion_challenge.resources import resource_root
from motion_challenge.simulation.warp_physics import WarpPhysics
from motion_challenge.simulation.tensor_reference import TensorMotionBank,TensorReference
from motion_challenge.evaluation.tensor_metrics import score,NAMES
from motion_challenge.simulation.scoring_cache import ScoringReferenceCache
from motion_challenge.validation import sha256_file


@torch.inference_mode()
def evaluate(root,dataset,data,cache,output,weights,slots=None,device="cuda:0",max_clips=None,max_steps=None,no_traces=False,data_format="original"):
    root=Path(root);output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError("Use a new result directory")
    output.mkdir(parents=True,exist_ok=True)
    runtime=json.loads((root/"runtime.mjwarp.json").read_text())
    admission=json.loads((root/"admission.mjwarp.json").read_text())
    if admission["status"]=="excluded":raise ValueError(admission["reason"])
    if not runtime.get("tensor_entrypoint"):raise ValueError("Model tensor adapter is not implemented")
    resources=resource_root()
    protocol_path=resources/f"suites/{dataset}/protocol.mjwarp311-no-linear-velocity-v1.json"
    protocol=json.loads(protocol_path.read_text())
    protocol_digest=sha256_file(protocol_path)
    runtime_digest=sha256_file(root/"runtime.mjwarp.json")
    if runtime["state_access"]!=protocol["state_access"]:raise ValueError("Observation contract mismatch")
    if sha256_file(root/runtime["inference_config"])!=runtime["inference_config_sha256"]:
        raise ValueError("Inference configuration hash mismatch")
    if sha256_file(root/"admission.mjwarp.json")!=runtime["observation_contract"]["admission_sha256"]:
        raise ValueError("Admission hash mismatch")
    (output/"protocol.json").write_text(protocol_path.read_text())
    (output/"runtime.json").write_text((root/"runtime.mjwarp.json").read_text())
    import motion_challenge
    code_root=Path(motion_challenge.__file__).resolve().parent
    source_files={str(p.relative_to(code_root)):sha256_file(p) for p in code_root.rglob("*.py")}
    adapter_files={p.name:sha256_file(p) for p in root.glob("*.py")}
    weight_files={str(p.relative_to(weights)):sha256_file(p) for p in Path(weights).rglob("*") if p.is_file()}
    if not weight_files:raise ValueError("No converted weights")
    if data_format=="lerobot-v3":
        from motion_challenge.data.prepare_v3 import prepare_v3
        prepared=prepare_v3(dataset,data,cache).resolve()
    elif data_format=="original":
        prepared=prepare(dataset,data,cache).resolve()
    else:raise ValueError("Unknown data format")
    meta=json.loads((prepared/"dataset.json").read_text())
    poses=np.load(prepared/"poses.npy",mmap_mode="r")
    count=len(meta["names"]) if max_clips is None else min(max_clips,len(meta["names"]))
    slots=min(4096 if slots is None else slots,count)
    dt=float(runtime["control_period_s"])
    if dt!=protocol["metric_period_s"] or dt!=protocol["control_period_s"]:
        raise ValueError("This tensor runner currently requires equal policy and metric periods")
    profile=resources/protocol["robot_profile"]
    if sha256_file(profile)!=protocol["robot_profile_sha256"]:raise ValueError("Robot profile hash mismatch")
    if abs(protocol["physics_substeps"]*protocol["physics_dt_s"]-dt)>1e-12:raise ValueError("Physics clock mismatch")
    physics=WarpPhysics(profile,slots,device,substeps=protocol["physics_substeps"])
    if physics.metadata()["versions"]!=protocol["engine_versions"]:raise ValueError("Runtime versions differ from protocol")
    bank=TensorMotionBank(np.asarray(poses),meta["starts"],meta["lengths"],np.full(len(meta["names"]),meta["fps"]),meta["names"],device)
    scoring=ScoringReferenceCache(physics.world,bank,protocol,prepared,device)
    offsets=torch.tensor(runtime["reference_offsets_s"],device=device,dtype=torch.float64)
    reference=TensorReference(physics.world,bank,slots*max(1,len(offsets)),device)
    sys.path.insert(0,str(root))
    module,cls=runtime["tensor_entrypoint"].split(":")
    adapter=getattr(importlib.import_module(module),cls)(root,runtime,physics.joint_names,slots,device,weights)
    durations=(np.asarray(meta["lengths"][:count])-1)/meta["fps"]
    lengths=np.ceil(durations/dt-1e-10).astype(int)
    order=np.argsort(-lengths,kind="stable");cursor=0
    clip_ids=torch.zeros(slots,device=device,dtype=torch.long)
    steps=torch.zeros(slots,device=device,dtype=torch.long)
    active=np.zeros(slots,dtype=bool);remaining=np.zeros(slots,dtype=int)
    sums=torch.zeros((slots,len(NAMES)),device=device,dtype=torch.float64)
    prefix=torch.zeros_like(sums);prefix_count=torch.zeros(slots,device=device,dtype=torch.long)
    seen_failure=torch.zeros(slots,device=device,dtype=torch.bool)
    failure_time=torch.full((slots,),float("nan"),device=device,dtype=torch.float64)
    last_valid=torch.zeros(slots,device=device,dtype=torch.float64)
    body_rows=scoring.body_rows
    root_row=scoring.root_row
    body_ids=[physics.world.model.body(n).id for n in scoring.body_names]
    records=[];traces=[];pending=[];iteration=0
    invalid=torch.zeros((),device=device,dtype=torch.bool)
    torch.cuda.synchronize(device)
    started=time.monotonic();last_progress=started
    def refill(free):
        nonlocal cursor
        n=min(len(free),count-cursor)
        if n<len(free):physics.park(torch.as_tensor(free[n:],device=device))
        if not n:return
        free=free[:n];ids=torch.as_tensor(free,device=device);clips=order[cursor:cursor+n]
        clip_ids[ids]=torch.as_tensor(clips,device=device);steps[ids]=0
        sums[ids]=0;prefix[ids]=0;prefix_count[ids]=0;seen_failure[ids]=False;failure_time[ids]=float("nan");last_valid[ids]=0
        q,v=reference.initial(clip_ids[ids]);physics.reset(ids,q,v);adapter.reset(ids)
        active[free]=True;remaining[free]=lengths[clips];cursor+=n
    def flush():
        if not pending:return
        data=torch.stack(pending).cpu().numpy()
        path=output/f"trace-{len(traces):05d}.npz"
        np.savez(path,clip_id=data[:,:,0].astype(np.int64),time_s=data[:,:,1],qpos=data[:,:,2:])
        traces.append({"path":path.name,"sha256":sha256_file(path),"samples":int((data[:,:,0]>=0).sum())})
        pending.clear()
    refill(np.arange(slots))
    torch.cuda.synchronize(device)
    started=time.monotonic();last_progress=started
    while active.any():
        advance=int(remaining[active].min())
        if max_steps is not None:advance=min(advance,max_steps-iteration)
        if advance<=0:break
        mask=torch.as_tensor(active,device=device)
        for _ in range(advance):
            clock=steps.double()*dt
            refs=reference.motion(clip_ids,clock[:,None]+offsets)
            commands=adapter.act(physics.observations(),refs)
            invalid |= (~torch.isfinite(commands).all(dim=(1,2)) & mask).any()
            invalid |= ((commands[:,2:4]<0).any(dim=(1,2)) & mask).any()
            duration=torch.minimum(torch.full_like(clock,dt),(bank.duration[clip_ids]-clock).clamp_min(1e-9))
            duration=torch.where(mask,duration,dt)
            physics.step(commands,duration)
            steps+=mask
            actual_time=torch.minimum(steps.double()*dt,bank.duration[clip_ids])
            ref_pos,ref_quat,ref_joints,ref_bodies=scoring.get(clip_ids,steps)
            ap=wp.to_torch(physics.data.xpos)[:,body_ids]
            aq=wp.to_torch(physics.data.xquat)[:,physics.root_id][:,[1,2,3,0]]
            values,failed=score(ap,aq,physics.qpos[:,physics.qids],ref_bodies,ref_quat,ref_joints,root_row,body_rows,protocol)
            invalid |= ~(torch.isfinite(values).all(dim=1) | ~mask).all()
            sums+=torch.where(mask[:,None],values,0.)
            first=failed&~seen_failure&mask
            prefix=torch.where(first[:,None],sums,prefix);prefix_count=torch.where(first,steps,prefix_count);failure_time=torch.where(first,actual_time,failure_time)
            seen_failure|=failed&mask
            good=~seen_failure&mask;last_valid=torch.where(good,actual_time,last_valid)
            trace_ids=torch.where(mask,clip_ids,-1)
            if not no_traces:
                pending.append(torch.cat((trace_ids[:,None],actual_time[:,None],physics.qpos.double()),dim=1))
            iteration+=1
            if len(pending)>=128:flush()
            if iteration%200==0:
                physics.check()
                if bool(invalid):raise FloatingPointError("Nonfinite controls/metrics or negative PD gains")
                print(json.dumps({'finished':len(records),'total':count,'steps':iteration,'elapsed_s':time.monotonic()-started}),flush=True)
        physics.check()
        if bool(invalid):raise FloatingPointError('Nonfinite controls or metrics')
        remaining[active]-=advance
        done=np.flatnonzero(active&(remaining==0))
        for slot in done:
            clip=int(clip_ids[slot]);n=int(steps[slot]);failed=bool(seen_failure[slot])
            full=(sums[slot]/n).cpu().tolist()
            early=(prefix[slot]/prefix_count[slot] if failed else sums[slot]/n).cpu().tolist()
            records.append({"clip_id":clip,"name":meta["names"][clip],"samples":n,"success":not failed,
                            "duration_s":float(durations[clip]),"completion":float(last_valid[slot])/durations[clip],
                            "failure_time_s":float(failure_time[slot]) if failed else None,
                            "full":dict(zip(NAMES,full)),"prefix":dict(zip(NAMES,early))})
        active[done]=False;refill(done)
        if time.monotonic()-last_progress>10:
            print(json.dumps({"finished":len(records),"total":count,"steps":iteration,"elapsed_s":time.monotonic()-started}),flush=True);last_progress=time.monotonic()
    flush()
    torch.cuda.synchronize(device)
    result={"status":"unverified_full_rollout" if len(records)==count and max_clips is None else "unverified_smoke",
            "traces_recorded":not no_traces,"vector_steps":iteration,"dataset":dataset,"dataset_identity":{k:v for k,v in meta.items() if k not in ("names","starts","lengths")},
            "protocol_id":protocol["id"],"protocol_sha256":protocol_digest,
            "state_access":runtime["state_access"],"runtime_sha256":runtime_digest,
            "source_sha256":source_files,"adapter_sha256":adapter_files,"converted_weights_sha256":weight_files,"checkpoint_sha256":runtime["weights_sha256"],"engine":physics.metadata(),"clips":sorted(records,key=lambda r:r["clip_id"]),
            "control_period_s":dt,"physics_dt_s":protocol["physics_dt_s"],"scoring_reference_cache":scoring.metadata,"traces":traces,"wall_s":time.monotonic()-started,"prepared":str(prepared)}
    if no_traces:result["status"]="completed_without_traces" if len(records)==count and max_clips is None else "smoke_without_traces"
    result["timing"]={"rollout_s":result["wall_s"],"vector_steps":iteration,"parallel_envs":slots,
      "active_transitions":sum(r["samples"] for r in records),
      "mean_vector_step_ms":result["wall_s"]/max(iteration,1)*1000,
      "transitions_per_s":sum(r["samples"] for r in records)/result["wall_s"],
      "scope":"policy, physics, scoring and slot refill; excludes setup and initial reset; includes traces when enabled"}
    if records:
        result["summary"]={"success_count":sum(r["success"] for r in records),"completion":float(np.mean([r["completion"] for r in records])),
          "full":{k:float(np.mean([r["full"][k] for r in records])) for k in NAMES},
          "prefix":{k:float(np.mean([r["prefix"][k] for r in records])) for k in NAMES}}
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"output":str(output),"summary":result.get("summary")}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ("root","data","output","weights"):p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--dataset",choices=("seed","lafan"),required=True)
    p.add_argument("--cache",type=Path,default=Path.home()/".cache/MotionChallenge/prepared")
    p.add_argument("--slots",type=int,help="Active worlds; default min(dataset motion count, 4096)");p.add_argument("--device",default="cuda:0")
    p.add_argument("--max-clips",type=int);p.add_argument("--max-steps",type=int)
    p.add_argument("--no-traces",action="store_true")
    p.add_argument("--data-format",choices=("original","lerobot-v3"),default="original")
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    evaluate(**vars(a))


if __name__=="__main__":main()
