"""Independent CPU trace verification for tensor rollouts.

Recomputes reference interpolation, FK, failures and scores without importing
the GPU scorer, GPU reference implementation or any model adapter.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from motion_challenge.resources import resource_root
from motion_challenge.validation import sha256_file
from motion_challenge.simulation.physics import World
from motion_challenge.data.array_reference import slerp_arrays
from motion_challenge.evaluation.batch_metrics import tracking_arrays

_STATE=None
KEYS=("mpjpe_world_m","mpjpe_xy_aligned_m","mpjpe_xy_yaw_aligned_m","joint_mae_rad","anchor_height_error_m","anchor_gravity_z_error")


def initialize(prepared,protocol):
    global _STATE
    root=Path(prepared);meta=json.loads((root/"dataset.json").read_text())
    world=World(resource_root()/protocol["robot_profile"])
    body_ids=[world.model.body(n).id for n in [protocol["failure"]["anchor_body"],*protocol["metric_bodies"]]]
    _STATE=(meta,np.load(root/"poses.npy",mmap_mode="r"),world,body_ids,protocol)


def chunk(task):
    path,expected_hash=task
    if sha256_file(path)!=expected_hash:raise ValueError("Trace hash mismatch")
    meta,poses,world,bids,protocol=_STATE
    trace=np.load(path)
    if trace["clip_id"].shape!=trace["time_s"].shape or trace["qpos"].shape[:-1]!=trace["clip_id"].shape:
        raise ValueError("Trace array shape mismatch")
    if trace["qpos"].shape[-1]!=world.model.nq:raise ValueError("Trace qpos width differs")
    ids=trace["clip_id"].ravel();t=trace["time_s"].ravel()
    if ids.dtype.kind not in "iu" or (ids < -1).any() or (ids >= len(meta["names"])).any():
        raise ValueError("Invalid trace clip identity")
    qpos=trace["qpos"].reshape(len(ids),-1);active=ids>=0
    ids,t,qpos=ids[active],t[active],qpos[active]
    starts=np.asarray(meta["starts"])[ids];lengths=np.asarray(meta["lengths"])[ids];fps=meta["fps"]
    duration=(lengths-1)/fps
    if not np.isfinite(qpos).all() or not np.isfinite(t).all() or (t<=0).any() or (t>duration+1e-8).any():
        raise ValueError("Invalid trace poses or time")
    frame=t*fps;lo=np.minimum(np.floor(frame).astype(int),lengths-2).clip(min=0)
    a,b=poses[starts+lo].astype(float),poses[starts+lo+1].astype(float)
    fraction=frame-lo;rp=a[:,:3]+fraction[:,None]*(b[:,:3]-a[:,:3])
    rq=slerp_arrays(a[:,3:7],b[:,3:7],fraction)
    rj=a[:,7:]+fraction[:,None]*(b[:,7:]-a[:,7:])
    actual=np.empty((len(ids),len(bids),7));desired=np.empty_like(actual)
    for i in range(len(ids)):
        world.data.qpos[:]=qpos[i];mujoco.mj_kinematics(world.model,world.data)
        actual[i,:,:3]=world.data.xpos[bids];actual[i,:,3:]=world.data.xquat[bids][:,[1,2,3,0]]
        world.data.qpos[:]=world.model.qpos0;world.data.qpos[:3]=rp[i]
        world.data.qpos[3:7]=rq[i,[3,0,1,2]];world.data.qpos[world.qadr]=rj[i]
        mujoco.mj_kinematics(world.model,world.data)
        desired[i,:,:3]=world.data.xpos[bids];desired[i,:,3:]=world.data.xquat[bids][:,[1,2,3,0]]
    failures,metrics=tracking_arrays(actual,desired,qpos[:,world.qadr],rj,qpos[:,:3],rp,qpos[:,[4,5,6,3]],rq,protocol)
    return ids,t,np.column_stack([metrics[k] for k in KEYS]),failures.any(1)


def verify(result_path,jobs=8):
    result_path=Path(result_path);result=json.loads(result_path.read_text())
    root=resource_root();protocol_path=root/f"suites/{result['dataset']}/protocol.mjwarp311-no-linear-velocity-v1.json"
    if sha256_file(protocol_path)!=result["protocol_sha256"]:raise ValueError("Protocol hash mismatch")
    protocol=json.loads(protocol_path.read_text())
    if sha256_file(root/protocol["robot_profile"])!=protocol["robot_profile_sha256"]:
        raise ValueError("Robot profile hash mismatch")
    if result["status"]!="unverified_full_rollout" or not result.get("traces_recorded",False):
        raise ValueError("Verification requires a full rollout with traces")
    prepared=Path(result["prepared"]);meta=json.loads((prepared/"dataset.json").read_text())
    if sha256_file(prepared/"poses.npy")!=result["dataset_identity"]["poses_sha256"]:
        raise ValueError("Reference poses changed")
    if any(result["dataset_identity"].get(k)!=v for k,v in meta.items() if k not in ("names","starts","lengths")):
        raise ValueError("Dataset identity mismatch")
    records={r["clip_id"]:r for r in result["clips"]}
    if len(records)!=len(result["clips"]) or set(records)!=set(range(len(meta["names"]))):
        raise ValueError("Result must contain every selected motion exactly once")
    for i,r in records.items():
        duration=(meta["lengths"][i]-1)/meta["fps"]
        if r["name"]!=meta["names"][i] or not math.isclose(duration,r["duration_s"],rel_tol=0,abs_tol=1e-10):
            raise ValueError("Result motion identity or duration differs")
    names=[t["path"] for t in result["traces"]]
    if len(names)!=len(set(names)) or any(Path(n).name!=n for n in names):
        raise ValueError("Duplicate or nonlocal trace path")
    if sum(t["samples"] for t in result["traces"])!=sum(r["samples"] for r in records.values()):
        raise ValueError("Trace sample receipt mismatch")
    accum={i:dict(count=0,sum=np.zeros(len(KEYS)),first=None,prefix=None,prefix_count=None,last_valid=0.) for i in records}
    tasks=[(result_path.parent/t["path"],t["sha256"]) for t in result["traces"]]
    maximum=0.
    with ProcessPoolExecutor(jobs,initializer=initialize,initargs=(prepared,protocol)) as pool:
        for ids,times,values,failed in pool.map(chunk,tasks):
            for i,t,v,f in zip(ids,times,values,failed):
                i=int(i)
                if i not in accum:raise ValueError("Incomplete clip trace cannot verify a published result")
                a=accum[i];a["count"]+=1
                end=records[i]["duration_s"];expected=min(a["count"]*protocol["metric_period_s"],end)
                if not math.isclose(t,expected,abs_tol=1e-8,rel_tol=0):raise ValueError("Missing/duplicate/shifted metric timestamp")
                a["sum"]+=v
                if f and a["first"] is None:
                    a["first"]=float(t);a["prefix"]=a["sum"].copy();a["prefix_count"]=a["count"]
                if a["first"] is None:a["last_valid"]=float(t)
    for i,a in accum.items():
        record=records[i]
        if a["count"]!=record["samples"]:raise ValueError("Sample count differs")
        if record["samples"]!=math.ceil(record["duration_s"]/protocol["metric_period_s"]-1e-10):
            raise ValueError("Trace does not cover exact endpoint")
        if (a["first"] is None)!=record["success"]:raise ValueError("Success classification differs")
        if a["first"] is not None and not math.isclose(a["first"],record["failure_time_s"],abs_tol=1e-8):
            raise ValueError("First failure differs")
        completion=a["last_valid"]/record["duration_s"]
        if not math.isclose(completion,record["completion"],abs_tol=1e-8):raise ValueError("Completion differs")
        full=a["sum"]/a["count"];prefix=a["prefix"]/a["prefix_count"] if a["first"] is not None else full
        for scope,expected in (("full",full),("prefix",prefix)):
            actual=np.array([record[scope][k] for k in KEYS])
            maximum=max(maximum,float(np.max(abs(actual-expected))))
            np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5,err_msg=f"{i} {scope}")
    if not records:raise ValueError("No complete motions")
    expected_summary={"success_count":sum(r["success"] for r in records.values()),
      "completion":float(np.mean([r["completion"] for r in records.values()]))}
    for key,value in expected_summary.items():
        if not math.isclose(result["summary"][key],value,rel_tol=0,abs_tol=1e-12):
            raise ValueError("Summary differs from per-motion records")
    for scope in ("full","prefix"):
        for key in KEYS:
            expected=float(np.mean([r[scope][key] for r in records.values()]))
            if not math.isclose(result["summary"][scope][key],expected,rel_tol=0,abs_tol=1e-12):
                raise ValueError("Summary metric differs from per-motion records")
    report={"verification":"independent_cpu_trace_recomputation_passed","clips":len(records),
            "verifier_sha256":sha256_file(__file__),"max_metric_abs_error":maximum,"result_sha256":sha256_file(result_path),
            "scope":"Trace consistency only; a separate clean rerun is still required"}
    (result_path.parent/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument("result",type=Path);p.add_argument("--jobs",type=int,default=8)
    a=p.parse_args();verify(a.result,a.jobs)


if __name__=="__main__":main()
