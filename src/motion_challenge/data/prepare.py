"""Prepare immutable, local-only motion caches from upstream downloads.

No network download is performed. A cache binds the original archive/manifest,
the unified selection and the converter source; never just path or mtime.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import tarfile
import time

import numpy as np
from scipy.spatial.transform import Rotation
from motion_challenge.resources import resource_root
from motion_challenge.validation import sha256_file
from motion_challenge.data.selection import load_selection


def decode_seed(stream, names):
    text=io.StringIO(stream.read().decode("utf-8"))
    header=next(csv.reader([text.readline()]))
    expected=["Frame",*("root_translate"+x for x in "XYZ"),*("root_rotate"+x for x in "XYZ"),*(n+"_dof" for n in names)]
    if len(header)!=len(set(header)) or set(header)!=set(expected):
        raise ValueError("Unexpected BONES-SEED CSV columns")
    x=np.loadtxt(text,delimiter=",",ndmin=2)
    x=x[:,[header.index(n) for n in expected]]
    if len(x)<2 or not np.isfinite(x).all() or not np.array_equal(x[:,0],np.arange(len(x))):
        raise ValueError("Invalid motion samples or frame indices")
    result=np.empty((len(x),7+len(names)),np.float32)
    result[:,:3]=x[:,1:4]/100
    result[:,3:7]=Rotation.from_euler("xyz",x[:,4:7],degrees=True).as_quat()
    result[:,7:]=np.deg2rad(x[:,7:])
    return result


def _decode_seed_bytes(payload, names):
    return decode_seed(io.BytesIO(payload), names)


def iter_seed_motions(archive, selected, joints, workers):
    """Stream one archive with bounded parallel parsing; yield in archive order."""
    if workers < 1:
        raise ValueError("Conversion workers must be positive")
    with tarfile.open(archive, "r|gz", bufsize=1024 * 1024) as tar:
        if workers == 1:
            for item in tar:
                if item.name in selected:
                    if not item.isfile():
                        raise ValueError("Selected archive entry is not a regular file")
                    yield item.name, decode_seed(tar.extractfile(item), joints)
            return
        # Spawn keeps workers independent of any CUDA context in the evaluator.
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = deque()
            for item in tar:
                if item.name not in selected:
                    continue
                if not item.isfile():
                    raise ValueError("Selected archive entry is not a regular file")
                payload = tar.extractfile(item).read()
                pending.append((item.name, pool.submit(_decode_seed_bytes, payload, joints)))
                if len(pending) >= workers * 3:
                    name, future = pending.popleft()
                    yield name, future.result()
            while pending:
                name, future = pending.popleft()
                yield name, future.result()


def prepare(dataset, data, cache, workers=None):
    started = time.perf_counter()
    if workers is None:
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
        workers = min(8, available)
    if workers < 1:
        raise ValueError("Conversion workers must be positive")
    root=resource_root();data=Path(data).expanduser().resolve();cache=Path(cache).expanduser().resolve()
    joints=json.loads((root/"robots/g1-29dof-mjwarp-v1.json").read_text())["joint_names"]
    identity={"dataset":dataset,"converter_sha256":sha256_file(__file__),"pose_format":"xyz-xyzw-joints-f32"}
    if dataset=="seed":
        config,rows=load_selection(root/"suites/seed/filter.json")
        archive=data if data.is_file() else data/"g1.tar.gz"
        if not archive.is_file():
            raise FileNotFoundError(f"Place the original BONES-SEED g1.tar.gz at {archive}; download it from https://huggingface.co/datasets/bones-studio/seed")
        digest=sha256_file(archive)
        if digest!=config["source_archive"]["sha256"]:
            raise ValueError("SEED archive revision differs from the frozen benchmark")
        identity.update(source_sha256=digest,selection_sha256=config["selection"]["sha256"])
        fps=120.
    elif dataset=="lafan":
        config=json.loads((root/"suites/lafan/dataset.json").read_text())
        files=config["files"];rows=[]
        for item in files:
            p=data/item["path"]
            if not p.is_file():raise FileNotFoundError(f"Missing {p}; download g1 CSVs from https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset")
            if sha256_file(p)!=item["sha256"]:raise ValueError(f"Source hash mismatch: {p}")
            with p.open("rb") as f:n=sum(1 for line in f if line.strip())
            rows.append({"path":item["path"],"frames":n})
        fps=config["source_fps"]
        identity.update(source_manifest_sha256=sha256_file(root/"suites/lafan/dataset.json"))
    else:raise ValueError("Choose seed or lafan")
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    dest=cache/key;dest.mkdir(parents=True,exist_ok=True)
    # Linux advisory lock serializes cache construction across simultaneous model runs.
    import fcntl
    with (dest/"build.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=dest/"dataset.json"
        if receipt.exists():
            meta=json.loads(receipt.read_text())
            if sha256_file(dest/"poses.npy")!=meta["poses_sha256"]:
                raise ValueError("Prepared pose cache failed integrity check")
            return dest
        source_validation_seconds = time.perf_counter() - started
        conversion_started = time.perf_counter()
        lengths=np.array([x["frames"] for x in rows],dtype=np.int64)
        starts=np.r_[0,np.cumsum(lengths)[:-1]]
        poses=np.lib.format.open_memmap(dest/"poses.partial.npy",mode="w+",dtype=np.float32,shape=(int(lengths.sum()),7+len(joints)))
        slots={x["path"]:i for i,x in enumerate(rows)};seen=set();prepared_frames=0
        def accept(name,values):
            nonlocal prepared_frames
            i=slots[name]
            if name in seen or len(values)!=lengths[i] or not np.isfinite(values).all():
                raise ValueError(f"Duplicate, invalid or changed motion: {name}")
            norms=np.linalg.norm(values[:,3:7],axis=1)
            if np.any(abs(norms-1)>1e-3):raise ValueError(f"Invalid quaternion: {name}")
            poses[starts[i]:starts[i]+lengths[i]]=values;seen.add(name)
            prepared_frames += len(values)
            if len(seen)%1000==0:
                elapsed=time.perf_counter()-conversion_started
                print(json.dumps({"prepared":len(seen),"total":len(rows),"frames":prepared_frames,
                    "elapsed_s":round(elapsed,2),"estimated_remaining_s":round(elapsed*(int(lengths.sum())-prepared_frames)/prepared_frames,2)}),flush=True)
        if dataset=="seed":
            for name, values in iter_seed_motions(archive, slots, joints, workers):
                accept(name, values)
        else:
            for row in rows:
                accept(row["path"],np.loadtxt(data/row["path"],delimiter=",",ndmin=2).astype(np.float32))
        if len(seen)!=len(rows):raise ValueError("Missing selected archive motions")
        poses.flush();del poses
        os.replace(dest/"poses.partial.npy",dest/"poses.npy")
        metadata={**identity,"key":key,"joint_names":joints,"fps":fps,"names":[x["path"] for x in rows],
                  "starts":starts.tolist(),"lengths":lengths.tolist(),"poses_sha256":sha256_file(dest/"poses.npy")}
        metadata["preparation"] = {"workers": workers if dataset == "seed" else 1,
            "source_validation_seconds": source_validation_seconds,
            "conversion_and_output_hash_seconds": time.perf_counter() - conversion_started,
            "total_seconds": time.perf_counter() - started}
        tmp=receipt.with_suffix(".tmp");tmp.write_text(json.dumps(metadata,separators=(",",":"))+"\n");os.replace(tmp,receipt)
        return dest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True,choices=("seed","lafan"))
    p.add_argument("--data",required=True,type=Path)
    p.add_argument("--cache",type=Path,default=Path.home()/".cache/MotionChallenge/prepared")
    p.add_argument("--workers",type=int,default=None)
    a=p.parse_args();print(prepare(a.dataset,a.data,a.cache,a.workers))


if __name__=="__main__":main()
