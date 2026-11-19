"""Prepare the canonical evaluation cohort from a local LeRobot v3 dataset."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from motion_challenge.data.lerobot import LeRobotMotionSource
from motion_challenge.data.selection import load_selection, seed_path_from_v3_name
from motion_challenge.resources import resource_root
from motion_challenge.validation import sha256_file


def prepare_v3(dataset, data, cache):
    root = resource_root()
    data, cache = Path(data).expanduser().resolve(), Path(cache).expanduser().resolve()
    joints = json.loads((root/"robots/g1-29dof-mjwarp-v1.json").read_text())["joint_names"]
    source = LeRobotMotionSource(data, joints)
    if dataset == "seed":
        config, rows = load_selection(root/"suites/seed/filter.json")
        names = [seed_path_from_v3_name(n) for n in source.names]
        expected = [r["path"] for r in rows]
        frame_counts = {r["path"]: r["frames"] for r in rows}
        if any(int(n) != frame_counts.get(name) for name, n in zip(names, source.lengths)):
            raise ValueError("SEED v3 motion lengths differ from the selected cohort")
        if source.rate != 120.: raise ValueError("SEED requires 120 Hz source timing")
        selection_hash = config["selection"]["sha256"]
    elif dataset == "lafan":
        manifest = root/"suites/lafan/dataset.json"
        config = json.loads(manifest.read_text())
        names = [f"g1/{n}.csv" for n in source.names]
        expected = [r["path"] for r in config["files"]]
        if source.rate != config["source_fps"]: raise ValueError("LAFAN source timing differs")
        selection_hash = sha256_file(manifest)
    else:
        raise ValueError("Choose seed or lafan")
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError("LeRobot source must contain the complete selected cohort exactly once")
    print(json.dumps({"phase": "hashing_v3_source", "motions": len(names), "frames": source.total_frames}), flush=True)
    files = [data/"meta/info.json"] + sorted((data/"meta/episodes").glob("*/*.parquet")) + source.data_files
    source_hashes = {str(p.relative_to(data)): sha256_file(p) for p in files}
    identity = {"dataset": dataset, "source_format": "lerobot-v3", "pose_format": "xyz-xyzw-joints-f32",
                "source_manifest_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
                "selection_sha256": selection_hash, "converter_sha256": sha256_file(__file__),
                "reader_sha256": sha256_file(Path(__file__).with_name("lerobot.py"))}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    dest = cache/key
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/"build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = dest/"dataset.json"
        if receipt.exists():
            meta = json.loads(receipt.read_text())
            if sha256_file(dest/"poses.npy") != meta["poses_sha256"]:
                raise ValueError("Prepared pose cache checksum differs")
            return dest
        lookup = {name: i for i, name in enumerate(names)}
        permutation = np.array([lookup[n] for n in expected])
        inverse = np.argsort(permutation)
        lengths = source.lengths[permutation]
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        poses = np.lib.format.open_memmap(dest/"poses.partial.npy", mode="w+", dtype=np.float32,
                                         shape=(source.total_frames, 7+len(joints)))
        ordered = names == expected
        for offset, block in source.iter_packed():
            block[:,3:7] = block[:,[4,5,6,3]]
            if np.any(np.abs(np.linalg.norm(block[:,3:7],axis=1)-1) > 1e-3):
                raise ValueError("Nonunit source quaternion")
            if ordered:
                poses[offset:offset+len(block)] = block
            else:
                end, cursor = offset+len(block), offset
                while cursor < end:
                    episode = int(np.searchsorted(source.ends, cursor, side="right"))
                    stop = min(end, int(source.ends[episode]))
                    target = int(starts[inverse[episode]]) + cursor-int(source.starts[episode])
                    poses[target:target+stop-cursor] = block[cursor-offset:stop-offset]
                    cursor = stop
            if offset//65536 % 100 == 0:
                print(json.dumps({"phase": "preparing_v3", "frames": offset+len(block), "total": source.total_frames}), flush=True)
        poses.flush()
        del poses
        os.replace(dest/"poses.partial.npy", dest/"poses.npy")
        meta = {**identity, "key": key, "joint_names": joints, "fps": source.rate, "names": expected,
                "starts": starts.tolist(), "lengths": lengths.tolist(), "poses_sha256": sha256_file(dest/"poses.npy")}
        (dest/"source-manifest.json").write_text(json.dumps(source_hashes, indent=2)+"\n")
        temp = receipt.with_suffix(".tmp")
        temp.write_text(json.dumps(meta, separators=(",", ":"))+"\n")
        os.replace(temp, receipt)
        return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=("seed", "lafan"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    print(prepare_v3(a.dataset, a.data, a.cache), flush=True)
