"""Generic headerless configuration-CSV audit driven entirely by a manifest."""

import csv
import math
from pathlib import Path
import urllib.parse
import urllib.request

from .reference import Motion
from .validation import confined_path, external_output_path, sha256_file


def fetch_dataset(manifest, root):
    """Fetch immutable, hash-locked inputs into an external cache; no retargeting."""
    root = external_output_path(root)
    base = manifest['download_base_url'].rstrip('/')+'/'
    if not base.startswith('https://'):
        raise ValueError('dataset downloads require HTTPS')
    fetched = cached = 0
    for item in manifest['files']:
        destination = confined_path(root,item['path'])
        if destination.is_file() and sha256_file(destination) == item['sha256']:
            cached += 1
            continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary = destination.with_name(destination.name+'.partial')
        url = base+urllib.parse.quote(item['path'],safe='/')
        try:
            with urllib.request.urlopen(url,timeout=60) as response, temporary.open('wb') as stream:
                while block := response.read(1024*1024):
                    stream.write(block)
            if temporary.stat().st_size != item['size'] or sha256_file(temporary) != item['sha256']:
                raise ValueError(f'download integrity mismatch: {item["path"]}')
            temporary.replace(destination)
            fetched += 1
        finally:
            temporary.unlink(missing_ok=True)
    return {'downloaded':fetched,'cached':cached,'root':str(root)}


def load_csv_motion(path, joint_names, fps):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("source FPS must be positive")
    rows = []
    with path.open(newline="") as stream:
        for number, row in enumerate(csv.reader(stream), 1):
            try:
                values = tuple(float(x) for x in row)
            except ValueError as exc:
                raise ValueError(f"invalid numeric row {number} in {path.name}") from exc
            if len(values) != 7 + len(joint_names):
                raise ValueError(f"unexpected column count at row {number} in {path.name}")
            rows.append(values)
    return Motion(tuple(joint_names), tuple(i/fps for i in range(len(rows))),
                  tuple(row[7:] for row in rows), tuple(row[:3] for row in rows),
                  tuple(row[3:7] for row in rows))


def audit_dataset(manifest, root):
    files = manifest["files"]
    if not files or len({x["path"] for x in files}) != len(files):
        raise ValueError("dataset manifest has empty or duplicate files")
    if len(files) != manifest["expected_clips"]:
        raise ValueError("manifest clip count differs from expected count")
    if manifest["format"] != "root_xyz_quat_xyzw_joint_csv":
        raise ValueError("unsupported dataset format")
    clips = []
    for item in files:
        path = confined_path(root, item["path"])
        if path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"dataset integrity mismatch: {item['path']}")
        motion = load_csv_motion(path, manifest["joint_names"], manifest["source_fps"])
        clips.append({"id": item["path"], "frames": len(motion.times_s),
                      "duration_s": motion.duration_s, "sha256": item["sha256"]})
    return {"audit": "passed", "suite": manifest["id"], "clips": clips,
            "clip_count": len(clips), "total_frames": sum(x["frames"] for x in clips),
            "total_duration_s": math.fsum(x["duration_s"] for x in clips),
            "filtered_clips": 0, "dynamic_feasibility_asserted": False}
