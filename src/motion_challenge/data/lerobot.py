"""Discover motion sources and stream LeRobot v3 poses into the shared CPU cache.

Only NumPy/PyArrow are needed here. Parquet is decoded once per cache identity,
not during simulation. Generic LeRobot datasets need the three named pose fields.
"""

from pathlib import Path
import json

import numpy as np

POSE_KEYS = (
    "observation.state.root_pos",
    "observation.state.root_quat",
    "observation.state.joint_pos",
)


def discover_motion_sources(path):
    """Stable recursive discovery; v3 directories take precedence over same-stem NPZ.

    Hidden/underscore directories are not datasets. A dataset is a leaf: its data
    and metadata folders are never scanned as independent sources.
    """
    path = Path(path).expanduser()
    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"Expected a motion NPZ or LeRobot dataset: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    found, visited = [], set()

    def scan(folder):
        actual = folder.resolve()
        if actual in visited:
            return
        visited.add(actual)
        if (folder / "meta/info.json").is_file():
            found.append(folder)
            return
        if (folder / "poses_0.npy").is_file() and (folder / "index.npz").is_file():
            return  # Runtime cache is not a source dataset.
        children = sorted(folder.iterdir())
        v3_names = {
            p.name for p in children if p.is_dir() and (p / "meta/info.json").is_file()
        }
        for child in children:
            if child.name.startswith((".", "_")):
                continue
            if child.is_dir():
                scan(child)
            elif child.suffix == ".npz" and child.stem not in v3_names:
                found.append(child)

    scan(path)
    if not found:
        raise ValueError(f"No motion NPZ or LeRobot v3 datasets in {path}")
    return sorted(found)


def source_signature(path):
    path = Path(path)
    if path.is_file():
        s = path.stat()
        return [str(path.resolve()), s.st_size, s.st_mtime_ns]
    files = [path / "meta/info.json"]
    files += sorted((path / "meta/episodes").glob("*/*.parquet"))
    files += sorted((path / "data").glob("*/*.parquet"))
    return {
        "root": str(path.resolve()),
        "files": [
            [str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns]
            for p in files
        ],
    }


class LeRobotMotionSource:
    """Small index resident in RAM, pose arrays streamed in bounded row batches."""

    def __init__(self, path, joint_names=None):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.path = Path(path)
        self.info = json.loads((self.path / "meta/info.json").read_text())
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError(f"{path}: expected LeRobot v3.0")
        features = self.info["features"]
        if any(k not in features for k in POSE_KEYS):
            raise ValueError(f"{path}: missing motion features {POSE_KEYS}")
        self.source_joints = tuple(features[POSE_KEYS[2]]["names"])
        self.joint_names = (
            tuple(joint_names) if joint_names is not None else self.source_joints
        )
        if (
            len(set(self.source_joints)) != len(self.source_joints)
            or len(set(self.joint_names)) != len(self.joint_names)
            or set(self.source_joints) != set(self.joint_names)
        ):
            raise ValueError(
                f"{path}: joint_names must uniquely match the robot joints"
            )
        self.permutation = [self.source_joints.index(j) for j in self.joint_names]
        self.quat_names = features[POSE_KEYS[1]].get("names")
        if self.quat_names is None or sorted(self.quat_names) != ["w", "x", "y", "z"]:
            raise ValueError(f"{path}: quaternion component names must declare w/x/y/z")
        self.quat_permutation = [self.quat_names.index(j) for j in ["w", "x", "y", "z"]]
        for k, dim in zip(POSE_KEYS, [3, 4, len(self.source_joints)]):
            if features[k]["shape"] != [dim] or features[k]["dtype"] != "float32":
                raise ValueError(f"{path}: expected float32 {k}[{dim}]")
        self.rate = float(self.info["fps"])
        if not np.isfinite(self.rate) or self.rate <= 0:
            raise ValueError(f"{path}: fps must be finite and positive")
        meta_files = sorted((self.path / "meta/episodes").glob("*/*.parquet"))
        self.data_files = sorted((self.path / "data").glob("*/*.parquet"))
        if not meta_files or not self.data_files:
            raise ValueError(f"{path}: missing data/episode shards")
        required = ["episode_index", "length", "dataset_from_index", "dataset_to_index"]
        tables = []
        for p in meta_files:
            cols = pq.ParquetFile(p).schema_arrow.names
            if not set(required) <= set(cols):
                raise ValueError(f"{p}: invalid episode index")
            selected = required + (
                ["source_motion_name"] if "source_motion_name" in cols else []
            )
            tables.append(pq.read_table(p, columns=selected))
        table = pa.concat_tables(tables).sort_by("episode_index")
        ids = table["episode_index"].to_numpy()
        self.lengths = table["length"].to_numpy().astype(np.int64)
        self.starts = table["dataset_from_index"].to_numpy().astype(np.int64)
        self.ends = table["dataset_to_index"].to_numpy().astype(np.int64)
        if not np.array_equal(ids, np.arange(len(ids))) or (self.lengths < 2).any():
            raise ValueError(f"{path}: invalid episode IDs/lengths")
        if not np.array_equal(
            self.starts, self.lengths.cumsum() - self.lengths
        ) or not np.array_equal(self.ends, self.lengths.cumsum()):
            raise ValueError(f"{path}: noncontiguous episode boundaries")
        self.total_frames = int(self.lengths.sum())
        if (
            len(ids) != self.info["total_episodes"]
            or self.total_frames != self.info["total_frames"]
        ):
            raise ValueError(f"{path}: inconsistent frame/episode totals")
        self.names = (
            tuple(table["source_motion_name"].to_pylist())
            if "source_motion_name" in table.column_names
            else tuple(f"{self.path.name}/episode_{i:06d}" for i in ids)
        )
        if any(not isinstance(n, str) or not n for n in self.names) or len(
            set(self.names)
        ) != len(self.names):
            raise ValueError(f"{path}: duplicate or invalid motion names")
        self.fps = np.full(len(ids), self.rate, dtype=np.float64)

    def iter_packed(self, batch_size=65536):
        import pyarrow.parquet as pq

        cursor = 0
        cols = list(POSE_KEYS) + ["index", "episode_index", "frame_index", "timestamp"]
        for path in self.data_files:
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=batch_size, columns=cols, use_threads=False
            ):
                n = len(batch)
                indices = np.arange(cursor, cursor + n, dtype=np.int64)
                ep = np.searchsorted(self.ends, indices, side="right")
                if n and ep[-1] >= len(self.ends):
                    raise ValueError(f"{path}: extra frames beyond episode metadata")
                frames = indices - self.starts[ep]
                for key, expected in [
                    ("index", indices),
                    ("episode_index", ep),
                    ("frame_index", frames),
                ]:
                    if not np.array_equal(batch.column(key).to_numpy(), expected):
                        raise ValueError(f"{path}: inconsistent {key} at row {cursor}")
                if not np.allclose(
                    batch.column("timestamp").to_numpy(),
                    frames / self.rate,
                    atol=1e-4,
                    rtol=0,
                ):
                    raise ValueError(f"{path}: inconsistent timestamps at row {cursor}")
                packed = np.empty((n, 7 + len(self.joint_names)), dtype=np.float32)
                for key, dest, perm in [
                    (POSE_KEYS[0], slice(0, 3), [0, 1, 2]),
                    (POSE_KEYS[1], slice(3, 7), self.quat_permutation),
                    (POSE_KEYS[2], slice(7, None), self.permutation),
                ]:
                    col = batch.column(key)
                    if col.null_count:
                        raise ValueError(f"{path}: null values in {key}")
                    # flatten respects a sliced list array's offsets; no Python row lists.
                    flat = col.flatten()
                    if flat.null_count or len(flat) != n * len(perm):
                        raise ValueError(f"{path}: invalid vectors in {key}")
                    packed[:, dest] = flat.to_numpy().reshape(n, len(perm))[:, perm]
                if (
                    not np.isfinite(packed).all()
                    or (np.linalg.norm(packed[:, 3:7], axis=1) < 1e-6).any()
                ):
                    raise ValueError(
                        f"{path}: invalid/non-finite poses at row {cursor}"
                    )
                yield cursor, packed
                cursor += n
        if cursor != self.total_frames:
            raise ValueError(
                f"{self.path}: missing frames ({cursor}/{self.total_frames})"
            )

    def write_cache(self, output):
        # Sequential writes keep process RSS bounded; consumers mmap after publish.
        with open(output, "wb") as stream:
            np.lib.format.write_array_header_2_0(
                stream,
                {
                    "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
                    "fortran_order": False,
                    "shape": (self.total_frames, 7 + len(self.joint_names)),
                },
            )
            for _, block in self.iter_packed():
                block.tofile(stream)
        return self.names, self.lengths, self.fps

    def read_arrays(self):
        target = np.empty(
            (self.total_frames, 7 + len(self.joint_names)), dtype=np.float32
        )
        for begin, block in self.iter_packed():
            target[begin : begin + len(block)] = block
        return target, self.fps, self.lengths, self.names
