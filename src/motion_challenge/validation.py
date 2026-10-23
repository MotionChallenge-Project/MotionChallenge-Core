"""Cheap declarative checks; official execution also requires runtime auditing."""

import hashlib
import math
from pathlib import Path, PurePosixPath
import re


def external_output_path(path):
    """Results/logs belong outside every Git checkout, including nested ones."""
    path = Path(path).expanduser().resolve()
    for parent in (path, *path.parents):
        if (parent / '.git').exists():
            raise ValueError(f'output must be outside Git repositories: {parent}')
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confined_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("artifact path must be relative and confined to its root")
    root = Path(root).resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("artifact symlink escapes its root")
    return resolved


def validate_controller(manifest, joint_names):
    required = {"api_version", "id", "robot_profile", "adapter", "artifacts",
                "joint_names", "control_period_s", "reference_offsets_s",
                "state_access", "action_mode", "history_s", "reference_history_s", "training_overlap",
                "license", "runtime_image_digest"}
    if set(manifest) != required:
        raise ValueError(f"controller keys differ: missing={required-set(manifest)}, extra={set(manifest)-required}")
    if manifest["api_version"] != "0.1-draft":
        raise ValueError("unsupported API version")
    if not manifest["id"] or manifest["robot_profile"] != "g1-29dof":
        raise ValueError("v0 requires a named G1 29 DoF controller")
    if manifest["joint_names"] != list(joint_names):
        raise ValueError("joint order must match the canonical robot profile")
    adapter = manifest["adapter"]
    if set(adapter) != {"repository", "commit", "entrypoint"}:
        raise ValueError("adapter requires repository, immutable commit and entrypoint")
    if not adapter["repository"].startswith("https://") or not re.fullmatch(r"[a-f0-9]{40}", adapter["commit"]):
        raise ValueError("adapter repository must use HTTPS and a full commit SHA")
    if not re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", adapter["entrypoint"]):
        raise ValueError("entrypoint must be module:factory")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", manifest["runtime_image_digest"]):
        raise ValueError("runtime image must have an immutable digest")
    for key in ("control_period_s", "history_s", "reference_history_s"):
        value = manifest[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {key}")
    if manifest["control_period_s"] <= 0:
        raise ValueError("control period must be positive")
    offsets = manifest["reference_offsets_s"]
    if not isinstance(offsets, list) or not offsets or any(type(x) not in (int, float) or not math.isfinite(x) or x < -manifest['reference_history_s'] for x in offsets):
        raise ValueError("invalid reference time offsets")
    if any(b <= a for a, b in zip(offsets, offsets[1:])):
        raise ValueError("reference time offsets must strictly increase")
    if manifest["state_access"] != "privileged-state-v0" or manifest["action_mode"] != "joint-pd-feedforward":
        raise ValueError("unsupported state/action contract")
    if manifest["training_overlap"] not in ("seen", "unseen-attested", "unknown"):
        raise ValueError("training overlap must be declared")
    if not isinstance(manifest["license"], str) or not manifest["license"].strip():
        raise ValueError("checkpoint license must be declared separately from source code")
    if not isinstance(manifest["artifacts"], list) or not manifest["artifacts"]:
        raise ValueError("controller needs weights and inference configuration artifacts")
    paths = []
    for artifact in manifest["artifacts"]:
        if set(artifact) != {"path", "url", "sha256", "role"}:
            raise ValueError("artifact keys must be path/url/sha256/role")
        confined_path(".", artifact["path"])
        if not artifact["url"].startswith("https://") or not re.fullmatch(r"[a-f0-9]{64}", artifact["sha256"]):
            raise ValueError("artifact requires HTTPS URL and SHA256")
        paths.append(artifact["path"])
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate artifact path")
    if not {"weights", "inference-config"} <= {x["role"] for x in manifest["artifacts"]}:
        raise ValueError("both weights and inference-config artifacts are required")
    return {"declarative_validation": "passed", "execution_verified": False,
            "max_preview_s": max(offsets), "official_submission_ready": False}


def verify_artifacts(manifest, root):
    for artifact in manifest["artifacts"]:
        path = confined_path(root, artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"artifact hash mismatch: {artifact['path']}")
    return len(manifest["artifacts"])
