"""One command: verified download, local weight conversion, rollout and verification."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

import torch
from motion_challenge.resources import resource_root
from motion_challenge.validation import sha256_file


def original_artifact(path, sha256, url, offline):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name+".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file() and sha256_file(path) == sha256:
            return path
        if offline:
            raise FileNotFoundError(f"No valid cached artifact: {path}")
        if not url or not url.startswith("https://"):
            raise ValueError("A pinned HTTPS artifact URL is required")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                with urllib.request.urlopen(url, timeout=120) as response:
                    shutil.copyfileobj(response, stream, length=8*1024*1024)
            if sha256_file(temporary) != sha256:
                raise ValueError("Downloaded artifact checksum mismatch")
            os.replace(temporary, path)
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)
    return path


def tensor_assets(root, runtime):
    """Verify the model's explicitly declared immutable production assets."""
    verified = {}
    root = Path(root).resolve()
    for name, expected in runtime.get("tensor_assets_sha256", {}).items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing model tensor asset: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Model tensor asset checksum mismatch: {name}")
        verified[name] = actual
    return verified


def prepared_checkpoint(root, runtime, destination, cache, offline):
    """Execute a model-owned source exporter and verify the resulting checkpoint."""
    tensor_assets(root, runtime)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = runtime["weights_sha256"]
    with destination.with_name(destination.name+".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_file() and sha256_file(destination) == expected:
            return destination
        if offline:
            raise FileNotFoundError(f"No valid prepared checkpoint: {destination}; supply --weights or prepare it online")
        sys.path.insert(0, str(root))
        module, function = runtime["prepare_checkpoint"].split(":")
        exporter = getattr(importlib.import_module(module), function)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            generated = Path(temporary)/destination.name
            exporter(generated, cache=cache/"checkpoint-preparation"/runtime["id"])
            if not generated.is_file() or sha256_file(generated) != expected:
                raise ValueError("Prepared checkpoint checksum mismatch")
            os.replace(generated, destination)
    return destination


def converted_weights(root, runtime, source, artifacts, cache, device):
    assets = tensor_assets(root, runtime)
    hook = runtime.get("tensor_conversion")
    if not hook: raise ValueError("This model has no batch weight converter")
    source_files = {p.name: sha256_file(p) for p in root.glob("*.py")}
    shared = resource_root()/"src/motion_challenge/networks"
    if not shared.is_dir():
        from motion_challenge import networks
        shared = Path(networks.__file__).parent
    common = {p.name: sha256_file(p) for p in shared.glob("*.py")}
    identity = {"source_sha256": runtime["weights_sha256"], "converter": hook, "tensor_assets_sha256": assets,
                "model_code_sha256": source_files, "common_code_sha256": common,
                "configuration_sha256": runtime["inference_config_sha256"],
                "tensor_configuration": {k:v for k,v in runtime.items() if k.startswith("tensor_")},
                "artifacts": {k:sha256_file(v) for k,v in artifacts.items()}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    parent = cache/"converted"/runtime["id"]
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent/key
    with (parent/(key+".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = destination/"conversion-receipt.json"
        if receipt.exists():
            recorded = json.loads(receipt.read_text())
            if recorded["identity"] == identity and all(
                    (destination/name).is_file() and sha256_file(destination/name) == digest
                    for name,digest in recorded["files_sha256"].items()):
                return destination
        temporary = Path(tempfile.mkdtemp(prefix=key+".partial-", dir=parent))
        try:
            sys.path.insert(0, str(root))
            module, function = hook.split(":")
            getattr(importlib.import_module(module), function)(root, runtime, source, temporary, artifacts, device)
            hashes = {str(p.relative_to(temporary)):sha256_file(p) for p in temporary.rglob("*") if p.is_file()}
            if not hashes: raise ValueError("Converter produced no artifacts")
            (temporary/"conversion-receipt.json").write_text(json.dumps(
                {"identity":identity, "files_sha256":hashes}, indent=2)+"\n")
            if destination.exists():
                invalid = parent/(key+".invalid-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
                destination.rename(invalid)
            temporary.rename(destination)
        finally:
            if temporary.exists(): shutil.rmtree(temporary)
    return destination


def main(adapter_root=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=adapter_root or Path.cwd())
    p.add_argument("--dataset", choices=("lafan", "seed"), required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--data-format", choices=("original", "lerobot-v3"), default="original")
    p.add_argument("--weights", type=Path, help="Optional local original checkpoint")
    p.add_argument("--artifact", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--cache", type=Path, default=Path.home()/".cache/MotionChallenge")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--slots", type=int, help="Default min(motion count, 4096)")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--verify-jobs", type=int, default=8)
    a = p.parse_args()
    if a.slots is not None and a.slots < 1: p.error("slots must be positive")
    if a.verify_jobs < 1: p.error("verify-jobs must be positive")
    root = a.model.resolve()
    cache = a.cache.expanduser().resolve()
    runtime = json.loads((root/"runtime.mjwarp.json").read_text())
    admission = json.loads((root/"admission.mjwarp.json").read_text())
    tensor_assets(root, runtime)
    if admission["status"] == "excluded": raise ValueError(admission["reason"])
    if not runtime.get("tensor_entrypoint"): raise ValueError("This model's batch adapter is not ready")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.weights:
        source = a.weights.expanduser().resolve()
        if sha256_file(source) != runtime["weights_sha256"]: raise ValueError("Original weight checksum mismatch")
    else:
        name = Path(runtime["weights"]).name
        destination = cache/"original"/runtime["weights_sha256"]/name
        if runtime.get("prepare_checkpoint"):
            source = prepared_checkpoint(root, runtime, destination, cache, a.offline)
        else:
            source = original_artifact(destination, runtime["weights_sha256"],
                                       runtime.get("weights_url"), a.offline)
    overrides = {}
    for value in a.artifact:
        name, separator, value = value.partition("=")
        if not separator or not value or name in overrides: p.error("Use unique --artifact NAME=PATH")
        overrides[name] = Path(value).expanduser().resolve()
    declared = runtime.get("artifacts", {})
    if set(overrides)-set(declared): p.error("Unknown artifact override")
    artifacts = {}
    for name,item in declared.items():
        if name in overrides:
            path = overrides[name]
            if sha256_file(path) != item["sha256"]: raise ValueError(f"Artifact checksum mismatch: {name}")
        else:
            path = original_artifact(cache/"original"/item["sha256"]/Path(item["path"]).name,
                                     item["sha256"], item.get("url"), a.offline)
        artifacts[name] = path
    weights = converted_weights(root, runtime, source, artifacts, cache, a.device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = a.output or cache/"runs"/runtime["id"]/a.dataset/stamp
    from motion_challenge.evaluation.tensor_run import evaluate
    evaluate(root, a.dataset, a.data, cache/"prepared", output, weights, a.slots, a.device,
             data_format=a.data_format)
    from motion_challenge.evaluation.tensor_verify import verify
    verify(output/"result.json", a.verify_jobs)


if __name__ == "__main__":
    main()
