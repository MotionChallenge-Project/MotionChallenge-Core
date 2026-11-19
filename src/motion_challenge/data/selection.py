"""Versioned local dataset selection; no downloads or motion data redistribution."""
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def seed_path_from_v3_name(name):
    """Recover a canonical source path from date__clip__local-hash IDs."""
    match = re.fullmatch(r"(\d{6})__(.+)__([0-9a-f]{12})", name)
    if not match:
        raise ValueError(f"Unsupported SEED v3 source identity: {name}")
    date, stem, _ = match.groups()
    if "/" in stem or "\\" in stem or stem in (".", ".."):
        raise ValueError("Invalid source clip name")
    return f"g1/csv/{date}/{stem}.csv"


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
        raise ValueError(f"Noncanonical source path: {name}")
    return name


def load_selection(filter_path):
    filter_path = Path(filter_path)
    config = json.loads(filter_path.read_text())
    if config.get("schema") != "motionchallenge/dataset-selection-v1":
        raise ValueError("Unknown dataset selection schema")
    ledger = filter_path.parent / safe_relative(config["selection"]["path"])
    if hashlib.sha256(ledger.read_bytes()).hexdigest() != config["selection"]["sha256"]:
        raise ValueError("Selection ledger checksum mismatch")
    with gzip.open(ledger, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    paths = [safe_relative(row["path"]) for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate selected source paths")
    if len(rows) != config["expected_episodes"]:
        raise ValueError("Selection episode count mismatch")
    if sum(row["frames"] for row in rows) != config["expected_frames"]:
        raise ValueError("Selection frame count mismatch")
    excluded = [safe_relative(row["path"]) for row in config["exclude"]]
    if len(excluded) != len(set(excluded)) or set(excluded) & set(paths):
        raise ValueError("Duplicate exclusions or included/excluded overlap")
    return config, rows


def select_source_files(filter_path, available):
    """Apply an exact frozen cohort; missing clips fail instead of shrinking it."""
    config, rows = load_selection(filter_path)
    available = list(available)
    for path in available:
        safe_relative(path)
    if len(available) != len(set(available)):
        raise ValueError("Duplicate source paths")
    known = set(available)
    missing = [row["path"] for row in rows if row["path"] not in known]
    if missing:
        raise ValueError(f"Missing {len(missing)} selected motions; first: {missing[:5]}")
    selected = {row["path"] for row in rows}
    excluded = {row["path"] for row in config["exclude"]}
    return rows, {
        "selection_id": config["id"],
        "filter_sha256": hashlib.sha256(Path(filter_path).read_bytes()).hexdigest(),
        "selected_episodes": len(rows),
        "selected_frames": sum(row["frames"] for row in rows),
        "explicitly_excluded": len(known & excluded),
        "outside_frozen_cohort": len(known - selected - excluded),
    }
