"""Compare independently verified runs of the same evaluation pipeline."""
import argparse
import hashlib
import json
from pathlib import Path

ROLLOUT_FILES = (
    "sdk.py", "simulation/warp_physics.py", "simulation/physics.py",
    "simulation/tensor_reference.py", "simulation/tensor_math.py",
    "simulation/scoring_cache.py", "evaluation/tensor_run.py",
    "evaluation/tensor_metrics.py",
)


def load_verified(path):
    path = Path(path)
    if path.is_dir():
        path /= "result.json"
    raw = path.read_bytes()
    result = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    verification = json.loads(path.with_name("verification.json").read_text())
    if verification.get("verification") != "independent_cpu_trace_recomputation_passed":
        raise ValueError("Independent trace verification is required")
    if verification["result_sha256"] != digest:
        raise ValueError("Verification receipt does not bind this result")
    if result["status"] != "unverified_full_rollout" or not result["traces_recorded"]:
        raise ValueError("A complete traced rollout is required")
    if verification["clips"] != len(result["clips"]):
        raise ValueError("Incomplete verified cohort")
    return result, digest


def compare(first, second):
    a, a_hash = load_verified(first)
    b, b_hash = load_verified(second)
    for key in ("dataset", "protocol_id", "protocol_sha256", "state_access",
                "runtime_sha256", "checkpoint_sha256", "engine", "control_period_s", "physics_dt_s"):
        if a[key] != b[key]:
            raise ValueError(f"Reproduction identity differs: {key}")
    if a["dataset_identity"]["poses_sha256"] != b["dataset_identity"]["poses_sha256"]:
        raise ValueError("Prepared motion data differs")
    for name in ROLLOUT_FILES:
        if a["source_sha256"][name] != b["source_sha256"][name]:
            raise ValueError(f"Rollout source differs: {name}")
    if a["adapter_sha256"] != b["adapter_sha256"]:
        raise ValueError("Model repository Python sources differ")
    # Conversion receipts include timings and paths; tensor/graph artifacts must match.
    def artifacts(result):
        return {k: v for k, v in result["converted_weights_sha256"].items()
                if not any(x in k.lower() for x in ("parity", "receipt", "validation", "benchmark"))}
    if artifacts(a) != artifacts(b):
        raise ValueError("Converted model artifacts differ")
    left = {x["name"]: x for x in a["clips"]}
    right = {x["name"]: x for x in b["clips"]}
    if len(left) != len(a["clips"]) or set(left) != set(right):
        raise ValueError("Motion cohort differs or contains duplicates")
    changed = []
    maximum = {"completion": 0.}
    for name, x in left.items():
        y = right[name]
        if x["samples"] != y["samples"] or x["duration_s"] != y["duration_s"]:
            raise ValueError(f"Motion clock differs: {name}")
        if x["success"] != y["success"]:
            changed.append(name)
        maximum["completion"] = max(maximum["completion"], abs(x["completion"] - y["completion"]))
        for span in ("full", "prefix"):
            for metric in x[span]:
                key = f"{span}.{metric}"
                maximum[key] = max(maximum.get(key, 0.), abs(x[span][metric] - y[span][metric]))
    delta = {"success_count": b["summary"]["success_count"] - a["summary"]["success_count"],
             "completion": b["summary"]["completion"] - a["summary"]["completion"]}
    for span in ("full", "prefix"):
        delta[span] = {key: b["summary"][span][key] - value for key, value in a["summary"][span].items()}
    return {"status": "pipeline_reproduction_passed", "first_result_sha256": a_hash,
            "second_result_sha256": b_hash, "clips": len(left), "same_pipeline_identity": True,
            "both_independently_trace_verified": True, "summary_delta_second_minus_first": delta,
            "max_per_motion_abs_difference": maximum, "changed_success_motions": changed,
            "numerically_identical_metrics": all(value == 0. for value in maximum.values()) and not changed,
            "scope": "Complete pipeline reproduction with matching data, model, physics and rollout code; numerical differences are reported separately."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("first", type=Path)
    p.add_argument("second", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = compare(args.first, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
