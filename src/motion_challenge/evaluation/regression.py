"""Compare complete, matched LAFAN result sets for internal regression review."""
import argparse
import json
import math
from pathlib import Path

from motion_challenge.validation import sha256_file


def normalized(value):
    result = value.get("result", value)
    clips = result["clips"]
    rows = {}
    for clip in clips:
        name = clip.get("name", clip.get("id"))
        if not name or name in rows:
            raise ValueError("Missing or duplicate motion identity")
        metrics = clip.get("metrics", {})
        rows[name] = {
            "duration_s": clip["duration_s"],
            "samples": clip.get("samples", clip.get("sample_count", {}).get("full_motion")),
            "success": clip["success"],
            "completion": clip["completion"],
            "failure_time_s": clip["failure_time_s"],
            "full": clip.get("full", metrics.get("full_motion")),
            "prefix": clip.get("prefix", metrics.get("until_first_failure")),
        }
    return result, rows


def compare(before, after):
    old, a = normalized(before)
    new, b = normalized(after)
    if not a or set(a) != set(b):
        raise ValueError("Comparison requires the same complete motion cohort")
    if not old.get("checkpoint_sha256") or old["checkpoint_sha256"] != new.get("checkpoint_sha256"):
        raise ValueError("Checkpoint identity differs")
    changes = []
    totals = {"full": {}, "prefix": {}}
    for name in sorted(a):
        x, y = a[name], b[name]
        if not math.isclose(x["duration_s"], y["duration_s"], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Motion duration differs: {name}")
        if x["samples"] is None or x["samples"] != y["samples"]:
            raise ValueError(f"Scoring sample count differs: {name}")
        row = {"name": name, "success_before": x["success"], "success_after": y["success"],
               "completion_delta": y["completion"] - x["completion"],
               "failure_time_before_s": x["failure_time_s"], "failure_time_after_s": y["failure_time_s"],
               "metrics": {}}
        for scope in totals:
            keys = set(x[scope]) & set(y[scope])
            if not keys:
                raise ValueError("No comparable metrics")
            row["metrics"][scope] = {}
            for key in sorted(keys):
                xv, yv = float(x[scope][key]), float(y[scope][key])
                if not math.isfinite(xv) or not math.isfinite(yv):
                    raise ValueError("Nonfinite metric")
                row["metrics"][scope][key] = yv - xv
                totals[scope].setdefault(key, []).append((xv, yv))
        changes.append(row)
    macro = {}
    for scope, metrics in totals.items():
        macro[scope] = {}
        for key, values in metrics.items():
            if len(values) != len(a):
                raise ValueError("Inconsistent metric schema across motions")
            x = sum(v[0] for v in values) / len(values)
            y = sum(v[1] for v in values) / len(values)
            macro[scope][key] = {"before": x, "after": y, "delta": y-x,
                                 "max_clip_abs_delta": max(abs(v[1]-v[0]) for v in values)}
    return {"status": "comparison_complete_requires_review", "clips": len(a),
            "checkpoint_sha256": old["checkpoint_sha256"],
            "success_before": sum(x["success"] for x in a.values()),
            "success_after": sum(x["success"] for x in b.values()),
            "success_changed": [x["name"] for x in changes if x["success_before"] != x["success_after"]],
            "completion_before": sum(x["completion"] for x in a.values())/len(a),
            "completion_after": sum(x["completion"] for x in b.values())/len(b),
            "metrics": macro, "per_clip": changes,
            "scope": "Matched checkpoint, motion names, durations and score sample counts. Aggregate similarity is not an implementation correctness certificate."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    report["inputs"] = {"before_sha256": sha256_file(args.before), "after_sha256": sha256_file(args.after)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("metrics", "per_clip")}, indent=2))


if __name__ == "__main__":
    main()
