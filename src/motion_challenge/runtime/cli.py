import argparse
import json
from pathlib import Path

from motion_challenge.data.dataset import audit_dataset, fetch_dataset
from motion_challenge.validation import external_output_path, validate_controller, verify_artifacts


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description="Motion Challenge dataset and controller verification tools")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-dataset")
    fetch.add_argument("--manifest", required=True)
    fetch.add_argument("--root", required=True)
    audit = commands.add_parser("audit-dataset")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--root", required=True)
    audit.add_argument("--output")
    validate = commands.add_parser("validate-controller")
    validate.add_argument("manifest")
    validate.add_argument("--robot", required=True)
    validate.add_argument("--artifacts-root")
    suite = commands.add_parser("inspect-suite")
    suite.add_argument("manifest")
    commands.add_parser("paths", help="Locate the packaged LAFAN and G1 protocol assets")
    scaffold = commands.add_parser("init-adapter", help="Create a separate SDK consumer project")
    scaffold.add_argument("--output", required=True)
    scaffold.add_argument("--core-path", required=True)
    args = parser.parse_args()
    try:
        if getattr(args, "output", None) and args.command != "init-adapter":
            args.output = external_output_path(args.output)
        if args.command == "paths":
            from motion_challenge.resources import lafan_paths
            result = {k:str(v) for k,v in lafan_paths().items()}
        elif args.command == "init-adapter":
            from motion_challenge.runtime.scaffold import create_project
            result = create_project(args.output,args.core_path)
        elif args.command == "fetch-dataset":
            result = fetch_dataset(read(args.manifest), args.root)
        elif args.command == "audit-dataset":
            result = audit_dataset(read(args.manifest), args.root)
        elif args.command == "validate-controller":
            manifest = read(args.manifest)
            result = validate_controller(manifest, read(args.robot)["joint_names"])
            if args.artifacts_root:
                result["artifacts_verified"] = verify_artifacts(manifest, args.artifacts_root)
        else:
            manifest = read(args.manifest)
            result = {"id": manifest["id"], "robot": manifest["robot_profile"],
                      "expected_clips": manifest["expected_clips"],
                      "status": "dataset audit available; official evaluation not implemented"}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"validation failed: {exc}\n")
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if getattr(args, "output", None) and args.command != "init-adapter":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(json.dumps({k: v for k, v in result.items() if k != "clips"}, indent=2))
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
