import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from .checkpoint import State
from .config import load_config
from .pipeline import Pipeline, export_project, validate_project
from .statistics import run_summary


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate creative, resumable 3D scenes with reproducible seeds")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument(
        "--config", required=True, type=Path, help="Category counts or a full scene configuration YAML"
    )
    generate.add_argument(
        "--settings",
        type=Path,
        help="Reuse the generation settings from an existing config (replaces generation settings in --config)",
    )
    generate.add_argument("--name", help="Readable run name (a numbered suffix prevents overwrites)")
    for command in ("resume", "status"):
        p = sub.add_parser(command)
        p.add_argument("--run-id", required=True)
        p.add_argument("--root", type=Path, default=Path(os.getenv("SCENE_GENERATOR_HOME", "runs")))
    for command in ("validate", "export"):
        p = sub.add_parser(command)
        p.add_argument("--scene", required=True, type=Path)
        if command == "export":
            p.add_argument("--format", choices=["glb", "obj", "ply"], default="glb")
    args = parser.parse_args(argv)
    pipeline = None
    try:
        if args.command == "generate":
            config = load_config(args.config, settings=args.settings)
            if args.name:
                config.generation.output.name = args.name
            pipeline = Pipeline.create(config)
            result = pipeline.run()
        elif args.command in {"resume", "status"}:
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,149}", args.run_id):
                raise ValueError("invalid run id")
            path = args.root.resolve() / args.run_id
            if not (path / "state.sqlite").is_file():
                raise ValueError("run not found; use --root for a custom output directory")
            if args.command == "resume":
                pipeline = Pipeline(path)
                result = pipeline.run()
            else:
                state = State(path / "state.sqlite")
                try:
                    result = run_summary(state, args.run_id)
                finally:
                    state.close()
        elif args.command == "validate":
            report = validate_project(args.scene)
            result = {"valid": report.valid, **report.model_dump(mode="json")}
            print(json.dumps(result, indent=2))
            return 0 if report.valid else 2
        else:
            result = export_project(args.scene, args.format)
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; checkpoints saved. Resume using the run ID printed above.", file=sys.stderr)
        return 130
    except ValidationError as exc:
        errors = [{"path": list(e["loc"]), "type": e["type"]} for e in exc.errors()]
        print("Invalid configuration or contract: " + json.dumps(errors), file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, OSError) as exc:
        # Only our controlled errors are useful here; never dump provider responses.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if pipeline:
            pipeline.close()


if __name__ == "__main__":
    sys.exit(main())
