"""Command-line entry point for the dependency-free control plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .doctor import DoctorLimits, ResourceDoctor, RequestPolicyError, validate_remote_request
from .target_snapshot import (
    DEFAULT_SNAPSHOT,
    SnapshotValidationError,
    inspect_snapshot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stratafold",
        description="Structural MoE compression without additional quantization",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser(
        "doctor", help="fail closed when host or remote-fetch safety limits are violated"
    )
    doctor.add_argument("--workspace", type=Path, default=Path.cwd())
    doctor.add_argument("--cache", type=Path)
    doctor.add_argument("--url", help="optionally validate a planned metadata URL")
    doctor.add_argument("--pinned-revision", help="required 40-hex revision for a URL")
    doctor.add_argument("--content-length", type=int)
    doctor.add_argument("--aggregate-bytes", type=int, default=0)
    doctor.add_argument("--json", action="store_true", help="emit compact JSON")

    inspect_target = subcommands.add_parser(
        "inspect-target",
        help="validate and summarize the committed target metadata offline",
    )
    inspect_target.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="snapshot directory containing manifest.json",
    )
    inspect_target.add_argument(
        "--json",
        action="store_true",
        help="emit compact JSON",
    )
    return parser


def _doctor(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    cache = (args.cache or workspace / ".cache").resolve()
    report = ResourceDoctor(DoctorLimits()).inspect(workspace=workspace, cache=cache)
    payload = report.to_dict()

    if args.url:
        try:
            request = validate_remote_request(
                args.url,
                pinned_revision=args.pinned_revision,
                content_length=args.content_length,
                aggregate_bytes=args.aggregate_bytes,
            )
            payload["remote_request"] = request.to_dict()
        except RequestPolicyError as exc:
            payload["remote_request"] = {
                "status": "rejected",
                "reason": str(exc),
                "payload_bytes_read": 0,
            }
            payload["ok"] = False

    print(json.dumps(payload, indent=None if args.json else 2, sort_keys=True))
    return 0 if payload["ok"] else 2


def _inspect_target(args: argparse.Namespace) -> int:
    try:
        payload = inspect_snapshot(args.snapshot)
    except SnapshotValidationError as exc:
        print(
            json.dumps(
                {"error": str(exc), "status": "rejected"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, indent=None if args.json else 2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "inspect-target":
        return _inspect_target(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

