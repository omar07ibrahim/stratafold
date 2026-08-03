"""Command-line entry point for the dependency-free control plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .doctor import DoctorLimits, ResourceDoctor, RequestPolicyError, validate_remote_request


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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
