#!/usr/bin/env python3
"""Verify the captured API projection against repository.json, entirely offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from stratafold.repository_projection import (
    ProjectionVerificationError,
    TARGET_REVISION,
    verify_projection_files,
)


DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "metadata"
    / "deepseek-v4-flash-0731"
    / TARGET_REVISION
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify repository.receipt.json and reproduce repository.json "
            "without network access."
        )
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=DEFAULT_ROOT / "repository.receipt.json",
        help="captured projection receipt",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=DEFAULT_ROOT / "repository.json",
        help="committed deterministic projection",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit compact deterministic JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_projection_files(args.receipt, args.repository)
    except ProjectionVerificationError as exc:
        error = {
            "error": str(exc),
            "status": "rejected",
        }
        print(
            json.dumps(error, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
        )
        return 2

    if args.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        projection = report["repository_projection"]
        projection_input = report["projection_input"]
        print("repository projection: verified")
        print(f"  input sha256: {projection_input['sha256']}")
        print(
            "  repository.json: "
            f"{projection['bytes']} bytes, sha256 {projection['sha256']}"
        )
        print("  network access: none")
        print("  weight/LFS payload bytes read: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
