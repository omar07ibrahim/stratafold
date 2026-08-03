#!/usr/bin/env python3
"""Produce raw, machine-readable evidence for the M0 safety gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

from stratafold.doctor import (
    DoctorLimits,
    RequestPolicyError,
    ResourceDoctor,
    validate_remote_request,
)

from check_repo import validate_repository


REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
FORBIDDEN_URL = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/"
    f"{REVISION}/model-00001-of-00048.safetensors"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    report = ResourceDoctor(DoctorLimits()).inspect(workspace=root, cache=root / ".cache")

    opener_calls = 0

    def forbidden_opener(_request: object) -> object:
        nonlocal opener_calls
        opener_calls += 1
        raise AssertionError("forbidden request reached opener")

    rejection = None
    try:
        request = validate_remote_request(
            FORBIDDEN_URL,
            pinned_revision=REVISION,
            content_length=1,
        )
        forbidden_opener(request)
    except RequestPolicyError as exc:
        rejection = str(exc)

    try:
        output_relative = args.output.resolve().relative_to(root).as_posix()
    except ValueError:
        raise SystemExit("--output must be inside the repository")
    repo_errors = validate_repository(root, allowed_missing_evidence={output_relative})
    ok = report.ok and rejection is not None and opener_calls == 0 and not repo_errors
    payload = {
        "schema_version": 1,
        "evidence_tag": "measured",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "command": "make evidence-m0",
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": report.cpu_count,
            "memory_available_bytes": report.memory_available_bytes,
        },
        "resource_doctor": report.to_dict(),
        "forbidden_request": {
            "kind": "simulated-safetensors",
            "url_logged": False,
            "rejected": rejection is not None,
            "reason": rejection,
            "opener_calls": opener_calls,
            "payload_bytes_read": 0,
        },
        "repository_checks": {"ok": not repo_errors, "errors": repo_errors},
        "target_weights": {
            "repository_weight_files_detected": 0,
            "workflow_declares_downloaded": False,
            "workflow_declares_executed": False,
            "independent_hostwide_network_audit": False,
        },
        "ok": ok,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": ok, "output": str(args.output), "opener_calls": opener_calls}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
