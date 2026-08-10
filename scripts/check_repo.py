#!/usr/bin/env python3
"""Dependency-free M0 claims, license, size, and secret checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
import sys
from typing import Iterable


ALLOWED_TAGS = {"measured", "source-reproduced", "derived", "projected", "unverified"}
REQUIRED_ROOT_FILES = {
    "CITATIONS.bib",
    "CLAIMS.json",
    "LICENSE",
    "LICENSE-DOCS",
    "NOTICE",
    "PROVENANCE.yaml",
    "THIRD_PARTY_NOTICES.md",
}
REQUIRED_PATHS = {
    "docs/BENCHMARK_CONTRACT.md",
    "docs/NAME_REVIEW.md",
    "docs/PRIOR_ART.md",
    "docs/RESEARCH_LEDGER.md",
    "docs/TARGET_LEDGER.md",
    "docs/THREAT_MODEL.md",
    "docs/adr/0001-clean-room.md",
}
BINARY_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx"}
# Detection rules only: findings expose rule labels and paths, never matched values.
CONTENT_GUARDS = {
    "private-key": re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "github-token": re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),
    "github-fine-grained-token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    "huggingface-token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "openai-api-key": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
}
FORBIDDEN_CLAIM_PATTERNS = {
    "target-unquantized": re.compile(
        r"(?:DeepSeek|target|release)[^\n.]{0,50}\b(?:is|was) unquantized\b", re.IGNORECASE
    ),
    "unsupported-sota": re.compile(
        r"\b(?:achieves?|delivers?|establishes?|is)\s+(?:a\s+|new\s+)*"
        r"(?:state[- ]of[- ]the[- ]art|SOTA)\b",
        re.IGNORECASE,
    ),
}


def _git_paths(root: Path, *arguments: str) -> tuple[set[str], str | None]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return set(), f"git path inventory failed: {exc}"
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        return set(), f"git path inventory failed: {detail or completed.returncode}"
    paths = {
        value.decode("utf-8", errors="strict")
        for value in completed.stdout.split(b"\0")
        if value
    }
    return paths, None


def repository_paths(root: Path) -> tuple[set[str], set[str], str | None]:
    tracked, tracked_error = _git_paths(root, "--cached")
    visible, visible_error = _git_paths(
        root, "--cached", "--others", "--exclude-standard"
    )
    return tracked, visible, tracked_error or visible_error


def _filesystem_fallback_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts or path.is_dir():
            continue
        paths.add(relative.as_posix())
    return paths


def public_files(root: Path, visible_paths: set[str]) -> Iterable[Path]:
    for relative_text in sorted(visible_paths):
        path = root / relative_text
        if path.is_symlink() or not path.is_file():
            continue
        yield path


def validate_claims(
    root: Path,
    *,
    allowed_missing_evidence: set[str] | None = None,
    tracked_paths: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads((root / "CLAIMS.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"CLAIMS.json cannot be loaded: {exc}"]
    if payload.get("schema_version") != 1:
        errors.append("CLAIMS.json schema_version must be 1")
    if set(payload.get("allowed_evidence_tags", [])) != ALLOWED_TAGS:
        errors.append("CLAIMS.json allowed_evidence_tags does not match policy")
    claims = payload.get("claims")
    if not isinstance(claims, list):
        return errors + ["CLAIMS.json claims must be a list"]
    seen: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            errors.append("CLAIMS.json claim entries must be objects")
            continue
        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not re.fullmatch(r"SF-C\d{4}", claim_id):
            errors.append(f"invalid claim id: {claim_id!r}")
            continue
        if claim_id in seen:
            errors.append(f"duplicate claim id: {claim_id}")
        seen.add(claim_id)
        if claim.get("tag") not in ALLOWED_TAGS:
            errors.append(f"{claim_id}: invalid evidence tag")
        if not claim.get("statement") or not claim.get("scope"):
            errors.append(f"{claim_id}: statement and scope are required")
        evidence = claim.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"{claim_id}: evidence must be a list")
            continue
        if claim.get("tag") in {"measured", "source-reproduced", "derived"}:
            if not evidence or not claim.get("reproduce"):
                errors.append(f"{claim_id}: tagged claim needs evidence and reproduce command")
        for evidence_path in evidence:
            if not isinstance(evidence_path, str):
                errors.append(f"{claim_id}: unsafe evidence path {evidence_path!r}")
                continue
            pure_path = PurePosixPath(evidence_path)
            if (
                pure_path.is_absolute()
                or ".." in pure_path.parts
                or "." in pure_path.parts
                or "\\" in evidence_path
                or "\x00" in evidence_path
            ):
                errors.append(f"{claim_id}: unsafe evidence path {evidence_path!r}")
                continue
            candidate = root / pure_path
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root.resolve())
            except (FileNotFoundError, OSError, RuntimeError, ValueError):
                resolved = None
            if (
                evidence_path not in (allowed_missing_evidence or set())
                and (resolved is None or not resolved.is_file())
            ):
                errors.append(f"{claim_id}: evidence file does not exist: {evidence_path}")
            elif (
                evidence_path not in (allowed_missing_evidence or set())
                and tracked_paths is not None
                and evidence_path not in tracked_paths
            ):
                errors.append(f"{claim_id}: evidence file is not Git-tracked: {evidence_path}")
    return errors


def validate_repository(
    root: Path, *, allowed_missing_evidence: set[str] | None = None
) -> list[str]:
    tracked_paths, visible_paths, inventory_error = repository_paths(root)
    if inventory_error is not None:
        visible_paths = _filesystem_fallback_paths(root)
    errors = [] if inventory_error is None else [inventory_error]
    errors.extend(
        validate_claims(
            root,
            allowed_missing_evidence=allowed_missing_evidence,
            tracked_paths=tracked_paths if inventory_error is None else None,
        )
    )
    missing = sorted(name for name in REQUIRED_ROOT_FILES if not (root / name).is_file())
    errors.extend(f"missing required file: {name}" for name in missing)
    missing_paths = sorted(name for name in REQUIRED_PATHS if not (root / name).is_file())
    errors.extend(f"missing required path: {name}" for name in missing_paths)
    for relative_text in sorted(visible_paths):
        path = root / relative_text
        relative = Path(relative_text)
        if path.is_symlink():
            errors.append(f"public repository symlink is forbidden: {relative}")
    for path in public_files(root, visible_paths):
        relative = path.relative_to(root)
        if path.suffix.lower() in BINARY_WEIGHT_SUFFIXES:
            errors.append(f"forbidden weight-like file: {relative}")
        size = path.stat().st_size
        if size > 32 * 1024**2:
            errors.append(f"file exceeds 32 MiB safety ceiling: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in CONTENT_GUARDS.items():
            if pattern.search(text):
                errors.append(f"possible {label} in {relative}")
        if relative.name not in {"AGENTS.md", "MISSION.md"}:
            for label, pattern in FORBIDDEN_CLAIM_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"forbidden claim pattern {label} in {relative}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    errors = validate_repository(args.root.resolve())
    result = {"ok": not errors, "errors": errors, "checks": ["claims", "licenses", "secrets", "weights", "file-size", "forbidden-claims"]}
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
