#!/usr/bin/env python3
"""Capture an actual deterministic M1 semantic-rejection record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Final, Sequence


SNAPSHOT_RELATIVE: Final = Path(
    "metadata/deepseek-v4-flash-0731/"
    "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
)
SNAPSHOT_NAMES: Final = (
    "manifest.json",
    "config.json",
    "model.safetensors.index.json",
    "LICENSE.target.txt",
    "repository.json",
    "repository.receipt.json",
)
OUTPUT_NAME: Final = "m1_rejection_path.json"
BEFORE_TOKEN: Final = b'"expert_dtype": "fp4"'
AFTER_TOKEN: Final = b'"expert_dtype": "fp3"'
EXPECTED_ERROR: Final = "config.expert_dtype: expected fp4, observed fp3"
MAX_FILE_BYTES: Final = 8 * 1024 * 1024
MAX_AGGREGATE_BYTES: Final = 16 * 1024 * 1024
MAX_PROCESS_BYTES: Final = 64 * 1024
TIMEOUT_SECONDS: Final = 30


class RejectionCaptureError(RuntimeError):
    """The controlled rejection experiment failed closed."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RejectionCaptureError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_object(data: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RejectionCaptureError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise RejectionCaptureError(f"{label} JSON root must be an object")
    return value


def _read_regular(path: Path, *, maximum: int = MAX_FILE_BYTES) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RejectionCaptureError(f"cannot safely open {path.name}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RejectionCaptureError(f"not a regular file: {path.name}")
        if metadata.st_size > maximum:
            raise RejectionCaptureError(f"bounded read exceeded for {path.name}")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(data) > maximum:
        raise RejectionCaptureError(f"file grew beyond bound: {path.name}")
    if len(data) != metadata.st_size:
        raise RejectionCaptureError(f"file changed during bounded read: {path.name}")
    return data


def _atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RejectionCaptureError(f"unsafe output path: {path.name}")
    descriptor = -1
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise RejectionCaptureError(f"cannot write {path.name}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _destination(root: Path, requested: Path) -> Path:
    destination = requested if requested.is_absolute() else root / requested
    if destination.is_symlink():
        raise RejectionCaptureError("output directory must not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or not destination.is_dir():
        raise RejectionCaptureError("output path must be a regular directory")
    return destination


def _copy_snapshot(source: Path, destination: Path) -> dict[str, bytes]:
    copied: dict[str, bytes] = {}
    aggregate = 0
    for name in SNAPSHOT_NAMES:
        data = _read_regular(source / name)
        aggregate += len(data)
        if aggregate > MAX_AGGREGATE_BYTES:
            raise RejectionCaptureError("snapshot exceeded aggregate byte bound")
        target = destination / name
        with target.open("xb") as stream:
            stream.write(data)
        copied[name] = data
    return copied


def _mutate_snapshot(snapshot: Path, copied: dict[str, bytes]) -> dict[str, object]:
    config_before = copied["config.json"]
    if config_before.count(BEFORE_TOKEN) != 1 or AFTER_TOKEN in config_before:
        raise RejectionCaptureError("expert_dtype mutation token is not unique")
    config_after = config_before.replace(BEFORE_TOKEN, AFTER_TOKEN, 1)
    if len(config_after) != len(config_before):
        raise RejectionCaptureError("expert_dtype mutation changed config byte length")
    (snapshot / "config.json").write_bytes(config_after)

    manifest = _load_object(copied["manifest.json"], label="manifest.json")
    files = manifest.get("files")
    if type(files) is not list:
        raise RejectionCaptureError("manifest files must be an array")
    matches = [
        entry
        for entry in files
        if type(entry) is dict and entry.get("path") == "config.json"
    ]
    if len(matches) != 1:
        raise RejectionCaptureError("manifest must list config.json exactly once")
    config_entry = matches[0]
    config_entry["bytes"] = len(config_after)
    config_entry["sha256"] = hashlib.sha256(config_after).hexdigest()
    manifest_after = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (snapshot / "manifest.json").write_bytes(manifest_after)

    return {
        "field": "config.expert_dtype",
        "from": "fp4",
        "to": "fp3",
        "same_length": True,
        "config_before": {
            "bytes": len(config_before),
            "sha256": hashlib.sha256(config_before).hexdigest(),
        },
        "config_after": {
            "bytes": len(config_after),
            "sha256": hashlib.sha256(config_after).hexdigest(),
        },
        "refreshed_manifest": {
            "bytes": len(manifest_after),
            "sha256": hashlib.sha256(manifest_after).hexdigest(),
            "config_entry_bytes": config_entry["bytes"],
            "config_entry_sha256": config_entry["sha256"],
        },
    }


def _environment(root: Path) -> dict[str, str]:
    return {
        "HOME": os.devnull,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(root / "src"),
        "PYTHONUTF8": "1",
        "TZ": "UTC",
    }


def _invoke(root: Path, snapshot: Path) -> subprocess.CompletedProcess[bytes]:
    command = (
        sys.executable,
        "-m",
        "stratafold",
        "inspect-target",
        "--snapshot",
        str(snapshot),
        "--json",
    )
    try:
        return subprocess.run(
            command,
            cwd=root,
            env=_environment(root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RejectionCaptureError(f"inspect-target invocation failed: {exc}") from exc


def _validate_result(result: subprocess.CompletedProcess[bytes]) -> dict[str, object]:
    if len(result.stdout) > MAX_PROCESS_BYTES or len(result.stderr) > MAX_PROCESS_BYTES:
        raise RejectionCaptureError("inspect-target output exceeded bounded limit")
    expected_payload = {"error": EXPECTED_ERROR, "status": "rejected"}
    expected_stderr = (
        json.dumps(expected_payload, sort_keys=True) + "\n"
    ).encode("utf-8")
    if result.returncode != 2:
        raise RejectionCaptureError(
            f"inspect-target returned {result.returncode}, expected 2"
        )
    if result.stdout != b"":
        raise RejectionCaptureError("inspect-target rejection stdout was not empty")
    if result.stderr != expected_stderr:
        raise RejectionCaptureError("inspect-target rejection stderr was not canonical")
    parsed = _load_object(result.stderr, label="inspect-target stderr")
    if parsed != expected_payload:
        raise RejectionCaptureError("inspect-target rejection payload drifted")
    return {
        "command": [
            "python3",
            "-m",
            "stratafold",
            "inspect-target",
            "--snapshot",
            "$MUTATED_SNAPSHOT",
            "--json",
        ],
        "returncode": result.returncode,
        "stdout": {
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "text": "",
        },
        "stderr": {
            "bytes": len(result.stderr),
            "sha256": hashlib.sha256(result.stderr).hexdigest(),
            "json": parsed,
            "canonical_text": result.stderr.decode("utf-8"),
        },
    }


def capture_rejection(root: Path, output_dir: Path) -> str:
    root = root.resolve(strict=True)
    source = root / SNAPSHOT_RELATIVE
    destination = _destination(root, output_dir)
    source_manifest = _read_regular(source / "manifest.json")
    source_config = _read_regular(source / "config.json")

    with tempfile.TemporaryDirectory(prefix="stratafold-m1-rejection-") as directory:
        temporary_snapshot = Path(directory) / "snapshot"
        temporary_snapshot.mkdir()
        copied = _copy_snapshot(source, temporary_snapshot)
        if (
            copied["manifest.json"] != source_manifest
            or copied["config.json"] != source_config
        ):
            raise RejectionCaptureError("source snapshot changed during controlled copy")
        mutation = _mutate_snapshot(temporary_snapshot, copied)
        invocation = _validate_result(_invoke(root, temporary_snapshot))

    record = {
        "schema_version": 1,
        "experiment": "m1-same-length-config-semantic-rejection",
        "source_snapshot": {
            "logical_path": SNAPSHOT_RELATIVE.as_posix(),
            "manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
            "config_sha256": hashlib.sha256(source_config).hexdigest(),
            "copied_files": list(SNAPSHOT_NAMES),
        },
        "mutation": mutation,
        "invocation": invocation,
        "gate_result": {
            "classification": "config-semantic-validation",
            "manifest_integrity_gate": "passed-after-config-entry-refresh",
            "semantic_gate": "rejected",
            "reviewed_identity_gates_reached": False,
            "reason": EXPECTED_ERROR,
        },
        "security": {
            "network_used": False,
            "target_code_imported_or_executed": False,
            "weight_or_lfs_payload_bytes_read": 0,
            "host_paths_recorded": False,
            "timestamps_recorded": False,
        },
    }
    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(destination / OUTPUT_NAME, encoded)
    return OUTPUT_NAME


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="capture the actual deterministic M1 semantic rejection path"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        output = capture_rejection(root, args.output_dir)
    except (OSError, RejectionCaptureError) as exc:
        print(f"capture_m1_rejection: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "output": output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
