#!/usr/bin/env python3
"""Capture deterministic raw evidence from the offline M1 target inspector."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import threading
from typing import Callable, Final, Sequence


TARGET_REPOSITORY: Final = "deepseek-ai/DeepSeek-V4-Flash-0731"
TARGET_REVISION: Final = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
COMPACT_COMMAND: Final = (
    "python3",
    "-m",
    "stratafold",
    "inspect-target",
    "--json",
)
PRETTY_COMMAND: Final = ("python3", "-m", "stratafold", "inspect-target")
TRANSCRIPT_COMMAND: Final = (
    b"$ PYTHONPATH=src python3 -m stratafold inspect-target\n"
)
JSON_NAME: Final = "m1_target_genome.json"
TEXT_NAME: Final = "m1_target_genome.txt"
OUTPUT_NAMES: Final = (JSON_NAME, TEXT_NAME)
MAX_STREAM_BYTES: Final = 256 * 1024
COMMAND_TIMEOUT_SECONDS: Final = 30
_READ_CHUNK_BYTES: Final = 8192


class CaptureError(RuntimeError):
    """The CLI surfaces or requested evidence operation failed closed."""


@dataclass(frozen=True)
class Invocation:
    returncode: int
    stdout: bytes
    stderr: bytes


Runner = Callable[[tuple[str, ...], Path], Invocation]


def _controlled_environment(root: Path) -> dict[str, str]:
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


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return


def _invoke_cli(command: tuple[str, ...], root: Path) -> Invocation:
    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=_controlled_environment(root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise CaptureError(f"could not start {command[-1]!r} surface: {exc}") from exc

    if process.stdout is None or process.stderr is None:
        _stop_process(process)
        process.wait()
        raise CaptureError("subprocess pipes were not created")

    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    overflow: list[str] = []
    read_errors: list[str] = []
    stop_lock = threading.Lock()

    def stop_once() -> None:
        with stop_lock:
            _stop_process(process)

    def read_stream(label: str, stream: object) -> None:
        try:
            while True:
                chunk = stream.read(_READ_CHUNK_BYTES)  # type: ignore[attr-defined]
                if not chunk:
                    return
                if len(buffers[label]) + len(chunk) > MAX_STREAM_BYTES:
                    overflow.append(label)
                    stop_once()
                    return
                buffers[label].extend(chunk)
        except OSError as exc:
            read_errors.append(f"{label}: {exc}")
            stop_once()

    readers = [
        threading.Thread(
            target=read_stream,
            args=("stdout", process.stdout),
            name="capture-m1-stdout",
        ),
        threading.Thread(
            target=read_stream,
            args=("stderr", process.stderr),
            name="capture-m1-stderr",
        ),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        process.wait(timeout=COMMAND_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        stop_once()
        process.wait(timeout=5)
    finally:
        for reader in readers:
            reader.join(timeout=5)
        process.stdout.close()
        process.stderr.close()

    if any(reader.is_alive() for reader in readers):
        stop_once()
        raise CaptureError("subprocess output readers did not terminate")
    if timed_out:
        raise CaptureError(
            f"CLI surface exceeded {COMMAND_TIMEOUT_SECONDS}-second timeout"
        )
    if overflow:
        raise CaptureError(
            f"CLI {overflow[0]} exceeded {MAX_STREAM_BYTES}-byte limit"
        )
    if read_errors:
        raise CaptureError(f"could not read CLI output: {read_errors[0]}")
    return Invocation(
        returncode=process.returncode,
        stdout=bytes(buffers["stdout"]),
        stderr=bytes(buffers["stderr"]),
    )


def _require_success(label: str, result: Invocation) -> bytes:
    if result.returncode != 0:
        raise CaptureError(f"{label} CLI exited with status {result.returncode}")
    if result.stderr:
        raise CaptureError(
            f"{label} CLI emitted {len(result.stderr)} unexpected stderr bytes"
        )
    if not result.stdout:
        raise CaptureError(f"{label} CLI emitted empty stdout")
    if len(result.stdout) > MAX_STREAM_BYTES:
        raise CaptureError(f"{label} CLI stdout exceeded the bounded limit")
    if not result.stdout.endswith(b"\n") or b"\r" in result.stdout:
        raise CaptureError(f"{label} CLI stdout is not canonical newline-delimited UTF-8")
    return result.stdout


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CaptureError(f"CLI JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _parse_surface(label: str, data: bytes) -> dict[str, object]:
    try:
        text = data.decode("utf-8", errors="strict")
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"{label} CLI stdout is not strict UTF-8 JSON") from exc
    if type(payload) is not dict:
        raise CaptureError(f"{label} CLI JSON root must be an object")
    return payload


def _require_contract(payload: dict[str, object]) -> None:
    if payload.get("status") != "validated":
        raise CaptureError("CLI target status is not validated")
    if payload.get("target") != {
        "repository": TARGET_REPOSITORY,
        "revision": TARGET_REVISION,
    }:
        raise CaptureError("CLI target identity is not the exact pinned target")
    if type(payload.get("safety")) is not dict:
        raise CaptureError("CLI safety surface is missing")


def _capture_bytes(root: Path, runner: Runner) -> dict[str, bytes]:
    compact = _require_success("compact", runner(COMPACT_COMMAND, root))
    pretty = _require_success("pretty", runner(PRETTY_COMMAND, root))
    compact_payload = _parse_surface("compact", compact)
    pretty_payload = _parse_surface("pretty", pretty)
    if compact_payload != pretty_payload:
        raise CaptureError("compact and pretty CLI JSON surfaces disagree")
    _require_contract(compact_payload)

    canonical_compact = (
        json.dumps(compact_payload, sort_keys=True) + "\n"
    ).encode("utf-8")
    canonical_pretty = (
        json.dumps(pretty_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if compact != canonical_compact:
        raise CaptureError("compact CLI JSON formatting drifted")
    if pretty != canonical_pretty:
        raise CaptureError("pretty CLI JSON formatting drifted")

    return {
        JSON_NAME: compact,
        TEXT_NAME: TRANSCRIPT_COMMAND + pretty,
    }


def _output_directory(root: Path, requested: Path, *, check: bool) -> Path:
    output_dir = requested if requested.is_absolute() else root / requested
    if output_dir.is_symlink():
        raise CaptureError("output directory must not be a symlink")
    if check:
        if not output_dir.is_dir():
            raise CaptureError("--check requires an existing output directory")
    else:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CaptureError(f"could not create output directory: {exc}") from exc
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise CaptureError("output path is not a regular directory")
    return output_dir


def _atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise CaptureError(f"refusing unsafe output path: {path.name}")
    descriptor = -1
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    except OSError as exc:
        raise CaptureError(f"could not atomically write {path.name}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _check_exact(path: Path, expected: bytes) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CaptureError(f"cannot safely open {path.name}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CaptureError(f"evidence output is not a regular file: {path.name}")
        if metadata.st_size != len(expected):
            raise CaptureError(f"evidence byte length drifted: {path.name}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            actual = stream.read(len(expected) + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if actual != expected:
        raise CaptureError(f"evidence bytes drifted: {path.name}")


def capture_evidence(
    root: Path,
    output_dir: Path,
    *,
    check: bool = False,
    runner: Runner | None = None,
) -> tuple[str, ...]:
    root = root.resolve(strict=True)
    destination = _output_directory(root, output_dir, check=check)
    outputs = _capture_bytes(root, runner or _invoke_cli)
    for name in OUTPUT_NAMES:
        path = destination / name
        if check:
            _check_exact(path, outputs[name])
        else:
            _atomic_write(path, outputs[name])
    return OUTPUT_NAMES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="capture deterministic raw evidence from the offline M1 CLI"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare existing outputs byte-for-byte without rewriting them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        outputs = capture_evidence(root, args.output_dir, check=args.check)
    except (CaptureError, OSError) as exc:
        print(f"capture_m1: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "mode": "check" if args.check else "generate",
                "ok": True,
                "outputs": list(outputs),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
