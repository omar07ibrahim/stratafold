from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from capture_m1 import (  # noqa: E402
    COMPACT_COMMAND,
    JSON_NAME,
    PRETTY_COMMAND,
    TEXT_NAME,
    TRANSCRIPT_COMMAND,
    CaptureError,
    Invocation,
    capture_evidence,
)


class CaptureM1Tests(unittest.TestCase):
    @staticmethod
    def _payload() -> dict[str, object]:
        return {
            "evidence_level": "pinned-official-metadata",
            "safety": {},
            "schema_version": 1,
            "status": "validated",
            "target": {
                "repository": "deepseek-ai/DeepSeek-V4-Flash-0731",
                "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            },
        }

    @staticmethod
    def _surface(payload: dict[str, object], *, pretty: bool) -> bytes:
        return (
            json.dumps(payload, indent=2 if pretty else None, sort_keys=True) + "\n"
        ).encode("utf-8")

    def _runner(
        self,
        *,
        compact_payload: dict[str, object] | None = None,
        pretty_payload: dict[str, object] | None = None,
        compact_returncode: int = 0,
        compact_stderr: bytes = b"",
    ):
        compact_payload = compact_payload or self._payload()
        pretty_payload = pretty_payload or compact_payload

        def run(command: tuple[str, ...], _root: Path) -> Invocation:
            if command == COMPACT_COMMAND:
                return Invocation(
                    compact_returncode,
                    self._surface(compact_payload, pretty=False),
                    compact_stderr,
                )
            if command == PRETTY_COMMAND:
                return Invocation(
                    0,
                    self._surface(pretty_payload, pretty=True),
                    b"",
                )
            raise AssertionError(f"unexpected command: {command!r}")

        return run

    def test_actual_generation_is_deterministic_and_check_does_not_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "raw"
            capture_evidence(ROOT, output_dir)
            first = {
                name: (output_dir / name).read_bytes()
                for name in (JSON_NAME, TEXT_NAME)
            }

            capture_evidence(ROOT, output_dir)
            second = {
                name: (output_dir / name).read_bytes()
                for name in (JSON_NAME, TEXT_NAME)
            }
            self.assertEqual(first, second)

            command, pretty = second[TEXT_NAME].split(b"\n", 1)
            self.assertEqual(command + b"\n", TRANSCRIPT_COMMAND)
            self.assertEqual(json.loads(second[JSON_NAME]), json.loads(pretty))

            before = {
                name: self._stable_metadata(output_dir / name)
                for name in (JSON_NAME, TEXT_NAME)
            }
            capture_evidence(ROOT, output_dir, check=True)
            after = {
                name: self._stable_metadata(output_dir / name)
                for name in (JSON_NAME, TEXT_NAME)
            }
            self.assertEqual(before, after)

    @staticmethod
    def _stable_metadata(path: Path) -> tuple[int, int, int, int, int]:
        metadata = path.stat()
        return (
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    def test_nonzero_cli_surface_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CaptureError, "exited with status 7"):
                capture_evidence(
                    ROOT,
                    Path(directory),
                    runner=self._runner(compact_returncode=7),
                )

    def test_stderr_cli_surface_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CaptureError, "unexpected stderr bytes"):
                capture_evidence(
                    ROOT,
                    Path(directory),
                    runner=self._runner(compact_stderr=b"warning\n"),
                )

    def test_mismatched_cli_surfaces_are_rejected(self) -> None:
        compact_payload = self._payload()
        pretty_payload = deepcopy(compact_payload)
        pretty_payload["status"] = "rejected"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CaptureError, "surfaces disagree"):
                capture_evidence(
                    ROOT,
                    Path(directory),
                    runner=self._runner(
                        compact_payload=compact_payload,
                        pretty_payload=pretty_payload,
                    ),
                )

    def test_check_rejects_drift_without_rewriting(self) -> None:
        runner = self._runner()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            capture_evidence(ROOT, output_dir, runner=runner)
            json_path = output_dir / JSON_NAME
            json_path.write_bytes(b"{}\n")
            before = json_path.read_bytes()
            with self.assertRaisesRegex(CaptureError, "byte length drifted"):
                capture_evidence(ROOT, output_dir, check=True, runner=runner)
            self.assertEqual(json_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
