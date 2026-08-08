from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from verify_m1_evidence import (  # noqa: E402
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_JSON_SHA256,
    EXPECTED_PROVENANCE_SHA256,
    EXPECTED_TEXT_SHA256,
    MAX_EVIDENCE_BYTES,
    SOURCE_HEAD,
    TRANSCRIPT_COMMAND,
    EvidenceVerificationError,
    loads_json_strict,
    main,
    read_regular_file,
    verify_m1_evidence,
    verify_m1_evidence_bytes,
)


EVIDENCE = ROOT / "evidence" / "raw"
JSON_PATH = EVIDENCE / "m1_target_genome.json"
TEXT_PATH = EVIDENCE / "m1_target_genome.txt"
PROVENANCE_PATH = EVIDENCE / "m1_target_genome.provenance.json"
MANIFEST_PATH = (
    ROOT
    / "metadata"
    / "deepseek-v4-flash-0731"
    / "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    / "manifest.json"
)
RECEIPT_PATH = MANIFEST_PATH.with_name("repository.receipt.json")
VERIFIER_PATH = ROOT / "scripts" / "verify_m1_evidence.py"


def inputs() -> list[bytes]:
    return [
        JSON_PATH.read_bytes(),
        TEXT_PATH.read_bytes(),
        PROVENANCE_PATH.read_bytes(),
        MANIFEST_PATH.read_bytes(),
        RECEIPT_PATH.read_bytes(),
    ]


def report_payload() -> dict[str, object]:
    value = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("M1 JSON is not an object")
    return value


def provenance_payload() -> dict[str, object]:
    value = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("M1 provenance is not an object")
    return value


def report_surfaces(report: dict[str, object]) -> tuple[bytes, bytes]:
    compact = (json.dumps(report, sort_keys=True) + "\n").encode("utf-8")
    pretty = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return compact, TRANSCRIPT_COMMAND + pretty


def provenance_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


class AdoptedEvidenceTests(unittest.TestCase):
    def test_valid_evidence_is_linked_to_exact_reviewed_artifact(self) -> None:
        report = verify_m1_evidence(ROOT)

        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["source_head"], SOURCE_HEAD)
        self.assertEqual(
            report["ci"],
            {
                "artifact_id": 9_024_264_228,
                "artifact_name": "m1-target-genome-31266366484",
                "conclusion": "success",
                "job_id": 93_125_047_136,
                "run_attempt": 1,
                "run_id": 31_266_366_484,
                "workflow_path": ".github/workflows/ci.yml",
            },
        )
        self.assertEqual(report["archive"]["bytes"], 12_568)
        self.assertEqual(
            report["archive"]["sha256"],
            EXPECTED_ARCHIVE_SHA256,
        )
        self.assertFalse(report["archive"]["committed"])
        self.assertEqual(report["archive"]["entry_count"], 2)

    def test_adopted_file_identities_are_exact(self) -> None:
        expected = {
            JSON_PATH: (5_674, EXPECTED_JSON_SHA256),
            TEXT_PATH: (6_606, EXPECTED_TEXT_SHA256),
            PROVENANCE_PATH: (4_240, EXPECTED_PROVENANCE_SHA256),
        }
        for path, (size, sha256) in expected.items():
            with self.subTest(path=path.name):
                data = path.read_bytes()
                self.assertEqual(len(data), size)
                self.assertEqual(hashlib.sha256(data).hexdigest(), sha256)

    def test_compact_and_transcript_surfaces_are_same_json(self) -> None:
        compact = loads_json_strict(JSON_PATH.read_bytes(), label="compact")
        transcript = TEXT_PATH.read_bytes()
        self.assertTrue(transcript.startswith(TRANSCRIPT_COMMAND))
        pretty = loads_json_strict(
            transcript[len(TRANSCRIPT_COMMAND):],
            label="pretty",
        )
        self.assertEqual(compact, pretty)

    def test_provenance_states_ephemeral_and_attestation_limits(self) -> None:
        provenance = provenance_payload()
        artifact = provenance["artifact"]
        self.assertIsInstance(artifact, dict)
        archive = artifact["archive"]
        self.assertIsInstance(archive, dict)
        self.assertFalse(archive["committed"])
        limitations = provenance["limitations"]
        self.assertIsInstance(limitations, list)
        self.assertTrue(any("ephemeral" in item for item in limitations))
        self.assertTrue(
            any("not an independent attestation" in item for item in limitations)
        )
        self.assertFalse(any(path.suffix == ".zip" for path in EVIDENCE.iterdir()))

    def test_verifier_does_not_open_a_socket_or_rewrite_evidence(self) -> None:
        paths = (JSON_PATH, TEXT_PATH, PROVENANCE_PATH)
        before = {
            path: (
                path.stat().st_ino,
                path.stat().st_size,
                path.stat().st_mtime_ns,
                path.stat().st_ctime_ns,
            )
            for path in paths
        }
        with patch.object(socket, "socket", side_effect=AssertionError("network used")):
            report = verify_m1_evidence(ROOT)
        self.assertEqual(report["status"], "verified")
        after = {
            path: (
                path.stat().st_ino,
                path.stat().st_size,
                path.stat().st_mtime_ns,
                path.stat().st_ctime_ns,
            )
            for path in paths
        }
        self.assertEqual(after, before)

    def test_machine_readable_cli(self) -> None:
        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            return_code = main(["--root", str(ROOT), "--json"])
        self.assertEqual(return_code, 0)
        self.assertEqual(error.getvalue(), "")
        self.assertEqual(json.loads(output.getvalue())["status"], "verified")


class EvidenceMutationTests(unittest.TestCase):
    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            EvidenceVerificationError,
            "duplicate JSON key",
        ):
            loads_json_strict(
                b'{"status": "validated", "status": "rejected"}',
                label="mutation.json",
            )

    def test_transcript_command_drift_is_rejected(self) -> None:
        values = inputs()
        values[1] = values[1].replace(
            b"PYTHONPATH=src",
            b"PYTHONPATH=/tmp",
            1,
        )
        with self.assertRaisesRegex(
            EvidenceVerificationError,
            "transcript command does not match",
        ):
            verify_m1_evidence_bytes(*values)

    def test_projection_identity_drift_is_rejected_semantically(self) -> None:
        report = report_payload()
        projection = report["repository_projection_verification"]
        self.assertIsInstance(projection, dict)
        projection_input = projection["projection_input"]
        self.assertIsInstance(projection_input, dict)
        projection_input["sha256"] = "0" * 64
        compact, transcript = report_surfaces(report)
        values = inputs()
        values[0] = compact
        values[1] = transcript
        with self.assertRaisesRegex(
            EvidenceVerificationError,
            "projection input identity does not match",
        ):
            verify_m1_evidence_bytes(*values)

    def test_absolute_host_path_is_rejected(self) -> None:
        report = report_payload()
        topology = report["topology"]
        self.assertIsInstance(topology, dict)
        topology["diagnostic_path"] = "/home/runner/work/stratafold"
        compact, transcript = report_surfaces(report)
        values = inputs()
        values[0] = compact
        values[1] = transcript
        with self.assertRaisesRegex(
            EvidenceVerificationError,
            "absolute host path",
        ):
            verify_m1_evidence_bytes(*values)

    def test_credential_shape_is_rejected(self) -> None:
        report = report_payload()
        topology = report["topology"]
        self.assertIsInstance(topology, dict)
        topology["diagnostic"] = "github_pat_" + "A" * 40
        compact, transcript = report_surfaces(report)
        values = inputs()
        values[0] = compact
        values[1] = transcript
        with self.assertRaisesRegex(
            EvidenceVerificationError,
            "credential-shaped text",
        ):
            verify_m1_evidence_bytes(*values)

    def test_runtime_timestamp_field_is_rejected(self) -> None:
        report = report_payload()
        topology = report["topology"]
        self.assertIsInstance(topology, dict)
        topology["generated_at"] = "2026-08-08T16:12:21Z"
        compact, transcript = report_surfaces(report)
        values = inputs()
        values[0] = compact
        values[1] = transcript
        with self.assertRaisesRegex(
            EvidenceVerificationError,
            "runtime timestamp field",
        ):
            verify_m1_evidence_bytes(*values)

    def test_unknown_provenance_schema_is_rejected(self) -> None:
        provenance = provenance_payload()
        provenance["unexpected"] = True
        values = inputs()
        values[2] = provenance_bytes(provenance)
        with self.assertRaisesRegex(
            EvidenceVerificationError,
            "unknown schema",
        ):
            verify_m1_evidence_bytes(*values)

    def test_provenance_limitation_drift_is_rejected(self) -> None:
        provenance = provenance_payload()
        limitations = provenance["limitations"]
        self.assertIsInstance(limitations, list)
        limitations.pop()
        values = inputs()
        values[2] = provenance_bytes(provenance)
        with self.assertRaisesRegex(
            EvidenceVerificationError,
            "limitations do not match",
        ):
            verify_m1_evidence_bytes(*values)

    def test_source_snapshot_identity_drift_is_rejected(self) -> None:
        values = inputs()
        values[3] = values[3] + b"\n"
        with self.assertRaisesRegex(
            EvidenceVerificationError,
            "committed source identity drifted",
        ):
            verify_m1_evidence_bytes(*values)

    def test_oversized_input_is_rejected_before_decode(self) -> None:
        values = inputs()
        values[0] = b"{" + b" " * MAX_EVIDENCE_BYTES
        with self.assertRaisesRegex(
            EvidenceVerificationError,
            "exceeds .* safety limit",
        ):
            verify_m1_evidence_bytes(*values)


class SafeReaderTests(unittest.TestCase):
    def test_symlink_is_not_followed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"safe")
            link = root / "link"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            with self.assertRaisesRegex(
                EvidenceVerificationError,
                "cannot safely read file",
            ):
                read_regular_file(link, maximum_bytes=32)

    def test_special_file_is_rejected(self) -> None:
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fifo"
            try:
                os.mkfifo(path)
            except OSError as exc:
                self.skipTest(f"FIFO unavailable: {exc}")
            with self.assertRaisesRegex(
                EvidenceVerificationError,
                "expected a regular file",
            ):
                read_regular_file(path, maximum_bytes=32)

    def test_sparse_oversized_file_is_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large"
            with path.open("wb") as stream:
                stream.truncate(MAX_EVIDENCE_BYTES + 1)
            with self.assertRaisesRegex(
                EvidenceVerificationError,
                "exceeds .* safety limit",
            ):
                read_regular_file(path, maximum_bytes=MAX_EVIDENCE_BYTES)

    def test_verifier_imports_no_network_or_process_transport(self) -> None:
        tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        self.assertTrue(
            imports
            <= {
                "__future__",
                "argparse",
                "hashlib",
                "json",
                "os",
                "pathlib",
                "re",
                "stat",
                "sys",
                "typing",
            }
        )
        self.assertTrue(
            {
                "http",
                "requests",
                "socket",
                "subprocess",
                "urllib",
            }.isdisjoint(imports)
        )


if __name__ == "__main__":
    unittest.main()
