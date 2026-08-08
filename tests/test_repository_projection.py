from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import ast
import importlib.util
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from stratafold.repository_projection import (
    EXPECTED_INPUT_BYTES,
    EXPECTED_INPUT_SHA256,
    EXPECTED_RECEIPT_BYTES,
    EXPECTED_RECEIPT_SHA256,
    EXPECTED_REPOSITORY_BYTES,
    EXPECTED_REPOSITORY_SHA256,
    MAX_RECEIPT_BYTES,
    ProjectionVerificationError,
    TARGET_REVISION,
    canonical_json_bytes,
    loads_json_strict,
    reconstruct_repository,
    verify_projection_bytes,
    verify_projection_files,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    ROOT
    / "metadata"
    / "deepseek-v4-flash-0731"
    / TARGET_REVISION
)
RECEIPT = SNAPSHOT / "repository.receipt.json"
REPOSITORY = SNAPSHOT / "repository.json"
CHECKER = ROOT / "scripts" / "check_repository_projection.py"


def receipt_payload() -> dict[str, object]:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("receipt is not an object")
    return value


def encoded(value: object) -> bytes:
    return canonical_json_bytes(value, pretty=True)


def checker_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "check_repository_projection_test",
        CHECKER,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProjectionReceiptTests(unittest.TestCase):
    def test_valid_receipt_reconstructs_exact_committed_projection(self) -> None:
        receipt_bytes = RECEIPT.read_bytes()
        repository_bytes = REPOSITORY.read_bytes()

        report = verify_projection_bytes(receipt_bytes, repository_bytes)

        self.assertEqual(report["status"], "verified")
        self.assertEqual(len(receipt_bytes), EXPECTED_RECEIPT_BYTES)
        self.assertEqual(
            report["projection_input"],
            {
                "bytes": EXPECTED_INPUT_BYTES,
                "sequence_order": "captured-but-non-semantic-and-untrusted",
                "sha256": EXPECTED_INPUT_SHA256,
                "sibling_count": 74,
            },
        )
        self.assertEqual(
            report["repository_projection"],
            {
                "bytes": EXPECTED_REPOSITORY_BYTES,
                "path": "repository.json",
                "sha256": EXPECTED_REPOSITORY_SHA256,
                "sibling_order": "lexicographic-by-rfilename",
            },
        )
        self.assertEqual(report["response"]["status"], 200)
        self.assertEqual(
            report["response"]["date"],
            "Sat, 08 Aug 2026 15:49:13 GMT",
        )
        self.assertFalse(report["response"]["body_committed"])
        self.assertEqual(
            report["safety"]["weight_or_lfs_payload_bytes_read"],
            0,
        )

    def test_receipt_and_projection_identities_are_exact(self) -> None:
        import hashlib

        receipt_bytes = RECEIPT.read_bytes()
        repository_bytes = REPOSITORY.read_bytes()
        self.assertEqual(len(receipt_bytes), EXPECTED_RECEIPT_BYTES)
        self.assertEqual(
            hashlib.sha256(receipt_bytes).hexdigest(),
            EXPECTED_RECEIPT_SHA256,
        )
        self.assertEqual(len(repository_bytes), EXPECTED_REPOSITORY_BYTES)
        self.assertEqual(
            hashlib.sha256(repository_bytes).hexdigest(),
            EXPECTED_REPOSITORY_SHA256,
        )

    def test_captured_sibling_order_is_non_semantic(self) -> None:
        receipt = receipt_payload()
        projection_input = receipt["projection_input"]
        self.assertIsInstance(projection_input, dict)
        siblings = projection_input["siblings"]
        self.assertIsInstance(siblings, list)

        original = canonical_json_bytes(
            reconstruct_repository(projection_input),
            pretty=True,
        )
        siblings.reverse()
        reordered = canonical_json_bytes(
            reconstruct_repository(projection_input),
            pretty=True,
        )

        self.assertEqual(reordered, original)
        self.assertEqual(reordered, REPOSITORY.read_bytes())

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ProjectionVerificationError,
            "duplicate JSON key",
        ):
            loads_json_strict(
                b'{"projection_input": {}, "projection_input": {}}',
                label="mutation.json",
            )

    def test_duplicate_sibling_name_is_rejected_before_identity_pin(self) -> None:
        receipt = receipt_payload()
        projection_input = receipt["projection_input"]
        self.assertIsInstance(projection_input, dict)
        siblings = projection_input["siblings"]
        self.assertIsInstance(siblings, list)
        first = siblings[0]
        second = siblings[1]
        self.assertIsInstance(first, dict)
        self.assertIsInstance(second, dict)
        second["rfilename"] = first["rfilename"]

        with self.assertRaisesRegex(
            ProjectionVerificationError,
            "duplicate sibling",
        ):
            verify_projection_bytes(encoded(receipt), REPOSITORY.read_bytes())

    def test_parent_path_is_rejected_before_identity_pin(self) -> None:
        receipt = receipt_payload()
        projection_input = receipt["projection_input"]
        self.assertIsInstance(projection_input, dict)
        siblings = projection_input["siblings"]
        self.assertIsInstance(siblings, list)
        first = siblings[0]
        self.assertIsInstance(first, dict)
        first["rfilename"] = "../config.json"

        with self.assertRaisesRegex(ProjectionVerificationError, "unsafe path"):
            verify_projection_bytes(encoded(receipt), REPOSITORY.read_bytes())

    def test_lfs_size_mismatch_is_rejected_before_identity_pin(self) -> None:
        receipt = receipt_payload()
        projection_input = receipt["projection_input"]
        self.assertIsInstance(projection_input, dict)
        siblings = projection_input["siblings"]
        self.assertIsInstance(siblings, list)
        lfs_sibling = next(
            item
            for item in siblings
            if isinstance(item, dict) and "lfs" in item
        )
        lfs = lfs_sibling["lfs"]
        self.assertIsInstance(lfs, dict)
        lfs["size"] += 1

        with self.assertRaisesRegex(
            ProjectionVerificationError,
            "LFS size mismatch",
        ):
            verify_projection_bytes(encoded(receipt), REPOSITORY.read_bytes())

    def test_storage_total_mismatch_is_rejected_before_identity_pin(self) -> None:
        receipt = receipt_payload()
        projection_input = receipt["projection_input"]
        self.assertIsInstance(projection_input, dict)
        projection_input["usedStorage"] += 1

        with self.assertRaisesRegex(
            ProjectionVerificationError,
            "usedStorage does not equal",
        ):
            verify_projection_bytes(encoded(receipt), REPOSITORY.read_bytes())

    def test_parameter_redistribution_is_rejected_before_identity_pin(self) -> None:
        receipt = receipt_payload()
        projection_input = receipt["projection_input"]
        self.assertIsInstance(projection_input, dict)
        safetensors = projection_input["safetensors"]
        self.assertIsInstance(safetensors, dict)
        parameters = safetensors["parameters"]
        self.assertIsInstance(parameters, dict)
        parameters["BF16"] += 1
        parameters["F32"] -= 1

        with self.assertRaisesRegex(
            ProjectionVerificationError,
            "parameter classes drifted",
        ):
            verify_projection_bytes(encoded(receipt), REPOSITORY.read_bytes())

    def test_bool_cannot_replace_integer(self) -> None:
        receipt = receipt_payload()
        projection_input = receipt["projection_input"]
        self.assertIsInstance(projection_input, dict)
        projection_input["usedStorage"] = True

        with self.assertRaisesRegex(
            ProjectionVerificationError,
            "booleans are not integers",
        ):
            verify_projection_bytes(encoded(receipt), REPOSITORY.read_bytes())

    def test_selection_allowlist_mutation_is_rejected(self) -> None:
        receipt = receipt_payload()
        policy = receipt["selection_policy"]
        self.assertIsInstance(policy, dict)
        included = policy["included_top_level_fields"]
        self.assertIsInstance(included, list)
        included.append("downloads")

        with self.assertRaisesRegex(
            ProjectionVerificationError,
            "does not match the capture policy",
        ):
            verify_projection_bytes(encoded(receipt), REPOSITORY.read_bytes())

    def test_benign_receipt_mutation_reaches_final_identity_gate(self) -> None:
        receipt = receipt_payload()
        policy = receipt["selection_policy"]
        self.assertIsInstance(policy, dict)
        policy["rationale"] += " "

        with self.assertRaisesRegex(
            ProjectionVerificationError,
            "receipt reviewed (byte|SHA-256) identity drifted",
        ):
            verify_projection_bytes(encoded(receipt), REPOSITORY.read_bytes())

    def test_repository_bytes_must_be_canonical_projection(self) -> None:
        repository = json.loads(REPOSITORY.read_text(encoding="utf-8"))
        self.assertIsInstance(repository, dict)
        repository["last_modified"] = "2026-08-01T03:07:42.000Z"

        with self.assertRaisesRegex(
            ProjectionVerificationError,
            "semantics do not match",
        ):
            verify_projection_bytes(RECEIPT.read_bytes(), encoded(repository))

    def test_oversized_receipt_is_rejected_before_json_decode(self) -> None:
        oversized = b"{" + b" " * MAX_RECEIPT_BYTES
        with self.assertRaisesRegex(
            ProjectionVerificationError,
            "exceeds .* safety limit",
        ):
            verify_projection_bytes(oversized, REPOSITORY.read_bytes())

    def test_verification_does_not_open_a_socket(self) -> None:
        with patch.object(socket, "socket", side_effect=AssertionError("network used")):
            report = verify_projection_files(RECEIPT, REPOSITORY)
        self.assertEqual(report["status"], "verified")

    def test_module_imports_are_stdlib_and_have_no_transport(self) -> None:
        module_path = ROOT / "src" / "stratafold" / "repository_projection.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        self.assertTrue(
            imports <= {
                "__future__",
                "hashlib",
                "json",
                "os",
                "pathlib",
                "re",
                "stat",
                "typing",
            }
        )
        self.assertTrue(
            {"socket", "urllib", "requests", "http", "subprocess"}.isdisjoint(imports)
        )


class ProjectionCheckerTests(unittest.TestCase):
    def test_checker_is_machine_readable_and_does_not_rewrite_inputs(self) -> None:
        before = {
            path: (
                path.stat().st_ino,
                path.stat().st_size,
                path.stat().st_mtime_ns,
                path.stat().st_ctime_ns,
            )
            for path in (RECEIPT, REPOSITORY)
        }
        output = io.StringIO()
        error = io.StringIO()
        module = checker_module()
        with redirect_stdout(output), redirect_stderr(error):
            return_code = module.main(["--json"])

        self.assertEqual(return_code, 0)
        self.assertEqual(error.getvalue(), "")
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "verified")
        after = {
            path: (
                path.stat().st_ino,
                path.stat().st_size,
                path.stat().st_mtime_ns,
                path.stat().st_ctime_ns,
            )
            for path in (RECEIPT, REPOSITORY)
        }
        self.assertEqual(after, before)

    def test_checker_rejects_invalid_receipt(self) -> None:
        module = checker_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text("{}", encoding="utf-8")
            output = io.StringIO()
            error = io.StringIO()
            with redirect_stdout(output), redirect_stderr(error):
                return_code = module.main(
                    [
                        "--receipt",
                        str(path),
                        "--repository",
                        str(REPOSITORY),
                        "--json",
                    ]
                )

        self.assertEqual(return_code, 2)
        self.assertEqual(output.getvalue(), "")
        payload = json.loads(error.getvalue())
        self.assertEqual(payload["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
