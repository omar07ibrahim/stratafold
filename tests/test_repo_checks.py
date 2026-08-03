from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from check_repo import validate_claims, validate_repository  # noqa: E402


class ClaimsPathTests(unittest.TestCase):
    @staticmethod
    def _claim_payload(evidence_path: str) -> dict[str, object]:
        return {
            "schema_version": 1,
            "allowed_evidence_tags": [
                "measured",
                "source-reproduced",
                "derived",
                "projected",
                "unverified",
            ],
            "claims": [
                {
                    "id": "SF-C9999",
                    "statement": "synthetic test claim",
                    "tag": "measured",
                    "scope": "unit test",
                    "evidence": [evidence_path],
                    "reproduce": "unit test",
                }
            ],
        }

    def test_nested_parent_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self._claim_payload("x/../../outside.json")
            (root / "CLAIMS.json").write_text(json.dumps(payload), encoding="utf-8")
            errors = validate_claims(root)
        self.assertTrue(any("unsafe evidence path" in error for error in errors), errors)

    def test_existing_but_untracked_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evidence.json").write_text("{}", encoding="utf-8")
            (root / "CLAIMS.json").write_text(
                json.dumps(self._claim_payload("evidence.json")), encoding="utf-8"
            )
            errors = validate_claims(root, tracked_paths={"CLAIMS.json"})
        self.assertTrue(any("not Git-tracked" in error for error in errors), errors)

    def test_external_symlink_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root_directory, tempfile.TemporaryDirectory() as outside_directory:
            root = Path(root_directory)
            outside = Path(outside_directory) / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            (root / "evidence.json").symlink_to(outside)
            (root / "CLAIMS.json").write_text(
                json.dumps(self._claim_payload("evidence.json")), encoding="utf-8"
            )
            errors = validate_claims(
                root, tracked_paths={"CLAIMS.json", "evidence.json"}
            )
        self.assertTrue(any("does not exist" in error for error in errors), errors)


class SecretPatternTests(unittest.TestCase):
    def _errors_for(self, secret: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "leak.txt").write_text(secret, encoding="utf-8")
            return validate_repository(root)

    def test_huggingface_token_shape_is_detected(self) -> None:
        errors = self._errors_for("hf_" + "A" * 32)
        self.assertTrue(any("huggingface-token" in error for error in errors), errors)

    def test_github_fine_grained_token_shape_is_detected(self) -> None:
        errors = self._errors_for("github_pat_" + "A" * 40)
        self.assertTrue(any("github-fine-grained-token" in error for error in errors), errors)

    def test_openai_project_key_shape_is_detected(self) -> None:
        errors = self._errors_for("sk-proj-" + "A" * 32)
        self.assertTrue(any("openai-api-key" in error for error in errors), errors)

    def test_encrypted_private_key_header_is_detected(self) -> None:
        errors = self._errors_for("-----BEGIN " + "ENCRYPTED PRIVATE KEY-----")
        self.assertTrue(any("private-key" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
