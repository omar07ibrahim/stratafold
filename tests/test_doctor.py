from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from urllib.request import Request

from stratafold.doctor import (
    DoctorLimits,
    RejectAllRedirects,
    RequestPolicyError,
    ResourceDoctor,
    validate_remote_request,
)


REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
BASE = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve"


class RemotePolicyTests(unittest.TestCase):
    def test_pinned_metadata_is_accepted(self) -> None:
        request = validate_remote_request(
            f"{BASE}/{REVISION}/config.json",
            pinned_revision=REVISION,
            content_length=4096,
        )
        self.assertEqual(request.status, "approved-metadata-plan")
        self.assertEqual(request.payload_bytes_read, 0)

    def test_unpinned_metadata_fails_closed(self) -> None:
        with self.assertRaisesRegex(RequestPolicyError, "not revision-bound"):
            validate_remote_request(
                f"{BASE}/main/config.json",
                pinned_revision=REVISION,
                content_length=4096,
            )

    def test_query_parameter_cannot_fake_revision_binding(self) -> None:
        with self.assertRaisesRegex(RequestPolicyError, "query strings"):
            validate_remote_request(
                f"{BASE}/main/config.json?ignored={REVISION}",
                pinned_revision=REVISION,
                content_length=4096,
            )

    def test_unknown_length_cannot_reach_opener(self) -> None:
        calls = 0

        def opener(_request: object) -> object:
            nonlocal calls
            calls += 1
            raise AssertionError("unknown-length request reached opener")

        with self.assertRaisesRegex(RequestPolicyError, "known content length"):
            request = validate_remote_request(
                f"{BASE}/{REVISION}/config.json",
                pinned_revision=REVISION,
            )
            opener(request)
        self.assertEqual(calls, 0)

    def test_traversal_cannot_escape_pinned_revision(self) -> None:
        with self.assertRaisesRegex(RequestPolicyError, "traversing"):
            validate_remote_request(
                f"{BASE}/{REVISION}/../main/config.json",
                pinned_revision=REVISION,
                content_length=4096,
            )

    def test_double_encoded_weight_suffix_is_rejected(self) -> None:
        with self.assertRaises(RequestPolicyError):
            validate_remote_request(
                f"{BASE}/{REVISION}/model%252esafetensors",
                pinned_revision=REVISION,
                content_length=4096,
            )

    def test_explicit_port_is_rejected(self) -> None:
        with self.assertRaisesRegex(RequestPolicyError, "ports"):
            validate_remote_request(
                "https://huggingface.co:443/deepseek-ai/DeepSeek-V4-Flash-0731/"
                f"resolve/{REVISION}/config.json",
                pinned_revision=REVISION,
                content_length=4096,
            )

    def test_weight_request_fails_before_opener(self) -> None:
        calls = 0

        def opener(_request: object) -> object:
            nonlocal calls
            calls += 1
            raise AssertionError("opener must not be constructed for a weight URL")

        with self.assertRaisesRegex(RequestPolicyError, "weight objects"):
            request = validate_remote_request(
                f"{BASE}/{REVISION}/model-00001-of-00048.safetensors",
                pinned_revision=REVISION,
                content_length=4096,
            )
            opener(request)
        self.assertEqual(calls, 0)

    def test_redirect_is_rejected_before_followup_request(self) -> None:
        handler = RejectAllRedirects()
        request = Request(f"{BASE}/{REVISION}/config.json")
        with self.assertRaisesRegex(RequestPolicyError, "redirects are forbidden"):
            handler.redirect_request(
                request,
                object(),
                302,
                "Found",
                {},
                "https://cas-bridge.xethub.hf.co/weight.safetensors",
            )

    def test_weight_index_metadata_is_allowed(self) -> None:
        request = validate_remote_request(
            f"{BASE}/{REVISION}/model.safetensors.index.json",
            pinned_revision=REVISION,
            content_length=2_000_000,
        )
        self.assertEqual(request.content_length, 2_000_000)

    def test_object_over_32_mib_is_rejected(self) -> None:
        with self.assertRaisesRegex(RequestPolicyError, "remote object exceeds"):
            validate_remote_request(
                f"{BASE}/{REVISION}/config.json",
                pinned_revision=REVISION,
                content_length=32 * 1024**2 + 1,
            )

    def test_aggregate_over_128_mib_is_rejected(self) -> None:
        with self.assertRaisesRegex(RequestPolicyError, "aggregate fetch exceeds"):
            validate_remote_request(
                f"{BASE}/{REVISION}/config.json",
                pinned_revision=REVISION,
                content_length=1,
                aggregate_bytes=128 * 1024**2,
            )

    def test_xet_host_is_rejected(self) -> None:
        with self.assertRaisesRegex(RequestPolicyError, "allowlist"):
            validate_remote_request(
                f"https://cas-bridge.xethub.hf.co/{REVISION}/config.json",
                pinned_revision=REVISION,
                content_length=10,
            )


class ResourceDoctorTests(unittest.TestCase):
    def test_small_workspace_passes_current_host_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "small.txt").write_text("bounded", encoding="utf-8")
            report = ResourceDoctor(DoctorLimits()).inspect(
                workspace=root, cache=root / ".cache"
            )
        self.assertTrue(report.ok, report.violations)
        self.assertGreaterEqual(report.free_bytes, 2 * 1024**3)
        self.assertGreater(report.cpu_count, 0)
        self.assertEqual(report.filesystem_path, "/")


if __name__ == "__main__":
    unittest.main()
