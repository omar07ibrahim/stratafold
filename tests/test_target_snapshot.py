from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch

from stratafold.__main__ import main
from stratafold.target_snapshot import (
    DEFAULT_SNAPSHOT,
    MAX_FILE_BYTES,
    MAX_MANIFEST_BYTES,
    SnapshotValidationError,
    inspect_snapshot,
    loads_json_strict,
)


@contextmanager
def copied_snapshot() -> object:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "snapshot"
        shutil.copytree(DEFAULT_SNAPSHOT, root)
        yield root


def load_json(root: Path, relative: str) -> dict[str, object]:
    payload = json.loads((root / relative).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} is not an object")
    return payload


def refresh_manifest(root: Path, relative: str) -> None:
    data = (root / relative).read_bytes()
    manifest = load_json(root, "manifest.json")
    files = manifest["files"]
    if not isinstance(files, list):
        raise AssertionError("manifest files is not a list")
    for raw_entry in files:
        if isinstance(raw_entry, dict) and raw_entry.get("path") == relative:
            raw_entry["bytes"] = len(data)
            raw_entry["sha256"] = hashlib.sha256(data).hexdigest()
            break
    else:
        raise AssertionError(f"manifest has no entry for {relative}")
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_json(root: Path, relative: str, payload: object) -> None:
    (root / relative).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    refresh_manifest(root, relative)


class PinnedSnapshotTests(unittest.TestCase):
    def test_valid_pinned_snapshot_reports_separate_ledgers(self) -> None:
        report = inspect_snapshot()

        self.assertEqual(report["status"], "validated")
        self.assertEqual(report["evidence_level"], "pinned-official-metadata")
        self.assertEqual(report["schema_version"], 2)
        verification = report["repository_projection_verification"]
        self.assertEqual(verification["status"], "verified")
        self.assertEqual(
            verification["projection_input"]["sha256"],
            "16dd1cea2018d8af2a84922bc6fff22a7988dbb4ecc58d9c92486fa01b178291",
        )
        self.assertEqual(
            verification["repository_projection"],
            {
                "bytes": 22_284,
                "path": "repository.json",
                "sha256": (
                    "6cacae22067d225351b46d30b3b4335db18b8941e342ac24ab945d81ebef4800"
                ),
                "sibling_order": "lexicographic-by-rfilename",
            },
        )
        self.assertFalse(verification["response"]["body_committed"])
        self.assertEqual(
            verification["safety"]["weight_or_lfs_payload_bytes_read"],
            0,
        )
        topology = report["topology"]
        self.assertEqual(topology["evidence_tag"], "source-reproduced")
        self.assertEqual(topology["hidden_layers"], 43)
        self.assertEqual(topology["hidden_size"], 4096)
        self.assertEqual(topology["routed_experts_per_layer"], 256)
        self.assertEqual(topology["shared_experts_per_layer"], 1)
        self.assertEqual(topology["experts_selected_per_token"], 6)
        self.assertEqual(topology["hash_layers"], 3)
        self.assertEqual(topology["dspark_target_layer_ids"], [40, 41, 42])

        declared = report["declared_representation"]
        self.assertEqual(declared["declared_expert_dtype"], "fp4")
        self.assertEqual(declared["declared_torch_dtype"], "bfloat16")
        self.assertEqual(
            declared["scope"],
            "configuration declarations only; no shard payload or header inspected",
        )
        self.assertNotIn("native_representation", report)
        self.assertEqual(
            declared["declared_quantization_config"]["quant_method"], "fp8"
        )
        self.assertEqual(declared["declared_quantization_config"]["fmt"], "e4m3")

        tensor_index = report["tensor_index"]
        self.assertEqual(tensor_index["tensor_entries"], 72_317)
        self.assertEqual(tensor_index["shard_count"], 48)
        self.assertEqual(tensor_index["shard_denominator"], 48)
        self.assertEqual(
            tensor_index["expert_census"],
            {
                "backbone_routed_expert_slots": 11_008,
                "backbone_routed_tensor_entries": 66_048,
                "backbone_shared_expert_slots": 43,
                "backbone_shared_tensor_entries": 258,
                "attachment_routed_expert_slots": 768,
                "attachment_routed_tensor_entries": 4_608,
                "attachment_shared_expert_slots": 3,
                "attachment_shared_tensor_entries": 18,
            },
        )

        parameter_classes = report["api_reported_parameter_classes"]
        self.assertEqual(
            parameter_classes["counts_by_storage_class"]["total"],
            304_180_418_494,
        )
        self.assertIn(
            "not recomputed from shard headers",
            parameter_classes["scope"],
        )

        ledgers = report["byte_ledgers"]
        self.assertEqual(
            ledgers["index_declared_tensor_payload"]["bytes"],
            166_878_536_440,
        )
        repository_bytes = ledgers["api_reported_repository_artifact_bytes"]
        self.assertEqual(repository_bytes["weight_shard_bytes"], 166_886_535_336)
        self.assertEqual(repository_bytes["non_weight_file_bytes"], 12_125_738)
        self.assertEqual(
            repository_bytes["all_repository_file_bytes"],
            166_898_661_074,
        )
        gap = ledgers["artifact_minus_index_payload_gap"]
        self.assertEqual(gap["bytes"], 7_998_896)
        self.assertEqual(
            gap["formula"],
            "api_reported_weight_shard_bytes - index_declared_tensor_payload_bytes",
        )
        self.assertEqual(
            gap["scope"],
            (
                "unattributed metadata difference; not measured compression or "
                "proven container overhead"
            ),
        )
        self.assertEqual(
            ledgers["committed_metadata_snapshot"]["listed_file_bytes"],
            5_652_705,
        )

        safety = report["safety"]
        observation = safety["decoder_observation"]
        self.assertEqual(observation["scope"], "this invocation and snapshot only")
        self.assertEqual(observation["snapshot_weight_shard_files_present"], 0)
        self.assertEqual(observation["weight_shard_files_opened_by_decoder"], 0)
        self.assertFalse(
            observation["target_code_imported_or_executed_by_decoder"]
        )
        self.assertFalse(observation["full_checkpoint_operation_performed"])
        attestation = safety["capture_attestation"]
        self.assertEqual(attestation["evidence_tag"], "unverified")
        self.assertEqual(
            attestation["scope"],
            "capture-time statement; not an independent or hostwide audit",
        )
        self.assertEqual(safety["hostwide_download_state"], "not_audited")
        self.assertNotIn("full_checkpoint", safety)

    def test_inspection_does_not_open_a_socket(self) -> None:
        with patch.object(socket, "socket", side_effect=AssertionError("network used")):
            report = inspect_snapshot()
        self.assertEqual(report["status"], "validated")

    def test_compact_cli_is_machine_readable(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            return_code = main(
                [
                    "inspect-target",
                    "--snapshot",
                    str(DEFAULT_SNAPSHOT),
                    "--json",
                ]
            )
        self.assertEqual(return_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "validated")
        self.assertEqual(payload["tensor_index"]["shard_count"], 48)

    def test_cli_rejects_missing_snapshot(self) -> None:
        error = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(error):
            return_code = main(
                ["inspect-target", "--snapshot", str(Path(directory) / "missing")]
            )
        self.assertEqual(return_code, 2)
        payload = json.loads(error.getvalue())
        self.assertEqual(payload["status"], "rejected")
        self.assertIn("does not exist", payload["error"])


class StrictJsonTests(unittest.TestCase):
    def test_duplicate_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(SnapshotValidationError, "duplicate JSON key"):
            loads_json_strict(b'{"same": 1, "same": 2}', label="mutation.json")

    def test_bool_cannot_replace_topology_integer(self) -> None:
        with copied_snapshot() as root:
            config = load_json(root, "config.json")
            config["num_hidden_layers"] = True
            write_json(root, "config.json", config)
            with self.assertRaisesRegex(
                SnapshotValidationError, "booleans are not integers"
            ):
                inspect_snapshot(root)


class SemanticMutationTests(unittest.TestCase):
    def test_topology_and_native_dtype_mutations_fail_closed(self) -> None:
        cases = (
            ("n_routed_experts", 255, "expected 256"),
            ("num_experts_per_tok", 5, "expected 6"),
            ("num_hidden_layers", 42, "expected 43"),
            ("expert_dtype", "fp8", "expected fp4"),
        )
        for key, value, message in cases:
            with self.subTest(key=key), copied_snapshot() as root:
                config = load_json(root, "config.json")
                config[key] = value
                write_json(root, "config.json", config)
                with self.assertRaisesRegex(SnapshotValidationError, message):
                    inspect_snapshot(root)

    def test_unsafe_shard_path_is_rejected(self) -> None:
        with copied_snapshot() as root:
            index = load_json(root, "model.safetensors.index.json")
            weight_map = index["weight_map"]
            self.assertIsInstance(weight_map, dict)
            first = next(iter(weight_map))
            weight_map[first] = "../model-00001-of-00048.safetensors"
            write_json(root, "model.safetensors.index.json", index)
            with self.assertRaisesRegex(
                SnapshotValidationError, "unsafe or malformed shard name"
            ):
                inspect_snapshot(root)

    def test_inconsistent_shard_denominator_is_rejected(self) -> None:
        with copied_snapshot() as root:
            index = load_json(root, "model.safetensors.index.json")
            weight_map = index["weight_map"]
            self.assertIsInstance(weight_map, dict)
            first = next(iter(weight_map))
            weight_map[first] = "model-00001-of-00047.safetensors"
            write_json(root, "model.safetensors.index.json", index)
            with self.assertRaisesRegex(
                SnapshotValidationError, "inconsistent shard denominators"
            ):
                inspect_snapshot(root)

    def test_shard_gap_is_rejected(self) -> None:
        with copied_snapshot() as root:
            index = load_json(root, "model.safetensors.index.json")
            weight_map = index["weight_map"]
            self.assertIsInstance(weight_map, dict)
            missing = "model-00024-of-00048.safetensors"
            replacement = "model-00023-of-00048.safetensors"
            changed = 0
            for tensor, shard in weight_map.items():
                if shard == missing:
                    weight_map[tensor] = replacement
                    changed += 1
            self.assertGreater(changed, 0)
            write_json(root, "model.safetensors.index.json", index)
            with self.assertRaisesRegex(SnapshotValidationError, "shard ordinal gap"):
                inspect_snapshot(root)

    def test_missing_expert_tensor_is_rejected(self) -> None:
        with copied_snapshot() as root:
            index = load_json(root, "model.safetensors.index.json")
            weight_map = index["weight_map"]
            self.assertIsInstance(weight_map, dict)
            del weight_map["layers.0.ffn.experts.0.w1.weight"]
            write_json(root, "model.safetensors.index.json", index)
            with self.assertRaisesRegex(
                SnapshotValidationError, "tensor census mismatch"
            ):
                inspect_snapshot(root)


class PinnedIdentityMutationTests(unittest.TestCase):
    def test_valid_looking_manifest_blob_identity_drift_is_rejected(self) -> None:
        with copied_snapshot() as root:
            manifest = load_json(root, "manifest.json")
            files = manifest["files"]
            self.assertIsInstance(files, list)
            config_entry = next(
                entry
                for entry in files
                if isinstance(entry, dict) and entry.get("path") == "config.json"
            )
            config_entry["upstream_blob_id"] = (
                "c3b10d45a829545fbf0d9d2880a1aa0b9ab3b43a"
            )
            (root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SnapshotValidationError, "reviewed upstream blob identity drifted"
            ):
                inspect_snapshot(root)

    def test_license_content_drift_with_refreshed_manifest_is_rejected(self) -> None:
        with copied_snapshot() as root:
            path = root / "LICENSE.target.txt"
            path.write_bytes(path.read_bytes() + b"\n")
            refresh_manifest(root, "LICENSE.target.txt")
            with self.assertRaisesRegex(
                SnapshotValidationError, "reviewed manifest byte identity drifted"
            ):
                inspect_snapshot(root)

    def test_benign_config_drift_reaches_final_content_gate(self) -> None:
        with copied_snapshot() as root:
            config = load_json(root, "config.json")
            config["attention_dropout"] = 0.125
            write_json(root, "config.json", config)
            with self.assertRaisesRegex(
                SnapshotValidationError, "reviewed manifest byte identity drifted"
            ):
                inspect_snapshot(root)

    def test_repository_blob_cross_link_is_pinned(self) -> None:
        with copied_snapshot() as root:
            repository = load_json(root, "repository.json")
            files = repository["files"]
            self.assertIsInstance(files, list)
            config_entry = next(
                entry
                for entry in files
                if isinstance(entry, dict) and entry.get("path") == "config.json"
            )
            config_entry["blob_id"] = (
                "c3b10d45a829545fbf0d9d2880a1aa0b9ab3b43a"
            )
            write_json(root, "repository.json", repository)
            with self.assertRaisesRegex(
                SnapshotValidationError, "repository.json blob cross-link drifted"
            ):
                inspect_snapshot(root)

    def test_format_only_manifest_drift_reaches_final_identity_gate(self) -> None:
        with copied_snapshot() as root:
            path = root / "manifest.json"
            original = path.read_bytes()
            changed = original.replace(
                b'\n  "files"',
                b'\n \t"files"',
                1,
            )
            self.assertEqual(len(changed), len(original))
            self.assertNotEqual(changed, original)
            self.assertEqual(json.loads(changed), json.loads(original))
            path.write_bytes(changed)
            with self.assertRaisesRegex(
                SnapshotValidationError,
                r"manifest\.json: reviewed SHA-256 identity drifted",
            ):
                inspect_snapshot(root)

    def test_receipt_semantics_precede_receipt_and_manifest_identity(self) -> None:
        with copied_snapshot() as root:
            receipt = load_json(root, "repository.receipt.json")
            projection_input = receipt["projection_input"]
            self.assertIsInstance(projection_input, dict)
            projection_input["usedStorage"] += 1
            write_json(root, "repository.receipt.json", receipt)
            with self.assertRaisesRegex(
                SnapshotValidationError,
                (
                    "repository projection verification failed: "
                    "receipt usedStorage does not equal the LFS byte total"
                ),
            ):
                inspect_snapshot(root)

    def test_arbitrary_non_expert_tensor_is_rejected(self) -> None:
        with copied_snapshot() as root:
            index = load_json(root, "model.safetensors.index.json")
            weight_map = index["weight_map"]
            self.assertIsInstance(weight_map, dict)
            weight_map["audit.non_expert.weight"] = next(iter(weight_map.values()))
            write_json(root, "model.safetensors.index.json", index)
            with self.assertRaisesRegex(
                SnapshotValidationError, "index tensor count drifted"
            ):
                inspect_snapshot(root)

    def test_tensor_payload_size_drift_is_rejected(self) -> None:
        with copied_snapshot() as root:
            index = load_json(root, "model.safetensors.index.json")
            metadata = index["metadata"]
            self.assertIsInstance(metadata, dict)
            metadata["total_size"] += 8
            write_json(root, "model.safetensors.index.json", index)
            with self.assertRaisesRegex(
                SnapshotValidationError, "tensor payload bytes drifted"
            ):
                inspect_snapshot(root)

    def test_parameter_redistribution_preserving_total_is_rejected(self) -> None:
        with copied_snapshot() as root:
            repository = load_json(root, "repository.json")
            classes = repository["safetensors_parameter_classes"]
            self.assertIsInstance(classes, dict)
            parameters = classes["parameters"]
            self.assertIsInstance(parameters, dict)
            parameters["BF16"] += 1
            parameters["F32"] -= 1
            write_json(root, "repository.json", repository)
            with self.assertRaisesRegex(
                SnapshotValidationError, "parameter classes drifted"
            ):
                inspect_snapshot(root)


class ManifestBoundaryTests(unittest.TestCase):
    def test_oversized_sparse_manifest_is_rejected_before_read(self) -> None:
        with copied_snapshot() as root:
            with (root / "manifest.json").open("wb") as stream:
                stream.truncate(MAX_MANIFEST_BYTES + 1)
            with self.assertRaisesRegex(
                SnapshotValidationError,
                rf"manifest\.json: file exceeds {MAX_MANIFEST_BYTES}-byte safety limit",
            ):
                inspect_snapshot(root)

    def test_declared_small_actual_file_is_rejected_before_read(self) -> None:
        with copied_snapshot() as root:
            with (root / "config.json").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(
                SnapshotValidationError,
                r"config\.json: byte length mismatch",
            ):
                inspect_snapshot(root)

    def test_oversized_sparse_listed_file_is_rejected_before_read(self) -> None:
        with copied_snapshot() as root:
            with (root / "config.json").open("wb") as stream:
                stream.truncate(MAX_FILE_BYTES + 1)
            with self.assertRaisesRegex(
                SnapshotValidationError,
                rf"config\.json: file exceeds {MAX_FILE_BYTES}-byte safety limit",
            ):
                inspect_snapshot(root)

    def test_non_regular_snapshot_entry_is_rejected_without_opening(self) -> None:
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation is unavailable")
        with copied_snapshot() as root:
            path = root / "config.json"
            path.unlink()
            try:
                os.mkfifo(path)
            except OSError as exc:
                self.skipTest(f"FIFO creation unavailable: {exc}")
            with self.assertRaisesRegex(SnapshotValidationError, "non-file entry"):
                inspect_snapshot(root)

    def test_original_retrieval_window_cannot_be_relabelled(self) -> None:
        with copied_snapshot() as root:
            manifest = load_json(root, "manifest.json")
            window = manifest["retrieval_window_utc"]
            self.assertIsInstance(window, dict)
            window["end"] = "2026-08-08T15:49:13Z"
            (root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SnapshotValidationError,
                "original retrieval window identity drifted",
            ):
                inspect_snapshot(root)

    def test_manifest_byte_length_is_enforced(self) -> None:
        with copied_snapshot() as root:
            manifest = load_json(root, "manifest.json")
            files = manifest["files"]
            self.assertIsInstance(files, list)
            config_entry = next(
                entry
                for entry in files
                if isinstance(entry, dict) and entry.get("path") == "config.json"
            )
            config_entry["bytes"] += 1
            (root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SnapshotValidationError, "byte length mismatch"
            ):
                inspect_snapshot(root)

    def test_manifest_sha_is_enforced(self) -> None:
        with copied_snapshot() as root:
            path = root / "config.json"
            data = path.read_bytes()
            changed = data.replace(b'"fp4"', b'"fp3"', 1)
            self.assertEqual(len(data), len(changed))
            self.assertNotEqual(data, changed)
            path.write_bytes(changed)
            with self.assertRaisesRegex(SnapshotValidationError, "SHA-256 mismatch"):
                inspect_snapshot(root)

    def test_manifest_parent_path_is_rejected(self) -> None:
        with copied_snapshot() as root:
            manifest = load_json(root, "manifest.json")
            files = manifest["files"]
            self.assertIsInstance(files, list)
            config_entry = next(
                entry
                for entry in files
                if isinstance(entry, dict) and entry.get("path") == "config.json"
            )
            config_entry["path"] = "../config.json"
            (root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SnapshotValidationError, "unsafe path"):
                inspect_snapshot(root)

    def test_snapshot_symlink_is_rejected(self) -> None:
        with copied_snapshot() as root:
            path = root / "config.json"
            path.unlink()
            try:
                path.symlink_to(DEFAULT_SNAPSHOT / "config.json")
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            with self.assertRaisesRegex(SnapshotValidationError, "symlink"):
                inspect_snapshot(root)


if __name__ == "__main__":
    unittest.main()



