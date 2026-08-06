from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch

from stratafold.__main__ import main
from stratafold.target_snapshot import (
    DEFAULT_SNAPSHOT,
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
        topology = report["topology"]
        self.assertEqual(topology["evidence_tag"], "source-reproduced")
        self.assertEqual(topology["hidden_layers"], 43)
        self.assertEqual(topology["hidden_size"], 4096)
        self.assertEqual(topology["routed_experts_per_layer"], 256)
        self.assertEqual(topology["shared_experts_per_layer"], 1)
        self.assertEqual(topology["experts_selected_per_token"], 6)
        self.assertEqual(topology["hash_layers"], 3)
        self.assertEqual(topology["dspark_target_layer_ids"], [40, 41, 42])

        native = report["native_representation"]
        self.assertEqual(native["expert_dtype"], "fp4")
        self.assertEqual(native["other_paths_dtype"], "bfloat16")
        self.assertEqual(native["quantization_config"]["quant_method"], "fp8")
        self.assertEqual(native["quantization_config"]["fmt"], "e4m3")

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

        self.assertEqual(
            report["parameter_ledger"]["counts_by_storage_class"]["total"],
            304_180_418_494,
        )
        ledgers = report["byte_ledgers"]
        self.assertEqual(
            ledgers["target_tensor_payload"]["bytes"],
            166_878_536_440,
        )
        self.assertEqual(
            ledgers["target_repository_artifacts"]["weight_shard_bytes"],
            166_886_535_336,
        )
        self.assertEqual(
            ledgers["target_repository_artifacts"]["non_weight_file_bytes"],
            12_125_738,
        )
        self.assertEqual(
            ledgers["target_repository_artifacts"]["all_repository_file_bytes"],
            166_898_661_074,
        )
        self.assertEqual(
            ledgers["derived_container_overhead"]["bytes"],
            7_998_896,
        )
        self.assertEqual(
            ledgers["committed_metadata_snapshot"]["listed_file_bytes"],
            5_628_127,
        )
        self.assertEqual(report["safety"]["decoder_mode"], "offline")
        self.assertFalse(report["safety"]["remote_code_execution"])
        self.assertFalse(report["safety"]["weight_shards_opened"])
        self.assertEqual(
            report["safety"]["full_checkpoint"],
            "NOT DOWNLOADED / NOT RUN",
        )

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


class ManifestBoundaryTests(unittest.TestCase):
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

