from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from capture_m1_rejection import (  # noqa: E402
    AFTER_TOKEN,
    BEFORE_TOKEN,
    EXPECTED_ERROR,
    OUTPUT_NAME,
    SNAPSHOT_RELATIVE,
    capture_rejection,
)
from render_m1_atlas import (  # noqa: E402
    GIF_DURATIONS,
    GIF_PALETTE,
    MANIFEST_NAME,
    SPEC_RELATIVE,
    SVG_NAMES,
    TRANSCRIPT_LABEL,
    render_atlas,
)
from stratafold import target_snapshot  # noqa: E402


SPEC_PATH = ROOT / SPEC_RELATIVE
REQUIREMENTS_PATH = ROOT / "requirements" / "visuals.txt"
EXPECTED_INVENTORY = [
    "m1_rejection_path.json",
    "atlas.manifest.json",
    "m1-cli-inspect.png",
    "m1-architecture.svg",
    "m1-topology.svg",
    "m1-expert-census.svg",
    "m1-shard-inventory.svg",
    "m1-byte-ledgers.svg",
    "m1-parameter-classes.svg",
    "m1-drift-boundary.svg",
    "m1-rejection-path.gif",
]
EXPECTED_SOURCES = {
    "evidence/raw/m1_target_genome.json":
        "a931facf2167616cbcba9e08787ddaaf55625913128b578e27263cf185fc0b9d",
    "evidence/raw/m1_target_genome.txt":
        "962dcf208aa7043c052c2e7ff9aec17cddf5b61607740a3c19d3a5187119eeee",
    (
        "metadata/deepseek-v4-flash-0731/"
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062/manifest.json"
    ): "991b2fbb9212b8dd8f49686e3e5f2b510627e4c5403d529afd54b0b7ce48474e",
    (
        "metadata/deepseek-v4-flash-0731/"
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json"
    ): "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
    (
        "metadata/deepseek-v4-flash-0731/"
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062/"
        "model.safetensors.index.json"
    ): "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
    (
        "metadata/deepseek-v4-flash-0731/"
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062/repository.json"
    ): "6cacae22067d225351b46d30b3b4335db18b8941e342ac24ab945d81ebef4800",
    (
        "metadata/deepseek-v4-flash-0731/"
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062/repository.receipt.json"
    ): "fb7d2406fccd6f326cef18df829457eb98ab2c61dd77a9a0436fd4a962b4bbbf",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path.name} must be a JSON object")
    return value


class M1VisualContractTests(unittest.TestCase):
    def test_contract_is_exact_and_adopts_no_generated_assets(self) -> None:
        spec = _load(SPEC_PATH)
        self.assertEqual(spec["schema_version"], 1)
        self.assertEqual(spec["contract"], "stratafold-m1-visual-atlas")
        self.assertEqual(spec["status"], "contract-only")
        self.assertEqual(spec["expected_inventory"], EXPECTED_INVENTORY)
        self.assertEqual(
            spec["adoption"],
            {
                "generated_assets_committed": False,
                "readme_visual_claims_allowed": False,
                "review_required_before_adoption": True,
            },
        )
        self.assertEqual(
            {
                entry["path"]: entry["sha256"]
                for entry in spec["source_inputs"]
            },
            EXPECTED_SOURCES,
        )
        for relative, expected in EXPECTED_SOURCES.items():
            self.assertEqual(_sha((ROOT / relative).read_bytes()), expected)
        for name in EXPECTED_INVENTORY:
            self.assertFalse(
                (ROOT / "docs" / "visuals" / name).exists(),
                f"generated asset was prematurely adopted: {name}",
            )

    def test_toolchain_dependency_and_security_contract_are_exact(self) -> None:
        spec = _load(SPEC_PATH)
        toolchain = spec["toolchain"]
        self.assertEqual(
            toolchain["python"],
            {
                "implementation": "CPython",
                "version": "3.12.3",
                "setup_python_action_sha":
                    "5fda3b95a4ea91299a34e894583c3862153e4b97",
            },
        )
        self.assertEqual(toolchain["pillow"]["version"], "12.3.0")
        self.assertEqual(
            toolchain["pillow"]["wheel_filename"],
            (
                "pillow-12.3.0-cp312-cp312-manylinux_2_27_x86_64."
                "manylinux_2_28_x86_64.whl"
            ),
        )
        self.assertEqual(
            toolchain["pillow"]["wheel_sha256"],
            "78cb2c6865a35ab8ff8b75fd122f6033b92a62c82801110e48ddd6c936a45d91",
        )
        self.assertEqual(
            toolchain["default_font"],
            {
                "loader": "PIL.ImageFont.load_default",
                "size": 14,
                "family": "Aileron",
                "style": "Regular",
                "source": "embedded in the pinned Pillow wheel",
            },
        )
        requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8")
        self.assertIn("--only-binary=:all:", requirements)
        self.assertIn("--require-hashes", requirements)
        self.assertIn("Pillow==12.3.0", requirements)
        self.assertIn(
            "--hash=sha256:"
            "78cb2c6865a35ab8ff8b75fd122f6033b92a62c82801110e48ddd6c936a45d91",
            requirements,
        )
        self.assertEqual(
            spec["deterministic_environment"],
            {
                "PYTHONHASHSEED": "0",
                "LC_ALL": "C.UTF-8",
                "LANG": "C.UTF-8",
                "TZ": "UTC",
            },
        )
        self.assertEqual(
            spec["security"]["svg_forbidden"],
            ["script", "external URL", "foreignObject"],
        )
        self.assertEqual(
            spec["assets"]["m1-cli-inspect.png"]["label"],
            TRANSCRIPT_LABEL,
        )
        self.assertEqual(
            spec["assets"]["m1-parameter-classes.svg"]["scale"],
            "log10(parameters) with exact numeric labels; no pie or trend encoding",
        )

    def test_actual_rejection_capture_is_byte_deterministic_and_path_free(self) -> None:
        source_snapshot = ROOT / SNAPSHOT_RELATIVE
        original_hashes = {
            path.name: _sha(path.read_bytes())
            for path in source_snapshot.iterdir()
            if path.is_file()
        }
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path = Path(first)
            second_path = Path(second)
            self.assertEqual(capture_rejection(ROOT, first_path), OUTPUT_NAME)
            self.assertEqual(capture_rejection(ROOT, second_path), OUTPUT_NAME)
            first_bytes = (first_path / OUTPUT_NAME).read_bytes()
            second_bytes = (second_path / OUTPUT_NAME).read_bytes()
            self.assertEqual(first_bytes, second_bytes)
            self.assertNotIn(first.encode(), first_bytes)
            self.assertNotIn(second.encode(), second_bytes)
            self.assertNotIn(str(ROOT).encode(), first_bytes)
            record = json.loads(first_bytes)

        self.assertEqual(
            {
                path.name: _sha(path.read_bytes())
                for path in source_snapshot.iterdir()
                if path.is_file()
            },
            original_hashes,
        )
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(
            record["experiment"],
            "m1-same-length-config-semantic-rejection",
        )
        self.assertEqual(
            record["source_snapshot"]["manifest_sha256"],
            EXPECTED_SOURCES[(SNAPSHOT_RELATIVE / "manifest.json").as_posix()],
        )
        self.assertEqual(
            record["source_snapshot"]["config_sha256"],
            EXPECTED_SOURCES[(SNAPSHOT_RELATIVE / "config.json").as_posix()],
        )
        mutation = record["mutation"]
        self.assertEqual(
            (mutation["field"], mutation["from"], mutation["to"]),
            ("config.expert_dtype", "fp4", "fp3"),
        )
        self.assertIs(mutation["same_length"], True)
        self.assertEqual(
            mutation["config_before"]["bytes"],
            mutation["config_after"]["bytes"],
        )
        self.assertEqual(
            mutation["config_after"]["sha256"],
            mutation["refreshed_manifest"]["config_entry_sha256"],
        )
        self.assertEqual(
            mutation["config_after"]["bytes"],
            mutation["refreshed_manifest"]["config_entry_bytes"],
        )
        invocation = record["invocation"]
        self.assertEqual(invocation["returncode"], 2)
        self.assertEqual(invocation["stdout"]["bytes"], 0)
        self.assertEqual(invocation["stdout"]["text"], "")
        expected_error = {"error": EXPECTED_ERROR, "status": "rejected"}
        self.assertEqual(invocation["stderr"]["json"], expected_error)
        self.assertEqual(
            invocation["stderr"]["canonical_text"],
            json.dumps(expected_error, sort_keys=True) + "\n",
        )
        self.assertEqual(
            record["gate_result"],
            {
                "classification": "config-semantic-validation",
                "manifest_integrity_gate": "passed-after-config-entry-refresh",
                "semantic_gate": "rejected",
                "reviewed_identity_gates_reached": False,
                "reason": EXPECTED_ERROR,
            },
        )
        self.assertEqual(
            record["security"],
            {
                "network_used": False,
                "target_code_imported_or_executed": False,
                "weight_or_lfs_payload_bytes_read": 0,
                "host_paths_recorded": False,
                "timestamps_recorded": False,
            },
        )

    def test_rejection_mutation_and_refreshed_manifest_are_reproducible(self) -> None:
        snapshot = ROOT / SNAPSHOT_RELATIVE
        config_before = (snapshot / "config.json").read_bytes()
        self.assertEqual(config_before.count(BEFORE_TOKEN), 1)
        config_after = config_before.replace(BEFORE_TOKEN, AFTER_TOKEN, 1)
        self.assertEqual(len(config_before), len(config_after))
        manifest = _load(snapshot / "manifest.json")
        entries = [
            entry for entry in manifest["files"]
            if entry["path"] == "config.json"
        ]
        self.assertEqual(len(entries), 1)
        entries[0]["bytes"] = len(config_after)
        entries[0]["sha256"] = _sha(config_after)
        manifest_after = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            capture_rejection(ROOT, Path(directory))
            record = _load(Path(directory) / OUTPUT_NAME)
        mutation = record["mutation"]
        self.assertEqual(mutation["config_after"]["sha256"], _sha(config_after))
        self.assertEqual(
            mutation["refreshed_manifest"],
            {
                "bytes": len(manifest_after),
                "sha256": _sha(manifest_after),
                "config_entry_bytes": len(config_after),
                "config_entry_sha256": _sha(config_after),
            },
        )

    def test_semantic_validation_precedes_reviewed_identity_gates(self) -> None:
        source = inspect.getsource(target_snapshot.inspect_snapshot)
        semantic = source.index("_validate_config(")
        identities = source.index("_validate_reviewed_identities(")
        self.assertLess(semantic, identities)
        self.assertIn(
            "config.expert_dtype: expected fp4, observed",
            inspect.getsource(target_snapshot._validate_config),
        )


@unittest.skipUnless(
    importlib.util.find_spec("PIL") is not None,
    "Pillow is installed only in the hosted visual-atlas job",
)
class M1VisualRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import PIL

        if (
            sys.implementation.name != "cpython"
            or sys.version_info[:3] != (3, 12, 3)
            or PIL.__version__ != "12.3.0"
        ):
            raise unittest.SkipTest(
                "exact CPython 3.12.3 + Pillow 12.3.0 toolchain required"
            )

    def _render_pair(self, parent: Path) -> tuple[Path, Path]:
        first = parent / "first"
        second = parent / "second"
        first.mkdir()
        second.mkdir()
        capture_rejection(ROOT, first)
        capture_rejection(ROOT, second)
        self.assertEqual(tuple(EXPECTED_INVENTORY), render_atlas(ROOT, first))
        self.assertEqual(tuple(EXPECTED_INVENTORY), render_atlas(ROOT, second))
        return first, second

    def test_two_atlases_are_byte_identical_and_manifest_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second = self._render_pair(Path(directory))
            self.assertEqual(
                sorted(path.name for path in first.iterdir()),
                sorted(EXPECTED_INVENTORY),
            )
            for name in EXPECTED_INVENTORY:
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                    name,
                )

            manifest = _load(first / MANIFEST_NAME)
            self.assertEqual(manifest["status"], "generated-not-adopted")
            self.assertEqual(len(manifest["inputs"]), 9)
            self.assertEqual(len(manifest["outputs"]), 10)
            self.assertNotIn(
                MANIFEST_NAME,
                {entry["path"] for entry in manifest["outputs"]},
            )
            self.assertEqual(
                [entry["path"] for entry in manifest["outputs"]],
                [name for name in EXPECTED_INVENTORY if name != MANIFEST_NAME],
            )
            for binding in manifest["outputs"]:
                data = (first / binding["path"]).read_bytes()
                self.assertEqual(binding["bytes"], len(data))
                self.assertEqual(binding["sha256"], _sha(data))
            for binding in manifest["inputs"]:
                if binding["path"].startswith("$ATLAS_OUTPUT/"):
                    data = first / binding["path"].split("/", 1)[1]
                else:
                    data = ROOT / binding["path"]
                raw = data.read_bytes()
                self.assertEqual(binding["bytes"], len(raw))
                self.assertEqual(binding["sha256"], _sha(raw))

    def test_png_svg_and_gif_contracts(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "atlas"
            output.mkdir()
            capture_rejection(ROOT, output)
            render_atlas(ROOT, output)

            with Image.open(output / "m1-cli-inspect.png") as png:
                self.assertEqual(png.size, (1920, 3000))
                self.assertEqual(png.mode, "RGB")
                self.assertEqual(png.info, {})

            for name in SVG_NAMES:
                data = (output / name).read_bytes()
                lowered = data.lower()
                self.assertNotIn(b"<script", lowered)
                self.assertNotIn(b"foreignobject", lowered)
                self.assertNotIn(b"http://", lowered)
                self.assertNotIn(b"https://", lowered)
                root = ET.fromstring(data)
                self.assertEqual(root.tag, "svg")
                self.assertEqual(root.attrib["viewBox"], "0 0 1200 675")
                self.assertIsNotNone(root.find("title"))
                self.assertIsNotNone(root.find("desc"))
                self.assertTrue(root.find("title").text)
                self.assertTrue(root.find("desc").text)

            parameter_svg = (
                output / "m1-parameter-classes.svg"
            ).read_text(encoding="utf-8")
            for label in (
                "1,483,567,488",
                "37,741,630",
                "6,304,038,912",
                "2,327,040",
                "296,352,743,424",
                "304,180,418,494",
                "log10(parameters)",
                "No pie or trend encoding",
            ):
                self.assertIn(label, parameter_svg)
            byte_svg = (output / "m1-byte-ledgers.svg").read_text(encoding="utf-8")
            self.assertIn("7,998,896", byte_svg)
            self.assertIn("NOT TO SCALE", byte_svg)
            self.assertIn("not proven container overhead", byte_svg)
            drift_svg = (output / "m1-drift-boundary.svg").read_text(encoding="utf-8")
            self.assertIn(EXPECTED_ERROR, drift_svg)
            self.assertIn("before reviewed identity gates", drift_svg)

            with Image.open(output / "m1-rejection-path.gif") as gif:
                self.assertEqual(gif.size, (960, 540))
                self.assertEqual(gif.n_frames, 4)
                self.assertEqual(gif.info.get("loop"), 0)
                self.assertNotIn("comment", gif.info)
                self.assertNotIn("exif", gif.info)
                observed_durations = []
                for frame in range(gif.n_frames):
                    gif.seek(frame)
                    observed_durations.append(gif.info["duration"])
                    if frame == 0:
                        expected_palette = []
                        for color in GIF_PALETTE:
                            expected_palette.extend(
                                int(color[index:index + 2], 16)
                                for index in (1, 3, 5)
                            )
                        self.assertEqual(
                            gif.getpalette()[:len(expected_palette)],
                            expected_palette,
                        )
                self.assertEqual(observed_durations, list(GIF_DURATIONS))


if __name__ == "__main__":
    unittest.main()
