#!/usr/bin/env python3
"""Render the deterministic, source-backed M1 visual atlas."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import importlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys
import tempfile
from typing import Final, Sequence


SPEC_RELATIVE: Final = Path("docs/visuals/m1_atlas.spec.json")
REJECTION_NAME: Final = "m1_rejection_path.json"
MANIFEST_NAME: Final = "atlas.manifest.json"
TRANSCRIPT_LABEL: Final = (
    "EXACT VERIFIED CLI TRANSCRIPT - committed bytes; "
    "not an OS-terminal screenshot"
)
EXPECTED_PYTHON: Final = "3.12.3"
EXPECTED_PILLOW: Final = "12.3.0"
MAX_INPUT_BYTES: Final = 8 * 1024 * 1024
SVG_VIEW_BOX: Final = "0 0 1200 675"
SVG_NAMES: Final = (
    "m1-architecture.svg",
    "m1-topology.svg",
    "m1-expert-census.svg",
    "m1-shard-inventory.svg",
    "m1-byte-ledgers.svg",
    "m1-parameter-classes.svg",
    "m1-drift-boundary.svg",
)
MEDIA_NAMES: Final = (
    "m1-cli-inspect.png",
    *SVG_NAMES,
    "m1-rejection-path.gif",
)
COLORS: Final = {
    "background": "#0b1220",
    "panel": "#152238",
    "text": "#dbeafe",
    "blue": "#60a5fa",
    "green": "#34d399",
    "yellow": "#fbbf24",
    "red": "#f87171",
    "muted": "#94a3b8",
}
GIF_PALETTE: Final = (
    "#0b1220",
    "#152238",
    "#dbeafe",
    "#60a5fa",
    "#34d399",
    "#fbbf24",
    "#f87171",
    "#94a3b8",
)
GIF_DURATIONS: Final = (900, 900, 900, 1200)


class AtlasRenderError(RuntimeError):
    """The atlas input, toolchain, or deterministic output contract failed."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AtlasRenderError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_object(data: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtlasRenderError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise AtlasRenderError(f"{label} root must be an object")
    return value


def _read_regular(path: Path, *, maximum: int = MAX_INPUT_BYTES) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AtlasRenderError(f"cannot safely open {path.name}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AtlasRenderError(f"not a regular file: {path.name}")
        if metadata.st_size > maximum:
            raise AtlasRenderError(f"bounded read exceeded for {path.name}")
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
    if len(data) > maximum or len(data) != metadata.st_size:
        raise AtlasRenderError(f"input changed or exceeded bound: {path.name}")
    return data


def _atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise AtlasRenderError(f"unsafe output path: {path.name}")
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
        raise AtlasRenderError(f"cannot write {path.name}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise AtlasRenderError(f"{label} must be an object")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise AtlasRenderError(f"{label} must be a non-negative integer")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str:
        raise AtlasRenderError(f"{label} must be a string")
    return value


def _toolchain() -> tuple[object, object, object]:
    if platform.python_implementation() != "CPython":
        raise AtlasRenderError("visual renderer requires CPython")
    if platform.python_version() != EXPECTED_PYTHON:
        raise AtlasRenderError(
            f"visual renderer requires Python {EXPECTED_PYTHON}, "
            f"observed {platform.python_version()}"
        )
    pillow = importlib.import_module("PIL")
    if getattr(pillow, "__version__", None) != EXPECTED_PILLOW:
        raise AtlasRenderError(
            f"visual renderer requires Pillow {EXPECTED_PILLOW}, "
            f"observed {getattr(pillow, '__version__', 'unknown')}"
        )
    image = importlib.import_module("PIL.Image")
    image_draw = importlib.import_module("PIL.ImageDraw")
    image_font = importlib.import_module("PIL.ImageFont")
    font = image_font.load_default(size=14)
    if tuple(font.getname()) != ("Aileron", "Regular"):
        raise AtlasRenderError(
            f"embedded default font drifted: {font.getname()!r}"
        )
    return image, image_draw, font


def _verify_sources(
    root: Path, spec: dict[str, object]
) -> tuple[dict[str, bytes], dict[str, object]]:
    entries = spec.get("source_inputs")
    if type(entries) is not list or not entries:
        raise AtlasRenderError("spec source_inputs must be a non-empty array")
    sources: dict[str, bytes] = {}
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"source_inputs[{index}]")
        relative = _text(entry.get("path"), f"source_inputs[{index}].path")
        expected = _text(entry.get("sha256"), f"source_inputs[{index}].sha256")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise AtlasRenderError(f"unsafe source path: {relative}")
        data = _read_regular(root / path)
        if _sha(data) != expected:
            raise AtlasRenderError(f"source SHA-256 drifted: {relative}")
        sources[relative] = data

    raw_name = "evidence/raw/m1_target_genome.json"
    transcript_name = "evidence/raw/m1_target_genome.txt"
    report = _load_object(sources[raw_name], label=raw_name)
    canonical_compact = (
        json.dumps(report, sort_keys=True) + "\n"
    ).encode("utf-8")
    if sources[raw_name] != canonical_compact:
        raise AtlasRenderError("M1 JSON is not the exact canonical compact surface")
    transcript = sources[transcript_name]
    expected_command = b"$ PYTHONPATH=src python3 -m stratafold inspect-target\n"
    if not transcript.startswith(expected_command):
        raise AtlasRenderError("M1 transcript command line drifted")
    expected_pretty = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if transcript[len(expected_command):] != expected_pretty:
        raise AtlasRenderError("M1 transcript is not the exact pretty JSON surface")
    return sources, report


def _verify_spec(spec: dict[str, object]) -> list[str]:
    if spec.get("schema_version") != 1:
        raise AtlasRenderError("unsupported atlas spec schema")
    if spec.get("contract") != "stratafold-m1-visual-atlas":
        raise AtlasRenderError("wrong atlas contract")
    adoption = _mapping(spec.get("adoption"), "adoption")
    if adoption != {
        "generated_assets_committed": False,
        "readme_visual_claims_allowed": False,
        "review_required_before_adoption": True,
    }:
        raise AtlasRenderError("atlas adoption boundary drifted")
    inventory = spec.get("expected_inventory")
    if type(inventory) is not list or not all(type(item) is str for item in inventory):
        raise AtlasRenderError("expected_inventory must contain strings")
    expected = [
        REJECTION_NAME,
        MANIFEST_NAME,
        "m1-cli-inspect.png",
        *SVG_NAMES,
        "m1-rejection-path.gif",
    ]
    if inventory != expected or len(set(inventory)) != len(inventory):
        raise AtlasRenderError("atlas exact inventory drifted")
    return inventory


def _verify_rejection(data: bytes) -> dict[str, object]:
    record = _load_object(data, label=REJECTION_NAME)
    if record.get("experiment") != "m1-same-length-config-semantic-rejection":
        raise AtlasRenderError("rejection experiment identity drifted")
    mutation = _mapping(record.get("mutation"), "rejection.mutation")
    if (
        mutation.get("field") != "config.expert_dtype"
        or mutation.get("from") != "fp4"
        or mutation.get("to") != "fp3"
        or mutation.get("same_length") is not True
    ):
        raise AtlasRenderError("rejection mutation contract drifted")
    invocation = _mapping(record.get("invocation"), "rejection.invocation")
    stderr = _mapping(invocation.get("stderr"), "rejection.invocation.stderr")
    error = {
        "error": "config.expert_dtype: expected fp4, observed fp3",
        "status": "rejected",
    }
    if (
        invocation.get("returncode") != 2
        or _mapping(invocation.get("stdout"), "rejection.invocation.stdout").get("bytes")
        != 0
        or stderr.get("json") != error
        or stderr.get("canonical_text")
        != json.dumps(error, sort_keys=True) + "\n"
    ):
        raise AtlasRenderError("actual rejection invocation contract drifted")
    gate = _mapping(record.get("gate_result"), "rejection.gate_result")
    if (
        gate.get("classification") != "config-semantic-validation"
        or gate.get("semantic_gate") != "rejected"
        or gate.get("reviewed_identity_gates_reached") is not False
    ):
        raise AtlasRenderError("rejection gate classification drifted")
    security = _mapping(record.get("security"), "rejection.security")
    if (
        security.get("host_paths_recorded") is not False
        or security.get("timestamps_recorded") is not False
        or security.get("network_used") is not False
    ):
        raise AtlasRenderError("rejection security boundary drifted")
    return record


def _svg_text(
    x: int,
    y: int,
    value: object,
    *,
    size: int = 22,
    color: str | None = None,
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{color or COLORS["text"]}" '
        f'font-family="sans-serif" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}">{escape(str(value))}</text>'
    )


def _svg_rect(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    fill: str,
    stroke: str | None = None,
    radius: int = 12,
) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="2"' if stroke else ""
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}"{stroke_attr}/>'
    )


def _svg_line(
    x1: int, y1: int, x2: int, y2: int, *, color: str, width: int = 3
) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}"/>'
    )


def _svg_document(title: str, description: str, body: list[str]) -> bytes:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg viewBox="{SVG_VIEW_BOX}" role="img" '
            'aria-labelledby="atlas-title atlas-description">'
        ),
        f'<title id="atlas-title">{escape(title)}</title>',
        f'<desc id="atlas-description">{escape(description)}</desc>',
        _svg_rect(0, 0, 1200, 675, fill=COLORS["background"], radius=0),
        *body,
        "</svg>",
        "",
    ]
    encoded = "\n".join(parts).encode("utf-8")
    forbidden = (b"<script", b"foreignObject", b"http://", b"https://")
    if any(token in encoded for token in forbidden):
        raise AtlasRenderError(f"unsafe SVG content in {title}")
    return encoded


def _header(title: str, subtitle: str) -> list[str]:
    return [
        _svg_text(60, 62, title, size=34, weight=700),
        _svg_text(60, 94, subtitle, size=17, color=COLORS["muted"]),
        _svg_line(60, 114, 1140, 114, color=COLORS["panel"], width=2),
    ]


def _architecture(report: dict[str, object]) -> bytes:
    target = _mapping(report.get("target"), "target")
    body = _header(
        "M1 evidence architecture",
        "Every visible claim flows from the committed, pinned metadata snapshot.",
    )
    columns = [
        (
            60,
            "Pinned inputs",
            [
                "config.json",
                "tensor index",
                "repository projection",
                "verification receipt",
            ],
            COLORS["blue"],
        ),
        (
            420,
            "Dependency-free validation",
            [
                "manifest byte + SHA gates",
                "semantic topology checks",
                "cross-file reconciliation",
                "reviewed identity gates",
            ],
            COLORS["green"],
        ),
        (
            780,
            "Evidence surfaces",
            [
                "canonical CLI JSON",
                "exact CLI transcript",
                "source-backed atlas",
                "hash-bound manifest",
            ],
            COLORS["yellow"],
        ),
    ]
    for x, label, rows, color in columns:
        body.append(_svg_rect(x, 150, 300, 360, fill=COLORS["panel"], stroke=color))
        body.append(_svg_text(x + 24, 190, label, size=23, color=color, weight=700))
        for index, row in enumerate(rows):
            body.append(_svg_rect(x + 24, 220 + index * 62, 252, 42, fill=COLORS["background"]))
            body.append(_svg_text(x + 38, 248 + index * 62, row, size=16))
    body.extend(
        [
            _svg_line(360, 330, 420, 330, color=COLORS["muted"]),
            _svg_line(720, 330, 780, 330, color=COLORS["muted"]),
            _svg_text(
                60,
                560,
                f'{target["repository"]} @ {_text(target["revision"], "target.revision")[:12]}…',
                size=18,
                color=COLORS["text"],
            ),
            _svg_text(
                60,
                595,
                "Boundary: configuration declarations only; no shard payload inspected.",
                size=18,
                color=COLORS["yellow"],
            ),
            _svg_text(
                60,
                625,
                "Full checkpoint: NOT DOWNLOADED / NOT RUN. Hostwide state: not audited.",
                size=18,
                color=COLORS["red"],
                weight=700,
            ),
        ]
    )
    return _svg_document(
        "M1 evidence architecture",
        "Flow from pinned metadata inputs through dependency-free validation to exact evidence surfaces.",
        body,
    )


def _topology(report: dict[str, object]) -> bytes:
    topology = _mapping(report.get("topology"), "topology")
    values = [
        ("Hidden layers", _integer(topology.get("hidden_layers"), "hidden_layers")),
        ("Hidden size", _integer(topology.get("hidden_size"), "hidden_size")),
        (
            "Routed experts / layer",
            _integer(topology.get("routed_experts_per_layer"), "routed_experts"),
        ),
        (
            "Shared experts / layer",
            _integer(topology.get("shared_experts_per_layer"), "shared_experts"),
        ),
        (
            "Experts selected / token",
            _integer(topology.get("experts_selected_per_token"), "top-k"),
        ),
        ("Hash layers", _integer(topology.get("hash_layers"), "hash_layers")),
        (
            "Next-token-prediction layers",
            _integer(topology.get("next_token_prediction_layers"), "next layers"),
        ),
    ]
    dspark = topology.get("dspark_target_layer_ids")
    if dspark != [40, 41, 42]:
        raise AtlasRenderError("DSpark target layer declaration drifted")
    body = _header(
        "Declared M1 topology",
        "Source-reproduced from config.json; declarations are not payload observations.",
    )
    for index, (label, value) in enumerate(values):
        column = index % 4
        row = index // 4
        x = 60 + column * 270
        y = 150 + row * 170
        body.append(_svg_rect(x, y, 240, 135, fill=COLORS["panel"]))
        body.append(_svg_text(x + 20, y + 42, label, size=17, color=COLORS["muted"]))
        body.append(_svg_text(x + 20, y + 98, f"{value:,}", size=38, color=COLORS["blue"], weight=700))
    body.extend(
        [
            _svg_rect(870, 320, 240, 135, fill=COLORS["panel"]),
            _svg_text(890, 362, "DSpark target layers", size=17, color=COLORS["muted"]),
            _svg_text(890, 418, "40 · 41 · 42", size=31, color=COLORS["green"], weight=700),
            _svg_text(60, 535, "Representation declarations", size=21, weight=700),
            _svg_text(60, 570, "expert_dtype = fp4  ·  torch_dtype = bfloat16", size=21, color=COLORS["yellow"]),
            _svg_text(60, 606, "quant_method = fp8  ·  fmt = e4m3  ·  weight block = 128 × 128", size=19),
            _svg_text(60, 640, "Evidence tag: source-reproduced", size=16, color=COLORS["muted"]),
        ]
    )
    return _svg_document(
        "Declared M1 topology",
        "Exact topology values reproduced from the committed configuration metadata.",
        body,
    )


def _expert_census(report: dict[str, object]) -> bytes:
    index = _mapping(report.get("tensor_index"), "tensor_index")
    census = _mapping(index.get("expert_census"), "expert_census")
    rows = [
        (
            "Backbone routed",
            _integer(census.get("backbone_routed_expert_slots"), "backbone slots"),
            _integer(census.get("backbone_routed_tensor_entries"), "backbone entries"),
            COLORS["blue"],
        ),
        (
            "Backbone shared",
            _integer(census.get("backbone_shared_expert_slots"), "shared slots"),
            _integer(census.get("backbone_shared_tensor_entries"), "shared entries"),
            COLORS["green"],
        ),
        (
            "Attachment routed",
            _integer(census.get("attachment_routed_expert_slots"), "attachment slots"),
            _integer(census.get("attachment_routed_tensor_entries"), "attachment entries"),
            COLORS["yellow"],
        ),
        (
            "Attachment shared",
            _integer(census.get("attachment_shared_expert_slots"), "attachment shared slots"),
            _integer(census.get("attachment_shared_tensor_entries"), "attachment shared entries"),
            COLORS["red"],
        ),
    ]
    maximum = max(item[2] for item in rows)
    body = _header(
        "Expert tensor census",
        "Exact expert slots and tensor-name entries reproduced from the committed index.",
    )
    for index_number, (label, slots, entries, color) in enumerate(rows):
        y = 160 + index_number * 105
        width = max(4, round(650 * entries / maximum))
        body.extend(
            [
                _svg_text(60, y + 22, label, size=20, weight=700),
                _svg_text(310, y + 22, f"{slots:,} expert slots", size=18, color=COLORS["muted"]),
                _svg_rect(60, y + 42, 650, 24, fill=COLORS["panel"], radius=5),
                _svg_rect(60, y + 42, width, 24, fill=color, radius=5),
                _svg_text(740, y + 63, f"{entries:,} tensor entries", size=20, color=color, weight=700),
            ]
        )
    body.extend(
        [
            _svg_text(60, 602, "Bar length encodes tensor-name entries within this census.", size=16, color=COLORS["muted"]),
            _svg_text(60, 632, "Index declarations only; no shard header or tensor payload inspected.", size=18, color=COLORS["yellow"]),
        ]
    )
    return _svg_document(
        "Expert tensor census",
        "Four exact expert-slot and tensor-entry categories from the M1 tensor index.",
        body,
    )


def _shard_inventory(report: dict[str, object]) -> bytes:
    index = _mapping(report.get("tensor_index"), "tensor_index")
    shard_count = _integer(index.get("shard_count"), "shard_count")
    denominator = _integer(index.get("shard_denominator"), "shard_denominator")
    entries = _integer(index.get("tensor_entries"), "tensor_entries")
    ledger = _mapping(report.get("byte_ledgers"), "byte_ledgers")
    payload = _integer(
        _mapping(ledger.get("index_declared_tensor_payload"), "index payload").get("bytes"),
        "index payload bytes",
    )
    if shard_count != 48 or denominator != 48:
        raise AtlasRenderError("exact 48-of-48 shard inventory drifted")
    body = _header(
        "Declared shard inventory",
        "All 48 shard names are present in the committed index and repository projection.",
    )
    for shard in range(48):
        column = shard % 12
        row = shard // 12
        x = 60 + column * 89
        y = 155 + row * 72
        body.append(_svg_rect(x, y, 72, 50, fill=COLORS["panel"], stroke=COLORS["blue"], radius=7))
        body.append(_svg_text(x + 36, y + 32, f"{shard + 1:02d}", size=17, color=COLORS["blue"], weight=700, anchor="middle"))
    body.extend(
        [
            _svg_text(60, 490, f"{shard_count} / {denominator} declared weight shards", size=28, color=COLORS["green"], weight=700),
            _svg_text(60, 535, f"{entries:,} tensor-name entries", size=24),
            _svg_text(60, 575, f"{payload:,} index-declared tensor payload bytes", size=24),
            _svg_text(60, 615, "Inventory metadata only; shard files were not opened by the decoder.", size=18, color=COLORS["yellow"]),
        ]
    )
    return _svg_document(
        "Declared shard inventory",
        "A 48-cell inventory with exact tensor-name and declared payload totals.",
        body,
    )


def _byte_ledgers(report: dict[str, object]) -> bytes:
    ledgers = _mapping(report.get("byte_ledgers"), "byte_ledgers")
    api = _mapping(
        ledgers.get("api_reported_repository_artifact_bytes"), "API byte ledger"
    )
    weight = _integer(api.get("weight_shard_bytes"), "weight_shard_bytes")
    nonweight = _integer(api.get("non_weight_file_bytes"), "non_weight_file_bytes")
    all_bytes = _integer(api.get("all_repository_file_bytes"), "all bytes")
    payload = _integer(
        _mapping(ledgers.get("index_declared_tensor_payload"), "index payload").get("bytes"),
        "index payload",
    )
    gap = _integer(
        _mapping(ledgers.get("artifact_minus_index_payload_gap"), "gap").get("bytes"),
        "gap",
    )
    body = _header(
        "M1 byte ledgers",
        "Separate API-reported artifact bytes from index-declared tensor payload bytes.",
    )
    comparisons = [
        ("API weight-shard bytes", weight, COLORS["blue"]),
        ("Index-declared tensor payload", payload, COLORS["green"]),
    ]
    for index_number, (label, value, color) in enumerate(comparisons):
        y = 165 + index_number * 105
        width = round(900 * value / max(weight, payload))
        body.extend(
            [
                _svg_text(60, y, label, size=20),
                _svg_rect(60, y + 20, 900, 34, fill=COLORS["panel"], radius=6),
                _svg_rect(60, y + 20, width, 34, fill=color, radius=6),
                _svg_text(990, y + 47, f"{value:,}", size=18, color=color, weight=700),
            ]
        )
    body.extend(
        [
            _svg_rect(60, 385, 1080, 120, fill=COLORS["panel"], stroke=COLORS["yellow"]),
            _svg_text(85, 423, "Explicit inset — NOT TO SCALE", size=18, color=COLORS["yellow"], weight=700),
            _svg_text(85, 462, f"Unattributed difference: {gap:,} bytes", size=28, color=COLORS["yellow"], weight=700),
            _svg_text(570, 423, f"API non-weight files: {nonweight:,} bytes", size=18),
            _svg_text(570, 462, f"API all repository files: {all_bytes:,} bytes", size=18),
            _svg_text(60, 565, "Difference = API weight-shard bytes − index-declared tensor payload bytes.", size=18),
            _svg_text(60, 602, "Not measured compression and not proven container overhead.", size=20, color=COLORS["red"], weight=700),
            _svg_text(60, 635, "No artifact payload was inspected.", size=17, color=COLORS["muted"]),
        ]
    )
    return _svg_document(
        "M1 byte ledgers",
        "Exact byte-ledger comparison plus a clearly not-to-scale unattributed-difference inset.",
        body,
    )


def _parameter_classes(report: dict[str, object]) -> bytes:
    classes = _mapping(
        _mapping(
            report.get("api_reported_parameter_classes"), "parameter classes"
        ).get("counts_by_storage_class"),
        "counts_by_storage_class",
    )
    ordered = ("BF16", "F32", "F8_E4M3", "I64", "I8")
    values = [
        (label, _integer(classes.get(label), f"parameter class {label}"))
        for label in ordered
    ]
    total = _integer(classes.get("total"), "parameter class total")
    if sum(value for _, value in values) != total:
        raise AtlasRenderError("parameter class values do not sum to exact total")
    body = _header(
        "API-reported parameter classes",
        "Bar height is log10(parameters); every bar carries its exact count.",
    )
    baseline = 545
    top = 170
    for index_number, (label, value) in enumerate(values):
        logarithm = math.log10(value)
        height = round((logarithm / 12.0) * (baseline - top))
        x = 90 + index_number * 210
        y = baseline - height
        body.extend(
            [
                _svg_rect(x, y, 130, height, fill=COLORS["blue"], radius=5),
                _svg_text(x + 65, y - 42, f"{value:,}", size=17, color=COLORS["text"], weight=700, anchor="middle"),
                _svg_text(x + 65, y - 17, f"log10 = {logarithm:.4f}", size=14, color=COLORS["muted"], anchor="middle"),
                _svg_text(x + 65, 578, label, size=20, color=COLORS["green"], weight=700, anchor="middle"),
            ]
        )
    body.extend(
        [
            _svg_line(60, baseline, 1140, baseline, color=COLORS["muted"], width=2),
            _svg_text(60, 625, f"Exact total: {total:,} API-reported parameters", size=22, color=COLORS["yellow"], weight=700),
            _svg_text(1140, 650, "No pie or trend encoding · not recomputed from shard headers", size=15, color=COLORS["muted"], anchor="end"),
        ]
    )
    return _svg_document(
        "API-reported parameter classes",
        "Five exact parameter-class counts on a log10 scale, with no pie or trend encoding.",
        body,
    )


def _drift_boundary(record: dict[str, object]) -> bytes:
    mutation = _mapping(record.get("mutation"), "mutation")
    before = _mapping(mutation.get("config_before"), "config_before")
    after = _mapping(mutation.get("config_after"), "config_after")
    refreshed = _mapping(mutation.get("refreshed_manifest"), "refreshed_manifest")
    invocation = _mapping(record.get("invocation"), "invocation")
    stderr = _mapping(invocation.get("stderr"), "stderr")
    body = _header(
        "Actual M1 drift boundary",
        "A copied snapshot is changed by one same-length token, then inspected by the real CLI.",
    )
    stages = [
        (
            60,
            "1 · copied source",
            'expert_dtype = "fp4"',
            f'{before["bytes"]:,} config bytes',
            COLORS["blue"],
        ),
        (
            335,
            "2 · semantic drift",
            'expert_dtype = "fp3"',
            f'{after["bytes"]:,} config bytes · same length',
            COLORS["yellow"],
        ),
        (
            610,
            "3 · integrity refreshed",
            "manifest config entry",
            f'{refreshed["config_entry_bytes"]:,} bytes + new SHA-256',
            COLORS["green"],
        ),
        (
            885,
            "4 · actual rejection",
            f'process return code {invocation["returncode"]}',
            "stdout = 0 bytes",
            COLORS["red"],
        ),
    ]
    for x, title, primary, secondary, color in stages:
        body.extend(
            [
                _svg_rect(x, 165, 245, 245, fill=COLORS["panel"], stroke=color),
                _svg_text(x + 18, 205, title, size=18, color=color, weight=700),
                _svg_text(x + 18, 270, primary, size=17),
                _svg_text(x + 18, 320, secondary, size=15, color=COLORS["muted"]),
            ]
        )
    body.extend(
        [
            _svg_line(305, 287, 335, 287, color=COLORS["muted"]),
            _svg_line(580, 287, 610, 287, color=COLORS["muted"]),
            _svg_line(855, 287, 885, 287, color=COLORS["muted"]),
            _svg_rect(60, 455, 1070, 120, fill=COLORS["panel"], stroke=COLORS["red"]),
            _svg_text(85, 495, "Canonical stderr from actual inspect-target --json subprocess:", size=18, color=COLORS["muted"]),
            _svg_text(85, 535, _mapping(stderr.get("json"), "stderr.json").get("error"), size=23, color=COLORS["red"], weight=700),
            _svg_text(60, 622, "Rejected at config semantics before reviewed identity gates.", size=22, color=COLORS["yellow"], weight=700),
        ]
    )
    return _svg_document(
        "Actual M1 drift boundary",
        "The exact copied-snapshot mutation, refreshed manifest integrity, and actual CLI semantic rejection.",
        body,
    )


def _render_png(
    transcript: bytes, image_module: object, draw_module: object, font: object
) -> bytes:
    image = image_module.new("RGB", (1920, 3000), COLORS["background"])
    draw = draw_module.Draw(image)
    draw.rectangle((32, 32, 1888, 2968), fill=COLORS["panel"], outline=COLORS["blue"], width=3)
    draw.text((65, 60), TRANSCRIPT_LABEL, font=font, fill=COLORS["yellow"])
    lines = transcript.decode("utf-8", errors="strict").splitlines()
    y = 108
    for line_number, line in enumerate(lines, start=1):
        box = draw.textbbox((0, 0), line, font=font)
        if box[2] - box[0] > 1780:
            raise AtlasRenderError(f"transcript line {line_number} does not fit exactly")
        if y + 15 > 2935:
            raise AtlasRenderError("exact transcript does not fit fixed PNG height")
        draw.text((65, y), line, font=font, fill=COLORS["text"])
        y += 16
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _hex_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))


def _render_gif(
    record: dict[str, object],
    image_module: object,
    draw_module: object,
    font: object,
) -> bytes:
    mutation = _mapping(record.get("mutation"), "mutation")
    invocation = _mapping(record.get("invocation"), "invocation")
    stderr = _mapping(invocation.get("stderr"), "stderr")
    frames_spec = [
        (
            "1 / 4 · copied source",
            [
                'config.expert_dtype = "fp4"',
                f'config bytes = {_mapping(mutation["config_before"], "before")["bytes"]:,}',
                "Source: committed reviewed snapshot",
            ],
            3,
        ),
        (
            "2 / 4 · same-length drift",
            [
                'config.expert_dtype = "fp3"',
                f'config bytes = {_mapping(mutation["config_after"], "after")["bytes"]:,}',
                "Mutation length delta = 0 bytes",
            ],
            5,
        ),
        (
            "3 / 4 · integrity refreshed",
            [
                "Manifest config entry SHA-256 refreshed",
                f'entry bytes = {_mapping(mutation["refreshed_manifest"], "manifest")["config_entry_bytes"]:,}',
                "Manifest integrity can now pass",
            ],
            4,
        ),
        (
            "4 / 4 · actual semantic rejection",
            [
                f'process return code = {invocation["returncode"]}',
                "stdout = 0 bytes",
                str(_mapping(stderr["json"], "stderr.json")["error"]),
                "Reviewed identity gates reached = false",
            ],
            6,
        ),
    ]
    palette: list[int] = []
    for color in GIF_PALETTE:
        palette.extend(_hex_rgb(color))
    palette.extend([0] * (768 - len(palette)))
    frames = []
    for heading, rows, accent in frames_spec:
        frame = image_module.new("P", (960, 540), 0)
        frame.putpalette(palette)
        draw = draw_module.Draw(frame)
        draw.rectangle((30, 30, 930, 510), fill=1, outline=accent, width=4)
        draw.text((65, 70), heading, font=font, fill=accent)
        for index, row in enumerate(rows):
            draw.rectangle((65, 130 + index * 75, 895, 185 + index * 75), fill=0)
            draw.text((85, 150 + index * 75), row, font=font, fill=2)
        draw.text(
            (65, 475),
            "Actual rejection record · deterministic fixed frame",
            font=font,
            fill=7,
        )
        frames.append(frame)
    import io

    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=list(GIF_DURATIONS),
        loop=0,
        disposal=2,
        optimize=False,
    )
    return buffer.getvalue()


def _require_entry_inventory(destination: Path) -> None:
    entries = list(destination.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise AtlasRenderError("output directory contains a non-regular entry")
    actual = {entry.name for entry in entries}
    if actual != {REJECTION_NAME}:
        raise AtlasRenderError(
            "renderer requires an output directory containing only "
            f"{REJECTION_NAME}; observed {sorted(actual)!r}"
        )


def _require_final_inventory(destination: Path, inventory: list[str]) -> None:
    entries = list(destination.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise AtlasRenderError("final atlas contains a non-regular entry")
    actual = sorted(entry.name for entry in entries)
    if actual != sorted(inventory):
        raise AtlasRenderError(f"final atlas inventory drifted: {actual!r}")


def render_atlas(root: Path, output_dir: Path) -> tuple[str, ...]:
    root = root.resolve(strict=True)
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    if destination.is_symlink() or not destination.is_dir():
        raise AtlasRenderError("output directory must already be a regular directory")
    _require_entry_inventory(destination)

    spec_data = _read_regular(root / SPEC_RELATIVE)
    spec = _load_object(spec_data, label=SPEC_RELATIVE.as_posix())
    inventory = _verify_spec(spec)
    sources, report = _verify_sources(root, spec)
    rejection_data = _read_regular(destination / REJECTION_NAME)
    rejection = _verify_rejection(rejection_data)
    image_module, draw_module, font = _toolchain()

    transcript = sources["evidence/raw/m1_target_genome.txt"]
    generated: dict[str, bytes] = {}
    generated["m1-cli-inspect.png"] = _render_png(
        transcript, image_module, draw_module, font
    )
    generated["m1-architecture.svg"] = _architecture(report)
    generated["m1-topology.svg"] = _topology(report)
    generated["m1-expert-census.svg"] = _expert_census(report)
    generated["m1-shard-inventory.svg"] = _shard_inventory(report)
    generated["m1-byte-ledgers.svg"] = _byte_ledgers(report)
    generated["m1-parameter-classes.svg"] = _parameter_classes(report)
    generated["m1-drift-boundary.svg"] = _drift_boundary(rejection)
    generated["m1-rejection-path.gif"] = _render_gif(
        rejection, image_module, draw_module, font
    )
    if tuple(generated) != MEDIA_NAMES:
        raise AtlasRenderError("renderer media order or inventory drifted")
    for name in MEDIA_NAMES:
        _atomic_write(destination / name, generated[name])

    input_bindings = [
        {
            "path": SPEC_RELATIVE.as_posix(),
            "bytes": len(spec_data),
            "sha256": _sha(spec_data),
        }
    ]
    for raw_entry in spec["source_inputs"]:
        entry = _mapping(raw_entry, "source input")
        relative = _text(entry.get("path"), "source input path")
        data = sources[relative]
        input_bindings.append(
            {"path": relative, "bytes": len(data), "sha256": _sha(data)}
        )
    input_bindings.append(
        {
            "path": f"$ATLAS_OUTPUT/{REJECTION_NAME}",
            "bytes": len(rejection_data),
            "sha256": _sha(rejection_data),
        }
    )
    output_data = {REJECTION_NAME: rejection_data, **generated}
    output_bindings = [
        {
            "path": name,
            "bytes": len(output_data[name]),
            "sha256": _sha(output_data[name]),
            "kind": _mapping(spec["assets"], "assets")[name]["kind"],
        }
        for name in inventory
        if name != MANIFEST_NAME
    ]
    manifest = {
        "schema_version": 1,
        "contract": "stratafold-m1-visual-atlas-manifest",
        "status": "generated-not-adopted",
        "inputs": input_bindings,
        "outputs": output_bindings,
        "toolchain": spec["toolchain"],
        "deterministic_environment": spec["deterministic_environment"],
        "truth_boundary": spec["truth_boundary"],
        "adoption": spec["adoption"],
    }
    _atomic_write(destination / MANIFEST_NAME, _canonical_json(manifest))
    _require_final_inventory(destination, inventory)
    return tuple(inventory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="render the deterministic source-backed M1 visual atlas"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        outputs = render_atlas(root, args.output_dir)
    except (AtlasRenderError, OSError) as exc:
        print(f"render_m1_atlas: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "outputs": list(outputs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
