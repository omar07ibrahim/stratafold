#!/usr/bin/env python3
"""Strict offline verifier for the adopted M1 evidence surfaces.

Only committed, bounded regular files are read.  This module has no network
transport and does not execute target-side code or inspect weight payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Final, Iterable


DEFAULT_ROOT: Final = Path(__file__).resolve().parents[1]
TARGET_REPOSITORY: Final = "deepseek-ai/DeepSeek-V4-Flash-0731"
TARGET_REVISION: Final = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
SOURCE_HEAD: Final = "b39e31359f03021357bfa350c9cdf77b0bd24505"
EVIDENCE_DIR: Final = Path("evidence/raw")
JSON_PATH: Final = EVIDENCE_DIR / "m1_target_genome.json"
TEXT_PATH: Final = EVIDENCE_DIR / "m1_target_genome.txt"
PROVENANCE_PATH: Final = EVIDENCE_DIR / "m1_target_genome.provenance.json"
WORKFLOW_PATH: Final = Path(".github/workflows/ci.yml")
MANIFEST_PATH: Final = Path(
    "metadata/deepseek-v4-flash-0731/"
    "7872f01b1d1fe23eabc4c98b48bffcef5a386062/manifest.json"
)
RECEIPT_PATH: Final = Path(
    "metadata/deepseek-v4-flash-0731/"
    "7872f01b1d1fe23eabc4c98b48bffcef5a386062/repository.receipt.json"
)
EXPECTED_M1_NAMES: Final = frozenset(
    {
        "m1_target_genome.json",
        "m1_target_genome.txt",
        "m1_target_genome.provenance.json",
    }
)
MAX_EVIDENCE_BYTES: Final = 64 * 1024
MAX_SOURCE_BYTES: Final = 64 * 1024
_READ_CHUNK_BYTES: Final = 16 * 1024

EXPECTED_JSON_BYTES: Final = 5_674
EXPECTED_JSON_SHA256: Final = (
    "a931facf2167616cbcba9e08787ddaaf55625913128b578e27263cf185fc0b9d"
)
EXPECTED_TEXT_BYTES: Final = 6_606
EXPECTED_TEXT_SHA256: Final = (
    "962dcf208aa7043c052c2e7ff9aec17cddf5b61607740a3c19d3a5187119eeee"
)
EXPECTED_PROVENANCE_BYTES: Final = 4_240
EXPECTED_PROVENANCE_SHA256: Final = (
    "ad034d745fca5d7da58ba1becac43a5aba13d277df27f58d2dfffbbf3c8729e2"
)
EXPECTED_ARCHIVE_BYTES: Final = 12_568
EXPECTED_ARCHIVE_SHA256: Final = (
    "110e0c1d65bb4f17ee5d72f210abe8c37f0ca1538072a05caf6ad95ff94bf1d8"
)
EXPECTED_INPUT_SHA256: Final = (
    "16dd1cea2018d8af2a84922bc6fff22a7988dbb4ecc58d9c92486fa01b178291"
)
EXPECTED_REPOSITORY_SHA256: Final = (
    "6cacae22067d225351b46d30b3b4335db18b8941e342ac24ab945d81ebef4800"
)
EXPECTED_BODY_SHA256: Final = (
    "cc906889b269c1a97632ed62bfd286b304b9602219064fe37feda529e3cd119c"
)
EXPECTED_VERIFICATION_TIMESTAMP: Final = "2026-08-08T15:49:13Z"
EXPECTED_RESPONSE_DATE: Final = "Sat, 08 Aug 2026 15:49:13 GMT"
TRANSCRIPT_COMMAND: Final = (
    b"$ PYTHONPATH=src python3 -m stratafold inspect-target\n"
)
EXPECTED_REPORT_KEYS: Final = frozenset(
    {
        "api_reported_parameter_classes",
        "byte_ledgers",
        "declared_representation",
        "evidence_level",
        "repository_projection_verification",
        "safety",
        "schema_version",
        "status",
        "target",
        "tensor_index",
        "topology",
    }
)
EXPECTED_LIMITATIONS: Final = (
    "The GitHub Actions artifact ZIP is ephemeral and is not committed; only its "
    "exact byte identity, archive structure, and adopted payload identities are "
    "recorded.",
    "GitHub Actions run, job, and artifact provenance is CI metadata, not an "
    "independent attestation of the model, upstream API response, or capture-time "
    "host state.",
)
EXPECTED_RECEIPT_LIMITATIONS: Final = (
    "The full API response body is deliberately not committed.",
    "Values of excluded fields cannot be reconstructed or audited offline from "
    "this receipt.",
    "A later full response body, including its field and sibling sequence order, "
    "may drift while the stable projection remains unchanged.",
    "This receipt covers repository metadata only.",
    "No model weight, LFS object, safetensors payload, or target-side code was "
    "requested, downloaded, opened, imported, or executed during this verification.",
    "This post-capture check verifies that selected stable fields reconstruct the "
    "earlier committed repository.json; it does not establish provenance for the "
    "original raw response used on 2026-08-06.",
)
EXPECTED_PROVENANCE: Final = json.loads(r'''{"adoption":{"policy":"Only the reviewed files from the pinned successful GitHub Actions artifact were adopted after independent in-memory archive and payload verification.","source_head":"b39e31359f03021357bfa350c9cdf77b0bd24505"},"artifact":{"archive":{"bytes":12568,"committed":false,"entries":[{"archive_entry":"m1_target_genome.json","bytes":5674,"compressed_bytes":5674,"compression_method":"stored","crc32":"566b7837","dos_datetime":[2026,8,8,16,12,16],"sha256":"a931facf2167616cbcba9e08787ddaaf55625913128b578e27263cf185fc0b9d","unix_mode":"100644"},{"archive_entry":"m1_target_genome.txt","bytes":6606,"compressed_bytes":6606,"compression_method":"stored","crc32":"5de42a87","dos_datetime":[2026,8,8,16,12,16],"sha256":"962dcf208aa7043c052c2e7ff9aec17cddf5b61607740a3c19d3a5187119eeee","unix_mode":"100644"}],"sha256":"110e0c1d65bb4f17ee5d72f210abe8c37f0ca1538072a05caf6ad95ff94bf1d8"},"created_at":"2026-08-08T16:12:19Z","expired_at_adoption":false,"expires_at":"2026-08-09T16:12:18Z","id":9024264228,"name":"m1-target-genome-31266366484","reported_size_bytes":12568,"updated_at":"2026-08-08T16:12:19Z"},"ci":{"job":{"completed_at":"2026-08-08T16:12:20Z","conclusion":"success","id":93125047136,"name":"dependency-free-control-plane","started_at":"2026-08-08T16:12:07Z","status":"completed"},"repository":{"full_name":"omar07ibrahim/stratafold","id":1321445804},"run":{"attempt":1,"conclusion":"success","created_at":"2026-08-08T16:12:05Z","event":"pull_request","head_branch":"feat/m1-target-genome","head_sha":"b39e31359f03021357bfa350c9cdf77b0bd24505","id":31266366484,"number":9,"started_at":"2026-08-08T16:12:05Z","status":"completed","updated_at":"2026-08-08T16:12:21Z"},"workflow":{"bytes":2058,"git_blob_id":"f4ca5765dcbd437ed1a7d711eb7f5d0cdb6d14e6","id":326041704,"name":"ci","path":".github/workflows/ci.yml","sha256":"e317d167f6b15cacc87ef601a6a87db26972bbb304c32807add02adb5132b7e4"}},"evidence_files":[{"artifact_entry":"m1_target_genome.json","bytes":5674,"path":"evidence/raw/m1_target_genome.json","sha256":"a931facf2167616cbcba9e08787ddaaf55625913128b578e27263cf185fc0b9d","unix_mode":"100644"},{"artifact_entry":"m1_target_genome.txt","bytes":6606,"path":"evidence/raw/m1_target_genome.txt","sha256":"962dcf208aa7043c052c2e7ff9aec17cddf5b61607740a3c19d3a5187119eeee","unix_mode":"100644"}],"limitations":["The GitHub Actions artifact ZIP is ephemeral and is not committed; only its exact byte identity, archive structure, and adopted payload identities are recorded.","GitHub Actions run, job, and artifact provenance is CI metadata, not an independent attestation of the model, upstream API response, or capture-time host state."],"schema_version":1,"source_snapshot_identities":{"manifest":{"bytes":2581,"git_blob_id":"1b3e823e3b9d82cf976f92307d59c21f5beb22ff","path":"metadata/deepseek-v4-flash-0731/7872f01b1d1fe23eabc4c98b48bffcef5a386062/manifest.json","sha256":"991b2fbb9212b8dd8f49686e3e5f2b510627e4c5403d529afd54b0b7ce48474e"},"repository_receipt":{"bytes":24578,"git_blob_id":"0644f29b0cab79a2d833163dcb74f82c943d52b6","path":"metadata/deepseek-v4-flash-0731/7872f01b1d1fe23eabc4c98b48bffcef5a386062/repository.receipt.json","sha256":"fb7d2406fccd6f326cef18df829457eb98ab2c61dd77a9a0436fd4a962b4bbbf"}}}''')

_CREDENTIAL_PATTERNS: Final = {
    "private-key": re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "github-token": re.compile(rb"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),
    "github-fine-grained-token": re.compile(
        rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"
    ),
    "huggingface-token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "openai-api-key": re.compile(
        rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b"
    ),
}
_ISO_TIMESTAMP = re.compile(
    r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\b"
)
_HTTP_DATE = re.compile(
    r"\b[A-Z][a-z]{2}, [0-9]{2} [A-Z][a-z]{2} [0-9]{4} "
    r"[0-9]{2}:[0-9]{2}:[0-9]{2} GMT\b"
)
_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_RUNTIME_KEYS = frozenset(
    {
        "captured_at",
        "captured_at_utc",
        "created_at",
        "generated_at",
        "generated_at_utc",
        "run_started_at",
        "runtime_timestamp",
        "updated_at",
    }
)


class EvidenceVerificationError(ValueError):
    """The adopted evidence violates its offline verification contract."""


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceVerificationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _no_constant(value: str) -> object:
    raise EvidenceVerificationError(f"non-finite JSON number is forbidden: {value}")


def loads_json_strict(data: bytes, *, label: str) -> object:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EvidenceVerificationError(f"{label}: expected UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicates,
            parse_constant=_no_constant,
        )
    except EvidenceVerificationError:
        raise
    except json.JSONDecodeError as exc:
        raise EvidenceVerificationError(
            f"{label}: invalid JSON: {exc.msg}"
        ) from exc


def _obj(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise EvidenceVerificationError(f"{label}: expected an object")
    return value


def _arr(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise EvidenceVerificationError(f"{label}: expected an array")
    return value


def _str(value: object, label: str, *, expected: str | None = None) -> str:
    if type(value) is not str or not value:
        raise EvidenceVerificationError(f"{label}: expected a non-empty string")
    if expected is not None and value != expected:
        raise EvidenceVerificationError(
            f"{label}: expected {expected!r}, observed {value!r}"
        )
    return value


def _int(value: object, label: str, *, expected: int | None = None) -> int:
    if type(value) is not int or value < 0:
        raise EvidenceVerificationError(
            f"{label}: expected a non-negative integer; booleans are not integers"
        )
    if expected is not None and value != expected:
        raise EvidenceVerificationError(
            f"{label}: expected {expected}, observed {value}"
        )
    return value


def _bool(value: object, label: str, *, expected: bool) -> bool:
    if type(value) is not bool or value is not expected:
        raise EvidenceVerificationError(
            f"{label}: expected {str(expected).lower()}"
        )
    return value


def _keys(value: dict[str, object], expected: Iterable[str], label: str) -> None:
    wanted = set(expected)
    actual = set(value)
    if actual != wanted:
        raise EvidenceVerificationError(
            f"{label}: unknown schema; "
            f"missing={sorted(wanted - actual)}, unexpected={sorted(actual - wanted)}"
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_id(data: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _canonical_compact(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def _canonical_pretty(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _walk(value: object) -> Iterable[tuple[str | None, object]]:
    if type(value) is dict:
        for key, child in value.items():
            yield key, child
            yield from _walk(child)
    elif type(value) is list:
        for child in value:
            yield None, child
            yield from _walk(child)


def _validate_no_environment_leaks(
    report: dict[str, object],
    json_bytes: bytes,
    text_bytes: bytes,
) -> None:
    combined = json_bytes + text_bytes
    for label, pattern in _CREDENTIAL_PATTERNS.items():
        if pattern.search(combined):
            raise EvidenceVerificationError(
                f"evidence contains credential-shaped text: {label}"
            )

    iso_timestamps: set[str] = set()
    http_dates: set[str] = set()
    for key, value in _walk(report):
        if key in _RUNTIME_KEYS:
            raise EvidenceVerificationError(
                f"evidence contains runtime timestamp field: {key}"
            )
        if type(value) is not str:
            continue
        if (
            value.startswith("/")
            or value.startswith("file://")
            or _DRIVE_PATH.match(value) is not None
        ):
            raise EvidenceVerificationError(
                "evidence contains an absolute host path"
            )
        iso_timestamps.update(_ISO_TIMESTAMP.findall(value))
        http_dates.update(_HTTP_DATE.findall(value))

    if iso_timestamps != {EXPECTED_VERIFICATION_TIMESTAMP}:
        raise EvidenceVerificationError(
            "evidence timestamp inventory does not match the pinned receipt"
        )
    if http_dates != {EXPECTED_RESPONSE_DATE}:
        raise EvidenceVerificationError(
            "evidence HTTP-date inventory does not match the pinned receipt"
        )


def _validate_report(report: dict[str, object]) -> None:
    _keys(report, EXPECTED_REPORT_KEYS, "m1_target_genome.json")
    _int(report["schema_version"], "report.schema_version", expected=2)
    _str(report["status"], "report.status", expected="validated")
    _str(
        report["evidence_level"],
        "report.evidence_level",
        expected="pinned-official-metadata",
    )
    target = _obj(report["target"], "report.target")
    _keys(target, {"repository", "revision"}, "report.target")
    _str(
        target["repository"],
        "report.target.repository",
        expected=TARGET_REPOSITORY,
    )
    _str(target["revision"], "report.target.revision", expected=TARGET_REVISION)

    verification = _obj(
        report["repository_projection_verification"],
        "report.repository_projection_verification",
    )
    _keys(
        verification,
        {
            "capture_kind",
            "evidence_tag",
            "limitations",
            "projection_input",
            "repository_projection",
            "response",
            "safety",
            "scope",
            "source",
            "status",
            "target",
            "verification_timestamp_utc",
        },
        "report.repository_projection_verification",
    )
    _str(
        verification["capture_kind"],
        "report.projection.capture_kind",
        expected="post-capture-api-projection-verification",
    )
    _str(
        verification["evidence_tag"],
        "report.projection.evidence_tag",
        expected="source-reproduced",
    )
    _str(
        verification["status"],
        "report.projection.status",
        expected="verified",
    )
    _str(
        verification["verification_timestamp_utc"],
        "report.projection.verification_timestamp_utc",
        expected=EXPECTED_VERIFICATION_TIMESTAMP,
    )
    if _arr(verification["limitations"], "report.projection.limitations") != list(
        EXPECTED_RECEIPT_LIMITATIONS
    ):
        raise EvidenceVerificationError(
            "report projection limitations do not match the receipt"
        )
    if verification["source"] != ["repository.receipt.json", "repository.json"]:
        raise EvidenceVerificationError(
            "report projection sources do not match"
        )
    _str(verification["scope"], "report.projection.scope")

    projection_target = _obj(
        verification["target"],
        "report.projection.target",
    )
    if projection_target != target:
        raise EvidenceVerificationError(
            "report projection target does not match the report target"
        )

    projection_input = _obj(
        verification["projection_input"],
        "report.projection.projection_input",
    )
    expected_input = {
        "bytes": 14_491,
        "sequence_order": "captured-but-non-semantic-and-untrusted",
        "sha256": EXPECTED_INPUT_SHA256,
        "sibling_count": 74,
    }
    if projection_input != expected_input:
        raise EvidenceVerificationError(
            "report projection input identity does not match"
        )

    repository_projection = _obj(
        verification["repository_projection"],
        "report.projection.repository_projection",
    )
    expected_projection = {
        "bytes": 22_284,
        "path": "repository.json",
        "sha256": EXPECTED_REPOSITORY_SHA256,
        "sibling_order": "lexicographic-by-rfilename",
    }
    if repository_projection != expected_projection:
        raise EvidenceVerificationError(
            "report repository projection identity does not match"
        )

    response = _obj(verification["response"], "report.projection.response")
    expected_response = {
        "body_bytes": 16_362,
        "body_committed": False,
        "body_sha256": EXPECTED_BODY_SHA256,
        "content_type": "application/json; charset=utf-8",
        "date": EXPECTED_RESPONSE_DATE,
        "status": 200,
    }
    if response != expected_response:
        raise EvidenceVerificationError(
            "report projection response identity does not match"
        )

    safety = _obj(verification["safety"], "report.projection.safety")
    expected_safety = {
        "metadata_only": True,
        "response_body_committed": False,
        "target_code_imported_or_executed": False,
        "weight_or_lfs_payload_bytes_read": 0,
    }
    if safety != expected_safety:
        raise EvidenceVerificationError(
            "report projection safety facts do not match"
        )


def _validate_surfaces(
    json_bytes: bytes,
    text_bytes: bytes,
) -> dict[str, object]:
    report = _obj(
        loads_json_strict(json_bytes, label=str(JSON_PATH)),
        str(JSON_PATH),
    )
    if _canonical_compact(report) != json_bytes:
        raise EvidenceVerificationError(
            "m1_target_genome.json is not canonical compact JSON"
        )
    if not text_bytes.startswith(TRANSCRIPT_COMMAND):
        raise EvidenceVerificationError(
            "m1_target_genome.txt transcript command does not match"
        )
    pretty_bytes = text_bytes[len(TRANSCRIPT_COMMAND):]
    pretty_report = _obj(
        loads_json_strict(pretty_bytes, label=str(TEXT_PATH)),
        str(TEXT_PATH),
    )
    if pretty_report != report:
        raise EvidenceVerificationError(
            "compact and pretty M1 JSON surfaces disagree"
        )
    if _canonical_pretty(pretty_report) != pretty_bytes:
        raise EvidenceVerificationError(
            "m1_target_genome.txt is not canonical pretty JSON"
        )
    _validate_report(report)
    _validate_no_environment_leaks(report, json_bytes, text_bytes)
    return report


def _validate_provenance(
    provenance_bytes: bytes,
    json_bytes: bytes,
    text_bytes: bytes,
    manifest_bytes: bytes,
    receipt_bytes: bytes,
) -> dict[str, object]:
    provenance = _obj(
        loads_json_strict(provenance_bytes, label=str(PROVENANCE_PATH)),
        str(PROVENANCE_PATH),
    )
    _keys(
        provenance,
        {
            "adoption",
            "artifact",
            "ci",
            "evidence_files",
            "limitations",
            "schema_version",
            "source_snapshot_identities",
        },
        str(PROVENANCE_PATH),
    )
    _int(
        provenance["schema_version"],
        "provenance.schema_version",
        expected=1,
    )
    if _canonical_pretty(provenance) != provenance_bytes:
        raise EvidenceVerificationError(
            "M1 provenance is not canonical pretty JSON"
        )
    limitations = _arr(provenance["limitations"], "provenance.limitations")
    if limitations != list(EXPECTED_LIMITATIONS):
        raise EvidenceVerificationError(
            "M1 provenance limitations do not match"
        )
    if provenance != EXPECTED_PROVENANCE:
        raise EvidenceVerificationError(
            "M1 provenance facts or nested schema drifted"
        )

    expected_sources = (
        (
            provenance["source_snapshot_identities"]["manifest"],
            manifest_bytes,
            str(MANIFEST_PATH),
            2_581,
            "991b2fbb9212b8dd8f49686e3e5f2b510627e4c5403d529afd54b0b7ce48474e",
            "1b3e823e3b9d82cf976f92307d59c21f5beb22ff",
        ),
        (
            provenance["source_snapshot_identities"]["repository_receipt"],
            receipt_bytes,
            str(RECEIPT_PATH),
            24_578,
            "fb7d2406fccd6f326cef18df829457eb98ab2c61dd77a9a0436fd4a962b4bbbf",
            "0644f29b0cab79a2d833163dcb74f82c943d52b6",
        ),
    )
    for recorded, data, path, size, sha256, blob_id in expected_sources:
        recorded_object = _obj(recorded, f"provenance identity for {path}")
        if recorded_object["path"] != path:
            raise EvidenceVerificationError(
                f"provenance source path does not match: {path}"
            )
        if len(data) != size or _sha256(data) != sha256:
            raise EvidenceVerificationError(
                f"committed source identity drifted: {path}"
            )
        if _git_blob_id(data) != blob_id:
            raise EvidenceVerificationError(
                f"committed source Git blob identity drifted: {path}"
            )

    manifest = _obj(
        loads_json_strict(manifest_bytes, label=str(MANIFEST_PATH)),
        str(MANIFEST_PATH),
    )
    _int(manifest.get("schema_version"), "manifest.schema_version", expected=2)
    receipt = _obj(
        loads_json_strict(receipt_bytes, label=str(RECEIPT_PATH)),
        str(RECEIPT_PATH),
    )
    _int(receipt.get("schema_version"), "receipt.schema_version", expected=1)
    _str(
        receipt.get("capture_kind"),
        "receipt.capture_kind",
        expected="post-capture-api-projection-verification",
    )

    evidence_identities = {
        "evidence/raw/m1_target_genome.json": (
            len(json_bytes),
            _sha256(json_bytes),
        ),
        "evidence/raw/m1_target_genome.txt": (
            len(text_bytes),
            _sha256(text_bytes),
        ),
    }
    for raw in _arr(provenance["evidence_files"], "provenance.evidence_files"):
        item = _obj(raw, "provenance.evidence_file")
        path = _str(item.get("path"), "provenance.evidence_file.path")
        if path not in evidence_identities:
            raise EvidenceVerificationError(
                f"provenance names unexpected evidence file: {path}"
            )
        size, sha256 = evidence_identities[path]
        if item.get("bytes") != size or item.get("sha256") != sha256:
            raise EvidenceVerificationError(
                f"provenance evidence identity drifted: {path}"
            )
    return provenance


def verify_m1_evidence_bytes(
    json_bytes: bytes,
    text_bytes: bytes,
    provenance_bytes: bytes,
    manifest_bytes: bytes,
    receipt_bytes: bytes,
) -> dict[str, object]:
    """Verify all adopted evidence and provenance from already-read bytes."""

    for label, data, maximum in (
        (str(JSON_PATH), json_bytes, MAX_EVIDENCE_BYTES),
        (str(TEXT_PATH), text_bytes, MAX_EVIDENCE_BYTES),
        (str(PROVENANCE_PATH), provenance_bytes, MAX_EVIDENCE_BYTES),
        (str(MANIFEST_PATH), manifest_bytes, MAX_SOURCE_BYTES),
        (str(RECEIPT_PATH), receipt_bytes, MAX_SOURCE_BYTES),
    ):
        if len(data) > maximum:
            raise EvidenceVerificationError(
                f"{label}: exceeds {maximum}-byte safety limit"
            )

    _validate_surfaces(json_bytes, text_bytes)
    provenance = _validate_provenance(
        provenance_bytes,
        json_bytes,
        text_bytes,
        manifest_bytes,
        receipt_bytes,
    )

    expected_identities = (
        (
            str(JSON_PATH),
            json_bytes,
            EXPECTED_JSON_BYTES,
            EXPECTED_JSON_SHA256,
        ),
        (
            str(TEXT_PATH),
            text_bytes,
            EXPECTED_TEXT_BYTES,
            EXPECTED_TEXT_SHA256,
        ),
        (
            str(PROVENANCE_PATH),
            provenance_bytes,
            EXPECTED_PROVENANCE_BYTES,
            EXPECTED_PROVENANCE_SHA256,
        ),
    )
    for label, data, size, sha256 in expected_identities:
        if len(data) != size:
            raise EvidenceVerificationError(
                f"{label}: reviewed byte identity drifted"
            )
        if _sha256(data) != sha256:
            raise EvidenceVerificationError(
                f"{label}: reviewed SHA-256 identity drifted"
            )

    artifact = _obj(provenance["artifact"], "provenance.artifact")
    archive = _obj(artifact["archive"], "provenance.artifact.archive")
    return {
        "status": "verified",
        "schema_version": 1,
        "source_head": SOURCE_HEAD,
        "ci": {
            "workflow_path": str(WORKFLOW_PATH),
            "run_id": 31_266_366_484,
            "run_attempt": 1,
            "job_id": 93_125_047_136,
            "artifact_id": 9_024_264_228,
            "artifact_name": "m1-target-genome-31266366484",
            "conclusion": "success",
        },
        "archive": {
            "committed": False,
            "bytes": _int(
                archive["bytes"],
                "provenance.artifact.archive.bytes",
                expected=EXPECTED_ARCHIVE_BYTES,
            ),
            "sha256": _str(
                archive["sha256"],
                "provenance.artifact.archive.sha256",
                expected=EXPECTED_ARCHIVE_SHA256,
            ),
            "entry_count": len(
                _arr(
                    archive["entries"],
                    "provenance.artifact.archive.entries",
                )
            ),
        },
        "evidence_files": [
            {
                "path": str(JSON_PATH),
                "bytes": len(json_bytes),
                "sha256": _sha256(json_bytes),
            },
            {
                "path": str(TEXT_PATH),
                "bytes": len(text_bytes),
                "sha256": _sha256(text_bytes),
            },
            {
                "path": str(PROVENANCE_PATH),
                "bytes": len(provenance_bytes),
                "sha256": _sha256(provenance_bytes),
            },
        ],
        "limitations": list(EXPECTED_LIMITATIONS),
    }


def _safe_os_error(exc: OSError) -> str:
    return exc.strerror or type(exc).__name__


def read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= getattr(os, name, 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise EvidenceVerificationError(f"{path}: expected a regular file")
        if metadata.st_size > maximum_bytes:
            raise EvidenceVerificationError(
                f"{path}: exceeds {maximum_bytes}-byte safety limit"
            )
        data = bytearray()
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise EvidenceVerificationError(
                    f"{path}: changed while being read"
                )
            data.extend(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise EvidenceVerificationError(f"{path}: grew while being read")
        return bytes(data)
    except EvidenceVerificationError:
        raise
    except OSError as exc:
        raise EvidenceVerificationError(
            f"{path}: cannot safely read file: {_safe_os_error(exc)}"
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_m1_evidence(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    root = Path(root)
    if not root.is_absolute():
        root = Path.cwd() / root
    if root.is_symlink() or not root.is_dir():
        raise EvidenceVerificationError(
            "repository root must be an existing non-symlink directory"
        )

    evidence_dir = root / EVIDENCE_DIR
    if evidence_dir.is_symlink() or not evidence_dir.is_dir():
        raise EvidenceVerificationError(
            "evidence/raw must be an existing non-symlink directory"
        )
    try:
        children = list(evidence_dir.iterdir())
        if any(child.suffix.lower() == ".zip" for child in children):
            raise EvidenceVerificationError(
                "artifact ZIP must not be committed under evidence/raw"
            )
        m1_entries = {
            child.name: child
            for child in children
            if child.name.startswith("m1_target_genome")
        }
    except EvidenceVerificationError:
        raise
    except OSError as exc:
        raise EvidenceVerificationError(
            f"cannot inspect evidence/raw: {_safe_os_error(exc)}"
        ) from None
    if set(m1_entries) != EXPECTED_M1_NAMES:
        raise EvidenceVerificationError(
            "M1 evidence inventory does not match; "
            f"observed={sorted(m1_entries)}"
        )
    for name, path in m1_entries.items():
        if path.is_symlink():
            raise EvidenceVerificationError(
                f"evidence entry must not be a symlink: {name}"
            )
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise EvidenceVerificationError(
                f"cannot inspect evidence entry {name}: {_safe_os_error(exc)}"
            ) from None
        if not stat.S_ISREG(mode):
            raise EvidenceVerificationError(
                f"evidence entry must be a regular file: {name}"
            )

    return verify_m1_evidence_bytes(
        read_regular_file(root / JSON_PATH, maximum_bytes=MAX_EVIDENCE_BYTES),
        read_regular_file(root / TEXT_PATH, maximum_bytes=MAX_EVIDENCE_BYTES),
        read_regular_file(
            root / PROVENANCE_PATH,
            maximum_bytes=MAX_EVIDENCE_BYTES,
        ),
        read_regular_file(root / MANIFEST_PATH, maximum_bytes=MAX_SOURCE_BYTES),
        read_regular_file(root / RECEIPT_PATH, maximum_bytes=MAX_SOURCE_BYTES),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the adopted M1 evidence and provenance offline."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="repository root",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit compact deterministic JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_m1_evidence(args.root)
    except EvidenceVerificationError as exc:
        print(
            json.dumps(
                {"error": str(exc), "status": "rejected"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2

    if args.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print("M1 evidence: verified")
        print(f"  source head: {report['source_head']}")
        print(
            "  artifact: "
            f"{report['ci']['artifact_name']} ({report['ci']['artifact_id']})"
        )
        print(
            "  archive: "
            f"{report['archive']['bytes']} bytes, "
            f"sha256 {report['archive']['sha256']} (not committed)"
        )
        print("  evidence files: 3")
        print("  network access: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
