"""Offline verification for the captured repository API projection.

The verifier consumes only committed JSON bytes.  It contains no transport code,
does not import target-side code, and never opens a model weight or LFS object.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Final, Iterable


TARGET_REPOSITORY: Final = "deepseek-ai/DeepSeek-V4-Flash-0731"
TARGET_REVISION: Final = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
SOURCE_URL: Final = (
    "https://huggingface.co/api/models/deepseek-ai/"
    f"DeepSeek-V4-Flash-0731/revision/{TARGET_REVISION}?blobs=true"
)
RECEIPT_NAME: Final = "repository.receipt.json"
REPOSITORY_NAME: Final = "repository.json"
MAX_RECEIPT_BYTES: Final = 64 * 1024
MAX_REPOSITORY_BYTES: Final = 64 * 1024
_READ_CHUNK_BYTES: Final = 16 * 1024

EXPECTED_RECEIPT_BYTES: Final = 24_578
EXPECTED_RECEIPT_SHA256: Final = (
    "fb7d2406fccd6f326cef18df829457eb98ab2c61dd77a9a0436fd4a962b4bbbf"
)
EXPECTED_INPUT_BYTES: Final = 14_491
EXPECTED_INPUT_SHA256: Final = (
    "16dd1cea2018d8af2a84922bc6fff22a7988dbb4ecc58d9c92486fa01b178291"
)
EXPECTED_REPOSITORY_BYTES: Final = 22_284
EXPECTED_REPOSITORY_SHA256: Final = (
    "6cacae22067d225351b46d30b3b4335db18b8941e342ac24ab945d81ebef4800"
)
EXPECTED_RESPONSE_BODY_BYTES: Final = 16_362
EXPECTED_RESPONSE_BODY_SHA256: Final = (
    "cc906889b269c1a97632ed62bfd286b304b9602219064fe37feda529e3cd119c"
)
EXPECTED_VERIFICATION_TIMESTAMP: Final = "2026-08-08T15:49:13Z"
EXPECTED_RESPONSE_DATE: Final = "Sat, 08 Aug 2026 15:49:13 GMT"
EXPECTED_RESPONSE_CONTENT_TYPE: Final = "application/json; charset=utf-8"
EXPECTED_LAST_MODIFIED: Final = "2026-08-01T03:07:41.000Z"
EXPECTED_ORIGINAL_RETRIEVAL_DATE: Final = "2026-08-06"
EXPECTED_SELECTION: Final = (
    "revision identity, stable file metadata, LFS object digests, and "
    "safetensors parameter classes; volatile popularity fields excluded"
)
EXPECTED_INPUT_CANONICALIZATION: Final = (
    "UTF-8 JSON, ensure_ascii=false, recursively sorted object keys, "
    "separators comma/colon, no trailing newline"
)
EXPECTED_REPOSITORY_CANONICALIZATION: Final = (
    "UTF-8 JSON, ensure_ascii=false, recursively sorted object keys, "
    "indent=2, trailing newline"
)
EXPECTED_INCLUDED_FIELDS: Final = (
    "id",
    "lastModified",
    "safetensors",
    "sha",
    "siblings",
    "usedStorage",
)
EXPECTED_EXCLUDED_FIELDS: Final = (
    "_id",
    "author",
    "cardData",
    "config",
    "createdAt",
    "disabled",
    "downloads",
    "gated",
    "library_name",
    "likes",
    "model-index",
    "modelId",
    "pipeline_tag",
    "private",
    "spaces",
    "tags",
    "transformersInfo",
    "widgetData",
)
EXPECTED_ARTIFACT_BYTES: Final = {
    "all_repository_files": 166_898_661_074,
    "non_weight_files": 12_125_738,
    "weight_shards": 166_886_535_336,
}
EXPECTED_PARAMETER_CLASSES: Final = {
    "BF16": 1_483_567_488,
    "F32": 37_741_630,
    "F8_E4M3": 6_304_038_912,
    "I64": 2_327_040,
    "I8": 296_352_743_424,
}
EXPECTED_PARAMETER_TOTAL: Final = 304_180_418_494
EXPECTED_LIMITATIONS: Final = (
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

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SHARD = re.compile(r"^model-([0-9]{5})-of-00048\.safetensors$")


class ProjectionVerificationError(ValueError):
    """The receipt or its deterministic projection violates the trust contract."""


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProjectionVerificationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _no_constant(value: str) -> object:
    raise ProjectionVerificationError(f"non-finite JSON number is forbidden: {value}")


def loads_json_strict(data: bytes, *, label: str) -> object:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProjectionVerificationError(f"{label}: expected UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicates,
            parse_constant=_no_constant,
        )
    except ProjectionVerificationError:
        raise
    except json.JSONDecodeError as exc:
        raise ProjectionVerificationError(
            f"{label}: invalid JSON: {exc.msg}"
        ) from exc


def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return text.encode("utf-8")


def _obj(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ProjectionVerificationError(f"{label}: expected an object")
    return value


def _arr(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise ProjectionVerificationError(f"{label}: expected an array")
    return value


def _str(value: object, label: str, *, expected: str | None = None) -> str:
    if type(value) is not str or not value:
        raise ProjectionVerificationError(f"{label}: expected a non-empty string")
    if expected is not None and value != expected:
        raise ProjectionVerificationError(
            f"{label}: expected {expected!r}, observed {value!r}"
        )
    return value


def _int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    expected: int | None = None,
) -> int:
    if type(value) is not int:
        raise ProjectionVerificationError(
            f"{label}: expected an integer; booleans are not integers"
        )
    if value < minimum:
        raise ProjectionVerificationError(f"{label}: expected a value >= {minimum}")
    if expected is not None and value != expected:
        raise ProjectionVerificationError(
            f"{label}: expected {expected}, observed {value}"
        )
    return value


def _bool(value: object, label: str, *, expected: bool) -> bool:
    if type(value) is not bool or value is not expected:
        raise ProjectionVerificationError(
            f"{label}: expected {str(expected).lower()}"
        )
    return value


def _keys(value: dict[str, object], expected: Iterable[str], label: str) -> None:
    wanted = set(expected)
    actual = set(value)
    if actual != wanted:
        raise ProjectionVerificationError(
            f"{label}: schema mismatch; "
            f"missing={sorted(wanted - actual)}, unexpected={sorted(actual - wanted)}"
        )


def _digest(
    value: object,
    label: str,
    pattern: re.Pattern[str],
    *,
    expected: str | None = None,
) -> str:
    text = _str(value, label)
    if pattern.fullmatch(text) is None:
        raise ProjectionVerificationError(f"{label}: malformed digest")
    if expected is not None and text != expected:
        raise ProjectionVerificationError(
            f"{label}: expected {expected}, observed {text}"
        )
    return text


def _path(value: object, label: str) -> str:
    text = _str(value, label)
    pure = PurePosixPath(text)
    if (
        "\x00" in text
        or "\\" in text
        or text in {".", ".."}
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != text
    ):
        raise ProjectionVerificationError(f"{label}: unsafe path")
    return text


def _expect_string_list(
    value: object,
    expected: tuple[str, ...],
    label: str,
) -> None:
    raw = _arr(value, label)
    if any(type(item) is not str for item in raw) or raw != list(expected):
        raise ProjectionVerificationError(f"{label}: does not match the capture policy")


def _validate_envelope(receipt: dict[str, object]) -> None:
    _keys(
        receipt,
        {
            "capture_kind",
            "limitations",
            "projection_input",
            "projection_input_canonicalization",
            "repository_projection",
            "request",
            "response",
            "schema_version",
            "selection_policy",
            "target",
            "verification_timestamp_utc",
        },
        RECEIPT_NAME,
    )
    _int(receipt["schema_version"], "receipt.schema_version", expected=1)
    _str(
        receipt["capture_kind"],
        "receipt.capture_kind",
        expected="post-capture-api-projection-verification",
    )
    _str(
        receipt["verification_timestamp_utc"],
        "receipt.verification_timestamp_utc",
        expected=EXPECTED_VERIFICATION_TIMESTAMP,
    )

    target = _obj(receipt["target"], "receipt.target")
    _keys(target, {"repository", "revision"}, "receipt.target")
    _str(
        target["repository"],
        "receipt.target.repository",
        expected=TARGET_REPOSITORY,
    )
    _str(target["revision"], "receipt.target.revision", expected=TARGET_REVISION)

    request = _obj(receipt["request"], "receipt.request")
    _keys(
        request,
        {"method", "recorded_headers", "redirects_followed", "url"},
        "receipt.request",
    )
    _str(request["method"], "receipt.request.method", expected="GET")
    _str(request["url"], "receipt.request.url", expected=SOURCE_URL)
    _int(
        request["redirects_followed"],
        "receipt.request.redirects_followed",
        expected=0,
    )
    headers = _obj(request["recorded_headers"], "receipt.request.recorded_headers")
    _keys(headers, {"accept", "user_agent"}, "receipt.request.recorded_headers")
    _str(
        headers["accept"],
        "receipt.request.recorded_headers.accept",
        expected="application/json",
    )
    _str(
        headers["user_agent"],
        "receipt.request.recorded_headers.user_agent",
        expected="stratafold-captured-projection-verifier/1",
    )

    response = _obj(receipt["response"], "receipt.response")
    _keys(
        response,
        {
            "body_bytes",
            "body_committed",
            "body_sha256",
            "content_length_header",
            "content_type",
            "date",
            "status",
        },
        "receipt.response",
    )
    _int(response["status"], "receipt.response.status", expected=200)
    _str(
        response["date"],
        "receipt.response.date",
        expected=EXPECTED_RESPONSE_DATE,
    )
    _str(
        response["content_type"],
        "receipt.response.content_type",
        expected=EXPECTED_RESPONSE_CONTENT_TYPE,
    )
    _str(
        response["content_length_header"],
        "receipt.response.content_length_header",
        expected=str(EXPECTED_RESPONSE_BODY_BYTES),
    )
    _int(
        response["body_bytes"],
        "receipt.response.body_bytes",
        expected=EXPECTED_RESPONSE_BODY_BYTES,
    )
    _digest(
        response["body_sha256"],
        "receipt.response.body_sha256",
        _HEX64,
        expected=EXPECTED_RESPONSE_BODY_SHA256,
    )
    _bool(
        response["body_committed"],
        "receipt.response.body_committed",
        expected=False,
    )

    limitations = _arr(receipt["limitations"], "receipt.limitations")
    if limitations != list(EXPECTED_LIMITATIONS):
        raise ProjectionVerificationError("receipt.limitations do not match")

    policy = _obj(receipt["selection_policy"], "receipt.selection_policy")
    _keys(
        policy,
        {
            "excluded_top_level_fields",
            "included_top_level_fields",
            "nested_fields",
            "nested_lfs_fields_excluded",
            "nested_sibling_fields_excluded",
            "rationale",
            "sibling_sequence_policy",
        },
        "receipt.selection_policy",
    )
    _expect_string_list(
        policy["included_top_level_fields"],
        EXPECTED_INCLUDED_FIELDS,
        "receipt.selection_policy.included_top_level_fields",
    )
    _expect_string_list(
        policy["excluded_top_level_fields"],
        EXPECTED_EXCLUDED_FIELDS,
        "receipt.selection_policy.excluded_top_level_fields",
    )
    if _arr(
        policy["nested_sibling_fields_excluded"],
        "receipt.selection_policy.nested_sibling_fields_excluded",
    ):
        raise ProjectionVerificationError(
            "receipt policy unexpectedly excludes sibling fields"
        )
    if _arr(
        policy["nested_lfs_fields_excluded"],
        "receipt.selection_policy.nested_lfs_fields_excluded",
    ):
        raise ProjectionVerificationError(
            "receipt policy unexpectedly excludes LFS fields"
        )
    nested = _obj(policy["nested_fields"], "receipt.selection_policy.nested_fields")
    _keys(nested, {"lfs", "safetensors", "sibling"}, "receipt.selection_policy.nested_fields")
    _expect_string_list(
        nested["lfs"],
        ("pointerSize", "sha256", "size"),
        "receipt.selection_policy.nested_fields.lfs",
    )
    _expect_string_list(
        nested["safetensors"],
        ("parameters", "total"),
        "receipt.selection_policy.nested_fields.safetensors",
    )
    _expect_string_list(
        nested["sibling"],
        ("blobId", "lfs", "rfilename", "size"),
        "receipt.selection_policy.nested_fields.sibling",
    )
    _str(policy["rationale"], "receipt.selection_policy.rationale")
    _str(
        policy["sibling_sequence_policy"],
        "receipt.selection_policy.sibling_sequence_policy",
        expected=(
            "Captured sequence order is retained for body-derived identity but is "
            "non-semantic and untrusted; reconstruction sorts unique safe rfilename "
            "values lexicographically."
        ),
    )

    canonicalization = _obj(
        receipt["projection_input_canonicalization"],
        "receipt.projection_input_canonicalization",
    )
    _keys(
        canonicalization,
        {"algorithm", "bytes", "sha256"},
        "receipt.projection_input_canonicalization",
    )
    _str(
        canonicalization["algorithm"],
        "receipt.projection_input_canonicalization.algorithm",
        expected=EXPECTED_INPUT_CANONICALIZATION,
    )
    _int(
        canonicalization["bytes"],
        "receipt.projection_input_canonicalization.bytes",
        expected=EXPECTED_INPUT_BYTES,
    )
    _digest(
        canonicalization["sha256"],
        "receipt.projection_input_canonicalization.sha256",
        _HEX64,
        expected=EXPECTED_INPUT_SHA256,
    )

    projection = _obj(receipt["repository_projection"], "receipt.repository_projection")
    _keys(
        projection,
        {
            "canonicalization",
            "expected_bytes",
            "expected_sha256",
            "original_retrieval_date_utc",
            "path",
            "selection",
            "sibling_ordering",
            "source_url",
        },
        "receipt.repository_projection",
    )
    _str(
        projection["canonicalization"],
        "receipt.repository_projection.canonicalization",
        expected=EXPECTED_REPOSITORY_CANONICALIZATION,
    )
    _int(
        projection["expected_bytes"],
        "receipt.repository_projection.expected_bytes",
        expected=EXPECTED_REPOSITORY_BYTES,
    )
    _digest(
        projection["expected_sha256"],
        "receipt.repository_projection.expected_sha256",
        _HEX64,
        expected=EXPECTED_REPOSITORY_SHA256,
    )
    _str(
        projection["original_retrieval_date_utc"],
        "receipt.repository_projection.original_retrieval_date_utc",
        expected=EXPECTED_ORIGINAL_RETRIEVAL_DATE,
    )
    _str(
        projection["path"],
        "receipt.repository_projection.path",
        expected=REPOSITORY_NAME,
    )
    _str(
        projection["selection"],
        "receipt.repository_projection.selection",
        expected=EXPECTED_SELECTION,
    )
    _str(
        projection["sibling_ordering"],
        "receipt.repository_projection.sibling_ordering",
        expected="lexicographic by rfilename",
    )
    _str(
        projection["source_url"],
        "receipt.repository_projection.source_url",
        expected=SOURCE_URL,
    )


def _validate_projection_input(value: object) -> dict[str, object]:
    projection_input = _obj(value, "receipt.projection_input")
    _keys(projection_input, EXPECTED_INCLUDED_FIELDS, "receipt.projection_input")
    _str(
        projection_input["id"],
        "receipt.projection_input.id",
        expected=TARGET_REPOSITORY,
    )
    _str(
        projection_input["sha"],
        "receipt.projection_input.sha",
        expected=TARGET_REVISION,
    )
    _str(
        projection_input["lastModified"],
        "receipt.projection_input.lastModified",
        expected=EXPECTED_LAST_MODIFIED,
    )

    safetensors = _obj(
        projection_input["safetensors"],
        "receipt.projection_input.safetensors",
    )
    _keys(
        safetensors,
        {"parameters", "total"},
        "receipt.projection_input.safetensors",
    )
    parameters = _obj(
        safetensors["parameters"],
        "receipt.projection_input.safetensors.parameters",
    )
    _keys(
        parameters,
        EXPECTED_PARAMETER_CLASSES,
        "receipt.projection_input.safetensors.parameters",
    )
    observed_parameters = {
        key: _int(
            parameters[key],
            f"receipt.projection_input.safetensors.parameters.{key}",
        )
        for key in EXPECTED_PARAMETER_CLASSES
    }
    if observed_parameters != EXPECTED_PARAMETER_CLASSES:
        raise ProjectionVerificationError(
            "receipt projection input parameter classes drifted"
        )
    total = _int(
        safetensors["total"],
        "receipt.projection_input.safetensors.total",
    )
    if sum(observed_parameters.values()) != total:
        raise ProjectionVerificationError(
            "receipt projection input parameter ledger does not add up"
        )
    if total != EXPECTED_PARAMETER_TOTAL:
        raise ProjectionVerificationError(
            "receipt projection input parameter total drifted"
        )

    siblings = _arr(
        projection_input["siblings"],
        "receipt.projection_input.siblings",
    )
    if len(siblings) != 74:
        raise ProjectionVerificationError(
            f"receipt projection input expected 74 siblings, observed {len(siblings)}"
        )

    names: set[str] = set()
    shard_ordinals: set[int] = set()
    git_count = 0
    lfs_count = 0
    all_bytes = 0
    lfs_bytes = 0
    for index, raw in enumerate(siblings):
        label = f"receipt.projection_input.siblings[{index}]"
        sibling = _obj(raw, label)
        has_lfs = "lfs" in sibling
        _keys(
            sibling,
            {"blobId", "rfilename", "size", "lfs"}
            if has_lfs
            else {"blobId", "rfilename", "size"},
            label,
        )
        name = _path(sibling["rfilename"], f"{label}.rfilename")
        if name in names:
            raise ProjectionVerificationError(
                f"receipt projection input contains duplicate sibling: {name}"
            )
        names.add(name)
        _digest(sibling["blobId"], f"{label}.blobId", _HEX40)
        size = _int(sibling["size"], f"{label}.size")
        all_bytes += size

        shard_match = _SHARD.fullmatch(name)
        if has_lfs:
            lfs_count += 1
            if shard_match is None:
                raise ProjectionVerificationError(
                    f"{label}: non-shard LFS object is outside the projection contract"
                )
            lfs = _obj(sibling["lfs"], f"{label}.lfs")
            _keys(lfs, {"pointerSize", "sha256", "size"}, f"{label}.lfs")
            if _int(lfs["size"], f"{label}.lfs.size") != size:
                raise ProjectionVerificationError(f"{label}: LFS size mismatch")
            _int(
                lfs["pointerSize"],
                f"{label}.lfs.pointerSize",
                expected=135,
            )
            _digest(lfs["sha256"], f"{label}.lfs.sha256", _HEX64)
            shard_ordinals.add(int(shard_match.group(1)))
            lfs_bytes += size
        else:
            git_count += 1
            if shard_match is not None:
                raise ProjectionVerificationError(
                    f"{label}: weight shard is not represented as LFS"
                )

    if git_count != 26 or lfs_count != 48:
        raise ProjectionVerificationError(
            "receipt projection input storage census drifted"
        )
    if shard_ordinals != set(range(1, 49)):
        raise ProjectionVerificationError(
            "receipt projection input shard ordinal census drifted"
        )
    used_storage = _int(
        projection_input["usedStorage"],
        "receipt.projection_input.usedStorage",
    )
    if used_storage != lfs_bytes:
        raise ProjectionVerificationError(
            "receipt usedStorage does not equal the LFS byte total"
        )
    observed_artifacts = {
        "all_repository_files": all_bytes,
        "non_weight_files": all_bytes - lfs_bytes,
        "weight_shards": lfs_bytes,
    }
    if observed_artifacts != EXPECTED_ARTIFACT_BYTES:
        raise ProjectionVerificationError(
            "receipt projection input artifact byte ledger drifted"
        )
    return projection_input


def reconstruct_repository(projection_input: object) -> dict[str, object]:
    """Validate an input and return its deterministic repository projection."""

    validated = _validate_projection_input(projection_input)
    siblings = sorted(
        _arr(validated["siblings"], "receipt.projection_input.siblings"),
        key=lambda item: _str(
            _obj(item, "receipt.projection_input.sibling")["rfilename"],
            "receipt.projection_input.sibling.rfilename",
        ),
    )
    files: list[dict[str, object]] = []
    all_bytes = 0
    weight_bytes = 0
    for raw in siblings:
        sibling = _obj(raw, "receipt.projection_input.sibling")
        size = _int(sibling["size"], "receipt.projection_input.sibling.size")
        storage: dict[str, object]
        if "lfs" in sibling:
            lfs = _obj(sibling["lfs"], "receipt.projection_input.sibling.lfs")
            storage = {
                "bytes": size,
                "kind": "lfs",
                "pointer_bytes": _int(
                    lfs["pointerSize"],
                    "receipt.projection_input.sibling.lfs.pointerSize",
                ),
                "sha256": _str(
                    lfs["sha256"],
                    "receipt.projection_input.sibling.lfs.sha256",
                ),
            }
            weight_bytes += size
        else:
            storage = {"kind": "git"}
        files.append(
            {
                "blob_id": _str(
                    sibling["blobId"],
                    "receipt.projection_input.sibling.blobId",
                ),
                "bytes": size,
                "path": _str(
                    sibling["rfilename"],
                    "receipt.projection_input.sibling.rfilename",
                ),
                "storage": storage,
            }
        )
        all_bytes += size

    safetensors = _obj(
        validated["safetensors"],
        "receipt.projection_input.safetensors",
    )
    return {
        "artifact_bytes": {
            "all_repository_files": all_bytes,
            "non_weight_files": all_bytes - weight_bytes,
            "weight_shards": weight_bytes,
        },
        "files": files,
        "last_modified": _str(
            validated["lastModified"],
            "receipt.projection_input.lastModified",
        ),
        "repository": _str(validated["id"], "receipt.projection_input.id"),
        "revision": _str(validated["sha"], "receipt.projection_input.sha"),
        "safetensors_parameter_classes": {
            "parameters": _obj(
                safetensors["parameters"],
                "receipt.projection_input.safetensors.parameters",
            ),
            "total": _int(
                safetensors["total"],
                "receipt.projection_input.safetensors.total",
            ),
        },
        "schema_version": 1,
        "source": {
            "retrieved_at_utc": EXPECTED_ORIGINAL_RETRIEVAL_DATE,
            "selection": EXPECTED_SELECTION,
            "url": SOURCE_URL,
        },
    }


def verify_projection_bytes(
    receipt_bytes: bytes,
    repository_bytes: bytes,
) -> dict[str, object]:
    """Verify the receipt and repository projection without any network access."""

    if len(receipt_bytes) > MAX_RECEIPT_BYTES:
        raise ProjectionVerificationError(
            f"{RECEIPT_NAME}: exceeds {MAX_RECEIPT_BYTES}-byte safety limit"
        )
    if len(repository_bytes) > MAX_REPOSITORY_BYTES:
        raise ProjectionVerificationError(
            f"{REPOSITORY_NAME}: exceeds {MAX_REPOSITORY_BYTES}-byte safety limit"
        )

    receipt = _obj(
        loads_json_strict(receipt_bytes, label=RECEIPT_NAME),
        RECEIPT_NAME,
    )
    repository = loads_json_strict(repository_bytes, label=REPOSITORY_NAME)

    # Semantic validation precedes exact identity gates so failures remain specific.
    _validate_envelope(receipt)
    projection_input = _validate_projection_input(receipt["projection_input"])
    input_bytes = canonical_json_bytes(projection_input)
    input_sha = hashlib.sha256(input_bytes).hexdigest()
    if len(input_bytes) != EXPECTED_INPUT_BYTES:
        raise ProjectionVerificationError(
            "projection input canonical byte length drifted"
        )
    if input_sha != EXPECTED_INPUT_SHA256:
        raise ProjectionVerificationError(
            "projection input canonical SHA-256 drifted"
        )

    reconstructed = reconstruct_repository(projection_input)
    projected_bytes = canonical_json_bytes(reconstructed, pretty=True)
    projected_sha = hashlib.sha256(projected_bytes).hexdigest()
    if len(projected_bytes) != EXPECTED_REPOSITORY_BYTES:
        raise ProjectionVerificationError(
            "reconstructed repository byte length drifted"
        )
    if projected_sha != EXPECTED_REPOSITORY_SHA256:
        raise ProjectionVerificationError(
            "reconstructed repository SHA-256 drifted"
        )
    if repository != reconstructed:
        raise ProjectionVerificationError(
            "repository.json semantics do not match the reconstructed projection"
        )
    if repository_bytes != projected_bytes:
        raise ProjectionVerificationError(
            "repository.json bytes are not the canonical reconstructed projection"
        )

    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if len(receipt_bytes) != EXPECTED_RECEIPT_BYTES:
        raise ProjectionVerificationError(
            "receipt reviewed byte identity drifted"
        )
    if receipt_sha != EXPECTED_RECEIPT_SHA256:
        raise ProjectionVerificationError(
            "receipt reviewed SHA-256 identity drifted"
        )

    return {
        "status": "verified",
        "capture_kind": "post-capture-api-projection-verification",
        "target": {
            "repository": TARGET_REPOSITORY,
            "revision": TARGET_REVISION,
        },
        "verification_timestamp_utc": EXPECTED_VERIFICATION_TIMESTAMP,
        "response": {
            "status": 200,
            "date": EXPECTED_RESPONSE_DATE,
            "content_type": EXPECTED_RESPONSE_CONTENT_TYPE,
            "body_bytes": EXPECTED_RESPONSE_BODY_BYTES,
            "body_sha256": EXPECTED_RESPONSE_BODY_SHA256,
            "body_committed": False,
        },
        "projection_input": {
            "bytes": len(input_bytes),
            "sha256": input_sha,
            "sibling_count": 74,
            "sequence_order": "captured-but-non-semantic-and-untrusted",
        },
        "repository_projection": {
            "path": REPOSITORY_NAME,
            "bytes": len(projected_bytes),
            "sha256": projected_sha,
            "sibling_order": "lexicographic-by-rfilename",
        },
        "limitations": list(EXPECTED_LIMITATIONS),
        "safety": {
            "metadata_only": True,
            "response_body_committed": False,
            "weight_or_lfs_payload_bytes_read": 0,
            "target_code_imported_or_executed": False,
        },
    }


def _safe_os_error(exc: OSError) -> str:
    return exc.strerror or type(exc).__name__


def read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    """Read a bounded regular file by descriptor without following a final symlink."""

    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= getattr(os, name, 0)
    descriptor = -1
    try:
        descriptor = os.open(Path(path), flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ProjectionVerificationError(f"{path}: expected a regular file")
        if metadata.st_size > maximum_bytes:
            raise ProjectionVerificationError(
                f"{path}: exceeds {maximum_bytes}-byte safety limit"
            )
        data = bytearray()
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise ProjectionVerificationError(f"{path}: changed while being read")
            data.extend(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ProjectionVerificationError(f"{path}: grew while being read")
        return bytes(data)
    except ProjectionVerificationError:
        raise
    except OSError as exc:
        raise ProjectionVerificationError(
            f"{path}: cannot safely read file: {_safe_os_error(exc)}"
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_projection_files(
    receipt_path: Path,
    repository_path: Path,
) -> dict[str, object]:
    receipt_bytes = read_regular_file(
        receipt_path,
        maximum_bytes=MAX_RECEIPT_BYTES,
    )
    repository_bytes = read_regular_file(
        repository_path,
        maximum_bytes=MAX_REPOSITORY_BYTES,
    )
    return verify_projection_bytes(receipt_bytes, repository_bytes)
