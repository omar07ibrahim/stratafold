"""Dependency-free offline decoder for the pinned target metadata snapshot.

Only committed JSON and license bytes are read. This module has no network
transport, never imports target-side Python, and never opens a weight shard.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Final, Iterable


TARGET_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash-0731"
TARGET_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DEFAULT_SNAPSHOT = (
    Path(__file__).resolve().parents[2]
    / "metadata"
    / "deepseek-v4-flash-0731"
    / TARGET_REVISION
)
EXPECTED_FILES = frozenset(
    {
        "config.json",
        "LICENSE.target.txt",
        "model.safetensors.index.json",
        "repository.json",
    }
)
EXPECTED_URLS = {
    "config.json": (
        "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/"
        f"{TARGET_REVISION}/config.json"
    ),
    "LICENSE.target.txt": (
        "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/"
        f"{TARGET_REVISION}/LICENSE"
    ),
    "model.safetensors.index.json": (
        "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/"
        f"{TARGET_REVISION}/model.safetensors.index.json"
    ),
    "repository.json": (
        "https://huggingface.co/api/models/deepseek-ai/"
        f"DeepSeek-V4-Flash-0731/revision/{TARGET_REVISION}?blobs=true"
    ),
}
MAX_FILE_BYTES = 32 * 1024**2
MAX_SNAPSHOT_BYTES = 128 * 1024**2

REVIEWED_FILE_IDENTITIES: Final[
    dict[str, tuple[int, str, str | None]]
] = {
    "config.json": (
        1888,
        "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
        "5f2da9100036b39e26ebe5ab493c5e8d4004d8a1",
    ),
    "model.safetensors.index.json": (
        5_602_871,
        "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
        "c3b10d45a829545fbf0d9d2880a1aa0b9ab3b43a",
    ),
    "LICENSE.target.txt": (
        1084,
        "f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040",
        "d62e3bef9f054f21b7fc616365850fbf879a99ff",
    ),
    "repository.json": (
        22_284,
        "6cacae22067d225351b46d30b3b4335db18b8941e342ac24ab945d81ebef4800",
        None,
    ),
}
REPOSITORY_PATHS: Final[dict[str, str]] = {
    "config.json": "config.json",
    "model.safetensors.index.json": "model.safetensors.index.json",
    "LICENSE.target.txt": "LICENSE",
}
EXPECTED_REPOSITORY_FILE_COUNT: Final = 74
EXPECTED_TENSOR_COUNT: Final = 72_317
EXPECTED_TENSOR_PAYLOAD_BYTES: Final = 166_878_536_440
EXPECTED_ARTIFACT_BYTES: Final = {
    "weight_shards": 166_886_535_336,
    "non_weight_files": 12_125_738,
    "all_repository_files": 166_898_661_074,
}
EXPECTED_PARAMETER_COUNTS: Final = {
    "BF16": 1_483_567_488,
    "I64": 2_327_040,
    "F32": 37_741_630,
    "F8_E4M3": 6_304_038_912,
    "I8": 296_352_743_424,
    "total": 304_180_418_494,
}

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SHARD = re.compile(r"^model-([0-9]{5})-of-([0-9]{5})\.safetensors$")
_LAYER = re.compile(r"^layers\.([0-9]+)\.")
_MTP = re.compile(r"^mtp\.([0-9]+)\.")
_ROUTED = re.compile(
    r"^layers\.([0-9]+)\.ffn\.experts\.([0-9]+)\.(w[123])\.(weight|scale)$"
)
_SHARED = re.compile(
    r"^layers\.([0-9]+)\.ffn\.shared_experts\.(w[123])\.(weight|scale)$"
)
_MTP_ROUTED = re.compile(
    r"^mtp\.([0-9]+)\.ffn\.experts\.([0-9]+)\.(w[123])\.(weight|scale)$"
)
_MTP_SHARED = re.compile(
    r"^mtp\.([0-9]+)\.ffn\.shared_experts\.(w[123])\.(weight|scale)$"
)
_COMPONENTS = frozenset(
    f"w{matrix}.{kind}"
    for matrix in (1, 2, 3)
    for kind in ("weight", "scale")
)


class SnapshotValidationError(ValueError):
    """The snapshot violates its offline trust contract."""


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _no_constant(value: str) -> object:
    raise SnapshotValidationError(f"non-finite JSON number is forbidden: {value}")


def loads_json_strict(data: bytes, *, label: str) -> object:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SnapshotValidationError(f"{label}: expected UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicates,
            parse_constant=_no_constant,
        )
    except SnapshotValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise SnapshotValidationError(f"{label}: invalid JSON: {exc.msg}") from exc


def _obj(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise SnapshotValidationError(f"{label}: expected an object")
    return value


def _arr(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise SnapshotValidationError(f"{label}: expected an array")
    return value


def _str(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise SnapshotValidationError(f"{label}: expected a non-empty string")
    return value


def _int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    expected: int | None = None,
) -> int:
    if type(value) is not int:
        raise SnapshotValidationError(
            f"{label}: expected an integer; booleans are not integers"
        )
    if value < minimum:
        raise SnapshotValidationError(f"{label}: expected a value >= {minimum}")
    if expected is not None and value != expected:
        raise SnapshotValidationError(
            f"{label}: expected {expected}, observed {value}"
        )
    return value


def _bool(value: object, label: str, expected: bool) -> bool:
    if type(value) is not bool:
        raise SnapshotValidationError(f"{label}: expected a boolean")
    if value is not expected:
        raise SnapshotValidationError(f"{label}: expected {str(expected).lower()}")
    return value


def _keys(value: dict[str, object], expected: Iterable[str], label: str) -> None:
    wanted = set(expected)
    actual = set(value)
    if actual != wanted:
        raise SnapshotValidationError(
            f"{label}: schema mismatch; "
            f"missing={sorted(wanted - actual)}, unexpected={sorted(actual - wanted)}"
        )


def _digest(value: object, label: str, pattern: re.Pattern[str]) -> str:
    text = _str(value, label)
    if pattern.fullmatch(text) is None:
        raise SnapshotValidationError(f"{label}: malformed digest")
    return text


def _path(value: object, label: str, *, basename: bool) -> str:
    text = _str(value, label)
    pure = PurePosixPath(text)
    if (
        "\x00" in text
        or "\\" in text
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != text
        or (basename and len(pure.parts) != 1)
    ):
        raise SnapshotValidationError(f"{label}: unsafe path")
    return text


def _read_file(root: Path, relative: str) -> bytes:
    candidate = root / relative
    try:
        if candidate.is_symlink():
            raise SnapshotValidationError(
                f"snapshot file must not be a symlink: {relative}"
            )
        if not candidate.is_file():
            raise SnapshotValidationError(
                f"snapshot file is missing or not regular: {relative}"
            )
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
        return candidate.read_bytes()
    except SnapshotValidationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise SnapshotValidationError(
            f"cannot safely read snapshot file {relative}: {exc}"
        ) from exc


def _validate_manifest(
    payload: object,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    manifest = _obj(payload, "manifest.json")
    _keys(
        manifest,
        {"files", "retrieval_window_utc", "safety", "schema_version", "target"},
        "manifest.json",
    )
    _int(manifest["schema_version"], "manifest.schema_version", expected=1)

    target = _obj(manifest["target"], "manifest.target")
    _keys(target, {"evidence_level", "repository", "revision"}, "manifest.target")
    if _str(target["evidence_level"], "manifest.target.evidence_level") != (
        "pinned-official-metadata"
    ):
        raise SnapshotValidationError("manifest evidence level is unsupported")
    if _str(target["repository"], "manifest.target.repository") != TARGET_REPOSITORY:
        raise SnapshotValidationError("manifest target repository does not match")
    if _str(target["revision"], "manifest.target.revision") != TARGET_REVISION:
        raise SnapshotValidationError("manifest target revision does not match")

    window = _obj(manifest["retrieval_window_utc"], "manifest.retrieval_window_utc")
    _keys(window, {"start", "end"}, "manifest.retrieval_window_utc")
    stamps: list[datetime] = []
    for name in ("start", "end"):
        value = _str(window[name], f"manifest.retrieval_window_utc.{name}")
        if not value.endswith("Z"):
            raise SnapshotValidationError("manifest timestamp must use UTC Z form")
        try:
            stamps.append(datetime.fromisoformat(value[:-1] + "+00:00"))
        except ValueError as exc:
            raise SnapshotValidationError("manifest contains an invalid timestamp") from exc
    if stamps[1] < stamps[0]:
        raise SnapshotValidationError("manifest retrieval window is reversed")

    safety = _obj(manifest["safety"], "manifest.safety")
    _keys(
        safety,
        {
            "remote_code_downloaded",
            "remote_code_executed",
            "trust_remote_code",
            "weight_payload_bytes_read",
        },
        "manifest.safety",
    )
    for name in ("remote_code_downloaded", "remote_code_executed", "trust_remote_code"):
        _bool(safety[name], f"manifest.safety.{name}", False)
    _int(
        safety["weight_payload_bytes_read"],
        "manifest.safety.weight_payload_bytes_read",
        expected=0,
    )

    entries: dict[str, dict[str, object]] = {}
    aggregate = 0
    required = {"bytes", "path", "sha256", "source_url"}
    for index, raw in enumerate(_arr(manifest["files"], "manifest.files")):
        label = f"manifest.files[{index}]"
        entry = _obj(raw, label)
        path = _path(entry.get("path"), f"{label}.path", basename=True)
        allowed = required | {"derivation", "upstream_blob_id"}
        if not required <= set(entry) or set(entry) - allowed:
            raise SnapshotValidationError(f"{label}: invalid file-entry schema")
        if path in entries:
            raise SnapshotValidationError(f"manifest contains duplicate path: {path}")
        size = _int(entry["bytes"], f"{label}.bytes")
        if size > MAX_FILE_BYTES:
            raise SnapshotValidationError(f"{label}: file exceeds 32 MiB")
        aggregate += size
        _digest(entry["sha256"], f"{label}.sha256", _HEX64)
        if _str(entry["source_url"], f"{label}.source_url") != EXPECTED_URLS.get(path):
            raise SnapshotValidationError(f"{label}: source URL is not allowlisted")
        if path == "repository.json":
            _keys(entry, required | {"derivation"}, label)
            if _str(entry["derivation"], f"{label}.derivation") != (
                "canonical stable-field projection documented in source.selection"
            ):
                raise SnapshotValidationError("repository derivation does not match")
        else:
            _keys(entry, required | {"upstream_blob_id"}, label)
            _digest(entry["upstream_blob_id"], f"{label}.upstream_blob_id", _HEX40)
        entries[path] = entry

    if set(entries) != EXPECTED_FILES:
        raise SnapshotValidationError("manifest file set does not match the snapshot")
    if aggregate > MAX_SNAPSHOT_BYTES:
        raise SnapshotValidationError("manifest aggregate exceeds 128 MiB")
    return entries, safety


def _load_snapshot(
    snapshot_dir: Path,
) -> tuple[
    dict[str, bytes],
    dict[str, object],
    dict[str, dict[str, object]],
    int,
    int,
]:
    root = Path(snapshot_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    try:
        if root.is_symlink():
            raise SnapshotValidationError("snapshot directory must not be a symlink")
        if not root.is_dir():
            raise SnapshotValidationError(f"snapshot directory does not exist: {root}")
        children = list(root.iterdir())
    except SnapshotValidationError:
        raise
    except OSError as exc:
        raise SnapshotValidationError(f"cannot inspect snapshot directory: {exc}") from exc

    for child in children:
        if child.is_symlink():
            raise SnapshotValidationError(
                f"snapshot entry must not be a symlink: {child.name}"
            )
        if not child.is_file():
            raise SnapshotValidationError(
                f"snapshot contains a non-file entry: {child.name}"
            )
    if {child.name for child in children} != EXPECTED_FILES | {"manifest.json"}:
        raise SnapshotValidationError("snapshot directory file set does not match")

    manifest_bytes = _read_file(root, "manifest.json")
    entries, safety = _validate_manifest(
        loads_json_strict(manifest_bytes, label="manifest.json")
    )
    verified: dict[str, bytes] = {}
    listed_bytes = 0
    for path in sorted(entries):
        data = _read_file(root, path)
        expected_bytes = _int(entries[path]["bytes"], f"manifest[{path}].bytes")
        if len(data) != expected_bytes:
            raise SnapshotValidationError(
                f"{path}: byte length mismatch; "
                f"expected {expected_bytes}, observed {len(data)}"
            )
        expected_sha = _str(entries[path]["sha256"], f"manifest[{path}].sha256")
        observed_sha = hashlib.sha256(data).hexdigest()
        if observed_sha != expected_sha:
            raise SnapshotValidationError(
                f"{path}: SHA-256 mismatch; "
                f"expected {expected_sha}, observed {observed_sha}"
            )
        verified[path] = data
        listed_bytes += len(data)
    return verified, safety, entries, listed_bytes, len(manifest_bytes)


def _config_int(config: dict[str, object], key: str, expected: int) -> int:
    if key not in config:
        raise SnapshotValidationError(f"config.json: missing {key}")
    return _int(config[key], f"config.{key}", expected=expected)


def _validate_config(payload: object) -> tuple[dict[str, object], dict[str, object]]:
    config = _obj(payload, "config.json")
    required = {
        "architectures",
        "dspark_target_layer_ids",
        "expert_dtype",
        "hidden_size",
        "model_type",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "num_hash_layers",
        "num_hidden_layers",
        "num_nextn_predict_layers",
        "quantization_config",
        "torch_dtype",
    }
    if required - set(config):
        raise SnapshotValidationError(
            f"config.json: missing critical keys {sorted(required - set(config))}"
        )
    if _arr(config["architectures"], "config.architectures") != [
        "DeepseekV4ForCausalLM"
    ]:
        raise SnapshotValidationError("config.architectures does not match")
    if _str(config["model_type"], "config.model_type") != "deepseek_v4":
        raise SnapshotValidationError("config.model_type does not match")

    layers = _config_int(config, "num_hidden_layers", 43)
    hidden = _config_int(config, "hidden_size", 4096)
    routed = _config_int(config, "n_routed_experts", 256)
    shared = _config_int(config, "n_shared_experts", 1)
    topk = _config_int(config, "num_experts_per_tok", 6)
    hashes = _config_int(config, "num_hash_layers", 3)
    nextn = _config_int(config, "num_nextn_predict_layers", 1)
    if topk > routed:
        raise SnapshotValidationError("config routes more experts than it stores")

    dspark = [
        _int(value, f"config.dspark_target_layer_ids[{index}]")
        for index, value in enumerate(
            _arr(config["dspark_target_layer_ids"], "config.dspark_target_layer_ids")
        )
    ]
    if dspark != [40, 41, 42]:
        raise SnapshotValidationError("config.dspark_target_layer_ids does not match")
    if any(layer >= layers for layer in dspark) or len(set(dspark)) != len(dspark):
        raise SnapshotValidationError("config contains invalid DSpark layers")

    ratios = _arr(config.get("compress_ratios"), "config.compress_ratios")
    for index, value in enumerate(ratios):
        _int(value, f"config.compress_ratios[{index}]")
    if len(ratios) != layers + len(dspark):
        raise SnapshotValidationError("config.compress_ratios has the wrong length")

    expert_dtype = _str(config["expert_dtype"], "config.expert_dtype")
    if expert_dtype != "fp4":
        raise SnapshotValidationError(
            f"config.expert_dtype: expected fp4, observed {expert_dtype}"
        )
    torch_dtype = _str(config["torch_dtype"], "config.torch_dtype")
    if torch_dtype != "bfloat16":
        raise SnapshotValidationError(
            f"config.torch_dtype: expected bfloat16, observed {torch_dtype}"
        )

    quant = _obj(config["quantization_config"], "config.quantization_config")
    expected_quant = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
    }
    _keys(quant, set(expected_quant) | {"weight_block_size"}, "config.quantization_config")
    for key, expected in expected_quant.items():
        observed = _str(quant[key], f"config.quantization_config.{key}")
        if observed != expected:
            raise SnapshotValidationError(
                f"config.quantization_config.{key}: expected {expected}, observed {observed}"
            )
    block = [
        _int(value, f"config.quantization_config.weight_block_size[{index}]")
        for index, value in enumerate(
            _arr(
                quant["weight_block_size"],
                "config.quantization_config.weight_block_size",
            )
        )
    ]
    if block != [128, 128]:
        raise SnapshotValidationError("config weight block size does not match")

    return (
        {
            "hidden_layers": layers,
            "hidden_size": hidden,
            "routed_experts_per_layer": routed,
            "shared_experts_per_layer": shared,
            "experts_selected_per_token": topk,
            "hash_layers": hashes,
            "next_token_prediction_layers": nextn,
            "dspark_target_layer_ids": dspark,
        },
        {
            "expert_dtype": expert_dtype,
            "declared_torch_dtype": torch_dtype,
            "quantization_config": {
                **expected_quant,
                "weight_block_size": block,
            },
        },
    )


def _shards(names: set[str], label: str) -> int:
    ordinals: set[int] = set()
    denominators: set[int] = set()
    for name in names:
        match = _SHARD.fullmatch(name)
        if match is None:
            raise SnapshotValidationError(
                f"{label}: unsafe or malformed shard name: {name!r}"
            )
        ordinals.add(int(match.group(1)))
        denominators.add(int(match.group(2)))
    if not names:
        raise SnapshotValidationError(f"{label}: empty shard inventory")
    if len(denominators) != 1:
        raise SnapshotValidationError(f"{label}: inconsistent shard denominators")
    denominator = next(iter(denominators))
    if denominator != 48:
        raise SnapshotValidationError(
            f"{label}: expected 48-shard denominator, observed {denominator}"
        )
    expected = set(range(1, denominator + 1))
    if ordinals != expected:
        raise SnapshotValidationError(
            f"{label}: shard ordinal gap; "
            f"missing={sorted(expected - ordinals)}, "
            f"unexpected={sorted(ordinals - expected)}"
        )
    if len(names) != denominator:
        raise SnapshotValidationError(
            f"{label}: expected {denominator} unique shards, observed {len(names)}"
        )
    return denominator


def _validate_repository(
    payload: object,
) -> tuple[dict[str, int], dict[str, int], set[str], dict[str, str]]:
    repository = _obj(payload, "repository.json")
    _keys(
        repository,
        {
            "artifact_bytes",
            "files",
            "last_modified",
            "repository",
            "revision",
            "safetensors_parameter_classes",
            "schema_version",
            "source",
        },
        "repository.json",
    )
    _int(repository["schema_version"], "repository.schema_version", expected=1)
    if _str(repository["repository"], "repository.repository") != TARGET_REPOSITORY:
        raise SnapshotValidationError("repository target does not match")
    if _str(repository["revision"], "repository.revision") != TARGET_REVISION:
        raise SnapshotValidationError("repository revision does not match")
    _str(repository["last_modified"], "repository.last_modified")
    source = _obj(repository["source"], "repository.source")
    _keys(source, {"retrieved_at_utc", "selection", "url"}, "repository.source")
    _str(source["retrieved_at_utc"], "repository.source.retrieved_at_utc")
    _str(source["selection"], "repository.source.selection")
    if _str(source["url"], "repository.source.url") != EXPECTED_URLS["repository.json"]:
        raise SnapshotValidationError("repository source URL does not match")

    remote_files = _arr(repository["files"], "repository.files")
    if len(remote_files) != EXPECTED_REPOSITORY_FILE_COUNT:
        raise SnapshotValidationError(
            "repository file count drifted; "
            f"expected {EXPECTED_REPOSITORY_FILE_COUNT}, observed {len(remote_files)}"
        )
    seen: set[str] = set()
    weight_names: set[str] = set()
    repository_blob_ids: dict[str, str] = {}
    all_sum = 0
    weight_sum = 0
    for index, raw in enumerate(remote_files):
        label = f"repository.files[{index}]"
        item = _obj(raw, label)
        _keys(item, {"blob_id", "bytes", "path", "storage"}, label)
        path = _path(item["path"], f"{label}.path", basename=False)
        if path in seen:
            raise SnapshotValidationError(f"repository contains duplicate path: {path}")
        seen.add(path)
        blob_id = _digest(item["blob_id"], f"{label}.blob_id", _HEX40)
        if path in set(REPOSITORY_PATHS.values()):
            repository_blob_ids[path] = blob_id
        size = _int(item["bytes"], f"{label}.bytes")
        storage = _obj(item["storage"], f"{label}.storage")
        kind = _str(storage.get("kind"), f"{label}.storage.kind")
        is_shard = _SHARD.fullmatch(path) is not None
        if kind == "git":
            _keys(storage, {"kind"}, f"{label}.storage")
            if is_shard:
                raise SnapshotValidationError("weight shard is not declared as LFS")
        elif kind == "lfs":
            _keys(
                storage,
                {"bytes", "kind", "pointer_bytes", "sha256"},
                f"{label}.storage",
            )
            if not is_shard:
                raise SnapshotValidationError("unexpected non-weight LFS object")
            if _int(storage["bytes"], f"{label}.storage.bytes") != size:
                raise SnapshotValidationError(f"{label}: LFS byte count mismatch")
            _int(storage["pointer_bytes"], f"{label}.storage.pointer_bytes")
            _digest(storage["sha256"], f"{label}.storage.sha256", _HEX64)
            weight_names.add(path)
            weight_sum += size
        else:
            raise SnapshotValidationError(f"{label}: unsupported storage kind {kind!r}")
        all_sum += size
    _shards(weight_names, "repository.json")

    artifacts = _obj(repository["artifact_bytes"], "repository.artifact_bytes")
    _keys(
        artifacts,
        {"all_repository_files", "non_weight_files", "weight_shards"},
        "repository.artifact_bytes",
    )
    ledger = {
        key: _int(value, f"repository.artifact_bytes.{key}")
        for key, value in artifacts.items()
    }
    if ledger["all_repository_files"] != all_sum:
        raise SnapshotValidationError("repository all-file byte ledger does not add up")
    if ledger["weight_shards"] != weight_sum:
        raise SnapshotValidationError("repository weight byte ledger does not add up")
    if ledger["non_weight_files"] != all_sum - weight_sum:
        raise SnapshotValidationError("repository non-weight byte ledger does not add up")
    if ledger["weight_shards"] + ledger["non_weight_files"] != ledger["all_repository_files"]:
        raise SnapshotValidationError("repository byte ledgers are inconsistent")
    if ledger != EXPECTED_ARTIFACT_BYTES:
        raise SnapshotValidationError(
            "repository byte ledger drifted; "
            f"expected {EXPECTED_ARTIFACT_BYTES}, observed {ledger}"
        )

    classes = _obj(
        repository["safetensors_parameter_classes"],
        "repository.safetensors_parameter_classes",
    )
    _keys(classes, {"parameters", "total"}, "repository.safetensors_parameter_classes")
    parameters = _obj(
        classes["parameters"], "repository.safetensors_parameter_classes.parameters"
    )
    _keys(
        parameters,
        {"BF16", "F32", "F8_E4M3", "I64", "I8"},
        "repository.safetensors_parameter_classes.parameters",
    )
    counts = {
        key: _int(
            value,
            f"repository.safetensors_parameter_classes.parameters.{key}",
        )
        for key, value in parameters.items()
    }
    total = _int(classes["total"], "repository.safetensors_parameter_classes.total")
    if sum(counts.values()) != total:
        raise SnapshotValidationError("repository parameter ledger does not add up")
    counts["total"] = total
    if counts != EXPECTED_PARAMETER_COUNTS:
        raise SnapshotValidationError(
            "repository parameter classes drifted; "
            f"expected {EXPECTED_PARAMETER_COUNTS}, observed {counts}"
        )
    return ledger, counts, weight_names, repository_blob_ids


def _slot_census(
    slots: dict[tuple[int, int], set[str]],
    outer_count: int,
    expert_count: int,
    label: str,
) -> None:
    expected = {
        (outer, expert)
        for outer in range(outer_count)
        for expert in range(expert_count)
    }
    if set(slots) != expected:
        raise SnapshotValidationError(
            f"{label}: expert-slot census mismatch; "
            f"missing={sorted(expected - set(slots))[:5]}, "
            f"unexpected={sorted(set(slots) - expected)[:5]}"
        )
    for position, components in slots.items():
        if components != _COMPONENTS:
            raise SnapshotValidationError(
                f"{label} {position}: tensor census mismatch; "
                f"missing={sorted(_COMPONENTS - components)}, "
                f"unexpected={sorted(components - _COMPONENTS)}"
            )


def _shared_census(slots: dict[int, set[str]], count: int, label: str) -> None:
    expected = set(range(count))
    if set(slots) != expected:
        raise SnapshotValidationError(
            f"{label}: shared-expert census mismatch; "
            f"missing={sorted(expected - set(slots))}, "
            f"unexpected={sorted(set(slots) - expected)}"
        )
    for position, components in slots.items():
        if components != _COMPONENTS:
            raise SnapshotValidationError(
                f"{label} {position}: tensor census mismatch; "
                f"missing={sorted(_COMPONENTS - components)}, "
                f"unexpected={sorted(components - _COMPONENTS)}"
            )


def _expert_census(
    names: Iterable[str], layers: int, experts: int, attachments: int
) -> dict[str, int]:
    routed: dict[tuple[int, int], set[str]] = {}
    shared: dict[int, set[str]] = {}
    mtp_routed: dict[tuple[int, int], set[str]] = {}
    mtp_shared: dict[int, set[str]] = {}

    for name in names:
        layer = _LAYER.match(name)
        if layer is not None and int(layer.group(1)) >= layers:
            raise SnapshotValidationError(f"tensor references out-of-range layer: {name}")
        mtp = _MTP.match(name)
        if mtp is not None and int(mtp.group(1)) >= attachments:
            raise SnapshotValidationError(
                f"tensor references out-of-range attachment: {name}"
            )

        match = _ROUTED.fullmatch(name)
        if match:
            position = (int(match.group(1)), int(match.group(2)))
            if position[1] >= experts:
                raise SnapshotValidationError(f"out-of-range routed expert: {name}")
            routed.setdefault(position, set()).add(f"{match.group(3)}.{match.group(4)}")
            continue
        match = _SHARED.fullmatch(name)
        if match:
            shared.setdefault(int(match.group(1)), set()).add(
                f"{match.group(2)}.{match.group(3)}"
            )
            continue
        match = _MTP_ROUTED.fullmatch(name)
        if match:
            position = (int(match.group(1)), int(match.group(2)))
            if position[1] >= experts:
                raise SnapshotValidationError(f"out-of-range attachment expert: {name}")
            mtp_routed.setdefault(position, set()).add(
                f"{match.group(3)}.{match.group(4)}"
            )
            continue
        match = _MTP_SHARED.fullmatch(name)
        if match:
            mtp_shared.setdefault(int(match.group(1)), set()).add(
                f"{match.group(2)}.{match.group(3)}"
            )
            continue
        if ".ffn.experts." in name or ".ffn.shared_experts." in name:
            raise SnapshotValidationError(f"unrecognized expert tensor name: {name}")

    _slot_census(routed, layers, experts, "backbone routed experts")
    _shared_census(shared, layers, "backbone shared experts")
    _slot_census(mtp_routed, attachments, experts, "attachment routed experts")
    _shared_census(mtp_shared, attachments, "attachment shared experts")
    return {
        "backbone_routed_expert_slots": len(routed),
        "backbone_routed_tensor_entries": sum(map(len, routed.values())),
        "backbone_shared_expert_slots": len(shared),
        "backbone_shared_tensor_entries": sum(map(len, shared.values())),
        "attachment_routed_expert_slots": len(mtp_routed),
        "attachment_routed_tensor_entries": sum(map(len, mtp_routed.values())),
        "attachment_shared_expert_slots": len(mtp_shared),
        "attachment_shared_tensor_entries": sum(map(len, mtp_shared.values())),
    }


def _validate_index(
    payload: object,
    topology: dict[str, object],
    repository_shards: set[str],
    repository_weight_bytes: int,
) -> tuple[int, int, int, int, dict[str, int]]:
    index = _obj(payload, "model.safetensors.index.json")
    _keys(index, {"metadata", "weight_map"}, "model.safetensors.index.json")
    metadata = _obj(index["metadata"], "index.metadata")
    _keys(metadata, {"total_size"}, "index.metadata")
    payload_bytes = _int(metadata["total_size"], "index.metadata.total_size")
    if payload_bytes != EXPECTED_TENSOR_PAYLOAD_BYTES:
        raise SnapshotValidationError(
            "index tensor payload bytes drifted; "
            f"expected {EXPECTED_TENSOR_PAYLOAD_BYTES}, observed {payload_bytes}"
        )
    if payload_bytes > repository_weight_bytes:
        raise SnapshotValidationError("tensor payload exceeds weight-shard bytes")

    raw_map = _obj(index["weight_map"], "index.weight_map")
    if not raw_map:
        raise SnapshotValidationError("index.weight_map is empty")
    weight_map: dict[str, str] = {}
    for tensor, raw_shard in raw_map.items():
        if (
            not tensor
            or tensor != tensor.strip()
            or "\x00" in tensor
            or "/" in tensor
            or "\\" in tensor
        ):
            raise SnapshotValidationError(f"unsafe tensor name: {tensor!r}")
        shard = _str(raw_shard, f"index.weight_map[{tensor!r}]")
        if _SHARD.fullmatch(shard) is None:
            raise SnapshotValidationError(
                f"index.weight_map: unsafe or malformed shard name: {shard!r}"
            )
        weight_map[tensor] = shard

    index_shards = set(weight_map.values())
    denominator = _shards(index_shards, "index.weight_map")
    if index_shards != repository_shards:
        raise SnapshotValidationError(
            "index and repository shard inventories differ; "
            f"missing={sorted(repository_shards - index_shards)}, "
            f"unexpected={sorted(index_shards - repository_shards)}"
        )

    layers = _int(topology["hidden_layers"], "topology.hidden_layers", expected=43)
    experts = _int(
        topology["routed_experts_per_layer"],
        "topology.routed_experts_per_layer",
        expected=256,
    )
    attachments = len(
        _arr(topology["dspark_target_layer_ids"], "topology.dspark_target_layer_ids")
    )
    census = _expert_census(weight_map, layers, experts, attachments)
    if len(weight_map) != EXPECTED_TENSOR_COUNT:
        raise SnapshotValidationError(
            "index tensor count drifted; "
            f"expected {EXPECTED_TENSOR_COUNT}, observed {len(weight_map)}"
        )
    return len(weight_map), len(index_shards), denominator, payload_bytes, census


def _git_blob_id(data: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _validate_reviewed_identities(
    files: dict[str, bytes],
    entries: dict[str, dict[str, object]],
    repository_blob_ids: dict[str, str],
) -> None:
    for path in ("config.json", "model.safetensors.index.json", "LICENSE.target.txt"):
        expected_bytes, expected_sha, expected_blob = REVIEWED_FILE_IDENTITIES[path]
        entry = entries[path]
        if _int(entry["bytes"], f"manifest[{path}].bytes") != expected_bytes:
            raise SnapshotValidationError(
                f"{path}: reviewed manifest byte identity drifted"
            )
        if _str(entry["sha256"], f"manifest[{path}].sha256") != expected_sha:
            raise SnapshotValidationError(
                f"{path}: reviewed manifest SHA-256 identity drifted"
            )
        if _str(entry["upstream_blob_id"], f"manifest[{path}].upstream_blob_id") != expected_blob:
            raise SnapshotValidationError(
                f"{path}: reviewed upstream blob identity drifted"
            )
        data = files[path]
        if len(data) != expected_bytes or hashlib.sha256(data).hexdigest() != expected_sha:
            raise SnapshotValidationError(
                f"{path}: reviewed official content identity drifted"
            )
        if _git_blob_id(data) != expected_blob:
            raise SnapshotValidationError(
                f"{path}: computed Git blob identity does not match the official blob"
            )
        repository_path = REPOSITORY_PATHS[path]
        if repository_blob_ids.get(repository_path) != expected_blob:
            raise SnapshotValidationError(
                f"{path}: repository.json blob cross-link drifted"
            )

    repository_bytes, repository_sha, _ = REVIEWED_FILE_IDENTITIES["repository.json"]
    repository_entry = entries["repository.json"]
    if _int(repository_entry["bytes"], "manifest[repository.json].bytes") != repository_bytes:
        raise SnapshotValidationError(
            "repository.json: reviewed manifest byte identity drifted"
        )
    if _str(repository_entry["sha256"], "manifest[repository.json].sha256") != repository_sha:
        raise SnapshotValidationError(
            "repository.json: reviewed manifest SHA-256 identity drifted"
        )
    repository_data = files["repository.json"]
    if (
        len(repository_data) != repository_bytes
        or hashlib.sha256(repository_data).hexdigest() != repository_sha
    ):
        raise SnapshotValidationError(
            "repository.json: reviewed official content identity drifted"
        )


def inspect_snapshot(snapshot_dir: Path = DEFAULT_SNAPSHOT) -> dict[str, object]:
    """Validate and summarize the snapshot without network or remote code."""

    files, safety, entries, listed_bytes, manifest_bytes = _load_snapshot(snapshot_dir)
    topology, native = _validate_config(
        loads_json_strict(files["config.json"], label="config.json")
    )
    (
        artifacts,
        parameters,
        repository_shards,
        repository_blob_ids,
    ) = _validate_repository(
        loads_json_strict(files["repository.json"], label="repository.json")
    )
    tensor_count, shard_count, denominator, payload_bytes, census = _validate_index(
        loads_json_strict(
            files["model.safetensors.index.json"],
            label="model.safetensors.index.json",
        ),
        topology,
        repository_shards,
        artifacts["weight_shards"],
    )
    _validate_reviewed_identities(files, entries, repository_blob_ids)
    return {
        "schema_version": 1,
        "status": "validated",
        "evidence_level": "pinned-official-metadata",
        "target": {
            "repository": TARGET_REPOSITORY,
            "revision": TARGET_REVISION,
        },
        "topology": {
            "evidence_tag": "source-reproduced",
            "source": "config.json",
            **topology,
        },
        "native_representation": {
            "evidence_tag": "source-reproduced",
            "source": "config.json",
            **native,
        },
        "tensor_index": {
            "evidence_tag": "source-reproduced",
            "source": "model.safetensors.index.json",
            "tensor_entries": tensor_count,
            "shard_count": shard_count,
            "shard_denominator": denominator,
            "expert_census": census,
        },
        "parameter_ledger": {
            "evidence_tag": "source-reproduced",
            "source": "repository.json",
            "counts_by_storage_class": parameters,
        },
        "byte_ledgers": {
            "target_tensor_payload": {
                "evidence_tag": "source-reproduced",
                "source": "model.safetensors.index.json",
                "bytes": payload_bytes,
            },
            "target_repository_artifacts": {
                "evidence_tag": "source-reproduced",
                "source": "repository.json",
                "weight_shard_bytes": artifacts["weight_shards"],
                "non_weight_file_bytes": artifacts["non_weight_files"],
                "all_repository_file_bytes": artifacts["all_repository_files"],
            },
            "derived_container_overhead": {
                "evidence_tag": "derived",
                "formula": "weight_shard_bytes - tensor_payload_bytes",
                "bytes": artifacts["weight_shards"] - payload_bytes,
            },
            "committed_metadata_snapshot": {
                "evidence_tag": "measured",
                "source": "manifest.json and verified local files",
                "listed_file_count": len(files),
                "listed_file_bytes": listed_bytes,
                "manifest_bytes": manifest_bytes,
                "directory_bytes": listed_bytes + manifest_bytes,
            },
        },
        "safety": {
            "decoder_mode": "offline",
            "remote_code_execution": False,
            "weight_shards_opened": False,
            "full_checkpoint": "NOT DOWNLOADED / NOT RUN",
            "snapshot_attestation": {
                "evidence_tag": "source-reproduced",
                "source": "manifest.json",
                **safety,
            },
        },
    }



