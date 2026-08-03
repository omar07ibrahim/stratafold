"""Resource and download policy enforcement.

The module intentionally uses only the Python standard library so the doctor
can run before any dependency installation or build. URL validation precedes
network-opener construction, which makes forbidden weight requests testably
fail before a payload byte can be transferred.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request


GIB = 1024**3
MIB = 1024**2


class DoctorError(RuntimeError):
    """Base class for fail-closed safety violations."""


class RequestPolicyError(DoctorError):
    """A planned network request is outside the metadata-only policy."""


@dataclass(frozen=True)
class DoctorLimits:
    min_free_bytes: int = 2 * GIB
    max_workspace_bytes: int = 750 * MIB
    max_cache_bytes: int = 128 * MIB
    max_remote_object_bytes: int = 32 * MIB
    max_remote_aggregate_bytes: int = 128 * MIB


@dataclass(frozen=True)
class ResourceReport:
    ok: bool
    free_bytes: int
    workspace_bytes: int
    cache_bytes: int
    memory_available_bytes: int | None
    cpu_count: int
    filesystem_path: str
    limits: DoctorLimits
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["status"] = "pass" if self.ok else "rejected"
        return result


@dataclass(frozen=True)
class RemoteRequest:
    url: str
    pinned_revision: str
    content_length: int | None
    aggregate_bytes: int
    status: str = "approved-metadata-plan"
    payload_bytes_read: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _tree_size(path: Path) -> int:
    """Return allocated-independent logical bytes without following symlinks."""

    if not path.exists():
        return 0
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except FileNotFoundError:
                        continue
        except FileNotFoundError:
            continue
    return total


def _memory_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None
    return None


class ResourceDoctor:
    def __init__(self, limits: DoctorLimits) -> None:
        self.limits = limits

    def inspect(self, *, workspace: Path, cache: Path) -> ResourceReport:
        workspace = workspace.resolve()
        cache = cache.resolve()
        filesystem_path = Path(workspace.anchor or os.sep)
        stat = os.statvfs(filesystem_path)
        free_bytes = stat.f_bavail * stat.f_frsize
        workspace_bytes = _tree_size(workspace)
        cache_bytes = _tree_size(cache)
        violations: list[str] = []

        if free_bytes < self.limits.min_free_bytes:
            violations.append(
                f"free space {free_bytes} is below required {self.limits.min_free_bytes} bytes"
            )
        if workspace_bytes > self.limits.max_workspace_bytes:
            violations.append(
                f"workspace {workspace_bytes} exceeds {self.limits.max_workspace_bytes} bytes"
            )
        if cache_bytes > self.limits.max_cache_bytes:
            violations.append(f"cache {cache_bytes} exceeds {self.limits.max_cache_bytes} bytes")

        return ResourceReport(
            ok=not violations,
            free_bytes=free_bytes,
            workspace_bytes=workspace_bytes,
            cache_bytes=cache_bytes,
            memory_available_bytes=_memory_available_bytes(),
            cpu_count=os.cpu_count() or 1,
            filesystem_path=str(filesystem_path),
            limits=self.limits,
            violations=tuple(violations),
        )


_REVISION = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".onnx",
)
_FORBIDDEN_HOST_FRAGMENTS = (
    "xethub",
    "cdn-lfs",
    "objects.githubusercontent.com",
)
_FORBIDDEN_PATH_FRAGMENTS = ("/xet/", "/lfs/", "git-lfs", "cas-bridge")
_ALLOWED_HOSTS = frozenset({"huggingface.co"})
_ALLOWED_METADATA_SUFFIXES = (
    ".json",
    ".md",
    ".txt",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
)


def _validate_url_shape(url: str, pinned_revision: str) -> None:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    try:
        explicit_port = parsed.port
    except ValueError as exc:
        raise RequestPolicyError("metadata URL contains an invalid port") from exc
    decoded_path = parsed.path
    for _ in range(3):
        expanded = unquote(decoded_path)
        if expanded == decoded_path:
            break
        decoded_path = expanded
    path = decoded_path.lower()

    if parsed.scheme != "https":
        raise RequestPolicyError("only HTTPS metadata requests are permitted")
    if parsed.username or parsed.password:
        raise RequestPolicyError("credentials are forbidden in metadata URLs")
    if explicit_port is not None:
        raise RequestPolicyError("explicit ports are forbidden in metadata URLs")
    if parsed.query or parsed.fragment:
        raise RequestPolicyError("query strings and fragments are forbidden in pinned metadata URLs")
    if host not in _ALLOWED_HOSTS:
        raise RequestPolicyError(f"host is outside the allowlist: {host or '<missing>'}")
    if any(fragment in host for fragment in _FORBIDDEN_HOST_FRAGMENTS):
        raise RequestPolicyError("LFS/Xet object hosts are forbidden")
    if any(fragment in path for fragment in _FORBIDDEN_PATH_FRAGMENTS):
        raise RequestPolicyError("LFS/Xet object paths are forbidden")
    if "%" in path or "\\" in path or any(
        part in {".", ".."} for part in path.split("/")
    ):
        raise RequestPolicyError("encoded or traversing metadata paths are forbidden")
    if path.endswith(_FORBIDDEN_WEIGHT_SUFFIXES):
        raise RequestPolicyError("model weight objects are forbidden on this host")
    if not path.endswith(_ALLOWED_METADATA_SUFFIXES):
        raise RequestPolicyError("only explicitly recognized metadata file types are permitted")
    if not isinstance(pinned_revision, str) or not _REVISION.fullmatch(pinned_revision):
        raise RequestPolicyError("pinned revision must be exactly 40 lowercase hex characters")
    pinned_resolve = f"/resolve/{pinned_revision}/"
    pinned_blob = f"/blob/{pinned_revision}/"
    pinned_raw = f"/raw/{pinned_revision}/"
    if (
        pinned_resolve not in parsed.path
        and pinned_blob not in parsed.path
        and pinned_raw not in parsed.path
    ):
        raise RequestPolicyError("metadata URL is not revision-bound in its path")


def validate_remote_request(
    url: str,
    *,
    pinned_revision: str | None,
    content_length: int | None = None,
    aggregate_bytes: int = 0,
    limits: DoctorLimits | None = None,
) -> RemoteRequest:
    """Validate a metadata request without opening a socket."""

    active_limits = limits or DoctorLimits()
    if pinned_revision is None:
        raise RequestPolicyError("an exact revision is required before remote access")
    _validate_url_shape(url, pinned_revision)
    if isinstance(aggregate_bytes, bool) or not isinstance(aggregate_bytes, int):
        raise RequestPolicyError("aggregate byte count must be an integer")
    if aggregate_bytes < 0:
        raise RequestPolicyError("aggregate byte count cannot be negative")
    if content_length is None:
        raise RequestPolicyError("known content length is required before a payload open")
    if isinstance(content_length, bool) or not isinstance(content_length, int):
        raise RequestPolicyError("content length must be an integer")
    if content_length < 0:
        raise RequestPolicyError("content length cannot be negative")
    if content_length > active_limits.max_remote_object_bytes:
        raise RequestPolicyError(
            f"remote object exceeds {active_limits.max_remote_object_bytes} bytes"
        )
    if aggregate_bytes + content_length > active_limits.max_remote_aggregate_bytes:
        raise RequestPolicyError(
            f"aggregate fetch exceeds {active_limits.max_remote_aggregate_bytes} bytes"
        )
    return RemoteRequest(
        url=url,
        pinned_revision=pinned_revision,
        content_length=content_length,
        aggregate_bytes=aggregate_bytes,
    )


class RejectAllRedirects(HTTPRedirectHandler):
    """Abort before urllib constructs any redirected request.

    M0 deliberately does not provide a network fetch command. M1 composes this
    handler with bounded HEAD/GET streaming and exact URL/header allowlists.
    """

    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> Request | None:
        del req, fp, code, msg, headers, newurl
        raise RequestPolicyError("redirects are forbidden for metadata requests")


def limits_as_mapping(limits: DoctorLimits | None = None) -> Mapping[str, int]:
    """Expose stable policy values to manifests and tests."""

    return asdict(limits or DoctorLimits())
