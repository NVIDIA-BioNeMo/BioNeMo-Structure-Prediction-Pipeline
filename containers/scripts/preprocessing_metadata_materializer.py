#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Safely materialize the UniRef mapping and taxonomy sidecars from tar+gzip.

The materializer is deliberately idempotent.  A completed success or failure is
published behind an immutable intent pointer; a later invocation validates that
pointer and the published bytes without reading the archive payload again.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import platform
import stat
import sys
import tarfile
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast

SCHEMA_VERSION = 1
MANIFEST_KIND = "preprocessing-metadata-materialization"
POINTER_KIND = "preprocessing-metadata-materialization-pointer"
EXPECTED_ARCHIVE_SIZE_BYTES = 102_918_187_842
LOGICAL_MAPPING = "uniref30_2302_db_mapping"
LOGICAL_TAXONOMY = "uniref30_2302_db_taxonomy"
CANDIDATE_BASENAMES: dict[str, tuple[str, ...]] = {
    LOGICAL_MAPPING: ("uniref30_2302_mapping", LOGICAL_MAPPING),
    LOGICAL_TAXONOMY: ("uniref30_2302_taxonomy", LOGICAL_TAXONOMY),
}
EXPECTED_OUTPUT_SIZE_BYTES: dict[str, int] = {
    LOGICAL_MAPPING: 5_797_891_705,
    LOGICAL_TAXONOMY: 667_957_493,
}
_HASH_CHUNK_BYTES = 1024 * 1024
_TAR_STREAM_BUFFER_BYTES = 64 * 1024
_HEX_DIGITS = frozenset("0123456789abcdef")
_UMASK_LOCK = threading.RLock()
_EXECUTION_CONTEXT_KEYS = frozenset(
    {
        "context_kind",
        "slurm_job_id",
        "slurmd_nodename",
        "materializer_sha256",
        "python_executable",
        "python_version",
    }
)


class MaterializationError(RuntimeError):
    """Base class for a materialization contract failure."""


class ArchiveValidationError(MaterializationError):
    """Raised before scanning when the archive cannot be safely bound."""


class ReuseValidationError(MaterializationError):
    """Raised when an existing immutable result no longer validates."""


class _HashingReader:
    """Count and hash every compressed byte read from a forward-only stream."""

    def __init__(self, source: BinaryIO) -> None:
        self._source = source
        self._digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        payload = self._source.read(size)
        self._digest.update(payload)
        self.bytes_read += len(payload)
        return payload

    def tell(self) -> int:
        return self._source.tell()

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


@dataclass(frozen=True)
class ArchiveIdentity:
    input_path: str
    resolved_path: str
    size_bytes: int
    mode: int
    mtime_ns: int
    device: int
    inode: int

    def as_json(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "resolved_path": self.resolved_path,
            "size_bytes": self.size_bytes,
            "mode": oct(self.mode),
            "mtime_ns": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
        }

    def intent_json(self) -> dict[str, Any]:
        return {
            "resolved_path": self.resolved_path,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
        }


@dataclass(frozen=True)
class PublicationLayout:
    root: Path
    objects: Path
    failures: Path
    by_intent: Path
    locks: Path
    staging: Path


@dataclass
class OutputCapture:
    logical_name: str
    source_member: str
    source_basename: str
    declared_size_bytes: int
    size_bytes: int
    sha256: str
    status: str

    def as_json(self, expected_size_bytes: int) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "file_name": self.logical_name,
            "source_member": self.source_member,
            "source_basename": self.source_basename,
            "declared_size_bytes": self.declared_size_bytes,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "status": self.status,
            "expected_size_bytes": expected_size_bytes,
            "size_matches_expected": self.size_bytes == expected_size_bytes,
        }


@dataclass(frozen=True)
class ExtractionAttempt:
    capture: OutputCapture | None
    output_error: str | None
    archive_error: str | None


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_sha256(value: str, field: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in _HEX_DIGITS for character in normalized):
        raise MaterializationError(f"{field} must be a 64-character SHA-256")
    if value != normalized:
        raise MaterializationError(f"{field} must use lowercase hexadecimal")
    return value


def materializer_sha256() -> str:
    """Return the digest of the exact committed tool bytes executing this call."""
    return _sha256_file(Path(__file__).resolve(strict=True))


def _build_execution_context(*, tool_sha256: str, local_test_context: bool) -> dict[str, Any]:
    _validate_sha256(tool_sha256, "materializer_sha256")
    if sys.version_info[:2] != (3, 12):
        raise MaterializationError(f"materializer requires pinned Python 3.12, observed {platform.python_version()}")
    try:
        python_executable = str(Path(sys.executable).resolve(strict=True))
    except OSError as exc:
        raise MaterializationError(f"Python executable cannot be resolved: {sys.executable}: {exc}") from exc
    if local_test_context:
        context_kind = "local-test"
        slurm_job_id: str | None = None
        slurmd_nodename: str | None = None
    else:
        context_kind = "slurm"
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        slurmd_nodename = os.environ.get("SLURMD_NODENAME")
        if not slurm_job_id or not slurmd_nodename:
            raise MaterializationError(
                "SLURM_JOB_ID and SLURMD_NODENAME are required; use --local-test-context only for tests"
            )
        if any(character.isspace() for character in slurm_job_id + slurmd_nodename):
            raise MaterializationError("Slurm job id and node name must not contain whitespace")
    return {
        "context_kind": context_kind,
        "slurm_job_id": slurm_job_id,
        "slurmd_nodename": slurmd_nodename,
        "materializer_sha256": tool_sha256,
        "python_executable": python_executable,
        "python_version": platform.python_version(),
    }


def _validate_execution_context(
    value: object,
    *,
    intent: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    if not isinstance(value, dict) or set(value) != _EXECUTION_CONTEXT_KEYS:
        raise ReuseValidationError("execution_context has an unexpected shape")
    context_kind = value.get("context_kind")
    if context_kind not in {"slurm", "local-test"}:
        raise ReuseValidationError("execution_context.context_kind is invalid")
    tool_sha256 = value.get("materializer_sha256")
    if not isinstance(tool_sha256, str):
        raise ReuseValidationError("execution_context.materializer_sha256 must be a string")
    try:
        _validate_sha256(tool_sha256, "execution_context.materializer_sha256")
    except MaterializationError as exc:
        raise ReuseValidationError(str(exc)) from exc
    if tool_sha256 != intent.get("materializer_sha256"):
        raise ReuseValidationError("execution_context materializer hash does not match the intent")
    python_executable = value.get("python_executable")
    python_version = value.get("python_version")
    if (
        not isinstance(python_executable, str)
        or not Path(python_executable).is_absolute()
        or not isinstance(python_version, str)
        or not python_version.startswith("3.12.")
    ):
        raise ReuseValidationError("execution_context does not identify a resolved Python 3.12 executable")
    slurm_job_id = value.get("slurm_job_id")
    slurmd_nodename = value.get("slurmd_nodename")
    if context_kind == "slurm":
        if not isinstance(slurm_job_id, str) or not slurm_job_id:
            raise ReuseValidationError("execution_context.slurm_job_id is missing")
        if not isinstance(slurmd_nodename, str) or not slurmd_nodename:
            raise ReuseValidationError("execution_context.slurmd_nodename is missing")
        if any(character.isspace() for character in slurm_job_id + slurmd_nodename):
            raise ReuseValidationError("execution_context Slurm identity contains whitespace")
    elif slurm_job_id is not None or slurmd_nodename is not None:
        raise ReuseValidationError("local-test execution context must not claim a Slurm job or node")
    # A completed Slurm result may be reused by another job.  The execution
    # identity must otherwise equal the current pinned tool/Python context.
    for key in ("context_kind", "materializer_sha256", "python_executable", "python_version"):
        if value.get(key) != current.get(key):
            raise ReuseValidationError(f"execution_context.{key} does not match the current runtime")


def _validate_configuration(
    candidate_basenames: Mapping[str, tuple[str, ...]],
    expected_output_sizes: Mapping[str, int],
) -> None:
    logical_names = {LOGICAL_MAPPING, LOGICAL_TAXONOMY}
    if set(candidate_basenames) != logical_names:
        raise MaterializationError("candidate_basenames must define exactly mapping and taxonomy")
    if set(expected_output_sizes) != logical_names:
        raise MaterializationError("expected_output_sizes must define exactly mapping and taxonomy")
    observed_basenames: set[str] = set()
    for logical_name, basenames in candidate_basenames.items():
        if not basenames:
            raise MaterializationError(f"candidate basename set is empty: {logical_name}")
        for basename in basenames:
            if (
                not basename
                or PurePosixPath(basename).name != basename
                or basename in {".", ".."}
                or "\\" in basename
                or "\x00" in basename
            ):
                raise MaterializationError(f"candidate is not a safe basename: {basename!r}")
            if basename in observed_basenames:
                raise MaterializationError(f"candidate basename is assigned more than once: {basename}")
            observed_basenames.add(basename)
    for logical_name, size in expected_output_sizes.items():
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise MaterializationError(f"expected output size must be a non-negative integer: {logical_name}")


def _stat_matches(identity: ArchiveIdentity, observed: os.stat_result) -> bool:
    return (
        stat.S_ISREG(observed.st_mode)
        and observed.st_size == identity.size_bytes
        and stat.S_IMODE(observed.st_mode) == identity.mode
        and observed.st_mtime_ns == identity.mtime_ns
        and observed.st_dev == identity.device
        and observed.st_ino == identity.inode
    )


def _capture_archive_identity(path: Path, expected_size_bytes: int | None) -> ArchiveIdentity:
    absolute = path.absolute()
    try:
        path_info = absolute.lstat()
    except OSError as exc:
        raise ArchiveValidationError(f"archive cannot be inspected: {absolute}: {exc}") from exc
    if stat.S_ISLNK(path_info.st_mode):
        raise ArchiveValidationError(f"archive must not be a symlink: {absolute}")
    if not stat.S_ISREG(path_info.st_mode):
        raise ArchiveValidationError(f"archive must be a regular file: {absolute}")
    try:
        resolved = absolute.resolve(strict=True)
        descriptor = os.open(absolute, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise ArchiveValidationError(
            f"archive must be readable without following a symlink: {absolute}: {exc}"
        ) from exc
    try:
        descriptor_info = os.fstat(descriptor)
        if path_info.st_dev != descriptor_info.st_dev or path_info.st_ino != descriptor_info.st_ino:
            raise ArchiveValidationError(f"archive changed while it was opened: {absolute}")
        if not stat.S_ISREG(descriptor_info.st_mode):
            raise ArchiveValidationError(f"archive descriptor is not regular: {absolute}")
    finally:
        os.close(descriptor)
    if expected_size_bytes is not None and descriptor_info.st_size != expected_size_bytes:
        raise ArchiveValidationError(
            f"archive size is {descriptor_info.st_size}, expected {expected_size_bytes}: {absolute}"
        )
    return ArchiveIdentity(
        input_path=str(absolute),
        resolved_path=str(resolved),
        size_bytes=descriptor_info.st_size,
        mode=stat.S_IMODE(descriptor_info.st_mode),
        mtime_ns=descriptor_info.st_mtime_ns,
        device=descriptor_info.st_dev,
        inode=descriptor_info.st_ino,
    )


def build_intent(
    archive: ArchiveIdentity,
    *,
    preflight_sha256: str,
    tool_sha256: str,
    candidate_basenames: Mapping[str, tuple[str, ...]] = CANDIDATE_BASENAMES,
    expected_output_sizes: Mapping[str, int] = EXPECTED_OUTPUT_SIZE_BYTES,
) -> tuple[str, dict[str, Any]]:
    """Build the canonical scan intent and its key."""
    _validate_configuration(candidate_basenames, expected_output_sizes)
    _validate_sha256(preflight_sha256, "preflight_sha256")
    _validate_sha256(tool_sha256, "materializer_sha256")
    candidates = {name: sorted(values) for name, values in sorted(candidate_basenames.items())}
    expected = {name: expected_output_sizes[name] for name in sorted(expected_output_sizes)}
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "archive": archive.intent_json(),
        "candidate_basenames": candidates,
        "expected_output_size_bytes": expected,
        "materializer_sha256": tool_sha256,
        "preflight_sha256": preflight_sha256,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(), payload


def _ensure_plain_directory(path: Path, *, create_mode: int) -> None:
    try:
        path.mkdir(mode=create_mode)
    except FileExistsError:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise MaterializationError(f"publication path is not a plain directory: {path}") from None


def _prepare_layout(root: Path) -> PublicationLayout:
    absolute = root.absolute()
    absolute.mkdir(mode=0o700, parents=True, exist_ok=True)
    root_info = absolute.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise MaterializationError(f"publication root is not a plain directory: {absolute}")
    resolved = absolute.resolve(strict=True)
    children = {name: resolved / name for name in ("objects", "failures", "by-intent", "locks", ".staging")}
    for child in children.values():
        _ensure_plain_directory(child, create_mode=0o700)
    return PublicationLayout(
        root=resolved,
        objects=children["objects"],
        failures=children["failures"],
        by_intent=children["by-intent"],
        locks=children["locks"],
        staging=children[".staging"],
    )


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise MaterializationError(f"lock path cannot be opened safely: {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise MaterializationError(f"lock path is not regular: {path}")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _private_umask() -> Iterator[None]:
    # umask is process-global.  Serializing the context keeps library callers in
    # different threads from restoring each other's previous value.
    with _UMASK_LOCK:
        previous = os.umask(0o077)
        try:
            yield
        finally:
            os.umask(previous)


def _member_type(member: tarfile.TarInfo) -> str:
    if member.isreg():
        return "regular"
    if member.isdir():
        return "directory"
    if member.issym():
        return "symlink"
    if member.islnk():
        return "hardlink"
    if member.ischr():
        return "character-device"
    if member.isblk():
        return "block-device"
    if member.isfifo():
        return "fifo"
    return "unknown"


def _unsafe_candidate_reasons(member: tarfile.TarInfo) -> list[str]:
    reasons: list[str] = []
    name = member.name
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        reasons.append("unsafe_path")
    if "\\" in name or "\x00" in name:
        reasons.append("non_posix_path")
    if not member.isreg():
        reasons.append("not_regular")
    sparse = member.sparse
    if member.type == tarfile.GNUTYPE_SPARSE or sparse:
        reasons.append("sparse_payload")
    risky_pax = sorted(
        key
        for key in member.pax_headers
        if key in {"path", "linkpath", "size"} or key.startswith("GNU.sparse") or key.startswith("SCHILY.realsize")
    )
    reasons.extend(f"pax_override:{key}" for key in risky_pax)
    return reasons


def _inventory_entry(index: int, member: tarfile.TarInfo) -> dict[str, Any]:
    raw_type = member.type.hex() if isinstance(member.type, bytes) else str(member.type)
    sparse_source = cast(list[tuple[int, int]] | None, member.sparse)
    sparse = [[offset, length] for offset, length in sparse_source] if sparse_source else []
    return {
        "index": index,
        "path": member.name,
        "basename": PurePosixPath(member.name).name,
        "type": _member_type(member),
        "raw_type": raw_type,
        "size_bytes": member.size,
        "link_name": member.linkname or None,
        "pax_headers": dict(sorted(member.pax_headers.items())),
        "sparse_map": sparse,
        "candidate_logical_name": None,
        "safe_candidate": None,
        "unsafe_reasons": [],
        "disposition": "ignored",
    }


def _open_output(path: Path) -> int:
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)


def _write_output_chunk(descriptor: int, chunk: bytes) -> None:
    remaining = memoryview(chunk)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("output write made no progress")
        remaining = remaining[written:]


def _extract_candidate(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    logical_name: str,
    output_path: Path,
) -> ExtractionAttempt:
    digest = hashlib.sha256()
    written = 0
    payload_read = 0
    output_error: str | None = None
    archive_error: str | None = None
    descriptor: int | None = None
    try:
        try:
            source = archive.extractfile(member)
            if source is None:
                raise tarfile.ExtractError(f"regular member has no payload stream: {member.name}")
        except Exception as exc:
            return ExtractionAttempt(
                capture=None,
                output_error=None,
                archive_error=f"{type(exc).__name__}: {exc}",
            )
        try:
            try:
                descriptor = _open_output(output_path)
            except Exception as exc:
                output_error = f"{type(exc).__name__}: {exc}"
            while archive_error is None:
                try:
                    chunk = source.read(_HASH_CHUNK_BYTES)
                except Exception as exc:
                    archive_error = f"{type(exc).__name__}: {exc}"
                    break
                if not chunk:
                    break
                payload_read += len(chunk)
                if descriptor is not None and output_error is None:
                    try:
                        _write_output_chunk(descriptor, chunk)
                    except Exception as exc:
                        output_error = f"{type(exc).__name__}: {exc}"
                    else:
                        digest.update(chunk)
                        written += len(chunk)
            if archive_error is None and payload_read != member.size:
                archive_error = f"ReadError: member {member.name} produced {payload_read} bytes, declared {member.size}"
            if descriptor is not None and output_error is None and archive_error is None:
                try:
                    os.fsync(descriptor)
                except Exception as exc:
                    output_error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                source.close()
            except Exception as exc:
                if archive_error is None:
                    archive_error = f"{type(exc).__name__}: {exc}"
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except Exception as exc:
                if output_error is None:
                    output_error = f"{type(exc).__name__}: {exc}"
    capture: OutputCapture | None
    if output_error is None and archive_error is None:
        capture = OutputCapture(
            logical_name=logical_name,
            source_member=member.name,
            source_basename=PurePosixPath(member.name).name,
            declared_size_bytes=member.size,
            size_bytes=written,
            sha256=digest.hexdigest(),
            status="complete",
        )
    else:
        capture = _partial_capture(logical_name, member, output_path)
    return ExtractionAttempt(capture=capture, output_error=output_error, archive_error=archive_error)


def _partial_capture(
    logical_name: str,
    member: tarfile.TarInfo,
    output_path: Path,
) -> OutputCapture | None:
    if not output_path.exists():
        return None
    return OutputCapture(
        logical_name=logical_name,
        source_member=member.name,
        source_basename=PurePosixPath(member.name).name,
        declared_size_bytes=member.size,
        size_bytes=output_path.stat().st_size,
        sha256=_sha256_file(output_path),
        status="partial",
    )


def _error(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _scan_archive(
    archive_path: Path,
    expected_identity: ArchiveIdentity,
    stage: Path,
    candidate_basenames: Mapping[str, tuple[str, ...]],
) -> tuple[list[dict[str, Any]], dict[str, OutputCapture], list[dict[str, str]], dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    captures: dict[str, OutputCapture] = {}
    errors: list[dict[str, str]] = []
    matches: dict[str, list[int]] = {logical: [] for logical in candidate_basenames}
    attempted_logicals: set[str] = set()
    candidate_lookup = {
        basename: logical for logical, basenames in candidate_basenames.items() for basename in basenames
    }
    descriptor = os.open(archive_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    raw: BinaryIO | None = None
    hashing_reader: _HashingReader | None = None
    compressed_bytes_consumed = 0
    archive_sha256 = hashlib.sha256(b"").hexdigest()
    raw_drain_after_gzip_bytes = 0
    gzip_eof_validated = False
    tar_iteration_complete = False
    post_tar_decompressed_bytes = 0
    post_tar_nonzero_bytes = 0
    descriptor_after: os.stat_result | None = None
    try:
        descriptor_before = os.fstat(descriptor)
        if not _stat_matches(expected_identity, descriptor_before):
            raise ArchiveValidationError("archive identity changed before the scan began")
        raw = os.fdopen(descriptor, "rb", buffering=0, closefd=False)
        hashing_reader = _HashingReader(raw)
        compressed = gzip.GzipFile(fileobj=cast(BinaryIO, hashing_reader), mode="rb")
        tar_stream: tarfile.TarFile | None = None
        try:
            try:
                tar_stream = tarfile.open(  # noqa: SIM115 - gzip must be drained before either stream closes
                    fileobj=compressed,
                    mode="r|",
                    bufsize=_TAR_STREAM_BUFFER_BYTES,
                )
                for index, member in enumerate(tar_stream):
                    record = _inventory_entry(index, member)
                    inventory.append(record)
                    logical_name = candidate_lookup.get(record["basename"])
                    if logical_name is None:
                        continue
                    matches[logical_name].append(index)
                    reasons = _unsafe_candidate_reasons(member)
                    record["candidate_logical_name"] = logical_name
                    record["safe_candidate"] = not reasons
                    record["unsafe_reasons"] = reasons
                    if reasons:
                        record["disposition"] = "candidate_rejected"
                        continue
                    if logical_name in attempted_logicals:
                        record["disposition"] = "duplicate_candidate_not_extracted"
                        continue
                    attempted_logicals.add(logical_name)
                    record["disposition"] = "extracting"
                    attempt = _extract_candidate(
                        tar_stream,
                        member,
                        logical_name,
                        stage / logical_name,
                    )
                    if attempt.capture is not None:
                        captures[logical_name] = attempt.capture
                    if attempt.output_error is not None:
                        record["disposition"] = (
                            "partial_extraction" if attempt.capture is not None else "output_creation_failed"
                        )
                        errors.append(
                            _error(
                                "local_output_error",
                                f"{member.name}: {attempt.output_error}",
                            )
                        )
                    if attempt.archive_error is not None:
                        if attempt.output_error is None:
                            record["disposition"] = (
                                "partial_extraction" if attempt.capture is not None else "payload_read_failed"
                            )
                        errors.append(
                            _error(
                                "payload_read_error",
                                f"{member.name}: {attempt.archive_error}",
                            )
                        )
                        raise tarfile.ReadError(attempt.archive_error)
                    if attempt.output_error is None:
                        record["disposition"] = "extracted"
                tar_iteration_complete = True
            except Exception as exc:
                if not errors or errors[-1]["code"] != "payload_read_error":
                    errors.append(_error("archive_read_error", f"{type(exc).__name__}: {exc}"))

            if tar_stream is not None and tar_iteration_complete:
                stream_buffer = getattr(tar_stream.fileobj, "buf", b"")
                if isinstance(stream_buffer, bytes):
                    post_tar_decompressed_bytes += len(stream_buffer)
                    post_tar_nonzero_bytes += len(stream_buffer) - stream_buffer.count(0)
            try:
                while True:
                    chunk = compressed.read(_HASH_CHUNK_BYTES)
                    if not chunk:
                        gzip_eof_validated = True
                        break
                    if tar_iteration_complete:
                        post_tar_decompressed_bytes += len(chunk)
                        post_tar_nonzero_bytes += len(chunk) - chunk.count(0)
            except Exception as exc:
                errors.append(_error("gzip_validation_error", f"{type(exc).__name__}: {exc}"))
        finally:
            if tar_stream is not None:
                tar_stream.close()
            compressed.close()
            try:
                while True:
                    trailing_compressed = hashing_reader.read(_HASH_CHUNK_BYTES)
                    if not trailing_compressed:
                        break
                    raw_drain_after_gzip_bytes += len(trailing_compressed)
            except OSError as exc:
                errors.append(_error("compressed_archive_drain_error", f"{type(exc).__name__}: {exc}"))
            compressed_bytes_consumed = hashing_reader.bytes_read
            archive_sha256 = hashing_reader.hexdigest()
        descriptor_after = os.fstat(descriptor)
    finally:
        if raw is not None:
            raw.close()
        os.close(descriptor)

    exact_compressed_eof = compressed_bytes_consumed == expected_identity.size_bytes
    archive_descriptor_unchanged = descriptor_after is not None and _stat_matches(expected_identity, descriptor_after)
    if not tar_iteration_complete:
        errors.append(_error("tar_scan_incomplete", "tar member iteration did not reach its terminator"))
    if not gzip_eof_validated:
        errors.append(_error("gzip_eof_unvalidated", "gzip EOF and CRC were not successfully validated"))
    if not exact_compressed_eof:
        errors.append(
            _error(
                "compressed_eof_mismatch",
                f"consumed {compressed_bytes_consumed} bytes, archive has {expected_identity.size_bytes}",
            )
        )
    if post_tar_nonzero_bytes:
        errors.append(
            _error(
                "nonzero_trailing_tar_data",
                f"{post_tar_nonzero_bytes} non-zero bytes follow the tar terminator",
            )
        )
    if tar_iteration_complete and post_tar_decompressed_bytes < tarfile.BLOCKSIZE:
        errors.append(
            _error(
                "tar_terminator_incomplete",
                "tar stream does not contain the required second zero terminator block",
            )
        )
    if not archive_descriptor_unchanged:
        errors.append(_error("archive_descriptor_changed", "archive descriptor stat changed during the scan"))

    for logical_name in sorted(candidate_basenames):
        logical_matches = matches[logical_name]
        if not logical_matches:
            errors.append(_error("candidate_missing", f"no candidate found for {logical_name}"))
            continue
        if len(logical_matches) != 1:
            errors.append(
                _error(
                    "candidate_count_mismatch",
                    f"{logical_name} has {len(logical_matches)} candidates; exactly one is required",
                )
            )
            continue
        record = inventory[logical_matches[0]]
        if record["safe_candidate"] is not True:
            joined_reasons = ",".join(record["unsafe_reasons"])
            errors.append(_error("unsafe_candidate", f"{logical_name}: {joined_reasons}"))
            continue
        capture = captures.get(logical_name)
        if capture is None or capture.status != "complete":
            errors.append(_error("candidate_incomplete", f"{logical_name} did not materialize completely"))

    scan = {
        "tar_iteration_complete": tar_iteration_complete,
        "inventory_member_count": len(inventory),
        "gzip_eof_validated": gzip_eof_validated,
        "compressed_bytes_consumed": compressed_bytes_consumed,
        "exact_compressed_eof": exact_compressed_eof,
        "archive_sha256": archive_sha256,
        "archive_sha256_complete": exact_compressed_eof,
        "raw_drain_after_gzip_bytes": raw_drain_after_gzip_bytes,
        "post_tar_decompressed_bytes": post_tar_decompressed_bytes,
        "post_tar_nonzero_bytes": post_tar_nonzero_bytes,
        "tar_zero_termination_validated": (
            tar_iteration_complete and post_tar_decompressed_bytes >= tarfile.BLOCKSIZE and post_tar_nonzero_bytes == 0
        ),
        "archive_descriptor_unchanged": archive_descriptor_unchanged,
    }
    return inventory, captures, errors, scan


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_file(path: Path, payload: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        _write_output_chunk(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _archive_is_unchanged(path: Path, expected: ArchiveIdentity) -> tuple[bool, dict[str, Any] | None]:
    try:
        observed = _capture_archive_identity(path, None)
    except ArchiveValidationError:
        return False, None
    return observed == expected, observed.as_json()


def _freeze_stage(stage: Path, captures: Mapping[str, OutputCapture], manifest_bytes: bytes) -> str:
    for logical_name in captures:
        output = stage / logical_name
        if output.exists():
            descriptor = os.open(output, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
            finally:
                os.close(descriptor)
    manifest_path = stage / "manifest.json"
    _write_new_file(manifest_path, manifest_bytes, 0o400)
    _fsync_directory(stage)
    return hashlib.sha256(manifest_bytes).hexdigest()


def _remove_owned_stage(stage: Path) -> None:
    os.chmod(stage, 0o700, follow_symlinks=False)
    for child in stage.iterdir():
        info = child.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise MaterializationError(f"unexpected entry in owned staging directory: {child}")
        os.chmod(child, 0o600, follow_symlinks=False)
        child.unlink()
    stage.rmdir()


def _path_exists_no_follow(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _require_mode(path: Path, expected_mode: int, *, directory: bool) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReuseValidationError(f"published path cannot be inspected: {path}: {exc}") from exc
    expected_type = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if stat.S_ISLNK(info.st_mode) or not expected_type:
        raise ReuseValidationError(f"published path has an unsafe type: {path}")
    observed_mode = stat.S_IMODE(info.st_mode)
    if observed_mode != expected_mode:
        raise ReuseValidationError(f"published path mode drifted: {path}: {oct(observed_mode)} != {oct(expected_mode)}")
    return info


def _validate_published_artifact(
    artifact_root: Path,
    *,
    manifest_sha256: str,
    intent_key: str,
    outcome: str,
) -> dict[str, Any]:
    _require_mode(artifact_root, 0o500, directory=True)
    manifest_path = artifact_root / "manifest.json"
    _require_mode(manifest_path, 0o400, directory=False)
    observed_manifest_sha256 = _sha256_file(manifest_path)
    if observed_manifest_sha256 != manifest_sha256:
        raise ReuseValidationError("published manifest hash mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReuseValidationError(f"published manifest is unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ReuseValidationError("published manifest must be an object")
    expected_manifest_keys = {
        "schema_version",
        "kind",
        "outcome",
        "intent_key",
        "intent",
        "execution_context",
        "archive_before",
        "archive_after",
        "archive_stat_unchanged",
        "archive_sha256",
        "scan",
        "inventory",
        "outputs",
        "size_findings",
        "errors",
    }
    if set(manifest) != expected_manifest_keys:
        raise ReuseValidationError("published manifest has an unexpected shape")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("kind") != MANIFEST_KIND:
        raise ReuseValidationError("published manifest contract mismatch")
    if manifest.get("intent_key") != intent_key or manifest.get("outcome") != outcome:
        raise ReuseValidationError("published manifest does not match its pointer")
    archive_sha256 = manifest.get("archive_sha256")
    scan = manifest.get("scan")
    if not isinstance(archive_sha256, str) or not isinstance(scan, dict):
        raise ReuseValidationError("published archive digest evidence is missing")
    try:
        _validate_sha256(archive_sha256, "archive_sha256")
    except MaterializationError as exc:
        raise ReuseValidationError(str(exc)) from exc
    if scan.get("archive_sha256") != archive_sha256:
        raise ReuseValidationError("published archive digest evidence is inconsistent")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ReuseValidationError("published outputs must be an object")
    if outcome == "success" and set(outputs) != set(CANDIDATE_BASENAMES):
        raise ReuseValidationError("successful publication does not contain both logical outputs")
    for logical_name, value in outputs.items():
        if logical_name not in CANDIDATE_BASENAMES or not isinstance(value, dict):
            raise ReuseValidationError("published output entry is unexpected")
        if value.get("file_name") != logical_name:
            raise ReuseValidationError(f"published output name mismatch: {logical_name}")
        output_path = artifact_root / logical_name
        info = _require_mode(output_path, 0o400, directory=False)
        if info.st_size != value.get("size_bytes") or _sha256_file(output_path) != value.get("sha256"):
            raise ReuseValidationError(f"published output hash or size mismatch: {logical_name}")
        if outcome == "success" and value.get("status") != "complete":
            raise ReuseValidationError(f"successful output is not complete: {logical_name}")
    return manifest


def _publish_artifact(
    layout: PublicationLayout,
    stage: Path,
    *,
    manifest_sha256: str,
    intent_key: str,
    outcome: str,
) -> Path:
    parent = layout.objects if outcome == "success" else layout.failures
    destination = parent / manifest_sha256
    with _exclusive_lock(layout.locks / "publication.lock"):
        if _path_exists_no_follow(destination):
            _validate_published_artifact(
                destination,
                manifest_sha256=manifest_sha256,
                intent_key=intent_key,
                outcome=outcome,
            )
            _remove_owned_stage(stage)
        else:
            os.rename(stage, destination)
            os.chmod(destination, 0o500, follow_symlinks=False)
            _fsync_directory(destination)
            _fsync_directory(parent)
    return destination


def _pointer_payload(intent_key: str, outcome: str, manifest_sha256: str) -> dict[str, Any]:
    category = "objects" if outcome == "success" else "failures"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": POINTER_KIND,
        "intent_key": intent_key,
        "outcome": outcome,
        "artifact_key": manifest_sha256,
        "artifact_relative_path": f"{category}/{manifest_sha256}",
        "manifest_relative_path": f"{category}/{manifest_sha256}/manifest.json",
        "manifest_sha256": manifest_sha256,
    }


def _publish_pointer(layout: PublicationLayout, intent_key: str, payload: dict[str, Any]) -> Path:
    final = layout.by_intent / intent_key
    temporary = Path(tempfile.mkdtemp(prefix=f".{intent_key}.", dir=layout.by_intent))
    try:
        _write_new_file(temporary / "result.json", _json_bytes(payload), 0o400)
        _fsync_directory(temporary)
        if _path_exists_no_follow(final):
            existing = _load_pointer(final / "result.json", intent_key)
            if existing != payload:
                raise ReuseValidationError("an existing intent pointer conflicts with the completed result")
            _remove_owned_stage(temporary)
        else:
            os.rename(temporary, final)
            os.chmod(final, 0o500, follow_symlinks=False)
            _fsync_directory(final)
            _fsync_directory(layout.by_intent)
    except Exception:
        if _path_exists_no_follow(temporary):
            _remove_owned_stage(temporary)
        raise
    return final / "result.json"


def _load_pointer(path: Path, intent_key: str) -> dict[str, Any]:
    _require_mode(path.parent, 0o500, directory=True)
    _require_mode(path, 0o400, directory=False)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReuseValidationError(f"intent pointer is unreadable: {exc}") from exc
    expected_keys = {
        "schema_version",
        "kind",
        "intent_key",
        "outcome",
        "artifact_key",
        "artifact_relative_path",
        "manifest_relative_path",
        "manifest_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ReuseValidationError("intent pointer has an unexpected shape")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != POINTER_KIND
        or payload.get("intent_key") != intent_key
        or payload.get("outcome") not in {"success", "failure"}
    ):
        raise ReuseValidationError("intent pointer contract mismatch")
    artifact_key = payload.get("artifact_key")
    manifest_sha256 = payload.get("manifest_sha256")
    if not isinstance(artifact_key, str) or artifact_key != manifest_sha256:
        raise ReuseValidationError("intent pointer artifact key mismatch")
    return payload


def _reuse_result(
    layout: PublicationLayout,
    pointer_path: Path,
    *,
    archive_path: Path,
    archive_identity: ArchiveIdentity,
    intent_key: str,
    intent: dict[str, Any],
    execution_context: dict[str, Any],
) -> dict[str, Any]:
    pointer = _load_pointer(pointer_path, intent_key)
    current = _capture_archive_identity(archive_path, archive_identity.size_bytes)
    if current != archive_identity:
        raise ReuseValidationError("archive stat no longer matches the completed intent")
    outcome = pointer["outcome"]
    category = "objects" if outcome == "success" else "failures"
    artifact_key = pointer["artifact_key"]
    expected_artifact_relative = f"{category}/{artifact_key}"
    if pointer["artifact_relative_path"] != expected_artifact_relative:
        raise ReuseValidationError("intent pointer artifact path mismatch")
    if pointer["manifest_relative_path"] != f"{expected_artifact_relative}/manifest.json":
        raise ReuseValidationError("intent pointer manifest path mismatch")
    artifact_root = layout.root / category / artifact_key
    manifest = _validate_published_artifact(
        artifact_root,
        manifest_sha256=pointer["manifest_sha256"],
        intent_key=intent_key,
        outcome=outcome,
    )
    if manifest.get("intent") != intent or manifest.get("archive_before") != archive_identity.as_json():
        raise ReuseValidationError("published manifest intent or archive identity mismatch")
    _validate_execution_context(manifest.get("execution_context"), intent=intent, current=execution_context)
    return {
        **pointer,
        "artifact_root": str(artifact_root),
        "manifest_path": str(artifact_root / "manifest.json"),
        "reused": True,
    }


def materialize_archive(
    archive_path: Path,
    publication_root: Path,
    *,
    preflight_sha256: str,
    expected_archive_size_bytes: int | None = EXPECTED_ARCHIVE_SIZE_BYTES,
    expected_output_sizes: Mapping[str, int] = EXPECTED_OUTPUT_SIZE_BYTES,
    candidate_basenames: Mapping[str, tuple[str, ...]] = CANDIDATE_BASENAMES,
    local_test_context: bool = False,
) -> dict[str, Any]:
    """Materialize or validate/reuse one immutable result for ``archive_path``."""
    _validate_configuration(candidate_basenames, expected_output_sizes)
    archive_identity = _capture_archive_identity(archive_path, expected_archive_size_bytes)
    tool_sha256 = materializer_sha256()
    execution_context = _build_execution_context(
        tool_sha256=tool_sha256,
        local_test_context=local_test_context,
    )
    intent_key, intent = build_intent(
        archive_identity,
        preflight_sha256=preflight_sha256,
        tool_sha256=tool_sha256,
        candidate_basenames=candidate_basenames,
        expected_output_sizes=expected_output_sizes,
    )
    with _private_umask():
        layout = _prepare_layout(publication_root)
        with _exclusive_lock(layout.locks / f"{intent_key}.lock"):
            intent_root = layout.by_intent / intent_key
            pointer_path = intent_root / "result.json"
            if _path_exists_no_follow(intent_root):
                if not _path_exists_no_follow(pointer_path):
                    raise ReuseValidationError("intent directory exists without a completed result pointer")
                return _reuse_result(
                    layout,
                    pointer_path,
                    archive_path=archive_path,
                    archive_identity=archive_identity,
                    intent_key=intent_key,
                    intent=intent,
                    execution_context=execution_context,
                )

            current = _capture_archive_identity(archive_path, expected_archive_size_bytes)
            if current != archive_identity:
                raise ArchiveValidationError("archive identity changed while waiting for the intent lock")

            stage = Path(tempfile.mkdtemp(prefix=f"{intent_key}.", dir=layout.staging))
            try:
                inventory, captures, errors, scan = _scan_archive(
                    archive_path,
                    archive_identity,
                    stage,
                    candidate_basenames,
                )
                unchanged, archive_after = _archive_is_unchanged(archive_path, archive_identity)
                if not unchanged:
                    errors.append(_error("archive_path_changed", "archive path stat changed during the scan"))
                output_payload = {
                    name: capture.as_json(expected_output_sizes[name]) for name, capture in sorted(captures.items())
                }
                size_findings = {
                    name: {
                        "expected_size_bytes": expected_output_sizes[name],
                        "observed_size_bytes": output_payload[name]["size_bytes"],
                        "matches_expected": output_payload[name]["size_matches_expected"],
                    }
                    for name in output_payload
                }
                for name, finding in sorted(size_findings.items()):
                    if finding["matches_expected"] is not True:
                        errors.append(
                            _error(
                                "expected_size_mismatch",
                                (
                                    f"{name}: observed {finding['observed_size_bytes']} bytes, "
                                    f"expected {finding['expected_size_bytes']}"
                                ),
                            )
                        )
                outcome = "success" if not errors else "failure"
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "kind": MANIFEST_KIND,
                    "outcome": outcome,
                    "intent_key": intent_key,
                    "intent": intent,
                    "execution_context": execution_context,
                    "archive_before": archive_identity.as_json(),
                    "archive_after": archive_after,
                    "archive_stat_unchanged": unchanged,
                    "archive_sha256": scan["archive_sha256"],
                    "scan": scan,
                    "inventory": inventory,
                    "outputs": output_payload,
                    "size_findings": size_findings,
                    "errors": errors,
                }
                manifest_bytes = _json_bytes(manifest)
                manifest_sha256 = _freeze_stage(stage, captures, manifest_bytes)
                artifact_root = _publish_artifact(
                    layout,
                    stage,
                    manifest_sha256=manifest_sha256,
                    intent_key=intent_key,
                    outcome=outcome,
                )
                pointer = _pointer_payload(intent_key, outcome, manifest_sha256)
                pointer_path = _publish_pointer(layout, intent_key, pointer)
                return {
                    **pointer,
                    "artifact_root": str(artifact_root),
                    "manifest_path": str(artifact_root / "manifest.json"),
                    "pointer_path": str(pointer_path),
                    "reused": False,
                }
            except Exception:
                if _path_exists_no_follow(stage):
                    _remove_owned_stage(stage)
                raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--publication-root", type=Path, required=True)
    preflight = parser.add_mutually_exclusive_group(required=True)
    preflight.add_argument("--preflight-manifest", type=Path)
    preflight.add_argument("--preflight-sha256")
    parser.add_argument("--expected-archive-size", type=int, default=EXPECTED_ARCHIVE_SIZE_BYTES)
    parser.add_argument("--expected-mapping-size", type=int, default=EXPECTED_OUTPUT_SIZE_BYTES[LOGICAL_MAPPING])
    parser.add_argument("--expected-taxonomy-size", type=int, default=EXPECTED_OUTPUT_SIZE_BYTES[LOGICAL_TAXONOMY])
    parser.add_argument(
        "--local-test-context",
        action="store_true",
        help="record an explicit non-Slurm test context instead of requiring Slurm evidence",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.preflight_manifest is not None:
        preflight_sha256 = _sha256_file(args.preflight_manifest)
    else:
        preflight_sha256 = args.preflight_sha256
    try:
        result = materialize_archive(
            args.archive,
            args.publication_root,
            preflight_sha256=preflight_sha256,
            expected_archive_size_bytes=args.expected_archive_size,
            expected_output_sizes={
                LOGICAL_MAPPING: args.expected_mapping_size,
                LOGICAL_TAXONOMY: args.expected_taxonomy_size,
            },
            local_test_context=args.local_test_context,
        )
    except MaterializationError as exc:
        print(json.dumps({"outcome": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
