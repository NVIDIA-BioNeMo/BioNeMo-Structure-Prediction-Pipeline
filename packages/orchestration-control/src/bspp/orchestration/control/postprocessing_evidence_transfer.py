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

"""Bounded remote-to-local transfer for postprocessing finalization metadata."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from bspp.orchestration.contract.postprocessing_action09_bundle import (
    PostprocessingFinalizationHandoffIndex,
    postprocessing_handoff_index_from_mapping,
)
from bspp.orchestration.contract.postprocessing_bundle_manifest import (
    postprocessing_tar_manifest_from_mapping,
)
from bspp.orchestration.contract.postprocessing_transfer_limits import (
    POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1,
)
from bspp.orchestration.control.postprocessing_authority_reader import require_postprocessing_v2_authority
from bspp.orchestration.control.postprocessing_phase_lifecycle import postprocessing_transport
from bspp.orchestration.control.postprocessing_phase_types import PostprocessingAuthority
from bspp.orchestration.control.transport import CommandRunner, default_command_runner


@dataclass(frozen=True)
class PostprocessingEvidenceFetchResult:
    phase_run_id: str
    attempt_id: str
    destination: Path
    indexed_file_count: int
    aggregate_bytes: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase_kind": "postprocessing",
            "operation": "evidence-fetch",
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "destination": str(self.destination),
            "indexed_file_count": self.indexed_file_count,
            "aggregate_bytes": self.aggregate_bytes,
            "status": "fetched",
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class PostprocessingHandoffAuthorityBinding:
    """The exact current-Attempt authority needed to fetch a Runtime handoff."""

    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_graph_digest: str
    execution_projection_sha256: str
    acceptance_policy_sha256: str
    evidence_dir: str

    @classmethod
    def from_authority(cls, authority: PostprocessingAuthority) -> PostprocessingHandoffAuthorityBinding:
        return cls(
            phase_run_id=authority.phase_run_id,
            attempt_id=authority.attempt_id,
            phase_runspec_digest=authority.runspec.digest,
            action_graph_digest=authority.runspec.payload.action_graph_digest,
            execution_projection_sha256=authority.runspec.payload.execution_projection.document_sha256,
            acceptance_policy_sha256=authority.runspec.payload.acceptance_policy.sha256,
            evidence_dir=authority.runspec.payload.attempt_paths.evidence_dir,
        )


class PostprocessingEvidenceSource(Protocol):
    def fetch_stable_artifact(self, remote_path: Path, *, remote_root: Path, maximum_bytes: int) -> bytes: ...


@dataclass(frozen=True)
class PostprocessingEvidenceFetchContext:
    """Typed test/operator seam after current authority has been established."""

    authority: PostprocessingHandoffAuthorityBinding
    source: PostprocessingEvidenceSource


@dataclass(frozen=True)
class PostprocessingEvidencePublicationStore:
    publish_directory_no_replace: Callable[[Path, Path], None]


CURRENT_POSTPROCESSING_EVIDENCE_PUBLICATION_STORE = PostprocessingEvidencePublicationStore(
    publish_directory_no_replace=lambda source, destination: _rename_directory_no_replace(source, destination)
)


def fetch_postprocessing_finalization_evidence(
    phase_run_id: str,
    *,
    authority_root: Path,
    destination: Path,
    runner: CommandRunner = default_command_runner,
    context: PostprocessingEvidenceFetchContext | None = None,
    publication_store: PostprocessingEvidencePublicationStore = CURRENT_POSTPROCESSING_EVIDENCE_PUBLICATION_STORE,
) -> PostprocessingEvidenceFetchResult:
    """Fetch only the bounded, indexed finalization bundle and publish it atomically."""
    source: PostprocessingEvidenceSource
    if context is None:
        full_authority = require_postprocessing_v2_authority(authority_root, phase_run_id)
        authority = PostprocessingHandoffAuthorityBinding.from_authority(full_authority)
        source = postprocessing_transport(full_authority, runner=runner)
    else:
        authority = context.authority
        source = context.source
        if authority.phase_run_id != phase_run_id:
            raise ValueError("postprocessing evidence context differs from the requested Phase Run")
    if destination.exists() or os.path.lexists(destination):
        index = validate_local_postprocessing_handoff(destination, authority=authority)
        return _result(index, destination)
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("postprocessing evidence destination parent must be an existing non-symlink directory")
    remote_root = Path(authority.evidence_dir) / "phase-finalization"
    index_bytes = source.fetch_stable_artifact(
        remote_root / "handoff-index.json",
        remote_root=remote_root,
        maximum_bytes=POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1.max_file_bytes,
    )
    index = _validated_index_bytes(index_bytes, authority=authority)
    # The strict index and all aggregate limits are validated before this first
    # descendant fetch. Remote locators embedded in any member are never read.
    fetched: dict[str, bytes] = {}
    for member in index.members:
        document = source.fetch_stable_artifact(
            remote_root / Path(*member.path.split("/")),
            remote_root=remote_root,
            maximum_bytes=member.size_bytes,
        )
        if len(document) != member.size_bytes or hashlib.sha256(document).hexdigest() != member.sha256:
            raise ValueError(f"fetched postprocessing evidence differs from its index: {member.path!r}")
        _canonical_mapping(document, label=f"postprocessing evidence {member.path!r}")
        fetched[member.path] = document

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.fetch-", dir=parent))
    os.chmod(temporary, 0o700)
    try:
        _write_new(temporary / "handoff-index.json", index_bytes)
        for relative, document in fetched.items():
            _write_new(temporary / Path(*relative.split("/")), document)
        _fsync_tree(temporary)
        validate_local_postprocessing_handoff(temporary, authority=authority)
        try:
            publication_store.publish_directory_no_replace(temporary, destination)
        except FileExistsError:
            validate_local_postprocessing_handoff(destination, authority=authority)
            shutil.rmtree(temporary)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _result(index, destination)


def validate_local_postprocessing_handoff(
    handoff: Path,
    *,
    authority: PostprocessingAuthority | PostprocessingHandoffAuthorityBinding | None = None,
) -> PostprocessingFinalizationHandoffIndex:
    """Validate an exact local JSON-only handoff without following its locators."""
    if handoff.is_symlink() or not handoff.is_dir():
        raise ValueError("postprocessing handoff must be a local non-symlink directory")
    observed_files, observed_directories = _local_tree(handoff)
    index_path = handoff / "handoff-index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("postprocessing handoff index is missing or unsafe")
    index_bytes = index_path.read_bytes()
    index = _validated_index_bytes(index_bytes, authority=authority)
    expected_files = {"handoff-index.json", *(member.path for member in index.members)}
    expected_directories = {parent for path in expected_files for parent in _relative_parents(path)}
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("postprocessing handoff local layout has missing or extra entries")
    by_path = {item.path: item for item in index.members}
    actual_tar_counts: list[tuple[str, int]] = []
    aggregate = 0
    for relative in sorted(by_path):
        member = by_path[relative]
        path = handoff / Path(*relative.split("/"))
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"postprocessing handoff member is unsafe: {relative!r}")
        document = path.read_bytes()
        aggregate += len(document)
        if len(document) != member.size_bytes or hashlib.sha256(document).hexdigest() != member.sha256:
            raise ValueError(f"postprocessing handoff member differs from its index: {relative!r}")
        payload = _canonical_mapping(document, label=f"postprocessing handoff member {relative!r}")
        if relative.startswith("outputs/tar-manifests/"):
            manifest = postprocessing_tar_manifest_from_mapping(payload)
            if relative != f"outputs/tar-manifests/{manifest.manifest_id}.json":
                raise ValueError("postprocessing tar manifest path differs from its content identity")
            actual_tar_counts.append((relative, len(manifest.members)))
    if aggregate != index.declared_aggregate_bytes or tuple(actual_tar_counts) != index.tar_manifest_member_counts:
        raise ValueError("postprocessing handoff actual aggregate or tar cardinality differs from its index")
    return index


def _validated_index_bytes(
    document: bytes,
    *,
    authority: PostprocessingAuthority | PostprocessingHandoffAuthorityBinding | None,
) -> PostprocessingFinalizationHandoffIndex:
    payload = _canonical_mapping(document, label="postprocessing handoff index")
    index = postprocessing_handoff_index_from_mapping(payload)
    binding = (
        PostprocessingHandoffAuthorityBinding.from_authority(authority)
        if isinstance(authority, PostprocessingAuthority)
        else authority
    )
    if binding is not None and (
        index.phase_run_id != binding.phase_run_id
        or index.attempt_id != binding.attempt_id
        or index.phase_runspec_digest != binding.phase_runspec_digest
        or index.action_graph_digest != binding.action_graph_digest
        or index.execution_projection_sha256 != binding.execution_projection_sha256
        or index.acceptance_policy_sha256 != binding.acceptance_policy_sha256
    ):
        raise ValueError("postprocessing handoff index differs from current Attempt authority")
    return index


def _local_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in names:
            path = current / name
            if path.is_symlink() or not path.is_dir():
                raise ValueError("postprocessing handoff contains an unsafe directory entry")
            directories.add(path.relative_to(root).as_posix())
        for name in filenames:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("postprocessing handoff contains an unsafe file entry")
            files.add(path.relative_to(root).as_posix())
        if len(files) > POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1.max_indexed_files + 1:
            raise ValueError("postprocessing handoff contains too many files")
    return files, directories


def _relative_parents(path: str) -> set[str]:
    parts = path.split("/")[:-1]
    return {"/".join(parts[:index]) for index in range(1, len(parts) + 1)}


def _canonical_mapping(document: bytes, *, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(document)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON mapping")
    canonical = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
    if document != canonical:
        raise ValueError(f"{label} must use canonical JSON bytes")
    return cast("Mapping[str, object]", payload)


def _write_new(path: Path, document: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _fsync_tree(root: Path) -> None:
    directories = [Path(directory) for directory, _, _ in os.walk(root, topdown=False, followlinks=False)]
    for directory in directories:
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


_RenameAt2 = Callable[[int, bytes, int, bytes, int], None]
_RENAME_NOREPLACE = 1
_RENAMEAT2_FALLBACK_ERRNOS = frozenset(
    {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)}
)


def _native_renameat2(
    source_fd: int,
    source_name: bytes,
    destination_fd: int,
    destination_name: bytes,
    flags: int,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS))
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(source_fd, source_name, destination_fd, destination_name, flags)
    if result == 0:
        return
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error), os.fsdecode(destination_name))


def _publication_paths(source: Path, destination: Path) -> tuple[Path, str, str]:
    if not source.is_absolute() or not destination.is_absolute():
        raise ValueError("postprocessing evidence publication paths must be absolute")
    if source.parent != destination.parent or source.name == destination.name:
        raise ValueError("postprocessing evidence publication requires distinct names in one parent")
    if not source.name or not destination.name or "/" in source.name or "/" in destination.name:
        raise ValueError("postprocessing evidence publication requires safe nonempty basenames")
    parent = source.parent
    if parent.resolve(strict=True) != parent or source.resolve(strict=True) != source:
        raise ValueError("postprocessing evidence publication paths must be canonical and symlink-free")
    return parent, source.name, destination.name


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _check_publication_state(parent: Path, parent_fd: int, source_name: str, destination_name: str) -> None:
    if not _same_identity(os.lstat(parent), os.fstat(parent_fd)):
        raise ValueError("postprocessing evidence publication parent identity changed")
    source_info = os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(source_info.st_mode) or source_info.st_uid != os.geteuid():
        raise ValueError("postprocessing evidence publication source must be an owned directory")
    try:
        os.stat(destination_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination_name)


def _rename_directory_no_replace(
    source: Path,
    destination: Path,
    *,
    renameat2: _RenameAt2 | None = None,
) -> None:
    """Publish a complete same-parent directory for cooperating BSPP writers."""
    parent, source_name, destination_name = _publication_paths(source, destination)
    before = os.lstat(parent)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not _same_identity(before, os.fstat(parent_fd)):
            raise ValueError("postprocessing evidence publication parent identity changed")
        _check_publication_state(parent, parent_fd, source_name, destination_name)
        try:
            (renameat2 or _native_renameat2)(
                parent_fd,
                os.fsencode(source_name),
                parent_fd,
                os.fsencode(destination_name),
                _RENAME_NOREPLACE,
            )
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination) from exc
            if exc.errno not in _RENAMEAT2_FALLBACK_ERRNOS:
                raise
        else:
            os.fsync(parent_fd)
            return
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        try:
            _check_publication_state(parent, parent_fd, source_name, destination_name)
            os.rename(source_name, destination_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
    finally:
        os.close(parent_fd)


def _result(index: PostprocessingFinalizationHandoffIndex, destination: Path) -> PostprocessingEvidenceFetchResult:
    return PostprocessingEvidenceFetchResult(
        phase_run_id=index.phase_run_id,
        attempt_id=index.attempt_id,
        destination=destination,
        indexed_file_count=index.declared_file_count,
        aggregate_bytes=index.declared_aggregate_bytes,
    )


__all__ = [
    "CURRENT_POSTPROCESSING_EVIDENCE_PUBLICATION_STORE",
    "PostprocessingEvidenceFetchContext",
    "PostprocessingEvidenceFetchResult",
    "PostprocessingEvidencePublicationStore",
    "PostprocessingEvidenceSource",
    "PostprocessingHandoffAuthorityBinding",
    "fetch_postprocessing_finalization_evidence",
    "validate_local_postprocessing_handoff",
]
