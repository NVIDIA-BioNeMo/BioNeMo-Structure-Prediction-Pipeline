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

"""Runtime-only scientific roots and atomic postprocessing finalization bundles.

Large scientific tar payloads never enter the bounded handoff.  Runtime streams
their regular-file members into content manifests, hashes ordinary small output
files, verifies all prior Runtime and acceptance evidence, and atomically
publishes the JSON-only Action 09 handoff.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from bspp.orchestration.contract.postprocessing_acceptance_capture import PostprocessingArtifactBinding
from bspp.orchestration.contract.postprocessing_action09_bundle import (
    PostprocessingBundleMemberIdentity,
    postprocessing_action09_assembly_witness_from_mapping,
    postprocessing_handoff_index_from_mapping,
)
from bspp.orchestration.contract.postprocessing_bundle_manifest import (
    postprocessing_tar_manifest_from_mapping,
)
from bspp.orchestration.contract.postprocessing_phase_ids import POSTPROCESSING_ACTION_IDS
from bspp.orchestration.contract.postprocessing_runspec import (
    ExecutablePostprocessingPhaseRunSpec,
    postprocessing_phase_runspec_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runtime_evidence import (
    PostprocessingCompletedRuntimeActionEvidence,
)
from bspp.orchestration.contract.postprocessing_transfer_limits import (
    POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TAR_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".tar.zst",
    ".tzst",
)
_ACCEPTANCE_STEPS = (
    "acceptance-tar-payload-parity",
    "acceptance-semantic",
    "acceptance-verify-evidence",
)
_CAPTURE_DESTINATIONS = {step: f"acceptance/captures/{step}.json" for step in _ACCEPTANCE_STEPS}
_REPORT_DESTINATIONS = {
    "acceptance-tar-payload-parity": "acceptance/reports/tar-payload-parity-report.json",
    "acceptance-semantic": "acceptance/reports/semantic-acceptance-summary.json",
    "acceptance-verify-evidence": "acceptance/reports/acceptance-evidence-report.json",
}
_ACTION09_ID = POSTPROCESSING_ACTION_IDS["acceptance-adjudication"]


@dataclass(frozen=True)
class _StableFile:
    path: Path
    device: int
    inode: int
    mode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class FinalizationPublicationStore:
    """Typed atomic-publication seam used for deterministic fault injection."""

    rename_directory_no_replace: Callable[[Path, Path], None]


def _publish_exact_directory(
    destination: Path,
    documents: Mapping[str, bytes],
    *,
    publication_store: FinalizationPublicationStore | None = None,
) -> None:
    if destination.exists() or os.path.lexists(destination):
        _verify_exact_directory(destination, documents)
        return
    parent = _absolute_directory(destination.parent, "finalization parent", create=False)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.assembly-", dir=parent))
    os.chmod(temporary, 0o700)
    try:
        for relative, document in sorted(documents.items()):
            _write_new(temporary / Path(*relative.split("/")), document)
        _fsync_tree(temporary)
        _verify_exact_directory(temporary, documents)
        try:
            (publication_store or LOCAL_FINALIZATION_PUBLICATION_STORE).rename_directory_no_replace(
                temporary, destination
            )
        except FileExistsError:
            _verify_exact_directory(destination, documents)
            shutil.rmtree(temporary)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verify_exact_directory(root: Path, documents: Mapping[str, bytes]) -> None:
    observed_files, observed_directories = _strict_tree(root)
    expected_directories = {parent for relative in documents for parent in _relative_parents(relative)}
    if observed_files != set(documents) or observed_directories != expected_directories:
        raise ValueError("existing postprocessing finalization bundle layout differs")
    for relative, expected in documents.items():
        observed = _stable_file_bytes(root / Path(*relative.split("/")))
        _canonical_mapping(observed, label=f"postprocessing finalization member {relative}")
        if observed != expected:
            raise ValueError(f"existing postprocessing finalization bundle differs: {relative!r}")
    index = postprocessing_handoff_index_from_mapping(
        _canonical_mapping(documents["handoff-index.json"], label="postprocessing handoff index")
    )
    for relative, count in index.tar_manifest_member_counts:
        manifest = postprocessing_tar_manifest_from_mapping(
            _canonical_mapping(documents[relative], label=f"postprocessing tar manifest {relative}")
        )
        if len(manifest.members) != count:
            raise ValueError("postprocessing tar manifest count differs from the handoff index")


def _existing_assembled_at(destination: Path) -> str | None:
    if not (destination.exists() or os.path.lexists(destination)):
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("existing postprocessing finalization bundle is not a safe directory")
    payload = _canonical_mapping(
        _stable_file_bytes(destination / "acceptance/bundle.json"),
        label="existing Action 09 assembly witness",
    )
    return postprocessing_action09_assembly_witness_from_mapping(payload).assembled_at


def _deterministic_assembled_at(
    completed_actions: tuple[PostprocessingCompletedRuntimeActionEvidence, ...],
) -> str:
    instants = tuple(task.completed_at for action in completed_actions for task in action.tasks)
    if not instants:
        raise ValueError("postprocessing Action 09 requires prior completed Runtime task timestamps")
    return max(instants)


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
        raise ValueError("postprocessing publication paths must be absolute")
    if source.parent != destination.parent or source.name == destination.name:
        raise ValueError("postprocessing publication requires distinct names in one parent")
    if not source.name or not destination.name or "/" in source.name or "/" in destination.name:
        raise ValueError("postprocessing publication requires safe nonempty basenames")
    parent = source.parent
    if parent.resolve(strict=True) != parent or source.resolve(strict=True) != source:
        raise ValueError("postprocessing publication paths must be canonical and symlink-free")
    return parent, source.name, destination.name


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _check_publication_state(parent: Path, parent_fd: int, source_name: str, destination_name: str) -> None:
    if not _same_identity(os.lstat(parent), os.fstat(parent_fd)):
        raise ValueError("postprocessing publication parent identity changed")
    source_info = os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(source_info.st_mode) or source_info.st_uid != os.geteuid():
        raise ValueError("postprocessing publication source must be an owned directory")
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
            raise ValueError("postprocessing publication parent identity changed")
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


LOCAL_FINALIZATION_PUBLICATION_STORE = FinalizationPublicationStore(
    rename_directory_no_replace=_rename_directory_no_replace
)


def _load_runspec(path: Path) -> ExecutablePostprocessingPhaseRunSpec:
    document = _stable_file_bytes(path)
    payload = _canonical_mapping(document, label="postprocessing Phase RunSpec")
    return postprocessing_phase_runspec_from_mapping(payload)


def _stable_file_bytes(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"authority path is not a regular file: {path}")
            document = handle.read()
            after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _assert_stable_stat(path, before, after)
    return document


def _authority_relative_document(root: Path, relative: str, *, label: str) -> bytes:
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"{label} path is not authority-relative")
    candidate = root / Path(*relative.split("/"))
    resolved = candidate.resolve(strict=True)
    if candidate != resolved or root not in resolved.parents:
        raise ValueError(f"{label} path escapes authority or traverses a link")
    return _stable_file_bytes(resolved)


def _verify_bound_artifact(root: Path, binding: PostprocessingArtifactBinding, *, label: str) -> None:
    document = _authority_relative_document(root, binding.path, label=label)
    if len(document) != binding.size_bytes or hashlib.sha256(document).hexdigest() != binding.sha256:
        raise ValueError(f"captured acceptance {label} changed")


def _regular_nonsymlink_file(path: Path, *, label: str) -> None:
    try:
        details = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        raise ValueError(f"{label} is missing: {path}") from None
    if path.is_symlink() or not stat.S_ISREG(details.st_mode):
        raise ValueError(f"{label} is not a regular non-symlink file: {path}")


def _assert_stable_stat(path: Path, before: os.stat_result, after: os.stat_result) -> None:
    observed = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or not (_same_stat(before, after) and _same_stat(before, observed))
    ):
        raise ValueError(f"file changed while it was being fingerprinted: {path}")


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def _stable_identity(path: Path, details: os.stat_result) -> _StableFile:
    return _StableFile(
        path=path,
        device=details.st_dev,
        inode=details.st_ino,
        mode=details.st_mode,
        size_bytes=details.st_size,
        mtime_ns=details.st_mtime_ns,
        ctime_ns=details.st_ctime_ns,
    )


def _verify_source_files(files: tuple[_StableFile, ...]) -> None:
    for expected in files:
        observed = expected.path.stat(follow_symlinks=False)
        if (
            expected.path.is_symlink()
            or not stat.S_ISREG(observed.st_mode)
            or (
                observed.st_dev,
                observed.st_ino,
                observed.st_mode,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
            )
            != (
                expected.device,
                expected.inode,
                expected.mode,
                expected.size_bytes,
                expected.mtime_ns,
                expected.ctime_ns,
            )
        ):
            raise ValueError(f"scientific output changed before finalization publication: {expected.path}")


def _strict_tree(root: Path) -> tuple[set[str], set[str]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"required authority directory is missing or unsafe: {root}")
    files: set[str] = set()
    directories: set[str] = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in names:
            path = current / name
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"authority tree contains an unsafe directory entry: {path}")
            directories.add(path.relative_to(root).as_posix())
        for name in filenames:
            path = current / name
            _regular_nonsymlink_file(path, label="authority tree member")
            files.add(path.relative_to(root).as_posix())
    return files, directories


def _absolute_directory(path: Path, label: str, *, create: bool) -> Path:
    if not path.is_absolute():
        raise ValueError(f"postprocessing {label} must be absolute")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError(f"postprocessing {label} must be an existing canonical non-symlink directory")
    return path


def _bundle_identity(path: str, document: bytes) -> PostprocessingBundleMemberIdentity:
    if len(document) > POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1.max_file_bytes:
        raise ValueError(f"postprocessing finalization document exceeds the file limit: {path!r}")
    return PostprocessingBundleMemberIdentity(
        path=path,
        sha256=hashlib.sha256(document).hexdigest(),
        size_bytes=len(document),
    )


def _canonical_mapping(document: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(document)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON mapping")
    mapping = cast("Mapping[str, Any]", payload)
    if document != _canonical_bytes(mapping):
        raise ValueError(f"{label} must use canonical JSON bytes")
    return mapping


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def _write_create_once(path: Path, document: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    try:
        _write_new(temporary, document)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or _stable_file_bytes(path) != document:
                raise ValueError(f"existing postprocessing Runtime evidence differs: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def _write_new(path: Path, document: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _fsync_tree(root: Path) -> None:
    for directory, _, _ in os.walk(root, topdown=False, followlinks=False):
        _fsync_directory(Path(directory))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _relative_parents(path: str) -> set[str]:
    parts = path.split("/")[:-1]
    return {"/".join(parts[:index]) for index in range(1, len(parts) + 1)}


def _sha(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
