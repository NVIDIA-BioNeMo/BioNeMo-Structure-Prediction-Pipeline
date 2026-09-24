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

"""Strict loading and immutable exclusive publication for placement evidence."""

from __future__ import annotations

import errno
import json
import os
import stat
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

import yaml

from bspp.orchestration.contract.database_direct_result import (
    DatabaseDirectResult,
    canonical_database_direct_result_bytes,
    database_direct_result_from_mapping,
)
from bspp.orchestration.contract.database_placement_result import (
    DatabasePlacementFailureEvidence,
    DatabasePlacementResult,
    canonical_database_placement_failure_evidence_bytes,
    canonical_database_placement_result_bytes,
    database_placement_failure_evidence_from_mapping,
    database_placement_result_from_mapping,
)
from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseSourceManifest,
    canonical_database_source_manifest_bytes,
    database_source_manifest_digest,
    database_source_manifest_from_mapping,
)
from bspp.orchestration.contract.phase import PhaseRunSpec

from ._database_placement_errors import DatabasePlacementError
from ._filesystem_authority import FilesystemBinding, _real_directory_stat, _stat_identity, filesystem_binding

_DirectoryBinding = FilesystemBinding
type _DescriptorLoader[T] = Callable[[int, str, Path], T]


class _ImmutableEvidencePublicationCollisionError(DatabasePlacementError):
    """An anchored immutable-publication name collision with a strict comparison."""

    def __init__(self, *, exact_match: bool, description: str) -> None:
        super().__init__(f"{description} destination already exists")
        self.exact_match = exact_match


def load_staged_manifest(runspec: PhaseRunSpec, path: Path) -> DatabaseSourceManifest:
    """Load exact canonical staged manifest bytes and bind them to RunSpec authority."""
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise DatabasePlacementError("staged Database Source Manifest must be a regular non-symlink file")
            raw = handle.read()
            if _stat_identity(info) != _stat_identity(os.fstat(descriptor)):
                raise DatabasePlacementError("staged Database Source Manifest changed while it was read")
        payload = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise DatabasePlacementError(f"staged Database Source Manifest is unavailable or invalid: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise DatabasePlacementError("staged Database Source Manifest must be a mapping")
    try:
        manifest = database_source_manifest_from_mapping(payload)
    except (TypeError, ValueError) as exc:
        raise DatabasePlacementError(f"staged Database Source Manifest is invalid: {exc}") from exc
    if raw != canonical_database_source_manifest_bytes(manifest):
        raise DatabasePlacementError("staged Database Source Manifest must contain exact canonical bytes")
    binding = runspec.payload.database
    if manifest != binding.source_manifest:
        raise DatabasePlacementError("staged Database Source Manifest does not match RunSpec authority")
    if database_source_manifest_digest(manifest) != binding.source_manifest_sha256:
        raise DatabasePlacementError("staged Database Source Manifest digest does not match RunSpec authority")
    return manifest


def load_database_placement_result(path: Path) -> DatabasePlacementResult:
    """Load exact canonical immutable Database Placement Result bytes."""
    return _load_from_path(path, description="Database Placement Result", loader=_load_result_at)


def load_database_direct_result(path: Path) -> DatabaseDirectResult:
    """Load one exact direct-requested or capacity-fallback Result."""
    return _load_from_path(path, description="Database direct Result", loader=_load_direct_result_at)


def load_database_placement_failure(path: Path) -> DatabasePlacementFailureEvidence:
    """Load exact canonical immutable Database Placement failure bytes."""
    return _load_from_path(path, description="Database Placement failure", loader=_load_failure_at)


def publish_database_placement_result(result: DatabasePlacementResult, destination: Path) -> None:
    """Exclusively publish and descriptor-verify one immutable Result."""
    _publish_immutable_exclusive(
        result,
        destination,
        serializer=canonical_database_placement_result_bytes,
        loader=_load_result_at,
        description="Database Placement Result",
    )


def publish_database_direct_result(result: DatabaseDirectResult, destination: Path) -> None:
    """Exclusively publish and descriptor-verify one direct-family Result."""
    _publish_immutable_exclusive(
        result,
        destination,
        serializer=canonical_database_direct_result_bytes,
        loader=_load_direct_result_at,
        description="Database direct Result",
    )


def publish_database_placement_failure(failure: DatabasePlacementFailureEvidence, destination: Path) -> None:
    """Exclusively publish and descriptor-verify one immutable classified failure."""
    _publish_immutable_exclusive(
        failure,
        destination,
        serializer=canonical_database_placement_failure_evidence_bytes,
        loader=_load_failure_at,
        description="Database Placement failure",
    )


def _load_from_path[T](path: Path, *, description: str, loader: _DescriptorLoader[T]) -> T:
    parent_descriptor, parent_identity = _open_anchored_directory(
        path.parent,
        description=f"{description} parent",
    )
    try:
        value = loader(parent_descriptor, path.name, path)
        _verify_directory_binding(
            path.parent,
            parent_descriptor,
            parent_identity,
            description=f"{description} parent",
        )
        return value
    finally:
        os.close(parent_descriptor)


def _load_result_at(parent_descriptor: int, name: str, display_path: Path) -> DatabasePlacementResult:
    return _load_canonical_json_at(
        parent_descriptor,
        name,
        display_path,
        parser=database_placement_result_from_mapping,
        serializer=canonical_database_placement_result_bytes,
        description="Database Placement Result",
    )


def _load_direct_result_at(parent_descriptor: int, name: str, display_path: Path) -> DatabaseDirectResult:
    return _load_canonical_json_at(
        parent_descriptor,
        name,
        display_path,
        parser=database_direct_result_from_mapping,
        serializer=canonical_database_direct_result_bytes,
        description="Database direct Result",
    )


def _load_failure_at(parent_descriptor: int, name: str, display_path: Path) -> DatabasePlacementFailureEvidence:
    return _load_canonical_json_at(
        parent_descriptor,
        name,
        display_path,
        parser=database_placement_failure_evidence_from_mapping,
        serializer=canonical_database_placement_failure_evidence_bytes,
        description="Database Placement failure",
    )


def _load_canonical_json_at[T](
    parent_descriptor: int,
    name: str,
    display_path: Path,
    *,
    parser: Callable[[Mapping[str, object]], T],
    serializer: Callable[[T], bytes],
    description: str,
) -> T:
    descriptor: int | None = None
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            info = os.fstat(descriptor)
            raw = handle.read()
            if _stat_identity(info) != _stat_identity(os.fstat(descriptor)):
                raise DatabasePlacementError(f"{description} changed while it was read")
        payload = json.loads(raw)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
            raise DatabasePlacementError(f"{description} must be an immutable 0444 regular file")
        if not isinstance(payload, Mapping):
            raise DatabasePlacementError(f"{description} must be a mapping")
        value = parser(payload)
        if raw != serializer(value):
            raise DatabasePlacementError(f"{description} must contain exact canonical bytes")
        return value
    except DatabasePlacementError:
        raise
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DatabasePlacementError(f"{description} is unavailable or invalid: {display_path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_immutable_exclusive[T](
    value: T,
    destination: Path,
    *,
    serializer: Callable[[T], bytes],
    loader: _DescriptorLoader[T],
    description: str,
) -> None:
    """One immutable hard-link transaction shared by Result and failure evidence."""
    content = serializer(value)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatabasePlacementError(f"cannot prepare {description} parent: {exc}") from exc
    parent_descriptor, parent_identity = _open_anchored_directory(
        destination.parent,
        description=f"{description} parent",
    )
    temporary_name = f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        with os.fdopen(temporary_descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            try:
                exact_match = loader(parent_descriptor, destination.name, destination) == value
            except DatabasePlacementError:
                exact_match = False
            _verify_directory_binding(
                destination.parent,
                parent_descriptor,
                parent_identity,
                description=f"{description} parent",
            )
            raise _ImmutableEvidencePublicationCollisionError(
                exact_match=exact_match,
                description=description,
            ) from exc
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        if loader(parent_descriptor, destination.name, destination) != value:
            raise DatabasePlacementError(f"published {description} failed exact verification")
        _verify_directory_binding(
            destination.parent,
            parent_descriptor,
            parent_identity,
            description=f"{description} parent",
        )
    except DatabasePlacementError:
        raise
    except OSError as exc:
        raise DatabasePlacementError(f"cannot exclusively publish {description}: {exc}") from exc
    finally:
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise DatabasePlacementError(f"cannot remove temporary {description}: {exc}") from exc
        finally:
            os.close(parent_descriptor)


def _open_anchored_directory(path: Path, *, description: str) -> tuple[int, _DirectoryBinding]:
    path_info = _real_directory_stat(path, description=description)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise DatabasePlacementError(f"{description} is unavailable: {path}") from exc
    identity = _directory_binding(path_info)
    if _directory_binding(os.fstat(descriptor)) != identity:
        os.close(descriptor)
        raise DatabasePlacementError(f"{description} changed while it was opened")
    return descriptor, identity


def _verify_directory_binding(
    path: Path,
    descriptor: int,
    expected_identity: _DirectoryBinding,
    *,
    description: str,
) -> None:
    descriptor_identity = _directory_binding(os.fstat(descriptor))
    path_identity = _directory_binding(_real_directory_stat(path, description=description))
    if descriptor_identity != expected_identity or path_identity != expected_identity:
        raise DatabasePlacementError(f"{description} changed during observation")


def _directory_binding(info: os.stat_result) -> _DirectoryBinding:
    return filesystem_binding(info)


__all__ = [
    "load_database_direct_result",
    "load_database_placement_failure",
    "load_database_placement_result",
    "load_staged_manifest",
    "publish_database_direct_result",
    "publish_database_placement_failure",
    "publish_database_placement_result",
]
