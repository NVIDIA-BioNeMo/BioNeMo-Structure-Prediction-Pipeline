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

"""Descriptor-anchored Database Source mount and inventory observation."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from bspp.orchestration.contract.database_placement import DATABASE_SOURCE_ROOT
from bspp.orchestration.contract.database_placement_result import (
    DatabasePostScienceObservation,
    DatabaseSourceObservation,
)
from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseAlias,
    DatabaseSourceManifest,
    DatabaseSourceMember,
    database_source_manifest_digest,
)

from ._database_placement_errors import (
    DatabasePlacementError,
    PostScienceSourceObservationError,
)
from ._filesystem_authority import (
    FilesystemObjectIdentity,
    FilesystemObservation,
    PathObservation,
    _real_directory_stat,
    _stat_identity,
)

_PathIdentity = PathObservation
_RootIdentity = FilesystemObservation
_SOURCE_ROOT_DESCRIPTION = "protected Database Source root"


def open_source_root(root: Path) -> tuple[int, _RootIdentity]:
    """Open a real directory root without following any path-component symlink."""
    path_info = _real_directory_stat(root, description=_SOURCE_ROOT_DESCRIPTION)
    try:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise DatabasePlacementError(f"protected Database Source root is unavailable: {root}") from exc
    identity = _stat_identity(path_info)
    if _stat_identity(os.fstat(descriptor)) != identity:
        os.close(descriptor)
        raise DatabasePlacementError("protected Database Source root changed while it was opened")
    return descriptor, identity


def observe_source_inventory(
    root: Path,
    root_descriptor: int,
    manifest: DatabaseSourceManifest,
) -> DatabaseSourceObservation:
    """Strictly observe the exact complete pre-science manifest closure."""
    root_info = os.fstat(root_descriptor)
    try:
        members: list[DatabaseSourceMember] = []
        inode_paths: dict[FilesystemObjectIdentity, str] = {}
        observed_paths: dict[str, _PathIdentity] = {}
        for expected in manifest.members:
            observed, resolved_info, member_paths = _observe_member(root, root_descriptor, expected)
            if observed != expected:
                raise DatabasePlacementError(
                    f"source member metadata does not match manifest for {expected.logical_name!r}"
                )
            inode = FilesystemObjectIdentity(device=resolved_info.st_dev, inode=resolved_info.st_ino)
            prior_path = inode_paths.get(inode)
            if prior_path is not None and prior_path != observed.resolved_path:
                raise DatabasePlacementError(
                    "distinct resolved Database Source paths must not bypass identity through a hard link: "
                    f"{prior_path!r}, {observed.resolved_path!r}"
                )
            inode_paths[inode] = observed.resolved_path
            _merge_observed_paths(observed_paths, member_paths)
            members.append(observed)
        _verify_complete_tree(root_descriptor, manifest, observed_paths)
        if _stat_identity(root_info) != _stat_identity(os.fstat(root_descriptor)):
            raise DatabasePlacementError("protected Database Source root changed during observation")
    except OSError as exc:
        raise DatabasePlacementError(f"cannot observe protected Database Source inventory: {exc}") from exc
    return DatabaseSourceObservation(
        source_container_root=DATABASE_SOURCE_ROOT,
        source_manifest_sha256=database_source_manifest_digest(manifest),
        members=tuple(members),
        verification="metadata-verified",
    )


def observe_post_science_source(
    root: Path,
    manifest: DatabaseSourceManifest,
    *,
    source_manifest_sha256: str,
) -> DatabasePostScienceObservation:
    """Capture one complete post-science tree or raise the dedicated failure variant error."""
    try:
        descriptor, identity = open_source_root(root)
    except DatabasePlacementError as exc:
        raise PostScienceSourceObservationError(str(exc)) from exc
    try:
        members: list[DatabaseSourceMember] = []
        errors: list[str] = []
        observed_paths: dict[str, _PathIdentity] = {}
        for expected in manifest.members:
            try:
                observed, _, member_paths = _observe_member(root, descriptor, expected)
                members.append(observed)
                _merge_observed_paths(observed_paths, member_paths, errors=errors)
                if observed != expected:
                    errors.append(f"source member metadata drift: {expected.logical_name}")
            except DatabasePlacementError as exc:
                errors.append(f"{expected.logical_name}: {exc}")
        inventory_paths = _observe_complete_tree_paths(descriptor, observed_paths, errors)
        try:
            verify_source_root_binding(root, descriptor, identity)
        except DatabasePlacementError as exc:
            errors.append(str(exc))
        return DatabasePostScienceObservation(
            source_container_root=DATABASE_SOURCE_ROOT,
            source_manifest_sha256=source_manifest_sha256,
            members=tuple(members),
            inventory_paths=inventory_paths,
            errors=_bounded_errors(errors),
        )
    except PostScienceSourceObservationError:
        raise
    except OSError as exc:
        raise PostScienceSourceObservationError(
            f"cannot completely observe selected Database Source root: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def verify_source_root_binding(
    root: Path,
    descriptor: int,
    expected_identity: _RootIdentity,
) -> None:
    """Verify the path and open descriptor still identify the same source root."""
    descriptor_identity = _stat_identity(os.fstat(descriptor))
    path_identity = _stat_identity(_real_directory_stat(root, description=_SOURCE_ROOT_DESCRIPTION))
    if descriptor_identity != expected_identity or path_identity != expected_identity:
        raise DatabasePlacementError("protected Database Source root changed during observation")


def _observe_member(
    root: Path,
    root_descriptor: int,
    expected: DatabaseSourceMember,
) -> tuple[DatabaseSourceMember, os.stat_result, dict[str, _PathIdentity]]:
    pending = list(Path(expected.source_path).parts)
    if not pending:
        raise DatabasePlacementError(f"empty source path for {expected.logical_name!r}")
    descriptor = os.dup(root_descriptor)
    current_parts: list[str] = []
    aliases: list[DatabaseAlias] = []
    visited_aliases: set[str] = set()
    observed_paths: dict[str, _PathIdentity] = {}
    source_kind: Literal["regular", "symlink"] = "regular"
    try:
        while pending:
            name = pending.pop(0)
            if name in {"", ".", ".."} or "/" in name:
                raise DatabasePlacementError(f"invalid source component for {expected.logical_name!r}: {name!r}")
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = Path(*current_parts, name).as_posix()
            if stat.S_ISLNK(info.st_mode):
                if not pending and not aliases:
                    source_kind = "symlink"
                if relative in visited_aliases:
                    raise DatabasePlacementError(
                        f"cyclic source alias for logical member {expected.logical_name!r}: {relative}"
                    )
                visited_aliases.add(relative)
                target = os.readlink(name, dir_fd=descriptor)
                stable_alias_info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if _stat_identity(info) != _stat_identity(stable_alias_info):
                    raise DatabasePlacementError(
                        f"source alias changed during observation for {expected.logical_name!r}: {relative}"
                    )
                observed_paths[relative] = PathObservation(_stat_identity(stable_alias_info), target)
                if not target or "\x00" in target:
                    raise DatabasePlacementError(
                        f"invalid source alias for logical member {expected.logical_name!r}: {relative}"
                    )
                aliases.append(DatabaseAlias(path=relative, target=target))
                target_path = Path(target)
                if target_path.is_absolute():
                    try:
                        target_parts = list(target_path.relative_to(root).parts)
                    except ValueError as exc:
                        raise DatabasePlacementError(
                            f"source alias escapes protected root for {expected.logical_name!r}: {target}"
                        ) from exc
                else:
                    normalized = os.path.normpath(str(Path(*current_parts) / target_path))
                    normalized_path = Path(normalized)
                    if normalized_path.is_absolute() or ".." in normalized_path.parts:
                        raise DatabasePlacementError(
                            f"source alias escapes protected root for {expected.logical_name!r}: {target}"
                        )
                    target_parts = [] if normalized == "." else list(normalized_path.parts)
                pending = [*target_parts, *pending]
                os.close(descriptor)
                descriptor = os.dup(root_descriptor)
                current_parts = []
                continue
            if pending:
                if not stat.S_ISDIR(info.st_mode):
                    raise DatabasePlacementError(
                        f"unsupported source path component for {expected.logical_name!r}: {relative}"
                    )
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                if _stat_identity(info) != _stat_identity(os.fstat(child)):
                    os.close(child)
                    raise DatabasePlacementError(
                        f"source directory changed during observation for {expected.logical_name!r}: {relative}"
                    )
                observed_paths[relative] = PathObservation(_stat_identity(info), None)
                os.close(descriptor)
                descriptor = child
                current_parts.append(name)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise DatabasePlacementError(
                    f"unsupported source leaf for logical member {expected.logical_name!r}: {relative}"
                )
            leaf_descriptor = os.open(name, os.O_PATH | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                if _stat_identity(info) != _stat_identity(os.fstat(leaf_descriptor)):
                    raise DatabasePlacementError(
                        f"source leaf changed during observation for {expected.logical_name!r}: {relative}"
                    )
            finally:
                os.close(leaf_descriptor)
            observed_paths[relative] = PathObservation(_stat_identity(info), None)
            observed = DatabaseSourceMember(
                role=expected.role,
                database_name=expected.database_name,
                logical_name=expected.logical_name,
                source_path=expected.source_path,
                source_kind=source_kind,
                resolved_path=relative,
                resolved_kind="regular",
                size_bytes=info.st_size,
                mtime_ns=info.st_mtime_ns,
                alias_topology=tuple(aliases),
                preexisting_checksum=expected.preexisting_checksum,
            )
            return observed, info, observed_paths
    except FileNotFoundError as exc:
        raise DatabasePlacementError(
            f"missing source member {expected.logical_name!r}: {expected.source_path}"
        ) from exc
    finally:
        os.close(descriptor)
    raise DatabasePlacementError(f"incomplete source member observation for {expected.logical_name!r}")


def _verify_complete_tree(
    root_descriptor: int,
    manifest: DatabaseSourceManifest,
    expected_identities: Mapping[str, _PathIdentity],
) -> None:
    allowed: set[str] = set()
    for member in manifest.members:
        for value in (
            member.source_path,
            member.resolved_path,
            *(alias.path for alias in member.alias_topology),
        ):
            path = Path(value)
            allowed.update(Path(*path.parts[:index]).as_posix() for index in range(1, len(path.parts) + 1))
    observed: set[str] = set()

    def visit(descriptor: int, prefix: tuple[str, ...]) -> None:
        for name in sorted(os.listdir(descriptor)):
            if name in {".", ".."} or "/" in name:
                raise DatabasePlacementError("protected Database Source inventory contains an invalid entry name")
            relative = Path(*prefix, name).as_posix()
            observed.add(relative)
            if relative not in allowed:
                raise DatabasePlacementError(f"unexpected Database Source inventory entry: {relative}")
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            target = os.readlink(name, dir_fd=descriptor) if stat.S_ISLNK(info.st_mode) else None
            if expected_identities.get(relative) != PathObservation(_stat_identity(info), target):
                raise DatabasePlacementError(f"Database Source path changed during observation: {relative}")
            if stat.S_ISDIR(info.st_mode):
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                try:
                    if _stat_identity(info) != _stat_identity(os.fstat(child)):
                        raise DatabasePlacementError(
                            f"Database Source directory changed during observation: {relative}"
                        )
                    visit(child, (*prefix, name))
                    if _stat_identity(info) != _stat_identity(os.fstat(child)):
                        raise DatabasePlacementError(
                            f"Database Source directory changed during observation: {relative}"
                        )
                finally:
                    os.close(child)
            elif not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                raise DatabasePlacementError(f"unsupported Database Source inventory entry: {relative}")

    visit(root_descriptor, ())
    missing = sorted(allowed - observed)
    if missing:
        raise DatabasePlacementError(f"Database Source inventory is incomplete: {missing!r}")


def _observe_complete_tree_paths(
    root_descriptor: int,
    expected_identities: Mapping[str, _PathIdentity],
    errors: list[str],
) -> tuple[str, ...]:
    observed: list[str] = []

    def visit(descriptor: int, prefix: tuple[str, ...]) -> None:
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as exc:
            raise PostScienceSourceObservationError(
                f"cannot completely list selected Database Source inventory: {exc}"
            ) from exc
        for name in names:
            if name in {".", ".."} or "/" in name:
                errors.append(f"invalid Database Source inventory name: {name!r}")
                continue
            relative = Path(*prefix, name).as_posix()
            observed.append(relative)
            try:
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                target = os.readlink(name, dir_fd=descriptor) if stat.S_ISLNK(info.st_mode) else None
                expected_identity = expected_identities.get(relative)
                if expected_identity is not None and expected_identity != PathObservation(_stat_identity(info), target):
                    errors.append(f"Database Source path changed during observation: {relative}")
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                    try:
                        if _stat_identity(info) != _stat_identity(os.fstat(child)):
                            errors.append(f"Database Source directory changed during observation: {relative}")
                        visit(child, (*prefix, name))
                        if _stat_identity(info) != _stat_identity(os.fstat(child)):
                            errors.append(f"Database Source directory changed during observation: {relative}")
                    finally:
                        os.close(child)
            except FileNotFoundError:
                errors.append(f"Database Source path disappeared during observation: {relative}")
            except OSError as exc:
                raise PostScienceSourceObservationError(
                    f"cannot completely observe selected Database Source path {relative!r}: {exc}"
                ) from exc

    visit(root_descriptor, ())
    observed_set = set(observed)
    for missing in sorted(set(expected_identities) - observed_set):
        errors.append(f"Database Source path disappeared during observation: {missing}")
    return tuple(sorted(observed_set))


def _merge_observed_paths(
    destination: dict[str, _PathIdentity],
    source: Mapping[str, _PathIdentity],
    *,
    errors: list[str] | None = None,
) -> None:
    for path, identity in source.items():
        prior = destination.setdefault(path, identity)
        if prior != identity:
            message = f"source path changed during observation: {path}"
            if errors is None:
                raise DatabasePlacementError(message)
            errors.append(message)


def _bounded_errors(errors: list[str]) -> tuple[str, ...]:
    return tuple(item[:512] for item in errors[:128])


__all__ = [
    "observe_post_science_source",
    "observe_source_inventory",
    "open_source_root",
    "verify_source_root_binding",
]
