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

"""Acceptance-cache profile selection and effective-user scope binding."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path

import yaml

from bspp.orchestration.contract.database_cache_maintenance import (
    DatabaseAcceptanceCacheProfile,
    select_database_acceptance_cache_profile,
)

from ._database_cache_maintenance_types import (
    DatabaseCacheMaintenanceError,
    _DatabaseAcceptanceCacheAuthority,
    _Entry,
    _PhysicalCacheRootBinding,
    _Scope,
)
from ._filesystem_authority import (
    DirectoryAuthority,
    _real_directory_stat,
    directory_authority,
    filesystem_authority,
    filesystem_object_identity,
)
from ._linux_mount_authority import (
    freeze_directory_mount_authority,
)

_FINAL = re.compile(r"[0-9a-f]{64}")
_POPULATION = re.compile(r"\.population-([0-9a-f]{64})-[0-9a-f]{32}")


def load_database_acceptance_cache_profile(config_path: Path, profile_name: str) -> DatabaseAcceptanceCacheProfile:
    """Load YAML once and select the contract-owned minimal projection."""
    raw = _load_database_cache_config(config_path)
    return _select_database_acceptance_cache_profile(raw, profile_name)


def load_database_acceptance_cache_authority(
    config_path: Path,
    profile_name: str,
) -> _DatabaseAcceptanceCacheAuthority:
    """Load the pure profile projection plus Runtime-only sibling path authority."""
    raw = _load_database_cache_config(config_path)
    profile = _select_database_acceptance_cache_profile(raw, profile_name)
    clusters = raw["clusters"]
    assert isinstance(clusters, Mapping)
    sibling_roots = tuple(
        Path(root)
        for name, item in clusters.items()
        if name != profile_name
        and isinstance(item, Mapping)
        and (root := item.get("database_cache_root")) is not None
        and isinstance(root, str)
    )
    return _DatabaseAcceptanceCacheAuthority(
        profile=profile,
        configured_sibling_cache_roots=sibling_roots,
    )


def _load_database_cache_config(config_path: Path) -> Mapping[str, object]:
    try:
        raw = yaml.safe_load(config_path.read_bytes())
    except (OSError, yaml.YAMLError) as exc:
        raise DatabaseCacheMaintenanceError(f"database cache maintenance config is unavailable: {config_path}") from exc
    if not isinstance(raw, Mapping):
        raise DatabaseCacheMaintenanceError("database cache maintenance config must be a mapping")
    return raw


def _select_database_acceptance_cache_profile(
    raw: Mapping[str, object],
    profile_name: str,
) -> DatabaseAcceptanceCacheProfile:
    try:
        return select_database_acceptance_cache_profile(raw, profile_name)
    except (TypeError, ValueError) as exc:
        raise DatabaseCacheMaintenanceError(str(exc)) from exc


def _bind_configured_sibling_roots(
    paths: tuple[Path, ...],
    *,
    mountinfo_path: Path,
) -> tuple[_PhysicalCacheRootBinding, ...]:
    bindings: list[_PhysicalCacheRootBinding] = []
    try:
        for path in paths:
            opened = _open_physical_cache_root(path)
            if opened is None:
                bindings.append(
                    _PhysicalCacheRootBinding(
                        path=path,
                        descriptor=None,
                        identity=None,
                        mount_authority=None,
                        configured_path_uses_symlink=False,
                    )
                )
                continue
            descriptor, identity = opened
            try:
                configured_path_uses_symlink = _configured_path_uses_symlink(path)
                authority_path = _descriptor_resolved_path(descriptor) if configured_path_uses_symlink else path
                mount_authority = freeze_directory_mount_authority(
                    role="configured-sibling-cache-root",
                    path=authority_path,
                    descriptor=descriptor,
                    mountinfo_path=mountinfo_path,
                )
            except BaseException:
                os.close(descriptor)
                raise
            bindings.append(
                _PhysicalCacheRootBinding(
                    path=path,
                    descriptor=descriptor,
                    identity=identity,
                    mount_authority=mount_authority,
                    configured_path_uses_symlink=configured_path_uses_symlink,
                )
            )
        return tuple(bindings)
    except BaseException:
        for binding in bindings:
            if binding.descriptor is not None:
                os.close(binding.descriptor)
        raise


def _configured_path_uses_symlink(path: Path) -> bool:
    """Observe every configured component without granting symlink authority."""
    absolute = Path(os.path.abspath(path))
    current = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in absolute.parts[1:]:
            try:
                info = os.stat(component, dir_fd=current, follow_symlinks=False)
            except FileNotFoundError:
                return False
            except OSError as exc:
                raise DatabaseCacheMaintenanceError(f"configured sibling cache root is unavailable: {path}") from exc
            if stat.S_ISLNK(info.st_mode):
                return True
            try:
                descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=current,
                )
            except OSError as exc:
                raise DatabaseCacheMaintenanceError(f"configured sibling cache root is unavailable: {path}") from exc
            opened = os.fstat(descriptor)
            if filesystem_object_identity(opened) != filesystem_object_identity(info):
                os.close(descriptor)
                raise DatabaseCacheMaintenanceError(f"configured sibling cache root changed while observed: {path}")
            os.close(current)
            current = descriptor
        return False
    finally:
        os.close(current)


def _descriptor_resolved_path(descriptor: int) -> Path:
    """Return the kernel-resolved visible path for one retained directory."""
    try:
        resolved = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
    except OSError as exc:
        raise DatabaseCacheMaintenanceError("configured sibling physical path is unavailable") from exc
    if not resolved.is_absolute() or resolved.name.endswith(" (deleted)"):
        raise DatabaseCacheMaintenanceError("configured sibling physical path is invalid")
    return resolved


def _open_physical_cache_root(path: Path) -> tuple[int, DirectoryAuthority] | None:
    descriptor: int | None = None
    try:
        visible = os.stat(path, follow_symlinks=True)
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except FileNotFoundError:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.lstat(path)
        except FileNotFoundError:
            return None
        raise DatabaseCacheMaintenanceError(f"configured sibling cache root is unresolved: {path}") from None
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise DatabaseCacheMaintenanceError(f"configured sibling cache root is unavailable: {path}") from exc
    try:
        identity = _physical_cache_root_identity(visible)
        if _physical_cache_root_identity(os.fstat(descriptor)) != identity:
            raise DatabaseCacheMaintenanceError(f"configured sibling cache root changed while opened: {path}")
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _physical_cache_root_identity(info: os.stat_result) -> DirectoryAuthority:
    if not stat.S_ISDIR(info.st_mode):
        raise DatabaseCacheMaintenanceError("configured sibling cache root must resolve to a directory")
    return directory_authority(info)


def _open_scope(
    profile: DatabaseAcceptanceCacheProfile,
    *,
    effective_user: str,
    effective_uid: int,
) -> _Scope:
    cache_root = Path(profile.cache_root)
    user_namespace = cache_root / "users" / effective_user
    root_fd, root_identity = _open_directory(cache_root, expected_uid=effective_uid, expected_device=None)
    users_fd: int | None = None
    user_fd: int | None = None
    try:
        users_fd, users_identity = _open_relative_directory(
            root_fd, "users", expected_uid=effective_uid, expected_device=root_identity.device
        )
        user_fd, user_identity = _open_relative_directory(
            users_fd, effective_user, expected_uid=effective_uid, expected_device=root_identity.device
        )
        return _Scope(
            profile=profile,
            effective_user=effective_user,
            effective_uid=effective_uid,
            cache_root=cache_root,
            user_namespace=user_namespace,
            replicas_path=user_namespace / "replicas",
            cache_root_descriptor=root_fd,
            users_descriptor=users_fd,
            user_descriptor=user_fd,
            cache_root_identity=root_identity,
            users_identity=users_identity,
            user_identity=user_identity,
        )
    except BaseException:
        if user_fd is not None:
            os.close(user_fd)
        if users_fd is not None:
            os.close(users_fd)
        os.close(root_fd)
        raise


def _open_directory(
    path: Path,
    *,
    expected_uid: int,
    expected_device: int | None,
) -> tuple[int, DirectoryAuthority]:
    info = _real_directory_stat(path, description="acceptance database cache authority")
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        identity = _directory_identity(info, expected_uid=expected_uid, expected_device=expected_device)
        if (
            _directory_identity(os.fstat(descriptor), expected_uid=expected_uid, expected_device=identity.device)
            != identity
        ):
            raise DatabaseCacheMaintenanceError("acceptance database cache authority changed while opened")
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_directory(
    parent_descriptor: int,
    name: str,
    *,
    expected_uid: int,
    expected_device: int,
) -> tuple[int, DirectoryAuthority]:
    try:
        info = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_descriptor)
    except OSError as exc:
        raise DatabaseCacheMaintenanceError(f"acceptance database cache directory is unavailable: {name!r}") from exc
    try:
        identity = _directory_identity(info, expected_uid=expected_uid, expected_device=expected_device)
        if (
            _directory_identity(os.fstat(descriptor), expected_uid=expected_uid, expected_device=expected_device)
            != identity
        ):
            raise DatabaseCacheMaintenanceError(f"acceptance database cache directory changed: {name!r}")
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _open_optional_locks(scope: _Scope) -> int | None:
    try:
        descriptor, _ = _open_relative_directory(
            scope.user_descriptor,
            ".locks",
            expected_uid=scope.effective_uid,
            expected_device=scope.user_identity.device,
        )
        return descriptor
    except DatabaseCacheMaintenanceError as exc:
        try:
            os.stat(".locks", dir_fd=scope.user_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None
        raise exc


def _open_optional_replicas(scope: _Scope) -> int | None:
    try:
        descriptor, _ = _open_relative_directory(
            scope.user_descriptor,
            "replicas",
            expected_uid=scope.effective_uid,
            expected_device=scope.user_identity.device,
        )
        return descriptor
    except DatabaseCacheMaintenanceError as exc:
        try:
            os.stat("replicas", dir_fd=scope.user_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None
        raise exc


def _scan_replicas(replicas_descriptor: int, *, cache_device: int) -> tuple[_Entry, ...]:
    try:
        names = tuple(sorted(os.listdir(replicas_descriptor)))
    except OSError as exc:
        raise DatabaseCacheMaintenanceError("cannot enumerate acceptance database replicas") from exc
    entries: list[_Entry] = []
    for name in names:
        try:
            info = os.stat(name, dir_fd=replicas_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise DatabaseCacheMaintenanceError(f"cannot bind database cache entry: {name!r}") from exc
        entries.append(_entry_from_stat(name, info, cache_device=cache_device))
    return tuple(entries)


def _entry_from_stat(name: str, info: os.stat_result, *, cache_device: int) -> _Entry:
    population = _POPULATION.fullmatch(name)
    if _FINAL.fullmatch(name) is not None:
        kind = "replica"
        identity = name
        expected_mode = 0o555
    elif population is not None:
        kind = "population"
        identity = population.group(1)
        expected_mode = 0o700
    else:
        raise DatabaseCacheMaintenanceError(f"unknown top-level database cache entry: {name!r}")
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_dev != cache_device
        or stat.S_IMODE(info.st_mode) != expected_mode
    ):
        raise DatabaseCacheMaintenanceError(f"database cache entry authority is invalid: {name!r}")
    return _Entry(
        kind=kind,
        identity=identity,
        basename=name,
        authority=filesystem_authority(info),
    )


def _verify_scope(scope: _Scope, *, replicas_descriptor: int | None = None) -> None:
    if (
        _visible_directory_identity(scope.cache_root, scope.effective_uid, scope.cache_root_identity.device)
        != scope.cache_root_identity
    ):
        raise DatabaseCacheMaintenanceError("acceptance database cache root changed")
    if (
        _directory_identity(
            os.fstat(scope.cache_root_descriptor),
            expected_uid=scope.effective_uid,
            expected_device=scope.cache_root_identity.device,
        )
        != scope.cache_root_identity
    ):
        raise DatabaseCacheMaintenanceError("acceptance database cache root descriptor changed")
    for parent, name, descriptor, expected in (
        (scope.cache_root_descriptor, "users", scope.users_descriptor, scope.users_identity),
        (scope.users_descriptor, scope.effective_user, scope.user_descriptor, scope.user_identity),
    ):
        visible = _directory_identity(
            os.stat(name, dir_fd=parent, follow_symlinks=False),
            expected_uid=scope.effective_uid,
            expected_device=scope.cache_root_identity.device,
        )
        opened = _directory_identity(
            os.fstat(descriptor),
            expected_uid=scope.effective_uid,
            expected_device=scope.cache_root_identity.device,
        )
        if visible != expected or opened != expected:
            raise DatabaseCacheMaintenanceError("acceptance database cache scope changed")
    if replicas_descriptor is not None:
        visible = _directory_identity(
            os.stat("replicas", dir_fd=scope.user_descriptor, follow_symlinks=False),
            expected_uid=scope.effective_uid,
            expected_device=scope.cache_root_identity.device,
        )
        opened = _directory_identity(
            os.fstat(replicas_descriptor),
            expected_uid=scope.effective_uid,
            expected_device=scope.cache_root_identity.device,
        )
        if visible != opened:
            raise DatabaseCacheMaintenanceError("acceptance database replicas authority changed")


def _visible_directory_identity(
    path: Path,
    uid: int,
    device: int,
) -> DirectoryAuthority:
    return _directory_identity(
        _real_directory_stat(path, description="acceptance database cache authority"),
        expected_uid=uid,
        expected_device=device,
    )


def _directory_identity(
    info: os.stat_result,
    *,
    expected_uid: int,
    expected_device: int | None,
) -> DirectoryAuthority:
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != expected_uid
        or stat.S_IMODE(info.st_mode) != 0o700
        or (expected_device is not None and info.st_dev != expected_device)
    ):
        raise DatabaseCacheMaintenanceError("acceptance database cache directory authority is invalid")
    return directory_authority(info)


def _require_filesystem_type(
    path: Path,
    descriptor: int,
    expected: str,
    *,
    mountinfo_path: Path,
) -> None:
    device = os.fstat(descriptor).st_dev
    candidates: list[tuple[int, str]] = []
    try:
        lines = mountinfo_path.read_text().splitlines()
    except OSError as exc:
        raise DatabaseCacheMaintenanceError("Linux mountinfo is unavailable for cache maintenance") from exc
    for line in lines:
        try:
            before, after = line.split(" - ", 1)
            fields = before.split()
            major, minor = (int(item) for item in fields[2].split(":"))
            mountpoint = Path(_decode_mount_path(fields[4]))
            filesystem_type = after.split()[0]
        except (IndexError, TypeError, ValueError) as exc:
            raise DatabaseCacheMaintenanceError("Linux mountinfo is malformed") from exc
        if os.makedev(major, minor) != device:
            continue
        try:
            path.relative_to(mountpoint)
        except ValueError:
            continue
        candidates.append((len(mountpoint.parts), filesystem_type))
    if not candidates or max(candidates)[1] != expected:
        raise DatabaseCacheMaintenanceError("acceptance database cache filesystem type does not match profile")


def _decode_mount_path(value: str) -> str:
    for encoded, decoded in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(encoded, decoded)
    return value


__all__ = [
    "_bind_configured_sibling_roots",
    "_descriptor_resolved_path",
    "_entry_from_stat",
    "_open_optional_locks",
    "_open_optional_replicas",
    "_open_physical_cache_root",
    "_open_scope",
    "_require_filesystem_type",
    "_scan_replicas",
    "_verify_scope",
    "load_database_acceptance_cache_authority",
    "load_database_acceptance_cache_profile",
]
