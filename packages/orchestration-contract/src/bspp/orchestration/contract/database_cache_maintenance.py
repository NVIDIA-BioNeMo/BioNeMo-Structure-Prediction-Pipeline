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

"""Acceptance-only Database Replica cache maintenance contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bspp.orchestration.contract.runplan import reject_environment_interpolation
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

DatabaseCacheMaintenanceResult = Literal["cleared", "already-empty", "refused", "failed"]
DatabaseCacheEntryKind = Literal["replica", "population"]
DatabaseCacheLockObservationKind = Literal["acquired", "contended"]

_INHERITANCE_KEYS = frozenset({"extends", "inherits", "parent", "base_profile"})
_SAFE_UNIX_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")
_FILESYSTEM_TOKEN = re.compile(r"[A-Za-z0-9_.+-]{1,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_EVIDENCE_KIND = "database-acceptance-cache-maintenance-v1"
_LOCK_PROTOCOL = "cache-exclusive-then-identity-exclusive-nonblocking-v1"
_MAX_LOCK_SAMPLES = 32
_MAX_REMOVED_SAMPLES = 32
_MAX_DIAGNOSTIC = 2048


@dataclass(frozen=True)
class DatabaseAcceptanceCacheProfile:
    """The sole cache-maintenance projection selected from a Cluster Profile."""

    profile_name: str
    database_cache_namespace: Literal["acceptance"]
    cache_root: str
    unix_user: str
    expected_filesystem_type: str
    lock_wait_seconds: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="DatabaseAcceptanceCacheProfile")
        if _PROFILE_NAME.fullmatch(self.profile_name) is None:
            raise ValueError("database cache maintenance profile name is invalid")
        if self.database_cache_namespace != "acceptance":
            raise ValueError("database cache maintenance requires the exact acceptance namespace marker")
        _validate_cache_root(self.cache_root)
        if _SAFE_UNIX_USER.fullmatch(self.unix_user) is None:
            raise ValueError("database cache maintenance requires a safe Unix user identity")
        if _FILESYSTEM_TOKEN.fullmatch(self.expected_filesystem_type) is None:
            raise ValueError("database cache maintenance expected filesystem type is invalid")
        if (
            not isinstance(self.lock_wait_seconds, int)
            or isinstance(self.lock_wait_seconds, bool)
            or self.lock_wait_seconds <= 0
        ):
            raise ValueError("database cache maintenance lock wait must be a positive integer")

    @property
    def user_cache_root(self) -> str:
        return str(Path(self.cache_root) / "users" / self.unix_user)

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_database_cache_maintenance_profile_bytes(self)).hexdigest()

    def to_mapping(self) -> dict[str, object]:
        return {
            "cache_root": self.cache_root,
            "database_cache_namespace": self.database_cache_namespace,
            "expected_filesystem_type": self.expected_filesystem_type,
            "lock_wait_seconds": self.lock_wait_seconds,
            "profile_name": self.profile_name,
            "schema_version": self.schema_version,
            "unix_user": self.unix_user,
        }


@dataclass(frozen=True)
class DatabaseCacheMaintenanceDescriptor:
    """Bounded descriptor identity proving one component of the selected scope."""

    role: Literal["cache-root", "users-root", "user-namespace", "locks", "replicas"]
    device: int
    inode: int
    uid: int
    mode: int

    def __post_init__(self) -> None:
        if self.role not in {"cache-root", "users-root", "user-namespace", "locks", "replicas"}:
            raise ValueError("database cache maintenance descriptor role is invalid")
        if any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in (self.device, self.inode, self.uid, self.mode)
        ):
            raise TypeError("database cache maintenance descriptor integers must be exact")
        if self.device < 0 or self.inode <= 0 or self.uid < 0 or self.mode not in {0o700}:
            raise ValueError("database cache maintenance descriptor identity is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "role": self.role,
            "uid": self.uid,
        }


@dataclass(frozen=True)
class DatabaseCacheIdentityLockObservation:
    """The nonblocking identity-lock outcome for one relevant replica identity."""

    source_manifest_sha256: str
    observation: DatabaseCacheLockObservationKind

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.source_manifest_sha256) is None:
            raise ValueError("database cache maintenance lock identity must be lowercase SHA-256")
        if self.observation not in {"acquired", "contended"}:
            raise ValueError("database cache maintenance lock observation is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "observation": self.observation,
            "source_manifest_sha256": self.source_manifest_sha256,
        }


@dataclass(frozen=True)
class DatabaseCacheIdentityLockSummary:
    """Bounded summary of the complete sorted maintenance lock sequence."""

    total_count: int
    acquired_count: int
    contended_count: int
    observations_sha256: str
    samples: tuple[DatabaseCacheIdentityLockObservation, ...]
    omitted_count: int

    def __post_init__(self) -> None:
        counts = (self.total_count, self.acquired_count, self.contended_count, self.omitted_count)
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in counts):
            raise ValueError("database cache maintenance identity-lock counts are invalid")
        if (
            self.total_count != self.acquired_count + self.contended_count
            or self.contended_count > 1
            or self.omitted_count != self.total_count - len(self.samples)
            or self.omitted_count < 0
            or len(self.samples) > _MAX_LOCK_SAMPLES
            or _SHA256.fullmatch(self.observations_sha256) is None
        ):
            raise ValueError("database cache maintenance identity-lock summary is invalid")
        if any(not isinstance(item, DatabaseCacheIdentityLockObservation) for item in self.samples):
            raise TypeError("database cache maintenance identity-lock samples must be exact records")
        identities = tuple(item.source_manifest_sha256 for item in self.samples)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("database cache maintenance identity-lock samples must be sorted and unique")
        sampled_acquired = sum(item.observation == "acquired" for item in self.samples)
        sampled_contended = sum(item.observation == "contended" for item in self.samples)
        if sampled_acquired > self.acquired_count or sampled_contended > self.contended_count:
            raise ValueError("database cache maintenance identity-lock samples exceed summary counts")
        if self.omitted_count == 0:
            if (
                sampled_acquired != self.acquired_count
                or sampled_contended != self.contended_count
                or self.observations_sha256 != database_cache_identity_lock_observations_digest(self.samples)
            ):
                raise ValueError("database cache maintenance identity-lock summary is inconsistent")
            if self.contended_count and self.samples[-1].observation != "contended":
                raise ValueError("database cache maintenance identity contention must be terminal")

    def to_mapping(self) -> dict[str, object]:
        return {
            "acquired_count": self.acquired_count,
            "contended_count": self.contended_count,
            "observations_sha256": self.observations_sha256,
            "omitted_count": self.omitted_count,
            "samples": [item.to_mapping() for item in self.samples],
            "total_count": self.total_count,
        }


@dataclass(frozen=True)
class DatabaseCacheRemovedEntry:
    """Bounded identity of one exact removed top-level cache entry."""

    kind: DatabaseCacheEntryKind
    source_manifest_sha256: str
    basename: str
    device: int
    inode: int

    def __post_init__(self) -> None:
        if self.kind not in {"replica", "population"}:
            raise ValueError("database cache maintenance removed-entry kind is invalid")
        if _SHA256.fullmatch(self.source_manifest_sha256) is None:
            raise ValueError("database cache maintenance removed-entry identity is invalid")
        expected = (
            self.source_manifest_sha256
            if self.kind == "replica"
            else re.compile(rf"\.population-{self.source_manifest_sha256}-[0-9a-f]{{32}}").fullmatch(self.basename)
        )
        if (self.kind == "replica" and self.basename != expected) or (self.kind == "population" and expected is None):
            raise ValueError("database cache maintenance removed-entry basename is invalid")
        if (
            not isinstance(self.device, int)
            or isinstance(self.device, bool)
            or self.device < 0
            or not isinstance(self.inode, int)
            or isinstance(self.inode, bool)
            or self.inode <= 0
        ):
            raise ValueError("database cache maintenance removed-entry filesystem identity is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "basename": self.basename,
            "device": self.device,
            "inode": self.inode,
            "kind": self.kind,
            "source_manifest_sha256": self.source_manifest_sha256,
        }


@dataclass(frozen=True)
class DatabaseCacheMaintenanceEvidence:
    """Strict bounded terminal record for one explicit maintenance attempt."""

    profile_name: str
    profile_digest: str
    effective_unix_user: str
    effective_uid: int
    cache_root: str
    user_namespace: str
    replicas_scope: str
    descriptors: tuple[DatabaseCacheMaintenanceDescriptor, ...]
    cache_lock_acquired: bool
    cache_lock_contended: bool
    identity_locks: DatabaseCacheIdentityLockSummary
    removed_count: int
    removed_entries_sha256: str
    removed_entry_samples: tuple[DatabaseCacheRemovedEntry, ...]
    omitted_count: int
    terminal_result: DatabaseCacheMaintenanceResult
    diagnostic: str | None
    database_cache_namespace: Literal["acceptance"] = "acceptance"
    lock_protocol: str = _LOCK_PROTOCOL
    kind: str = _EVIDENCE_KIND
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="DatabaseCacheMaintenanceEvidence")
        if self.kind != _EVIDENCE_KIND or self.lock_protocol != _LOCK_PROTOCOL:
            raise ValueError("database cache maintenance evidence protocol is invalid")
        if _PROFILE_NAME.fullmatch(self.profile_name) is None or _SHA256.fullmatch(self.profile_digest) is None:
            raise ValueError("database cache maintenance profile authority is invalid")
        if self.database_cache_namespace != "acceptance":
            raise ValueError("database cache maintenance evidence must bind acceptance")
        if _SAFE_UNIX_USER.fullmatch(self.effective_unix_user) is None:
            raise ValueError("database cache maintenance effective user is invalid")
        if not isinstance(self.effective_uid, int) or isinstance(self.effective_uid, bool) or self.effective_uid < 0:
            raise ValueError("database cache maintenance effective UID is invalid")
        _validate_cache_root(self.cache_root)
        if self.user_namespace != str(Path(self.cache_root) / "users" / self.effective_unix_user):
            raise ValueError("database cache maintenance user namespace is not exact")
        if self.replicas_scope != str(Path(self.user_namespace) / "replicas"):
            raise ValueError("database cache maintenance replicas scope is not exact")
        if any(not isinstance(item, DatabaseCacheMaintenanceDescriptor) for item in self.descriptors):
            raise TypeError("database cache maintenance descriptors must be exact records")
        roles = tuple(item.role for item in self.descriptors)
        scope_prefix = ("cache-root", "users-root", "user-namespace")
        replicas_roles = (*scope_prefix, "replicas")
        locked_roles = (*replicas_roles, "locks")
        if roles not in {(), scope_prefix, replicas_roles, locked_roles}:
            raise ValueError("database cache maintenance descriptor sequence is unreachable")
        if any(item.uid != self.effective_uid for item in self.descriptors):
            raise ValueError("database cache maintenance descriptor UID does not match effective UID")
        scope_device = self.descriptors[0].device if self.descriptors else None
        if scope_device is not None and any(item.device != scope_device for item in self.descriptors):
            raise ValueError("database cache maintenance descriptors must bind one scope device")
        if not isinstance(self.cache_lock_acquired, bool) or not isinstance(self.cache_lock_contended, bool):
            raise TypeError("database cache maintenance cache-lock observations must be boolean")
        if not isinstance(self.identity_locks, DatabaseCacheIdentityLockSummary):
            raise TypeError("database cache maintenance identity-lock summary must be an exact record")
        if any(not isinstance(item, DatabaseCacheRemovedEntry) for item in self.removed_entry_samples):
            raise TypeError("database cache maintenance removed samples must be exact records")
        if self.removed_entry_samples and (
            scope_device is None or any(item.device != scope_device for item in self.removed_entry_samples)
        ):
            raise ValueError("database cache maintenance removed samples must bind the scope device")
        removed_names = tuple(item.basename for item in self.removed_entry_samples)
        if removed_names != tuple(sorted(set(removed_names))):
            raise ValueError("database cache maintenance removed samples must be sorted and unique")
        if len(self.removed_entry_samples) > _MAX_REMOVED_SAMPLES:
            raise ValueError("database cache maintenance evidence samples are unbounded")
        if (
            not isinstance(self.removed_count, int)
            or isinstance(self.removed_count, bool)
            or self.removed_count < 0
            or not isinstance(self.omitted_count, int)
            or isinstance(self.omitted_count, bool)
            or self.omitted_count != self.removed_count - len(self.removed_entry_samples)
            or self.omitted_count < 0
            or _SHA256.fullmatch(self.removed_entries_sha256) is None
        ):
            raise ValueError("database cache maintenance removed-entry summary is invalid")
        if self.terminal_result not in {"cleared", "already-empty", "refused", "failed"}:
            raise ValueError("database cache maintenance terminal result is invalid")
        if (self.terminal_result in {"cleared", "already-empty"}) != (self.diagnostic is None):
            raise ValueError("database cache maintenance diagnostic does not match terminal result")
        if self.diagnostic is not None and not 0 < len(self.diagnostic) <= _MAX_DIAGNOSTIC:
            raise ValueError("database cache maintenance diagnostic is invalid")
        if self.terminal_result == "cleared" and self.removed_count == 0:
            raise ValueError("cleared maintenance evidence must record a removal")
        if self.terminal_result == "already-empty" and self.removed_count != 0:
            raise ValueError("already-empty maintenance evidence cannot record removals")
        if self.terminal_result == "refused" and self.removed_count != 0:
            raise ValueError("refused maintenance evidence cannot record removals")
        if self.omitted_count == 0 and self.removed_entries_sha256 != removed_database_cache_entries_digest(
            self.removed_entry_samples
        ):
            raise ValueError("database cache maintenance removed-entry digest is inconsistent")
        if self.terminal_result == "already-empty":
            if roles not in {scope_prefix, replicas_roles}:
                raise ValueError("already-empty maintenance evidence has invalid scope descriptors")
            if self.cache_lock_acquired or self.cache_lock_contended or self.identity_locks.total_count:
                raise ValueError("already-empty maintenance evidence cannot claim lock ownership")
        if self.terminal_result in {"cleared", "failed"}:
            if roles != locked_roles or not self.cache_lock_acquired:
                raise ValueError("mutating maintenance evidence requires complete held authority")
            if (
                not self.identity_locks.total_count
                or self.identity_locks.acquired_count != self.identity_locks.total_count
                or self.identity_locks.contended_count
            ):
                raise ValueError("mutating maintenance evidence requires every relevant identity lock")
        if self.terminal_result == "refused" and self.cache_lock_acquired:
            if roles != locked_roles:
                raise ValueError("locked refusal evidence requires complete scope descriptors")
            if self.identity_locks.contended_count > 1:
                raise ValueError("refused maintenance lock contention must be terminal")
        if (
            self.terminal_result == "refused"
            and not self.cache_lock_acquired
            and roles not in {(), scope_prefix, replicas_roles}
        ):
            raise ValueError("unlocked refusal evidence has invalid scope descriptors")
        if not self.cache_lock_acquired and self.identity_locks.total_count:
            raise ValueError("identity lock observations require cache ownership")
        acquired_identities = {
            item.source_manifest_sha256 for item in self.identity_locks.samples if item.observation == "acquired"
        }
        if self.identity_locks.omitted_count == 0 and any(
            item.source_manifest_sha256 not in acquired_identities for item in self.removed_entry_samples
        ):
            raise ValueError("removed maintenance samples require matching acquired identity locks")

    @property
    def identity_lock_observations(self) -> tuple[DatabaseCacheIdentityLockObservation, ...]:
        """Compatibility view of the bounded lock-observation samples."""
        return self.identity_locks.samples

    def to_mapping(self) -> dict[str, object]:
        return {
            "database_cache_maintenance": {
                "cache_lock": {
                    "acquired": self.cache_lock_acquired,
                    "contended": self.cache_lock_contended,
                },
                "database_cache_namespace": self.database_cache_namespace,
                "descriptors": [item.to_mapping() for item in self.descriptors],
                "diagnostic": self.diagnostic,
                "effective_uid": self.effective_uid,
                "effective_unix_user": self.effective_unix_user,
                "identity_locks": self.identity_locks.to_mapping(),
                "kind": self.kind,
                "lock_protocol": self.lock_protocol,
                "profile_digest": self.profile_digest,
                "profile_name": self.profile_name,
                "removed": {
                    "count": self.removed_count,
                    "entries_sha256": self.removed_entries_sha256,
                    "omitted_count": self.omitted_count,
                    "samples": [item.to_mapping() for item in self.removed_entry_samples],
                },
                "scope": {
                    "cache_root": self.cache_root,
                    "replicas": self.replicas_scope,
                    "user_namespace": self.user_namespace,
                },
                "schema_version": self.schema_version,
                "terminal_result": self.terminal_result,
            }
        }


def reject_profile_inheritance(raw: Mapping[str, object], *, source: str) -> None:
    """Reject every unsupported profile inheritance spelling at the shared seam."""
    used_keys = sorted(key for key in _INHERITANCE_KEYS if key in raw)
    if used_keys:
        raise ValueError(f"{source} does not support inheritance keys: {', '.join(used_keys)}")


def select_database_acceptance_cache_profile(
    config: Mapping[str, object],
    profile_name: str,
) -> DatabaseAcceptanceCacheProfile:
    """Select one exact acceptance-only maintenance projection from user config."""
    if _PROFILE_NAME.fullmatch(profile_name) is None:
        raise ValueError("database cache maintenance profile selection is invalid")
    clusters = config.get("clusters")
    if not isinstance(clusters, Mapping):
        raise TypeError("database cache maintenance config requires a top-level clusters mapping")
    if profile_name not in clusters:
        raise ValueError(f"unknown database cache maintenance profile: {profile_name!r}")
    mappings: dict[str, Mapping[str, object]] = {}
    for name, value in clusters.items():
        if not isinstance(name, str) or _PROFILE_NAME.fullmatch(name) is None or not isinstance(value, Mapping):
            raise TypeError("database cache maintenance Cluster Profiles must be named mappings")
        reject_profile_inheritance(value, source=f"Cluster Profile {name!r}")
        reject_environment_interpolation(value, context=f"Cluster Profile {name!r}")
        mappings[name] = value
    selected = mappings[profile_name]
    marker = selected.get("database_cache_namespace")
    if marker != "acceptance":
        raise ValueError("database cache maintenance requires an explicitly acceptance-marked profile")
    staging_keys = (
        "database_cache_root",
        "database_cache_unix_user",
        "database_cache_filesystem_type",
        "database_cache_reserve_bytes",
        "database_lock_wait_seconds",
    )
    if any(selected.get(key) is None for key in staging_keys):
        raise ValueError("database cache maintenance requires complete staging authority")
    reserve = selected["database_cache_reserve_bytes"]
    if not isinstance(reserve, int) or isinstance(reserve, bool) or reserve < 0:
        raise ValueError("database cache maintenance reserve bytes must be a non-negative integer")
    cache_root = selected["database_cache_root"]
    unix_user = selected["database_cache_unix_user"]
    filesystem_type = selected["database_cache_filesystem_type"]
    lock_wait = selected["database_lock_wait_seconds"]
    if not all(isinstance(item, str) for item in (cache_root, unix_user, filesystem_type)):
        raise TypeError("database cache maintenance staging strings must be exact")
    if not isinstance(lock_wait, int) or isinstance(lock_wait, bool):
        raise TypeError("database cache maintenance lock wait must be an exact integer")
    assert isinstance(cache_root, str)
    selected_canonical = _canonical_cache_root(cache_root)
    for sibling_name, sibling in mappings.items():
        if sibling_name == profile_name:
            continue
        sibling_root = sibling.get("database_cache_root")
        if sibling_root is None:
            continue
        if not isinstance(sibling_root, str):
            raise TypeError("configured sibling database cache root must be a string")
        sibling_canonical = _canonical_cache_root(sibling_root)
        if _cache_roots_overlap(selected_canonical, sibling_canonical):
            raise ValueError(f"acceptance database cache root aliases configured sibling profile {sibling_name!r}")
    return DatabaseAcceptanceCacheProfile(
        profile_name=profile_name,
        database_cache_namespace="acceptance",
        cache_root=cache_root,
        unix_user=unix_user,  # type: ignore[arg-type]
        expected_filesystem_type=filesystem_type,  # type: ignore[arg-type]
        lock_wait_seconds=lock_wait,
    )


def canonical_database_cache_maintenance_profile_bytes(profile: DatabaseAcceptanceCacheProfile) -> bytes:
    return _canonical_json_bytes(profile.to_mapping())


def canonical_database_cache_maintenance_evidence_bytes(evidence: DatabaseCacheMaintenanceEvidence) -> bytes:
    return _canonical_json_bytes(evidence.to_mapping())


def removed_database_cache_entries_digest(entries: tuple[DatabaseCacheRemovedEntry, ...]) -> str:
    """Digest the full ordered removed-entry sequence without payload inventory."""
    ordered = sorted(
        entries,
        key=lambda item: (
            item.kind,
            item.source_manifest_sha256,
            item.basename,
            item.device,
            item.inode,
        ),
    )
    return hashlib.sha256(
        _canonical_json_bytes({"removed_entries": [item.to_mapping() for item in ordered]})
    ).hexdigest()


def database_cache_identity_lock_observations_digest(
    observations: tuple[DatabaseCacheIdentityLockObservation, ...],
) -> str:
    """Digest every sorted lock observation while publishing only bounded samples."""
    ordered = sorted(observations, key=lambda item: (item.source_manifest_sha256, item.observation))
    return hashlib.sha256(
        _canonical_json_bytes({"identity_lock_observations": [item.to_mapping() for item in ordered]})
    ).hexdigest()


def empty_database_cache_entries_digest() -> str:
    return removed_database_cache_entries_digest(())


def load_database_cache_maintenance_evidence(
    path: Path,
    *,
    dir_fd: int | None = None,
) -> DatabaseCacheMaintenanceEvidence:
    """Load one exact canonical immutable maintenance record without following links."""
    if dir_fd is not None and (path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}):
        raise ValueError("descriptor-relative maintenance evidence path must be one exact basename")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError("database cache maintenance evidence authority is invalid")
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise TypeError("database cache maintenance evidence must be a mapping")
        evidence = database_cache_maintenance_evidence_from_mapping(payload)
        if raw != canonical_database_cache_maintenance_evidence_bytes(evidence):
            raise ValueError("database cache maintenance evidence must contain exact canonical bytes")
        return evidence
    finally:
        if descriptor is not None:
            os.close(descriptor)


def database_cache_maintenance_evidence_from_mapping(
    payload: Mapping[str, object],
) -> DatabaseCacheMaintenanceEvidence:
    _require_fields(payload, {"database_cache_maintenance"}, "maintenance envelope")
    body = _mapping(payload["database_cache_maintenance"], "maintenance body")
    _require_fields(
        body,
        {
            "cache_lock",
            "database_cache_namespace",
            "descriptors",
            "diagnostic",
            "effective_uid",
            "effective_unix_user",
            "identity_locks",
            "kind",
            "lock_protocol",
            "profile_digest",
            "profile_name",
            "removed",
            "schema_version",
            "scope",
            "terminal_result",
        },
        "maintenance body",
    )
    cache_lock = _mapping(body["cache_lock"], "cache lock")
    _require_fields(cache_lock, {"acquired", "contended"}, "cache lock")
    scope = _mapping(body["scope"], "maintenance scope")
    _require_fields(scope, {"cache_root", "replicas", "user_namespace"}, "maintenance scope")
    removed = _mapping(body["removed"], "removed summary")
    _require_fields(removed, {"count", "entries_sha256", "omitted_count", "samples"}, "removed summary")
    return DatabaseCacheMaintenanceEvidence(
        profile_name=_string(body["profile_name"], "profile name"),
        profile_digest=_string(body["profile_digest"], "profile digest"),
        effective_unix_user=_string(body["effective_unix_user"], "effective Unix user"),
        effective_uid=_integer(body["effective_uid"], "effective UID"),
        cache_root=_string(scope["cache_root"], "cache root"),
        user_namespace=_string(scope["user_namespace"], "user namespace"),
        replicas_scope=_string(scope["replicas"], "replicas scope"),
        descriptors=tuple(_descriptor_from_mapping(item) for item in _sequence(body["descriptors"], "descriptors")),
        cache_lock_acquired=_boolean(cache_lock["acquired"], "cache lock acquired"),
        cache_lock_contended=_boolean(cache_lock["contended"], "cache lock contended"),
        identity_locks=_lock_summary_from_mapping(body["identity_locks"]),
        removed_count=_integer(removed["count"], "removed count"),
        removed_entries_sha256=_string(removed["entries_sha256"], "removed entries digest"),
        removed_entry_samples=tuple(
            _removed_entry_from_mapping(item) for item in _sequence(removed["samples"], "removed samples")
        ),
        omitted_count=_integer(removed["omitted_count"], "removed omitted count"),
        terminal_result=_string(body["terminal_result"], "terminal result"),  # type: ignore[arg-type]
        diagnostic=_optional_string(body["diagnostic"], "diagnostic"),
        database_cache_namespace=_string(body["database_cache_namespace"], "namespace"),  # type: ignore[arg-type]
        lock_protocol=_string(body["lock_protocol"], "lock protocol"),
        kind=_string(body["kind"], "kind"),
        schema_version=_integer(body["schema_version"], "schema version"),
    )


def _descriptor_from_mapping(value: object) -> DatabaseCacheMaintenanceDescriptor:
    item = _mapping(value, "descriptor")
    _require_fields(item, {"device", "inode", "mode", "role", "uid"}, "descriptor")
    return DatabaseCacheMaintenanceDescriptor(
        role=_string(item["role"], "descriptor role"),  # type: ignore[arg-type]
        device=_integer(item["device"], "descriptor device"),
        inode=_integer(item["inode"], "descriptor inode"),
        uid=_integer(item["uid"], "descriptor uid"),
        mode=_integer(item["mode"], "descriptor mode"),
    )


def _lock_observation_from_mapping(value: object) -> DatabaseCacheIdentityLockObservation:
    item = _mapping(value, "identity lock observation")
    _require_fields(item, {"observation", "source_manifest_sha256"}, "identity lock observation")
    return DatabaseCacheIdentityLockObservation(
        source_manifest_sha256=_string(item["source_manifest_sha256"], "identity lock identity"),
        observation=_string(item["observation"], "identity lock observation"),  # type: ignore[arg-type]
    )


def _lock_summary_from_mapping(value: object) -> DatabaseCacheIdentityLockSummary:
    item = _mapping(value, "identity lock summary")
    _require_fields(
        item,
        {
            "acquired_count",
            "contended_count",
            "observations_sha256",
            "omitted_count",
            "samples",
            "total_count",
        },
        "identity lock summary",
    )
    return DatabaseCacheIdentityLockSummary(
        total_count=_integer(item["total_count"], "identity lock total count"),
        acquired_count=_integer(item["acquired_count"], "identity lock acquired count"),
        contended_count=_integer(item["contended_count"], "identity lock contended count"),
        observations_sha256=_string(item["observations_sha256"], "identity lock observations digest"),
        samples=tuple(
            _lock_observation_from_mapping(sample) for sample in _sequence(item["samples"], "identity lock samples")
        ),
        omitted_count=_integer(item["omitted_count"], "identity lock omitted count"),
    )


def _removed_entry_from_mapping(value: object) -> DatabaseCacheRemovedEntry:
    item = _mapping(value, "removed entry")
    _require_fields(item, {"basename", "device", "inode", "kind", "source_manifest_sha256"}, "removed entry")
    return DatabaseCacheRemovedEntry(
        kind=_string(item["kind"], "removed entry kind"),  # type: ignore[arg-type]
        source_manifest_sha256=_string(item["source_manifest_sha256"], "removed entry identity"),
        basename=_string(item["basename"], "removed entry basename"),
        device=_integer(item["device"], "removed entry device"),
        inode=_integer(item["inode"], "removed entry inode"),
    )


def _validate_cache_root(value: str) -> None:
    canonical = _canonical_cache_root(value)
    path = Path(canonical)
    if len(path.parts) < 4:
        raise ValueError("database cache maintenance cache root is shallow or broad")


def _canonical_cache_root(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("database cache maintenance cache root must be specified")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("database cache maintenance cache root must be absolute")
    canonical = os.path.normpath(value)
    if canonical != value or any(part in {".", ".."} for part in path.parts):
        raise ValueError("database cache maintenance cache root must not normalize or escape")
    if canonical == "/":
        raise ValueError("database cache maintenance cache root is broad")
    return canonical


def _cache_roots_overlap(left: str, right: str) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    return left_path == right_path or left_path in right_path.parents or right_path in left_path.parents


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return value


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON array")
    return tuple(value)


def _require_fields(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


__all__ = [
    "DatabaseAcceptanceCacheProfile",
    "DatabaseCacheIdentityLockObservation",
    "DatabaseCacheIdentityLockSummary",
    "DatabaseCacheMaintenanceDescriptor",
    "DatabaseCacheMaintenanceEvidence",
    "DatabaseCacheRemovedEntry",
    "canonical_database_cache_maintenance_evidence_bytes",
    "canonical_database_cache_maintenance_profile_bytes",
    "database_cache_identity_lock_observations_digest",
    "database_cache_maintenance_evidence_from_mapping",
    "empty_database_cache_entries_digest",
    "load_database_cache_maintenance_evidence",
    "reject_profile_inheritance",
    "removed_database_cache_entries_digest",
    "select_database_acceptance_cache_profile",
]
