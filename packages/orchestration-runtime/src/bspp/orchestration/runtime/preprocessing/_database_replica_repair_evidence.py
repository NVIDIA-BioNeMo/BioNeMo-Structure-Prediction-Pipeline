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

"""Runtime-private immutable evidence for Database Replica repair actions."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ._database_placement_errors import DatabasePlacementError
from ._database_placement_evidence_io import (
    _ImmutableEvidencePublicationCollisionError,
    _load_canonical_json_at,
    _load_from_path,
    _publish_immutable_exclusive,
)
from ._database_replica_errors import ClassifiedDatabaseReplicaError

_INVALIDATION_KIND = "database-replica-invalidation-v1"
_CLEANUP_KIND = "database-replica-stale-population-cleanup-v1"
_LOCK_PROTOCOL = "identity-exclusive-then-cache-exclusive-v1"
_OWNERLESS_PROOF = "identity-exclusive-and-cache-exclusive-v1"


@dataclass(frozen=True)
class RepairActionAuthority:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_id: str
    database_set_identifier: str
    database_set_version: str
    source_manifest_sha256: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "attempt_id": self.attempt_id,
            "database_set": {
                "identifier": self.database_set_identifier,
                "version": self.database_set_version,
            },
            "phase_run_id": self.phase_run_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "requested_policy": "stage-required",
            "source_manifest_sha256": self.source_manifest_sha256,
        }


@dataclass(frozen=True)
class DatabaseReplicaInvalidationEvidence:
    authority: RepairActionAuthority
    original_replica_device: int
    original_replica_inode: int
    validation_error: str
    retired_population_basename: str
    visible_final_absent: bool = True
    science_started: bool = False
    lock_protocol: str = _LOCK_PROTOCOL

    def to_mapping(self) -> dict[str, object]:
        return {
            "database_replica_invalidation": {
                "authority": self.authority.to_mapping(),
                "kind": _INVALIDATION_KIND,
                "lock_protocol": self.lock_protocol,
                "original_replica": {
                    "device": self.original_replica_device,
                    "inode": self.original_replica_inode,
                },
                "retired_population_basename": self.retired_population_basename,
                "science_started": self.science_started,
                "validation_error": self.validation_error,
                "visible_final_absent": self.visible_final_absent,
            }
        }


@dataclass(frozen=True)
class DatabaseReplicaCleanupEvidence:
    authority: RepairActionAuthority
    removed_count: int
    removed_basenames_sha256: str
    removed_basename_samples: tuple[str, ...]
    omitted_count: int
    failure: str | None
    ownerless_proof: str = _OWNERLESS_PROOF

    def to_mapping(self) -> dict[str, object]:
        return {
            "database_replica_stale_population_cleanup": {
                "authority": self.authority.to_mapping(),
                "failure": self.failure,
                "kind": _CLEANUP_KIND,
                "omitted_count": self.omitted_count,
                "ownerless_proof": self.ownerless_proof,
                "removed_basename_samples": list(self.removed_basename_samples),
                "removed_basenames_sha256": self.removed_basenames_sha256,
                "removed_count": self.removed_count,
            }
        }


def removed_basenames_digest(names: tuple[str, ...]) -> str:
    """Hash the full sorted cleanup inventory without retaining an unbounded list."""
    return hashlib.sha256("".join(f"{name}\n" for name in names).encode()).hexdigest()


def publish_database_replica_invalidation(
    evidence: DatabaseReplicaInvalidationEvidence,
    destination: Path,
) -> None:
    _publish_or_accept_exact(
        evidence,
        destination,
        serializer=_canonical_invalidation_bytes,
        loader=_load_invalidation_at,
        description="Database Replica invalidation evidence",
    )


def publish_database_replica_cleanup(
    evidence: DatabaseReplicaCleanupEvidence,
    destination: Path,
) -> None:
    _publish_or_accept_exact(
        evidence,
        destination,
        serializer=_canonical_cleanup_bytes,
        loader=_load_cleanup_at,
        description="Database Replica stale-population cleanup evidence",
    )


def load_database_replica_invalidation(path: Path) -> DatabaseReplicaInvalidationEvidence:
    return _load_from_path(path, description="Database Replica invalidation evidence", loader=_load_invalidation_at)


def load_database_replica_cleanup(path: Path) -> DatabaseReplicaCleanupEvidence:
    return _load_from_path(
        path,
        description="Database Replica stale-population cleanup evidence",
        loader=_load_cleanup_at,
    )


def _publish_or_accept_exact[T](
    evidence: T,
    destination: Path,
    *,
    serializer: Callable[[T], bytes],
    loader: Callable[[int, str, Path], T],
    description: str,
) -> None:
    try:
        _publish_immutable_exclusive(
            evidence,
            destination,
            serializer=serializer,
            loader=loader,
            description=description,
        )
    except _ImmutableEvidencePublicationCollisionError as exc:
        if exc.exact_match:
            return
        raise ClassifiedDatabaseReplicaError(
            "publication-failed",
            f"{description} collision is not an exact canonical match",
        ) from exc
    except DatabasePlacementError as exc:
        raise ClassifiedDatabaseReplicaError(
            "publication-failed",
            f"{description} publication failed",
        ) from exc


def _canonical_invalidation_bytes(evidence: DatabaseReplicaInvalidationEvidence) -> bytes:
    return _canonical_json_bytes(evidence.to_mapping())


def _canonical_cleanup_bytes(evidence: DatabaseReplicaCleanupEvidence) -> bytes:
    return _canonical_json_bytes(evidence.to_mapping())


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _load_invalidation_at(
    parent_descriptor: int,
    name: str,
    display_path: Path,
) -> DatabaseReplicaInvalidationEvidence:
    return _load_canonical_json_at(
        parent_descriptor,
        name,
        display_path,
        parser=_invalidation_from_mapping,
        serializer=_canonical_invalidation_bytes,
        description="Database Replica invalidation evidence",
    )


def _load_cleanup_at(parent_descriptor: int, name: str, display_path: Path) -> DatabaseReplicaCleanupEvidence:
    return _load_canonical_json_at(
        parent_descriptor,
        name,
        display_path,
        parser=_cleanup_from_mapping,
        serializer=_canonical_cleanup_bytes,
        description="Database Replica stale-population cleanup evidence",
    )


def _invalidation_from_mapping(payload: Mapping[str, object]) -> DatabaseReplicaInvalidationEvidence:
    _require_fields(payload, {"database_replica_invalidation"}, "invalidation envelope")
    body = _mapping(payload["database_replica_invalidation"], "invalidation body")
    _require_fields(
        body,
        {
            "authority",
            "kind",
            "lock_protocol",
            "original_replica",
            "retired_population_basename",
            "science_started",
            "validation_error",
            "visible_final_absent",
        },
        "invalidation body",
    )
    original = _mapping(body["original_replica"], "original replica")
    _require_fields(original, {"device", "inode"}, "original replica")
    evidence = DatabaseReplicaInvalidationEvidence(
        authority=_authority_from_mapping(_mapping(body["authority"], "repair authority")),
        original_replica_device=_integer(original["device"], "original replica device"),
        original_replica_inode=_integer(original["inode"], "original replica inode"),
        validation_error=_string(body["validation_error"], "validation error"),
        retired_population_basename=_string(body["retired_population_basename"], "retired basename"),
        visible_final_absent=_boolean(body["visible_final_absent"], "visible final absent"),
        science_started=_boolean(body["science_started"], "science started"),
        lock_protocol=_string(body["lock_protocol"], "lock protocol"),
    )
    if body["kind"] != _INVALIDATION_KIND or evidence.lock_protocol != _LOCK_PROTOCOL:
        raise ValueError("Database Replica invalidation protocol is invalid")
    expected_retired = re.fullmatch(
        rf"\.population-{re.escape(evidence.authority.source_manifest_sha256)}-[0-9a-f]{{32}}",
        evidence.retired_population_basename,
    )
    if (
        not evidence.visible_final_absent
        or evidence.science_started
        or evidence.original_replica_device < 0
        or evidence.original_replica_inode <= 0
        or not 0 < len(evidence.validation_error) <= 2048
        or expected_retired is None
    ):
        raise ValueError("Database Replica invalidation terminal facts are invalid")
    return evidence


def _cleanup_from_mapping(payload: Mapping[str, object]) -> DatabaseReplicaCleanupEvidence:
    _require_fields(payload, {"database_replica_stale_population_cleanup"}, "cleanup envelope")
    body = _mapping(payload["database_replica_stale_population_cleanup"], "cleanup body")
    _require_fields(
        body,
        {
            "authority",
            "failure",
            "kind",
            "omitted_count",
            "ownerless_proof",
            "removed_basename_samples",
            "removed_basenames_sha256",
            "removed_count",
        },
        "cleanup body",
    )
    samples_value = body["removed_basename_samples"]
    if not isinstance(samples_value, list) or not all(isinstance(value, str) for value in samples_value):
        raise ValueError("cleanup basename samples are invalid")
    failure_value = body["failure"]
    if failure_value is not None and not isinstance(failure_value, str):
        raise ValueError("cleanup failure is invalid")
    evidence = DatabaseReplicaCleanupEvidence(
        authority=_authority_from_mapping(_mapping(body["authority"], "repair authority")),
        removed_count=_integer(body["removed_count"], "removed count"),
        removed_basenames_sha256=_string(body["removed_basenames_sha256"], "removed basename digest"),
        removed_basename_samples=tuple(samples_value),
        omitted_count=_integer(body["omitted_count"], "omitted count"),
        failure=failure_value,
        ownerless_proof=_string(body["ownerless_proof"], "ownerless proof"),
    )
    if body["kind"] != _CLEANUP_KIND or evidence.ownerless_proof != _OWNERLESS_PROOF:
        raise ValueError("Database Replica cleanup protocol is invalid")
    expected_name = re.compile(rf"\.population-{re.escape(evidence.authority.source_manifest_sha256)}-[0-9a-f]{{32}}")
    expected_sample_count = min(evidence.removed_count, 32)
    expected_omitted_count = max(0, evidence.removed_count - 32)
    if (
        evidence.removed_count < 0
        or evidence.omitted_count < 0
        or (evidence.removed_count == 0 and evidence.failure is None)
        or len(evidence.removed_basename_samples) != expected_sample_count
        or evidence.omitted_count != expected_omitted_count
        or evidence.removed_count != len(evidence.removed_basename_samples) + evidence.omitted_count
        or evidence.removed_basename_samples != tuple(sorted(set(evidence.removed_basename_samples)))
        or any(expected_name.fullmatch(name) is None for name in evidence.removed_basename_samples)
        or re.fullmatch(r"[0-9a-f]{64}", evidence.removed_basenames_sha256) is None
        or (
            evidence.omitted_count == 0
            and removed_basenames_digest(evidence.removed_basename_samples) != evidence.removed_basenames_sha256
        )
        or (evidence.failure is not None and not 0 < len(evidence.failure) <= 2048)
    ):
        raise ValueError("Database Replica cleanup bounds are invalid")
    return evidence


def _authority_from_mapping(payload: Mapping[str, object]) -> RepairActionAuthority:
    _require_fields(
        payload,
        {
            "action_id",
            "attempt_id",
            "database_set",
            "phase_run_id",
            "phase_runspec_digest",
            "requested_policy",
            "source_manifest_sha256",
        },
        "repair authority",
    )
    database_set = _mapping(payload["database_set"], "repair database set")
    _require_fields(database_set, {"identifier", "version"}, "repair database set")
    if payload["requested_policy"] != "stage-required":
        raise ValueError("repair requested policy is invalid")
    authority = RepairActionAuthority(
        phase_run_id=_string(payload["phase_run_id"], "phase run id"),
        attempt_id=_string(payload["attempt_id"], "attempt id"),
        phase_runspec_digest=_string(payload["phase_runspec_digest"], "RunSpec digest"),
        action_id=_string(payload["action_id"], "action id"),
        database_set_identifier=_string(database_set["identifier"], "database set identifier"),
        database_set_version=_string(database_set["version"], "database set version"),
        source_manifest_sha256=_string(payload["source_manifest_sha256"], "source manifest digest"),
    )
    if (
        not authority.phase_run_id
        or not authority.attempt_id
        or not authority.action_id
        or not authority.database_set_identifier
        or not authority.database_set_version
        or re.fullmatch(r"[0-9a-f]{64}", authority.phase_runspec_digest) is None
        or re.fullmatch(r"[0-9a-f]{64}", authority.source_manifest_sha256) is None
    ):
        raise ValueError("repair authority values are invalid")
    return authority


def _require_fields(payload: Mapping[str, object], expected: set[str], description: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{description} fields are invalid")


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{description} must be a mapping")
    return value


def _string(value: object, description: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{description} must be a string")
    return value


def _integer(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{description} must be an integer")
    return value


def _boolean(value: object, description: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{description} must be a boolean")
    return value


__all__ = [
    "DatabaseReplicaCleanupEvidence",
    "DatabaseReplicaInvalidationEvidence",
    "RepairActionAuthority",
    "load_database_replica_cleanup",
    "load_database_replica_invalidation",
    "publish_database_replica_cleanup",
    "publish_database_replica_invalidation",
    "removed_basenames_digest",
]
