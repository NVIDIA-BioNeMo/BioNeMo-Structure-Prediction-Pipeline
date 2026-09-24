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

"""Strict stage-preferred direct-capacity-fallback Result authority."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, cast

from bspp.orchestration.contract._database_validation import (
    require_fields as _require_fields,
)
from bspp.orchestration.contract._database_validation import (
    required_mapping as _required_mapping,
)
from bspp.orchestration.contract._database_validation import required_nonempty_str as _required_str
from bspp.orchestration.contract._database_validation import validate_schema as _validate_schema
from bspp.orchestration.contract.database_placement import (
    DATABASE_SOURCE_ROOT,
    LEGACY_DATABASE_SOURCE_ROOT,
    LEGACY_SELECTED_DATABASE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
)
from bspp.orchestration.contract.database_placement_result import (
    DatabaseSourceMountFacts,
    DatabaseSourceObservation,
)
from bspp.orchestration.contract.database_replica_facts import (
    DatabaseCacheMountFacts,
    DatabaseCapacityGate,
    _cache_mount_from_mapping,
    _capacity_from_mapping,
    _database_set_from_mapping,
    _source_mount_from_mapping,
    _source_observation_from_mapping,
    _unwrap_exact,
)
from bspp.orchestration.contract.database_set_provisioning import DatabaseSetIdentity
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ACTION_ID = re.compile(r"preprocessing-chunk-[0-9]{6}")


@dataclass(frozen=True)
class DatabaseCapacityFallbackResult:
    """Exclusive successful authority for one positive capacity fallback."""

    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_id: str
    database_set: DatabaseSetIdentity
    requested_policy: DatabaseAccessPolicy
    source_manifest_sha256: str
    branch_kind: Literal["direct-capacity-fallback"]
    outcome: DatabasePlacementOutcomeKind
    selected_container_root: str
    verification: Literal["metadata-verified"]
    source_mount: DatabaseSourceMountFacts
    cache_mount: DatabaseCacheMountFacts
    capacity_gate: DatabaseCapacityGate
    pre_science_observation: DatabaseSourceObservation
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseCapacityFallbackResult")
        if _PHASE_RUN_ID.fullmatch(self.phase_run_id) is None:
            raise ValueError("capacity fallback phase_run_id must be an opaque phase-run id")
        if _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise ValueError("capacity fallback attempt_id must use attempt-NNNN")
        if _SHA256.fullmatch(self.phase_runspec_digest) is None:
            raise ValueError("capacity fallback phase_runspec_digest must be lowercase SHA-256")
        if _ACTION_ID.fullmatch(self.action_id) is None:
            raise ValueError("capacity fallback action_id must identify a preprocessing chunk")
        if not isinstance(self.database_set, DatabaseSetIdentity):
            raise ValueError("capacity fallback requires an exact Database Set identity")
        if self.requested_policy != DatabaseAccessPolicy.STAGE_PREFERRED:
            raise ValueError("capacity fallback Result supports stage-preferred policy only")
        if _SHA256.fullmatch(self.source_manifest_sha256) is None:
            raise ValueError("capacity fallback source-manifest digest must be lowercase SHA-256")
        if (
            self.branch_kind != "direct-capacity-fallback"
            or self.outcome != DatabasePlacementOutcomeKind.DIRECT_CAPACITY_FALLBACK
        ):
            raise ValueError("capacity fallback Result requires the direct-capacity-fallback branch")
        if self.selected_container_root not in (SELECTED_DATABASE_ROOT, LEGACY_SELECTED_DATABASE_ROOT):
            raise ValueError(f"capacity fallback selected root must be {SELECTED_DATABASE_ROOT!r}")
        if self.verification != "metadata-verified":
            raise ValueError("capacity fallback verification must be 'metadata-verified'")
        if not isinstance(self.source_mount, DatabaseSourceMountFacts):
            raise ValueError("capacity fallback requires exact source mount facts")
        source_root = PurePosixPath(DATABASE_SOURCE_ROOT)
        legacy_source_root = PurePosixPath(LEGACY_DATABASE_SOURCE_ROOT)
        mount_point = PurePosixPath(self.source_mount.mount_point)
        if (mount_point != source_root and mount_point not in source_root.parents) and (
            mount_point != legacy_source_root and mount_point not in legacy_source_root.parents
        ):
            raise ValueError("capacity fallback source mount must contain the protected source root")
        if not isinstance(self.cache_mount, DatabaseCacheMountFacts):
            raise ValueError("capacity fallback requires exact cache mount facts")
        if self.source_mount.mount_id == self.cache_mount.mount_id or (
            self.source_mount.device_major,
            self.source_mount.device_minor,
        ) == (self.cache_mount.device_major, self.cache_mount.device_minor):
            raise ValueError("capacity fallback source and cache must be different mounted filesystems")
        if not isinstance(self.capacity_gate, DatabaseCapacityGate) or self.capacity_gate.decision != "insufficient":
            raise ValueError("capacity fallback requires an insufficient completed capacity gate")
        if not isinstance(self.pre_science_observation, DatabaseSourceObservation):
            raise ValueError("capacity fallback requires one exact pre-science observation")
        if self.pre_science_observation.source_manifest_sha256 != self.source_manifest_sha256:
            raise ValueError("capacity fallback observation must bind the same source-manifest digest")
        unique_payload_metadata: dict[str, tuple[int, int]] = {}
        for member in self.pre_science_observation.members:
            identity = (member.size_bytes, member.mtime_ns)
            prior_identity = unique_payload_metadata.get(member.resolved_path)
            if prior_identity is not None and prior_identity != identity:
                raise ValueError(
                    f"capacity fallback aliases disagree about resolved source metadata: {member.resolved_path!r}"
                )
            unique_payload_metadata[member.resolved_path] = identity
        unique_payload_bytes = sum(size_bytes for size_bytes, _mtime_ns in unique_payload_metadata.values())
        if self.capacity_gate.allocated_replica_bytes != unique_payload_bytes:
            raise ValueError("capacity fallback allocation must equal the unique observed payload bytes")

    def to_mapping(self) -> dict[str, object]:
        return {
            "database_capacity_fallback_result": {
                "schema_version": self.schema_version,
                "phase_run_id": self.phase_run_id,
                "attempt_id": self.attempt_id,
                "phase_runspec_digest": self.phase_runspec_digest,
                "action_id": self.action_id,
                "database_set": self.database_set.to_mapping(),
                "requested_policy": self.requested_policy.value,
                "source_manifest_sha256": self.source_manifest_sha256,
                "branch_kind": self.branch_kind,
                "outcome": self.outcome.value,
                "selected_container_root": self.selected_container_root,
                "verification": self.verification,
                "source_mount": self.source_mount.to_mapping(),
                "cache_mount": self.cache_mount.to_mapping(),
                "capacity_gate": self.capacity_gate.to_mapping(),
                "pre_science_observation": self.pre_science_observation.to_mapping(),
            }
        }


def database_capacity_fallback_result_from_mapping(
    payload: Mapping[str, object],
) -> DatabaseCapacityFallbackResult:
    """Strict-load the sole direct-capacity-fallback Result shape."""
    inner = _unwrap_exact(payload, "database_capacity_fallback_result")
    _require_fields(
        inner,
        {
            "schema_version",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "action_id",
            "database_set",
            "requested_policy",
            "source_manifest_sha256",
            "branch_kind",
            "outcome",
            "selected_container_root",
            "verification",
            "source_mount",
            "cache_mount",
            "capacity_gate",
            "pre_science_observation",
        },
        "Database capacity fallback Result",
    )
    if _required_str(inner, "requested_policy") != DatabaseAccessPolicy.STAGE_PREFERRED.value:
        raise ValueError("capacity fallback Result requested_policy must be stage-preferred")
    if _required_str(inner, "branch_kind") != "direct-capacity-fallback":
        raise ValueError("capacity fallback Result branch must be direct-capacity-fallback")
    if _required_str(inner, "outcome") != DatabasePlacementOutcomeKind.DIRECT_CAPACITY_FALLBACK.value:
        raise ValueError("capacity fallback Result outcome must be direct-capacity-fallback")
    if _required_str(inner, "verification") != "metadata-verified":
        raise ValueError("capacity fallback Result verification must be metadata-verified")
    return DatabaseCapacityFallbackResult(
        phase_run_id=_required_str(inner, "phase_run_id"),
        attempt_id=_required_str(inner, "attempt_id"),
        phase_runspec_digest=_required_str(inner, "phase_runspec_digest"),
        action_id=_required_str(inner, "action_id"),
        database_set=_database_set_from_mapping(_required_mapping(inner, "database_set")),
        requested_policy=DatabaseAccessPolicy.STAGE_PREFERRED,
        source_manifest_sha256=_required_str(inner, "source_manifest_sha256"),
        branch_kind="direct-capacity-fallback",
        outcome=DatabasePlacementOutcomeKind.DIRECT_CAPACITY_FALLBACK,
        selected_container_root=_required_str(inner, "selected_container_root"),
        verification=cast("Literal['metadata-verified']", _required_str(inner, "verification")),
        source_mount=_source_mount_from_mapping(_required_mapping(inner, "source_mount")),
        cache_mount=_cache_mount_from_mapping(_required_mapping(inner, "cache_mount")),
        capacity_gate=_capacity_from_mapping(_required_mapping(inner, "capacity_gate")),
        pre_science_observation=_source_observation_from_mapping(_required_mapping(inner, "pre_science_observation")),
        schema_version=validate_schema_version(
            inner.get("schema_version"), record_name="DatabaseCapacityFallbackResult"
        ),
    )


def canonical_database_capacity_fallback_result_bytes(result: DatabaseCapacityFallbackResult) -> bytes:
    return (json.dumps(result.to_mapping(), indent=2, sort_keys=True) + "\n").encode()


def database_capacity_fallback_result_digest(result: DatabaseCapacityFallbackResult) -> str:
    return hashlib.sha256(canonical_database_capacity_fallback_result_bytes(result)).hexdigest()


__all__ = [
    "DatabaseCapacityFallbackResult",
    "canonical_database_capacity_fallback_result_bytes",
    "database_capacity_fallback_result_digest",
    "database_capacity_fallback_result_from_mapping",
]
