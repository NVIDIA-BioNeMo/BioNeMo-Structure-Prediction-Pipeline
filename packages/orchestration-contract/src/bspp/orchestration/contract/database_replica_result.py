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

"""Strict cold Database Replica Result and bounded failure authority."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
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
    DATABASE_CACHE_ROOT,
    LEGACY_DATABASE_CACHE_ROOT,
    LEGACY_SELECTED_DATABASE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
)
from bspp.orchestration.contract.database_placement_result import DatabaseSourceMountFacts
from bspp.orchestration.contract.database_replica_facts import (
    DatabaseCacheMountFacts,
    DatabaseCapacityGate,
    DatabaseReplicaCopyEvidence,
    DatabaseWarmCacheMountFacts,
    _cache_mount_from_mapping,
    _capacity_from_mapping,
    _copy_evidence_from_mapping,
    _database_set_from_mapping,
    _source_mount_from_mapping,
    _unwrap_exact,
    _validate_sha256,
    database_warm_cache_mount_facts_from_mapping,
)
from bspp.orchestration.contract.database_set_provisioning import DatabaseSetIdentity
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

DatabaseReplicaColdFailureClassification = Literal[
    "authority-invalid",
    "source-manifest-invalid",
    "source-mount-invalid",
    "source-inventory-invalid",
    "cache-authority-invalid",
    "capacity-observation-failed",
    "insufficient-capacity",
    "lock-unavailable",
    "copy-failed",
    "hard-links-unsupported",
    "replica-validation-failed",
    "publication-failed",
    "result-publication-failed",
]


_SHA256 = re.compile(r"[0-9a-f]{64}")


_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")


_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")


_ACTION_ID = re.compile(r"preprocessing-chunk-[0-9]{6}")


_COLD_FAILURE_CLASSIFICATIONS = frozenset(
    {
        "authority-invalid",
        "source-manifest-invalid",
        "source-mount-invalid",
        "source-inventory-invalid",
        "cache-authority-invalid",
        "capacity-observation-failed",
        "insufficient-capacity",
        "lock-unavailable",
        "copy-failed",
        "hard-links-unsupported",
        "replica-validation-failed",
        "publication-failed",
        "result-publication-failed",
    }
)


@dataclass(frozen=True)
class DatabaseReplicaColdResult:
    """Exclusive successful authority for one newly populated cold replica."""

    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_id: str
    database_set: DatabaseSetIdentity
    requested_policy: DatabaseAccessPolicy
    source_manifest_sha256: str
    branch_kind: Literal["staged"]
    outcome: DatabasePlacementOutcomeKind
    selected_container_root: str
    replica_container_root: str
    verification: Literal["metadata-verified"]
    source_mount: DatabaseSourceMountFacts
    cache_mount: DatabaseCacheMountFacts
    capacity_gate: DatabaseCapacityGate
    replica_manifest_sha256: str
    copy_evidence: DatabaseReplicaCopyEvidence
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseReplicaColdResult")
        if _PHASE_RUN_ID.fullmatch(self.phase_run_id) is None:
            raise ValueError("cold replica phase_run_id must be an opaque phase-run id")
        if _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise ValueError("cold replica attempt_id must use attempt-NNNN")
        _validate_sha256(self.phase_runspec_digest, "cold replica Phase RunSpec digest")
        if _ACTION_ID.fullmatch(self.action_id) is None:
            raise ValueError("cold replica action_id must identify a preprocessing chunk")
        if not isinstance(self.database_set, DatabaseSetIdentity):
            raise ValueError("cold replica Result requires an exact Database Set identity")
        if self.requested_policy not in {
            DatabaseAccessPolicy.STAGE_REQUIRED,
            DatabaseAccessPolicy.STAGE_PREFERRED,
        }:
            raise ValueError("cold replica Result supports a staging-capable policy only")
        _validate_sha256(self.source_manifest_sha256, "cold replica source-manifest digest")
        if self.branch_kind != "staged" or self.outcome != DatabasePlacementOutcomeKind.REPLICA_COLD:
            raise ValueError("cold replica Result requires the staged replica-cold branch")
        if self.selected_container_root not in (SELECTED_DATABASE_ROOT, LEGACY_SELECTED_DATABASE_ROOT):
            raise ValueError(f"cold replica selected root must be {SELECTED_DATABASE_ROOT!r}")
        expected_replica = f"{DATABASE_CACHE_ROOT}/replicas/{self.source_manifest_sha256}"
        legacy_expected_replica = f"{LEGACY_DATABASE_CACHE_ROOT}/replicas/{self.source_manifest_sha256}"
        if self.replica_container_root not in (expected_replica, legacy_expected_replica):
            raise ValueError("cold replica Result must use the manifest-addressed cache path")
        if self.verification != "metadata-verified":
            raise ValueError("cold replica Result verification must be 'metadata-verified'")
        if not isinstance(self.source_mount, DatabaseSourceMountFacts):
            raise ValueError("cold replica Result requires exact source mount facts")
        if not isinstance(self.cache_mount, DatabaseCacheMountFacts):
            raise ValueError("cold replica Result requires exact cache mount facts")
        if (self.source_mount.device_major, self.source_mount.device_minor) == (
            self.cache_mount.device_major,
            self.cache_mount.device_minor,
        ):
            raise ValueError("cold replica source and cache must be different mounted filesystems")
        if not isinstance(self.capacity_gate, DatabaseCapacityGate) or self.capacity_gate.decision != "sufficient":
            raise ValueError("cold replica Result requires a sufficient completed capacity gate")
        _validate_sha256(self.replica_manifest_sha256, "Database Replica Manifest digest")
        if not isinstance(self.copy_evidence, DatabaseReplicaCopyEvidence):
            raise ValueError("cold replica Result requires exact copy evidence")
        if self.capacity_gate.allocated_replica_bytes != self.copy_evidence.total_copied_bytes:
            raise ValueError("cold replica capacity allocation must equal the unique copied payload bytes")

    def to_mapping(self) -> dict[str, object]:
        return {
            "database_replica_cold_result": {
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
                "replica_container_root": self.replica_container_root,
                "verification": self.verification,
                "source_mount": self.source_mount.to_mapping(),
                "cache_mount": self.cache_mount.to_mapping(),
                "capacity_gate": self.capacity_gate.to_mapping(),
                "replica_manifest_sha256": self.replica_manifest_sha256,
                "copy_evidence": self.copy_evidence.to_mapping(),
            }
        }


@dataclass(frozen=True)
class DatabaseReplicaColdFailureEvidence:
    """Bounded staged-placement proof that science never started and no Result exists."""

    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_id: str
    database_set: DatabaseSetIdentity
    requested_policy: DatabaseAccessPolicy
    source_manifest_sha256: str
    source_mount: DatabaseSourceMountFacts | None
    cache_mount: DatabaseCacheMountFacts | DatabaseWarmCacheMountFacts | None
    capacity_gate: DatabaseCapacityGate | None
    science_started: Literal[False]
    classification: DatabaseReplicaColdFailureClassification
    error: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseReplicaColdFailureEvidence")
        if _PHASE_RUN_ID.fullmatch(self.phase_run_id) is None:
            raise ValueError("cold failure phase_run_id must be an opaque phase-run id")
        if _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise ValueError("cold failure attempt_id must use attempt-NNNN")
        _validate_sha256(self.phase_runspec_digest, "cold failure Phase RunSpec digest")
        if _ACTION_ID.fullmatch(self.action_id) is None:
            raise ValueError("cold failure action_id must identify a preprocessing chunk")
        if not isinstance(self.database_set, DatabaseSetIdentity):
            raise ValueError("cold failure requires an exact Database Set identity")
        if self.requested_policy not in {
            DatabaseAccessPolicy.STAGE_REQUIRED,
            DatabaseAccessPolicy.STAGE_PREFERRED,
        }:
            raise ValueError("cold failure supports a staging-capable policy only")
        _validate_sha256(self.source_manifest_sha256, "cold failure source-manifest digest")
        if self.source_mount is not None and not isinstance(self.source_mount, DatabaseSourceMountFacts):
            raise ValueError("cold failure source_mount must be exact facts when available")
        if self.cache_mount is not None and not isinstance(
            self.cache_mount, DatabaseCacheMountFacts | DatabaseWarmCacheMountFacts
        ):
            raise ValueError("cold failure cache_mount must be exact facts when available")
        if self.capacity_gate is not None and not isinstance(self.capacity_gate, DatabaseCapacityGate):
            raise ValueError("cold failure capacity_gate must be exact when available")
        if self.classification == "insufficient-capacity" and (
            self.capacity_gate is None or self.capacity_gate.decision != "insufficient"
        ):
            raise ValueError("insufficient-capacity failure requires an insufficient completed gate")
        if (
            self.requested_policy == DatabaseAccessPolicy.STAGE_PREFERRED
            and self.classification == "insufficient-capacity"
        ):
            raise ValueError("stage-preferred insufficient capacity must be a successful fallback Result")
        if self.science_started is not False:
            raise ValueError("cold failure science_started must be false")
        if self.classification not in _COLD_FAILURE_CLASSIFICATIONS:
            raise ValueError(f"unsupported cold failure classification: {self.classification!r}")
        if not self.error or len(self.error) > 2048 or "\x00" in self.error:
            raise ValueError("cold failure error must be non-empty and bounded to 2048 characters")

    def to_mapping(self) -> dict[str, object]:
        return {
            "database_replica_cold_failure": {
                "schema_version": self.schema_version,
                "phase_run_id": self.phase_run_id,
                "attempt_id": self.attempt_id,
                "phase_runspec_digest": self.phase_runspec_digest,
                "action_id": self.action_id,
                "database_set": self.database_set.to_mapping(),
                "requested_policy": self.requested_policy.value,
                "source_manifest_sha256": self.source_manifest_sha256,
                "source_mount": None if self.source_mount is None else self.source_mount.to_mapping(),
                "cache_mount": None if self.cache_mount is None else self.cache_mount.to_mapping(),
                "capacity_gate": None if self.capacity_gate is None else self.capacity_gate.to_mapping(),
                "science_started": self.science_started,
                "classification": self.classification,
                "error": self.error,
            }
        }


def database_replica_cold_result_from_mapping(payload: Mapping[str, object]) -> DatabaseReplicaColdResult:
    inner = _unwrap_exact(payload, "database_replica_cold_result")
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
            "replica_container_root",
            "verification",
            "source_mount",
            "cache_mount",
            "capacity_gate",
            "replica_manifest_sha256",
            "copy_evidence",
        },
        "Database Replica cold Result",
    )
    policy = _required_str(inner, "requested_policy")
    outcome = _required_str(inner, "outcome")
    if policy not in {
        DatabaseAccessPolicy.STAGE_REQUIRED.value,
        DatabaseAccessPolicy.STAGE_PREFERRED.value,
    }:
        raise ValueError("cold replica Result requested_policy must be staging-capable")
    if outcome != DatabasePlacementOutcomeKind.REPLICA_COLD.value:
        raise ValueError("cold replica Result outcome must be replica-cold")
    return DatabaseReplicaColdResult(
        phase_run_id=_required_str(inner, "phase_run_id"),
        attempt_id=_required_str(inner, "attempt_id"),
        phase_runspec_digest=_required_str(inner, "phase_runspec_digest"),
        action_id=_required_str(inner, "action_id"),
        database_set=_database_set_from_mapping(_required_mapping(inner, "database_set")),
        requested_policy=DatabaseAccessPolicy(policy),
        source_manifest_sha256=_required_str(inner, "source_manifest_sha256"),
        branch_kind=cast("Literal['staged']", _required_str(inner, "branch_kind")),
        outcome=DatabasePlacementOutcomeKind.REPLICA_COLD,
        selected_container_root=_required_str(inner, "selected_container_root"),
        replica_container_root=_required_str(inner, "replica_container_root"),
        verification=cast("Literal['metadata-verified']", _required_str(inner, "verification")),
        source_mount=_source_mount_from_mapping(_required_mapping(inner, "source_mount")),
        cache_mount=_cache_mount_from_mapping(_required_mapping(inner, "cache_mount")),
        capacity_gate=_capacity_from_mapping(_required_mapping(inner, "capacity_gate")),
        replica_manifest_sha256=_required_str(inner, "replica_manifest_sha256"),
        copy_evidence=_copy_evidence_from_mapping(_required_mapping(inner, "copy_evidence")),
        schema_version=validate_schema_version(inner.get("schema_version"), record_name="DatabaseReplicaColdResult"),
    )


def canonical_database_replica_cold_result_bytes(result: DatabaseReplicaColdResult) -> bytes:
    return (json.dumps(result.to_mapping(), indent=2, sort_keys=True) + "\n").encode()


def database_replica_cold_result_digest(result: DatabaseReplicaColdResult) -> str:
    return hashlib.sha256(canonical_database_replica_cold_result_bytes(result)).hexdigest()


def database_replica_cold_failure_evidence_from_mapping(
    payload: Mapping[str, object],
) -> DatabaseReplicaColdFailureEvidence:
    inner = _unwrap_exact(payload, "database_replica_cold_failure")
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
            "source_mount",
            "cache_mount",
            "capacity_gate",
            "science_started",
            "classification",
            "error",
        },
        "Database Replica cold failure",
    )
    policy = _required_str(inner, "requested_policy")
    if policy not in {
        DatabaseAccessPolicy.STAGE_REQUIRED.value,
        DatabaseAccessPolicy.STAGE_PREFERRED.value,
    }:
        raise ValueError("cold failure requested_policy must be staging-capable")
    if inner.get("science_started") is not False:
        raise ValueError("cold failure science_started must be false")
    source_mount = inner.get("source_mount")
    cache_mount = inner.get("cache_mount")
    capacity = inner.get("capacity_gate")
    for name, value in (("source_mount", source_mount), ("cache_mount", cache_mount), ("capacity_gate", capacity)):
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"cold failure {name} must be a mapping or null")
    return DatabaseReplicaColdFailureEvidence(
        phase_run_id=_required_str(inner, "phase_run_id"),
        attempt_id=_required_str(inner, "attempt_id"),
        phase_runspec_digest=_required_str(inner, "phase_runspec_digest"),
        action_id=_required_str(inner, "action_id"),
        database_set=_database_set_from_mapping(_required_mapping(inner, "database_set")),
        requested_policy=DatabaseAccessPolicy(policy),
        source_manifest_sha256=_required_str(inner, "source_manifest_sha256"),
        source_mount=_source_mount_from_mapping(source_mount) if isinstance(source_mount, Mapping) else None,
        cache_mount=_failure_cache_mount_from_mapping(cache_mount) if isinstance(cache_mount, Mapping) else None,
        capacity_gate=_capacity_from_mapping(capacity) if isinstance(capacity, Mapping) else None,
        science_started=False,
        classification=cast(
            "DatabaseReplicaColdFailureClassification",
            _required_str(inner, "classification"),
        ),
        error=_required_str(inner, "error"),
        schema_version=validate_schema_version(
            inner.get("schema_version"), record_name="DatabaseReplicaColdFailureEvidence"
        ),
    )


def _failure_cache_mount_from_mapping(
    payload: Mapping[str, object],
) -> DatabaseCacheMountFacts | DatabaseWarmCacheMountFacts:
    if "writable_verified" in payload:
        return _cache_mount_from_mapping(payload)
    return database_warm_cache_mount_facts_from_mapping(payload)


def canonical_database_replica_cold_failure_evidence_bytes(
    failure: DatabaseReplicaColdFailureEvidence,
) -> bytes:
    return (json.dumps(failure.to_mapping(), indent=2, sort_keys=True) + "\n").encode()


def database_replica_cold_failure_evidence_digest(failure: DatabaseReplicaColdFailureEvidence) -> str:
    return hashlib.sha256(canonical_database_replica_cold_failure_evidence_bytes(failure)).hexdigest()


__all__ = [
    "DatabaseReplicaColdFailureClassification",
    "DatabaseReplicaColdFailureEvidence",
    "DatabaseReplicaColdResult",
    "canonical_database_replica_cold_failure_evidence_bytes",
    "canonical_database_replica_cold_result_bytes",
    "database_replica_cold_failure_evidence_digest",
    "database_replica_cold_failure_evidence_from_mapping",
    "database_replica_cold_result_digest",
    "database_replica_cold_result_from_mapping",
]
