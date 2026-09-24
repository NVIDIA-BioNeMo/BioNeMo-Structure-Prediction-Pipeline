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

"""Strict Result authority for reuse of an immutable warm Database Replica."""

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
from bspp.orchestration.contract.database_replica_facts import (
    DatabaseWarmCacheMountFacts,
    _database_set_from_mapping,
    _unwrap_exact,
    _validate_sha256,
    database_warm_cache_mount_facts_from_mapping,
)
from bspp.orchestration.contract.database_set_provisioning import DatabaseSetIdentity
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ACTION_ID = re.compile(r"preprocessing-chunk-[0-9]{6}")


@dataclass(frozen=True)
class DatabaseReplicaWarmResult:
    """Successful authority for reuse of one fully validated warm replica."""

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
    cache_mount: DatabaseWarmCacheMountFacts
    replica_manifest_sha256: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseReplicaWarmResult")
        if _PHASE_RUN_ID.fullmatch(self.phase_run_id) is None:
            raise ValueError("warm replica phase_run_id must be an opaque phase-run id")
        if _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise ValueError("warm replica attempt_id must use attempt-NNNN")
        _validate_sha256(self.phase_runspec_digest, "warm replica Phase RunSpec digest")
        if _ACTION_ID.fullmatch(self.action_id) is None:
            raise ValueError("warm replica action_id must identify a preprocessing chunk")
        if not isinstance(self.database_set, DatabaseSetIdentity):
            raise ValueError("warm replica Result requires an exact Database Set identity")
        if self.requested_policy not in {
            DatabaseAccessPolicy.STAGE_REQUIRED,
            DatabaseAccessPolicy.STAGE_PREFERRED,
        }:
            raise ValueError("warm replica Result supports a staging-capable policy only")
        _validate_sha256(self.source_manifest_sha256, "warm replica source-manifest digest")
        if self.branch_kind != "staged" or self.outcome != DatabasePlacementOutcomeKind.REPLICA_WARM:
            raise ValueError("warm replica Result requires the staged replica-warm branch")
        if self.selected_container_root not in (SELECTED_DATABASE_ROOT, LEGACY_SELECTED_DATABASE_ROOT):
            raise ValueError(f"warm replica selected root must be {SELECTED_DATABASE_ROOT!r}")
        expected_replica = f"{DATABASE_CACHE_ROOT}/replicas/{self.source_manifest_sha256}"
        legacy_expected_replica = f"{LEGACY_DATABASE_CACHE_ROOT}/replicas/{self.source_manifest_sha256}"
        if self.replica_container_root not in (expected_replica, legacy_expected_replica):
            raise ValueError("warm replica Result must use the manifest-addressed cache path")
        if self.verification != "metadata-verified":
            raise ValueError("warm replica Result verification must be 'metadata-verified'")
        if not isinstance(self.cache_mount, DatabaseWarmCacheMountFacts):
            raise ValueError("warm replica Result requires exact read-only cache mount facts")
        _validate_sha256(self.replica_manifest_sha256, "Database Replica Manifest digest")

    def to_mapping(self) -> dict[str, object]:
        return {
            "database_replica_warm_result": {
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
                "cache_mount": self.cache_mount.to_mapping(),
                "replica_manifest_sha256": self.replica_manifest_sha256,
            }
        }


def database_replica_warm_result_from_mapping(payload: Mapping[str, object]) -> DatabaseReplicaWarmResult:
    inner = _unwrap_exact(payload, "database_replica_warm_result")
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
            "cache_mount",
            "replica_manifest_sha256",
        },
        "Database Replica warm Result",
    )
    policy = _required_str(inner, "requested_policy")
    if policy not in {
        DatabaseAccessPolicy.STAGE_REQUIRED.value,
        DatabaseAccessPolicy.STAGE_PREFERRED.value,
    }:
        raise ValueError("warm replica Result requested_policy must be staging-capable")
    if _required_str(inner, "outcome") != DatabasePlacementOutcomeKind.REPLICA_WARM.value:
        raise ValueError("warm replica Result outcome must be replica-warm")
    return DatabaseReplicaWarmResult(
        phase_run_id=_required_str(inner, "phase_run_id"),
        attempt_id=_required_str(inner, "attempt_id"),
        phase_runspec_digest=_required_str(inner, "phase_runspec_digest"),
        action_id=_required_str(inner, "action_id"),
        database_set=_database_set_from_mapping(_required_mapping(inner, "database_set")),
        requested_policy=DatabaseAccessPolicy(policy),
        source_manifest_sha256=_required_str(inner, "source_manifest_sha256"),
        branch_kind=cast("Literal['staged']", _required_str(inner, "branch_kind")),
        outcome=DatabasePlacementOutcomeKind.REPLICA_WARM,
        selected_container_root=_required_str(inner, "selected_container_root"),
        replica_container_root=_required_str(inner, "replica_container_root"),
        verification=cast("Literal['metadata-verified']", _required_str(inner, "verification")),
        cache_mount=database_warm_cache_mount_facts_from_mapping(_required_mapping(inner, "cache_mount")),
        replica_manifest_sha256=_required_str(inner, "replica_manifest_sha256"),
        schema_version=validate_schema_version(inner.get("schema_version"), record_name="DatabaseReplicaWarmResult"),
    )


def canonical_database_replica_warm_result_bytes(result: DatabaseReplicaWarmResult) -> bytes:
    return (json.dumps(result.to_mapping(), indent=2, sort_keys=True) + "\n").encode()


def database_replica_warm_result_digest(result: DatabaseReplicaWarmResult) -> str:
    return hashlib.sha256(canonical_database_replica_warm_result_bytes(result)).hexdigest()


__all__ = [
    "DatabaseReplicaWarmResult",
    "canonical_database_replica_warm_result_bytes",
    "database_replica_warm_result_digest",
    "database_replica_warm_result_from_mapping",
]
