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

"""Closed direct/staged action placement dispatch and reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bspp.orchestration.contract.database_direct_result import (
    DatabaseDirectResult,
    database_direct_result_digest,
)
from bspp.orchestration.contract.database_placement import (
    DATABASE_REPLICA_LEASE_TARGET,
    DatabaseAccessPolicy,
)
from bspp.orchestration.contract.database_placement_result import (
    DatabasePlacementFailureEvidence,
    DatabasePostScienceEvidence,
)
from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaColdFailureEvidence,
    DatabaseReplicaColdResult,
    DatabaseReplicaResult,
    DatabaseReplicaWarmResult,
    database_replica_result_digest,
    database_replica_warm_result_digest,
)
from bspp.orchestration.contract.database_replica_lease import (
    DatabaseReplicaLeaseEvidence,
    DatabaseReplicaLeaseFailureClassification,
    DatabaseReplicaLeaseFailureEvidence,
    DatabaseReplicaWarmLeaseFailureEvidence,
    PreprocessingStagedDatabasePlacementEvidence,
)
from bspp.orchestration.contract.phase import PhaseRunSpec
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingDatabasePlacementCommandFailureEvidence,
    PreprocessingDatabasePlacementEvidence,
)

from ._database_placement_errors import DatabasePlacementError
from ._database_replica_evidence_io import (
    load_database_replica_cold_failure,
    load_database_replica_result,
)
from ._database_replica_lease import (
    reconcile_database_replica_result,
    reconcile_staged_database_placement_evidence,
)
from .database_placement import (
    load_database_direct_result,
    load_database_placement_failure,
    load_database_placement_result,
    reconcile_database_direct_result,
)


class DatabaseActionPlacementError(RuntimeError):
    """Action placement evidence is absent, contradictory, or unauthorized."""


@dataclass(frozen=True)
class DirectActionDatabasePlacement:
    result: DatabaseDirectResult | None
    failure: DatabasePlacementFailureEvidence | None


@dataclass(frozen=True)
class StagedActionDatabasePlacement:
    result: DatabaseReplicaResult | None
    failure: DatabaseReplicaColdFailureEvidence | None


ActionDatabasePlacement = DirectActionDatabasePlacement | StagedActionDatabasePlacement
ActionDatabasePlacementEvidence = (
    PreprocessingDatabasePlacementEvidence
    | PreprocessingStagedDatabasePlacementEvidence
    | PreprocessingDatabasePlacementCommandFailureEvidence
)
ResolvedActionDatabasePlacement = ActionDatabasePlacement | PreprocessingDatabasePlacementCommandFailureEvidence


def load_action_database_placement(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    result_path: Path,
    failure_path: Path | None,
) -> ActionDatabasePlacement:
    """Strict-load exactly one policy-specific Result or placement failure."""
    policy = runspec.payload.database.requested_policy
    if policy not in {
        DatabaseAccessPolicy.DIRECT,
        DatabaseAccessPolicy.STAGE_REQUIRED,
        DatabaseAccessPolicy.STAGE_PREFERRED,
    }:
        raise DatabaseActionPlacementError("execute-chunk supports only a canonical Database Access Policy")
    result_visible = result_path.exists() or result_path.is_symlink()
    failure_visible = failure_path is not None and (failure_path.exists() or failure_path.is_symlink())
    if result_visible and failure_visible:
        raise DatabaseActionPlacementError(
            "contradictory Database Placement Result and failure evidence cannot coexist"
        )
    if result_visible:
        if policy == DatabaseAccessPolicy.DIRECT:
            direct_result = load_database_placement_result(result_path)
            reconcile_database_direct_result(runspec, direct_result, action_id=action_id)
            return DirectActionDatabasePlacement(result=direct_result, failure=None)
        if policy == DatabaseAccessPolicy.STAGE_REQUIRED:
            staged_result = load_database_replica_result(result_path)
            reconcile_database_replica_result(runspec, staged_result)
            return StagedActionDatabasePlacement(result=staged_result, failure=None)
        try:
            preferred_direct_result = load_database_direct_result(result_path)
        except DatabasePlacementError:
            try:
                staged_result = load_database_replica_result(result_path)
            except DatabasePlacementError as staged_error:
                raise DatabaseActionPlacementError(
                    "stage-preferred Result is not one exact fallback, cold, or warm envelope"
                ) from staged_error
            reconcile_database_replica_result(runspec, staged_result)
            return StagedActionDatabasePlacement(result=staged_result, failure=None)
        reconcile_database_direct_result(runspec, preferred_direct_result, action_id=action_id)
        return DirectActionDatabasePlacement(result=preferred_direct_result, failure=None)
    if failure_visible:
        assert failure_path is not None
        failure: DatabasePlacementFailureEvidence | DatabaseReplicaColdFailureEvidence
        if policy == DatabaseAccessPolicy.DIRECT:
            failure = load_database_placement_failure(failure_path)
        else:
            failure = load_database_replica_cold_failure(failure_path)
        binding = runspec.payload.database
        if (
            failure.phase_run_id != runspec.phase_run_id
            or failure.attempt_id != runspec.attempt_id
            or failure.phase_runspec_digest != runspec.digest
            or failure.action_id != action_id
            or failure.database_set != binding.database_set
            or failure.requested_policy != binding.requested_policy
            or failure.source_manifest_sha256 != binding.source_manifest_sha256
        ):
            raise DatabaseActionPlacementError("Database Placement failure does not match RunSpec authority")
        if isinstance(failure, DatabasePlacementFailureEvidence):
            return DirectActionDatabasePlacement(result=None, failure=failure)
        return StagedActionDatabasePlacement(result=None, failure=failure)
    raise DatabaseActionPlacementError("Database Placement produced neither a Result nor classified failure evidence")


def database_action_placement_evidence(
    placement: ResolvedActionDatabasePlacement,
    *,
    lease_outcome: DatabaseReplicaLeaseEvidence | None,
    science_started: bool,
    post_science_observation: DatabasePostScienceEvidence | None,
) -> ActionDatabasePlacementEvidence:
    """Construct the exact concrete wrapper without widening direct serialization."""
    if isinstance(placement, PreprocessingDatabasePlacementCommandFailureEvidence):
        if lease_outcome is not None or science_started or post_science_observation is not None:
            raise ValueError("placement command failure cannot retain science or lease evidence")
        return placement
    if isinstance(placement, DirectActionDatabasePlacement):
        return PreprocessingDatabasePlacementEvidence(
            result=placement.result,
            result_digest=(database_direct_result_digest(placement.result) if placement.result is not None else None),
            failure=placement.failure,
            science_started=science_started,
            post_science_observation=post_science_observation,
        )
    return PreprocessingStagedDatabasePlacementEvidence(
        result=placement.result,
        result_digest=(database_replica_result_digest(placement.result) if placement.result is not None else None),
        failure=placement.failure,
        lease_outcome=lease_outcome,
        science_started=science_started,
    )


def lease_failure_evidence(
    result: DatabaseReplicaResult,
    error: DatabasePlacementError,
) -> DatabaseReplicaLeaseFailureEvidence | DatabaseReplicaWarmLeaseFailureEvidence:
    """Classify a bounded pre-science lease/revalidation failure."""
    message = str(error)
    classification: DatabaseReplicaLeaseFailureClassification
    if "contended" in message:
        classification = "lease-contended"
    elif "protocol file is unavailable" in message:
        classification = "lease-open-failed"
    elif "RunSpec authority" in message or "protocol file authority" in message:
        classification = "lease-authority-invalid"
    else:
        classification = "replica-revalidation-failed"
    if isinstance(result, DatabaseReplicaColdResult):
        return DatabaseReplicaLeaseFailureEvidence(
            database_replica_cold_result_digest=database_replica_result_digest(result),
            source_manifest_sha256=result.source_manifest_sha256,
            replica_manifest_sha256=result.replica_manifest_sha256,
            selected_container_root=result.selected_container_root,
            lease_target=DATABASE_REPLICA_LEASE_TARGET,
            verification="metadata-verified",
            classification=classification,
            error=message[:2048],
        )
    return DatabaseReplicaWarmLeaseFailureEvidence(
        database_replica_warm_result_digest=database_replica_warm_result_digest(result),
        source_manifest_sha256=result.source_manifest_sha256,
        replica_manifest_sha256=result.replica_manifest_sha256,
        selected_container_root=result.selected_container_root,
        lease_target=DATABASE_REPLICA_LEASE_TARGET,
        verification="metadata-verified",
        classification=classification,
        error=message[:2048],
    )


def reconcile_action_database_placement(
    runspec: PhaseRunSpec,
    evidence: ActionDatabasePlacementEvidence,
    *,
    action_id: str,
    require_success: bool,
) -> None:
    """Dispatch immediate and finalization reconciliation by concrete wrapper."""
    if isinstance(evidence, PreprocessingDatabasePlacementCommandFailureEvidence):
        _reconcile_command_failure(runspec, evidence, action_id=action_id)
        if require_success:
            raise ValueError("successful preprocessing reconciliation rejects placement command failure")
        return
    if isinstance(evidence, PreprocessingDatabasePlacementEvidence):
        if evidence.result is None:
            if require_success:
                raise ValueError("successful preprocessing reconciliation requires a Database Placement Result")
            assert evidence.failure is not None
            _reconcile_failure_binding(runspec, evidence.failure, action_id=action_id)
            return
        reconcile_database_direct_result(runspec, evidence.result, action_id=action_id)
        return
    reconcile_staged_database_placement_evidence(runspec, evidence, require_success=require_success)


def resolve_action_database_placement(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    result_path: Path,
    failure_path: Path | None,
    placement_process_status: int,
) -> ResolvedActionDatabasePlacement:
    """Resolve placement command status and durable files into one action state."""
    if (
        not isinstance(placement_process_status, int)
        or isinstance(placement_process_status, bool)
        or not 0 <= placement_process_status <= 255
    ):
        raise DatabaseActionPlacementError("placement process status must be a non-boolean integer in 0..255")
    expected_action_id = runspec.payload.actions[0].action_id
    try:
        if action_id != expected_action_id:
            raise DatabaseActionPlacementError(
                f"action id {action_id!r} does not match the sole declared action {expected_action_id!r}"
            )
        placement = load_action_database_placement(
            runspec,
            action_id=expected_action_id,
            result_path=result_path,
            failure_path=failure_path,
        )
    except (OSError, ValueError, DatabasePlacementError, DatabaseActionPlacementError) as exc:
        return _command_failure_evidence(
            runspec,
            action_id=expected_action_id,
            placement_process_status=placement_process_status,
            classification="evidence-reconciliation-failed",
            error=str(exc),
            result=None,
        )
    if placement_process_status == 0 or placement.failure is not None:
        return placement
    assert placement.result is not None
    return _command_failure_evidence(
        runspec,
        action_id=expected_action_id,
        placement_process_status=placement_process_status,
        classification="nonzero-after-result",
        error=f"Database Placement exited with status {placement_process_status} after publishing its exact Result",
        result=placement.result,
    )


def _command_failure_evidence(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    placement_process_status: int,
    classification: Literal["nonzero-after-result", "evidence-reconciliation-failed"],
    error: str,
    result: DatabaseDirectResult | DatabaseReplicaResult | None,
) -> PreprocessingDatabasePlacementCommandFailureEvidence:
    binding = runspec.payload.database
    result_digest: str | None = None
    if result is not None:
        result_digest = (
            database_replica_result_digest(result)
            if isinstance(result, DatabaseReplicaColdResult | DatabaseReplicaWarmResult)
            else database_direct_result_digest(result)
        )
    return PreprocessingDatabasePlacementCommandFailureEvidence(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        action_id=action_id,
        database_set=binding.database_set,
        requested_policy=binding.requested_policy,
        source_manifest_sha256=binding.source_manifest_sha256,
        placement_process_status=placement_process_status,
        result=result,
        result_digest=result_digest,
        science_started=False,
        classification=classification,
        error=(error or "Database Placement evidence reconciliation failed")[:2048],
    )


def _reconcile_failure_binding(
    runspec: PhaseRunSpec,
    failure: DatabasePlacementFailureEvidence | DatabaseReplicaColdFailureEvidence,
    *,
    action_id: str,
) -> None:
    binding = runspec.payload.database
    if (
        failure.phase_run_id != runspec.phase_run_id
        or failure.attempt_id != runspec.attempt_id
        or failure.phase_runspec_digest != runspec.digest
        or failure.action_id != action_id
        or failure.database_set != binding.database_set
        or failure.requested_policy != binding.requested_policy
        or failure.source_manifest_sha256 != binding.source_manifest_sha256
    ):
        raise ValueError("Database Placement failure does not match RunSpec authority")


def _reconcile_command_failure(
    runspec: PhaseRunSpec,
    evidence: PreprocessingDatabasePlacementCommandFailureEvidence,
    *,
    action_id: str,
) -> None:
    binding = runspec.payload.database
    if (
        evidence.phase_run_id != runspec.phase_run_id
        or evidence.attempt_id != runspec.attempt_id
        or evidence.phase_runspec_digest != runspec.digest
        or evidence.action_id != action_id
        or evidence.database_set != binding.database_set
        or evidence.requested_policy != binding.requested_policy
        or evidence.source_manifest_sha256 != binding.source_manifest_sha256
    ):
        raise ValueError("Database Placement command failure does not match RunSpec authority")
    if evidence.result is None:
        return
    if isinstance(evidence.result, DatabaseReplicaColdResult | DatabaseReplicaWarmResult):
        reconcile_database_replica_result(runspec, evidence.result)
    else:
        reconcile_database_direct_result(runspec, evidence.result, action_id=action_id)


def resolve_database_science_branch(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    result_path: Path,
    failure_path: Path | None,
) -> Literal["staged", "direct-capacity-fallback", "placement-failure"]:
    """Resolve one strict reconciled stage-preferred branch without payload access."""
    if runspec.payload.database.requested_policy != DatabaseAccessPolicy.STAGE_PREFERRED:
        raise DatabaseActionPlacementError("science branch selection requires stage-preferred policy")
    placement = load_action_database_placement(
        runspec,
        action_id=action_id,
        result_path=result_path,
        failure_path=failure_path,
    )
    if placement.failure is not None:
        return "placement-failure"
    if isinstance(placement, StagedActionDatabasePlacement):
        if placement.result is None:
            raise DatabaseActionPlacementError("staged branch selection requires a Result")
        return "staged"
    if placement.result is None or placement.result.branch_kind != "direct-capacity-fallback":
        raise DatabaseActionPlacementError("direct branch selection requires a capacity-fallback Result")
    return "direct-capacity-fallback"


__all__ = [
    "ActionDatabasePlacement",
    "ActionDatabasePlacementEvidence",
    "DatabaseActionPlacementError",
    "DirectActionDatabasePlacement",
    "StagedActionDatabasePlacement",
    "database_action_placement_evidence",
    "lease_failure_evidence",
    "load_action_database_placement",
    "reconcile_action_database_placement",
    "resolve_action_database_placement",
    "resolve_database_science_branch",
]
