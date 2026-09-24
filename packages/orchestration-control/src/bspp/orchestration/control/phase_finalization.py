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

"""Control-side acceptance and terminal sealing for one preprocessing Phase Run."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import yaml

from bspp.orchestration.contract.database_capacity_fallback_result import (
    DatabaseCapacityFallbackResult,
)
from bspp.orchestration.contract.database_placement import (
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
)
from bspp.orchestration.contract.database_placement_result import DatabasePostScienceObservation
from bspp.orchestration.contract.database_replica import DatabaseReplicaColdResult, DatabaseReplicaWarmResult
from bspp.orchestration.contract.database_replica_lease import (
    DatabaseReplicaLeaseTerminalEvidence,
    DatabaseReplicaWarmLeaseTerminalEvidence,
    PreprocessingStagedDatabasePlacementEvidence,
)
from bspp.orchestration.contract.phase import FoldingPhaseRunSpec, PhaseRunSpec, canonical_mapping_digest
from bspp.orchestration.contract.phase_action_evidence_attestation import (
    PhaseActionEvidenceAttestedEvent,
)
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardAdoptionEvidence,
    AttemptCarryForwardReceiptReference,
    AttemptCarryForwardRecord,
)
from bspp.orchestration.contract.phase_receipt import (
    PhaseFinalizedEvent,
    PhaseFinalizedPayload,
    PhaseReceipt,
    ProvidedSuccessfulSchedulerEvidence,
    phase_receipt_id,
    provided_successful_scheduler_evidence_from_mapping,
)
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingChunkActionEvidence,
    PreprocessingDatabasePlacementEvidence,
    preprocessing_chunk_action_evidence_from_mapping,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    PreprocessingHandoffBundle,
    msa_artifact_set_manifest_from_mapping,
    msa_chunk_manifest_from_mapping,
    preprocessing_content_validation_evidence_from_mapping,
    verified_local_bundled_artifact_location_from_mapping,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION
from bspp.orchestration.control.folding_phase_adapter import validate_folding_action_evidence
from bspp.orchestration.control.folding_phase_types import CanonicalPairIndex
from bspp.orchestration.control.phase_adapters import phase_authority_family
from bspp.orchestration.control.phase_authority import (
    PhaseAuthoritySealedError,
    PhaseAuthorityStore,
    PhaseAuthorityValidation,
    require_complete_current_runspec,
)
from bspp.orchestration.control.phase_lifecycle import require_finalizable_phase_attempt
from bspp.orchestration.control.postprocessing_authority_reader import (
    reject_historical_postprocessing_mutation,
)
from bspp.orchestration.control.postprocessing_phase_finalization import (
    PostprocessingPhaseFinalizationResult,
    finalize_postprocessing_phase,
)

Clock = Callable[[], datetime]

logger = logging.getLogger(__name__)

FOLDING_ADAPTER_VERSION = "folding-scientific-backend-v1"


@dataclass(frozen=True)
class PhaseFinalizationResult:
    """Small CLI result for one accepted or idempotently replayed receipt."""

    phase_run_id: str
    attempt_id: str
    phase_receipt_id: str
    artifact_set_id: str | None = None
    artifact_location_id: str | None = None
    status: str = "accepted"

    def to_mapping(self) -> dict[str, object]:
        return {
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_receipt_id": self.phase_receipt_id,
            "artifact_set_id": self.artifact_set_id,
            "artifact_location_id": self.artifact_location_id,
            "status": self.status,
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


def finalize_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    scheduler_evidence_path: Path | None = None,
    action_evidence_path: Path | None = None,
    handoff_path: Path | None = None,
    acceptance_adjudication_path: Path | None = None,
    clock: Clock | None = None,
    authority_store: PhaseAuthorityStore | None = None,
) -> PhaseFinalizationResult | PostprocessingPhaseFinalizationResult:
    """Validate small evidence records and append one terminal receipt event."""
    family = phase_authority_family(authority_root, phase_run_id)
    if family == "postprocessing":
        reject_historical_postprocessing_mutation(authority_root, phase_run_id)
        if any(
            path is None
            for path in (
                scheduler_evidence_path,
                action_evidence_path,
                handoff_path,
                acceptance_adjudication_path,
            )
        ):
            raise ValueError(
                "postprocessing Phase Finalization requires scheduler, aggregate action, handoff, "
                "and acceptance adjudication evidence"
            )
        if authority_store is not None:
            raise ValueError("postprocessing Phase Finalization does not accept a preprocessing authority store")
        assert scheduler_evidence_path is not None
        assert action_evidence_path is not None
        assert handoff_path is not None
        assert acceptance_adjudication_path is not None
        return finalize_postprocessing_phase(
            phase_run_id,
            authority_root=authority_root,
            scheduler_evidence_path=scheduler_evidence_path,
            aggregate_action_evidence_path=action_evidence_path,
            handoff_path=handoff_path,
            acceptance_adjudication_path=acceptance_adjudication_path,
            clock=clock,
        )
    if family == "folding":
        if acceptance_adjudication_path is not None:
            raise ValueError("folding Phase Finalization rejects --acceptance-adjudication")
        if scheduler_evidence_path is None or action_evidence_path is None:
            raise ValueError("folding Phase Finalization requires scheduler and action evidence")
        store = authority_store or PhaseAuthorityStore(authority_root)
        with store.phase_operation_lock(phase_run_id):
            return _finalize_folding_phase_under_lock(
                phase_run_id,
                store=store,
                scheduler_evidence_path=scheduler_evidence_path,
                action_evidence_path=action_evidence_path,
                handoff_path=handoff_path,
                clock=clock,
            )
    if acceptance_adjudication_path is not None:
        raise ValueError("preprocessing Phase Finalization rejects --acceptance-adjudication")
    if scheduler_evidence_path is None or action_evidence_path is None or handoff_path is None:
        raise ValueError("preprocessing Phase Finalization requires scheduler, action, and handoff evidence")
    store = authority_store or PhaseAuthorityStore(authority_root)
    with store.phase_operation_lock(phase_run_id):
        return _finalize_phase_under_lock(
            phase_run_id,
            store=store,
            scheduler_evidence_path=scheduler_evidence_path,
            action_evidence_path=action_evidence_path,
            handoff_path=handoff_path,
            clock=clock,
        )


def _finalize_phase_under_lock(
    phase_run_id: str,
    *,
    store: PhaseAuthorityStore,
    scheduler_evidence_path: Path,
    action_evidence_path: Path,
    handoff_path: Path,
    clock: Clock | None,
) -> PhaseFinalizationResult:
    authority = store.validate(phase_run_id)
    require_finalizable_phase_attempt(authority.lifecycle)
    require_complete_current_runspec(authority, operation="Phase Finalization")
    scheduler = _load_scheduler_evidence(scheduler_evidence_path)
    action_evidence = _load_action_evidence(action_evidence_path)
    handoff = _load_handoff(handoff_path, authority)
    finalized_at = (
        authority.receipt.finalized_at if authority.receipt is not None else _format_timestamp((clock or _utc_now)())
    )
    payload = _build_finalized_payload(
        authority,
        scheduler=scheduler,
        action_evidence=action_evidence,
        handoff=handoff,
        finalized_at=finalized_at,
    )
    if authority.receipt is not None:
        _require_identical_retry(authority, payload)
        return _result(authority.receipt)

    logger.warning(
        "accepting first-time preprocessing Phase finalization for %s attempt %s",
        phase_run_id,
        authority.phase_runspec.attempt_id,
    )

    def event_factory(sequence: int) -> PhaseFinalizedEvent:
        return PhaseFinalizedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at=finalized_at,
            payload=payload,
        )

    try:
        accepted = store.append_finalized_event(phase_run_id, event_factory)
    except PhaseAuthoritySealedError:
        accepted = store.validate(phase_run_id)
        assert accepted.receipt is not None
        concurrent_candidate = _build_finalized_payload(
            accepted,
            scheduler=scheduler,
            action_evidence=action_evidence,
            handoff=handoff,
            finalized_at=accepted.receipt.finalized_at,
        )
        _require_identical_retry(accepted, concurrent_candidate)
    assert accepted.receipt is not None
    return _result(accepted.receipt)


def _finalize_folding_phase_under_lock(
    phase_run_id: str,
    *,
    store: PhaseAuthorityStore,
    scheduler_evidence_path: Path,
    action_evidence_path: Path,
    clock: Clock | None,
    handoff_path: Path | None = None,
) -> PhaseFinalizationResult:
    """Finalize one folding Attempt through the generalized authority path."""
    authority = store.validate(phase_run_id)
    require_finalizable_phase_attempt(authority.lifecycle)
    require_complete_current_runspec(authority, operation="Phase Finalization")
    runspec = authority.phase_runspec
    if not isinstance(runspec, FoldingPhaseRunSpec):
        raise ValueError("folding Phase Finalization requires a folding Phase RunSpec")
    scheduler = _load_scheduler_evidence(scheduler_evidence_path)
    if runspec.payload.evidence_profile is not None:
        from .folding_artifact_evidence import _require_terminal, validate_local_folding_bundle

        _require_terminal(authority)
        if handoff_path is None:
            raise ValueError("artifact-backed folding finalization requires --handoff")
        evidence = validate_local_folding_bundle(
            handoff_path, runspec=runspec, action_evidence_path=action_evidence_path
        )
    else:
        evidence = _load_folding_action_evidence(action_evidence_path)
    index = validate_folding_action_evidence(phase_runspec=runspec, evidence=evidence)
    if index is None:
        raise ValueError("folding Phase Finalization requires a canonical-pair index")
    finalized_at = (
        authority.receipt.finalized_at if authority.receipt is not None else _format_timestamp((clock or _utc_now)())
    )
    payload = _build_folding_finalized_payload(
        authority,
        scheduler=scheduler,
        evidence=evidence,
        index=index,
        finalized_at=finalized_at,
    )
    if authority.receipt is not None:
        _require_identical_retry(authority, payload)
        return _result(authority.receipt)

    def event_factory(sequence: int) -> PhaseFinalizedEvent:
        return PhaseFinalizedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=runspec.attempt_id,
            occurred_at=finalized_at,
            payload=payload,
        )

    try:
        accepted = store.append_finalized_event(phase_run_id, event_factory)
    except PhaseAuthoritySealedError:
        accepted = store.validate(phase_run_id)
        assert accepted.receipt is not None
        concurrent_candidate = _build_folding_finalized_payload(
            accepted,
            scheduler=scheduler,
            evidence=evidence,
            index=index,
            finalized_at=accepted.receipt.finalized_at,
        )
        _require_identical_retry(accepted, concurrent_candidate)
    assert accepted.receipt is not None
    return _result(accepted.receipt)


def _build_folding_finalized_payload(
    authority: PhaseAuthorityValidation,
    *,
    scheduler: ProvidedSuccessfulSchedulerEvidence,
    evidence: Mapping[str, Mapping[str, object]],
    index: CanonicalPairIndex,
    finalized_at: str,
) -> PhaseFinalizedPayload:
    """Build one attempt-bound folding receipt and terminal finalized payload."""
    runspec = authority.phase_runspec
    if not isinstance(runspec, FoldingPhaseRunSpec):
        raise ValueError("folding Phase Finalization requires a folding Phase RunSpec")
    action = next((item for item in runspec.payload.actions if item.action_kind == "canonical-pair"), None)
    if action is None:
        raise ValueError("folding Phase Finalization requires the terminal canonical-pair action")
    submission = authority.submission
    if submission is None or submission.status != "submitted":
        raise ValueError("Phase Finalization requires a complete durable Phase Submission")
    submitted_action = next((item for item in submission.actions if item.action_id == action.action_id), None)
    if submitted_action is None or submitted_action.status != "submitted" or submitted_action.job_id is None:
        raise ValueError("Phase Finalization action has no durable Slurm assignment")
    if scheduler.job_id != submitted_action.job_id:
        raise ValueError("scheduler evidence does not match Control's assigned Slurm job")
    terminal = next(
        (item for item in authority.terminal_observations if item.action_id == action.action_id),
        None,
    )
    if terminal is not None and (
        terminal.outcome != "succeeded"
        or terminal.job_id != scheduler.job_id
        or terminal.source != scheduler.source
        or terminal.state != scheduler.state
        or terminal.exit_code != scheduler.exit_code
    ):
        raise ValueError("scheduler evidence conflicts with durable terminal accounting")
    folding_digests = tuple(canonical_mapping_digest(evidence[item.action_id]) for item in runspec.payload.actions)
    index_digest = canonical_mapping_digest(index.to_mapping())
    location = runspec.input_location
    input_sha256 = location.lz4_sha256
    input_size_bytes = location.lz4_size_bytes
    receipt_body: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "phase_run_id": runspec.phase_run_id,
        "attempt_id": runspec.attempt_id,
        "phase_plan_digest": authority.phase_plan.digest,
        "phase_runspec_location": authority.current_attempt.phase_runspec_location,
        "phase_runspec_digest": runspec.digest,
        "input_sha256": input_sha256,
        "input_size_bytes": input_size_bytes,
        "input_location_digest": canonical_mapping_digest(runspec.input_location.to_mapping()),
        "cluster_snapshot_digest": canonical_mapping_digest(runspec.cluster.to_mapping()),
        "runtime_action_digest": canonical_mapping_digest(action.to_mapping()),
        "adapter_version": FOLDING_ADAPTER_VERSION,
        "slurm_job_id": scheduler.job_id,
        "action_id": action.action_id,
        "canonical_pair_index_digest": index_digest,
        "folding_action_evidence_digests": list(folding_digests),
        "finalized_at": finalized_at,
        "result": "passed",
    }
    receipt = PhaseReceipt(
        phase_receipt_id=phase_receipt_id(receipt_body),
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_plan_digest=authority.phase_plan.digest,
        phase_runspec_location=authority.current_attempt.phase_runspec_location,
        phase_runspec_digest=runspec.digest,
        input_sha256=input_sha256,
        input_size_bytes=input_size_bytes,
        input_location_digest=canonical_mapping_digest(runspec.input_location.to_mapping()),
        cluster_snapshot_digest=canonical_mapping_digest(runspec.cluster.to_mapping()),
        runtime_action_digest=canonical_mapping_digest(action.to_mapping()),
        adapter_version=FOLDING_ADAPTER_VERSION,
        slurm_job_id=scheduler.job_id,
        action_id=action.action_id,
        finalized_at=finalized_at,
        canonical_pair_index_digest=index_digest,
        folding_action_evidence_digests=folding_digests,
    )
    return PhaseFinalizedPayload(
        scheduler_evidence=scheduler,
        folding_action_evidence_digests=folding_digests,
        canonical_pair_index_digest=index_digest,
        receipt=receipt,
    )


def _load_folding_action_evidence(path: Path) -> Mapping[str, Mapping[str, object]]:
    # Generated folding JSON carries full PAE arrays. Native JSON avoids the
    # additional YAML node tree; retain YAML compatibility on the same bytes.
    data = path.read_bytes()
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        try:
            payload = yaml.safe_load(data)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid folding action evidence document in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected folding action evidence mapping in {path}")
    if not payload or any(not isinstance(value, Mapping) for value in payload.values()):
        raise ValueError("folding action evidence must be a non-empty mapping of action evidence records")
    return {str(key): dict(value) for key, value in payload.items() if isinstance(value, Mapping)}


def _build_finalized_payload(
    authority: PhaseAuthorityValidation,
    *,
    scheduler: ProvidedSuccessfulSchedulerEvidence,
    action_evidence: PreprocessingChunkActionEvidence,
    handoff: PreprocessingHandoffBundle,
    finalized_at: str,
) -> PhaseFinalizedPayload:
    runspec = authority.phase_runspec
    if not isinstance(runspec, PhaseRunSpec):
        raise ValueError("preprocessing Phase Finalization requires a preprocessing Phase RunSpec")
    action = runspec.payload.actions[0]
    submission = authority.submission
    if submission is None or submission.status != "submitted":
        raise ValueError("Phase Finalization requires a complete durable Phase Submission")
    submitted_action = next((item for item in submission.actions if item.action_id == action.action_id), None)
    if submitted_action is None or submitted_action.status != "submitted" or submitted_action.job_id is None:
        raise ValueError("Phase Finalization action has no durable Slurm assignment")
    if scheduler.job_id != submitted_action.job_id:
        raise ValueError("scheduler evidence does not match Control's assigned Slurm job")
    terminal = next(
        (item for item in authority.terminal_observations if item.action_id == action.action_id),
        None,
    )
    if terminal is not None and (
        terminal.outcome != "succeeded"
        or terminal.job_id != scheduler.job_id
        or terminal.source != scheduler.source
        or terminal.state != scheduler.state
        or terminal.exit_code != scheduler.exit_code
    ):
        raise ValueError("scheduler evidence conflicts with durable terminal accounting")
    validation = handoff.content_validation
    action_digest = canonical_mapping_digest(action_evidence.to_mapping())
    expected_identity = (runspec.phase_run_id, runspec.attempt_id, runspec.digest, action.action_id)
    if (
        (scheduler.phase_run_id, scheduler.attempt_id, scheduler.phase_runspec_digest, scheduler.action_id)
        != expected_identity
        or (
            action_evidence.phase_run_id,
            action_evidence.attempt_id,
            action_evidence.phase_runspec_digest,
            action_evidence.action_id,
        )
        != expected_identity
        or (
            validation.phase_run_id,
            validation.attempt_id,
            validation.phase_runspec_digest,
            validation.action_id,
        )
        != expected_identity
    ):
        raise ValueError("finalization inputs do not bind the current authoritative attempt")
    if action_evidence.outcome != "succeeded" or validation.outcome != "passed":
        raise ValueError("Phase Finalization requires successful scientific and content evidence")
    placement = action_evidence.database_placement
    database = runspec.payload.database
    requested_policy: DatabaseAccessPolicy
    source_manifest_sha256: str
    placement_outcome: DatabasePlacementOutcomeKind
    if isinstance(placement, PreprocessingDatabasePlacementEvidence):
        direct_result = placement.result
        post_science_observation = placement.post_science_observation
        if direct_result is None or placement.failure is not None:
            raise ValueError("Phase Finalization requires successful Database Placement evidence")
        if not isinstance(post_science_observation, DatabasePostScienceObservation):
            raise ValueError("Phase Finalization requires a complete post-science Database Source observation")
        if isinstance(direct_result, DatabaseCapacityFallbackResult) and (
            database.staging is None
            or direct_result.pre_science_observation.members != database.source_manifest.members
            or direct_result.capacity_gate.reserved_bytes != database.staging.reserve_bytes
            or direct_result.cache_mount.filesystem_type != database.staging.expected_filesystem_type
        ):
            raise ValueError("Phase Finalization database placement does not reconcile with the authoritative RunSpec")
        valid_direct_family = (
            direct_result.requested_policy == DatabaseAccessPolicy.DIRECT
            and direct_result.outcome == DatabasePlacementOutcomeKind.DIRECT_REQUESTED
            and direct_result.branch_kind == "direct-requested"
        ) or (
            direct_result.requested_policy == DatabaseAccessPolicy.STAGE_PREFERRED
            and direct_result.outcome == DatabasePlacementOutcomeKind.DIRECT_CAPACITY_FALLBACK
            and direct_result.branch_kind == "direct-capacity-fallback"
        )
        if (
            direct_result.requested_policy != database.requested_policy
            or direct_result.source_manifest_sha256 != database.source_manifest_sha256
            or direct_result.database_set != database.database_set
            or not valid_direct_family
            or direct_result.selected_container_root != SELECTED_DATABASE_ROOT
            or not placement.science_started
            or not post_science_observation.matches(direct_result.pre_science_observation)
        ):
            raise ValueError("Phase Finalization database placement does not reconcile with the authoritative RunSpec")
        requested_policy = direct_result.requested_policy
        source_manifest_sha256 = direct_result.source_manifest_sha256
        placement_outcome = direct_result.outcome
    elif isinstance(placement, PreprocessingStagedDatabasePlacementEvidence):
        staged_result = placement.result
        lease = placement.lease_outcome
        valid_family = (
            isinstance(staged_result, DatabaseReplicaColdResult)
            and staged_result.outcome == DatabasePlacementOutcomeKind.REPLICA_COLD
            and isinstance(lease, DatabaseReplicaLeaseTerminalEvidence)
        ) or (
            isinstance(staged_result, DatabaseReplicaWarmResult)
            and staged_result.outcome == DatabasePlacementOutcomeKind.REPLICA_WARM
            and isinstance(lease, DatabaseReplicaWarmLeaseTerminalEvidence)
        )
        if (
            not isinstance(staged_result, DatabaseReplicaColdResult | DatabaseReplicaWarmResult)
            or placement.failure is not None
            or database.requested_policy
            not in {DatabaseAccessPolicy.STAGE_REQUIRED, DatabaseAccessPolicy.STAGE_PREFERRED}
            or staged_result.requested_policy != database.requested_policy
            or staged_result.source_manifest_sha256 != database.source_manifest_sha256
            or staged_result.database_set != database.database_set
            or staged_result.branch_kind != "staged"
            or staged_result.selected_container_root != SELECTED_DATABASE_ROOT
            or not placement.science_started
            or not valid_family
            or not isinstance(
                lease,
                DatabaseReplicaLeaseTerminalEvidence | DatabaseReplicaWarmLeaseTerminalEvidence,
            )
            or not lease.kernel_started
            or not lease.held_through_kernel_exit
        ):
            raise ValueError("Phase Finalization staged placement does not reconcile with authoritative RunSpec")
        requested_policy = staged_result.requested_policy
        source_manifest_sha256 = staged_result.source_manifest_sha256
        placement_outcome = staged_result.outcome
    else:
        raise ValueError("Phase Finalization rejects unsupported Database Placement evidence")
    if validation.action_evidence_digest != action_digest:
        raise ValueError("content validation does not bind the exact action evidence document")
    if action_evidence.chunk_name != action.payload.chunk_name or validation.chunk_name != action.payload.chunk_name:
        raise ValueError("finalization inputs do not bind the declared chunk")
    finalized_timestamp = _parse_timestamp(finalized_at)
    if any(
        _parse_timestamp(observed) > finalized_timestamp
        for observed in (scheduler.observed_at, action_evidence.finished_at, validation.finished_at)
    ):
        raise ValueError("Phase Finalization cannot predate required terminal evidence")
    carry_closure = _carry_forward_closure(authority, action_evidence)

    receipt_body: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "phase_run_id": runspec.phase_run_id,
        "attempt_id": runspec.attempt_id,
        "phase_plan_digest": authority.phase_plan.digest,
        "phase_runspec_location": authority.current_attempt.phase_runspec_location,
        "phase_runspec_digest": runspec.digest,
        "input_sha256": runspec.input_location.sha256,
        "input_size_bytes": runspec.input_location.size_bytes,
        "input_location_digest": canonical_mapping_digest(runspec.input_location.to_mapping()),
        "output_artifact_set_id": handoff.artifact_set.artifact_set_id,
        "artifact_location_id": handoff.artifact_location.artifact_location_id,
        "scheduler_evidence_digest": scheduler.digest,
        "action_evidence_digest": action_digest,
        "content_validation_evidence_digest": canonical_mapping_digest(validation.to_mapping()),
        "cluster_snapshot_digest": canonical_mapping_digest(runspec.cluster.to_mapping()),
        "runtime_action_digest": canonical_mapping_digest(action.to_mapping()),
        "database_requested_policy": requested_policy.value,
        "database_source_manifest_sha256": source_manifest_sha256,
        "database_placement_outcome": placement_outcome.value,
        "adapter_version": action_evidence.adapter_version,
        "slurm_job_id": scheduler.job_id,
        "action_id": action.action_id,
        "member_count": handoff.artifact_set.member_count,
        "logical_bytes": handoff.artifact_set.logical_bytes,
        "finalized_at": finalized_at,
        "result": "passed",
    }
    if carry_closure:
        receipt_body["carry_forward_closure"] = [item.to_mapping() for item in carry_closure]
    receipt = PhaseReceipt(
        phase_receipt_id=phase_receipt_id(receipt_body),
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_plan_digest=authority.phase_plan.digest,
        phase_runspec_location=authority.current_attempt.phase_runspec_location,
        phase_runspec_digest=runspec.digest,
        input_sha256=runspec.input_location.sha256,
        input_size_bytes=runspec.input_location.size_bytes,
        input_location_digest=canonical_mapping_digest(runspec.input_location.to_mapping()),
        output_artifact_set_id=handoff.artifact_set.artifact_set_id,
        artifact_location_id=handoff.artifact_location.artifact_location_id,
        scheduler_evidence_digest=scheduler.digest,
        action_evidence_digest=action_digest,
        content_validation_evidence_digest=canonical_mapping_digest(validation.to_mapping()),
        cluster_snapshot_digest=canonical_mapping_digest(runspec.cluster.to_mapping()),
        runtime_action_digest=canonical_mapping_digest(action.to_mapping()),
        database_requested_policy=requested_policy,
        database_source_manifest_sha256=source_manifest_sha256,
        database_placement_outcome=placement_outcome,
        adapter_version=action_evidence.adapter_version,
        slurm_job_id=scheduler.job_id,
        action_id=action.action_id,
        member_count=handoff.artifact_set.member_count,
        logical_bytes=handoff.artifact_set.logical_bytes,
        finalized_at=finalized_at,
        carry_forward_closure=carry_closure,
    )
    return PhaseFinalizedPayload(
        scheduler_evidence=scheduler,
        action_evidence=action_evidence,
        chunk_manifest=handoff.chunk_manifest,
        artifact_set=handoff.artifact_set,
        artifact_location=handoff.artifact_location,
        content_validation=validation,
        receipt=receipt,
    )


def _carry_forward_closure(
    authority: PhaseAuthorityValidation,
    current_evidence: PreprocessingChunkActionEvidence,
) -> tuple[AttemptCarryForwardReceiptReference, ...]:
    records_list: list[AttemptCarryForwardRecord] = []
    for item in (
        *(prior.carry_forward for prior in authority.prior_attempts),
        authority.current_carry_forward,
    ):
        if item is None:
            continue
        if not isinstance(item, AttemptCarryForwardRecord):
            raise ValueError("carry provenance lineage requires preprocessing carry records")
        records_list.append(item)
    records = tuple(records_list)
    if not records:
        if current_evidence.carry_forward_adoption is not None:
            raise ValueError("no-carry finalization cannot contain adoption evidence")
        return ()
    if authority.current_carry_forward is None:
        raise ValueError("carry provenance lineage does not end at the current Attempt")
    expected_targets = tuple(record.target_attempt_ordinal for record in records)
    if expected_targets != tuple(sorted(set(expected_targets))):
        raise ValueError("carry provenance lineage must be unique and Attempt-ordered")
    for predecessor, successor in pairwise(records):
        if predecessor.target_attempt_id != successor.source_attempt_id:
            raise ValueError("carry provenance lineage is not one immediate-predecessor chain")
    closure: list[AttemptCarryForwardReceiptReference] = []
    for record in records:
        adoption = (
            current_evidence.carry_forward_adoption
            if record.target_attempt_id == authority.current_attempt.attempt_id
            else _historical_adoption(authority, record)
        )
        _validate_adoption_for_record(authority, record, adoption)
        assert adoption is not None
        closure.append(
            AttemptCarryForwardReceiptReference(
                attempt_carry_forward_id=record.attempt_carry_forward_id,
                attempt_carry_forward_digest=record.digest,
                source_attempt_id=record.source_attempt_id,
                target_attempt_id=record.target_attempt_id,
                adopted_content_digest=record.content_digest,
                source_verification_evidence_digest=record.verification.evidence_mapping_digest,
                target_adoption_evidence_digest=adoption.digest,
            )
        )
    return tuple(closure)


def _historical_adoption(
    authority: PhaseAuthorityValidation,
    record: AttemptCarryForwardRecord,
) -> AttemptCarryForwardAdoptionEvidence:
    matches = tuple(
        (event.payload.evidence.carry_forward_adoption, event.payload.submission_id)
        for event in authority.events
        if isinstance(event, PhaseActionEvidenceAttestedEvent)
        and event.attempt_id == record.target_attempt_id
        and event.payload.evidence.carry_forward_adoption is not None
        and event.payload.evidence.carry_forward_adoption.attempt_carry_forward_id == record.attempt_carry_forward_id
    )
    if len(matches) != 1 or matches[0][0] is None:
        raise ValueError("carry provenance ancestor lacks exactly one durable target adoption")
    adoption, submission_id = matches[0]
    assert adoption is not None
    if adoption.phase_submission_id != submission_id:
        raise ValueError("carry provenance ancestor adoption does not bind its attested submission")
    return adoption


def _validate_adoption_for_record(
    authority: PhaseAuthorityValidation,
    record: AttemptCarryForwardRecord,
    adoption: AttemptCarryForwardAdoptionEvidence | None,
) -> None:
    if adoption is None:
        raise ValueError("carried finalization requires passing current adoption evidence")
    expected = tuple(
        (item.member_name, item.source_private_mount_path, item.target_declared_path, item.size_bytes, item.sha256)
        for item in record.content
    )
    observed = tuple(
        (item.member_name, item.source_path, item.target_path, item.size_bytes, item.sha256)
        for item in adoption.content
    )
    if (
        adoption.phase_run_id != authority.phase_run.phase_run_id
        or adoption.attempt_id != record.target_attempt_id
        or adoption.action_id != record.workspace.target_action_id
        or adoption.attempt_carry_forward_id != record.attempt_carry_forward_id
        or adoption.attempt_carry_forward_digest != record.digest
        or adoption.remaining_search_input_sha256 != record.remaining_search_input_sha256
        or observed != expected
    ):
        raise ValueError("carry provenance target adoption does not match its exact record")
    if record.target_attempt_id == authority.current_attempt.attempt_id and (
        authority.submission is None or adoption.phase_submission_id != authority.submission.submission_id
    ):
        raise ValueError("current carry adoption does not bind the authoritative submission")


def _require_identical_retry(
    authority: PhaseAuthorityValidation,
    candidate: PhaseFinalizedPayload,
) -> None:
    if authority.finalized_event is None or authority.receipt is None:
        raise ValueError("accepted Phase authority is missing its terminal receipt event")
    if authority.finalized_event.payload != candidate:
        raise ValueError("Phase Run is sealed and finalization retry inputs differ from its receipt")


def _load_scheduler_evidence(path: Path) -> ProvidedSuccessfulSchedulerEvidence:
    return provided_successful_scheduler_evidence_from_mapping(_load_mapping(path, "scheduler evidence"))


def _load_action_evidence(path: Path) -> PreprocessingChunkActionEvidence:
    return preprocessing_chunk_action_evidence_from_mapping(_load_mapping(path, "action evidence"))


def _load_handoff(path: Path, authority: PhaseAuthorityValidation) -> PreprocessingHandoffBundle:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"missing preprocessing handoff directory: {path}")
    runspec = authority.phase_runspec
    if not isinstance(runspec, PhaseRunSpec):
        raise ValueError("preprocessing handoff requires a preprocessing Phase RunSpec")
    chunk_name = runspec.payload.actions[0].payload.chunk_name
    chunk_relative = Path(f"chunks/{chunk_name.removesuffix('.fa')}.json")
    expected_files = {
        chunk_relative,
        Path("artifact-set.json"),
        Path("artifact-location.json"),
        Path("content-validation.json"),
    }
    expected_directories = {Path("chunks")}
    observed_files: set[Path] = set()
    observed_directories: set[Path] = set()
    for candidate in path.rglob("*"):
        relative = candidate.relative_to(path)
        if candidate.is_symlink():
            raise ValueError(f"preprocessing handoff must not contain symlinks: {relative}")
        if candidate.is_file():
            observed_files.add(relative)
        elif candidate.is_dir():
            observed_directories.add(relative)
        else:
            raise ValueError(f"preprocessing handoff contains an unsupported entry: {relative}")
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("preprocessing handoff must contain the exact successful four-file layout")
    return PreprocessingHandoffBundle(
        chunk_manifest=msa_chunk_manifest_from_mapping(_load_mapping(path / chunk_relative, "chunk manifest")),
        artifact_set=msa_artifact_set_manifest_from_mapping(_load_mapping(path / "artifact-set.json", "Artifact Set")),
        artifact_location=verified_local_bundled_artifact_location_from_mapping(
            _load_mapping(path / "artifact-location.json", "Artifact Location")
        ),
        content_validation=preprocessing_content_validation_evidence_from_mapping(
            _load_mapping(path / "content-validation.json", "content validation")
        ),
    )


def _load_mapping(path: Path, name: str) -> Mapping[str, object]:
    try:
        payload = yaml.safe_load(path.read_bytes())
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid {name} document in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected {name} mapping in {path}")
    return payload


def _result(receipt: PhaseReceipt) -> PhaseFinalizationResult:
    return PhaseFinalizationResult(
        phase_run_id=receipt.phase_run_id,
        attempt_id=receipt.attempt_id,
        phase_receipt_id=receipt.phase_receipt_id,
        artifact_set_id=receipt.output_artifact_set_id,
        artifact_location_id=receipt.artifact_location_id,
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Phase Finalization clock must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = ["PhaseFinalizationResult", "finalize_phase"]
