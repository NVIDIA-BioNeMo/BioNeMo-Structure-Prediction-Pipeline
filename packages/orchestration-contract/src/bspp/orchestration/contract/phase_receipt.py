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

"""Attempt-bound Phase Receipt and terminal finalization event contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Set
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

from bspp.orchestration.contract.database_placement import (
    LEGACY_SELECTED_DATABASE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
)
from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaColdResult,
    DatabaseReplicaWarmResult,
    database_replica_cold_result_digest,
    database_replica_warm_result_digest,
)
from bspp.orchestration.contract.database_replica_lease import (
    DatabaseReplicaLeaseTerminalEvidence,
    DatabaseReplicaWarmLeaseTerminalEvidence,
    PreprocessingStagedDatabasePlacementEvidence,
)
from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardReceiptReference,
    attempt_carry_forward_receipt_reference_from_mapping,
)
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingChunkActionEvidence,
    PreprocessingDatabasePlacementEvidence,
    preprocessing_chunk_action_evidence_from_mapping,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    MsaArtifactSetManifest,
    MsaChunkManifest,
    PreprocessingContentValidationEvidence,
    PreprocessingHandoffBundle,
    VerifiedLocalBundledArtifactLocation,
    msa_artifact_set_manifest_from_mapping,
    msa_chunk_manifest_from_mapping,
    preprocessing_content_validation_evidence_from_mapping,
    verified_local_bundled_artifact_location_from_mapping,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

SchedulerEvidenceKind = Literal["provided-successful-terminal-observation"]
SchedulerEvidenceSource = Literal["sacct"]
PhaseFinalizedEventType = Literal["phase-finalized"]
ReceiptFamily = Literal["preprocessing", "folding"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_ACTION_ID = re.compile(r"(preprocessing-chunk|msa-flatten|split|preprocess|fold|canonical-pair)-[0-9]{6}")
_JOB_ID = re.compile(r"[0-9]+")
_ARTIFACT_SET_ID = re.compile(r"sha256:[0-9a-f]{64}")
_LOCATION_ID = re.compile(r"artifact-location-[0-9a-f]{64}")
_RECEIPT_ID = re.compile(r"phase-receipt-[0-9a-f]{64}")

_FOLDING_ACTION_KINDS = frozenset({"msa-flatten", "split", "preprocess", "fold", "canonical-pair"})


def _receipt_family(action_id: str) -> ReceiptFamily:
    kind = action_id.rsplit("-", 1)[0]
    if kind == "preprocessing-chunk":
        return "preprocessing"
    if kind in _FOLDING_ACTION_KINDS:
        return "folding"
    raise ValueError("record has invalid Runtime Action identity")


_SHARED_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "phase_receipt_id",
        "phase_run_id",
        "attempt_id",
        "phase_plan_digest",
        "phase_runspec_location",
        "phase_runspec_digest",
        "input_sha256",
        "input_size_bytes",
        "input_location_digest",
        "cluster_snapshot_digest",
        "runtime_action_digest",
        "adapter_version",
        "slurm_job_id",
        "action_id",
        "finalized_at",
        "result",
        "carry_forward_closure",
    }
)
_PREPROCESSING_RECEIPT_FIELDS = _SHARED_RECEIPT_FIELDS | {
    "output_artifact_set_id",
    "artifact_location_id",
    "scheduler_evidence_digest",
    "action_evidence_digest",
    "content_validation_evidence_digest",
    "database_requested_policy",
    "database_source_manifest_sha256",
    "database_placement_outcome",
    "member_count",
    "logical_bytes",
}
_FOLDING_RECEIPT_FIELDS = _SHARED_RECEIPT_FIELDS | {
    "canonical_pair_index_digest",
    "folding_action_evidence_digests",
}

_PREPROCESSING_FINALIZED_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "scheduler_evidence",
        "action_evidence",
        "chunk_manifest",
        "artifact_set",
        "artifact_location",
        "content_validation",
        "receipt",
    }
)
_FOLDING_FINALIZED_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "scheduler_evidence",
        "folding_action_evidence_digests",
        "canonical_pair_index_digest",
        "receipt",
    }
)


@dataclass(frozen=True)
class ProvidedSuccessfulSchedulerEvidence:
    """Caller-supplied terminal Slurm observation, not submission authority."""

    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_id: str
    job_id: str
    observed_at: str
    source: SchedulerEvidenceSource = "sacct"
    state: str = "COMPLETED"
    exit_code: str = "0:0"
    evidence_kind: SchedulerEvidenceKind = "provided-successful-terminal-observation"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "ProvidedSuccessfulSchedulerEvidence")
        _validate_run_attempt_action(self.phase_run_id, self.attempt_id, self.action_id)
        _sha(self.phase_runspec_digest, "scheduler evidence RunSpec digest")
        if self.evidence_kind != "provided-successful-terminal-observation":
            raise ValueError("scheduler evidence must be explicitly marked as a supplied terminal observation")
        if self.source != "sacct" or self.state != "COMPLETED" or self.exit_code != "0:0":
            raise ValueError("scheduler evidence must be a successful terminal sacct observation")
        if _JOB_ID.fullmatch(self.job_id) is None:
            raise ValueError("scheduler evidence job_id must be numeric")
        _timestamp(self.observed_at, "scheduler evidence observed_at")

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evidence_kind": self.evidence_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_id": self.action_id,
            "job_id": self.job_id,
            "source": self.source,
            "state": self.state,
            "exit_code": self.exit_code,
            "observed_at": self.observed_at,
        }


@dataclass(frozen=True)
class PhaseReceipt:
    """Immutable successful handoff authority for one exact Phase Attempt."""

    phase_receipt_id: str
    phase_run_id: str
    attempt_id: str
    phase_plan_digest: str
    phase_runspec_location: str
    phase_runspec_digest: str
    input_sha256: str
    input_size_bytes: int
    input_location_digest: str
    cluster_snapshot_digest: str
    runtime_action_digest: str
    adapter_version: str
    slurm_job_id: str
    action_id: str
    finalized_at: str
    # preprocessing-only (required for preprocessing, omitted for folding)
    output_artifact_set_id: str | None = None
    artifact_location_id: str | None = None
    scheduler_evidence_digest: str | None = None
    action_evidence_digest: str | None = None
    content_validation_evidence_digest: str | None = None
    database_requested_policy: DatabaseAccessPolicy | None = None
    database_source_manifest_sha256: str | None = None
    database_placement_outcome: DatabasePlacementOutcomeKind | None = None
    member_count: int | None = None
    logical_bytes: int | None = None
    # folding-only
    canonical_pair_index_digest: str | None = None
    folding_action_evidence_digests: tuple[str, ...] = ()
    # shared defaults
    carry_forward_closure: tuple[AttemptCarryForwardReceiptReference, ...] = ()
    result: str = "passed"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PhaseReceipt")
        _validate_run_attempt_action(self.phase_run_id, self.attempt_id, self.action_id)
        if _receipt_family(self.action_id) == "preprocessing":
            self._validate_preprocessing_receipt()
        else:
            self._validate_folding_receipt()

    def _validate_preprocessing_receipt(self) -> None:
        for label, digest in (
            ("Phase Plan", self.phase_plan_digest),
            ("Phase RunSpec", self.phase_runspec_digest),
            ("input", self.input_sha256),
            ("input location", self.input_location_digest),
            ("scheduler evidence", cast("str", self.scheduler_evidence_digest)),
            ("action evidence", cast("str", self.action_evidence_digest)),
            ("content validation", cast("str", self.content_validation_evidence_digest)),
            ("cluster snapshot", self.cluster_snapshot_digest),
            ("runtime action", self.runtime_action_digest),
            ("database source manifest", cast("str", self.database_source_manifest_sha256)),
        ):
            _sha(digest, f"Phase Receipt {label} digest")
        requested_policy = cast("DatabaseAccessPolicy", self.database_requested_policy)
        placement_outcome = cast("DatabasePlacementOutcomeKind", self.database_placement_outcome)
        if (requested_policy, placement_outcome) not in {
            (DatabaseAccessPolicy.DIRECT, DatabasePlacementOutcomeKind.DIRECT_REQUESTED),
            (DatabaseAccessPolicy.STAGE_REQUIRED, DatabasePlacementOutcomeKind.REPLICA_COLD),
            (DatabaseAccessPolicy.STAGE_REQUIRED, DatabasePlacementOutcomeKind.REPLICA_WARM),
            (DatabaseAccessPolicy.STAGE_PREFERRED, DatabasePlacementOutcomeKind.REPLICA_COLD),
            (DatabaseAccessPolicy.STAGE_PREFERRED, DatabasePlacementOutcomeKind.REPLICA_WARM),
            (
                DatabaseAccessPolicy.STAGE_PREFERRED,
                DatabasePlacementOutcomeKind.DIRECT_CAPACITY_FALLBACK,
            ),
        }:
            raise ValueError("Phase Receipt requires an accepted direct or staged database pair")
        if self.phase_runspec_location != f"attempts/{self.attempt_id}/phase-runspec.json":
            raise ValueError("Phase Receipt RunSpec location must be authority-relative and attempt-bound")
        if _ARTIFACT_SET_ID.fullmatch(cast("str", self.output_artifact_set_id)) is None:
            raise ValueError("Phase Receipt has an invalid output Artifact Set id")
        if _LOCATION_ID.fullmatch(cast("str", self.artifact_location_id)) is None:
            raise ValueError("Phase Receipt has an invalid Artifact Location id")
        if not self.adapter_version or _JOB_ID.fullmatch(self.slurm_job_id) is None:
            raise ValueError("Phase Receipt requires adapter version and numeric Slurm job id")
        _positive(self.input_size_bytes, "Phase Receipt input_size_bytes")
        _positive(cast("int", self.member_count), "Phase Receipt member_count")
        _positive(cast("int", self.logical_bytes), "Phase Receipt logical_bytes")
        if self.result != "passed":
            raise ValueError("a Phase Receipt can only record a passed finalization")
        _timestamp(self.finalized_at, "Phase Receipt finalized_at")
        self._validate_carry_forward_closure()
        if self.phase_receipt_id != phase_receipt_id(self.identity_mapping()):
            raise ValueError("Phase Receipt id does not match its canonical attempt-bound content")

    def _validate_folding_receipt(self) -> None:
        for label, digest in (
            ("Phase Plan", self.phase_plan_digest),
            ("Phase RunSpec", self.phase_runspec_digest),
            ("input", self.input_sha256),
            ("input location", self.input_location_digest),
            ("cluster snapshot", self.cluster_snapshot_digest),
            ("runtime action", self.runtime_action_digest),
            ("canonical-pair index", cast("str", self.canonical_pair_index_digest)),
        ):
            _sha(digest, f"Phase Receipt {label} digest")
        if not self.folding_action_evidence_digests or any(
            _SHA256.fullmatch(digest) is None for digest in self.folding_action_evidence_digests
        ):
            raise ValueError(
                "Phase Receipt folding action evidence digests must be a non-empty tuple of lowercase SHA-256"
            )
        if (
            self.output_artifact_set_id is not None
            or self.artifact_location_id is not None
            or self.scheduler_evidence_digest is not None
            or self.action_evidence_digest is not None
            or self.content_validation_evidence_digest is not None
            or self.database_requested_policy is not None
            or self.database_source_manifest_sha256 is not None
            or self.database_placement_outcome is not None
            or self.member_count is not None
            or self.logical_bytes is not None
        ):
            raise ValueError("folding Phase Receipt rejects preprocessing chunk and database fields")
        if self.phase_runspec_location != f"attempts/{self.attempt_id}/phase-runspec.json":
            raise ValueError("Phase Receipt RunSpec location must be authority-relative and attempt-bound")
        if not self.adapter_version or _JOB_ID.fullmatch(self.slurm_job_id) is None:
            raise ValueError("Phase Receipt requires adapter version and numeric Slurm job id")
        _positive(self.input_size_bytes, "Phase Receipt input_size_bytes")
        if self.result != "passed":
            raise ValueError("a Phase Receipt can only record a passed finalization")
        _timestamp(self.finalized_at, "Phase Receipt finalized_at")
        self._validate_carry_forward_closure()
        if self.phase_receipt_id != phase_receipt_id(self.identity_mapping()):
            raise ValueError("Phase Receipt id does not match its canonical attempt-bound content")

    def _validate_carry_forward_closure(self) -> None:
        if not isinstance(self.carry_forward_closure, tuple) or any(
            not isinstance(item, AttemptCarryForwardReceiptReference) for item in self.carry_forward_closure
        ):
            raise ValueError("Phase Receipt carry-forward closure must be an immutable tuple")
        targets = tuple(item.target_attempt_id for item in self.carry_forward_closure)
        if len(set(item.attempt_carry_forward_id for item in self.carry_forward_closure)) != len(
            self.carry_forward_closure
        ) or targets != tuple(sorted(targets)):
            raise ValueError("Phase Receipt carry-forward closure must be unique and Attempt-ordered")
        if self.carry_forward_closure and self.carry_forward_closure[-1].target_attempt_id != self.attempt_id:
            raise ValueError("Phase Receipt carry-forward closure must end at the successful Attempt")

    def identity_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_plan_digest": self.phase_plan_digest,
            "phase_runspec_location": self.phase_runspec_location,
            "phase_runspec_digest": self.phase_runspec_digest,
            "input_sha256": self.input_sha256,
            "input_size_bytes": self.input_size_bytes,
            "input_location_digest": self.input_location_digest,
            "cluster_snapshot_digest": self.cluster_snapshot_digest,
            "runtime_action_digest": self.runtime_action_digest,
            "adapter_version": self.adapter_version,
            "slurm_job_id": self.slurm_job_id,
            "action_id": self.action_id,
            "finalized_at": self.finalized_at,
            "result": self.result,
        }
        if _receipt_family(self.action_id) == "preprocessing":
            result.update(
                {
                    "output_artifact_set_id": self.output_artifact_set_id,
                    "artifact_location_id": self.artifact_location_id,
                    "scheduler_evidence_digest": self.scheduler_evidence_digest,
                    "action_evidence_digest": self.action_evidence_digest,
                    "content_validation_evidence_digest": self.content_validation_evidence_digest,
                    "database_requested_policy": cast("DatabaseAccessPolicy", self.database_requested_policy).value,
                    "database_source_manifest_sha256": self.database_source_manifest_sha256,
                    "database_placement_outcome": cast(
                        "DatabasePlacementOutcomeKind", self.database_placement_outcome
                    ).value,
                    "member_count": self.member_count,
                    "logical_bytes": self.logical_bytes,
                }
            )
        else:
            result["canonical_pair_index_digest"] = self.canonical_pair_index_digest
            result["folding_action_evidence_digests"] = list(self.folding_action_evidence_digests)
        if self.carry_forward_closure:
            result["carry_forward_closure"] = [item.to_mapping() for item in self.carry_forward_closure]
        return result

    def to_mapping(self) -> dict[str, object]:
        return {"phase_receipt_id": self.phase_receipt_id, **self.identity_mapping()}


def phase_receipt_id(payload: Mapping[str, object]) -> str:
    return f"phase-receipt-{canonical_mapping_digest(payload)}"


@dataclass(frozen=True)
class PhaseFinalizedPayload:
    """Self-contained records retained by one terminal Phase event."""

    scheduler_evidence: ProvidedSuccessfulSchedulerEvidence
    receipt: PhaseReceipt
    action_evidence: PreprocessingChunkActionEvidence | None = None
    chunk_manifest: MsaChunkManifest | None = None
    artifact_set: MsaArtifactSetManifest | None = None
    artifact_location: VerifiedLocalBundledArtifactLocation | None = None
    content_validation: PreprocessingContentValidationEvidence | None = None
    folding_action_evidence_digests: tuple[str, ...] = ()
    canonical_pair_index_digest: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PhaseFinalizedPayload")
        if _receipt_family(self.receipt.action_id) == "preprocessing":
            self._validate_preprocessing_payload()
        else:
            self._validate_folding_payload()

    def _validate_preprocessing_payload(self) -> None:
        receipt = self.receipt
        action_evidence = cast("PreprocessingChunkActionEvidence", self.action_evidence)
        chunk_manifest = cast("MsaChunkManifest", self.chunk_manifest)
        artifact_set = cast("MsaArtifactSetManifest", self.artifact_set)
        artifact_location = cast("VerifiedLocalBundledArtifactLocation", self.artifact_location)
        content_validation = cast("PreprocessingContentValidationEvidence", self.content_validation)
        if (
            self.scheduler_evidence.phase_run_id != receipt.phase_run_id
            or action_evidence.phase_run_id != receipt.phase_run_id
            or content_validation.phase_run_id != receipt.phase_run_id
            or self.scheduler_evidence.attempt_id != receipt.attempt_id
            or action_evidence.attempt_id != receipt.attempt_id
            or content_validation.attempt_id != receipt.attempt_id
        ):
            raise ValueError("Phase finalized payload records must bind one run and attempt")
        if self.scheduler_evidence.digest != receipt.scheduler_evidence_digest:
            raise ValueError("Phase Receipt does not bind the exact scheduler evidence")
        if canonical_mapping_digest(action_evidence.to_mapping()) != receipt.action_evidence_digest:
            raise ValueError("Phase Receipt does not bind the exact action evidence")
        placement = action_evidence.database_placement
        if isinstance(placement, PreprocessingDatabasePlacementEvidence):
            direct_result = placement.result
            if direct_result is None or placement.failure is not None:
                raise ValueError("Phase finalized payload requires successful Database Placement evidence")
            if (
                receipt.database_requested_policy != direct_result.requested_policy
                or receipt.database_source_manifest_sha256 != direct_result.source_manifest_sha256
                or receipt.database_placement_outcome != direct_result.outcome
            ):
                raise ValueError("Phase Receipt database summary does not bind the exact placement Result")
        elif isinstance(placement, PreprocessingStagedDatabasePlacementEvidence):
            staged_result = placement.result
            lease = placement.lease_outcome
            cold_family = (
                isinstance(staged_result, DatabaseReplicaColdResult)
                and staged_result.outcome == DatabasePlacementOutcomeKind.REPLICA_COLD
                and isinstance(lease, DatabaseReplicaLeaseTerminalEvidence)
                and lease.database_replica_cold_result_digest == database_replica_cold_result_digest(staged_result)
            )
            warm_family = (
                isinstance(staged_result, DatabaseReplicaWarmResult)
                and staged_result.outcome == DatabasePlacementOutcomeKind.REPLICA_WARM
                and isinstance(lease, DatabaseReplicaWarmLeaseTerminalEvidence)
                and lease.database_replica_warm_result_digest == database_replica_warm_result_digest(staged_result)
            )
            if (
                not isinstance(staged_result, DatabaseReplicaColdResult | DatabaseReplicaWarmResult)
                or placement.failure is not None
                or not placement.science_started
                or staged_result.requested_policy
                not in {DatabaseAccessPolicy.STAGE_REQUIRED, DatabaseAccessPolicy.STAGE_PREFERRED}
                or staged_result.branch_kind != "staged"
                or staged_result.selected_container_root not in (SELECTED_DATABASE_ROOT, LEGACY_SELECTED_DATABASE_ROOT)
                or staged_result.verification != "metadata-verified"
                or not (cold_family or warm_family)
                or not isinstance(
                    lease,
                    DatabaseReplicaLeaseTerminalEvidence | DatabaseReplicaWarmLeaseTerminalEvidence,
                )
                or not lease.kernel_started
                or not lease.held_through_kernel_exit
                or lease.replica_manifest_sha256 != staged_result.replica_manifest_sha256
                or receipt.database_requested_policy != staged_result.requested_policy
                or receipt.database_source_manifest_sha256 != staged_result.source_manifest_sha256
                or receipt.database_placement_outcome != staged_result.outcome
            ):
                raise ValueError("Phase finalized payload requires exact successful staged database authority")
        else:
            raise ValueError("Phase finalized payload rejects unsupported Database Placement evidence")
        adoption = action_evidence.carry_forward_adoption
        if (adoption is None) != (not receipt.carry_forward_closure):
            raise ValueError("Phase Receipt carry closure must match current adoption evidence")
        if adoption is not None:
            current = receipt.carry_forward_closure[-1]
            if (
                current.attempt_carry_forward_id != adoption.attempt_carry_forward_id
                or current.attempt_carry_forward_digest != adoption.attempt_carry_forward_digest
                or current.target_adoption_evidence_digest != adoption.digest
            ):
                raise ValueError("Phase Receipt carry closure does not bind current adoption")
        if canonical_mapping_digest(content_validation.to_mapping()) != receipt.content_validation_evidence_digest:
            raise ValueError("Phase Receipt does not bind the exact content-validation evidence")
        if artifact_set.artifact_set_id != receipt.output_artifact_set_id:
            raise ValueError("Phase Receipt does not bind the exact output Artifact Set")
        if artifact_location.artifact_location_id != receipt.artifact_location_id:
            raise ValueError("Phase Receipt does not bind the exact Artifact Location")
        if artifact_set.chunks[0].sha256 != chunk_manifest.digest:
            raise ValueError("Phase finalized payload root does not reference its chunk manifest")
        PreprocessingHandoffBundle(
            chunk_manifest=chunk_manifest,
            artifact_set=artifact_set,
            artifact_location=artifact_location,
            content_validation=content_validation,
        )
        if action_evidence.outcome != "succeeded" or content_validation.outcome != "passed":
            raise ValueError("Phase finalized payload requires successful scientific and content evidence")
        action_digest = canonical_mapping_digest(action_evidence.to_mapping())
        if content_validation.action_evidence_digest != action_digest:
            raise ValueError("content validation does not bind the exact action evidence")
        if (
            self.scheduler_evidence.action_id != receipt.action_id
            or action_evidence.action_id != receipt.action_id
            or content_validation.action_id != receipt.action_id
            or self.scheduler_evidence.job_id != receipt.slurm_job_id
            or action_evidence.adapter_version != receipt.adapter_version
        ):
            raise ValueError("Phase finalized payload provenance does not match its receipt")
        if artifact_set.member_count != receipt.member_count or artifact_set.logical_bytes != receipt.logical_bytes:
            raise ValueError("Phase finalized payload counts do not match its receipt")
        finalized = _parse_timestamp(receipt.finalized_at)
        if any(
            _parse_timestamp(observed) > finalized
            for observed in (
                self.scheduler_evidence.observed_at,
                action_evidence.finished_at,
                content_validation.finished_at,
            )
        ):
            raise ValueError("Phase Receipt cannot predate required terminal evidence")

    def _validate_folding_payload(self) -> None:
        receipt = self.receipt
        if (
            self.action_evidence is not None
            or self.chunk_manifest is not None
            or self.artifact_set is not None
            or self.artifact_location is not None
            or self.content_validation is not None
        ):
            raise ValueError("folding Phase finalized payload rejects preprocessing chunk and handoff fields")
        if (
            self.scheduler_evidence.phase_run_id != receipt.phase_run_id
            or self.scheduler_evidence.attempt_id != receipt.attempt_id
            or self.scheduler_evidence.phase_runspec_digest != receipt.phase_runspec_digest
            or self.scheduler_evidence.action_id != receipt.action_id
            or self.scheduler_evidence.job_id != receipt.slurm_job_id
        ):
            raise ValueError("Phase finalized payload scheduler evidence does not bind its receipt")
        if not receipt.action_id.startswith("canonical-pair-"):
            raise ValueError("Phase finalized payload requires the terminal canonical-pair action")
        if self.folding_action_evidence_digests != receipt.folding_action_evidence_digests:
            raise ValueError("Phase Receipt does not bind the exact folding action evidence")
        if self.canonical_pair_index_digest != receipt.canonical_pair_index_digest:
            raise ValueError("Phase Receipt does not bind the exact canonical-pair index digest")
        finalized = _parse_timestamp(receipt.finalized_at)
        if _parse_timestamp(self.scheduler_evidence.observed_at) > finalized:
            raise ValueError("Phase Receipt cannot predate required terminal evidence")

    def to_mapping(self) -> dict[str, object]:
        if _receipt_family(self.receipt.action_id) == "preprocessing":
            return {
                "schema_version": self.schema_version,
                "scheduler_evidence": self.scheduler_evidence.to_mapping(),
                "action_evidence": cast("PreprocessingChunkActionEvidence", self.action_evidence).to_mapping(),
                "chunk_manifest": cast("MsaChunkManifest", self.chunk_manifest).to_mapping(),
                "artifact_set": cast("MsaArtifactSetManifest", self.artifact_set).to_mapping(),
                "artifact_location": cast("VerifiedLocalBundledArtifactLocation", self.artifact_location).to_mapping(),
                "content_validation": cast(
                    "PreprocessingContentValidationEvidence", self.content_validation
                ).to_mapping(),
                "receipt": self.receipt.to_mapping(),
            }
        return {
            "schema_version": self.schema_version,
            "scheduler_evidence": self.scheduler_evidence.to_mapping(),
            "folding_action_evidence_digests": list(self.folding_action_evidence_digests),
            "canonical_pair_index_digest": self.canonical_pair_index_digest,
            "receipt": self.receipt.to_mapping(),
        }


@dataclass(frozen=True)
class PhaseFinalizedEvent:
    """Terminal append-only event that publishes one Phase Receipt."""

    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseFinalizedPayload
    event_type: PhaseFinalizedEventType = "phase-finalized"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PhaseFinalizedEvent")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence <= 1:
            raise ValueError("phase-finalized event sequence must be greater than one")
        if self.event_type != "phase-finalized":
            raise ValueError("terminal Phase Event must be named 'phase-finalized'")
        if self.phase_run_id != self.payload.receipt.phase_run_id or self.attempt_id != self.payload.receipt.attempt_id:
            raise ValueError("phase-finalized event identity must match its receipt")
        if self.occurred_at != self.payload.receipt.finalized_at:
            raise ValueError("phase-finalized event timestamp must match its receipt")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "occurred_at": self.occurred_at,
            "payload": self.payload.to_mapping(),
        }


def provided_successful_scheduler_evidence_from_mapping(
    payload: Mapping[str, object],
) -> ProvidedSuccessfulSchedulerEvidence:
    _strict(
        payload,
        {
            "schema_version",
            "evidence_kind",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "action_id",
            "job_id",
            "source",
            "state",
            "exit_code",
            "observed_at",
        },
        "ProvidedSuccessfulSchedulerEvidence",
    )
    return ProvidedSuccessfulSchedulerEvidence(
        schema_version=_schema(payload, "ProvidedSuccessfulSchedulerEvidence"),
        evidence_kind=cast("SchedulerEvidenceKind", _str(payload, "evidence_kind")),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        phase_runspec_digest=_str(payload, "phase_runspec_digest"),
        action_id=_str(payload, "action_id"),
        job_id=_str(payload, "job_id"),
        source=cast("SchedulerEvidenceSource", _str(payload, "source")),
        state=_str(payload, "state"),
        exit_code=_str(payload, "exit_code"),
        observed_at=_str(payload, "observed_at"),
    )


def phase_receipt_from_mapping(payload: Mapping[str, object]) -> PhaseReceipt:
    action_id = _str(payload, "action_id")
    family = _receipt_family(action_id)
    _strict(
        payload,
        _PREPROCESSING_RECEIPT_FIELDS if family == "preprocessing" else _FOLDING_RECEIPT_FIELDS,
        "PhaseReceipt",
    )
    closure = payload.get("carry_forward_closure")
    if "carry_forward_closure" in payload and not isinstance(closure, list):
        raise ValueError("carry_forward_closure must be a list when present")
    carry_forward_closure = tuple(
        attempt_carry_forward_receipt_reference_from_mapping(item)
        for item in cast("list[Mapping[str, object]]", closure or [])
    )
    schema_version = _schema(payload, "PhaseReceipt")
    phase_receipt_id = _str(payload, "phase_receipt_id")
    phase_run_id = _str(payload, "phase_run_id")
    attempt_id = _str(payload, "attempt_id")
    phase_plan_digest = _str(payload, "phase_plan_digest")
    phase_runspec_location = _str(payload, "phase_runspec_location")
    phase_runspec_digest = _str(payload, "phase_runspec_digest")
    input_sha256 = _str(payload, "input_sha256")
    input_size_bytes = _int(payload, "input_size_bytes")
    input_location_digest = _str(payload, "input_location_digest")
    cluster_snapshot_digest = _str(payload, "cluster_snapshot_digest")
    runtime_action_digest = _str(payload, "runtime_action_digest")
    adapter_version = _str(payload, "adapter_version")
    slurm_job_id = _str(payload, "slurm_job_id")
    finalized_at = _str(payload, "finalized_at")
    result = _str(payload, "result")
    if family == "preprocessing":
        return PhaseReceipt(
            schema_version=schema_version,
            phase_receipt_id=phase_receipt_id,
            phase_run_id=phase_run_id,
            attempt_id=attempt_id,
            phase_plan_digest=phase_plan_digest,
            phase_runspec_location=phase_runspec_location,
            phase_runspec_digest=phase_runspec_digest,
            input_sha256=input_sha256,
            input_size_bytes=input_size_bytes,
            input_location_digest=input_location_digest,
            output_artifact_set_id=_str(payload, "output_artifact_set_id"),
            artifact_location_id=_str(payload, "artifact_location_id"),
            scheduler_evidence_digest=_str(payload, "scheduler_evidence_digest"),
            action_evidence_digest=_str(payload, "action_evidence_digest"),
            content_validation_evidence_digest=_str(payload, "content_validation_evidence_digest"),
            cluster_snapshot_digest=cluster_snapshot_digest,
            runtime_action_digest=runtime_action_digest,
            database_requested_policy=DatabaseAccessPolicy(_str(payload, "database_requested_policy")),
            database_source_manifest_sha256=_str(payload, "database_source_manifest_sha256"),
            database_placement_outcome=DatabasePlacementOutcomeKind(_str(payload, "database_placement_outcome")),
            adapter_version=adapter_version,
            slurm_job_id=slurm_job_id,
            action_id=action_id,
            member_count=_int(payload, "member_count"),
            logical_bytes=_int(payload, "logical_bytes"),
            finalized_at=finalized_at,
            result=result,
            carry_forward_closure=carry_forward_closure,
        )
    return PhaseReceipt(
        schema_version=schema_version,
        phase_receipt_id=phase_receipt_id,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        phase_plan_digest=phase_plan_digest,
        phase_runspec_location=phase_runspec_location,
        phase_runspec_digest=phase_runspec_digest,
        input_sha256=input_sha256,
        input_size_bytes=input_size_bytes,
        input_location_digest=input_location_digest,
        cluster_snapshot_digest=cluster_snapshot_digest,
        runtime_action_digest=runtime_action_digest,
        adapter_version=adapter_version,
        slurm_job_id=slurm_job_id,
        action_id=action_id,
        finalized_at=finalized_at,
        result=result,
        canonical_pair_index_digest=_str(payload, "canonical_pair_index_digest"),
        folding_action_evidence_digests=_str_tuple(payload, "folding_action_evidence_digests"),
        carry_forward_closure=carry_forward_closure,
    )


def phase_finalized_payload_from_mapping(payload: Mapping[str, object]) -> PhaseFinalizedPayload:
    receipt = phase_receipt_from_mapping(_mapping(payload, "receipt"))
    family = _receipt_family(receipt.action_id)
    if family == "preprocessing":
        _strict(payload, _PREPROCESSING_FINALIZED_PAYLOAD_FIELDS, "PhaseFinalizedPayload")
        return PhaseFinalizedPayload(
            schema_version=_schema(payload, "PhaseFinalizedPayload"),
            scheduler_evidence=provided_successful_scheduler_evidence_from_mapping(
                _mapping(payload, "scheduler_evidence")
            ),
            action_evidence=preprocessing_chunk_action_evidence_from_mapping(_mapping(payload, "action_evidence")),
            chunk_manifest=msa_chunk_manifest_from_mapping(_mapping(payload, "chunk_manifest")),
            artifact_set=msa_artifact_set_manifest_from_mapping(_mapping(payload, "artifact_set")),
            artifact_location=verified_local_bundled_artifact_location_from_mapping(
                _mapping(payload, "artifact_location")
            ),
            content_validation=preprocessing_content_validation_evidence_from_mapping(
                _mapping(payload, "content_validation")
            ),
            receipt=receipt,
        )
    _strict(payload, _FOLDING_FINALIZED_PAYLOAD_FIELDS, "PhaseFinalizedPayload")
    return PhaseFinalizedPayload(
        schema_version=_schema(payload, "PhaseFinalizedPayload"),
        scheduler_evidence=provided_successful_scheduler_evidence_from_mapping(_mapping(payload, "scheduler_evidence")),
        folding_action_evidence_digests=_str_tuple(payload, "folding_action_evidence_digests"),
        canonical_pair_index_digest=_str(payload, "canonical_pair_index_digest"),
        receipt=receipt,
    )


def phase_finalized_event_from_mapping(payload: Mapping[str, object]) -> PhaseFinalizedEvent:
    _strict(
        payload,
        {"schema_version", "sequence", "event_type", "phase_run_id", "attempt_id", "occurred_at", "payload"},
        "PhaseFinalizedEvent",
    )
    event_type = _str(payload, "event_type")
    if event_type != "phase-finalized":
        raise ValueError(f"unsupported terminal Phase Event type: {event_type!r}")
    return PhaseFinalizedEvent(
        schema_version=_schema(payload, "PhaseFinalizedEvent"),
        sequence=_int(payload, "sequence"),
        event_type=cast("PhaseFinalizedEventType", event_type),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        occurred_at=_str(payload, "occurred_at"),
        payload=phase_finalized_payload_from_mapping(_mapping(payload, "payload")),
    )


def _validate_run_attempt_action(run_id: str, attempt_id: str, action_id: str) -> None:
    if _PHASE_RUN_ID.fullmatch(run_id) is None or _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise ValueError("record has invalid Phase Run or Attempt identity")
    if _ACTION_ID.fullmatch(action_id) is None:
        raise ValueError("record has invalid Runtime Action identity")


def _strict(payload: Mapping[str, object], allowed: Set[str], name: str) -> None:
    if "schema_version" not in payload:
        raise ValueError(f"missing explicit schema_version at {name}")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")


def _schema(payload: Mapping[str, object], name: str) -> int:
    return validate_schema_version(payload.get("schema_version"), record_name=name)


def _str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _str_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError("folding_action_evidence_digests must be a list of non-empty strings")
    return tuple(value)


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _validate_schema(value: int, name: str) -> None:
    if validate_schema_version(value, record_name=name) != value:
        raise ValueError(f"{name} schema_version must be explicit")


def _sha(value: str, name: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _positive(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _timestamp(value: str, name: str) -> None:
    if not value.endswith("Z"):
        raise ValueError(f"{name} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be an explicit UTC timestamp") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{name} must be an explicit UTC timestamp")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")


__all__ = [
    "PhaseFinalizedEvent",
    "PhaseFinalizedEventType",
    "PhaseFinalizedPayload",
    "PhaseReceipt",
    "ProvidedSuccessfulSchedulerEvidence",
    "SchedulerEvidenceKind",
    "phase_finalized_event_from_mapping",
    "phase_finalized_payload_from_mapping",
    "phase_receipt_from_mapping",
    "phase_receipt_id",
    "provided_successful_scheduler_evidence_from_mapping",
]
