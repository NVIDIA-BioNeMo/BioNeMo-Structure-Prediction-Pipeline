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

"""Append-only contracts and identities for immutable Phase Attempt retry."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.folding_carry_forward import (
    FoldingCarryForwardRecord,
    FoldingCarryForwardReference,
    folding_carry_forward_record_from_mapping,
)
from bspp.orchestration.contract.phase import (
    FoldingPhasePlan,
    FoldingPhaseRunSpec,
    PhasePlan,
    PhaseRunSpec,
    canonical_mapping_digest,
    phase_runspec_family_from_mapping,
)
from bspp.orchestration.contract.phase_cancellation import PhaseCancellationLifecycleView
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardRecord,
    AttemptCarryForwardReference,
    attempt_carry_forward_record_from_mapping,
)
from bspp.orchestration.contract.phase_receipt import PhaseFinalizedEvent, PhaseReceipt
from bspp.orchestration.contract.phase_reconciliation import PhaseActionTerminalObservationView
from bspp.orchestration.contract.phase_state import PhaseAttempt, phase_attempt_from_mapping
from bspp.orchestration.contract.phase_submission import PhaseSubmissionLifecycleView
from bspp.orchestration.contract.preprocessing_execution import (
    scientific_canonical_mapping,
    validate_scientific_schema_version,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

PhaseRetryPredecessorOutcome = Literal["failed", "cancelled"]
PhaseRetryEventType = Literal["phase-attempt-retried"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RETRY_ID = re.compile(r"phase-retry-[0-9a-f]{64}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")


def phase_input_set_identity_digest(phase_plan: PhasePlan | FoldingPhasePlan) -> str:
    """Digest the stable ordered input-set identity for a Phase Run."""
    if isinstance(phase_plan, FoldingPhasePlan):
        return _folding_identity_digest(phase_plan, identity_kind="phase-input-set")
    work_input = phase_plan.payload.work_plan.input
    return canonical_mapping_digest(
        {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "identity_kind": "phase-input-set",
            "phase_kind": phase_plan.phase_kind,
            "content_sha256": phase_plan.input_location.sha256,
            "size_bytes": phase_plan.input_location.size_bytes,
            "normalization": work_input.normalization_mode,
            "records": [record.to_mapping() for record in work_input.records],
        }
    )


def phase_scientific_identity_digest(phase_plan: PhasePlan | FoldingPhasePlan) -> str:
    """Digest the frozen scientific and command semantics of one Phase Plan."""
    if isinstance(phase_plan, FoldingPhasePlan):
        return _folding_identity_digest(phase_plan, identity_kind="phase-scientific-execution")
    execution = phase_plan.payload.chunk_execution_intent
    return canonical_mapping_digest(
        {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "identity_kind": "phase-scientific-execution",
            "phase_kind": phase_plan.phase_kind,
            "work_plan": phase_plan.payload.work_plan.to_mapping(),
            "scientific": scientific_canonical_mapping(execution.scientific),
            "database": phase_plan.payload.database.to_mapping(),
            "expected_a3ms": [item.to_mapping() for item in execution.expected_a3ms],
            "action_kind": "preprocessing-chunk",
            "action_id": f"preprocessing-chunk-{phase_plan.payload.work_plan.chunks[0].ordinal:06d}",
            "dependencies": [],
            "runtime": execution.runtime.to_mapping(),
        }
    )


def compare_retry_invariants(
    phase_plan: PhasePlan | FoldingPhasePlan,
    predecessor: PhaseRunSpec | FoldingPhaseRunSpec,
    successor: PhaseRunSpec | FoldingPhaseRunSpec,
) -> None:
    """Fail unless only the reviewed Retry operational fields changed."""
    if isinstance(phase_plan, FoldingPhasePlan):
        _compare_folding_retry_invariants(phase_plan, predecessor, successor)
        return
    if predecessor.phase_plan_digest != phase_plan.digest or successor.phase_plan_digest != phase_plan.digest:
        raise ValueError("Phase Retry must preserve the exact stored Phase Plan digest")
    if predecessor.phase_run_id != successor.phase_run_id:
        raise ValueError("Phase Retry successor must preserve the Phase Run id")
    if predecessor.input_location != phase_plan.input_location or successor.input_location != phase_plan.input_location:
        raise ValueError("Phase Retry must preserve the complete input location")
    predecessor_frozen = _retry_frozen_mapping(predecessor)
    successor_frozen = _retry_frozen_mapping(successor)
    if predecessor_frozen != successor_frozen:
        raise ValueError("Phase Retry changed a non-allowlisted scientific or execution field")
    action = successor.payload.actions[0]
    if action.payload.site.container_image != successor.cluster.runtime_image:
        raise ValueError("Phase Retry action image must equal the qualified successor runtime image")


def _folding_identity_digest(phase_plan: FoldingPhasePlan, *, identity_kind: str) -> str:
    identity: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "identity_kind": identity_kind,
        "phase_kind": "folding",
        "backend": phase_plan.payload.backend,
        "msa_set": phase_plan.payload.msa_set.to_mapping(),
    }
    if identity_kind == "phase-scientific-execution" and phase_plan.payload.bioir_model_policy is not None:
        identity["bioir_model_policy"] = phase_plan.payload.bioir_model_policy.to_mapping()
    return canonical_mapping_digest(identity)


def _compare_folding_retry_invariants(
    phase_plan: FoldingPhasePlan,
    predecessor: PhaseRunSpec | FoldingPhaseRunSpec,
    successor: PhaseRunSpec | FoldingPhaseRunSpec,
) -> None:
    if not isinstance(predecessor, FoldingPhaseRunSpec) or not isinstance(successor, FoldingPhaseRunSpec):
        raise ValueError("folding Phase Retry requires folding Phase RunSpecs")
    if predecessor.phase_plan_digest != phase_plan.digest or successor.phase_plan_digest != phase_plan.digest:
        raise ValueError("Phase Retry must preserve the exact stored Phase Plan digest")
    if predecessor.phase_run_id != successor.phase_run_id:
        raise ValueError("Phase Retry successor must preserve the Phase Run id")
    if predecessor.input_location != phase_plan.input_location or successor.input_location != phase_plan.input_location:
        raise ValueError("Phase Retry must preserve the complete input location")
    if _folding_retry_frozen_mapping(predecessor) != _folding_retry_frozen_mapping(successor):
        raise ValueError("Phase Retry changed a non-allowlisted scientific or execution field")


def _folding_retry_frozen_mapping(runspec: FoldingPhaseRunSpec) -> dict[str, object]:
    mapping = deepcopy(runspec.to_mapping())
    mapping.pop("carry_forward", None)
    mapping.pop("attempt_id")
    mapping.pop("materialized_at")
    mapping.pop("cluster")
    payload = cast("dict[str, object]", mapping["payload"])
    # The attempt-scoped fold shard projection binding (location, SHA-256,
    # size) is re-derived identically for the successor and is not a scientific
    # or execution field; its identity invariance check lands in e13s11.
    payload.pop("fold_shard_projection", None)
    actions = cast("list[dict[str, object]]", payload["actions"])
    for action in actions:
        action.pop("resources")
    return mapping


def _retry_frozen_mapping(runspec: PhaseRunSpec) -> dict[str, object]:
    mapping = deepcopy(runspec.to_mapping())
    mapping.pop("carry_forward", None)
    mapping.pop("attempt_id")
    mapping.pop("materialized_at")
    mapping.pop("cluster")
    payload = cast("dict[str, object]", mapping["payload"])
    database = cast("dict[str, object]", payload["database"])
    _normalize_retry_database_site_authority(database)
    actions = cast("list[dict[str, object]]", payload["actions"])
    for action in actions:
        action.pop("resources")
        action_payload = cast("dict[str, object]", action["payload"])
        site = cast("dict[str, object]", action_payload["site"])
        site.pop("container_image")
    return mapping


def _normalize_retry_database_site_authority(database: dict[str, object]) -> None:
    """Normalize only current-profile database fields reviewed as Retry-operational."""
    database["source_manifest"] = "<current-site-source-manifest>"
    database.pop("source_manifest_sha256")
    database.pop("source_manifest_projection")
    database["staging"] = "<current-site-staging-authority>"
    branches = cast("list[dict[str, object]]", database["branches"])
    for branch in branches:
        for mount_group in ("placement_mounts", "scientific_mounts", "finalization_mounts"):
            mounts = cast("list[dict[str, object]]", branch[mount_group])
            for mount in mounts:
                mount["source"] = "<current-site-mount-source>"


def phase_retry_id(
    *,
    phase_run_id: str,
    predecessor_attempt_id: str,
    successor_attempt_id: str,
    predecessor_outcome: PhaseRetryPredecessorOutcome,
    predecessor_phase_runspec_digest: str,
    phase_plan_digest: str,
    input_set_identity_digest: str,
    scientific_identity_digest: str,
    selected_cluster_profile: str,
    successor_phase_runspec_digest: str,
) -> str:
    body: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "identity_kind": "phase-retry",
        "phase_run_id": phase_run_id,
        "predecessor_attempt_id": predecessor_attempt_id,
        "successor_attempt_id": successor_attempt_id,
        "predecessor_outcome": predecessor_outcome,
        "predecessor_phase_runspec_digest": predecessor_phase_runspec_digest,
        "phase_plan_digest": phase_plan_digest,
        "input_set_identity_digest": input_set_identity_digest,
        "scientific_identity_digest": scientific_identity_digest,
        "selected_cluster_profile": selected_cluster_profile,
        "successor_phase_runspec_digest": successor_phase_runspec_digest,
    }
    return f"phase-retry-{canonical_mapping_digest(body)}"


@dataclass(frozen=True)
class PhaseAttemptRetriedPayload:
    retry_id: str
    predecessor_attempt_id: str
    predecessor_phase_runspec_digest: str
    predecessor_outcome: PhaseRetryPredecessorOutcome
    phase_plan_digest: str
    input_set_identity_digest: str
    scientific_identity_digest: str
    selected_cluster_profile: str
    successor_attempt: PhaseAttempt
    successor_phase_runspec: PhaseRunSpec | FoldingPhaseRunSpec
    carry_forward_record: AttemptCarryForwardRecord | FoldingCarryForwardRecord | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PhaseAttemptRetriedPayload")
        if _RETRY_ID.fullmatch(self.retry_id) is None:
            raise ValueError("Phase Retry id must be a canonical phase-retry id")
        if self.predecessor_outcome not in {"failed", "cancelled"}:
            raise ValueError("Phase Retry predecessor outcome must be failed or cancelled")
        if _ATTEMPT_ID.fullmatch(self.predecessor_attempt_id) is None:
            raise ValueError("Phase Retry predecessor Attempt id must be canonical attempt-NNNN")
        for name, value in (
            ("predecessor Phase RunSpec", self.predecessor_phase_runspec_digest),
            ("Phase Plan", self.phase_plan_digest),
            ("input-set identity", self.input_set_identity_digest),
            ("scientific identity", self.scientific_identity_digest),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError(f"Phase Retry {name} digest must be a lowercase SHA-256")
        if not self.selected_cluster_profile:
            raise ValueError("Phase Retry selected Cluster Profile must be non-empty")
        if self.successor_attempt.ordinal <= 1:
            raise ValueError("Phase Retry successor must follow the initial Attempt")
        expected_predecessor = f"attempt-{self.successor_attempt.ordinal - 1:04d}"
        if self.predecessor_attempt_id != expected_predecessor:
            raise ValueError("Phase Retry predecessor Attempt id must be contiguous with its successor")
        if self.predecessor_attempt_id == self.successor_attempt.attempt_id:
            raise ValueError("Phase Retry successor must differ from its predecessor")
        if self.successor_phase_runspec.attempt_id != self.successor_attempt.attempt_id:
            raise ValueError("Phase Retry successor Attempt and RunSpec ids must match")
        if self.successor_phase_runspec.digest != self.successor_attempt.phase_runspec_digest:
            raise ValueError("Phase Retry successor RunSpec digest must match its Attempt")
        if self.successor_phase_runspec.materialized_at != self.successor_attempt.created_at:
            raise ValueError("Phase Retry successor timestamps must match")
        if self.successor_phase_runspec.phase_plan_digest != self.phase_plan_digest:
            raise ValueError("Phase Retry successor must bind the exact Phase Plan digest")
        if self.successor_phase_runspec.cluster.profile_name != self.selected_cluster_profile:
            raise ValueError("Phase Retry successor cluster must match the selected profile")
        reference = self.successor_phase_runspec.carry_forward
        record = self.carry_forward_record
        if (reference is None) != (record is None):
            raise ValueError("Phase Retry carry-forward record/reference must be present together")
        if reference is not None and record is not None:
            if isinstance(self.successor_phase_runspec, FoldingPhaseRunSpec):
                if not isinstance(reference, FoldingCarryForwardReference) or not isinstance(
                    record, FoldingCarryForwardRecord
                ):
                    raise ValueError("folding Phase Retry carry-forward record/reference must both be folding records")
                if (
                    reference.folding_carry_forward_id != record.folding_carry_forward_id
                    or reference.digest != record.digest
                    or record.target_attempt_id != self.successor_attempt.attempt_id
                    or record.source_attempt_id != self.predecessor_attempt_id
                ):
                    raise ValueError("Phase Retry carry-forward record does not match successor reference")
            else:
                if not isinstance(reference, AttemptCarryForwardReference) or not isinstance(
                    record, AttemptCarryForwardRecord
                ):
                    raise ValueError(
                        "preprocessing Phase Retry carry-forward record/reference must both be preprocessing records"
                    )
                if (
                    reference.attempt_carry_forward_id != record.attempt_carry_forward_id
                    or reference.digest != record.digest
                    or record.target_attempt_id != self.successor_attempt.attempt_id
                    or record.source_attempt_id != self.predecessor_attempt_id
                ):
                    raise ValueError("Phase Retry carry-forward record does not match successor reference")
        expected = phase_retry_id(
            phase_run_id=self.successor_phase_runspec.phase_run_id,
            predecessor_attempt_id=self.predecessor_attempt_id,
            successor_attempt_id=self.successor_attempt.attempt_id,
            predecessor_outcome=self.predecessor_outcome,
            predecessor_phase_runspec_digest=self.predecessor_phase_runspec_digest,
            phase_plan_digest=self.phase_plan_digest,
            input_set_identity_digest=self.input_set_identity_digest,
            scientific_identity_digest=self.scientific_identity_digest,
            selected_cluster_profile=self.selected_cluster_profile,
            successor_phase_runspec_digest=self.successor_phase_runspec.digest,
        )
        if self.retry_id != expected:
            raise ValueError("Phase Retry id does not match its canonical identity preimage")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "retry_id": self.retry_id,
            "predecessor_attempt_id": self.predecessor_attempt_id,
            "predecessor_phase_runspec_digest": self.predecessor_phase_runspec_digest,
            "predecessor_outcome": self.predecessor_outcome,
            "phase_plan_digest": self.phase_plan_digest,
            "input_set_identity_digest": self.input_set_identity_digest,
            "scientific_identity_digest": self.scientific_identity_digest,
            "selected_cluster_profile": self.selected_cluster_profile,
            "successor_attempt": self.successor_attempt.to_mapping(),
            "successor_phase_runspec": self.successor_phase_runspec.to_mapping(),
        }
        if self.carry_forward_record is not None:
            result["carry_forward_record"] = self.carry_forward_record.to_mapping()
        return result


@dataclass(frozen=True)
class PhaseAttemptRetriedEvent:
    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseAttemptRetriedPayload
    event_type: PhaseRetryEventType = "phase-attempt-retried"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PhaseAttemptRetriedEvent")
        if self.sequence <= 1 or self.event_type != "phase-attempt-retried":
            raise ValueError("Phase Retry event must follow initial materialization")
        successor = self.payload.successor_phase_runspec
        if self.phase_run_id != successor.phase_run_id or self.attempt_id != successor.attempt_id:
            raise ValueError("Phase Retry event must bind its successor run and Attempt")
        if (
            self.occurred_at != successor.materialized_at
            or self.occurred_at != self.payload.successor_attempt.created_at
        ):
            raise ValueError("Phase Retry event timestamp must equal successor materialization")

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


@dataclass(frozen=True)
class PhaseAttemptHistoryView:
    """Immutable audit projection of one predecessor Attempt."""

    attempt: PhaseAttempt
    phase_runspec: PhaseRunSpec | FoldingPhaseRunSpec
    outcome: PhaseRetryPredecessorOutcome
    submission: PhaseSubmissionLifecycleView | None
    terminal_observations: tuple[PhaseActionTerminalObservationView, ...]
    cancellation: PhaseCancellationLifecycleView | None
    retry_id: str
    carry_forward: AttemptCarryForwardRecord | FoldingCarryForwardRecord | None = None
    finalized_event: PhaseFinalizedEvent | None = None
    receipt: PhaseReceipt | None = None

    def __post_init__(self) -> None:
        if self.phase_runspec.attempt_id != self.attempt.attempt_id:
            raise ValueError("Phase Attempt history RunSpec must bind its Attempt")
        if self.phase_runspec.digest != self.attempt.phase_runspec_digest:
            raise ValueError("Phase Attempt history RunSpec digest must bind its Attempt")
        if self.outcome not in {"failed", "cancelled"}:
            raise ValueError("Phase Attempt history outcome must be failed or cancelled")
        if _RETRY_ID.fullmatch(self.retry_id) is None:
            raise ValueError("Phase Attempt history requires a canonical Retry id")
        if self.finalized_event is not None or self.receipt is not None:
            raise ValueError("retryable Phase Attempt history cannot contain finalization authority")
        if (self.phase_runspec.carry_forward is None) != (self.carry_forward is None):
            raise ValueError("Phase Attempt history carry record/reference must be present together")
        if self.carry_forward is not None:
            if isinstance(self.phase_runspec, FoldingPhaseRunSpec):
                folding_reference = self.phase_runspec.carry_forward
                assert folding_reference is not None
                if not isinstance(folding_reference, FoldingCarryForwardReference) or not isinstance(
                    self.carry_forward, FoldingCarryForwardRecord
                ):
                    raise ValueError(
                        "folding Phase Attempt history carry record/reference must both be folding records"
                    )
                if folding_reference.folding_carry_forward_id != self.carry_forward.folding_carry_forward_id:
                    raise ValueError("Phase Attempt history carry record does not match its RunSpec")
            else:
                preprocessing_reference = self.phase_runspec.carry_forward
                assert preprocessing_reference is not None
                if not isinstance(preprocessing_reference, AttemptCarryForwardReference) or not isinstance(
                    self.carry_forward, AttemptCarryForwardRecord
                ):
                    raise ValueError(
                        "preprocessing Phase Attempt history carry record/reference must both be preprocessing records"
                    )
                if preprocessing_reference.attempt_carry_forward_id != self.carry_forward.attempt_carry_forward_id:
                    raise ValueError("Phase Attempt history carry record does not match its RunSpec")


def phase_attempt_retried_payload_from_mapping(payload: Mapping[str, object]) -> PhaseAttemptRetriedPayload:
    _require_versions(payload, path=("phase_attempt_retried_payload",))
    _reject_unknown(
        payload,
        {
            "schema_version",
            "retry_id",
            "predecessor_attempt_id",
            "predecessor_phase_runspec_digest",
            "predecessor_outcome",
            "phase_plan_digest",
            "input_set_identity_digest",
            "scientific_identity_digest",
            "selected_cluster_profile",
            "successor_attempt",
            "successor_phase_runspec",
            "carry_forward_record",
        },
        "PhaseAttemptRetriedPayload",
    )
    outcome = _required_str(payload, "predecessor_outcome")
    if outcome not in {"failed", "cancelled"}:
        raise ValueError("unsupported Phase Retry predecessor outcome")
    carry_mapping = payload.get("carry_forward_record")
    if "carry_forward_record" in payload and not isinstance(carry_mapping, Mapping):
        raise ValueError("carry_forward_record must be a mapping when present")
    successor_runspec = phase_runspec_family_from_mapping(_required_mapping(payload, "successor_phase_runspec"))
    if not isinstance(successor_runspec, PhaseRunSpec | FoldingPhaseRunSpec):
        raise ValueError("Phase Retry successor must be a preprocessing or folding Phase RunSpec")
    carry_forward_record: AttemptCarryForwardRecord | FoldingCarryForwardRecord | None
    if carry_mapping is not None:
        if isinstance(successor_runspec, FoldingPhaseRunSpec):
            carry_forward_record = folding_carry_forward_record_from_mapping(
                cast("Mapping[str, object]", carry_mapping)
            )
        else:
            carry_forward_record = attempt_carry_forward_record_from_mapping(
                cast("Mapping[str, object]", carry_mapping)
            )
    else:
        carry_forward_record = None
    return PhaseAttemptRetriedPayload(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PhaseAttemptRetriedPayload"),
        retry_id=_required_str(payload, "retry_id"),
        predecessor_attempt_id=_required_str(payload, "predecessor_attempt_id"),
        predecessor_phase_runspec_digest=_required_str(payload, "predecessor_phase_runspec_digest"),
        predecessor_outcome=cast("PhaseRetryPredecessorOutcome", outcome),
        phase_plan_digest=_required_str(payload, "phase_plan_digest"),
        input_set_identity_digest=_required_str(payload, "input_set_identity_digest"),
        scientific_identity_digest=_required_str(payload, "scientific_identity_digest"),
        selected_cluster_profile=_required_str(payload, "selected_cluster_profile"),
        successor_attempt=phase_attempt_from_mapping(_required_mapping(payload, "successor_attempt")),
        successor_phase_runspec=successor_runspec,
        carry_forward_record=carry_forward_record,
    )


def phase_attempt_retried_event_from_mapping(payload: Mapping[str, object]) -> PhaseAttemptRetriedEvent:
    _require_versions(payload, path=("phase_attempt_retried_event",))
    _reject_unknown(
        payload,
        {"schema_version", "sequence", "event_type", "phase_run_id", "attempt_id", "occurred_at", "payload"},
        "PhaseAttemptRetriedEvent",
    )
    event_type = _required_str(payload, "event_type")
    if event_type != "phase-attempt-retried":
        raise ValueError(f"unsupported Phase Retry event type: {event_type!r}")
    return PhaseAttemptRetriedEvent(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PhaseAttemptRetriedEvent"),
        sequence=_required_int(payload, "sequence"),
        event_type=cast("PhaseRetryEventType", event_type),
        phase_run_id=_required_str(payload, "phase_run_id"),
        attempt_id=_required_str(payload, "attempt_id"),
        occurred_at=_required_str(payload, "occurred_at"),
        payload=phase_attempt_retried_payload_from_mapping(_required_mapping(payload, "payload")),
    )


def _validate_schema(value: int, name: str) -> None:
    if validate_schema_version(value, record_name=name) != value:
        raise ValueError(f"{name} schema_version must be explicit")


def _require_versions(value: object, *, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        if "schema_version" not in value:
            raise ValueError(f"missing explicit schema_version at {'.'.join(path)}")
        if path[-1] == "scientific":
            validate_scientific_schema_version(value["schema_version"], record_name=".".join(path))
        else:
            validate_schema_version(value["schema_version"], record_name=".".join(path))
        for key, nested in value.items():
            if key not in {"schema_version", "database_set", "source_manifest"}:
                _require_versions(nested, path=(*path, str(key)))
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            _require_versions(nested, path=(*path, str(index)))


def _reject_unknown(payload: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


__all__ = [
    "PhaseAttemptHistoryView",
    "PhaseAttemptRetriedEvent",
    "PhaseAttemptRetriedPayload",
    "PhaseRetryEventType",
    "PhaseRetryPredecessorOutcome",
    "compare_retry_invariants",
    "phase_attempt_retried_event_from_mapping",
    "phase_attempt_retried_payload_from_mapping",
    "phase_input_set_identity_digest",
    "phase_retry_id",
    "phase_scientific_identity_digest",
]
