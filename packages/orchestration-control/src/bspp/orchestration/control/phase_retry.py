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

"""Locked, scheduler-free creation of one immutable successor Phase Attempt.

Folding Retry auto-derives a folding-specific carry-forward record from the
predecessor's authenticated per-rank journals: every completion whose full
authority tuple and output path/size/SHA-256 still verify is carried forward,
with no caller-authored subset and no option to omit a valid completion; a
zero-result journal scan omits the record. This revises the prior
folding carry-forward prohibition. Preprocessing Retry keeps the explicit
caller-authored subset request pattern; postprocessing Retry
rejects carry-forward and delegates to its dedicated coordinator.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from bspp.orchestration.contract.folding_carry_forward import (
    FoldingCarryAncestorReference,
    FoldingCarryForwardContent,
    FoldingCarryForwardOutput,
    FoldingCarryForwardRecord,
    FoldingCarryForwardReference,
    folding_carry_forward_id,
)
from bspp.orchestration.contract.folding_shard import (
    FoldShardProjection,
    FoldShardProjectionBinding,
    fold_shard_projection_from_mapping,
)
from bspp.orchestration.contract.phase import (
    FoldingPhasePlan,
    FoldingPhaseRunSpec,
    PhaseRunSpec,
    canonical_mapping_digest,
)
from bspp.orchestration.contract.phase_carry_forward import AttemptCarryForwardRecord
from bspp.orchestration.contract.phase_retry import (
    PhaseAttemptRetriedEvent,
    PhaseAttemptRetriedPayload,
    compare_retry_invariants,
    phase_input_set_identity_digest,
    phase_retry_id,
    phase_scientific_identity_digest,
)
from bspp.orchestration.contract.phase_state import PhaseAttempt
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION
from bspp.orchestration.control.folding_phase_adapter import (
    _kernel_image_for_action,
    materialize_folding_attempt_runspec,
)
from bspp.orchestration.control.folding_phase_types import (
    FoldingPhaseAttemptOperationalSelection,
    folding_qualification_tuple_id,
)
from bspp.orchestration.control.phase_action_evidence_attestation import (
    ensure_phase_action_evidence_attestation,
)
from bspp.orchestration.control.phase_adapters import phase_authority_family
from bspp.orchestration.control.phase_attempt_materialization import (
    PhaseAttemptOperationalSelection,
    materialize_preprocessing_attempt_runspec,
    resolve_phase_attempt_operational_selection,
)
from bspp.orchestration.control.phase_authority import (
    PhaseAuthorityStore,
    PhaseAuthorityValidation,
    validate_phase_run_id,
)
from bspp.orchestration.control.phase_carry_forward import (
    derive_attempt_carry_forward,
    load_attempt_carry_forward_request,
)
from bspp.orchestration.control.phase_lifecycle import require_retryable_phase_attempt
from bspp.orchestration.control.phase_materialization import _resolve_folding_operational_selection
from bspp.orchestration.control.postprocessing_authority_reader import (
    reject_historical_postprocessing_mutation,
)
from bspp.orchestration.control.postprocessing_phase_retry import (
    PostprocessingRetryResult,
    retry_postprocessing_phase,
)
from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.control.transport import RemoteSlurmTransport

_SHA256 = re.compile(r"[0-9a-f]{64}")

Clock = Callable[[], datetime]


@dataclass(frozen=True)
class PhaseRetryResult:
    phase_run_id: str
    predecessor_attempt_id: str
    successor_attempt_id: str
    retry_id: str
    phase_runspec_digest: str
    phase_runspec_location: str
    phase_plan_digest: str
    input_set_identity_digest: str
    scientific_identity_digest: str
    selected_cluster_profile: str
    attempt_carry_forward_id: str | None = None
    attempt_carry_forward_digest: str | None = None
    carried_content_count: int | None = None
    status: str = "materialized"

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "phase_run_id": self.phase_run_id,
            "predecessor_attempt_id": self.predecessor_attempt_id,
            "successor_attempt_id": self.successor_attempt_id,
            "retry_id": self.retry_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "phase_runspec_location": self.phase_runspec_location,
            "phase_plan_digest": self.phase_plan_digest,
            "input_set_identity_digest": self.input_set_identity_digest,
            "scientific_identity_digest": self.scientific_identity_digest,
            "selected_cluster_profile": self.selected_cluster_profile,
            "status": self.status,
        }
        if self.attempt_carry_forward_id is not None:
            result.update(
                {
                    "attempt_carry_forward_id": self.attempt_carry_forward_id,
                    "attempt_carry_forward_digest": self.attempt_carry_forward_digest,
                    "carried_content_count": self.carried_content_count,
                }
            )
        return result

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


def retry_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    config_path: Path,
    profile_name: str | None = None,
    source_repo: Path | None = None,
    carry_forward_path: Path | None = None,
    clock: Clock | None = None,
    authority_store: PhaseAuthorityStore | None = None,
    evidence_transport: RemoteSlurmTransport | None = None,
) -> PhaseRetryResult | PostprocessingRetryResult:
    """Materialize exactly one clean successor Attempt without external effects.

    Folding successor Attempts auto-derive their carry-forward record from the
    predecessor's verified rank journals; preprocessing uses an
    explicit caller-authored subset, and postprocessing rejects carry-forward.
    """
    family = phase_authority_family(authority_root, phase_run_id)
    if family == "postprocessing":
        reject_historical_postprocessing_mutation(authority_root, phase_run_id)
        if carry_forward_path is not None:
            raise ValueError("postprocessing Phase Retry does not support --carry-forward")
        if authority_store is not None or evidence_transport is not None:
            raise ValueError("postprocessing Phase Retry does not accept preprocessing authority dependencies")
        return retry_postprocessing_phase(
            phase_run_id,
            authority_root=authority_root,
            config_path=config_path,
            profile_name=profile_name,
            source_repo=source_repo,
            clock=clock,
        )
    validate_phase_run_id(phase_run_id)
    if authority_store is not None and authority_store.authority_root != authority_root:
        raise ValueError("injected PhaseAuthorityStore root must match authority_root")
    store = authority_store or PhaseAuthorityStore(authority_root)
    with store.phase_operation_lock(phase_run_id):
        authority = store.validate(phase_run_id)
        if not authority.current_runspec_projection_complete:
            completed = store.publish_current_attempt_runspec_projection(phase_run_id)
            return _result_from_authority(completed)

        predecessor_outcome = require_retryable_phase_attempt(authority.lifecycle)
        predecessor = authority.current_attempt
        if predecessor.ordinal >= 9999:
            raise ValueError("Phase Retry cannot exceed Attempt ordinal 9999")
        successor_ordinal = predecessor.ordinal + 1
        successor_attempt_id = f"attempt-{successor_ordinal:04d}"
        now = (clock or _utc_now)()
        materialized_at = _format_timestamp(now)
        carry_request = None
        if carry_forward_path is not None and family != "folding":
            carry_request = load_attempt_carry_forward_request(carry_forward_path)
            transport = evidence_transport or RemoteSlurmTransport(
                kind=authority.phase_runspec.cluster.transport,
                ssh_target=authority.phase_runspec.cluster.ssh_target,
            )
            authority = ensure_phase_action_evidence_attestation(
                authority,
                store=store,
                transport=transport,
                clock=lambda: now,
            )
        selected_profile = profile_name or authority.phase_runspec.cluster.profile_name
        phase_plan = authority.phase_plan
        operational: PhaseAttemptOperationalSelection | FoldingPhaseAttemptOperationalSelection
        successor_runspec: PhaseRunSpec | FoldingPhaseRunSpec
        carry_record: AttemptCarryForwardRecord | FoldingCarryForwardRecord | None
        if isinstance(phase_plan, FoldingPhasePlan):
            profile = resolve_cluster_profile(selected_profile, config_path=config_path)
            operational = _resolve_folding_operational_selection(profile, phase_plan.payload.backend)
            successor_runspec = materialize_folding_attempt_runspec(
                phase_run_id=phase_run_id,
                attempt_id=successor_attempt_id,
                phase_plan=phase_plan,
                materialized_at=materialized_at,
                operational=operational,
            )
            carry_record, successor_runspec = _derive_folding_carry_forward(
                authority=authority,
                successor_runspec=successor_runspec,
                declared_at=materialized_at,
                transport=evidence_transport
                or RemoteSlurmTransport(
                    kind=authority.phase_runspec.cluster.transport,
                    ssh_target=authority.phase_runspec.cluster.ssh_target,
                ),
            )
        else:
            operational = resolve_phase_attempt_operational_selection(
                selected_profile,
                config_path=config_path,
                source_repo=source_repo or Path.cwd(),
                now=now,
            )
            successor_runspec = materialize_preprocessing_attempt_runspec(
                phase_run_id=phase_run_id,
                attempt_id=successor_attempt_id,
                phase_plan=phase_plan,
                materialized_at=materialized_at,
                operational=operational,
                rematerialize_runtime_image=True,
            )
            carry_record = None
            if carry_request is not None:
                carry_record, successor_runspec = derive_attempt_carry_forward(
                    authority=authority,
                    successor_runspec=successor_runspec,
                    target_attempt_ordinal=successor_ordinal,
                    request=carry_request,
                    declared_at=materialized_at,
                )
        compare_retry_invariants(authority.phase_plan, authority.phase_runspec, successor_runspec)
        if isinstance(phase_plan, FoldingPhasePlan):
            predecessor_runspec = authority.phase_runspec
            if not isinstance(predecessor_runspec, FoldingPhaseRunSpec) or not isinstance(
                successor_runspec, FoldingPhaseRunSpec
            ):
                raise ValueError("folding Phase Retry requires folding predecessor and successor RunSpecs")
            _require_folding_topology_invariance(predecessor_runspec, successor_runspec)
        successor = PhaseAttempt(
            attempt_id=successor_attempt_id,
            ordinal=successor_ordinal,
            phase_runspec_location=f"attempts/{successor_attempt_id}/phase-runspec.json",
            phase_runspec_digest=successor_runspec.digest,
            created_at=materialized_at,
        )
        input_digest = phase_input_set_identity_digest(authority.phase_plan)
        scientific_digest = phase_scientific_identity_digest(authority.phase_plan)
        retry_identity = phase_retry_id(
            phase_run_id=phase_run_id,
            predecessor_attempt_id=predecessor.attempt_id,
            successor_attempt_id=successor_attempt_id,
            predecessor_outcome=predecessor_outcome,
            predecessor_phase_runspec_digest=authority.phase_runspec.digest,
            phase_plan_digest=authority.phase_plan.digest,
            input_set_identity_digest=input_digest,
            scientific_identity_digest=scientific_digest,
            selected_cluster_profile=selected_profile,
            successor_phase_runspec_digest=successor_runspec.digest,
        )
        payload = PhaseAttemptRetriedPayload(
            retry_id=retry_identity,
            predecessor_attempt_id=predecessor.attempt_id,
            predecessor_phase_runspec_digest=authority.phase_runspec.digest,
            predecessor_outcome=predecessor_outcome,
            phase_plan_digest=authority.phase_plan.digest,
            input_set_identity_digest=input_digest,
            scientific_identity_digest=scientific_digest,
            selected_cluster_profile=selected_profile,
            successor_attempt=successor,
            successor_phase_runspec=successor_runspec,
            carry_forward_record=carry_record,
        )
        completed = store.append_attempt_retried_event(
            phase_run_id,
            lambda sequence: PhaseAttemptRetriedEvent(
                sequence=sequence,
                phase_run_id=phase_run_id,
                attempt_id=successor_attempt_id,
                occurred_at=materialized_at,
                payload=payload,
            ),
        )
        return _result_from_authority(completed)


def _result_from_authority(authority: PhaseAuthorityValidation) -> PhaseRetryResult:
    matches = tuple(
        event
        for event in authority.events
        if isinstance(event, PhaseAttemptRetriedEvent) and event.attempt_id == authority.current_attempt.attempt_id
    )
    if len(matches) != 1:
        raise ValueError("current retried Phase Attempt requires exactly one Retry authority event")
    payload = matches[0].payload
    record = payload.carry_forward_record
    return PhaseRetryResult(
        phase_run_id=authority.phase_run.phase_run_id,
        predecessor_attempt_id=payload.predecessor_attempt_id,
        successor_attempt_id=authority.current_attempt.attempt_id,
        retry_id=payload.retry_id,
        phase_runspec_digest=authority.phase_runspec.digest,
        phase_runspec_location=authority.current_attempt.phase_runspec_location,
        phase_plan_digest=payload.phase_plan_digest,
        input_set_identity_digest=payload.input_set_identity_digest,
        scientific_identity_digest=payload.scientific_identity_digest,
        selected_cluster_profile=payload.selected_cluster_profile,
        attempt_carry_forward_id=(_carry_forward_id(record) if record is not None else None),
        attempt_carry_forward_digest=(record.digest if record is not None else None),
        carried_content_count=(len(record.content) if record is not None else None),
    )


def _carry_forward_id(record: AttemptCarryForwardRecord | FoldingCarryForwardRecord) -> str:
    """Return one carry record's canonical id across both phase families."""
    if isinstance(record, FoldingCarryForwardRecord):
        return record.folding_carry_forward_id
    return record.attempt_carry_forward_id


@dataclass(frozen=True)
class _FoldingJournalOutput:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class _FoldingJournalEvent:
    """Control-side strict mirror of the Runtime rank journal event."""

    phase_run_id: str
    attempt_id: str
    rank: int
    fold_action_id: str
    fold_action_digest: str
    shard_projection_sha256: str
    shard_projection_worker_count: int
    shard_projection_lpt_version: int
    predecessor_digest: str
    target_id: str
    sequence_sha256: str
    backend: str
    qualification_tuple_id: str
    outputs: tuple[_FoldingJournalOutput, ...]
    event_kind: str = "native"
    source_attempt_id: str | None = None
    carry_record_digest: str | None = None


_FOLDING_JOURNAL_FIELDS = frozenset(
    {
        "schema_version",
        "phase_run_id",
        "attempt_id",
        "rank",
        "fold_action_id",
        "fold_action_digest",
        "shard_projection_sha256",
        "shard_projection_worker_count",
        "shard_projection_lpt_version",
        "predecessor_digest",
        "target_id",
        "sequence_sha256",
        "description",
        "backend",
        "qualification_tuple_id",
        "outputs",
        "event_kind",
        "source_attempt_id",
        "carry_record_digest",
    }
)


def _parse_folding_journal_event(payload: Mapping[str, object]) -> _FoldingJournalEvent:
    unknown = sorted(set(payload) - _FOLDING_JOURNAL_FIELDS)
    if unknown:
        raise ValueError(f"unknown folding journal field(s): {', '.join(unknown)}")
    event_kind = payload.get("event_kind")
    if event_kind is None:
        event_kind = "native"
    elif event_kind not in {"native", "adopted"}:
        raise ValueError("folding journal event_kind must be native or adopted")
    source_attempt_id = _journal_optional_str(payload, "source_attempt_id")
    carry_record_digest = _journal_optional_sha256(payload, "carry_record_digest")
    if event_kind == "adopted":
        if source_attempt_id is None or carry_record_digest is None:
            raise ValueError("adopted folding journal event requires source_attempt_id and carry_record_digest")
    elif source_attempt_id is not None or carry_record_digest is not None:
        raise ValueError("native folding journal event must not declare adopted provenance fields")
    return _FoldingJournalEvent(
        phase_run_id=_journal_str(payload, "phase_run_id"),
        attempt_id=_journal_str(payload, "attempt_id"),
        rank=_journal_int(payload, "rank"),
        fold_action_id=_journal_str(payload, "fold_action_id"),
        fold_action_digest=_journal_str(payload, "fold_action_digest"),
        shard_projection_sha256=_journal_str(payload, "shard_projection_sha256"),
        shard_projection_worker_count=_journal_int(payload, "shard_projection_worker_count"),
        shard_projection_lpt_version=_journal_int(payload, "shard_projection_lpt_version"),
        predecessor_digest=_journal_sha256(payload, "predecessor_digest"),
        target_id=_journal_str(payload, "target_id"),
        sequence_sha256=_journal_sha256(payload, "sequence_sha256"),
        backend=_journal_str(payload, "backend"),
        qualification_tuple_id=_journal_str(payload, "qualification_tuple_id"),
        event_kind=event_kind,
        source_attempt_id=source_attempt_id,
        carry_record_digest=carry_record_digest,
        outputs=tuple(
            _FoldingJournalOutput(
                path=_journal_str(item, "path"),
                size=_journal_int(item, "size"),
                sha256=_journal_str(item, "sha256"),
            )
            for item in _journal_outputs(payload)
        ),
    )


def _journal_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"folding journal {key} must be a non-empty string")
    return value


def _journal_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"folding journal {key} must be an integer")
    return value


def _journal_sha256(payload: Mapping[str, object], key: str) -> str:
    value = _journal_str(payload, key)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"folding journal {key} must be a lowercase SHA-256")
    return value


def _journal_optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"folding journal {key} must be a non-empty string")
    return value


def _journal_optional_sha256(payload: Mapping[str, object], key: str) -> str | None:
    value = _journal_optional_str(payload, key)
    if value is not None and _SHA256.fullmatch(value) is None:
        raise ValueError(f"folding journal {key} must be a lowercase SHA-256")
    return value


def _journal_outputs(payload: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    value = payload.get("outputs")
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError("folding journal outputs must be a list of mappings")
    return tuple(item for item in value if isinstance(item, Mapping))


def _read_folding_journal(transport: RemoteSlurmTransport, path: str) -> tuple[_FoldingJournalEvent, ...]:
    raw = transport.read_immutable_text_artifact_no_follow(path)
    if not raw:
        return ()
    terminated = raw.endswith(b"\n")
    lines = raw.split(b"\n")
    if terminated:
        # The trailing empty element is the record terminator, not a line.
        lines = lines[:-1]
    events: list[_FoldingJournalEvent] = []
    for index, line in enumerate(lines):
        is_final = index == len(lines) - 1
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # Only an unterminated final append is a torn write and may be
            # ignored; a complete (newline-terminated) malformed line is a
            # scan error, never an empty scan.
            if is_final and not terminated:
                break
            raise ValueError(f"malformed folding journal line {index} in {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"malformed folding journal line {index} in {path}: expected a JSON object")
        events.append(_parse_folding_journal_event(payload))
    return tuple(events)


def _load_folding_shard_projection(
    authority: PhaseAuthorityValidation,
    binding: FoldShardProjectionBinding,
) -> FoldShardProjection:
    projection_path = authority.authority_path / binding.location
    try:
        raw = projection_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read fold shard projection {projection_path}: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != binding.sha256:
        raise ValueError("fold shard projection SHA-256 mismatch")
    if len(raw) != binding.size_bytes:
        raise ValueError("fold shard projection size mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed fold shard projection {projection_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("fold shard projection must be a JSON object")
    projection = fold_shard_projection_from_mapping(payload)
    if projection.worker_count != binding.worker_count or projection.lpt_version != binding.lpt_version:
        raise ValueError("fold shard projection binding mismatch")
    return projection


def _verify_folding_journal_event(
    event: _FoldingJournalEvent,
    *,
    fold_action_id: str,
    fold_action_digest: str,
    qualification_id: str,
    projection_digest: str,
    worker_count: int,
    lpt_version: int,
    plan_backend: str,
    phase_run_id: str,
    attempt_id: str,
    rank_targets: Mapping[int, tuple[str, ...]],
    predecessor_carry: FoldingCarryForwardRecord | None,
    transport: RemoteSlurmTransport,
) -> None:
    if event.phase_run_id != phase_run_id or event.attempt_id != attempt_id:
        raise ValueError("folding journal event does not bind the predecessor Phase Run/Attempt")
    if event.fold_action_id != fold_action_id or event.fold_action_digest != fold_action_digest:
        raise ValueError("folding journal event does not bind the predecessor fold action")
    if (
        event.shard_projection_sha256 != projection_digest
        or event.shard_projection_worker_count != worker_count
        or event.shard_projection_lpt_version != lpt_version
    ):
        raise ValueError("folding journal event does not bind the predecessor shard projection")
    declared = rank_targets.get(event.rank)
    if declared is None or event.target_id not in declared:
        raise ValueError("folding journal event target is outside its rank's declared shard assignment")
    if event.backend != plan_backend:
        raise ValueError("folding journal event backend does not bind the Phase Plan")
    if event.qualification_tuple_id != qualification_id:
        raise ValueError("folding journal event qualification does not bind the current operational selection")
    if event.event_kind == "adopted":
        if predecessor_carry is None:
            raise ValueError("adopted folding journal event requires a folding predecessor carry record")
        if event.source_attempt_id != predecessor_carry.source_attempt_id:
            raise ValueError(
                "adopted folding journal event source_attempt_id does not bind the predecessor carry record"
            )
        if event.carry_record_digest != predecessor_carry.digest:
            raise ValueError(
                "adopted folding journal event carry_record_digest does not bind the predecessor carry record"
            )
    for output in event.outputs:
        _verify_folding_journal_output(transport, output)


def _verify_folding_journal_output(transport: RemoteSlurmTransport, output: _FoldingJournalOutput) -> None:
    sha_result = transport.command(("sha256sum", output.path))
    if sha_result.returncode != 0:
        raise ValueError(f"folding journal output hash command failed for {output.path}")
    observed_sha = sha_result.stdout.strip().split(maxsplit=1)[0] if sha_result.stdout.strip() else ""
    if observed_sha != output.sha256:
        raise ValueError(f"folding journal output SHA-256 mismatch for {output.path}")
    stat_result = transport.command(("stat", "-c", "%s", output.path))
    if stat_result.returncode != 0:
        raise ValueError(f"folding journal output size command failed for {output.path}")
    try:
        observed_size = int(stat_result.stdout.strip())
    except ValueError as exc:
        raise ValueError(f"folding journal output size is not an integer for {output.path}") from exc
    if observed_size != output.size:
        raise ValueError(f"folding journal output size mismatch for {output.path}")


def _folding_ancestor_closure(
    current: AttemptCarryForwardRecord | FoldingCarryForwardRecord | None,
) -> tuple[FoldingCarryAncestorReference, ...]:
    if current is None:
        return ()
    if not isinstance(current, FoldingCarryForwardRecord):
        raise ValueError("folding carry-forward ancestor closure requires a folding predecessor carry record")
    closure: list[FoldingCarryAncestorReference] = [
        FoldingCarryAncestorReference(
            folding_carry_forward_id=current.folding_carry_forward_id,
            digest=current.digest,
        )
    ]
    seen = {current.folding_carry_forward_id}
    for ancestor in current.ancestor_closure:
        if ancestor.folding_carry_forward_id not in seen:
            closure.append(ancestor)
            seen.add(ancestor.folding_carry_forward_id)
    return tuple(closure)


def _derive_folding_carry_forward(
    *,
    authority: PhaseAuthorityValidation,
    successor_runspec: FoldingPhaseRunSpec,
    declared_at: str,
    transport: RemoteSlurmTransport,
) -> tuple[FoldingCarryForwardRecord | None, FoldingPhaseRunSpec]:
    plan = authority.phase_plan
    runspec = authority.phase_runspec
    if not isinstance(plan, FoldingPhasePlan):
        raise ValueError("folding carry-forward derivation requires a folding Phase Plan")
    if not isinstance(runspec, FoldingPhaseRunSpec):
        raise ValueError("folding carry-forward derivation requires a folding predecessor RunSpec")
    fold_action = next((action for action in runspec.payload.actions if action.action_kind == "fold"), None)
    if fold_action is None:
        raise ValueError("folding carry-forward derivation requires a fold action")
    binding = runspec.payload.fold_shard_projection
    if binding is None or binding.worker_count <= 1:
        # A legacy/unenriched predecessor has no shard authority to enumerate,
        # and a scalar fold dispatch writes no per-rank journals; either way
        # there is no carryable completion set.
        return None, successor_runspec
    projection = _load_folding_shard_projection(authority, binding)
    fold_action_digest = canonical_mapping_digest(fold_action.to_mapping())
    qualification_id = folding_qualification_tuple_id(
        backend=plan.payload.backend,
        kernel_image=_kernel_image_for_action(runspec, fold_action),
        cluster_snapshot_digest=canonical_mapping_digest(runspec.cluster.to_mapping()),
    )
    attempt_root = (
        PurePosixPath(runspec.cluster.project_root) / "bspp-phase-runs" / runspec.phase_run_id / runspec.attempt_id
    )
    fold_action_root = attempt_root / "actions" / fold_action.action_id
    rank_targets = {rank.global_rank: tuple(target.target_id for target in rank.targets) for rank in projection.ranks}
    current_carry = authority.current_carry_forward
    predecessor_carry = current_carry if isinstance(current_carry, FoldingCarryForwardRecord) else None
    has_adopted = predecessor_carry is not None
    events: list[_FoldingJournalEvent] = []
    for rank in projection.ranks:
        native_path = str(fold_action_root / "ranks" / str(rank.global_rank) / "journal.jsonl")
        events.extend(_read_folding_journal(transport, native_path))
        if has_adopted:
            adopted_path = str(fold_action_root / "ranks" / str(rank.global_rank) / "adopted.jsonl")
            events.extend(_read_folding_journal(transport, adopted_path))
    if not events:
        return None, successor_runspec
    verified: dict[tuple[int, str], _FoldingJournalEvent] = {}
    for event in events:
        _verify_folding_journal_event(
            event,
            fold_action_id=fold_action.action_id,
            fold_action_digest=fold_action_digest,
            qualification_id=qualification_id,
            projection_digest=binding.sha256,
            worker_count=binding.worker_count,
            lpt_version=binding.lpt_version,
            plan_backend=plan.payload.backend,
            phase_run_id=runspec.phase_run_id,
            attempt_id=runspec.attempt_id,
            rank_targets=rank_targets,
            predecessor_carry=predecessor_carry,
            transport=transport,
        )
        key = (event.rank, event.target_id)
        if key in verified:
            raise ValueError(f"duplicate folding journal target {event.target_id} for rank {event.rank}")
        verified[key] = event
    if not verified:
        return None, successor_runspec
    predecessor_digests = {event.predecessor_digest for event in verified.values()}
    if len(predecessor_digests) != 1:
        raise ValueError("folding journal events disagree on the predecessor handoff digest")
    content = tuple(
        FoldingCarryForwardContent(
            target_id=event.target_id,
            sequence_sha256=event.sequence_sha256,
            source_action_id=fold_action.action_id,
            source_rank=event.rank,
            outputs=tuple(
                FoldingCarryForwardOutput(output_path=output.path, size_bytes=output.size, sha256=output.sha256)
                for output in event.outputs
            ),
        )
        for _, event in sorted(verified.items())
    )
    ancestor_closure = _folding_ancestor_closure(authority.current_carry_forward)
    content_digest = canonical_mapping_digest(
        {"schema_version": CURRENT_CONTRACT_SCHEMA_VERSION, "content": [item.to_mapping() for item in content]}
    )
    target_attempt_ordinal = authority.current_attempt.ordinal + 1
    identity: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "phase_run_id": runspec.phase_run_id,
        "phase_plan_digest": plan.digest,
        "source_attempt_id": authority.current_attempt.attempt_id,
        "source_attempt_ordinal": authority.current_attempt.ordinal,
        "source_runspec_digest": runspec.digest,
        "target_attempt_id": successor_runspec.attempt_id,
        "target_attempt_ordinal": target_attempt_ordinal,
        "backend": plan.payload.backend,
        "content": [item.to_mapping() for item in content],
        "ancestor_closure": [item.to_mapping() for item in ancestor_closure],
        "content_digest": content_digest,
        "declared_at": declared_at,
    }
    record = FoldingCarryForwardRecord(
        folding_carry_forward_id=folding_carry_forward_id(identity),
        phase_run_id=runspec.phase_run_id,
        phase_plan_digest=plan.digest,
        source_attempt_id=authority.current_attempt.attempt_id,
        source_attempt_ordinal=authority.current_attempt.ordinal,
        source_runspec_digest=runspec.digest,
        target_attempt_id=successor_runspec.attempt_id,
        target_attempt_ordinal=target_attempt_ordinal,
        backend=plan.payload.backend,
        content=content,
        ancestor_closure=ancestor_closure,
        content_digest=content_digest,
        declared_at=declared_at,
    )
    reference = FoldingCarryForwardReference(
        folding_carry_forward_id=record.folding_carry_forward_id,
        digest=record.digest,
        location=f"attempts/{successor_runspec.attempt_id}/folding-carry-forward.json",
    )
    return record, replace(successor_runspec, carry_forward=reference)


def _require_folding_topology_invariance(
    predecessor: FoldingPhaseRunSpec,
    successor: FoldingPhaseRunSpec,
) -> None:
    """Fail closed unless Retry preserves the exact packed fold topology.

    A partial-carry successor must submit the unchanged full N-element array and
    each rank adopts/skips its valid carried targets in place. Any change to the
    array element count, concurrency cap, per-node task count, per-task GPU
    count, derived worker count, LPT version, or canonical shard identity would
    renumber targets and is rejected before materialization.
    """
    predecessor_topology = _folding_topology_identity(predecessor)
    successor_topology = _folding_topology_identity(successor)
    if predecessor_topology != successor_topology:
        raise ValueError(
            "folding Phase Retry topology must be invariant: "
            f"predecessor {predecessor_topology!r} != successor {successor_topology!r}"
        )


def _folding_topology_identity(runspec: FoldingPhaseRunSpec) -> dict[str, object]:
    fold_action = next(action for action in runspec.payload.actions if action.action_kind == "fold")
    resources = fold_action.resources
    binding = runspec.payload.fold_shard_projection
    if binding is None:
        raise ValueError("folding Retry topology invariance requires a fold shard projection binding")
    return {
        "nodes": resources.nodes,
        "tasks_per_node": resources.tasks_per_node,
        "gpus_per_task": resources.gpus_per_task,
        "max_parallel": resources.max_parallel,
        "worker_count": resources.workers,
        "lpt_version": binding.lpt_version,
        "shard_sha256": binding.sha256,
        "shard_size_bytes": binding.size_bytes,
    }


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Phase Retry clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = ["PhaseRetryResult", "retry_phase"]
