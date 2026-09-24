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

"""Asynchronous direct-Slurm submission for one materialized Phase Attempt."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from bspp.orchestration.contract.folding_carry_forward import FoldingCarryForwardRecord
from bspp.orchestration.contract.folding_shard import (
    FOLD_SHARD_PROJECTION_FILENAME,
    FoldShardProjection,
    fold_shard_projection_from_mapping,
)
from bspp.orchestration.contract.phase import (
    FoldingPhaseRunSpec,
    FoldingRuntimeAction,
    PhaseRunSpec,
    PreprocessingRuntimeAction,
)
from bspp.orchestration.contract.phase_carry_forward import AttemptCarryForwardRecord
from bspp.orchestration.contract.phase_submission import (
    PhaseActionDispatchIntendedEvent,
    PhaseActionDispatchIntendedPayload,
    PhaseActionDispatchRejectedEvent,
    PhaseActionDispatchRejectedPayload,
    PhaseActionSatisfiedWithoutDispatchEvent,
    PhaseActionSatisfiedWithoutDispatchPayload,
    PhaseActionSubmissionPlan,
    PhaseActionSubmittedEvent,
    PhaseActionSubmittedPayload,
    PhaseSubmissionIntendedEvent,
    PhaseSubmissionIntendedPayload,
    PhaseSubmissionLifecycleView,
)
from bspp.orchestration.control.folding_phase_adapter import render_folding_submission_intent
from bspp.orchestration.control.phase_adapters import phase_authority_family
from bspp.orchestration.control.phase_authority import (
    PhaseAuthorityStore,
    PhaseAuthorityValidation,
    require_complete_current_runspec,
)
from bspp.orchestration.control.phase_lifecycle import classify_phase_lifecycle, require_active_phase_attempt
from bspp.orchestration.control.phase_rendering import render_phase_submission_intent
from bspp.orchestration.control.postprocessing_authority_reader import (
    reject_historical_postprocessing_mutation,
)
from bspp.orchestration.control.postprocessing_phase_adapter import (
    PostprocessingLifecycleResult,
    submit_postprocessing_phase,
)
from bspp.orchestration.control.transport import (
    CommandResult,
    CommandRunner,
    RemoteSlurmTransport,
    SlurmAction,
    SlurmSubmissionRejected,
    command_argv,
    default_command_runner,
)

Clock = Callable[[], datetime]


class PhaseSubmissionCorrelationUnresolvedError(ValueError):
    """The original Slurm assignment cannot be proven by durable correlation."""


@dataclass(frozen=True)
class PhaseActionSubmissionResult:
    action_id: str
    job_id: str | None = None

    def to_mapping(self) -> dict[str, object]:
        return {"action_id": self.action_id, "job_id": self.job_id}


@dataclass(frozen=True)
class PhaseSubmissionResult:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    submission_id: str
    actions: tuple[PhaseActionSubmissionResult, ...]
    status: str = "submitted"

    def to_mapping(self) -> dict[str, object]:
        return {
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "submission_id": self.submission_id,
            "status": self.status,
            "actions": [action.to_mapping() for action in self.actions],
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


def submit_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    clock: Clock | None = None,
    authority_store: PhaseAuthorityStore | None = None,
    runner: CommandRunner = default_command_runner,
) -> PhaseSubmissionResult | PostprocessingLifecycleResult:
    """Dispatch every declared action and return after durable job assignment."""
    family = phase_authority_family(authority_root, phase_run_id)
    if family == "postprocessing":
        reject_historical_postprocessing_mutation(authority_root, phase_run_id)
        if authority_store is not None:
            raise ValueError("postprocessing Phase Submission does not accept a preprocessing authority store")
        return submit_postprocessing_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=clock,
            runner=runner,
        )
    is_folding = family == "folding"
    store = authority_store or PhaseAuthorityStore(authority_root)
    now = clock or _utc_now
    with store.phase_operation_lock(phase_run_id):
        authority = store.validate(phase_run_id)
        if authority.submission is not None and authority.submission.status == "failed":
            raise ValueError(f"Phase Submission already has a rejected Runtime Action: {phase_run_id}")
        lifecycle = classify_phase_lifecycle(authority.lifecycle)
        if lifecycle == "accepted":
            raise ValueError(f"Phase Run is already sealed: {phase_run_id}")
        if lifecycle == "failed":
            raise ValueError(f"Phase Submission cannot cross terminal accounting failure: {phase_run_id}")
        require_active_phase_attempt(authority.lifecycle, operation="Phase Submission")
        require_complete_current_runspec(authority, operation="Phase Submission")
        runspec_path, document_sha256, intended = _prepare_submission_material(authority)
        if authority.submission is None:
            intent_time = now()
            _require_current_qualification(authority, now=intent_time, is_folding=is_folding)
            intended_at = _format_timestamp(intent_time)
            authority = store.append_event(
                phase_run_id,
                lambda sequence: PhaseSubmissionIntendedEvent(
                    sequence=sequence,
                    phase_run_id=phase_run_id,
                    attempt_id=authority.phase_runspec.attempt_id,
                    occurred_at=intended_at,
                    payload=intended,
                ),
            )
        else:
            _require_identical_intent(authority.submission, intended)
            intended_at = _submission_intended_at(authority)
        if authority.submission is None:
            raise AssertionError("submission intent append did not replay a submission")
        if authority.submission.status == "submitted":
            return _result(authority)

        transport = RemoteSlurmTransport(
            kind=authority.phase_runspec.cluster.transport,
            ssh_target=authority.phase_runspec.cluster.ssh_target,
            runner=runner,
        )
        for action in _topological_actions(authority.phase_runspec.payload.actions):
            assert authority.submission is not None
            view = next(item for item in authority.submission.actions if item.action_id == action.action_id)
            if view.status == "submitted" or view.status == "satisfied":
                continue
            if view.status == "rejected":
                raise ValueError(f"Runtime Action submission was rejected: {view.action_id}")
            if view.status == "dispatching":
                authority = _recover_dispatched_action(
                    store,
                    authority,
                    view.plan,
                    transport=transport,
                    intended_at=intended_at,
                    now=now,
                )
                continue
            if _is_fully_carried_fold(authority, view.plan):
                authority = _append_satisfied_without_dispatch_event(store, authority, view.plan, now=now)
                continue
            authority = _dispatch_planned_action(
                store,
                authority,
                view.plan,
                runspec_path=runspec_path,
                runspec_document_sha256=document_sha256,
                transport=transport,
                now=now,
            )
        if authority.submission is None or authority.submission.status != "submitted":
            raise ValueError("Phase Submission did not durably assign every Runtime Action")
        return _result(authority)


def _prepare_submission_material(
    authority: PhaseAuthorityValidation,
) -> tuple[Path, str, PhaseSubmissionIntendedPayload]:
    """Rebuild exact submission material without taking a lifecycle lock."""
    runspec_path = authority.authority_path / authority.current_attempt.phase_runspec_location
    runspec_bytes = runspec_path.read_bytes()
    expected_runspec_bytes = (
        json.dumps(authority.phase_runspec.to_mapping(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    if runspec_bytes != expected_runspec_bytes:
        raise ValueError("stored Phase RunSpec bytes differ from canonical materialized authority")
    document_sha256 = hashlib.sha256(runspec_bytes).hexdigest()
    carry_record = authority.current_carry_forward
    carry_document_sha256: str | None = None
    if carry_record is not None:
        if authority.phase_runspec.carry_forward is None:
            raise ValueError("carried authority RunSpec lacks its carry-forward reference")
        carry_path = authority.authority_path / authority.phase_runspec.carry_forward.location
        carry_bytes = carry_path.read_bytes()
        expected_carry_bytes = (
            json.dumps(carry_record.to_mapping(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode()
        if carry_bytes != expected_carry_bytes:
            raise ValueError("stored carry-forward bytes differ from canonical Retry authority")
        carry_document_sha256 = hashlib.sha256(carry_bytes).hexdigest()
    if isinstance(authority.phase_runspec, FoldingPhaseRunSpec):
        if carry_record is not None and not isinstance(carry_record, FoldingCarryForwardRecord):
            raise ValueError("folding Phase Submission requires a folding carry-forward record")
        intended = render_folding_submission_intent(
            phase_runspec=authority.phase_runspec,
            phase_runspec_location=authority.current_attempt.phase_runspec_location,
            phase_runspec_document_sha256=document_sha256,
            carry_forward_record=carry_record if isinstance(carry_record, FoldingCarryForwardRecord) else None,
            carry_forward_document_sha256=carry_document_sha256,
        )
    else:
        if carry_record is not None and not isinstance(carry_record, AttemptCarryForwardRecord):
            raise ValueError("preprocessing Phase Submission requires a preprocessing carry-forward record")
        intended = render_phase_submission_intent(
            authority.phase_runspec,
            phase_runspec_location=authority.current_attempt.phase_runspec_location,
            phase_runspec_document_sha256=document_sha256,
            carry_forward_record=carry_record,
            carry_forward_document_sha256=carry_document_sha256,
        )
    return runspec_path, document_sha256, intended


def _require_existing_submission_intent(
    authority: PhaseAuthorityValidation,
    intended: PhaseSubmissionIntendedPayload,
) -> None:
    """Require the exact previously frozen intent for Resume continuation."""
    if authority.submission is None:
        raise ValueError("Phase Resume requires an existing Phase Submission intent")
    _require_identical_intent(authority.submission, intended)


def _dispatch_planned_action(
    store: PhaseAuthorityStore,
    authority: PhaseAuthorityValidation,
    plan: PhaseActionSubmissionPlan,
    *,
    runspec_path: Path,
    runspec_document_sha256: str,
    transport: RemoteSlurmTransport,
    now: Clock,
) -> PhaseAuthorityValidation:
    assert authority.submission is not None
    cluster_runspec_path = str(
        PurePosixPath(authority.phase_runspec.cluster.staging_root)
        / "bspp-phase-runs"
        / authority.phase_run.phase_run_id
        / authority.phase_runspec.attempt_id
        / "phase-runspec.json"
    )
    staging_token = f"{authority.submission.submission_id[-16:]}-{plan.action_id}"
    transport.stage_immutable_artifact(
        runspec_path,
        cluster_runspec_path,
        expected_sha256=runspec_document_sha256,
        staging_token=staging_token,
    )
    if isinstance(authority.phase_runspec, FoldingPhaseRunSpec):
        binding = authority.phase_runspec.payload.fold_shard_projection
        if binding is not None:
            local_projection_path = authority.authority_path / binding.location
            cluster_projection_path = str(PurePosixPath(cluster_runspec_path).parent / FOLD_SHARD_PROJECTION_FILENAME)
            transport.stage_immutable_artifact(
                local_projection_path,
                cluster_projection_path,
                expected_sha256=binding.sha256,
                staging_token=staging_token,
            )
    if not isinstance(authority.phase_runspec, FoldingPhaseRunSpec):
        local_manifest_path = (
            authority.authority_path / authority.phase_runspec.payload.database.source_manifest_projection
        )
        cluster_manifest_path = str(PurePosixPath(cluster_runspec_path).parent / "database-source-manifest.json")
        transport.stage_immutable_artifact(
            local_manifest_path,
            cluster_manifest_path,
            expected_sha256=authority.phase_runspec.payload.database.source_manifest_sha256,
            staging_token=staging_token,
        )
    if plan.carry_forward_record_path is not None:
        if authority.phase_runspec.carry_forward is None or plan.carry_forward_record_sha256 is None:
            raise ValueError("carried submission plan lacks authoritative carry projection")
        local_carry_path = authority.authority_path / authority.phase_runspec.carry_forward.location
        transport.stage_immutable_artifact(
            local_carry_path,
            plan.carry_forward_record_path,
            expected_sha256=plan.carry_forward_record_sha256,
            staging_token=staging_token,
        )
    with tempfile.TemporaryDirectory(prefix="bspp-phase-action-") as temporary_directory:
        local_script = Path(temporary_directory) / f"{plan.action_id}.sbatch"
        local_script.write_text(plan.script_body, encoding="utf-8")
        transport.stage_immutable_artifact(
            local_script,
            plan.cluster_script_path,
            expected_sha256=plan.script_sha256,
            staging_token=staging_token,
        )
    _require_command_success(
        transport.command(("mkdir", "-p", str(PurePosixPath(plan.action_evidence_path).parent))),
        operation="Phase Action output directory creation",
    )
    dependency_job_ids = _dependency_job_ids(authority.submission, plan)
    dispatch_at = _format_timestamp(now())
    dispatch_payload = PhaseActionDispatchIntendedPayload(
        submission_id=authority.submission.submission_id,
        action_id=plan.action_id,
        script_sha256=plan.script_sha256,
        scheduler_correlation_token=plan.scheduler_correlation_token,
        dependency_job_ids=dependency_job_ids,
    )
    authority = store.append_event(
        authority.phase_run.phase_run_id,
        lambda sequence: PhaseActionDispatchIntendedEvent(
            sequence=sequence,
            phase_run_id=authority.phase_run.phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at=dispatch_at,
            payload=dispatch_payload,
        ),
    )
    try:
        submission = transport.submit_action(
            SlurmAction(
                action_id=plan.action_id,
                script_path=Path(plan.cluster_script_path),
                dependency_job_ids=dependency_job_ids,
            )
        )
    except SlurmSubmissionRejected as exc:
        rejected_at = _format_timestamp(now())
        rejected_payload = PhaseActionDispatchRejectedPayload(
            submission_id=dispatch_payload.submission_id,
            action_id=plan.action_id,
            script_sha256=plan.script_sha256,
            scheduler_correlation_token=plan.scheduler_correlation_token,
            dependency_job_ids=dependency_job_ids,
            sbatch_argv=exc.result.argv,
            return_code=exc.result.returncode,
            stdout=exc.result.stdout,
            stderr=exc.result.stderr,
        )
        store.append_event(
            authority.phase_run.phase_run_id,
            lambda sequence: PhaseActionDispatchRejectedEvent(
                sequence=sequence,
                phase_run_id=authority.phase_run.phase_run_id,
                attempt_id=authority.phase_runspec.attempt_id,
                occurred_at=rejected_at,
                payload=rejected_payload,
            ),
        )
        raise
    submitted_at = _format_timestamp(now())
    return _append_submitted_event(
        store,
        authority,
        plan,
        dependency_job_ids=dependency_job_ids,
        job_id=submission.job_id,
        sbatch_argv=submission.command,
        occurred_at=submitted_at,
    )


def _recover_dispatched_action(
    store: PhaseAuthorityStore,
    authority: PhaseAuthorityValidation,
    plan: PhaseActionSubmissionPlan,
    *,
    transport: RemoteSlurmTransport,
    intended_at: str,
    now: Clock,
) -> PhaseAuthorityValidation:
    assert authority.submission is not None
    view = next(item for item in authority.submission.actions if item.action_id == plan.action_id)
    job_id = _correlated_job_id(plan, transport=transport, intended_at=intended_at)
    return _append_submitted_event(
        store,
        authority,
        plan,
        dependency_job_ids=view.dependency_job_ids,
        job_id=job_id,
        sbatch_argv=_sbatch_argv(
            plan.cluster_script_path,
            view.dependency_job_ids,
            transport=transport,
        ),
        occurred_at=_format_timestamp(now()),
    )


def _correlated_job_id(
    plan: PhaseActionSubmissionPlan,
    *,
    transport: RemoteSlurmTransport,
    intended_at: str,
) -> str:
    try:
        matches = transport.query_submissions_by_correlation(
            job_name=plan.job_name,
            comment=plan.scheduler_correlation_token,
            submitted_after=intended_at,
        )
    except (OSError, ValueError) as exc:
        raise PhaseSubmissionCorrelationUnresolvedError(str(exc)) from exc
    if not matches:
        raise PhaseSubmissionCorrelationUnresolvedError(
            f"Runtime Action submission remains uncertain and was not repeated: {plan.action_id}"
        )
    if len(matches) != 1:
        raise PhaseSubmissionCorrelationUnresolvedError(
            f"multiple Slurm jobs match durable Runtime Action correlation: {plan.action_id}"
        )
    return matches[0].job_id


def _append_submitted_event(
    store: PhaseAuthorityStore,
    authority: PhaseAuthorityValidation,
    plan: PhaseActionSubmissionPlan,
    *,
    dependency_job_ids: tuple[str, ...],
    job_id: str,
    sbatch_argv: tuple[str, ...],
    occurred_at: str,
) -> PhaseAuthorityValidation:
    assert authority.submission is not None
    payload = PhaseActionSubmittedPayload(
        submission_id=authority.submission.submission_id,
        action_id=plan.action_id,
        script_sha256=plan.script_sha256,
        scheduler_correlation_token=plan.scheduler_correlation_token,
        dependency_job_ids=dependency_job_ids,
        job_id=job_id,
        sbatch_argv=sbatch_argv,
        expected_task_indexes=plan.expected_task_indexes,
    )
    return store.append_event(
        authority.phase_run.phase_run_id,
        lambda sequence: PhaseActionSubmittedEvent(
            sequence=sequence,
            phase_run_id=authority.phase_run.phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at=occurred_at,
            payload=payload,
        ),
    )


def _require_identical_intent(
    recorded: PhaseSubmissionLifecycleView,
    rendered: PhaseSubmissionIntendedPayload,
) -> None:
    recorded_plans = tuple(action.plan for action in recorded.actions)
    if (
        recorded.submission_id != rendered.submission_id
        or recorded.phase_runspec_location != rendered.phase_runspec_location
        or recorded.phase_runspec_digest != rendered.phase_runspec_digest
        or recorded.phase_runspec_document_sha256 != rendered.phase_runspec_document_sha256
        or recorded.qualification_tuple_id != rendered.qualification_tuple_id
        or recorded_plans != rendered.actions
    ):
        raise ValueError("re-rendered Phase Submission differs from its durable intent")


def _dependency_job_ids(
    submission: PhaseSubmissionLifecycleView,
    plan: PhaseActionSubmissionPlan,
) -> tuple[str, ...]:
    by_id = {view.action_id: view for view in submission.actions}
    result: list[str] = []
    for dependency_id in plan.dependency_action_ids:
        dependency = by_id[dependency_id]
        if dependency.status == "satisfied":
            continue
        if dependency.status != "submitted" or dependency.job_id is None:
            raise ValueError(f"Runtime Action dependency lacks durable job assignment: {dependency_id}")
        result.append(dependency.job_id)
    return tuple(result)


def _load_fold_shard_projection(
    authority: PhaseAuthorityValidation,
    runspec: FoldingPhaseRunSpec,
) -> FoldShardProjection:
    """Load and verify the canonical fold shard projection from its binding."""
    binding = runspec.payload.fold_shard_projection
    if binding is None:
        raise ValueError("folding carry closure requires a fold shard projection binding")
    projection_path = authority.authority_path / binding.location
    try:
        raw = projection_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read fold shard projection {projection_path}: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != binding.sha256 or len(raw) != binding.size_bytes:
        raise ValueError("fold shard projection does not bind its authoritative bytes")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed fold shard projection {projection_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("fold shard projection must be a JSON object")
    projection = fold_shard_projection_from_mapping(payload)
    if projection.worker_count != binding.worker_count or projection.lpt_version != binding.lpt_version:
        raise ValueError("fold shard projection binding mismatch")
    return projection


def _is_fully_carried_fold(
    authority: PhaseAuthorityValidation,
    plan: PhaseActionSubmissionPlan,
) -> bool:
    """True iff the current Attempt's carry record covers the complete fold target set."""
    runspec = authority.phase_runspec
    if not isinstance(runspec, FoldingPhaseRunSpec):
        return False
    if not plan.action_id.startswith("fold-"):
        return False
    record = authority.current_carry_forward
    if not isinstance(record, FoldingCarryForwardRecord):
        return False
    projection = _load_fold_shard_projection(authority, runspec)
    projection_target_ids = [target.target_id for rank in projection.ranks for target in rank.targets]
    carried_target_ids = [item.target_id for item in record.content]
    return set(projection_target_ids) == set(carried_target_ids) and len(carried_target_ids) == len(
        set(carried_target_ids)
    )


def _append_satisfied_without_dispatch_event(
    store: PhaseAuthorityStore,
    authority: PhaseAuthorityValidation,
    plan: PhaseActionSubmissionPlan,
    *,
    now: Clock,
) -> PhaseAuthorityValidation:
    """Satisfy a fully carried fold action without any Slurm dispatch."""
    assert authority.submission is not None
    runspec = authority.phase_runspec
    if not isinstance(runspec, FoldingPhaseRunSpec):
        raise ValueError("no-dispatch satisfaction requires a folding Phase RunSpec")
    record = authority.current_carry_forward
    if not isinstance(record, FoldingCarryForwardRecord):
        raise ValueError("satisfied action requires a sealed folding carry record")
    binding = runspec.payload.fold_shard_projection
    if binding is None:
        raise ValueError("satisfied action requires a fold shard projection binding")
    projection = _load_fold_shard_projection(authority, runspec)
    carried_target_ids = tuple(target.target_id for rank in projection.ranks for target in rank.targets)
    payload = PhaseActionSatisfiedWithoutDispatchPayload(
        submission_id=authority.submission.submission_id,
        action_id=plan.action_id,
        runtime_action_digest=plan.runtime_action_digest,
        scheduler_correlation_token=plan.scheduler_correlation_token,
        phase_runspec_digest=runspec.digest,
        carry_record_digest=record.digest,
        shard_manifest_digest=binding.sha256,
        carried_target_ids=carried_target_ids,
    )
    satisfied_at = _format_timestamp(now())
    return store.append_event(
        authority.phase_run.phase_run_id,
        lambda sequence: PhaseActionSatisfiedWithoutDispatchEvent(
            sequence=sequence,
            phase_run_id=authority.phase_run.phase_run_id,
            attempt_id=runspec.attempt_id,
            occurred_at=satisfied_at,
            payload=payload,
        ),
    )


def _topological_actions(
    actions: tuple[PreprocessingRuntimeAction | FoldingRuntimeAction, ...],
) -> tuple[PreprocessingRuntimeAction | FoldingRuntimeAction, ...]:
    by_id = {action.action_id: action for action in actions}
    if any(dependency not in by_id for action in actions for dependency in action.dependencies):
        raise ValueError("Phase RunSpec Runtime Action graph has a dangling dependency")
    remaining = list(actions)
    dispatched: set[str] = set()
    ordered: list[PreprocessingRuntimeAction | FoldingRuntimeAction] = []
    while remaining:
        ready = next((action for action in remaining if set(action.dependencies) <= dispatched), None)
        if ready is None:
            raise ValueError("Phase RunSpec Runtime Action graph is not acyclic")
        remaining.remove(ready)
        ordered.append(ready)
        dispatched.add(ready.action_id)
    return tuple(ordered)


def _submission_intended_at(authority: PhaseAuthorityValidation) -> str:
    submission = authority.submission
    if submission is None:
        raise ValueError("Phase Submission lifecycle is missing its durable intent event")
    matches = tuple(
        item
        for item in authority.events
        if (
            isinstance(item, PhaseSubmissionIntendedEvent)
            and item.phase_run_id == authority.phase_run.phase_run_id
            and item.attempt_id == authority.current_attempt.attempt_id
            and item.payload.submission_id == submission.submission_id
            and item.payload.phase_runspec_digest == authority.phase_runspec.digest
        )
    )
    if len(matches) != 1:
        raise ValueError("current Phase Submission lifecycle requires exactly one durable intent event")
    return matches[0].occurred_at


def _require_current_qualification(
    authority: PhaseAuthorityValidation,
    *,
    now: datetime,
    is_folding: bool = False,
) -> None:
    if is_folding:
        # Folding has no preprocessing runtime qualification; the
        # qualification tuple id is derived from the backend/kernel/cluster
        # snapshot by the folding adapter during intent rendering.
        return
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Phase Submission clock must be timezone-aware")
    runspec = authority.phase_runspec
    if not isinstance(runspec, PhaseRunSpec):
        raise ValueError("preprocessing Runtime Qualification requires a preprocessing Phase RunSpec")
    expires_at = datetime.fromisoformat(runspec.cluster.preprocessing_runtime.expires_at)
    if expires_at <= now.astimezone(UTC):
        raise ValueError("materialized preprocessing Runtime Qualification has expired")


def _sbatch_argv(
    script_path: str,
    dependency_job_ids: tuple[str, ...],
    *,
    transport: RemoteSlurmTransport,
) -> tuple[str, ...]:
    argv: tuple[str, ...] = ("sbatch", "--parsable")
    if dependency_job_ids:
        argv = (*argv, "--dependency=afterok:" + ":".join(dependency_job_ids))
    return command_argv(
        (*argv, script_path),
        transport=transport.kind,
        ssh_target=transport.ssh_target,
    )


def _require_command_success(result: CommandResult, *, operation: str) -> None:
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"{operation} failed"
        raise ValueError(detail)


def _result(authority: PhaseAuthorityValidation) -> PhaseSubmissionResult:
    submission = authority.submission
    if submission is None or submission.status != "submitted":
        raise ValueError("Phase Submission result requires a complete durable assignment set")
    actions: list[PhaseActionSubmissionResult] = []
    for view in submission.actions:
        if view.status == "satisfied":
            actions.append(PhaseActionSubmissionResult(action_id=view.action_id, job_id=None))
            continue
        if view.job_id is None:
            raise ValueError("submitted Runtime Action is missing its durable Slurm job id")
        actions.append(PhaseActionSubmissionResult(action_id=view.action_id, job_id=view.job_id))
    return PhaseSubmissionResult(
        phase_run_id=authority.phase_run.phase_run_id,
        attempt_id=authority.phase_runspec.attempt_id,
        phase_runspec_digest=authority.phase_runspec.digest,
        submission_id=submission.submission_id,
        actions=tuple(actions),
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Phase Submission clock must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = [
    "PhaseActionSubmissionResult",
    "PhaseSubmissionCorrelationUnresolvedError",
    "PhaseSubmissionResult",
    "submit_phase",
]
