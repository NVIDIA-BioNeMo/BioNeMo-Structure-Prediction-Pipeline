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

"""Read-only Phase authority and Slurm status observation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tabulate import tabulate

from bspp.orchestration.contract.folding_carry_forward import FoldingCarryForwardRecord
from bspp.orchestration.contract.phase import FoldingRuntimeAction, PreprocessingRuntimeAction
from bspp.orchestration.contract.phase_cancellation import PhaseCancellationLifecycleView
from bspp.orchestration.contract.phase_carry_forward import AttemptCarryForwardRecord
from bspp.orchestration.contract.phase_reconciliation import (
    FoldingActionTerminalObservationView,
    PhaseActionTerminalObservationView,
)
from bspp.orchestration.contract.phase_submission import (
    PhaseActionSubmissionView,
    PhaseSubmissionLifecycleView,
)
from bspp.orchestration.control.monitoring import SlurmCommandSnapshot, SlurmObservation
from bspp.orchestration.control.phase_adapters import phase_authority_family
from bspp.orchestration.control.phase_authority import (
    PhaseAuthorityStore,
    PhaseAuthorityValidation,
    _select_terminal_observations_by_action,
    validate_phase_run_id,
)
from bspp.orchestration.control.postprocessing_phase_adapter import (
    PostprocessingLifecycleResult,
    status_postprocessing_phase,
)
from bspp.orchestration.control.transport import (
    CommandRunner,
    RemoteSlurmTransport,
    default_command_runner,
)

PhaseSummaryStatus = Literal["materialized", "submitting", "submitted", "failed", "cancelling", "cancelled", "accepted"]
PhaseActionDurableStatus = Literal["not-submitted", "planned", "dispatching", "submitted", "satisfied", "rejected"]
SchedulerObservationStatus = Literal["not-applicable", "observed", "missing", "unknown"]
SchedulerSummaryStatus = Literal["not-requested", "complete", "degraded", "unavailable"]
SchedulerSourceAvailability = Literal["skipped", "available", "unavailable"]
_INCOMPLETE_PROJECTION_WARNING = "current_runspec_projection_incomplete: rerun Phase Retry before mutation"


def _carry_forward_id(record: AttemptCarryForwardRecord | FoldingCarryForwardRecord) -> str:
    """Return one carry record's canonical id across both phase families."""
    if isinstance(record, FoldingCarryForwardRecord):
        return record.folding_carry_forward_id
    return record.attempt_carry_forward_id


@dataclass(frozen=True)
class PhaseAttemptHistoryStatus:
    attempt_id: str
    ordinal: int
    status: str
    phase_runspec_digest: str
    phase_runspec_location: str
    retry_id: str | None = None
    attempt_carry_forward_id: str | None = None

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "attempt_id": self.attempt_id,
            "ordinal": self.ordinal,
            "status": self.status,
            "phase_runspec_digest": self.phase_runspec_digest,
            "phase_runspec_location": self.phase_runspec_location,
            "retry_id": self.retry_id,
        }
        if self.attempt_carry_forward_id is not None:
            result["attempt_carry_forward_id"] = self.attempt_carry_forward_id
        return result


@dataclass(frozen=True)
class PhaseCarryForwardStatus:
    attempt_carry_forward_id: str
    source_attempt_id: str
    content_count: int
    content_digest: str
    record_projection_complete: bool

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_carry_forward_id": self.attempt_carry_forward_id,
            "source_attempt_id": self.source_attempt_id,
            "content_count": self.content_count,
            "content_digest": self.content_digest,
            "record_projection_complete": self.record_projection_complete,
        }


@dataclass(frozen=True)
class PhaseSchedulerSourceStatus:
    """Availability of one scheduler observation source."""

    source: str
    availability: SchedulerSourceAvailability
    parser: str
    returncode: int
    warning: str | None = None

    def __post_init__(self) -> None:
        if self.source not in {"squeue", "sacct"}:
            raise ValueError(f"unsupported scheduler source: {self.source!r}")
        if self.availability not in {"skipped", "available", "unavailable"}:
            raise ValueError(f"unsupported scheduler source availability: {self.availability!r}")
        if self.availability == "skipped" and self.parser != "skipped":
            raise ValueError("skipped scheduler source requires the skipped parser")
        if self.availability == "unavailable" and self.parser != "unavailable":
            raise ValueError("unavailable scheduler source requires the unavailable parser")
        if self.availability == "available" and self.parser in {"skipped", "unavailable"}:
            raise ValueError("available scheduler source requires a successful parser")
        if self.availability == "unavailable" and self.warning is None:
            raise ValueError("unavailable scheduler source requires a warning")
        if self.warning == "":
            raise ValueError("scheduler source warning must be non-empty when present")

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "availability": self.availability,
            "parser": self.parser,
            "returncode": self.returncode,
        }
        if self.warning is not None:
            mapping["warning"] = self.warning
        return mapping


PhaseActionTerminalView = PhaseActionTerminalObservationView | FoldingActionTerminalObservationView


@dataclass(frozen=True)
class PhaseActionSchedulerStatus:
    """Selected scheduler observation for one durably declared action."""

    status: SchedulerObservationStatus
    state: str | None = None
    source: str | None = None
    exit_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"not-applicable", "observed", "missing", "unknown"}:
            raise ValueError(f"unsupported scheduler observation status: {self.status!r}")
        if self.status == "observed":
            if not self.state or self.source not in {"squeue", "sacct"}:
                raise ValueError("observed scheduler status requires a state and scheduler source")
            return
        if self.state is not None or self.source is not None or self.exit_code is not None:
            raise ValueError("unobserved scheduler status cannot contain selected job fields")

    def to_mapping(self) -> dict[str, object]:
        return {
            "status": self.status,
            "state": self.state,
            "source": self.source,
            "exit_code": self.exit_code,
        }


@dataclass(frozen=True)
class PhaseActionStatus:
    """Durable and observed state for one Runtime Action."""

    action_id: str
    dependency_action_ids: tuple[str, ...]
    durable_status: PhaseActionDurableStatus
    job_id: str | None
    scheduler: PhaseActionSchedulerStatus
    terminal: PhaseActionTerminalView | None = None

    def __post_init__(self) -> None:
        if not self.action_id or not isinstance(self.dependency_action_ids, tuple):
            raise ValueError("Phase Action status requires an id and immutable dependencies")
        if self.durable_status not in {"not-submitted", "planned", "dispatching", "submitted", "satisfied", "rejected"}:
            raise ValueError(f"unsupported durable Phase Action status: {self.durable_status!r}")
        if self.durable_status == "submitted":
            if self.job_id is None:
                raise ValueError("submitted Phase Action status requires a durable job id")
        elif self.job_id is not None:
            raise ValueError("only a submitted Phase Action status may contain a durable job id")
        if (self.job_id is None) != (self.scheduler.status == "not-applicable"):
            raise ValueError("scheduler applicability must match durable job-id presence")
        if self.terminal is not None:
            if self.durable_status != "submitted":
                raise ValueError("durable terminal accounting must bind a submitted action job")
            if self.action_id != self.terminal.action_id:
                raise ValueError("durable terminal accounting must bind its action")
            if isinstance(self.terminal, PhaseActionTerminalObservationView):
                if self.job_id != self.terminal.job_id:
                    raise ValueError("durable terminal accounting must bind a submitted action job")
            elif self.job_id != self.terminal.parent_job_id:
                raise ValueError("durable terminal accounting must bind a submitted action job")

    def to_mapping(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "dependency_action_ids": list(self.dependency_action_ids),
            "durable_status": self.durable_status,
            "job_id": self.job_id,
            "terminal": self.terminal.to_mapping() if self.terminal is not None else None,
            "scheduler": self.scheduler.to_mapping(),
        }


@dataclass(frozen=True)
class PhaseStatusReport:
    """Stable read-only status report for one current Phase Attempt."""

    phase_run_id: str
    attempt_id: str
    attempt_status: str
    status: PhaseSummaryStatus
    sealed: bool
    phase_receipt_id: str | None
    submission_id: str | None
    submission_status: str | None
    cancellation: PhaseCancellationLifecycleView | None
    actions: tuple[PhaseActionStatus, ...]
    scheduler_status: SchedulerSummaryStatus
    requested_job_ids: tuple[str, ...]
    scheduler_sources: tuple[PhaseSchedulerSourceStatus, PhaseSchedulerSourceStatus]
    attempt_history: tuple[PhaseAttemptHistoryStatus, ...]
    runspec_projection_complete: bool
    carry_forward: PhaseCarryForwardStatus | None = None
    carry_forward_closure_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_phase_run_id(self.phase_run_id)
        if not self.attempt_id or not self.actions:
            raise ValueError("Phase Status requires the current Attempt and every declared action")
        if self.attempt_status not in {"materialized", "failed", "cancelling", "cancelled", "succeeded"}:
            raise ValueError(f"unsupported Phase Attempt status: {self.attempt_status!r}")
        if self.status not in {
            "materialized",
            "submitting",
            "submitted",
            "failed",
            "cancelling",
            "cancelled",
            "accepted",
        }:
            raise ValueError(f"unsupported Phase summary status: {self.status!r}")
        if self.scheduler_status not in {"not-requested", "complete", "degraded", "unavailable"}:
            raise ValueError(f"unsupported Phase scheduler summary: {self.scheduler_status!r}")
        if len({action.action_id for action in self.actions}) != len(self.actions):
            raise ValueError("Phase Status action ids must be unique")
        if tuple(source.source for source in self.scheduler_sources) != ("squeue", "sacct"):
            raise ValueError("Phase Status scheduler sources must be ordered squeue then sacct")
        expected_job_ids = _ordered_unique_jobs(self.actions)
        if self.requested_job_ids != expected_job_ids:
            raise ValueError("Phase Status requested jobs must match durable assignments in action order")
        if self.requested_job_ids:
            if any(source.availability == "skipped" for source in self.scheduler_sources):
                raise ValueError("Phase Status with durable jobs cannot skip a scheduler source")
        elif any(source.availability != "skipped" for source in self.scheduler_sources):
            raise ValueError("Phase Status without durable jobs must skip scheduler sources")
        expected_scheduler_status = _scheduler_summary_status(self.requested_job_ids, self.scheduler_sources)
        if self.scheduler_status != expected_scheduler_status:
            raise ValueError("Phase Status scheduler summary does not match source availability")
        expected_warnings = tuple(source.warning for source in self.scheduler_sources if source.warning is not None)
        if not self.runspec_projection_complete:
            expected_warnings = (*expected_warnings, _INCOMPLETE_PROJECTION_WARNING)
        if self.warnings != expected_warnings:
            raise ValueError("Phase Status warnings must match scheduler sources in stable order")
        if not self.attempt_history or self.attempt_history[-1].attempt_id != self.attempt_id:
            raise ValueError("Phase Status Attempt history must end at the current Attempt")
        if tuple(item.ordinal for item in self.attempt_history) != tuple(range(1, len(self.attempt_history) + 1)):
            raise ValueError("Phase Status Attempt history must be contiguous")
        if (self.submission_id is None) != (self.submission_status is None):
            raise ValueError("Phase Status submission id and state must appear together")
        if self.submission_status is not None and (
            not self.submission_id or self.submission_status not in {"submitting", "submitted", "failed"}
        ):
            raise ValueError("Phase Status has an invalid durable submission summary")
        if self.carry_forward is None and self.carry_forward_closure_ids:
            raise ValueError("Phase Status carry closure requires a current carry record")
        if len(set(self.carry_forward_closure_ids)) != len(self.carry_forward_closure_ids):
            raise ValueError("Phase Status carry closure ids must be unique and ordered")
        if self.attempt_status == "materialized":
            if self.cancellation is not None:
                raise ValueError("active Phase Status cannot contain cancellation authority")
            if self.sealed or self.phase_receipt_id is not None:
                raise ValueError("materialized Phase Status must be unsealed and unreceipted")
            expected_status = self.submission_status or "materialized"
            if self.status != expected_status:
                raise ValueError("materialized Phase summary must preserve its durable submission status")
        elif self.attempt_status == "failed":
            if self.cancellation is not None:
                raise ValueError("failed Phase Status cannot contain cancellation authority")
            if self.sealed or self.phase_receipt_id is not None or self.status != "failed":
                raise ValueError("failed Phase Status must be failed, unsealed, and unreceipted")
            if self.submission_status not in {"submitting", "submitted", "failed"}:
                raise ValueError("failed Phase Status requires durable submission authority")
        elif self.attempt_status in {"cancelling", "cancelled"}:
            if self.sealed or self.phase_receipt_id is not None:
                raise ValueError("cancellation Phase Status must be unsealed and unreceipted")
            if self.cancellation is None or self.cancellation.status != self.attempt_status:
                raise ValueError("cancellation Phase Status requires matching durable cancellation authority")
            if (
                self.cancellation.phase_run_id != self.phase_run_id
                or self.cancellation.attempt_id != self.attempt_id
                or tuple(item.action_id for item in self.cancellation.actions)
                != tuple(item.action_id for item in self.actions)
            ):
                raise ValueError("cancellation Phase Status must bind the exact report run, attempt, and action order")
            if self.status != self.attempt_status:
                raise ValueError("cancellation Phase summary must match its lifecycle")
        else:
            if self.cancellation is not None:
                raise ValueError("accepted Phase Status cannot contain cancellation authority")
            if not self.sealed or self.phase_receipt_id is None or self.status != "accepted":
                raise ValueError("succeeded Phase Status requires a sealed accepted receipt")
            if self.submission_status != "submitted":
                raise ValueError("accepted Phase Status requires a complete durable submission")
        for action in self.actions:
            if action.job_id is None:
                continue
            expected_observation_states = (
                {"observed", "missing"} if self.scheduler_status == "complete" else {"observed", "unknown"}
            )
            if action.scheduler.status not in expected_observation_states:
                raise ValueError("Phase Action observation does not match scheduler source availability")

    def to_mapping(self) -> dict[str, object]:
        submission: dict[str, object] | None = None
        if self.submission_id is not None:
            submission = {"submission_id": self.submission_id, "status": self.submission_status}
        result: dict[str, object] = {
            "phase_run_id": self.phase_run_id,
            "current_attempt": {"attempt_id": self.attempt_id, "status": self.attempt_status},
            "attempt_history": [item.to_mapping() for item in self.attempt_history],
            "runspec_projection_complete": self.runspec_projection_complete,
            "phase": {
                "status": self.status,
                "sealed": self.sealed,
                "phase_receipt_id": self.phase_receipt_id,
            },
            "submission": submission,
            "cancellation": self.cancellation.to_mapping() if self.cancellation is not None else None,
            "actions": [action.to_mapping() for action in self.actions],
            "scheduler": {
                "status": self.scheduler_status,
                "requested_job_ids": list(self.requested_job_ids),
                "sources": {source.source: source.to_mapping() for source in self.scheduler_sources},
                "warnings": list(self.warnings),
            },
        }
        if self.carry_forward is not None:
            result["carry_forward"] = self.carry_forward.to_mapping()
            result["carry_forward_closure_ids"] = list(self.carry_forward_closure_ids)
        return result

    def render_json(self) -> str:
        """Render stable structured JSON."""
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"

    def render_table(self) -> str:
        """Render a stable human-readable summary and action table."""
        summary_rows: tuple[tuple[str, str], ...] = (
            ("phase_run_id", self.phase_run_id),
            ("attempt_id", self.attempt_id),
            ("attempt_status", self.attempt_status),
            ("runspec_projection_complete", str(self.runspec_projection_complete).lower()),
            ("phase_status", self.status),
            ("sealed", str(self.sealed).lower()),
            ("phase_receipt_id", self.phase_receipt_id or ""),
            ("submission_status", self.submission_status or ""),
            ("cancellation_status", self.cancellation.status if self.cancellation is not None else ""),
            ("scheduler_status", self.scheduler_status),
        )
        if self.carry_forward is not None:
            summary_rows = (
                *summary_rows,
                ("attempt_carry_forward_id", self.carry_forward.attempt_carry_forward_id),
                ("carry_source_attempt", self.carry_forward.source_attempt_id),
                ("carried_content_count", str(self.carry_forward.content_count)),
                (
                    "carry_record_projection_complete",
                    str(self.carry_forward.record_projection_complete).lower(),
                ),
            )
        action_rows = tuple(
            (
                action.action_id,
                action.durable_status,
                action.job_id or "",
                action.scheduler.status,
                action.scheduler.state or "",
                action.scheduler.source or "",
                action.scheduler.exit_code or "",
                action.terminal.outcome if action.terminal is not None else "",
                action.terminal.state if isinstance(action.terminal, PhaseActionTerminalObservationView) else "",
            )
            for action in self.actions
        )
        lines = [
            "phase_status:",
            "  summary:",
            "    "
            + tabulate(
                summary_rows,
                headers=("field", "value"),
                tablefmt="github",
                stralign="left",
                disable_numparse=True,
            ).replace("\n", "\n    "),
            "  actions:",
        ]
        action_table = tabulate(
            action_rows,
            headers=(
                "action",
                "durable",
                "job_id",
                "observation",
                "slurm_state",
                "source",
                "exit_code",
                "terminal_outcome",
                "terminal_state",
            ),
            tablefmt="github",
            stralign="left",
            disable_numparse=True,
        )
        lines.extend(f"    {line}" for line in action_table.splitlines())
        lines.append("  attempt_history:")
        history_table = tabulate(
            tuple(
                (
                    item.attempt_id,
                    str(item.ordinal),
                    item.status,
                    item.phase_runspec_digest,
                    item.retry_id or "",
                )
                for item in self.attempt_history
            ),
            headers=("attempt", "ordinal", "status", "runspec_digest", "retry_id"),
            tablefmt="github",
            stralign="left",
            disable_numparse=True,
        )
        lines.extend(f"    {line}" for line in history_table.splitlines())
        if self.cancellation is not None:
            lines.append("  cancellation:")
            cancellation_rows = tuple(
                (
                    action.action_id,
                    action.target.initial_submission_status,
                    action.disposition,
                    action.bound_job_id or "",
                    str(len(action.requests)),
                    ""
                    if not action.requests or action.requests[-1].return_code is None
                    else str(action.requests[-1].return_code),
                )
                for action in self.cancellation.actions
            )
            cancellation_table = tabulate(
                cancellation_rows,
                headers=("action", "initial_submission", "disposition", "job_id", "requests", "last_return_code"),
                tablefmt="github",
                stralign="left",
                disable_numparse=True,
            )
            lines.extend(f"    {line}" for line in cancellation_table.splitlines())
        if self.warnings:
            lines.append("  warnings:")
            lines.extend(f"    - {warning}" for warning in self.warnings)
        return "\n".join(lines) + "\n"


def status_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    runner: CommandRunner = default_command_runner,
) -> PhaseStatusReport | PostprocessingLifecycleResult:
    """Observe one Phase Run without locking, reconciling, or writing authority."""
    family = phase_authority_family(authority_root, phase_run_id)
    if family == "postprocessing":
        return status_postprocessing_phase(
            phase_run_id,
            authority_root=authority_root,
            runner=runner,
        )
    if family == "folding":
        # Folding status flows through the generic read-only body below; the
        # action graph and scheduler observation are already family-agnostic.
        pass
    validate_phase_run_id(phase_run_id)
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    declared_actions = authority.phase_runspec.payload.actions
    submission = authority.submission
    _require_submission_action_alignment(declared_actions, submission)
    job_ids = _durable_job_ids(declared_actions, submission)

    observation: SlurmObservation | None = None
    if job_ids:
        transport = RemoteSlurmTransport(
            kind=authority.phase_runspec.cluster.transport,
            ssh_target=authority.phase_runspec.cluster.ssh_target,
            runner=runner,
        )
        observation = transport.query_observation_best_effort(job_ids)

    sources = _scheduler_sources(observation)
    scheduler_status = _scheduler_summary_status(job_ids, sources)
    actions = _build_action_statuses(
        declared_actions,
        submission=submission,
        terminal_observations=authority.terminal_observations,
        array_terminal_observations=authority.array_terminal_observations,
        observation=observation,
        scheduler_status=scheduler_status,
    )
    return _build_report(
        authority,
        actions=actions,
        job_ids=job_ids,
        sources=sources,
        scheduler_status=scheduler_status,
        warnings=(
            *(observation.warnings if observation is not None else ()),
            *(() if authority.current_runspec_projection_complete else (_INCOMPLETE_PROJECTION_WARNING,)),
        ),
    )


def _build_report(
    authority: PhaseAuthorityValidation,
    *,
    actions: tuple[PhaseActionStatus, ...],
    job_ids: tuple[str, ...],
    sources: tuple[PhaseSchedulerSourceStatus, PhaseSchedulerSourceStatus],
    scheduler_status: SchedulerSummaryStatus,
    warnings: tuple[str, ...],
) -> PhaseStatusReport:
    submission = authority.submission
    history: list[PhaseAttemptHistoryStatus] = []
    for index, prior in enumerate(authority.prior_attempts):
        incoming_retry_id = authority.prior_attempts[index - 1].retry_id if index else None
        history.append(
            PhaseAttemptHistoryStatus(
                attempt_id=prior.attempt.attempt_id,
                ordinal=prior.attempt.ordinal,
                status=prior.outcome,
                phase_runspec_digest=prior.phase_runspec.digest,
                phase_runspec_location=prior.attempt.phase_runspec_location,
                retry_id=incoming_retry_id,
                attempt_carry_forward_id=(
                    _carry_forward_id(prior.carry_forward) if prior.carry_forward is not None else None
                ),
            )
        )
    history.append(
        PhaseAttemptHistoryStatus(
            attempt_id=authority.current_attempt.attempt_id,
            ordinal=authority.current_attempt.ordinal,
            status=authority.lifecycle.attempt_status,
            phase_runspec_digest=authority.phase_runspec.digest,
            phase_runspec_location=authority.current_attempt.phase_runspec_location,
            retry_id=authority.prior_attempts[-1].retry_id if authority.prior_attempts else None,
            attempt_carry_forward_id=(
                _carry_forward_id(authority.current_carry_forward)
                if authority.current_carry_forward is not None
                else None
            ),
        )
    )
    return PhaseStatusReport(
        phase_run_id=authority.phase_run.phase_run_id,
        attempt_id=authority.lifecycle.current_attempt_id,
        attempt_status=authority.lifecycle.attempt_status,
        status=_phase_summary_status(authority),
        sealed=authority.lifecycle.sealed,
        phase_receipt_id=authority.lifecycle.phase_receipt_id,
        submission_id=submission.submission_id if submission is not None else None,
        submission_status=submission.status if submission is not None else None,
        cancellation=authority.cancellation,
        actions=actions,
        scheduler_status=scheduler_status,
        requested_job_ids=job_ids,
        scheduler_sources=sources,
        attempt_history=tuple(history),
        runspec_projection_complete=authority.current_runspec_projection_complete,
        carry_forward=(
            PhaseCarryForwardStatus(
                attempt_carry_forward_id=_carry_forward_id(authority.current_carry_forward),
                source_attempt_id=authority.current_carry_forward.source_attempt_id,
                content_count=len(authority.current_carry_forward.content),
                content_digest=authority.current_carry_forward.content_digest,
                record_projection_complete=(
                    authority.phase_runspec.carry_forward is not None
                    and (authority.authority_path / authority.phase_runspec.carry_forward.location).is_file()
                ),
            )
            if authority.current_carry_forward is not None
            else None
        ),
        carry_forward_closure_ids=(
            tuple(item.attempt_carry_forward_id for item in authority.receipt.carry_forward_closure)
            if authority.receipt is not None
            else ()
        ),
        warnings=warnings,
    )


def _phase_summary_status(authority: PhaseAuthorityValidation) -> PhaseSummaryStatus:
    if authority.lifecycle.attempt_status == "cancelling" and authority.lifecycle.run_status == "cancelling":
        return "cancelling"
    if authority.lifecycle.attempt_status == "cancelled" and authority.lifecycle.run_status == "cancelled":
        return "cancelled"
    if authority.lifecycle.sealed and authority.lifecycle.run_status == "accepted":
        return "accepted"
    if authority.lifecycle.attempt_status == "failed" and authority.lifecycle.run_status == "failed":
        return "failed"
    if authority.submission is None:
        return "materialized"
    return authority.submission.status


def _require_submission_action_alignment(
    declared_actions: tuple[PreprocessingRuntimeAction | FoldingRuntimeAction, ...],
    submission: PhaseSubmissionLifecycleView | None,
) -> None:
    if submission is None:
        return
    declared_ids = tuple(action.action_id for action in declared_actions)
    submitted_ids = tuple(action.action_id for action in submission.actions)
    if declared_ids != submitted_ids:
        raise ValueError("Phase Status submission actions do not match authoritative RunSpec order")


def _durable_job_ids(
    declared_actions: tuple[PreprocessingRuntimeAction | FoldingRuntimeAction, ...],
    submission: PhaseSubmissionLifecycleView | None,
) -> tuple[str, ...]:
    if submission is None:
        return ()
    by_id = {action.action_id: action for action in submission.actions}
    seen: set[str] = set()
    job_ids: list[str] = []
    for declared in declared_actions:
        job_id = by_id[declared.action_id].job_id
        if job_id is not None and job_id not in seen:
            seen.add(job_id)
            job_ids.append(job_id)
    return tuple(job_ids)


def _scheduler_sources(
    observation: SlurmObservation | None,
) -> tuple[PhaseSchedulerSourceStatus, PhaseSchedulerSourceStatus]:
    if observation is None:
        return (
            PhaseSchedulerSourceStatus("squeue", "skipped", "skipped", 0),
            PhaseSchedulerSourceStatus("sacct", "skipped", "skipped", 0),
        )
    return (_scheduler_source(observation.squeue), _scheduler_source(observation.sacct))


def _scheduler_source(snapshot: SlurmCommandSnapshot) -> PhaseSchedulerSourceStatus:
    availability: SchedulerSourceAvailability = "unavailable" if snapshot.parser == "unavailable" else "available"
    return PhaseSchedulerSourceStatus(
        source=snapshot.kind,
        availability=availability,
        parser=snapshot.parser,
        returncode=snapshot.returncode,
        warning=snapshot.warning,
    )


def _scheduler_summary_status(
    requested_job_ids: tuple[str, ...],
    sources: tuple[PhaseSchedulerSourceStatus, PhaseSchedulerSourceStatus],
) -> SchedulerSummaryStatus:
    if not requested_job_ids:
        return "not-requested"
    unavailable = sum(source.availability == "unavailable" for source in sources)
    if unavailable == 2:
        return "unavailable"
    if unavailable == 1:
        return "degraded"
    return "complete"


def _build_action_statuses(
    declared_actions: tuple[PreprocessingRuntimeAction | FoldingRuntimeAction, ...],
    *,
    submission: PhaseSubmissionLifecycleView | None,
    terminal_observations: tuple[PhaseActionTerminalObservationView, ...] = (),
    array_terminal_observations: tuple[FoldingActionTerminalObservationView, ...] = (),
    observation: SlurmObservation | None,
    scheduler_status: SchedulerSummaryStatus,
) -> tuple[PhaseActionStatus, ...]:
    submitted_by_id: dict[str, PhaseActionSubmissionView] = {}
    if submission is not None:
        submitted_by_id = {action.action_id: action for action in submission.actions}
    observed_by_job_id = observation.selected_state_by_job_id() if observation is not None else {}
    terminal_by_action = _select_terminal_observations_by_action(
        terminal_observations,
        array_terminal_observations,
    )
    statuses: list[PhaseActionStatus] = []
    for declared in declared_actions:
        durable = submitted_by_id.get(declared.action_id)
        durable_status: PhaseActionDurableStatus = durable.status if durable is not None else "not-submitted"
        job_id = durable.job_id if durable is not None else None
        if job_id is None:
            scheduler = PhaseActionSchedulerStatus("not-applicable")
        else:
            selected = observed_by_job_id.get(job_id)
            if selected is not None:
                scheduler = PhaseActionSchedulerStatus(
                    "observed",
                    state=selected.state,
                    source=selected.source,
                    exit_code=selected.exit_code,
                )
            else:
                missing_status: SchedulerObservationStatus = "missing" if scheduler_status == "complete" else "unknown"
                scheduler = PhaseActionSchedulerStatus(missing_status)
        statuses.append(
            PhaseActionStatus(
                action_id=declared.action_id,
                dependency_action_ids=declared.dependencies,
                durable_status=durable_status,
                job_id=job_id,
                scheduler=scheduler,
                terminal=terminal_by_action.get(declared.action_id),
            )
        )
    return tuple(statuses)


def _ordered_unique_jobs(actions: tuple[PhaseActionStatus, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    job_ids: list[str] = []
    for action in actions:
        if action.job_id is not None and action.job_id not in seen:
            seen.add(action.job_id)
            job_ids.append(action.job_id)
    return tuple(job_ids)


__all__ = [
    "PhaseActionSchedulerStatus",
    "PhaseActionStatus",
    "PhaseAttemptHistoryStatus",
    "PhaseCarryForwardStatus",
    "PhaseSchedulerSourceStatus",
    "PhaseStatusReport",
    "status_phase",
]
