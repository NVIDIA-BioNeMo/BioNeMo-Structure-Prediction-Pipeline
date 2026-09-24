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

"""Bounded and crash-safe Phase cancellation tests."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from click.testing import CliRunner

from bspp.orchestration.contract.phase_cancellation import (
    PhaseCancellationCompletedEvent,
    PhaseCancellationIntendedEvent,
    PhaseJobCancellationRequestIntendedEvent,
    PhaseJobCancellationRequestIntendedPayload,
    PhaseJobCancellationRequestResultEvent,
)
from bspp.orchestration.contract.phase_reconciliation import PhaseActionTerminalObservedEvent
from bspp.orchestration.contract.phase_submission import (
    PhaseActionSubmissionPlan,
    PhaseActionSubmissionView,
    PhaseActionSubmittedEvent,
    PhaseActionSubmittedPayload,
    PhaseSubmissionLifecycleView,
    phase_action_scheduler_correlation_token,
    phase_action_submission_identity_mapping,
    phase_submission_id,
)
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore, _replay_action_submitted_event
from bspp.orchestration.control.phase_cancellation import cancel_phase
from bspp.orchestration.control.phase_finalization import finalize_phase
from bspp.orchestration.control.phase_rendering import render_phase_submission_intent
from bspp.orchestration.control.phase_resume import resume_phase
from bspp.orchestration.control.phase_status import status_phase
from bspp.orchestration.control.phase_submission import submit_phase
from bspp.orchestration.control.transport import CommandResult, RemoteSlurmTransport
from tests.support.transport_argv import wrap_remote_command
from tests.test_phase_finalization import _accepted_inputs, _finalize, _record_submission
from tests.test_phase_resume import _materialized_authority, _record_submission_state, _record_terminal

NOW = datetime(2026, 8, 20, 13, 0, tzinfo=UTC)


class CancellationRunner:
    def __init__(
        self,
        *,
        accounting_states: list[tuple[str, str | None]] | None = None,
        scancel_returncodes: list[int] | None = None,
        correlation_job_id: str | None = None,
        crash_scancel: bool = False,
    ) -> None:
        self.accounting_states = accounting_states or [("RUNNING", None)]
        self.scancel_returncodes = list(scancel_returncodes or [0])
        self.correlation_job_id = correlation_job_id
        self.crash_scancel = crash_scancel
        self.observation_index = 0
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "scancel":
            if self.crash_scancel:
                raise OSError("simulated scancel transport interruption")
            returncode = self.scancel_returncodes.pop(0) if self.scancel_returncodes else 0
            return CommandResult(argv, returncode, "", "rejected\n" if returncode else "")
        if argv[0] == "squeue" and any(item.startswith("--name=") for item in argv):
            jobs: list[dict[str, object]] = []
            if self.correlation_job_id is not None:
                token = next(item.removeprefix("--name=") for item in argv if item.startswith("--name="))
                jobs = [
                    {
                        "job_id": int(self.correlation_job_id),
                        "name": token,
                        "comment": token,
                        "job_state": "RUNNING",
                    }
                ]
            return CommandResult(argv, 0, json.dumps({"jobs": jobs}), "")
        if argv[0] == "sacct" and any(item.startswith("--name=") for item in argv):
            return CommandResult(argv, 0, json.dumps({"jobs": []}), "")
        if argv[0] == "squeue":
            job_ids = argv[argv.index("-j") + 1].split(",")
            jobs = [{"job_id": int(job_id), "job_state": "RUNNING"} for job_id in job_ids]
            return CommandResult(argv, 0, json.dumps({"jobs": jobs}), "")
        if argv[0] == "sacct":
            job_ids = argv[argv.index("-j") + 1].split(",")
            state, exit_code = self.accounting_states[min(self.observation_index, len(self.accounting_states) - 1)]
            self.observation_index += 1
            jobs = [{"job_id_raw": job_id, "state": state, "exit_code": exit_code} for job_id in job_ids]
            return CommandResult(argv, 0, json.dumps({"jobs": jobs}), "")
        raise AssertionError(f"unexpected command: {argv}")


def _reject_remote(_argv: tuple[str, ...]) -> CommandResult:
    raise AssertionError("unexpected remote call")


@pytest.mark.parametrize("submission_state", [None, "planned"])
def test_true_zero_cancellation_completes_without_scheduler_calls_and_is_idempotent(
    tmp_path: Path,
    submission_state: str | None,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    if submission_state is not None:
        _record_submission_state(authority_root, phase_run_id, submission_state)

    first = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=_reject_remote,
    )
    second = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=_reject_remote,
    )

    assert first == second
    assert first.status == "cancelled"
    assert first.target_job_ids == first.terminal_job_ids == first.pending_job_ids == ()
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.lifecycle.attempt_status == replay.lifecycle.run_status == "cancelled"
    assert sum(isinstance(event, PhaseCancellationIntendedEvent) for event in replay.events) == 1
    assert sum(isinstance(event, PhaseCancellationCompletedEvent) for event in replay.events) == 1


@pytest.mark.parametrize(
    ("terminal_state", "exit_code", "expected_outcome"),
    [
        ("COMPLETED", "0:0", "succeeded"),
        ("FAILED", "1:0", "failed"),
        ("CANCELLED", "0:15", "failed"),
    ],
)
def test_submitted_cancellation_scancels_once_then_uses_terminal_sacct_as_completion(
    tmp_path: Path,
    terminal_state: str,
    exit_code: str,
    expected_outcome: str,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    runner = CancellationRunner(accounting_states=[("RUNNING", None), (terminal_state, exit_code)])

    result = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=runner,
    )

    assert result.status == "cancelled"
    assert result.target_job_ids == result.terminal_job_ids == ("123456",)
    assert [source.availability for source in result.scheduler_sources] == ["available", "available"]
    assert result.warnings == ()
    assert [call for call in runner.calls if call[0] == "scancel"] == [("scancel", "123456")]
    assert len([call for call in runner.calls if call[0] == "sacct" and "-j" in call]) == 2
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.lifecycle.attempt_status == replay.lifecycle.run_status == "cancelled"
    assert replay.terminal_observations[0].state == terminal_state
    assert replay.terminal_observations[0].outcome == expected_outcome
    assert sum(isinstance(event, PhaseActionTerminalObservedEvent) for event in replay.events) == 1


def test_completed_cancellation_retains_degraded_scheduler_observation_and_warnings(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        if argv[0] == "squeue":
            return CommandResult(argv, 1, "", "queue unavailable")
        if argv[0] == "sacct":
            return CommandResult(
                argv,
                0,
                json.dumps({"jobs": [{"job_id_raw": "123456", "state": "CANCELLED", "exit_code": "0:15"}]}),
                "",
            )
        raise AssertionError(f"unexpected command: {argv}")

    result = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=runner,
    )

    assert result.status == "cancelled"
    assert [source.availability for source in result.scheduler_sources] == ["unavailable", "available"]
    assert any("squeue_unavailable" in warning for warning in result.warnings)
    assert not any(call[0] == "scancel" for call in calls)


@pytest.mark.parametrize("sacct_mode", ["empty", "unavailable"])
def test_squeue_terminal_without_sacct_evidence_cannot_complete(
    tmp_path: Path,
    sacct_mode: str,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        if argv[0] == "squeue":
            return CommandResult(
                argv,
                0,
                json.dumps({"jobs": [{"job_id": 123456, "job_state": "CANCELLED"}]}),
                "",
            )
        if argv[0] == "sacct" and sacct_mode == "empty":
            return CommandResult(argv, 0, json.dumps({"jobs": []}), "")
        if argv[0] == "sacct":
            return CommandResult(argv, 1, "", "accounting unavailable")
        if argv[0] == "scancel":
            return CommandResult(argv, 0, "", "")
        raise AssertionError(f"unexpected command: {argv}")

    result = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=runner,
    )

    assert result.status == "cancelling"
    assert result.terminal_job_ids == ()
    assert [call for call in calls if call[0] == "scancel"] == [("scancel", "123456")]
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.cancellation is not None
    assert replay.cancellation.status == "cancelling"
    assert replay.terminal_observations == ()
    assert not any(isinstance(event, PhaseCancellationCompletedEvent) for event in replay.events)


def test_preexisting_successful_terminal_evidence_completes_without_query_or_scancel(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    _record_terminal(authority_root, phase_run_id)

    result = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=_reject_remote,
    )

    assert result.status == "cancelled"
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert len(replay.terminal_observations) == 1
    assert sum(isinstance(event, PhaseActionTerminalObservedEvent) for event in replay.events) == 1


def test_nonzero_scancel_result_is_durable_and_next_invocation_uses_next_request_ordinal(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    first_runner = CancellationRunner(accounting_states=[("RUNNING", None)], scancel_returncodes=[1])

    first = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=first_runner,
    )
    assert first.status == "cancelling"
    second_runner = CancellationRunner(accounting_states=[("RUNNING", None), ("CANCELLED", "0:15")])
    second = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=second_runner,
    )
    assert second.status == "cancelled"

    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    intents = [event for event in replay.events if isinstance(event, PhaseJobCancellationRequestIntendedEvent)]
    results = [event for event in replay.events if isinstance(event, PhaseJobCancellationRequestResultEvent)]
    assert [event.payload.request_ordinal for event in intents] == [1, 2]
    assert [event.payload.request_ordinal for event in results] == [1, 2]
    assert [event.payload.return_code for event in results] == [1, 0]


def test_scancel_interruption_leaves_request_intent_and_recovery_reuses_it(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    crashing = CancellationRunner(crash_scancel=True)
    with pytest.raises(OSError, match="transport interruption"):
        cancel_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=lambda: NOW,
            runner=crashing,
        )
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.cancellation is not None
    assert replay.cancellation.actions[0].disposition == "cancel-requesting"
    assert len([event for event in replay.events if isinstance(event, PhaseJobCancellationRequestIntendedEvent)]) == 1

    recovered = CancellationRunner(accounting_states=[("RUNNING", None), ("CANCELLED", "0:15")])
    result = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=recovered,
    )
    assert result.status == "cancelled"
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert len([event for event in replay.events if isinstance(event, PhaseJobCancellationRequestIntendedEvent)]) == 1
    assert len([call for call in recovered.calls if call[0] == "scancel"]) == 1


def test_dispatching_target_is_only_correlation_recovered_and_never_resubmitted(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "dispatching")
    runner = CancellationRunner(
        correlation_job_id="5150",
        accounting_states=[("RUNNING", None), ("CANCELLED", "0:15")],
    )

    result = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=runner,
    )

    assert result.status == "cancelled"
    assert result.target_job_ids == ("5150",)
    assert not any(call[0] == "sbatch" for call in runner.calls)
    assert [call for call in runner.calls if call[0] == "scancel"] == [("scancel", "5150")]
    event_types = [
        getattr(event, "event_type", "") for event in PhaseAuthorityStore(authority_root).validate(phase_run_id).events
    ]
    cancellation_index = event_types.index("phase-cancellation-intended")
    assignment_index = event_types.index("phase-action-submitted")
    request_index = event_types.index("phase-job-cancellation-request-intended")
    assert cancellation_index < assignment_index < request_index


def test_unresolved_dispatch_correlation_remains_cancelling_without_sbatch_or_scancel(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "dispatching")
    runner = CancellationRunner(correlation_job_id=None)

    result = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=runner,
    )

    assert result.status == "cancelling"
    assert result.unresolved_action_ids == ("preprocessing-chunk-000000",)
    assert not any(call[0] in {"sbatch", "scancel"} for call in runner.calls)

    store = PhaseAuthorityStore(authority_root)
    replay = store.validate(phase_run_id)
    assert replay.cancellation is not None
    with pytest.raises(ValueError, match="pending job-bound"):
        store.append_event(
            phase_run_id,
            lambda sequence: PhaseJobCancellationRequestIntendedEvent(
                sequence=sequence,
                phase_run_id=phase_run_id,
                attempt_id=replay.phase_runspec.attempt_id,
                occurred_at="2026-08-20T13:05:00.000000Z",
                payload=PhaseJobCancellationRequestIntendedPayload(
                    cancellation_id=replay.cancellation.cancellation_id,
                    action_id="preprocessing-chunk-000000",
                    job_id="5150",
                    request_ordinal=1,
                    scancel_argv=("scancel", "5150"),
                ),
            ),
        )


def test_multiple_dispatch_correlations_remain_unresolved_without_external_effects(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "dispatching")

    class MultipleCorrelationRunner(CancellationRunner):
        def __call__(self, argv: tuple[str, ...]) -> CommandResult:
            if argv[0] == "squeue" and any(item.startswith("--name=") for item in argv):
                self.calls.append(argv)
                token = next(item.removeprefix("--name=") for item in argv if item.startswith("--name="))
                jobs = [
                    {"job_id": job_id, "name": token, "comment": token, "job_state": "RUNNING"}
                    for job_id in (5150, 5151)
                ]
                return CommandResult(argv, 0, json.dumps({"jobs": jobs}), "")
            return super().__call__(argv)

    runner = MultipleCorrelationRunner()
    result = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=runner,
    )

    assert result.status == "cancelling"
    assert result.unresolved_action_ids == ("preprocessing-chunk-000000",)
    assert any("multiple Slurm jobs" in warning for warning in result.warnings)
    assert not any(call[0] in {"sbatch", "scancel"} for call in runner.calls)


def test_unavailable_dispatch_correlation_remains_unresolved_without_external_effects(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "dispatching")
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        raise OSError("scheduler transport unavailable")

    result = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=runner,
    )

    assert result.status == "cancelling"
    assert result.unresolved_action_ids == ("preprocessing-chunk-000000",)
    assert result.warnings == ("scheduler transport unavailable",)
    assert not any(call[0] in {"sbatch", "scancel"} for call in calls)


def test_assignment_publication_value_error_is_not_masked_as_unresolved_correlation(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "dispatching")

    def reject_assignment(staged: Path) -> None:
        if json.loads(staged.read_text())["event_type"] == "phase-action-submitted":
            raise ValueError("simulated assignment publication invariant")

    store = PhaseAuthorityStore(authority_root, before_event_publish=reject_assignment)
    runner = CancellationRunner(correlation_job_id="5150")
    with pytest.raises(ValueError, match="assignment publication invariant"):
        cancel_phase(
            phase_run_id,
            authority_root=authority_root,
            authority_store=store,
            clock=lambda: NOW,
            runner=runner,
        )

    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.cancellation is not None
    assert replay.cancellation.actions[0].disposition == "correlating"
    assert not any(call[0] in {"sbatch", "scancel"} for call in runner.calls)


def test_failed_boundary_is_rejected_before_remote_effects(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    _record_terminal(authority_root, phase_run_id, state="FAILED", exit_code="1:0", outcome="failed")

    with pytest.raises(ValueError, match="failed Phase Attempt boundary"):
        cancel_phase(phase_run_id, authority_root=authority_root, runner=_reject_remote)


def test_rejected_dispatch_boundary_is_rejected_without_cancellation_intent_or_remote_effects(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission(authority_root, phase_run_id, outcome="rejected")

    with pytest.raises(ValueError, match="failed Phase Attempt boundary"):
        cancel_phase(phase_run_id, authority_root=authority_root, runner=_reject_remote)
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.cancellation is None


def test_accepted_boundary_is_rejected_before_remote_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    _finalize(inputs)
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert hasattr(fixture, "runspec")
    assert isinstance(authority_root, Path)

    with pytest.raises(ValueError, match="accepted Phase Attempt boundary"):
        cancel_phase(fixture.runspec.phase_run_id, authority_root=authority_root, runner=_reject_remote)


def test_cancellation_intent_blocks_submit_resume_and_finalize_before_external_or_evidence_io(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    runner = CancellationRunner(accounting_states=[("RUNNING", None)])
    result = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=runner,
    )
    assert result.status == "cancelling"

    with pytest.raises(ValueError, match="cancelling Phase Attempt boundary"):
        submit_phase(phase_run_id, authority_root=authority_root, runner=_reject_remote)
    with pytest.raises(ValueError, match="cancelling Phase Attempt boundary"):
        resume_phase(phase_run_id, authority_root=authority_root, runner=_reject_remote)
    with pytest.raises(ValueError, match="cancelling Phase Attempt boundary"):
        finalize_phase(
            phase_run_id,
            authority_root=authority_root,
            scheduler_evidence_path=tmp_path / "missing-scheduler.json",
            action_evidence_path=tmp_path / "missing-action.json",
            handoff_path=tmp_path / "missing-handoff",
        )


def test_acknowledged_request_is_not_repeated_while_accounting_remains_live_and_status_is_read_only(
    tmp_path: Path,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    first_runner = CancellationRunner(accounting_states=[("RUNNING", None)])
    first = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=first_runner,
    )
    assert first.status == "cancelling"
    assert first.cancellation_requested_job_ids == ("123456",)

    second_runner = CancellationRunner(accounting_states=[("RUNNING", None)])
    second = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=second_runner,
    )
    assert second.status == "cancelling"
    assert not any(call[0] == "scancel" for call in second_runner.calls)
    before_status = {
        path.relative_to(authority_root).as_posix(): path.read_bytes()
        for path in authority_root.rglob("*")
        if path.is_file()
    }
    status_runner = CancellationRunner(accounting_states=[("CANCELLED", "0:15")])
    report = status_phase(phase_run_id, authority_root=authority_root, runner=status_runner)
    after_status = {
        path.relative_to(authority_root).as_posix(): path.read_bytes()
        for path in authority_root.rglob("*")
        if path.is_file()
    }
    assert report.status == "cancelling"
    assert before_status == after_status
    assert not any(call[0] == "scancel" for call in status_runner.calls)


def test_request_result_publication_fault_preserves_ambiguous_request_for_exact_recovery(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")

    def interrupt_result(staged: Path) -> None:
        if json.loads(staged.read_text())["event_type"] == "phase-job-cancellation-request-result":
            raise OSError("simulated result publication interruption")

    store = PhaseAuthorityStore(authority_root, before_event_publish=interrupt_result)
    runner = CancellationRunner(accounting_states=[("RUNNING", None)])
    with pytest.raises(OSError, match="result publication interruption"):
        cancel_phase(
            phase_run_id,
            authority_root=authority_root,
            authority_store=store,
            clock=lambda: NOW,
            runner=runner,
        )
    assert len([call for call in runner.calls if call[0] == "scancel"]) == 1
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.cancellation is not None
    assert replay.cancellation.actions[0].disposition == "cancel-requesting"


def test_graph_and_per_job_intents_are_replayable_before_scancel_effect(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")

    class InspectingRunner(CancellationRunner):
        def __call__(self, argv: tuple[str, ...]) -> CommandResult:
            if argv[0] == "scancel":
                replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
                assert replay.lifecycle.attempt_status == "cancelling"
                assert replay.cancellation is not None
                assert replay.cancellation.actions[0].disposition == "cancel-requesting"
                assert isinstance(replay.events[-1], PhaseJobCancellationRequestIntendedEvent)
            return super().__call__(argv)

    runner = InspectingRunner(accounting_states=[("RUNNING", None), ("CANCELLED", "0:15")])
    result = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=runner,
    )
    assert result.status == "cancelled"


def test_concurrent_cancel_calls_serialize_and_the_completed_repeat_is_remote_free(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    scancel_started = threading.Event()
    release_scancel = threading.Event()

    class BlockingRunner(CancellationRunner):
        def __call__(self, argv: tuple[str, ...]) -> CommandResult:
            if argv[0] == "scancel":
                scancel_started.set()
                if not release_scancel.wait(timeout=5):
                    raise TimeoutError("test did not release scancel")
            return super().__call__(argv)

    runner = BlockingRunner(accounting_states=[("RUNNING", None), ("CANCELLED", "0:15")])
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            cancel_phase,
            phase_run_id,
            authority_root=authority_root,
            clock=lambda: NOW,
            runner=runner,
        )
        assert scancel_started.wait(timeout=5)
        second = pool.submit(
            cancel_phase,
            phase_run_id,
            authority_root=authority_root,
            clock=lambda: NOW,
            runner=_reject_remote,
        )
        release_scancel.set()
        assert first.result(timeout=5).status == "cancelled"
        assert second.result(timeout=5).status == "cancelled"

    assert len([call for call in runner.calls if call[0] == "scancel"]) == 1


def test_cancellation_preserves_every_preexisting_authority_byte(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    authority_path = authority_root / phase_run_id
    before = {
        path.relative_to(authority_path).as_posix(): path.read_bytes()
        for path in authority_path.rglob("*")
        if path.is_file()
    }
    runner = CancellationRunner(accounting_states=[("RUNNING", None), ("CANCELLED", "0:15")])

    cancel_phase(phase_run_id, authority_root=authority_root, clock=lambda: NOW, runner=runner)

    for relative, content in before.items():
        assert (authority_path / relative).read_bytes() == content


def test_status_and_cli_expose_durable_cancellation_without_changing_legacy_run_help(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    cancel_phase(phase_run_id, authority_root=authority_root, clock=lambda: NOW, runner=_reject_remote)

    report = status_phase(phase_run_id, authority_root=authority_root, runner=_reject_remote)
    assert report.status == report.attempt_status == "cancelled"
    assert report.cancellation is not None
    assert report.to_mapping()["cancellation"] is not None
    assert "  cancellation:" in report.render_table()

    cli_runner = CliRunner()
    legacy_help = cli_runner.invoke(cli, ["run", "--help"]).output
    result = cli_runner.invoke(
        cli,
        ["phase", "cancel", phase_run_id, "--authority-root", str(authority_root)],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "cancelled"
    assert cli_runner.invoke(cli, ["run", "--help"]).output == legacy_help


def test_one_job_cancellation_transport_returns_nonzero_result_and_legacy_wrapper_still_raises() -> None:
    calls: list[tuple[str, ...]] = []

    def reject(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        return CommandResult(argv, 7, "", "denied\n")

    transport = RemoteSlurmTransport(kind="ssh", ssh_target="login.example", runner=reject)
    result = transport.request_job_cancellation("42")

    assert result.returncode == 7
    assert result.argv[0:2] == ("ssh", "login.example")
    assert result.argv[2] == wrap_remote_command("scancel 42")
    with pytest.raises(ValueError, match="denied"):
        transport.cancel_jobs(("42",))
    with pytest.raises(ValueError, match="numeric"):
        transport.request_job_cancellation("42; rm")
    assert len(calls) == 2


def test_assignment_replay_rejects_one_job_id_bound_to_two_synthetic_actions(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    base = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    original = render_phase_submission_intent(
        base.phase_runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="6" * 64,
    )
    original_plan = original.actions[0]
    second_action_id = "preprocessing-chunk-000001"
    second_identity = phase_action_submission_identity_mapping(
        action_id=second_action_id,
        runtime_action_digest=original_plan.runtime_action_digest,
        dependency_action_ids=(),
        cluster_script_path=original_plan.cluster_script_path.replace(original_plan.action_id, second_action_id),
        action_evidence_path=original_plan.action_evidence_path.replace(original_plan.action_id, second_action_id),
        handoff_path=original_plan.handoff_path.replace(original_plan.action_id, second_action_id),
    )
    identities = (original_plan.submission_identity_mapping(), second_identity)
    submission_id = phase_submission_id(
        phase_run_id=phase_run_id,
        attempt_id=base.phase_runspec.attempt_id,
        phase_runspec_digest=base.phase_runspec.digest,
        phase_runspec_document_sha256=original.phase_runspec_document_sha256,
        qualification_tuple_id=original.qualification_tuple_id,
        actions=identities,
    )

    def plan(identity: dict[str, object]) -> PhaseActionSubmissionPlan:
        action_id = str(identity["action_id"])
        token = phase_action_scheduler_correlation_token(submission_id, action_id)
        body = original_plan.script_body.replace(original_plan.scheduler_correlation_token, token)
        body = body.replace(original_plan.action_id, action_id)
        return PhaseActionSubmissionPlan(
            action_id=action_id,
            runtime_action_digest=str(identity["runtime_action_digest"]),
            dependency_action_ids=(),
            cluster_script_path=str(identity["cluster_script_path"]),
            script_sha256=sha256(body.encode()).hexdigest(),
            script_body=body,
            job_name=token,
            scheduler_correlation_token=token,
            action_evidence_path=str(identity["action_evidence_path"]),
            handoff_path=str(identity["handoff_path"]),
        )

    first_plan, second_plan = (plan(identity) for identity in identities)
    submission = PhaseSubmissionLifecycleView(
        phase_run_id=phase_run_id,
        attempt_id=base.phase_runspec.attempt_id,
        submission_id=submission_id,
        phase_runspec_location=original.phase_runspec_location,
        phase_runspec_digest=original.phase_runspec_digest,
        phase_runspec_document_sha256=original.phase_runspec_document_sha256,
        qualification_tuple_id=original.qualification_tuple_id,
        actions=(
            PhaseActionSubmissionView(
                plan=first_plan,
                status="submitted",
                job_id="77",
                sbatch_argv=("sbatch", "--parsable", first_plan.cluster_script_path),
            ),
            PhaseActionSubmissionView(plan=second_plan, status="dispatching"),
        ),
        status="submitting",
    )
    synthetic_authority = replace(base, submission=submission)
    event = PhaseActionSubmittedEvent(
        sequence=2,
        phase_run_id=phase_run_id,
        attempt_id=base.phase_runspec.attempt_id,
        occurred_at="2026-08-20T13:00:00.000000Z",
        payload=PhaseActionSubmittedPayload(
            submission_id=submission.submission_id,
            action_id=second_plan.action_id,
            script_sha256=second_plan.script_sha256,
            scheduler_correlation_token=second_plan.scheduler_correlation_token,
            dependency_job_ids=(),
            job_id="77",
            sbatch_argv=("sbatch", "--parsable", second_plan.cluster_script_path),
        ),
    )

    with pytest.raises(ValueError, match="already assigned to another Runtime Action"):
        _replay_action_submitted_event(event, synthetic_authority)
