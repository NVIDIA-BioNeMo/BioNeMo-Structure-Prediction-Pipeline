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

"""Bounded idempotent Phase Resume coordination tests."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from bspp.orchestration.contract.database_placement import DatabaseSetSelection
from bspp.orchestration.contract.phase import PhasePlan, PreprocessingPhasePlanPayload
from bspp.orchestration.contract.phase_reconciliation import (
    PhaseActionTerminalObservedEvent,
    PhaseActionTerminalObservedPayload,
    PhaseActionTerminalOutcome,
)
from bspp.orchestration.contract.phase_state import (
    PhaseAttempt,
    PhaseMaterializedEvent,
    PhaseMaterializedPayload,
    PhaseRun,
)
from bspp.orchestration.contract.phase_submission import (
    PhaseActionDispatchIntendedEvent,
    PhaseActionDispatchIntendedPayload,
    PhaseActionDispatchRejectedEvent,
    PhaseActionSubmittedEvent,
    PhaseActionSubmittedPayload,
    PhaseSubmissionIntendedEvent,
)
from bspp.orchestration.contract.preprocessing_execution import (
    preprocessing_chunk_execution_intent_from_plan,
)
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore
from bspp.orchestration.control.phase_rendering import render_phase_submission_intent
from bspp.orchestration.control.phase_resume import (
    PHASE_RESUME_NEW_ACTION_LIMIT,
    resume_phase,
)
from bspp.orchestration.control.phase_status import status_phase
from bspp.orchestration.control.transport import CommandResult, default_command_runner
from tests.support.preprocessing_execution import preprocessing_execution_fixture

NOW = datetime(2026, 8, 20, 13, 0, tzinfo=UTC)


def _materialized_authority(tmp_path: Path) -> tuple[Path, str]:
    fixture = preprocessing_execution_fixture(tmp_path / "work")
    plan = PhasePlan(
        target_cluster=fixture.runspec.cluster.profile_name,
        input_location=fixture.runspec.input_location,
        payload=PreprocessingPhasePlanPayload(
            work_plan=fixture.runspec.payload.work_plan,
            chunk_execution_intent=preprocessing_chunk_execution_intent_from_plan(fixture.action.payload),
            database=DatabaseSetSelection(
                database_set=fixture.runspec.payload.database.database_set,
                requested_policy=fixture.runspec.payload.database.requested_policy,
            ),
        ),
    )
    runspec = replace(fixture.runspec, phase_plan_digest=plan.digest)
    attempt = PhaseAttempt(
        attempt_id=runspec.attempt_id,
        ordinal=1,
        phase_runspec_location=f"attempts/{runspec.attempt_id}/phase-runspec.json",
        phase_runspec_digest=runspec.digest,
        created_at=runspec.materialized_at,
    )
    phase_run = PhaseRun(
        phase_run_id=runspec.phase_run_id,
        phase_plan_location="phase-plan.json",
        phase_plan_digest=plan.digest,
        created_at=runspec.materialized_at,
        current_attempt_id=attempt.attempt_id,
        attempts=(attempt,),
    )
    event = PhaseMaterializedEvent(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        occurred_at=runspec.materialized_at,
        payload=PhaseMaterializedPayload(phase_run=phase_run, phase_runspec=runspec),
    )
    authority_root = tmp_path / "authority"
    PhaseAuthorityStore(authority_root).publish(
        phase_plan=plan,
        phase_run=phase_run,
        phase_runspec=runspec,
        materialized_event=event,
    )
    return authority_root, runspec.phase_run_id


def _record_submission_state(
    authority_root: Path,
    phase_run_id: str,
    state: str,
    *,
    job_id: str = "123456",
    intended_at: str = "2026-08-20T12:00:00.000000Z",
) -> None:
    store = PhaseAuthorityStore(authority_root)
    authority = store.validate(phase_run_id)
    attempt = authority.current_attempt
    document_sha = hashlib.sha256((authority.authority_path / attempt.phase_runspec_location).read_bytes()).hexdigest()
    carry_record = authority.current_carry_forward
    carry_document_sha = None
    if carry_record is not None:
        reference = authority.phase_runspec.carry_forward
        assert reference is not None
        carry_document_sha = hashlib.sha256((authority.authority_path / reference.location).read_bytes()).hexdigest()
    intent = render_phase_submission_intent(
        authority.phase_runspec,
        phase_runspec_location=attempt.phase_runspec_location,
        phase_runspec_document_sha256=document_sha,
        carry_forward_record=carry_record,
        carry_forward_document_sha256=carry_document_sha,
    )
    authority = store.append_event(
        phase_run_id,
        lambda sequence: PhaseSubmissionIntendedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at=intended_at,
            payload=intent,
        ),
    )
    if state == "planned":
        return
    plan = intent.actions[0]
    dispatch = PhaseActionDispatchIntendedPayload(
        submission_id=intent.submission_id,
        action_id=plan.action_id,
        script_sha256=plan.script_sha256,
        scheduler_correlation_token=plan.scheduler_correlation_token,
        dependency_job_ids=(),
    )
    authority = store.append_event(
        phase_run_id,
        lambda sequence: PhaseActionDispatchIntendedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at="2026-08-20T12:01:00.000000Z",
            payload=dispatch,
        ),
    )
    if state == "dispatching":
        return
    if state != "submitted":
        raise AssertionError(f"unsupported submission state: {state}")
    store.append_event(
        phase_run_id,
        lambda sequence: PhaseActionSubmittedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at="2026-08-20T12:02:00.000000Z",
            payload=PhaseActionSubmittedPayload(
                submission_id=intent.submission_id,
                action_id=plan.action_id,
                script_sha256=plan.script_sha256,
                scheduler_correlation_token=plan.scheduler_correlation_token,
                dependency_job_ids=(),
                job_id=job_id,
                sbatch_argv=("sbatch", "--parsable", plan.cluster_script_path),
            ),
        ),
    )


def _record_terminal(
    authority_root: Path,
    phase_run_id: str,
    *,
    state: str = "COMPLETED",
    exit_code: str | None = "0:0",
    outcome: PhaseActionTerminalOutcome = "succeeded",
    observed_at: str = "2026-08-20T12:03:00.000000Z",
) -> None:
    store = PhaseAuthorityStore(authority_root)
    authority = store.validate(phase_run_id)
    assert authority.submission is not None
    action = authority.submission.actions[0]
    assert action.job_id is not None
    payload = PhaseActionTerminalObservedPayload(
        submission_id=authority.submission.submission_id,
        phase_runspec_digest=authority.phase_runspec.digest,
        action_id=action.action_id,
        runtime_action_digest=action.plan.runtime_action_digest,
        scheduler_correlation_token=action.scheduler_correlation_token,
        job_id=action.job_id,
        state=state,
        exit_code=exit_code,
        outcome=outcome,
    )
    store.append_event(
        phase_run_id,
        lambda sequence: PhaseActionTerminalObservedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at=observed_at,
            payload=payload,
        ),
    )


class FakeRunner:
    def __init__(
        self,
        *,
        squeue_jobs: list[dict[str, object]] | None = None,
        sacct_jobs: list[dict[str, object]] | None = None,
        sacct_parsable_stdout: str | None = None,
        sbatch_job_id: str = "123456",
        sbatch_returncode: int = 0,
        sbatch_stderr: str = "",
    ) -> None:
        self.squeue_jobs = squeue_jobs or []
        self.sacct_jobs = sacct_jobs or []
        self.sacct_parsable_stdout = sacct_parsable_stdout
        self.sbatch_job_id = sbatch_job_id
        self.sbatch_returncode = sbatch_returncode
        self.sbatch_stderr = sbatch_stderr
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "squeue":
            return CommandResult(argv, 0, json.dumps({"jobs": self.squeue_jobs}), "")
        if argv[0] == "sacct":
            if "--parsable2" in argv and self.sacct_parsable_stdout is not None:
                return CommandResult(argv, 0, self.sacct_parsable_stdout, "")
            return CommandResult(argv, 0, json.dumps({"jobs": self.sacct_jobs}), "")
        if argv[0] == "sbatch":
            stdout = self.sbatch_job_id + "\n" if self.sbatch_returncode == 0 else ""
            return CommandResult(argv, self.sbatch_returncode, stdout, self.sbatch_stderr)
        return default_command_runner(argv)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _unset_signal_sacct_job(state: str, return_code: int) -> dict[str, object]:
    """Observed Slurm 24.05 accounting shape, with no exact JSON signal."""
    return {
        "job_id": 123456,
        "state": {"current": [state]},
        "exit_code": {
            "return_code": {"set": True, "infinite": False, "number": return_code},
            "signal": {"id": {"set": False, "infinite": False, "number": 0}, "name": ""},
        },
    }


@pytest.mark.parametrize(("state", "return_code", "outcome"), [("COMPLETED", 0, "succeeded"), ("FAILED", 1, "failed")])
def test_resume_scalar_uses_exact_fallback_for_unset_json_signal(
    tmp_path: Path, state: str, return_code: int, outcome: str
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    runner = FakeRunner(
        sacct_jobs=[_unset_signal_sacct_job(state, return_code)],
        sacct_parsable_stdout=f"123456|123456|{state}|{return_code}:0|0|\n",
    )

    result = resume_phase(phase_run_id, authority_root=authority_root, clock=lambda: NOW, runner=runner)

    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert result.terminal_action_ids == ("preprocessing-chunk-000000",)
    assert replay.terminal_observations[0].outcome == outcome
    assert replay.terminal_observations[0].exit_code == f"{return_code}:0"
    assert any("sacct_json_incomplete" in warning for warning in result.warnings)
    fallbacks = [call for call in runner.calls if "--parsable2" in call]
    assert len(fallbacks) == 1
    assert "--format=JobIDRaw,JobID,State,ExitCode,Restarts" in fallbacks[0]
    assert not any(call[0] == "sbatch" for call in runner.calls)


@pytest.mark.parametrize(
    "fallback",
    ["", "123456|123456|COMPLETED|0|0|\n", "123456|999999|COMPLETED|0:0|0|\n"],
)
def test_resume_scalar_does_not_infer_success_without_exact_fallback(tmp_path: Path, fallback: str) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    before = _tree_bytes(authority_root / phase_run_id)
    runner = FakeRunner(sacct_jobs=[_unset_signal_sacct_job("COMPLETED", 0)], sacct_parsable_stdout=fallback)

    result = resume_phase(phase_run_id, authority_root=authority_root, clock=lambda: NOW, runner=runner)

    assert result.outcome == "no-op"
    assert result.terminal_action_ids == ()
    assert _tree_bytes(authority_root / phase_run_id) == before
    assert any("sacct_json_incomplete" in warning for warning in result.warnings)
    assert not any(call[0] == "sbatch" for call in runner.calls)


def test_resume_requires_existing_intent_without_remote_effect_or_new_attempt(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    authority_path = authority_root / phase_run_id
    before = _tree_bytes(authority_path)

    with pytest.raises(ValueError, match="existing Phase Submission intent"):
        resume_phase(
            phase_run_id,
            authority_root=authority_root,
            runner=lambda _argv: (_ for _ in ()).throw(AssertionError("unexpected remote call")),
        )

    assert _tree_bytes(authority_path) == before
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert len(authority.phase_run.attempts) == 1


def test_resume_dispatches_one_planned_remnant_once_then_never_duplicates_sbatch(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "planned")
    runner = FakeRunner()

    result = resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=runner,
    )

    assert result.outcome == "reconciled"
    assert result.submitted_action_ids == ("preprocessing-chunk-000000",)
    assert len([call for call in runner.calls if call[0] == "sbatch"]) == PHASE_RESUME_NEW_ACTION_LIMIT == 1
    second_runner = FakeRunner(squeue_jobs=[{"job_id": 123456, "job_state": "RUNNING"}])
    second = resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=second_runner,
    )
    assert second.outcome == "no-op"
    assert not any(call[0] == "sbatch" for call in second_runner.calls)
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.submission is not None
    assert replay.submission.actions[0].job_id == "123456"
    assert sum(isinstance(event, PhaseActionSubmittedEvent) for event in replay.events) == 1


def test_resume_returns_failed_after_one_durable_planned_action_rejection(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "planned")
    runner = FakeRunner(sbatch_returncode=1, sbatch_stderr="partition unavailable\n")

    result = resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=runner,
    )

    assert result.outcome == "failed"
    assert result.actions[0].durable_status == "rejected"
    assert result.submitted_action_ids == ()
    assert len([call for call in runner.calls if call[0] == "sbatch"]) == 1
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.submission is not None
    assert replay.submission.status == "failed"
    assert replay.submission.actions[0].status == "rejected"
    assert replay.lifecycle.attempt_status == replay.lifecycle.run_status == "failed"
    assert sum(isinstance(event, PhaseActionDispatchRejectedEvent) for event in replay.events) == 1

    with pytest.raises(ValueError, match="failed Phase Attempt boundary"):
        resume_phase(
            phase_run_id,
            authority_root=authority_root,
            runner=lambda _argv: (_ for _ in ()).throw(AssertionError("failed resume must be remote-free")),
        )


def test_resume_does_not_mask_rejection_event_append_failure(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "planned")

    def interrupt_rejection(staged_event: Path) -> None:
        if json.loads(staged_event.read_text())["event_type"] == "phase-action-dispatch-rejected":
            raise OSError("simulated rejection append interruption")

    store = PhaseAuthorityStore(authority_root, before_event_publish=interrupt_rejection)
    runner = FakeRunner(sbatch_returncode=1, sbatch_stderr="partition unavailable\n")
    with pytest.raises(OSError, match="rejection append interruption"):
        resume_phase(
            phase_run_id,
            authority_root=authority_root,
            authority_store=store,
            clock=lambda: NOW,
            runner=runner,
        )

    assert len([call for call in runner.calls if call[0] == "sbatch"]) == 1
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.submission is not None
    assert replay.submission.actions[0].status == "dispatching"
    assert not any(isinstance(event, PhaseActionDispatchRejectedEvent) for event in replay.events)


def test_resume_repairs_dispatching_by_original_correlation_then_observes_same_job(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "dispatching")
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert authority.submission is not None
    token = authority.submission.actions[0].scheduler_correlation_token
    runner = FakeRunner(
        squeue_jobs=[
            {
                "job_id": 5150,
                "name": token,
                "comment": token,
                "job_state": "RUNNING",
            }
        ]
    )

    result = resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=runner,
    )

    assert result.recovered_action_ids == ("preprocessing-chunk-000000",)
    assert result.actions[0].job_id == "5150"
    assert not any(call[0] == "sbatch" for call in runner.calls)
    correlation_sacct = next(call for call in runner.calls if call[0] == "sacct" and "--name=" + token in call)
    assert "--starttime=2026-08-20T12:00:00" in correlation_sacct
    assert len([call for call in runner.calls if call[0] == "squeue" and "-j" in call]) == 1


@pytest.mark.parametrize("matched", [[], ["7001", "7002"]])
def test_resume_uncertain_correlation_never_submits(tmp_path: Path, matched: list[str]) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "dispatching")
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert authority.submission is not None
    token = authority.submission.actions[0].scheduler_correlation_token
    runner = FakeRunner(
        squeue_jobs=[{"job_id": int(job), "name": token, "comment": token, "job_state": "PENDING"} for job in matched]
    )

    expected = "remains uncertain" if not matched else "multiple Slurm jobs"
    with pytest.raises(ValueError, match=expected):
        resume_phase(phase_run_id, authority_root=authority_root, clock=lambda: NOW, runner=runner)
    assert not any(call[0] == "sbatch" for call in runner.calls)
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.submission is not None and replay.submission.actions[0].status == "dispatching"


def test_resume_unavailable_correlation_remains_dispatching_without_sbatch(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "dispatching")
    authority_path = authority_root / phase_run_id
    before = _tree_bytes(authority_path)
    calls: list[tuple[str, ...]] = []

    def unavailable_runner(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        return CommandResult(argv, 1, "", "scheduler unavailable")

    with pytest.raises(ValueError, match="scheduler unavailable"):
        resume_phase(phase_run_id, authority_root=authority_root, runner=unavailable_runner)

    assert not any(call[0] == "sbatch" for call in calls)
    assert _tree_bytes(authority_path) == before
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.submission is not None and replay.submission.actions[0].status == "dispatching"


def test_resume_records_terminal_sacct_once_and_scheduler_success_is_not_acceptance(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    runner = FakeRunner(
        squeue_jobs=[{"job_id": 123456, "job_state": "RUNNING"}],
        sacct_jobs=[{"job_id_raw": "123456", "state": "COMPLETED", "exit_code": "0:0"}],
    )

    result = resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=runner,
    )

    assert result.outcome == "reconciled"
    assert result.terminal_action_ids == ("preprocessing-chunk-000000",)
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.lifecycle.attempt_status == "materialized"
    assert not replay.lifecycle.sealed and replay.receipt is None
    assert replay.terminal_observations[0].outcome == "succeeded"
    assert len([call for call in runner.calls if call[0] == "squeue" and "-j" in call]) == 1
    assert len([call for call in runner.calls if call[0] == "sacct" and "-j" in call]) == 1

    second = resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=lambda _argv: (_ for _ in ()).throw(AssertionError("terminal action must not be queried again")),
    )
    assert second.outcome == "no-op"
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert sum(isinstance(event, PhaseActionTerminalObservedEvent) for event in replay.events) == 1


def test_resume_retries_terminal_observation_after_event_append_interruption(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")

    def interrupt_terminal(staged_event: Path) -> None:
        if json.loads(staged_event.read_text())["event_type"] == "phase-action-terminal-observed":
            raise OSError("simulated terminal append interruption")

    interrupted_store = PhaseAuthorityStore(authority_root, before_event_publish=interrupt_terminal)
    observation = [{"job_id_raw": "123456", "state": "COMPLETED", "exit_code": "0:0"}]
    with pytest.raises(OSError, match="terminal append interruption"):
        resume_phase(
            phase_run_id,
            authority_root=authority_root,
            authority_store=interrupted_store,
            clock=lambda: NOW,
            runner=FakeRunner(sacct_jobs=observation),
        )

    assert not PhaseAuthorityStore(authority_root).validate(phase_run_id).terminal_observations
    recovered = resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: NOW,
        runner=FakeRunner(sacct_jobs=observation),
    )
    assert recovered.terminal_action_ids == ("preprocessing-chunk-000000",)
    assert len(PhaseAuthorityStore(authority_root).validate(phase_run_id).terminal_observations) == 1


@pytest.mark.parametrize(
    ("squeue_jobs", "sacct_jobs", "warning"),
    [
        ([], [], "missing from current scheduler observation"),
        ([{"job_id": 123456, "job_state": "COMPLETED"}], [], "without sacct confirmation"),
        ([], [{"job_id_raw": "123456", "state": "COMPLETED"}], "lacks an exit code"),
        ([], [{"job_id_raw": "123456", "state": "RUNNING"}], None),
    ],
)
def test_resume_inconclusive_observations_are_noop(
    tmp_path: Path,
    squeue_jobs: list[dict[str, object]],
    sacct_jobs: list[dict[str, object]],
    warning: str | None,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    authority_path = authority_root / phase_run_id
    before = _tree_bytes(authority_path)
    runner = FakeRunner(squeue_jobs=squeue_jobs, sacct_jobs=sacct_jobs)

    result = resume_phase(phase_run_id, authority_root=authority_root, clock=lambda: NOW, runner=runner)

    assert result.outcome == "no-op"
    assert not result.terminal_action_ids
    assert (warning is None) == (not result.warnings)
    if warning is not None:
        assert any(warning in item for item in result.warnings)
    assert _tree_bytes(authority_path) == before
    assert not any(call[0] == "sbatch" for call in runner.calls)


@pytest.mark.parametrize(("unavailable", "available"), [("squeue", "sacct"), ("sacct", "squeue")])
def test_resume_preserves_unavailable_source_warning_without_mutation(
    tmp_path: Path,
    unavailable: str,
    available: str,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    authority_path = authority_root / phase_run_id
    before = _tree_bytes(authority_path)

    def runner(argv: tuple[str, ...]) -> CommandResult:
        if argv[0] == unavailable:
            return CommandResult(argv, 1, "", f"{unavailable} unavailable")
        assert argv[0] == available
        return CommandResult(argv, 0, json.dumps({"jobs": []}), "")

    result = resume_phase(phase_run_id, authority_root=authority_root, runner=runner)

    assert result.outcome == "no-op"
    assert any(unavailable in warning for warning in result.warnings)
    assert _tree_bytes(authority_path) == before


def test_resume_preserves_both_unavailable_source_warnings_without_mutation(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    authority_path = authority_root / phase_run_id
    before = _tree_bytes(authority_path)
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        return CommandResult(argv, 1, "", f"{argv[0]} unavailable")

    result = resume_phase(phase_run_id, authority_root=authority_root, runner=runner)

    assert result.outcome == "no-op"
    assert len(result.warnings) == 2
    assert "squeue_unavailable" in result.warnings[0]
    assert "sacct_unavailable" in result.warnings[1]
    assert {call[0] for call in calls} == {"squeue", "sacct"}
    assert not any(call[0] == "sbatch" for call in calls)
    assert _tree_bytes(authority_path) == before


@pytest.mark.parametrize(
    ("state", "exit_code"),
    [("FAILED", "1:0"), ("TIMEOUT", None), ("COMPLETED", "2:0")],
)
def test_resume_terminal_failure_creates_failed_boundary_and_status_projection(
    tmp_path: Path,
    state: str,
    exit_code: str | None,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    runner = FakeRunner(sacct_jobs=[{"job_id_raw": "123456", "state": state, "exit_code": exit_code}])

    result = resume_phase(phase_run_id, authority_root=authority_root, clock=lambda: NOW, runner=runner)

    assert result.outcome == "failed"
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.lifecycle.attempt_status == replay.lifecycle.run_status == "failed"
    assert replay.submission is not None and replay.submission.status == "submitted"
    status = status_phase(
        phase_run_id,
        authority_root=authority_root,
        runner=FakeRunner(),
    )
    assert status.status == "failed" and status.attempt_status == "failed"
    assert status.actions[0].durable_status == "submitted"
    assert status.actions[0].terminal is not None and status.actions[0].terminal.outcome == "failed"

    before_status = _tree_bytes(authority_root / phase_run_id)
    repeated_status = status_phase(phase_run_id, authority_root=authority_root, runner=FakeRunner())
    assert repeated_status.actions[0].terminal == status.actions[0].terminal
    assert _tree_bytes(authority_root / phase_run_id) == before_status

    with pytest.raises(ValueError, match="failed Phase Attempt boundary"):
        resume_phase(
            phase_run_id,
            authority_root=authority_root,
            runner=lambda _argv: (_ for _ in ()).throw(AssertionError("failed resume must be remote-free")),
        )


def test_resume_uses_frozen_intent_after_qualification_expiry(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "planned")

    result = resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: datetime(2099, 1, 1, tzinfo=UTC),
        runner=FakeRunner(),
    )

    assert result.submitted_action_ids == ("preprocessing-chunk-000000",)


def test_resume_rejects_stored_runspec_byte_drift_before_remote_contact(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "planned")
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    runspec_path = authority.authority_path / authority.phase_run.attempts[0].phase_runspec_location
    runspec_path.write_bytes(runspec_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="RunSpec bytes"):
        resume_phase(
            phase_run_id,
            authority_root=authority_root,
            runner=lambda _argv: (_ for _ in ()).throw(AssertionError("unexpected remote call")),
        )


def test_resume_rejects_invalid_run_id_before_remote_contact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="valid Phase Run id"):
        resume_phase(
            "../not-a-phase-run",
            authority_root=tmp_path,
            runner=lambda _argv: (_ for _ in ()).throw(AssertionError("unexpected remote call")),
        )


def test_resume_rejects_preexisting_accepted_boundary_before_remote_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_phase_finalization import _accepted_inputs, _finalize

    inputs = _accepted_inputs(tmp_path, monkeypatch)
    finalized = _finalize(inputs)
    authority_root = inputs["authority_root"]
    assert isinstance(authority_root, Path)

    with pytest.raises(ValueError, match="accepted Phase Attempt boundary"):
        resume_phase(
            finalized.phase_run_id,
            authority_root=authority_root,
            runner=lambda _argv: (_ for _ in ()).throw(AssertionError("accepted resume must be remote-free")),
        )


def test_duplicate_terminal_observation_is_rejected_by_authority_replay(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    _record_terminal(authority_root, phase_run_id)
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    first = next(event for event in authority.events if isinstance(event, PhaseActionTerminalObservedEvent))

    with pytest.raises(ValueError, match="exactly one durable terminal observation"):
        PhaseAuthorityStore(authority_root).append_event(
            phase_run_id,
            lambda sequence: replace(first, sequence=sequence),
        )


def test_concurrent_resumes_serialize_one_planned_dispatch(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "planned")
    runner = FakeRunner()

    def invoke() -> object:
        return resume_phase(phase_run_id, authority_root=authority_root, clock=lambda: NOW, runner=runner)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _index: invoke(), range(2)))

    assert {result.outcome for result in results} == {"reconciled", "no-op"}
    assert len([call for call in runner.calls if call[0] == "sbatch"]) == 1
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert sum(isinstance(event, PhaseActionSubmittedEvent) for event in replay.events) == 1


def test_resume_takes_exactly_one_operation_lock_and_never_imports_public_lifecycle_calls(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")

    class CountingStore(PhaseAuthorityStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.acquisitions = 0

        @contextmanager
        def phase_operation_lock(self, locked_phase_run_id: str) -> Iterator[None]:
            self.acquisitions += 1
            with super().phase_operation_lock(locked_phase_run_id):
                yield

    store = CountingStore(authority_root)
    resume_phase(phase_run_id, authority_root=authority_root, authority_store=store, runner=FakeRunner())
    assert store.acquisitions == 1

    import bspp.orchestration.control.phase_resume as phase_resume_module

    source = inspect.getsource(phase_resume_module)
    assert "submit_phase" not in source
    assert "finalize_phase" not in source
    assert "sleep(" not in source


def test_phase_resume_cli_json_and_legacy_help_are_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission_state(authority_root, phase_run_id, "submitted")
    cli_runner = CliRunner()
    legacy_help = cli_runner.invoke(cli, ["run", "--help"]).output

    import bspp.orchestration.control.phase_resume as phase_resume_module

    real_resume = phase_resume_module.resume_phase

    def resume_with_fake_runner(
        requested_phase_run_id: str,
        *,
        authority_root: Path,
    ) -> object:
        return real_resume(
            requested_phase_run_id,
            authority_root=authority_root,
            runner=FakeRunner(),
        )

    monkeypatch.setattr(phase_resume_module, "resume_phase", resume_with_fake_runner)
    result = cli_runner.invoke(
        cli,
        ["phase", "resume", phase_run_id, "--authority-root", str(authority_root)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["outcome"] == "no-op"
    assert cli_runner.invoke(cli, ["run", "--help"]).output == legacy_help
