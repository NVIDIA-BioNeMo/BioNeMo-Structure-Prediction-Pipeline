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

"""Read-only Phase Status service and CLI tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

from bspp.orchestration.contract.database_placement import DatabaseSetSelection
from bspp.orchestration.contract.phase import PhasePlan, PreprocessingPhasePlanPayload
from bspp.orchestration.contract.phase_reconciliation import (
    FoldingActionTerminalObservationView,
    FoldingTaskTerminalEvidence,
    PhaseActionTerminalObservationView,
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
    PhaseActionDispatchRejectedPayload,
    PhaseActionSubmittedEvent,
    PhaseActionSubmittedPayload,
    PhaseSubmissionIntendedEvent,
)
from bspp.orchestration.contract.preprocessing_execution import (
    preprocessing_chunk_execution_intent_from_plan,
)
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.phase_authority import (
    PhaseAuthorityStore,
    _select_terminal_observations_by_action,
)
from bspp.orchestration.control.phase_rendering import render_phase_submission_intent
from bspp.orchestration.control.phase_status import (
    PhaseActionSchedulerStatus,
    PhaseActionStatus,
    _build_action_statuses,
    _ordered_unique_jobs,
    status_phase,
)
from bspp.orchestration.control.transport import CommandResult
from tests.support.preprocessing_execution import preprocessing_execution_fixture


class RecordingRunner:
    def __init__(self, responses: list[CommandResult]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        response = self.responses.pop(0)
        return replace(response, argv=argv)


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
    materialized = PhaseMaterializedEvent(
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
        materialized_event=materialized,
    )
    return authority_root, runspec.phase_run_id


def _record_submission(
    authority_root: Path,
    phase_run_id: str,
    *,
    outcome: str,
    job_id: str = "123456",
) -> None:
    store = PhaseAuthorityStore(authority_root)
    authority = store.validate(phase_run_id)
    attempt = authority.phase_run.attempts[0]
    document_sha256 = hashlib.sha256(
        (authority.authority_path / attempt.phase_runspec_location).read_bytes()
    ).hexdigest()
    intent = render_phase_submission_intent(
        authority.phase_runspec,
        phase_runspec_location=attempt.phase_runspec_location,
        phase_runspec_document_sha256=document_sha256,
    )
    authority = store.append_event(
        phase_run_id,
        lambda sequence: PhaseSubmissionIntendedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at="2026-08-20T12:00:00.000000Z",
            payload=intent,
        ),
    )
    if outcome == "planned":
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
    if outcome == "dispatching":
        return
    sbatch_argv = ("sbatch", "--parsable", plan.cluster_script_path)
    if outcome == "rejected":
        store.append_event(
            phase_run_id,
            lambda sequence: PhaseActionDispatchRejectedEvent(
                sequence=sequence,
                phase_run_id=phase_run_id,
                attempt_id=authority.phase_runspec.attempt_id,
                occurred_at="2026-08-20T12:02:00.000000Z",
                payload=PhaseActionDispatchRejectedPayload(
                    submission_id=intent.submission_id,
                    action_id=plan.action_id,
                    script_sha256=plan.script_sha256,
                    scheduler_correlation_token=plan.scheduler_correlation_token,
                    dependency_job_ids=(),
                    sbatch_argv=sbatch_argv,
                    return_code=1,
                    stdout="",
                    stderr="partition unavailable\n",
                ),
            ),
        )
        return
    if outcome != "submitted":
        raise AssertionError(f"unsupported test submission outcome: {outcome}")
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
                sbatch_argv=sbatch_argv,
            ),
        ),
    )


def _authority_snapshot(authority_root: Path) -> tuple[tuple[str, str, bytes | None], ...]:
    entries: list[tuple[str, str, bytes | None]] = []
    for path in sorted(authority_root.rglob("*")):
        relative = str(path.relative_to(authority_root))
        if path.is_symlink():
            entries.append((relative, "symlink", None))
        elif path.is_dir():
            entries.append((relative, "directory", None))
        else:
            entries.append((relative, "file", path.read_bytes()))
    return tuple(entries)


def _empty_scheduler_results() -> list[CommandResult]:
    return [
        CommandResult(argv=(), returncode=0, stdout='{"jobs": []}', stderr=""),
        CommandResult(argv=(), returncode=0, stdout='{"jobs": []}', stderr=""),
    ]


def test_materialized_status_reports_every_action_without_scheduler_or_authority_mutation(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    before = _authority_snapshot(authority_root)

    def reject_scheduler(_argv: tuple[str, ...]) -> CommandResult:
        raise AssertionError("unsubmitted Phase Status must not contact Slurm")

    report = status_phase(phase_run_id, authority_root=authority_root, runner=reject_scheduler)

    assert report.phase_run_id == phase_run_id
    assert report.attempt_id == "attempt-0001"
    assert report.attempt_status == "materialized"
    assert report.status == "materialized"
    assert report.scheduler_status == "not-requested"
    assert report.requested_job_ids == ()
    assert [(action.action_id, action.durable_status, action.scheduler.status) for action in report.actions] == [
        ("preprocessing-chunk-000000", "not-submitted", "not-applicable")
    ]
    assert _authority_snapshot(authority_root) == before


@pytest.mark.parametrize("outcome", ["planned", "dispatching", "rejected"])
def test_status_preserves_unassigned_durable_states_without_correlation(
    tmp_path: Path,
    outcome: str,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission(authority_root, phase_run_id, outcome=outcome)

    def reject_scheduler(_argv: tuple[str, ...]) -> CommandResult:
        raise AssertionError("an action without a durable job id must not be correlated")

    report = status_phase(phase_run_id, authority_root=authority_root, runner=reject_scheduler)

    assert report.actions[0].durable_status == outcome
    assert report.actions[0].job_id is None
    assert report.actions[0].scheduler.status == "not-applicable"
    assert report.status == ("failed" if outcome == "rejected" else "submitting")
    if outcome == "rejected":
        assert "partition unavailable" not in report.render_json()


def test_terminal_sacct_wins_and_repeated_observations_render_identically(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission(authority_root, phase_run_id, outcome="submitted")
    observations = [
        CommandResult(
            argv=(),
            returncode=0,
            stdout=json.dumps({"jobs": [{"job_id": 123456, "job_state": "RUNNING"}]}),
            stderr="",
        ),
        CommandResult(
            argv=(),
            returncode=0,
            stdout=json.dumps({"jobs": [{"job_id_raw": "123456", "state": "COMPLETED", "exit_code": "0:0"}]}),
            stderr="",
        ),
    ]
    runner = RecordingRunner([*observations, *observations])
    before = _authority_snapshot(authority_root)

    first = status_phase(phase_run_id, authority_root=authority_root, runner=runner)
    second = status_phase(phase_run_id, authority_root=authority_root, runner=runner)

    assert first == second
    assert first.to_mapping() == second.to_mapping()
    assert first.render_json() == second.render_json()
    assert first.render_table() == second.render_table()
    assert first.status == "submitted"
    assert first.scheduler_status == "complete"
    assert first.actions[0].durable_status == "submitted"
    assert first.actions[0].scheduler == PhaseActionSchedulerStatus(
        "observed",
        state="COMPLETED",
        source="sacct",
        exit_code="0:0",
    )
    assert [call[0] for call in runner.calls] == ["squeue", "sacct", "squeue", "sacct"]
    assert "observed_at" not in first.render_json()
    assert _authority_snapshot(authority_root) == before


def test_successful_empty_sources_report_missing_assigned_job(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission(authority_root, phase_run_id, outcome="submitted")

    report = status_phase(
        phase_run_id,
        authority_root=authority_root,
        runner=RecordingRunner(_empty_scheduler_results()),
    )

    assert report.status == "submitted"
    assert report.scheduler_status == "complete"
    assert report.actions[0].scheduler.status == "missing"


def test_finalized_authority_remains_accepted_when_accounting_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_phase_finalization import _accepted_inputs, _finalize

    inputs = _accepted_inputs(tmp_path, monkeypatch)
    finalized = _finalize(inputs)
    authority_root = inputs["authority_root"]
    assert isinstance(authority_root, Path)

    report = status_phase(
        finalized.phase_run_id,
        authority_root=authority_root,
        runner=RecordingRunner(_empty_scheduler_results()),
    )

    assert report.status == "accepted"
    assert report.sealed is True
    assert report.phase_receipt_id == finalized.phase_receipt_id
    assert report.submission_status == "submitted"
    assert report.actions[0].durable_status == "submitted"
    assert report.actions[0].scheduler.status == "missing"


def test_one_unavailable_and_one_empty_source_report_unknown_degraded_observation(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission(authority_root, phase_run_id, outcome="submitted")
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=1, stdout="", stderr="json unavailable"),
            CommandResult(argv=(), returncode=2, stdout="", stderr="queue unavailable"),
            CommandResult(argv=(), returncode=0, stdout='{"jobs": []}', stderr=""),
        ]
    )

    report = status_phase(phase_run_id, authority_root=authority_root, runner=runner)

    assert report.status == "submitted"
    assert report.scheduler_status == "degraded"
    assert report.actions[0].scheduler.status == "unknown"
    assert report.scheduler_sources[0].availability == "unavailable"
    assert report.scheduler_sources[1].availability == "available"
    assert len(report.warnings) == 1


def test_sacct_failure_retains_live_squeue_state_in_degraded_report(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission(authority_root, phase_run_id, outcome="submitted")
    runner = RecordingRunner(
        [
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps({"jobs": [{"job_id": 123456, "job_state": "RUNNING"}]}),
                stderr="",
            ),
            CommandResult(argv=(), returncode=1, stdout="", stderr="accounting JSON unavailable"),
            CommandResult(argv=(), returncode=2, stdout="", stderr="accounting unavailable"),
        ]
    )

    report = status_phase(phase_run_id, authority_root=authority_root, runner=runner)

    assert report.scheduler_status == "degraded"
    assert report.actions[0].scheduler == PhaseActionSchedulerStatus(
        "observed",
        state="RUNNING",
        source="squeue",
    )
    assert [call[0] for call in runner.calls] == ["squeue", "sacct", "sacct"]


def test_both_unavailable_sources_report_unknown_without_raising(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission(authority_root, phase_run_id, outcome="submitted")
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=1, stdout="", stderr="squeue json unavailable"),
            CommandResult(argv=(), returncode=2, stdout="", stderr="squeue unavailable"),
            CommandResult(argv=(), returncode=1, stdout="", stderr="sacct json unavailable"),
            CommandResult(argv=(), returncode=2, stdout="", stderr="sacct unavailable"),
        ]
    )

    report = status_phase(phase_run_id, authority_root=authority_root, runner=runner)

    assert report.scheduler_status == "unavailable"
    assert report.actions[0].scheduler.status == "unknown"
    assert [source.availability for source in report.scheduler_sources] == ["unavailable", "unavailable"]
    assert len(report.warnings) == 2


def test_status_never_uses_lifecycle_locks_or_append_apis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission(authority_root, phase_run_id, outcome="submitted")
    before = _authority_snapshot(authority_root)

    def reject_mutation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Phase Status must not call a lifecycle mutation API")

    monkeypatch.setattr(PhaseAuthorityStore, "phase_operation_lock", reject_mutation)
    monkeypatch.setattr(PhaseAuthorityStore, "append_event", reject_mutation)
    runner = RecordingRunner(_empty_scheduler_results())

    status_phase(phase_run_id, authority_root=authority_root, runner=runner)

    assert [call[0] for call in runner.calls] == ["squeue", "sacct"]
    assert _authority_snapshot(authority_root) == before


def test_invalid_id_and_malformed_authority_fail_before_scheduler_without_repair(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)

    def reject_scheduler(_argv: tuple[str, ...]) -> CommandResult:
        raise AssertionError("invalid authority must fail before scheduler contact")

    before = _authority_snapshot(authority_root)
    with pytest.raises(ValueError, match="valid Phase Run id"):
        status_phase("../phase-run-00000000000000000000000000000000", authority_root=authority_root)
    assert _authority_snapshot(authority_root) == before

    event_path = authority_root / phase_run_id / "events/000001-phase-materialized.json"
    event_path.write_bytes(b"not-json\n")
    malformed = _authority_snapshot(authority_root)
    with pytest.raises(ValueError, match="invalid JSON"):
        status_phase(phase_run_id, authority_root=authority_root, runner=reject_scheduler)
    assert _authority_snapshot(authority_root) == malformed


def test_pure_action_projection_preserves_synthetic_declaration_and_dependency_order(tmp_path: Path) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work")
    first = replace(fixture.action, action_id="preprocessing-chunk-000001")
    second = replace(
        fixture.action,
        action_id="preprocessing-chunk-000002",
        dependencies=(first.action_id,),
    )

    statuses = _build_action_statuses(
        (second, first),
        submission=None,
        observation=None,
        scheduler_status="not-requested",
    )

    assert [(status.action_id, status.dependency_action_ids) for status in statuses] == [
        (second.action_id, (first.action_id,)),
        (first.action_id, ()),
    ]
    duplicate_jobs = (
        PhaseActionStatus("a", (), "submitted", "7", PhaseActionSchedulerStatus("missing")),
        PhaseActionStatus("b", (), "submitted", "7", PhaseActionSchedulerStatus("missing")),
        PhaseActionStatus("c", (), "submitted", "8", PhaseActionSchedulerStatus("missing")),
    )
    assert _ordered_unique_jobs(duplicate_jobs) == ("7", "8")


def test_build_action_statuses_prefers_array_terminal_on_cross_family_overlap(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    _record_submission(authority_root, phase_run_id, outcome="submitted")
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert authority.submission is not None
    action = authority.phase_runspec.payload.actions[0]

    scalar = PhaseActionTerminalObservationView(
        action_id=action.action_id,
        job_id="123456",
        state="CANCELLED",
        exit_code="0:15",
        outcome="failed",
        source="sacct",
        observed_at="2026-08-20T12:00:00Z",
    )
    array = FoldingActionTerminalObservationView(
        action_id=action.action_id,
        parent_job_id="123456",
        expected_task_indexes=(0,),
        tasks=(
            FoldingTaskTerminalEvidence(
                task_index=0,
                scheduler_job_id="123456_0",
                state="COMPLETED",
                exit_code="0:0",
                source="sacct",
            ),
        ),
        outcome="succeeded",
        observed_at="2026-08-20T12:00:00Z",
    )

    selected = _select_terminal_observations_by_action((scalar,), (array,))
    assert selected == {action.action_id: array}

    statuses = _build_action_statuses(
        (action,),
        submission=authority.submission,
        terminal_observations=(scalar,),
        array_terminal_observations=(array,),
        observation=None,
        scheduler_status="not-requested",
    )

    assert len(statuses) == 1
    assert isinstance(statuses[0].terminal, FoldingActionTerminalObservationView)
    assert statuses[0].terminal.parent_job_id == "123456"
    assert statuses[0].terminal.expected_task_indexes == (0,)
    assert statuses[0].terminal.outcome == "succeeded"


def test_scalar_terminal_status_mapping_remains_compatible() -> None:
    scalar = PhaseActionTerminalObservationView(
        action_id="preprocessing-chunk-000000",
        job_id="7",
        state="COMPLETED",
        exit_code="0:0",
        outcome="succeeded",
        source="sacct",
        observed_at="2026-08-20T12:00:00Z",
    )
    status = PhaseActionStatus(
        "preprocessing-chunk-000000",
        (),
        "submitted",
        "7",
        PhaseActionSchedulerStatus("missing"),
        terminal=scalar,
    )
    mapping = status.to_mapping()
    assert mapping["job_id"] == "7"
    assert mapping["terminal"]["job_id"] == "7"
    assert mapping["terminal"]["outcome"] == "succeeded"
    assert "parent_job_id" not in mapping["terminal"]


def test_status_report_records_reject_invalid_runtime_discriminators_and_relations(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    report = status_phase(phase_run_id, authority_root=authority_root)
    invalid = cast(Any, "invalid")

    with pytest.raises(ValueError, match="source availability"):
        replace(report.scheduler_sources[0], availability=invalid)
    with pytest.raises(ValueError, match="scheduler observation status"):
        replace(report.actions[0].scheduler, status=invalid)
    with pytest.raises(ValueError, match="durable Phase Action status"):
        replace(report.actions[0], durable_status=invalid)
    with pytest.raises(ValueError, match="Phase summary status"):
        replace(report, status=invalid)
    with pytest.raises(ValueError, match="scheduler summary"):
        replace(report, scheduler_status=invalid)
    with pytest.raises(ValueError, match="preserve its durable submission status"):
        replace(report, status="submitted")

    submitted_root, submitted_id = _materialized_authority(tmp_path / "submitted")
    _record_submission(submitted_root, submitted_id, outcome="submitted")
    submitted = status_phase(
        submitted_id,
        authority_root=submitted_root,
        runner=RecordingRunner(_empty_scheduler_results()),
    )
    with pytest.raises(ValueError, match="preserve its durable submission status"):
        replace(submitted, submission_status="failed")


def test_phase_status_cli_emits_table_text_and_stable_json_without_changing_legacy_help(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    cli_runner = CliRunner()
    legacy_help = cli_runner.invoke(cli, ["run", "--help"]).output

    table = cli_runner.invoke(
        cli,
        ["phase", "status", phase_run_id, "--authority-root", str(authority_root)],
    )
    text = cli_runner.invoke(
        cli,
        ["phase", "status", phase_run_id, "--authority-root", str(authority_root), "--format", "text"],
    )
    structured = cli_runner.invoke(
        cli,
        ["phase", "status", phase_run_id, "--authority-root", str(authority_root), "--format", "json"],
    )

    assert table.exit_code == 0, table.output
    assert table.output == text.output
    assert "phase_status:" in table.output
    payload = json.loads(structured.output)
    assert payload["phase_run_id"] == phase_run_id
    assert payload["current_attempt"] == {"attempt_id": "attempt-0001", "status": "materialized"}
    assert payload["actions"][0]["action_id"] == "preprocessing-chunk-000000"
    assert cli_runner.invoke(cli, ["run", "--help"]).output == legacy_help


def test_phase_status_cli_requires_an_existing_authority_root(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "phase",
            "status",
            "phase-run-00000000000000000000000000000000",
            "--authority-root",
            str(tmp_path / "missing"),
        ],
    )

    assert result.exit_code != 0
    assert "Directory" in result.output
    assert "does not exist" in result.output
