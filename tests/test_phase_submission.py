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

"""Direct-Slurm Phase Submission rendering, coordination, and CLI tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from bspp.orchestration.contract.database_placement import DatabaseSetSelection
from bspp.orchestration.contract.phase import (
    PhaseMountSnapshot,
    PhasePlan,
    PreprocessingPhasePlanPayload,
)
from bspp.orchestration.contract.phase_reconciliation import (
    PhaseActionTerminalObservedEvent,
    PhaseActionTerminalObservedPayload,
)
from bspp.orchestration.contract.phase_state import (
    PhaseAttempt,
    PhaseMaterializedEvent,
    PhaseMaterializedPayload,
    PhaseRun,
)
from bspp.orchestration.contract.phase_submission import (
    PhaseActionDispatchIntendedEvent,
    PhaseActionDispatchRejectedEvent,
    PhaseActionSubmittedEvent,
    PhaseSubmissionIntendedEvent,
)
from bspp.orchestration.contract.preprocessing_execution import (
    preprocessing_chunk_execution_intent_from_plan,
)
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore
from bspp.orchestration.control.phase_rendering import render_phase_submission_intent
from bspp.orchestration.control.phase_submission import (
    _require_current_qualification,
    _topological_actions,
    submit_phase,
)
from bspp.orchestration.control.transport import CommandResult, default_command_runner, legacy_command_argv
from tests.support.preprocessing_execution import preprocessing_execution_fixture
from tests.support.transport_argv import maybe_unwrap_remote_command

SUBMITTED_AT = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def _materialized_authority(tmp_path: Path, *, transport: str = "local-slurm") -> tuple[Path, str]:
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
    cluster = fixture.runspec.cluster
    if transport == "ssh":
        cluster = replace(cluster, transport="ssh", ssh_target="example-cluster-login")
    runspec = replace(fixture.runspec, phase_plan_digest=plan.digest, cluster=cluster)
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


class LocalSubmissionRunner:
    """Run local staging commands and synthesize only the Slurm assignment."""

    def __init__(self, authority_root: Path, phase_run_id: str, *, sbatch_result: CommandResult) -> None:
        self.authority_root = authority_root
        self.phase_run_id = phase_run_id
        self.sbatch_result = sbatch_result
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        intent = self.authority_root / self.phase_run_id / "events/000002-phase-submission-intended.json"
        assert intent.is_file(), "the complete submission intent must precede every remote effect"
        if argv[0] == "sbatch":
            return replace(self.sbatch_result, argv=argv)
        return default_command_runner(argv)


def test_renderer_freezes_qualified_direct_slurm_action_and_rejects_unsupported_shape(
    tmp_path: Path,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    attempt = authority.phase_run.attempts[0]
    runspec_path = authority.authority_path / attempt.phase_runspec_location
    document_hash = hashlib.sha256(runspec_path.read_bytes()).hexdigest()

    intent = render_phase_submission_intent(
        authority.phase_runspec,
        phase_runspec_location=attempt.phase_runspec_location,
        phase_runspec_document_sha256=document_hash,
    )

    plan = intent.actions[0]
    script = plan.script_body
    qualification = authority.phase_runspec.cluster.preprocessing_runtime.qualification_tuple
    assert plan.job_name == plan.scheduler_correlation_token
    assert f"#SBATCH --job-name={plan.scheduler_correlation_token}" in script
    assert f"#SBATCH --comment={plan.scheduler_correlation_token}" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=32G" in script
    assert "#SBATCH --time=01:00:00" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert qualification.cluster_image_sha256 in script
    assert qualification.source_bundle_sha256 in script
    assert document_hash in script
    assert script.count("  execute-chunk \\") == 2
    assert script.count("--placement-process-status \\") == 2
    assert script.count('"$_placement_status"') == 3
    assert script.count("  finalize-chunk \\") == 1
    assert "\n+  " not in script
    assert script.count("/usr/local/bin/entrypoint.sh") == 5
    assert "stage-input" in script
    assert script.count("  stage-input \\") == 1
    assert "nextflow" not in script.lower()
    assert "workflow-engine" not in script.lower()
    assert "PYTHONPATH" not in script


def test_renderer_derives_gres_for_packed_topology_action(tmp_path: Path) -> None:
    """A packed-topology action (gres=None, nodes+gpus_per_task) renders the
    derived --gres and passes the parity check against a derived qualification."""
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    attempt = authority.phase_run.attempts[0]
    runspec_path = authority.authority_path / attempt.phase_runspec_location
    document_hash = hashlib.sha256(runspec_path.read_bytes()).hexdigest()

    action = authority.phase_runspec.payload.actions[0]
    qualification = authority.phase_runspec.cluster.preprocessing_runtime.qualification_tuple
    assert qualification.gpu_worker_gres == "gpu:1"

    packed_action = replace(
        action,
        resources=replace(
            action.resources,
            gres=None,
            nodes=2,
            tasks_per_node=5,
            gpus_per_task=1,
        ),
    )
    packed_runspec = replace(
        authority.phase_runspec,
        payload=replace(authority.phase_runspec.payload, actions=(packed_action,)),
    )
    intent = render_phase_submission_intent(
        packed_runspec,
        phase_runspec_location=attempt.phase_runspec_location,
        phase_runspec_document_sha256=document_hash,
    )
    packed_script = intent.actions[0].script_body
    assert "#SBATCH --gres=gpu:1" in packed_script
    assert qualification.cluster_image_sha256 in packed_script


def test_renderer_rejects_packed_action_with_mismatched_derived_gres(tmp_path: Path) -> None:
    """A packed action with gpus_per_task=2 derives gpu:2, which must mismatch a
    qualification that derived gpu:1."""
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    attempt = authority.phase_run.attempts[0]
    runspec_path = authority.authority_path / attempt.phase_runspec_location
    document_hash = hashlib.sha256(runspec_path.read_bytes()).hexdigest()

    action = authority.phase_runspec.payload.actions[0]
    mismatched_action = replace(
        action,
        resources=replace(
            action.resources,
            gres=None,
            nodes=2,
            tasks_per_node=5,
            gpus_per_task=2,
        ),
    )
    mismatched_runspec = replace(
        authority.phase_runspec,
        payload=replace(authority.phase_runspec.payload, actions=(mismatched_action,)),
    )
    with pytest.raises(ValueError, match="GRES does not match"):
        render_phase_submission_intent(
            mismatched_runspec,
            phase_runspec_location=attempt.phase_runspec_location,
            phase_runspec_document_sha256=document_hash,
        )

    action = authority.phase_runspec.payload.actions[0]
    array_runspec = replace(
        authority.phase_runspec,
        payload=replace(
            authority.phase_runspec.payload,
            actions=(replace(action, resources=replace(action.resources, array="0-3")),),
        ),
    )
    with pytest.raises(ValueError, match="does not support Slurm arrays"):
        render_phase_submission_intent(
            array_runspec,
            phase_runspec_location=attempt.phase_runspec_location,
            phase_runspec_document_sha256=document_hash,
        )

    mismatched_node_runspec = replace(
        authority.phase_runspec,
        payload=replace(
            authority.phase_runspec.payload,
            actions=(replace(action, resources=replace(action.resources, nodelist="gpu-node-018")),),
        ),
    )
    with pytest.raises(ValueError, match="nodelist does not match qualified"):
        render_phase_submission_intent(
            mismatched_node_runspec,
            phase_runspec_location=attempt.phase_runspec_location,
            phase_runspec_document_sha256=document_hash,
        )

    protected_database_root = authority.phase_runspec.payload.database.source_manifest.source_root
    conflict = PhaseMountSnapshot(source="/other-databases", target=protected_database_root)
    conflict_runspec = replace(
        authority.phase_runspec,
        cluster=replace(authority.phase_runspec.cluster, extra_mounts=(conflict,)),
    )
    with pytest.raises(ValueError, match="overlaps protected database namespace"):
        render_phase_submission_intent(
            conflict_runspec,
            phase_runspec_location=attempt.phase_runspec_location,
            phase_runspec_document_sha256=document_hash,
        )


def test_submit_phase_stages_exact_runspec_records_assignment_and_is_idempotent(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    runner = LocalSubmissionRunner(
        authority_root,
        phase_run_id,
        sbatch_result=CommandResult(argv=(), returncode=0, stdout="4242;local\n", stderr=""),
    )

    result = submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: SUBMITTED_AT,
        runner=runner,
    )

    assert result.status == "submitted"
    assert [(item.action_id, item.job_id) for item in result.actions] == [("preprocessing-chunk-000000", "4242")]
    sbatch_calls = [call for call in runner.calls if call[0] == "sbatch"]
    assert len(sbatch_calls) == 1
    assert not any(call[0] in {"squeue", "sacct"} for call in runner.calls)
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert [type(event) for event in authority.events[1:]] == [
        PhaseSubmissionIntendedEvent,
        PhaseActionDispatchIntendedEvent,
        PhaseActionSubmittedEvent,
    ]
    assert authority.submission is not None
    view = authority.submission.actions[0]
    assert view.job_id == "4242"
    assert view.sbatch_argv == sbatch_calls[0]
    staged_runspec = (
        Path(authority.phase_runspec.cluster.staging_root)
        / "bspp-phase-runs"
        / phase_run_id
        / authority.phase_runspec.attempt_id
        / "phase-runspec.json"
    )
    stored_runspec = authority.authority_path / authority.phase_run.attempts[0].phase_runspec_location
    assert staged_runspec.read_bytes() == stored_runspec.read_bytes()

    def reject_remote(_argv: tuple[str, ...]) -> CommandResult:
        raise AssertionError("a fully submitted retry must make no remote call")

    assert (
        submit_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=lambda: datetime(2027, 1, 1, tzinfo=UTC),
            runner=reject_remote,
        )
        == result
    )


def test_current_qualification_accepts_contract_validated_canonical_z_expiry(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)

    assert authority.phase_runspec.cluster.preprocessing_runtime.expires_at == "2026-08-26T10:00:00.000000Z"
    _require_current_qualification(authority, now=SUBMITTED_AT)


def test_authority_validation_rejects_noncanonical_qualification_expiry_before_submission(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    runspec_path = authority.authority_path / authority.current_attempt.phase_runspec_location
    runspec_mapping = json.loads(runspec_path.read_text())
    runspec_mapping["cluster"]["preprocessing_runtime"]["expires_at"] = "2026-08-26T10:00:00.000000+00:00"
    runspec_path.write_text(json.dumps(runspec_mapping, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="expires_at must be an RFC 3339 UTC timestamp ending in Z"):
        submit_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=lambda: SUBMITTED_AT,
            runner=lambda _argv: (_ for _ in ()).throw(AssertionError("unexpected remote call")),
        )


def test_authority_replay_rejects_nonexact_recorded_sbatch_argv(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    runner = LocalSubmissionRunner(
        authority_root,
        phase_run_id,
        sbatch_result=CommandResult(argv=(), returncode=0, stdout="4242\n", stderr=""),
    )
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: SUBMITTED_AT,
        runner=runner,
    )
    event_path = authority_root / phase_run_id / "events/000004-phase-action-submitted.json"
    event = json.loads(event_path.read_text())
    event["payload"]["sbatch_argv"] = ["sbatch", "--parsable", "/wrong/action.sbatch"]
    event_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="exact invoked sbatch argv"):
        PhaseAuthorityStore(authority_root).validate(phase_run_id)


_POST_WIRE_CUTOVER_AT = "2026-09-19T00:00:00Z"


def _ssh_submitted_authority(tmp_path: Path) -> tuple[Path, str]:
    """Materialize and submit an ssh-transport authority through a recording fake runner."""
    authority_root, phase_run_id = _materialized_authority(tmp_path, transport="ssh")
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    runspec_path = authority.authority_path / authority.phase_run.attempts[0].phase_runspec_location
    document_hash = hashlib.sha256(runspec_path.read_bytes()).hexdigest()
    intent = render_phase_submission_intent(
        authority.phase_runspec,
        phase_runspec_location=authority.phase_run.attempts[0].phase_runspec_location,
        phase_runspec_document_sha256=document_hash,
    )
    hash_results = iter(
        (
            document_hash,
            authority.phase_runspec.payload.database.source_manifest_sha256,
            intent.actions[0].script_sha256,
        )
    )

    def ssh_runner(argv: tuple[str, ...]) -> CommandResult:
        remote_command = maybe_unwrap_remote_command(argv[-1])
        if "sha256sum" in remote_command and "awk" not in remote_command:
            return CommandResult(argv=argv, returncode=0, stdout=f"{next(hash_results)}  staged\n", stderr="")
        if "sbatch --parsable" in remote_command:
            return CommandResult(argv=argv, returncode=0, stdout="4242\n", stderr="")
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")

    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: SUBMITTED_AT,
        runner=ssh_runner,
    )
    return authority_root, phase_run_id


def _rewrite_submitted_event(
    authority_root: Path,
    phase_run_id: str,
    *,
    sbatch_argv: list[str],
    occurred_at: str | None = None,
) -> None:
    event_path = authority_root / phase_run_id / "events/000004-phase-action-submitted.json"
    event = json.loads(event_path.read_text())
    event["payload"]["sbatch_argv"] = sbatch_argv
    if occurred_at is not None:
        event["occurred_at"] = occurred_at
    event_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")


def _legacy_wire_shape(authority_root: Path, phase_run_id: str) -> list[str]:
    """The exact legacy wire derivation of the authority's recorded sbatch invocation."""
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert authority.submission is not None
    view = authority.submission.actions[0]
    cluster = authority.phase_runspec.cluster
    return list(
        legacy_command_argv(
            ("sbatch", "--parsable", view.plan.cluster_script_path),
            transport=cluster.transport,
            ssh_target=cluster.ssh_target,
        )
    )


def test_authority_replay_rejects_nonexact_recorded_sbatch_argv_ssh(tmp_path: Path) -> None:
    """An ssh-transport tampered sbatch_argv matching neither shape is rejected."""
    authority_root, phase_run_id = _ssh_submitted_authority(tmp_path)
    # Tampered value that matches neither the current base64 shape nor the legacy shape
    _rewrite_submitted_event(
        authority_root,
        phase_run_id,
        sbatch_argv=["ssh", "wrong-target", "bash -lc 'sbatch --parsable /wrong/action.sbatch'"],
    )

    with pytest.raises(ValueError, match="exact invoked sbatch argv"):
        PhaseAuthorityStore(authority_root).validate(phase_run_id)


def test_authority_replay_accepts_legacy_wire_shape_before_cutover(tmp_path: Path) -> None:
    """A legacy event (SUBMITTED_AT predates the cutover) may record the legacy wire shape."""
    authority_root, phase_run_id = _ssh_submitted_authority(tmp_path)
    _rewrite_submitted_event(
        authority_root,
        phase_run_id,
        sbatch_argv=_legacy_wire_shape(authority_root, phase_run_id),
    )

    PhaseAuthorityStore(authority_root).validate(phase_run_id)


def test_authority_replay_rejects_legacy_wire_shape_after_cutover(tmp_path: Path) -> None:
    """A post-cutover event recording the legacy wire shape is rejected even when byte-exact."""
    authority_root, phase_run_id = _ssh_submitted_authority(tmp_path)
    _rewrite_submitted_event(
        authority_root,
        phase_run_id,
        sbatch_argv=_legacy_wire_shape(authority_root, phase_run_id),
        occurred_at=_POST_WIRE_CUTOVER_AT,
    )

    with pytest.raises(ValueError, match="exact invoked sbatch argv"):
        PhaseAuthorityStore(authority_root).validate(phase_run_id)


def test_definitive_local_rejection_is_durable_and_never_retried(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    runner = LocalSubmissionRunner(
        authority_root,
        phase_run_id,
        sbatch_result=CommandResult(argv=(), returncode=1, stdout="", stderr="partition unavailable\n"),
    )

    with pytest.raises(ValueError, match="partition unavailable"):
        submit_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=lambda: SUBMITTED_AT,
            runner=runner,
        )

    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert authority.submission is not None
    assert authority.submission.status == "failed"
    assert authority.lifecycle.attempt_status == authority.lifecycle.run_status == "failed"
    assert isinstance(authority.events[-1], PhaseActionDispatchRejectedEvent)

    with pytest.raises(ValueError, match="rejected Runtime Action"):
        submit_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=lambda: SUBMITTED_AT,
            runner=lambda _argv: (_ for _ in ()).throw(AssertionError("unexpected remote call")),
        )


def test_submit_rejects_scheduler_terminal_failed_boundary_before_submitted_fast_return(
    tmp_path: Path,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: SUBMITTED_AT,
        runner=LocalSubmissionRunner(
            authority_root,
            phase_run_id,
            sbatch_result=CommandResult(argv=(), returncode=0, stdout="4242\n", stderr=""),
        ),
    )
    store = PhaseAuthorityStore(authority_root)
    authority = store.validate(phase_run_id)
    assert authority.submission is not None
    action = authority.submission.actions[0]
    assert action.job_id == "4242"
    terminal = PhaseActionTerminalObservedPayload(
        submission_id=authority.submission.submission_id,
        phase_runspec_digest=authority.phase_runspec.digest,
        action_id=action.action_id,
        runtime_action_digest=action.plan.runtime_action_digest,
        scheduler_correlation_token=action.scheduler_correlation_token,
        job_id=action.job_id,
        state="FAILED",
        exit_code="1:0",
        outcome="failed",
    )
    store.append_event(
        phase_run_id,
        lambda sequence: PhaseActionTerminalObservedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at="2026-08-20T13:00:00.000000Z",
            payload=terminal,
        ),
    )

    with pytest.raises(ValueError, match="terminal accounting failure"):
        submit_phase(
            phase_run_id,
            authority_root=authority_root,
            runner=lambda _argv: (_ for _ in ()).throw(AssertionError("unexpected remote call")),
        )


def test_assignment_append_interruption_recovers_exact_job_without_second_sbatch(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)

    def interrupt_assignment(staged_event: Path) -> None:
        if json.loads(staged_event.read_text())["event_type"] == "phase-action-submitted":
            raise OSError("simulated assignment append interruption")

    interrupted_store = PhaseAuthorityStore(authority_root, before_event_publish=interrupt_assignment)
    first_runner = LocalSubmissionRunner(
        authority_root,
        phase_run_id,
        sbatch_result=CommandResult(argv=(), returncode=0, stdout="5150\n", stderr=""),
    )
    with pytest.raises(OSError, match="assignment append interruption"):
        submit_phase(
            phase_run_id,
            authority_root=authority_root,
            authority_store=interrupted_store,
            clock=lambda: SUBMITTED_AT,
            runner=first_runner,
        )
    assert len([call for call in first_runner.calls if call[0] == "sbatch"]) == 1

    interrupted = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert interrupted.submission is not None
    assert interrupted.submission.actions[0].status == "dispatching"
    correlation = interrupted.submission.actions[0].scheduler_correlation_token
    recovery_calls: list[tuple[str, ...]] = []

    def recovery_runner(argv: tuple[str, ...]) -> CommandResult:
        recovery_calls.append(argv)
        if argv[0] == "squeue":
            return CommandResult(
                argv=argv,
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {
                                "job_id": 5150,
                                "name": correlation,
                                "comment": correlation,
                                "job_state": "RUNNING",
                            }
                        ]
                    }
                ),
                stderr="",
            )
        if argv[0] == "sacct":
            return CommandResult(argv=argv, returncode=0, stdout='{"jobs": []}', stderr="")
        raise AssertionError(f"unexpected recovery command: {argv}")

    recovered = submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: SUBMITTED_AT,
        runner=recovery_runner,
    )

    assert recovered.actions[0].job_id == "5150"
    assert not any(call[0] == "sbatch" for call in recovery_calls)
    assert [call[0] for call in recovery_calls] == ["squeue", "sacct"]
    assert "--starttime=2026-08-20T12:00:00" in recovery_calls[1]


@pytest.mark.parametrize("matched_job_ids", [(), ("8101", "8102")])
def test_uncertain_submission_retry_never_resubmits_without_one_exact_match(
    tmp_path: Path,
    matched_job_ids: tuple[str, ...],
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    uncertain_runner = LocalSubmissionRunner(
        authority_root,
        phase_run_id,
        sbatch_result=CommandResult(argv=(), returncode=0, stdout="accepted without id\n", stderr=""),
    )
    with pytest.raises(ValueError, match="could not parse sbatch job id"):
        submit_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=lambda: SUBMITTED_AT,
            runner=uncertain_runner,
        )
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert authority.submission is not None
    assert authority.submission.actions[0].status == "dispatching"
    correlation = authority.submission.actions[0].scheduler_correlation_token
    retry_calls: list[tuple[str, ...]] = []

    def retry_runner(argv: tuple[str, ...]) -> CommandResult:
        retry_calls.append(argv)
        if argv[0] == "squeue":
            jobs = [
                {
                    "job_id": job_id,
                    "name": correlation,
                    "comment": correlation,
                    "job_state": "PENDING",
                }
                for job_id in matched_job_ids
            ]
            return CommandResult(
                argv=argv,
                returncode=0,
                stdout=json.dumps({"jobs": jobs}),
                stderr="",
            )
        if argv[0] == "sacct":
            return CommandResult(argv=argv, returncode=0, stdout='{"jobs": []}', stderr="")
        raise AssertionError(f"uncertain retry must only query correlation: {argv}")

    message = "remains uncertain" if not matched_job_ids else "multiple Slurm jobs"
    with pytest.raises(ValueError, match=message):
        submit_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=lambda: SUBMITTED_AT,
            runner=retry_runner,
        )
    assert not any(call[0] == "sbatch" for call in retry_calls)
    still_dispatching = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert still_dispatching.submission is not None
    assert still_dispatching.submission.actions[0].status == "dispatching"


def test_runner_exception_after_dispatch_intent_remains_reconstructable(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)

    class ExceptionRunner(LocalSubmissionRunner):
        def __call__(self, argv: tuple[str, ...]) -> CommandResult:
            if argv[0] == "sbatch":
                self.calls.append(argv)
                raise OSError("simulated local exec interruption")
            return super().__call__(argv)

    runner = ExceptionRunner(
        authority_root,
        phase_run_id,
        sbatch_result=CommandResult(argv=(), returncode=0, stdout="unused", stderr=""),
    )
    with pytest.raises(OSError, match="local exec interruption"):
        submit_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=lambda: SUBMITTED_AT,
            runner=runner,
        )

    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert authority.submission is not None
    assert authority.submission.actions[0].status == "dispatching"
    assert not any(isinstance(event, PhaseActionSubmittedEvent) for event in authority.events)
    assert not any(isinstance(event, PhaseActionDispatchRejectedEvent) for event in authority.events)


def test_submit_fails_before_remote_effect_for_stale_qualification_or_intent_append(
    tmp_path: Path,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path / "stale")
    remote_calls: list[tuple[str, ...]] = []

    def record_remote(argv: tuple[str, ...]) -> CommandResult:
        remote_calls.append(argv)
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")

    with pytest.raises(ValueError, match="Qualification has expired"):
        submit_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=lambda: datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
            runner=record_remote,
        )
    assert remote_calls == []
    assert PhaseAuthorityStore(authority_root).validate(phase_run_id).submission is None

    interrupted_root, interrupted_id = _materialized_authority(tmp_path / "interrupted")

    def interrupt_intent(staged_event: Path) -> None:
        if json.loads(staged_event.read_text())["event_type"] == "phase-submission-intended":
            raise OSError("simulated intent append interruption")

    with pytest.raises(OSError, match="intent append interruption"):
        submit_phase(
            interrupted_id,
            authority_root=interrupted_root,
            authority_store=PhaseAuthorityStore(
                interrupted_root,
                before_event_publish=interrupt_intent,
            ),
            clock=lambda: SUBMITTED_AT,
            runner=record_remote,
        )
    assert remote_calls == []
    assert PhaseAuthorityStore(interrupted_root).validate(interrupted_id).submission is None


def test_missing_malformed_and_runspec_byte_drift_fail_without_remote_effect(tmp_path: Path) -> None:
    remote_calls: list[tuple[str, ...]] = []

    def record_remote(argv: tuple[str, ...]) -> CommandResult:
        remote_calls.append(argv)
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")

    missing_root = tmp_path / "missing-authority"
    with pytest.raises(ValueError, match="missing Phase Run authority"):
        submit_phase(
            "phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            authority_root=missing_root,
            clock=lambda: SUBMITTED_AT,
            runner=record_remote,
        )

    malformed_root, malformed_id = _materialized_authority(tmp_path / "malformed")
    malformed_run = malformed_root / malformed_id
    (malformed_run / "unexpected.json").write_text("{}\n")
    with pytest.raises(ValueError, match="layout mismatch"):
        submit_phase(
            malformed_id,
            authority_root=malformed_root,
            clock=lambda: SUBMITTED_AT,
            runner=record_remote,
        )

    drift_root, drift_id = _materialized_authority(tmp_path / "drift")
    drift_authority = PhaseAuthorityStore(drift_root).validate(drift_id)
    runspec_path = drift_authority.authority_path / drift_authority.phase_run.attempts[0].phase_runspec_location
    runspec_mapping = json.loads(runspec_path.read_text())
    runspec_path.write_text(json.dumps(runspec_mapping, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="RunSpec bytes differ"):
        submit_phase(
            drift_id,
            authority_root=drift_root,
            clock=lambda: SUBMITTED_AT,
            runner=record_remote,
        )

    assert remote_calls == []


def test_ssh_submission_records_exact_invoked_transport_argv(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path, transport="ssh")
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    runspec_path = authority.authority_path / authority.phase_run.attempts[0].phase_runspec_location
    document_hash = hashlib.sha256(runspec_path.read_bytes()).hexdigest()
    intent = render_phase_submission_intent(
        authority.phase_runspec,
        phase_runspec_location=authority.phase_run.attempts[0].phase_runspec_location,
        phase_runspec_document_sha256=document_hash,
    )
    hash_results = iter(
        (
            document_hash,
            authority.phase_runspec.payload.database.source_manifest_sha256,
            intent.actions[0].script_sha256,
        )
    )
    calls: list[tuple[str, ...]] = []

    def ssh_runner(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        remote_command = maybe_unwrap_remote_command(argv[-1])
        if "sha256sum" in remote_command and "awk" not in remote_command:
            return CommandResult(argv=argv, returncode=0, stdout=f"{next(hash_results)}  staged\n", stderr="")
        if "sbatch --parsable" in remote_command:
            return CommandResult(argv=argv, returncode=0, stdout="6060;example-cluster\n", stderr="")
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")

    result = submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: SUBMITTED_AT,
        runner=ssh_runner,
    )

    assert result.actions[0].job_id == "6060"
    sbatch_call = next(call for call in calls if "sbatch --parsable" in maybe_unwrap_remote_command(call[-1]))
    submitted = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert submitted.submission is not None
    assert submitted.submission.actions[0].sbatch_argv == sbatch_call
    assert sbatch_call[:2] == ("ssh", "example-cluster-login")


def test_topological_order_uses_declaration_order_as_tie_breaker(tmp_path: Path) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work")
    base = fixture.action
    parent_b = replace(base, action_id="preprocessing-chunk-000001")
    parent_a = replace(base, action_id="preprocessing-chunk-000002")
    child = replace(
        base,
        action_id="preprocessing-chunk-000003",
        dependencies=(parent_a.action_id, parent_b.action_id),
    )

    ordered = _topological_actions((child, parent_b, parent_a))

    assert tuple(action.action_id for action in ordered) == (
        parent_b.action_id,
        parent_a.action_id,
        child.action_id,
    )
    assert child.dependencies == (parent_a.action_id, parent_b.action_id)


def test_phase_submit_cli_emits_stable_json_and_preserves_legacy_run_help(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    runner = LocalSubmissionRunner(
        authority_root,
        phase_run_id,
        sbatch_result=CommandResult(argv=(), returncode=0, stdout="7171\n", stderr=""),
    )
    import bspp.orchestration.control.phase_submission as phase_submission

    real_submit = phase_submission.submit_phase
    monkeypatch.setattr(
        phase_submission,
        "submit_phase",
        lambda requested_id, *, authority_root: real_submit(
            requested_id,
            authority_root=authority_root,
            clock=lambda: SUBMITTED_AT,
            runner=runner,
        ),
    )
    cli_runner = CliRunner()
    legacy_help = cli_runner.invoke(cli, ["run", "--help"]).output

    result = cli_runner.invoke(
        cli,
        ["phase", "submit", phase_run_id, "--authority-root", str(authority_root)],
    )

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    payload = json.loads(result.output)
    assert payload["phase_run_id"] == phase_run_id
    assert payload["status"] == "submitted"
    assert payload["actions"] == [{"action_id": "preprocessing-chunk-000000", "job_id": "7171"}]
    assert cli_runner.invoke(cli, ["run", "--help"]).output == legacy_help
