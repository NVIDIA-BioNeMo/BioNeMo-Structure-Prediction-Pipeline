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

"""Bounded diagnostics policy for coordinator failure paths."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import bspp.orchestration.control.postprocessing_phase_diagnostics as diagnostics_module
from bspp.orchestration.contract.postprocessing_acceptance_adjudication import (
    PostprocessingAcceptanceAdjudication,
    PostprocessingReconciliationResult,
    PostprocessingResidualCardinality,
)
from bspp.orchestration.contract.postprocessing_acceptance_diagnostics import canonical_allowance_id
from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    PostprocessingAcceptancePolicySnapshot,
    PostprocessingBaselineReportBinding,
    PostprocessingCompletionExitContract,
    PostprocessingCrossReportReconciliation,
    PostprocessingRawExitReportOutcome,
    PostprocessingResidualAllowance,
)
from bspp.orchestration.contract.postprocessing_phase_ids import postprocessing_action_id
from bspp.orchestration.contract.postprocessing_terminal_events import (
    PostprocessingActionTerminalObservedPayload,
    PostprocessingTaskTerminalEvidence,
)
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.monitoring import (
    SlurmCommandSnapshot,
    SlurmJobRecord,
    SlurmJobState,
    SlurmObservation,
)
from bspp.orchestration.control.postprocessing_phase_diagnostics import (
    MAX_REMOTE_LOG_BYTES,
    capture_postprocessing_phase_diagnostics,
)
from bspp.orchestration.control.transport import CommandResult, RemoteSlurmTransport
from bspp.orchestration.runtime.postprocessing.phase_acceptance import capture_acceptance

PHASE_RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
ATTEMPT_ID = "attempt-0001"
SUBMISSION_ID = f"postprocessing-submission-{'1' * 64}"


class _Submission:
    def __init__(self, renderer: int | tuple[int, ...]) -> None:
        renderers = renderer if isinstance(renderer, tuple) else (renderer,)
        self.actions = tuple(SimpleNamespace(renderer_contract_version=value) for value in renderers)

    def job_ids_by_action(self) -> dict[str, str]:
        return {
            "postprocessing-01-preflight": "10",
            "postprocessing-02-recipe": "11",
            "postprocessing-04-slurm": "20",
            "postprocessing-05-analysis-finalize": "30",
        }


class _Transport:
    def __init__(self) -> None:
        self.read_calls: list[tuple[str, int]] = []

    def query_observation_best_effort(self, job_ids: tuple[str, ...], **kwargs: object) -> object:
        assert job_ids == ("10", "11", "20", "30")
        assert kwargs["require_exact_terminal_exit"] is True
        assert kwargs["expected_terminal_job_ids"]
        return _slurm_observation(
            job_ids,
            selected_states=tuple(
                SlurmJobState(job_id=job_id, state="FAILED", source="sacct", exit_code="1:0") for job_id in job_ids
            ),
            warnings=("scheduler warning https://user:secret@example.invalid/path?token=secret",),
        )

    def read_immutable_bytes_artifact_no_follow(self, path: str, *, max_bytes: int) -> bytes:
        self.read_calls.append((path, max_bytes))
        return (
            b"prefix\xff https://user:password@example.invalid/path"
            b"?X-Amz-Credential=access-key&X-Amz-Security-Token=session-secret"
            b"&X-Amz-Signature=signed&AWSAccessKeyId=legacy-access"
            b"&api_key=api-secret&client_secret=oauth-secret&benign=also-private\n"
        )


def _terminal(action_id: str, parent: str, *, task_indexes: tuple[int, ...], failed: bool) -> object:
    tasks = tuple(
        PostprocessingTaskTerminalEvidence(
            scheduler_job_id=parent if index is None else f"{parent}_{index}",
            task_index=index,
            state="FAILED" if failed else "COMPLETED",
            exit_code="1:0" if failed else "0:0",
            source="sacct",
        )
        for index in (task_indexes or (None,))
    )
    return PostprocessingActionTerminalObservedPayload(
        submission_id=SUBMISSION_ID,
        phase_runspec_digest="2" * 64,
        action_id=action_id,
        runtime_action_digest="3" * 64,
        parent_job_id=parent,
        expected_task_indexes=task_indexes,
        tasks=tasks,
        outcome="failed" if failed else "succeeded",
    )


def _slurm_observation(
    requested_job_ids: tuple[str, ...],
    *,
    sacct_jobs: tuple[SlurmJobRecord, ...] = (),
    selected_states: tuple[SlurmJobState, ...] = (),
    warnings: tuple[str, ...] = (),
) -> SlurmObservation:
    return SlurmObservation(
        requested_job_ids=requested_job_ids,
        squeue=SlurmCommandSnapshot(kind="squeue", argv=(), returncode=0, parser="fixture"),
        sacct=SlurmCommandSnapshot(kind="sacct", argv=(), returncode=0, parser="fixture"),
        squeue_jobs=(),
        sacct_jobs=sacct_jobs,
        selected_states=selected_states,
        warnings=warnings,
    )


def _authority(renderer: int | tuple[int, ...] = 3) -> object:
    actions = (
        SimpleNamespace(action_id="postprocessing-01-preflight", step_name="preflight", expected_task_indexes=()),
        SimpleNamespace(action_id="postprocessing-02-recipe", step_name="recipe", expected_task_indexes=()),
        SimpleNamespace(action_id="postprocessing-04-slurm", step_name="slurm", expected_task_indexes=(853,)),
        SimpleNamespace(
            action_id="postprocessing-05-analysis-finalize",
            step_name="analysis-finalize",
            expected_task_indexes=(),
        ),
    )
    return SimpleNamespace(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        status="failed",
        submission_state=_Submission(renderer),
        runspec=SimpleNamespace(
            payload=SimpleNamespace(
                actions=actions,
                attempt_paths=SimpleNamespace(evidence_dir="/remote/evidence"),
            )
        ),
        terminal_payloads=(
            _terminal("postprocessing-01-preflight", "10", task_indexes=(), failed=True),
            _terminal("postprocessing-02-recipe", "11", task_indexes=(), failed=False),
            _terminal("postprocessing-04-slurm", "20", task_indexes=(853,), failed=True),
            # analysis-finalize is assigned but has no durable terminal event.
        ),
    )


@pytest.mark.parametrize("renderer_contract", (3, 4, 5))
def test_v3_diagnostics_reads_only_exact_durable_failed_action_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    renderer_contract: int,
) -> None:
    authority = _authority(renderer=renderer_contract)
    transport = _Transport()
    monkeypatch.setattr(diagnostics_module, "require_postprocessing_v2_authority", lambda *_args: authority)
    monkeypatch.setattr(diagnostics_module, "postprocessing_transport", lambda *_args, **_kwargs: transport)

    result = capture_postprocessing_phase_diagnostics(
        PHASE_RUN_ID,
        authority_root=tmp_path / "authority",
        diagnostics_root=tmp_path / "diagnostics",
        clock=lambda: datetime(2026, 9, 5, tzinfo=UTC),
    )

    assert [path for path, _maximum in transport.read_calls] == [
        "/remote/evidence/slurm-logs/preflight.10.out",
        "/remote/evidence/slurm-logs/preflight.10.err",
        "/remote/evidence/slurm-logs/slurm.20_853.out",
        "/remote/evidence/slurm-logs/slurm.20_853.err",
    ]
    assert {maximum for _path, maximum in transport.read_calls} == {MAX_REMOTE_LOG_BYTES}
    summary = json.loads(result.output.read_bytes())
    rendered = json.dumps(summary)
    assert "recipe.11" not in rendered
    assert "analysis-finalize.30" not in rendered
    assert "access-key" not in rendered
    assert "session-secret" not in rendered
    assert "signed" not in rendered
    assert "legacy-access" not in rendered
    assert "api-secret" not in rendered
    assert "oauth-secret" not in rendered
    assert "also-private" not in rendered
    assert "user:password" not in rendered
    assert "X-Amz-Security-Token=<redacted>" in rendered
    assert all("\ufffd" in row["tail"] for row in summary["logs"])
    assert result.captured_log_files == 4


@pytest.mark.parametrize(
    ("renderer_contract", "error"),
    ((6, "unsupported renderer contract version"), ((3, 5), "mixed renderer contract versions")),
)
def test_diagnostics_rejects_unknown_or_mixed_renderer_contracts_before_reading_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    renderer_contract: int | tuple[int, ...],
    error: str,
) -> None:
    authority = _authority(renderer=renderer_contract)
    transport = _Transport()
    monkeypatch.setattr(diagnostics_module, "require_postprocessing_v2_authority", lambda *_args: authority)
    monkeypatch.setattr(diagnostics_module, "postprocessing_transport", lambda *_args, **_kwargs: transport)

    with pytest.raises(ValueError, match=error):
        capture_postprocessing_phase_diagnostics(
            PHASE_RUN_ID,
            authority_root=tmp_path / "authority",
            diagnostics_root=tmp_path / "diagnostics",
        )
    assert transport.read_calls == []


def test_v3_diagnostics_preserves_invalid_bytes_through_production_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    calls: list[tuple[str, ...]] = []
    log_document = b"bad-utf8:\xff https://example.invalid/log?X-Amz-Security-Token=session&key=value\n"

    def runner(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        if argv[0] == "squeue":
            stdout = '{"jobs": []}'
        elif argv[0] == "sacct":
            stdout = json.dumps(
                {
                    "jobs": [
                        {
                            "job_id_raw": job_id,
                            "state": "CANCELLED" if job_id == "30" else "FAILED",
                            "exit_code": "0:0" if job_id == "30" else "1:0",
                        }
                        for job_id in ("10", "11", "20", "30")
                    ]
                }
            )
        elif argv[0] == "python3":
            stdout = base64.b64encode(log_document).decode("ascii")
        else:
            raise AssertionError(f"unexpected diagnostics transport command: {argv}")
        return CommandResult(argv=argv, returncode=0, stdout=stdout, stderr="")

    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)
    monkeypatch.setattr(diagnostics_module, "require_postprocessing_v2_authority", lambda *_args: authority)
    monkeypatch.setattr(diagnostics_module, "postprocessing_transport", lambda *_args, **_kwargs: transport)

    result = capture_postprocessing_phase_diagnostics(
        PHASE_RUN_ID,
        authority_root=tmp_path / "authority",
        diagnostics_root=tmp_path / "diagnostics",
    )

    summary = json.loads(result.output.read_bytes())
    assert result.captured_log_files == 4
    assert all("\ufffd" in row["tail"] for row in summary["logs"])
    rendered = json.dumps(summary)
    assert "session" not in rendered and "key=value" not in rendered
    assert "X-Amz-Security-Token=<redacted>" in rendered
    reads = [argv for argv in calls if argv[0] == "python3"]
    assert len(reads) == 4
    assert all(argv[-1] == str(MAX_REMOTE_LOG_BYTES) for argv in reads)


def test_v2_diagnostics_are_scheduler_only_even_for_durable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(renderer=2)
    transport = _Transport()
    monkeypatch.setattr(diagnostics_module, "require_postprocessing_v2_authority", lambda *_args: authority)
    monkeypatch.setattr(diagnostics_module, "postprocessing_transport", lambda *_args, **_kwargs: transport)

    result = capture_postprocessing_phase_diagnostics(
        PHASE_RUN_ID,
        authority_root=tmp_path / "authority",
        diagnostics_root=tmp_path / "diagnostics",
    )

    assert transport.read_calls == []
    summary = json.loads(result.output.read_bytes())
    assert summary["logs"] == []
    assert summary["log_policy"] == "scheduler-only-for-renderer-contracts-1-and-2"
    assert summary["acceptance_adjudication"] == {"status": "not-applicable"}
    assert summary["acceptance_support"] == {"status": "not-applicable", "files": [], "omitted_file_count": 0}


def test_diagnostics_selects_fresh_root_failure_and_excludes_dependency_cancellations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_names = (
        "preflight",
        "recipe",
        "preprocess",
        "slurm",
        "analysis-finalize",
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
        "acceptance-adjudication",
    )
    actions = tuple(
        SimpleNamespace(
            action_id=f"postprocessing-{index:02d}-{name}",
            step_name=name,
            expected_task_indexes=(853,) if index == 4 else (),
        )
        for index, name in enumerate(action_names, start=1)
    )
    assigned = {action.action_id: str(13210720 + index) for index, action in enumerate(actions, start=1)}
    assigned[actions[3].action_id] = "13210749"

    class Submission:
        actions = (SimpleNamespace(renderer_contract_version=3),)

        def job_ids_by_action(self) -> dict[str, str]:
            return assigned

    terminals = tuple(
        _terminal(
            action.action_id,
            assigned[action.action_id],
            task_indexes=(),
            failed=False,
        )
        for action in actions[:3]
    ) + tuple(
        PostprocessingActionTerminalObservedPayload(
            submission_id=SUBMISSION_ID,
            phase_runspec_digest="2" * 64,
            action_id=action.action_id,
            runtime_action_digest="3" * 64,
            parent_job_id=assigned[action.action_id],
            expected_task_indexes=(),
            tasks=(
                PostprocessingTaskTerminalEvidence(
                    scheduler_job_id=assigned[action.action_id],
                    task_index=None,
                    state="CANCELLED",
                    exit_code="0:0",
                    source="sacct",
                ),
            ),
            outcome="failed",
        )
        for action in actions[4:]
    )
    authority = SimpleNamespace(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        status="failed",
        submission_state=Submission(),
        runspec=SimpleNamespace(
            payload=SimpleNamespace(
                actions=actions,
                attempt_paths=SimpleNamespace(evidence_dir="/remote/evidence"),
            )
        ),
        terminal_payloads=terminals,
    )

    class Transport:
        def __init__(self) -> None:
            self.read_calls: list[str] = []

        def query_observation_best_effort(self, job_ids: tuple[str, ...], **kwargs: object) -> object:
            assert job_ids == tuple(assigned[action.action_id] for action in actions)
            assert kwargs["require_exact_terminal_exit"] is True
            assert kwargs["expected_terminal_job_ids"]
            states = tuple(
                SlurmJobState(
                    job_id=assigned[action.action_id],
                    state=(
                        "FAILED"
                        if action in (actions[0], actions[3])
                        else "COMPLETED"
                        if action in actions[1:3]
                        else "CANCELLED"
                    ),
                    source="sacct",
                    exit_code=("1:0" if action in (actions[0], actions[3]) else "0:0"),
                )
                for action in actions
            )
            records = tuple(
                SlurmJobRecord(
                    job_id=(f"{state.job_id}_853" if state.job_id == assigned[actions[3].action_id] else state.job_id),
                    requested_job_id=state.job_id,
                    state=state.state,
                    exit_code=state.exit_code,
                    source="sacct",
                )
                for state in states
            )
            return _slurm_observation(job_ids, sacct_jobs=records, selected_states=states)

        def read_immutable_bytes_artifact_no_follow(self, path: str, *, max_bytes: int) -> bytes:
            assert max_bytes == MAX_REMOTE_LOG_BYTES
            self.read_calls.append(path)
            return b"worker failure\n"

    transport = Transport()
    monkeypatch.setattr(diagnostics_module, "require_postprocessing_v2_authority", lambda *_args: authority)
    monkeypatch.setattr(diagnostics_module, "postprocessing_transport", lambda *_args, **_kwargs: transport)

    result = capture_postprocessing_phase_diagnostics(
        PHASE_RUN_ID,
        authority_root=tmp_path / "authority",
        diagnostics_root=tmp_path / "diagnostics",
    )

    assert transport.read_calls == [
        "/remote/evidence/slurm-logs/slurm.13210749_853.out",
        "/remote/evidence/slurm-logs/slurm.13210749_853.err",
    ]
    summary = json.loads(result.output.read_bytes())
    assert summary["schema_version"] == 3
    assert summary["diagnostic_kind"] == "postprocessing-phase-diagnostics-v3"
    assert summary["acceptance_adjudication"] == {"status": "not-selected"}
    assert summary["selected_failure_actions"] == [
        {
            "action_id": actions[3].action_id,
            "source": "fresh-scheduler",
            "scheduler_job_id": "13210749_853",
            "task_index": 853,
            "log_job_token": "13210749_853",
            "state": "FAILED",
            "exit_code": "1:0",
        }
    ]
    assert summary["missing_durable_terminal_action_ids"] == [actions[3].action_id]


def test_diagnostics_reads_only_the_exact_late_failing_array_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = SimpleNamespace(
        action_id="postprocessing-04-slurm",
        step_name="slurm",
        expected_task_indexes=tuple(range(10)),
    )
    parent_job_id = "700"
    records = tuple(
        SlurmJobRecord(
            job_id=f"{parent_job_id}_{task_index}",
            requested_job_id=parent_job_id,
            state="FAILED" if task_index == 9 else "COMPLETED",
            exit_code="9:0" if task_index == 9 else "0:0",
            source="sacct",
        )
        for task_index in action.expected_task_indexes
    )
    observation = _slurm_observation(
        (parent_job_id,),
        sacct_jobs=records,
        selected_states=(SlurmJobState(parent_job_id, "FAILED", "sacct", "9:0"),),
    )
    authority = SimpleNamespace(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        status="failed",
        submission_state=SimpleNamespace(
            actions=(SimpleNamespace(renderer_contract_version=4),),
            job_ids_by_action=lambda: {action.action_id: parent_job_id},
        ),
        runspec=SimpleNamespace(
            payload=SimpleNamespace(
                actions=(action,),
                attempt_paths=SimpleNamespace(evidence_dir="/remote/evidence"),
            )
        ),
        terminal_payloads=(),
    )

    class Transport:
        def __init__(self) -> None:
            self.read_calls: list[str] = []

        def query_observation_best_effort(self, *_args: object, **_kwargs: object) -> SlurmObservation:
            return observation

        def read_immutable_bytes_artifact_no_follow(self, path: str, *, max_bytes: int) -> bytes:
            assert max_bytes == MAX_REMOTE_LOG_BYTES
            self.read_calls.append(path)
            return b"late array task failure\n"

    transport = Transport()
    monkeypatch.setattr(diagnostics_module, "require_postprocessing_v2_authority", lambda *_args: authority)
    monkeypatch.setattr(diagnostics_module, "postprocessing_transport", lambda *_args, **_kwargs: transport)

    result = capture_postprocessing_phase_diagnostics(
        PHASE_RUN_ID,
        authority_root=tmp_path / "authority",
        diagnostics_root=tmp_path / "diagnostics",
    )

    assert transport.read_calls == [
        "/remote/evidence/slurm-logs/slurm.700_9.out",
        "/remote/evidence/slurm-logs/slurm.700_9.err",
    ]
    summary = json.loads(result.output.read_bytes())
    assert summary["omitted_log_file_count"] == 0
    assert summary["selected_failure_actions"] == [
        {
            "action_id": action.action_id,
            "source": "fresh-scheduler",
            "scheduler_job_id": "700_9",
            "task_index": 9,
            "log_job_token": "700_9",
            "state": "FAILED",
            "exit_code": "9:0",
        }
    ]


@pytest.mark.parametrize(
    "records",
    (
        (SlurmJobRecord("800_1", "sacct", "800", state="FAILED", exit_code="1:0"),),
        (
            SlurmJobRecord("800_1", "sacct", "800", state="FAILED", exit_code="1:0"),
            SlurmJobRecord("800_1", "sacct", "800", state="FAILED", exit_code="1:0"),
            SlurmJobRecord("800_2", "sacct", "800", state="COMPLETED", exit_code="0:0"),
        ),
        (
            SlurmJobRecord("800_1", "sacct", "800", state="FAILED", exit_code="1:0"),
            SlurmJobRecord("800_2", "sacct", "800", state="COMPLETED", exit_code="0:0"),
            SlurmJobRecord("800_[1-2]", "sacct", "800", state="FAILED", exit_code="1:0"),
        ),
        (
            SlurmJobRecord("800_1", "sacct", "800", state="FAILED", exit_code="unknown"),
            SlurmJobRecord("800_2", "sacct", "800", state="COMPLETED", exit_code="0:0"),
        ),
    ),
    ids=("missing-child", "ambiguous-child", "non-exact-child-id", "non-exact-exit"),
)
def test_diagnostics_ignores_unproven_fresh_array_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: tuple[SlurmJobRecord, ...],
) -> None:
    action = SimpleNamespace(
        action_id="postprocessing-04-slurm",
        step_name="slurm",
        expected_task_indexes=(1, 2),
    )
    authority = SimpleNamespace(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        status="failed",
        submission_state=SimpleNamespace(
            actions=(SimpleNamespace(renderer_contract_version=4),),
            job_ids_by_action=lambda: {action.action_id: "800"},
        ),
        runspec=SimpleNamespace(
            payload=SimpleNamespace(
                actions=(action,),
                attempt_paths=SimpleNamespace(evidence_dir="/remote/evidence"),
            )
        ),
        terminal_payloads=(),
    )
    observation = _slurm_observation(
        ("800",),
        sacct_jobs=records,
        selected_states=(SlurmJobState("800", "FAILED", "sacct", "1:0"),),
    )

    class Transport:
        def query_observation_best_effort(self, *_args: object, **_kwargs: object) -> SlurmObservation:
            return observation

        def read_immutable_bytes_artifact_no_follow(self, *_args: object, **_kwargs: object) -> bytes:
            raise AssertionError("unproven child evidence must not authorize a log read")

    monkeypatch.setattr(diagnostics_module, "require_postprocessing_v2_authority", lambda *_args: authority)
    monkeypatch.setattr(diagnostics_module, "postprocessing_transport", lambda *_args, **_kwargs: Transport())

    result = capture_postprocessing_phase_diagnostics(
        PHASE_RUN_ID,
        authority_root=tmp_path / "authority",
        diagnostics_root=tmp_path / "diagnostics",
    )

    summary = json.loads(result.output.read_bytes())
    assert summary["selected_failure_actions"] == []
    assert summary["logs"] == []


def test_public_diagnostics_command_is_a_read_only_thin_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    diagnostics_root = tmp_path / "diagnostics"
    captured: dict[str, object] = {}

    def capture(phase_run_id: str, **kwargs: object) -> object:
        captured["phase_run_id"] = phase_run_id
        captured.update(kwargs)
        return SimpleNamespace(render_json=lambda: '{"status":"captured"}\n')

    monkeypatch.setattr(diagnostics_module, "capture_postprocessing_phase_diagnostics", capture)
    result = CliRunner().invoke(
        cli,
        [
            "phase",
            "diagnostics",
            PHASE_RUN_ID,
            "--authority-root",
            str(authority_root),
            "--diagnostics-root",
            str(diagnostics_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"status": "captured"}
    assert captured == {
        "phase_run_id": PHASE_RUN_ID,
        "authority_root": authority_root,
        "diagnostics_root": diagnostics_root,
    }
    assert list(authority_root.iterdir()) == []


def test_diagnostics_rejects_authority_destination_alias_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(authority_root, target_is_directory=True)
    authority = _authority()
    monkeypatch.setattr(diagnostics_module, "require_postprocessing_v2_authority", lambda *_args: authority)
    monkeypatch.setattr(
        diagnostics_module,
        "postprocessing_transport",
        lambda *_args, **_kwargs: pytest.fail("transport must not be created"),
    )

    with pytest.raises(ValueError, match="outside Phase authority"):
        capture_postprocessing_phase_diagnostics(
            PHASE_RUN_ID,
            authority_root=authority_root,
            diagnostics_root=alias / "diagnostics",
        )

    assert list(authority_root.iterdir()) == []


class _Action09Transport:
    def __init__(self, documents: dict[str, bytes], *, unreadable: bool = False) -> None:
        self.documents = documents
        self.unreadable = unreadable
        self.read_calls: list[tuple[str, int]] = []

    def query_observation_best_effort(self, job_ids: tuple[str, ...], **kwargs: object) -> SlurmObservation:
        assert job_ids == ("90",)
        assert kwargs == {"require_exact_terminal_exit": True, "expected_terminal_job_ids": ("90",)}
        return _slurm_observation(
            job_ids,
            selected_states=(SlurmJobState("90", "FAILED", "sacct", "1:0"),),
        )

    def read_immutable_bytes_artifact_no_follow(self, path: str, *, max_bytes: int) -> bytes:
        self.read_calls.append((path, max_bytes))
        if self.unreadable and path.endswith("phase-acceptance/adjudication.json"):
            raise OSError("fixture transport failure")
        if path not in self.documents:
            raise FileNotFoundError(path)
        return self.documents[path]


def _action09_fixture(
    tmp_path: Path,
    *,
    result: str,
    non_allowlisted_errors: int,
    canonical_error_count: int = 1,
) -> tuple[object, dict[str, bytes]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    evidence = tmp_path / "evidence"
    reports = {
        "acceptance/tar_payload_parity/tar_payload_parity_report.json": {
            "baseline_dir": "/baseline",
            "candidate_dir": "/candidate",
            "ok": True,
            "inventory_errors": [],
            "baseline_only_tars": [],
            "candidate_only_tars": [],
            "payload_mismatch_count": 0,
            "error_count": 0,
            "files": [],
        },
        "acceptance/semantic_acceptance/semantic_acceptance_summary.json": {
            "baseline_dir": "/baseline",
            "candidate_dir": "/candidate",
            "ok": False,
            "errors": [
                f"https://user:secret-{index}@example.invalid/archive-{index}.tar?token=secret-{index}"
                for index in range(canonical_error_count)
            ],
        },
        "acceptance/verify_evidence/acceptance_evidence_report.json": {
            "schema_version": 1,
            "ok": False,
            "parity_report_path": "acceptance/tar_payload_parity/tar_payload_parity_report.json",
            "semantic_report_path": "acceptance/semantic_acceptance/semantic_acceptance_summary.json",
            "issues": [],
        },
    }
    report_paths = tuple(reports)
    step_reports = (
        ("acceptance-tar-payload-parity", report_paths[0], "tar-payload-parity-report"),
        ("acceptance-semantic", report_paths[1], "semantic-acceptance-summary"),
        ("acceptance-verify-evidence", report_paths[2], "acceptance-evidence-report"),
    )
    policy = PostprocessingAcceptancePolicySnapshot(
        baseline_id="fixture",
        baseline_version="v1",
        policy_schema="fixture",
        policy_version="1",
        residual_allowances=(
            PostprocessingResidualAllowance(
                report=report_paths[1],
                json_pointer="/missing-one",
                match_kind="json-pointer-equals",
                expected_value="ordinary-expected-one",
                cardinality_kind="exact",
                required_count=1,
                permitted_min=1,
                permitted_max=1,
            ),
            PostprocessingResidualAllowance(
                report=report_paths[2],
                json_pointer="/missing-two",
                match_kind="json-pointer-equals",
                expected_value={"filename": "ordinary-expected-two"},
                cardinality_kind="exact",
                required_count=1,
                permitted_min=1,
                permitted_max=1,
            ),
        ),
        completion_exit_contracts=tuple(
            PostprocessingCompletionExitContract(
                step_name=step,
                allowed_raw_exit_codes=(0, 1),
                report_paths=(path,),
                report_schema=schema,
                report_schema_version="1",
                outcome_report_path=path,
                outcome_json_pointer="/ok",
                raw_exit_report_outcomes=(
                    PostprocessingRawExitReportOutcome(raw_exit_code=0, report_ok=True),
                    PostprocessingRawExitReportOutcome(raw_exit_code=1, report_ok=False),
                ),
            )
            for step, path, schema in step_reports
        ),
        baseline_report_bindings=(
            PostprocessingBaselineReportBinding(report=report_paths[0], baseline_locator_json_pointer="/baseline_dir"),
            PostprocessingBaselineReportBinding(report=report_paths[1], baseline_locator_json_pointer="/baseline_dir"),
        ),
        cross_report_reconciliations=(
            PostprocessingCrossReportReconciliation(
                left_report=report_paths[0],
                left_json_pointer="/candidate_dir",
                right_report=report_paths[1],
                right_json_pointer="/candidate_dir",
            ),
        ),
    )
    policy_bytes = (json.dumps(policy.to_mapping(), sort_keys=True) + "\n").encode()
    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(policy_bytes)
    for path, payload in reports.items():
        location = evidence / path
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_text(json.dumps(payload, sort_keys=True))
    captures = []
    for step, _path, _schema in step_reports:
        raw = evidence / "phase-acceptance/raw"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / f"{step}.stdout").write_text("out\n")
        (raw / f"{step}.stderr").write_text("err\n")
        captures.append(
            capture_acceptance(
                policy_path=policy_path,
                expected_policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
                evidence_root=evidence,
                phase_run_id=PHASE_RUN_ID,
                attempt_id=ATTEMPT_ID,
                action_id=postprocessing_action_id(step),
                step_name=step,
                raw_exit_code=0 if step == "acceptance-tar-payload-parity" else 1,
                raw_stdout_path=raw / f"{step}.stdout",
                raw_stderr_path=raw / f"{step}.stderr",
                output_path=evidence / "phase-acceptance" / f"{step}-capture.json",
                completed_at="2026-09-05T00:00:00Z",
            )
        )
    residuals = (
        ()
        if result == "passed"
        else tuple(
            PostprocessingResidualCardinality(canonical_allowance_id(allowance), 0, 1, 1)
            for allowance in policy.residual_allowances
        )
    )
    adjudication = PostprocessingAcceptanceAdjudication(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        policy_id=policy.policy_id,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        capture_digests=tuple(item.digest for item in captures),
        non_allowlisted_errors=non_allowlisted_errors,
        residual_cardinalities=residuals,
        reconciliation_results=(PostprocessingReconciliationResult("fixture", True),),
        adjudicated_at="2026-09-05T00:00:00Z",
        result=result,
    )
    action_id = postprocessing_action_id("acceptance-adjudication")
    action = SimpleNamespace(action_id=action_id, step_name="acceptance-adjudication", expected_task_indexes=())
    authority = SimpleNamespace(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        status="failed",
        acceptance_policy=policy,
        acceptance_policy_bytes=policy_bytes,
        submission_state=SimpleNamespace(
            actions=(SimpleNamespace(renderer_contract_version=5),), job_ids_by_action=lambda: {action_id: "90"}
        ),
        runspec=SimpleNamespace(
            payload=SimpleNamespace(actions=(action,), attempt_paths=SimpleNamespace(evidence_dir="/remote/evidence"))
        ),
        terminal_payloads=(_terminal(action_id, "90", task_indexes=(), failed=True),),
    )
    documents = {
        "/remote/evidence/slurm-logs/acceptance-adjudication.90.out": b"action09 failed\n",
        "/remote/evidence/slurm-logs/acceptance-adjudication.90.err": b"action09 failed\n",
    }
    for local in evidence.rglob("*"):
        if local.is_file():
            documents["/remote/evidence/" + local.relative_to(evidence).as_posix()] = local.read_bytes()
    documents["/remote/evidence/phase-acceptance/adjudication.json"] = (
        json.dumps(adjudication.to_mapping(), sort_keys=True) + "\n"
    ).encode()
    return authority, documents


def _capture_action09(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: str = "failed",
    non_allowlisted_errors: int = 0,
    canonical_error_count: int = 1,
    documents_mutator: Callable[[dict[str, bytes]], None] | None = None,
    unreadable: bool = False,
) -> tuple[object, _Action09Transport, dict[str, object]]:
    authority, documents = _action09_fixture(
        tmp_path,
        result=result,
        non_allowlisted_errors=non_allowlisted_errors,
        canonical_error_count=canonical_error_count,
    )
    if documents_mutator is not None:
        documents_mutator(documents)
    transport = _Action09Transport(documents, unreadable=unreadable)
    monkeypatch.setattr(diagnostics_module, "require_postprocessing_v2_authority", lambda *_args: authority)
    monkeypatch.setattr(diagnostics_module, "postprocessing_transport", lambda *_args, **_kwargs: transport)
    result_value = capture_postprocessing_phase_diagnostics(
        PHASE_RUN_ID,
        authority_root=tmp_path / "authority",
        diagnostics_root=tmp_path / "diagnostics",
    )
    return result_value, transport, json.loads(result_value.output.read_bytes())


def test_action09_failed_adjudication_uses_the_exact_attempt_bound_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, transport, summary = _capture_action09(tmp_path, monkeypatch)

    assert transport.read_calls == [
        ("/remote/evidence/slurm-logs/acceptance-adjudication.90.out", MAX_REMOTE_LOG_BYTES),
        ("/remote/evidence/slurm-logs/acceptance-adjudication.90.err", MAX_REMOTE_LOG_BYTES),
        ("/remote/evidence/phase-acceptance/adjudication.json", MAX_REMOTE_LOG_BYTES),
    ]
    assert summary["acceptance_adjudication"]["status"] == "captured"
    assert summary["acceptance_adjudication"]["result"] == "failed"
    assert summary["acceptance_adjudication"]["non_allowlisted_errors"] == 0
    residuals = summary["acceptance_adjudication"]["residual_cardinalities"]
    assert [item["reference"]["report"] for item in residuals] == [
        "acceptance/semantic_acceptance/semantic_acceptance_summary.json",
        "acceptance/verify_evidence/acceptance_evidence_report.json",
    ]
    assert {item["allowance_id"] for item in residuals} == {"<redacted>"}
    assert len({item["reference"]["expected_value_sha256"] for item in residuals}) == 2
    assert "ordinary-expected" not in json.dumps(residuals)
    assert summary["acceptance_support"]["status"] == "not-applicable"
    assert result.to_mapping()["schema_version"] == 2
    assert result.to_mapping()["acceptance_adjudication_status"] == "captured"


@pytest.mark.parametrize(
    ("mutator", "unreadable", "expected"),
    (
        (lambda documents: documents.pop("/remote/evidence/phase-acceptance/adjudication.json"), False, "missing"),
        (None, True, "unreadable"),
        (
            lambda documents: documents.__setitem__("/remote/evidence/phase-acceptance/adjudication.json", b"not-json"),
            False,
            "invalid",
        ),
        (
            lambda documents: documents.__setitem__(
                "/remote/evidence/phase-acceptance/adjudication.json",
                documents["/remote/evidence/phase-acceptance/adjudication.json"].replace(
                    PHASE_RUN_ID.encode(), b"phase-run-fedcba9876543210fedcba9876543210"
                ),
            ),
            False,
            "invalid",
        ),
    ),
    ids=("missing", "unreadable", "invalid-json", "identity-mismatch"),
)
def test_action09_adjudication_failure_states_preserve_log_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: Callable[[dict[str, bytes]], None] | None,
    unreadable: bool,
    expected: str,
) -> None:
    _result, transport, summary = _capture_action09(
        tmp_path,
        monkeypatch,
        documents_mutator=mutator,
        unreadable=unreadable,
    )

    assert summary["acceptance_adjudication"]["status"] == expected
    assert len(summary["logs"]) == 2
    assert transport.read_calls[-1] == ("/remote/evidence/phase-acceptance/adjudication.json", MAX_REMOTE_LOG_BYTES)


def test_action09_passed_adjudication_is_identified_as_post_adjudication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _result, _transport, summary = _capture_action09(tmp_path, monkeypatch, result="passed")

    assert summary["acceptance_adjudication"]["status"] == "captured"
    assert summary["acceptance_adjudication"]["result"] == "passed"
    assert summary["acceptance_adjudication"]["post_adjudication_failure"] is True


def test_action09_positive_count_support_requires_adjudication_capture_digest_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _result, transport, summary = _capture_action09(tmp_path, monkeypatch, non_allowlisted_errors=2)

    support = summary["acceptance_support"]
    assert support["status"] == "captured"
    assert support["complete"] is True
    assert (
        support["files"][0]["remote_path"]
        == "/remote/evidence/phase-acceptance/acceptance-tar-payload-parity-capture.json"
    )
    assert support["unallowlisted_projection"]["rows"]
    assert all("<redacted>" in row["value"] for row in support["unallowlisted_projection"]["rows"])
    assert "secret-0" not in json.dumps(support["unallowlisted_projection"])
    assert all(maximum == MAX_REMOTE_LOG_BYTES for _path, maximum in transport.read_calls)

    def mismatch(documents: dict[str, bytes]) -> None:
        payload = json.loads(documents["/remote/evidence/phase-acceptance/adjudication.json"])
        payload["capture_digests"][0] = "0" * 64
        documents["/remote/evidence/phase-acceptance/adjudication.json"] = json.dumps(payload).encode()

    _result, _transport, mismatched = _capture_action09(
        tmp_path / "mismatch",
        monkeypatch,
        non_allowlisted_errors=2,
        documents_mutator=mismatch,
    )
    assert mismatched["acceptance_adjudication"]["status"] == "captured"
    assert mismatched["acceptance_support"]["status"] == "partial"
    assert "consistency_failure" in mismatched["acceptance_support"]
    assert "unallowlisted_projection" not in mismatched["acceptance_support"]
    assert "unattributed_non_allowlisted_errors" not in mismatched["acceptance_support"]


def test_acceptance_support_deduplicates_normalized_absolute_path_aliases() -> None:
    root = diagnostics_module.PurePosixPath("/remote/evidence")
    requests = [
        ("report", "a/./b.json", "acceptance-semantic", root / "a/./b.json"),
        ("report", "a/b.json", "acceptance-semantic", root / "a/b.json"),
    ]

    deduplicated = diagnostics_module._deduplicate_acceptance_support_requests(requests)

    assert len(deduplicated) == 1
    assert deduplicated[0][1] == "a/./b.json"


def test_action09_support_rejects_canonical_count_above_adjudicated_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _result, _transport, summary = _capture_action09(
        tmp_path,
        monkeypatch,
        non_allowlisted_errors=1,
        canonical_error_count=2,
    )

    assert summary["acceptance_adjudication"]["status"] == "captured"
    support = summary["acceptance_support"]
    assert support["status"] == "partial"
    assert support["complete"] is False
    assert "consistency_failure" in support
    assert "unallowlisted_projection" not in support
    assert "unattributed_non_allowlisted_errors" not in support
