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

"""Cancellation must recover exact accounting exits without inventing unset signals.

The compact JSON fixture preserves an observed Slurm exit-code shape: a set
return code and an unset signal ID. Job IDs and queue responses are synthetic.
All scheduler commands use recording runners; the folding fixture stages only
its small local files through the existing submission fixture.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bspp.orchestration.contract.phase_cancellation import PhaseCancellationCompletedEvent
from bspp.orchestration.contract.phase_reconciliation import PhaseActionTerminalObservedEvent
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore
from bspp.orchestration.control.phase_cancellation import cancel_phase
from bspp.orchestration.control.phase_submission import submit_phase
from bspp.orchestration.control.transport import CommandResult
from tests.test_phase_cancellation import CancellationRunner, _reject_remote
from tests.test_phase_folding_lifecycle import FoldingSubmissionRunner, _materialize_folding
from tests.test_phase_resume import _materialized_authority, _record_submission_state

NOW = datetime(2026, 9, 19, 12, tzinfo=UTC)


class SparseExitRunner:
    def __init__(
        self,
        *,
        states: dict[str, tuple[str, str]] | None = None,
        fallback_rows: str | None = None,
        unavailable: bool = False,
        reject_restarts: bool = True,
    ) -> None:
        self.states = states or {"123456": ("COMPLETED", "0:0")}
        self.fallback_rows = fallback_rows
        self.unavailable = unavailable
        self.reject_restarts = reject_restarts
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "squeue":
            # Expired queue records must not prevent exact accounting recovery.
            return CommandResult(argv, 1, "", "slurm_load_jobs error: Invalid job id specified")
        if argv[0] == "scancel":
            return CommandResult(argv, 0, "", "")
        assert argv[0] == "sacct", argv
        if self.reject_restarts and any("Restarts" in part for part in argv):
            return CommandResult(argv, 1, "", 'sacct: error: Invalid field requested: "Restarts"\n')
        ids = argv[argv.index("-j") + 1].split(",")
        if "--json" in argv:
            jobs = []
            for job in ids:
                state, exit_code = self.states[job]
                jobs.append(
                    {
                        "job_id": int(job),
                        "array": {
                            "job_id": 0,
                            "task_id": {"set": False, "infinite": False, "number": 0},
                        },
                        "state": {"current": [state]},
                        "exit_code": {
                            "return_code": {"set": True, "infinite": False, "number": int(exit_code.split(":")[0])},
                            "signal": {"id": {"set": False, "infinite": False, "number": 0}, "name": ""},
                        },
                    }
                )
            return CommandResult(argv, 0, json.dumps({"jobs": jobs}), "")
        assert any(part.startswith("--format=JobIDRaw,JobID,State,ExitCode") for part in argv), argv
        if self.unavailable:
            return CommandResult(argv, 1, "", "accounting unavailable")
        rows = self.fallback_rows
        if rows is None:
            rows = "".join(f"{job}|{job}|{self.states[job][0]}|{self.states[job][1]}\n" for job in ids)
        return CommandResult(argv, 0, rows, "")


@pytest.mark.parametrize("reject_restarts", [True, False])
@pytest.mark.parametrize("state,exit_code", [("COMPLETED", "0:0"), ("FAILED", "15:0"), ("CANCELLED", "0:15")])
def test_cancellation_uses_exact_accounting_fallback_without_recancelling_terminal_job(
    tmp_path: Path, reject_restarts: bool, state: str, exit_code: str
) -> None:
    root, phase = _materialized_authority(tmp_path)
    _record_submission_state(root, phase, "submitted")
    runner = SparseExitRunner(states={"123456": (state, exit_code)}, reject_restarts=reject_restarts)

    result = cancel_phase(phase, authority_root=root, clock=lambda: NOW, runner=runner)

    assert result.status == "cancelled"
    assert result.terminal_job_ids == ("123456",)
    assert not any(call[0] == "scancel" for call in runner.calls)
    assert any("sacct_json_incomplete" in warning for warning in result.warnings)
    assert any("squeue_unavailable" in warning for warning in result.warnings)
    assert [(source.source, source.availability) for source in result.scheduler_sources] == [
        ("squeue", "unavailable"),
        ("sacct", "available"),
    ]
    replay = PhaseAuthorityStore(root).validate(phase)
    assert [(r.state, r.exit_code) for r in replay.terminal_observations] == [(state, exit_code)]
    assert sum(isinstance(event, PhaseCancellationCompletedEvent) for event in replay.events) == 1
    before = {str(f): f.read_bytes() for f in (root / phase).rglob("*") if f.is_file()}
    again = cancel_phase(phase, authority_root=root, clock=lambda: NOW, runner=_reject_remote)
    assert again.status == "cancelled"
    assert {str(f): f.read_bytes() for f in (root / phase).rglob("*") if f.is_file()} == before


@pytest.mark.parametrize(
    "rows",
    [
        "",
        "123456|123456|COMPLETED|\n",
        "malformed\n",
        "123456|123456|COMPLETED|0\n",
        "999999|999999|COMPLETED|0:0\n",
        "123456|999999|COMPLETED|0:0\n",
        "123456|123456_0|COMPLETED|0:0\n",
        "123457|123456_0|COMPLETED|0:0\n",
        "123456.batch|123456.batch|COMPLETED|0:0\n",
        "123456|123456|COMPLETED|0:0\n123456|123456|FAILED|1:0\n",
    ],
)
def test_inconclusive_fallback_does_not_close_bound_parent(tmp_path: Path, rows: str) -> None:
    root, phase = _materialized_authority(tmp_path)
    _record_submission_state(root, phase, "submitted")
    runner = SparseExitRunner(fallback_rows=rows)

    result = cancel_phase(phase, authority_root=root, clock=lambda: NOW, runner=runner)

    assert result.status == "cancelling"
    assert result.terminal_job_ids == ()
    assert any("sacct_json_incomplete" in warning for warning in result.warnings)
    replay = PhaseAuthorityStore(root).validate(phase)
    assert replay.terminal_observations == ()
    assert not any(isinstance(event, PhaseCancellationCompletedEvent) for event in replay.events)


def test_unavailable_exact_fallback_preserves_cancellation_intent(tmp_path: Path) -> None:
    root, phase = _materialized_authority(tmp_path)
    _record_submission_state(root, phase, "submitted")
    runner = SparseExitRunner(unavailable=True)

    result = cancel_phase(phase, authority_root=root, clock=lambda: NOW, runner=runner)

    assert result.status == "cancelling"
    assert result.terminal_job_ids == ()
    assert any("accounting unavailable" in warning for warning in result.warnings)
    assert PhaseAuthorityStore(root).validate(phase).terminal_observations == ()


def test_existing_folding_cancellation_recovers_three_completed_predecessors(tmp_path: Path) -> None:
    root, phase = _materialize_folding(tmp_path)
    submit_phase(phase, authority_root=root, clock=lambda: NOW, runner=FoldingSubmissionRunner())
    first = cancel_phase(
        phase, authority_root=root, clock=lambda: NOW, runner=CancellationRunner(accounting_states=[("RUNNING", None)])
    )
    assert first.status == "cancelling"
    before = {str(f): f.read_bytes() for f in (root / phase).rglob("*") if f.is_file()}
    states = {str(job): ("COMPLETED", "0:0") if job < 1004 else ("CANCELLED", "0:15") for job in range(1001, 1006)}
    runner = SparseExitRunner(states=states)

    result = cancel_phase(phase, authority_root=root, clock=lambda: NOW, runner=runner)

    assert result.status == "cancelled"
    assert result.terminal_job_ids == tuple(states)
    assert not any(call[0] == "scancel" for call in runner.calls)
    replay = PhaseAuthorityStore(root).validate(phase)
    events = [event for event in replay.events if isinstance(event, PhaseActionTerminalObservedEvent)]
    assert [(e.payload.job_id, e.payload.state, e.payload.exit_code) for e in events] == [
        (job, *state_exit) for job, state_exit in states.items()
    ]
    assert all(Path(path).read_bytes() == data for path, data in before.items())
