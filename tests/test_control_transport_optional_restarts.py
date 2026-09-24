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

"""Accounting compatibility when a Slurm build does not expose Restarts."""

from __future__ import annotations

import json

import pytest

from bspp.orchestration.control.monitoring import (
    parse_sacct_identity_parsable_rows,
    parse_sacct_parsable_rows,
)
from bspp.orchestration.control.postprocessing_phase_lifecycle import _complete_action_task_set
from bspp.orchestration.control.transport import CommandResult, RemoteSlurmTransport, command_argv
from tests.test_control_transport import RecordingRunner


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("suffix", ["", "|", "||"])
def test_optional_restarts_remains_unknown(identity: bool, suffix: str) -> None:
    parser = parse_sacct_identity_parsable_rows if identity else parse_sacct_parsable_rows
    prefix = "18917854|18917854" if identity else "18917854"
    row = f"{prefix}|COMPLETED|0:0{suffix}"

    records = parser(row + "\n", requested=("18917854",))

    assert [(record.job_id, record.state, record.exit_code, record.restarts, record.raw) for record in records] == [
        ("18917854", "COMPLETED", "0:0", None, row)
    ]
    action = type("ScalarAction", (), {"expected_task_indexes": ()})()
    assert _complete_action_task_set(action, "18917854", records, autorequeue_enabled=True) is None


@pytest.mark.parametrize(
    ("identity", "row"),
    [
        (False, "18917854|COMPLETED"),
        (False, "18917854|COMPLETED|0"),
        (False, "18917854|COMPLETED|0:0|0|extra"),
        (False, "not-a-job|COMPLETED|0:0"),
        (True, "18917854|18917854|COMPLETED"),
        (True, "18917854|18917854|COMPLETED|0"),
        (True, "18917854|18917854|COMPLETED|0:0|0|extra"),
        (True, "18917854|99999|COMPLETED|0:0"),
        (True, "18917854.batch|18917854.extern|COMPLETED|0:0"),
        (True, "18917854|18917854|RUNNING|0:0"),
    ],
)
def test_optional_restarts_does_not_hide_malformed_required_fields(identity: bool, row: str) -> None:
    parser = parse_sacct_identity_parsable_rows if identity else parse_sacct_parsable_rows

    with pytest.raises(ValueError, match="malformed sacct"):
        parser(row + "\n", requested=("18917854",))


@pytest.mark.parametrize("kind", ["local-slurm", "ssh"])
@pytest.mark.parametrize(("state", "exit_code"), [("FAILED", "1:0"), ("COMPLETED", "0:0")])
def test_transport_recovers_exact_exit_when_slurm_rejects_restarts(kind: str, state: str, exit_code: str) -> None:
    # FAILED rows are the exact response captured from Slurm job 18917854.
    rows = (
        "18917854|18917854|FAILED|1:0\n"
        "18917854.batch|18917854.batch|FAILED|1:0\n"
        "18917854.extern|18917854.extern|COMPLETED|0:0\n"
        "18917854.0|18917854.0|COMPLETED|0:0\n"
        "18917854.1|18917854.1|FAILED|1:0\n"
        if state == "FAILED"
        else "18917854|18917854|COMPLETED|0:0\n"
    )
    sparse_job = {
        "job_id": 18917854,
        "state": {"current": [state]},
        "exit_code": {
            "return_code": {"set": True, "number": int(exit_code.split(":")[0])},
            "signal": {"id": {"set": False, "number": 0}},
        },
    }
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout='{"jobs": []}', stderr=""),
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": [sparse_job]}), stderr=""),
            CommandResult(
                argv=(), returncode=1, stdout="", stderr='sacct: error: Invalid field requested: "Restarts"\n'
            ),
            CommandResult(argv=(), returncode=0, stdout=rows, stderr=""),
        ]
    )
    ssh_target = "example-cluster-login" if kind == "ssh" else None
    transport = RemoteSlurmTransport(kind=kind, ssh_target=ssh_target, runner=runner)

    observation = transport.query_observation(("18917854",), require_exact_terminal_exit=True)

    assert [(record.state, record.exit_code) for record in observation.selected_states] == [(state, exit_code)]
    assert observation.sacct.parser == "parsable-fallback"
    assert observation.sacct.raw_text == rows
    assert all(record.restarts is None for record in observation.sacct_jobs)
    assert [record.job_id for record in observation.sacct_jobs] == [row.split("|")[1] for row in rows.splitlines()]
    assert "sacct_json_incomplete" in observation.warnings[0]
    assert not any("sacct_unavailable" in warning for warning in observation.warnings)
    for offset, suffix in [(-2, ",Restarts"), (-1, "")]:
        expected = command_argv(
            ("sacct", "-j", "18917854", f"--format=JobIDRaw,JobID,State,ExitCode{suffix}", "--noheader", "--parsable2"),
            transport=kind,
            ssh_target=ssh_target,
        )
        assert runner.calls[offset] == expected
    assert observation.sacct.argv == runner.calls[-1]
