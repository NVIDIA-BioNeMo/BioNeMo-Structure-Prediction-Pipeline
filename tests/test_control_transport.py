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

"""Tests for Control Plane command transport helpers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from bspp.orchestration.control.monitoring import (
    SlurmCommandSnapshot,
    SlurmObservation,
    parse_sacct_identity_parsable_rows,
    parse_sacct_json,
    parse_sacct_parsable_rows,
    parse_squeue_json,
)
from bspp.orchestration.control.postprocessing_phase_lifecycle import (
    _array_parent_cancelled_before_task_instantiation,
    _complete_parent_hierarchy_observation,
)
from bspp.orchestration.control.transport import (
    CommandResult,
    RemoteSlurmTransport,
    SlurmAction,
    SlurmActionTransport,
    SlurmSubmissionRejected,
    SlurmSubmissionUncertain,
    command_argv,
    default_command_runner,
    legacy_command_argv,
    normalize_slurm_state,
    parse_sbatch_job_id,
    run_probe,
    shell_argv,
)
from tests.support.transport_argv import legacy_remote_command, wrap_remote_command


class RecordingRunner:
    def __init__(self, responses: list[CommandResult]) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.responses = responses

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        response = self.responses.pop(0)
        return CommandResult(argv=argv, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr)


@pytest.mark.parametrize(
    ("raw_job", "expected_job_id"),
    [
        ({"job_id": {"number": 13048721}}, "13048721"),
        (
            {
                "job_id": {"number": 13048721},
                "array_job_id": {"number": 0},
                "array_task_id": {"set": False},
            },
            "13048721",
        ),
        (
            {
                "job_id": {"number": 13048726},
                "array_job_id": {"number": 13048726},
                "array_task_id": {"set": False},
            },
            "13048726",
        ),
        (
            {
                "job_id": {"number": 13048726},
                "array_job_id": {"number": 13048726},
                "array_task_id": {"set": True, "number": 853},
            },
            "13048726_853",
        ),
    ],
)
def test_squeue_json_decodes_example_cluster_wrapped_scalar_and_array_ids(
    raw_job: dict[str, object], expected_job_id: str
) -> None:
    records = parse_squeue_json(json.dumps({"jobs": [raw_job]}), requested=())

    assert [(record.job_id, record.requested_job_id) for record in records] == [(expected_job_id, None)]


def test_sparse_terminal_sacct_json_is_replaced_by_exact_parsable_fallback() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {
                                "job_id_raw": "13048721",
                                "state": {"current": "FAILED"},
                                "exit_code": {"return_code": 0, "status": "SUCCESS"},
                            }
                        ]
                    }
                ),
                stderr="",
            ),
            CommandResult(argv=(), returncode=0, stdout="13048721|13048721|FAILED|1:0|0|\n", stderr=""),
        ]
    )
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    observation = transport.query_observation(("13048721",), require_exact_terminal_exit=True)

    assert observation.sacct.parser == "parsable-fallback"
    assert observation.sacct.warning is not None and "sacct_json_incomplete" in observation.sacct.warning
    assert [(state.state, state.exit_code, state.source) for state in observation.selected_states] == [
        ("FAILED", "1:0", "sacct")
    ]
    assert observation.sacct.raw_json is None
    assert runner.calls[-1] == (
        "sacct",
        "-j",
        "13048721",
        "--format=JobIDRaw,JobID,State,ExitCode,Restarts",
        "--noheader",
        "--parsable2",
    )


def test_exact_terminal_exit_observation_keeps_only_requested_raw_jobs() -> None:
    runner = RecordingRunner(
        [
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {"meta": {"cluster": "example-cluster"}, "jobs": [{"job_id": 1001}, {"job_id": 2002}]}
                ),
                stderr="",
            ),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "meta": {"cluster": "example-cluster"},
                        "jobs": [
                            {"job_id_raw": "1001", "state": "RUNNING"},
                            {"job_id_raw": "2002", "state": "COMPLETED", "exit_code": "0:0"},
                        ],
                    }
                ),
                stderr="",
            ),
        ]
    )

    observation = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner).query_observation(
        ("1001",),
        require_exact_terminal_exit=True,
    )

    assert observation.squeue.raw_json == {"jobs": [{"job_id": 1001}]}
    assert observation.sacct.raw_json == {"jobs": [{"job_id_raw": "1001", "state": "RUNNING"}]}
    assert [item.job_id for item in observation.squeue_jobs] == ["1001"]
    assert [item.job_id for item in observation.sacct_jobs] == ["1001"]


@pytest.mark.parametrize(
    ("derived_job_id", "parser"),
    [
        ("1001.batch", "json"),
        ("1001_853", "json"),
        ("1001_[853-900]", "json"),
        ("1001.batch", "parsable-fallback"),
        ("1001_853", "parsable-fallback"),
    ],
)
def test_exact_terminal_observation_preserves_complete_requested_parent_hierarchy(
    derived_job_id: str,
    parser: str,
) -> None:
    parent_row = {"job_id_raw": "1001", "state": "CANCELLED", "exit_code": "0:0"}
    derived_row = {"job_id_raw": derived_job_id, "state": "CANCELLED", "exit_code": "0:0"}
    responses = [CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr="")]
    if parser == "json":
        responses.append(
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            parent_row,
                            derived_row,
                            {"job_id_raw": "2002.batch", "state": "COMPLETED", "exit_code": "0:0"},
                        ]
                    }
                ),
                stderr="",
            )
        )
    else:
        responses.extend(
            (
                CommandResult(argv=(), returncode=1, stdout="", stderr="sacct JSON unsupported"),
                CommandResult(
                    argv=(),
                    returncode=0,
                    stdout=(
                        "1001|1001|CANCELLED|0:0|0|\n"
                        + (
                            "1001.batch|1001.batch|CANCELLED|0:0|0|\n"
                            if derived_job_id == "1001.batch"
                            else "1001|1001_853|CANCELLED|0:0|0|\n"
                        )
                    ),
                    stderr="",
                ),
            )
        )
    observation = RemoteSlurmTransport(
        kind="local-slurm",
        ssh_target=None,
        runner=RecordingRunner(responses),
    ).query_observation(("1001",), require_exact_terminal_exit=True)

    assert [record.job_id for record in observation.sacct_jobs] == ["1001", derived_job_id]
    assert [(state.job_id, state.state) for state in observation.selected_states] == [("1001", "CANCELLED")]
    assert _complete_parent_hierarchy_observation(observation) is True
    action = type("ArrayAction", (), {"expected_task_indexes": (853,)})()
    evidence = _array_parent_cancelled_before_task_instantiation(action, "1001", observation.sacct_jobs)
    if derived_job_id == "1001.batch" and parser == "json":
        assert evidence is not None and evidence.scheduler_job_id == "1001"
    else:
        assert evidence is None


def test_parent_hierarchy_absence_proof_uses_only_complete_sacct_snapshot() -> None:
    base = {
        "requested_job_ids": ("1001",),
        "squeue": SlurmCommandSnapshot(kind="squeue", argv=(), returncode=1, parser="unavailable"),
        "squeue_jobs": (),
        "sacct_jobs": (),
        "selected_states": (),
        "warnings": ("unrelated squeue warning",),
    }
    clean_json = SlurmObservation(
        **base,
        sacct=SlurmCommandSnapshot(kind="sacct", argv=("sacct", "--json"), returncode=0, parser="json"),
    )
    strict_fallback = SlurmObservation(
        **base,
        sacct=SlurmCommandSnapshot(
            kind="sacct",
            argv=("sacct", "--format=JobIDRaw,JobID,State,ExitCode,Restarts"),
            returncode=0,
            parser="parsable-fallback",
            raw_text="",
            warning="sacct_json_unavailable: unsupported",
        ),
    )
    malformed_json = SlurmObservation(
        **base,
        sacct=SlurmCommandSnapshot(
            kind="sacct",
            argv=("sacct", "--json"),
            returncode=0,
            parser="json",
            warning="sacct_json_incomplete: malformed row",
        ),
    )
    unavailable = SlurmObservation(
        **base,
        sacct=SlurmCommandSnapshot(kind="sacct", argv=("sacct",), returncode=1, parser="unavailable"),
    )

    assert _complete_parent_hierarchy_observation(clean_json) is True
    assert _complete_parent_hierarchy_observation(strict_fallback) is True
    assert _complete_parent_hierarchy_observation(malformed_json) is False
    assert _complete_parent_hierarchy_observation(unavailable) is False


def test_exact_terminal_json_malformed_row_requires_strict_fallback_before_absence_proof() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {"job_id_raw": "1001", "state": "CANCELLED", "exit_code": "0:0"},
                            {"state": "CANCELLED", "exit_code": "0:0"},
                        ]
                    }
                ),
                stderr="",
            ),
            CommandResult(argv=(), returncode=0, stdout="1001|1001|CANCELLED|0:0|0|\n", stderr=""),
        ]
    )

    observation = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner).query_observation(
        ("1001",),
        require_exact_terminal_exit=True,
    )

    assert observation.sacct.parser == "parsable-fallback"
    assert "malformed sacct JSON row" in (observation.sacct.warning or "")
    assert [record.job_id for record in observation.sacct_jobs] == ["1001"]
    assert _complete_parent_hierarchy_observation(observation) is True


def test_default_observation_preserves_rich_json_without_exact_exit_probe() -> None:
    squeue_payload = {"meta": {"cluster": "example-cluster"}, "jobs": [{"job_id": 1001}, {"job_id": 2002}]}
    sacct_payload = {
        "meta": {"cluster": "example-cluster", "query": "rich"},
        "jobs": [
            {
                "job_id_raw": "1001",
                "state": "FAILED",
                "exit_code": {"return_code": 0, "status": "SUCCESS"},
                "elapsed": "00:00:10",
            },
            {"job_id_raw": "2002", "state": "RUNNING", "elapsed": "00:00:11"},
        ],
    }
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps(squeue_payload), stderr=""),
            CommandResult(argv=(), returncode=0, stdout=json.dumps(sacct_payload), stderr=""),
        ]
    )

    observation = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner).query_observation(("1001",))

    assert observation.squeue.parser == observation.sacct.parser == "json"
    assert observation.squeue.raw_json == squeue_payload
    assert observation.sacct.raw_json == sacct_payload
    assert observation.warnings == ()
    assert [call[0] for call in runner.calls] == ["squeue", "sacct"]


def test_sacct_json_keeps_sparse_exit_code_uninterpreted() -> None:
    records = parse_sacct_json(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id_raw": "13048721",
                        "state": "FAILED",
                        "exit_code": {"return_code": 0, "status": "SUCCESS"},
                    }
                ]
            }
        ),
        requested=("13048721",),
    )

    assert records[0].exit_code is None


def test_sacct_json_decodes_complete_wrapped_exit_code_components() -> None:
    records = parse_sacct_json(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id_raw": "13048721",
                        "state": "FAILED",
                        "exit_code": {
                            "return_code": {"number": 1, "set": True},
                            "signal": {"number": 0, "set": True},
                        },
                    }
                ]
            }
        ),
        requested=("13048721",),
    )

    assert records[0].exit_code == "1:0"


def test_strict_sacct_fallback_preserves_array_children_without_fabricating_them() -> None:
    child = parse_sacct_parsable_rows("13048726_853|FAILED|1:0|0|\n", requested=("13048726",))
    parent = parse_sacct_parsable_rows("13048726|CANCELLED|0:0|0|\n", requested=("13048726",))

    assert [(record.job_id, record.exit_code) for record in child] == [("13048726_853", "1:0")]
    assert [(record.job_id, record.exit_code) for record in parent] == [("13048726", "0:0")]


def test_collapsed_array_json_uses_dual_identity_fallback_for_exact_child() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps({"jobs": [{"job_id_raw": "13285540", "state": "COMPLETED", "exit_code": "0:0"}]}),
                stderr="",
            ),
            CommandResult(
                argv=(),
                returncode=0,
                stdout="13285540|13285540_853|COMPLETED|0:0|0|\n",
                stderr="",
            ),
        ]
    )

    observation = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner).query_observation(
        ("13285540",),
        require_exact_terminal_exit=True,
        expected_terminal_job_ids=("13285540_853",),
    )

    assert [(record.job_id, record.requested_job_id) for record in observation.sacct_jobs] == [
        ("13285540_853", "13285540")
    ]
    assert len(runner.calls) == 3
    assert runner.calls[-1][3] == "--format=JobIDRaw,JobID,State,ExitCode,Restarts"


def test_live_shaped_sparse_collapsed_array_parent_enriches_from_exact_child_only() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {
                                "job_id_raw": "13285540",
                                "state": "COMPLETED",
                                "exit_code": {"return_code": 0, "status": "SUCCESS"},
                            }
                        ]
                    }
                ),
                stderr="",
            ),
            CommandResult(
                argv=(),
                returncode=0,
                stdout="13285540|13285540_853|COMPLETED|0:0|0|\n",
                stderr="",
            ),
        ]
    )

    observation = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner).query_observation(
        ("13285540",),
        require_exact_terminal_exit=True,
        expected_terminal_job_ids=("13285540_853",),
    )

    assert observation.sacct.parser == "parsable-fallback"
    assert [(record.job_id, record.exit_code) for record in observation.sacct_jobs] == [("13285540_853", "0:0")]
    assert len(runner.calls) == 3


def test_batched_sparse_scalar_and_collapsed_array_enrich_exact_endpoints() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {
                                "job_id_raw": "13285539",
                                "state": "COMPLETED",
                                "exit_code": {"return_code": 0, "status": "SUCCESS"},
                            },
                            {
                                "job_id_raw": "13285540",
                                "state": "COMPLETED",
                                "exit_code": {"return_code": 0, "status": "SUCCESS"},
                            },
                        ]
                    }
                ),
                stderr="",
            ),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=("13285539|13285539|COMPLETED|0:0|0|\n13285540|13285540_853|COMPLETED|0:0|0|\n"),
                stderr="",
            ),
        ]
    )

    observation = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner).query_observation(
        ("13285539", "13285540"),
        require_exact_terminal_exit=True,
        expected_terminal_job_ids=("13285539", "13285540_853"),
    )

    assert [(record.job_id, record.exit_code) for record in observation.sacct_jobs] == [
        ("13285539", "0:0"),
        ("13285540_853", "0:0"),
    ]
    assert len(runner.calls) == 3


def test_exact_array_json_fast_path_does_not_issue_fallback_call() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps({"jobs": [{"job_id_raw": "13285540_853", "state": "COMPLETED", "exit_code": "0:0"}]}),
                stderr="",
            ),
        ]
    )

    observation = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner).query_observation(
        ("13285540",),
        require_exact_terminal_exit=True,
        expected_terminal_job_ids=("13285540_853",),
    )

    assert observation.sacct.parser == "json"
    assert len(runner.calls) == 2


def test_best_effort_invalid_collapsed_array_fallback_keeps_scoped_json_without_child() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps({"jobs": [{"job_id_raw": "13285540", "state": "COMPLETED", "exit_code": "0:0"}]}),
                stderr="",
            ),
            CommandResult(
                argv=(),
                returncode=0,
                stdout="13285540|99999_853|COMPLETED|0:0|0|\n",
                stderr="",
            ),
        ]
    )

    observation = RemoteSlurmTransport(
        kind="local-slurm", ssh_target=None, runner=runner
    ).query_observation_best_effort(
        ("13285540",),
        require_exact_terminal_exit=True,
        expected_terminal_job_ids=("13285540_853",),
    )

    assert observation.sacct.parser == "json"
    assert [record.job_id for record in observation.sacct_jobs] == ["13285540"]
    assert "sacct_unavailable" in observation.warnings[-1]


def test_parent_only_cancelled_dual_fallback_proves_absent_array_child_without_synthesis() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps({"jobs": [{"job_id_raw": "13285540", "state": "CANCELLED", "exit_code": "0:0"}]}),
                stderr="",
            ),
            CommandResult(argv=(), returncode=0, stdout="13285540|13285540|CANCELLED|0:0|0|\n", stderr=""),
        ]
    )

    observation = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner).query_observation(
        ("13285540",),
        require_exact_terminal_exit=True,
        expected_terminal_job_ids=("13285540_853",),
    )

    assert [record.job_id for record in observation.sacct_jobs] == ["13285540"]
    assert _complete_parent_hierarchy_observation(observation) is True
    action = type("ArrayAction", (), {"expected_task_indexes": (853,)})()
    assert _array_parent_cancelled_before_task_instantiation(action, "13285540", observation.sacct_jobs) is not None


@pytest.mark.parametrize(
    "row",
    [
        "13285540|13285540_853+|COMPLETED|0:0|0|\n",
        "13285540|13285540_[853-900]|COMPLETED|0:0|0|\n",
        "13285540|99999_853|COMPLETED|0:0|0|\n",
        "13285540_7|13285540_853|COMPLETED|0:0|0|\n",
        "13285540.batch|13285540_853.extern|COMPLETED|0:0|0|\n",
        "13285540|13285540_853|RUNNING|0:0|0|\n",
        "13285540|13285540_853|COMPLETED|0|0|\n",
    ],
)
def test_dual_identity_terminal_parser_rejects_ambiguous_or_malformed_rows(row: str) -> None:
    with pytest.raises(ValueError, match="malformed sacct identity parsable row"):
        parse_sacct_identity_parsable_rows(row, requested=("13285540",))


def test_dual_identity_terminal_parser_leaves_malformed_restarts_unresolved() -> None:
    records = parse_sacct_identity_parsable_rows(
        "13285540|13285540_853|COMPLETED|0:0|extra|\n",
        requested=("13285540",),
    )

    assert [(record.job_id, record.exit_code, record.restarts) for record in records] == [("13285540_853", "0:0", None)]


def test_dual_identity_terminal_parser_retains_cross_checked_step_hierarchy() -> None:
    records = parse_sacct_identity_parsable_rows(
        "13285540|13285540_853|FAILED|1:0|0|\n13285540.batch|13285540_853.batch|FAILED|1:0|0|\n",
        requested=("13285540",),
    )

    assert [record.job_id for record in records] == ["13285540_853", "13285540_853.batch"]


def test_dual_identity_terminal_parser_accepts_normal_array_raw_ids_and_steps() -> None:
    records = parse_sacct_identity_parsable_rows(
        "13285540_853|13285540_853|COMPLETED|0:0|0|\n13285540_853.batch|13285540_853.batch|COMPLETED|0:0|0|\n",
        requested=("13285540",),
    )

    assert [(record.job_id, record.requested_job_id) for record in records] == [
        ("13285540_853", "13285540"),
        ("13285540_853.batch", "13285540"),
    ]


def test_collapsed_array_fallback_scopes_out_structurally_valid_unrelated_rows() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps({"jobs": [{"job_id_raw": "13285540", "state": "COMPLETED", "exit_code": "0:0"}]}),
                stderr="",
            ),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=(
                    "13285540|13285540_853|COMPLETED|0:0|0|\n"
                    "99999|99999_7|COMPLETED|0:0|0|\n"
                    "99999_7.batch|99999_7.batch|COMPLETED|0:0|0|\n"
                ),
                stderr="",
            ),
        ]
    )

    observation = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner).query_observation(
        ("13285540",),
        require_exact_terminal_exit=True,
        expected_terminal_job_ids=("13285540_853",),
    )

    assert [record.job_id for record in observation.sacct_jobs] == ["13285540_853"]


def test_parent_only_array_cancellation_cannot_mask_another_sparse_exit_obligation() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {"job_id_raw": "13285540", "state": "CANCELLED", "exit_code": "0:0"},
                            {
                                "job_id_raw": "13285541",
                                "state": "FAILED",
                                "exit_code": {"return_code": 0, "status": "SUCCESS"},
                            },
                        ]
                    }
                ),
                stderr="",
            ),
            CommandResult(argv=(), returncode=0, stdout="13285540|13285540|CANCELLED|0:0|0|\n", stderr=""),
        ]
    )

    observation = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner).query_observation(
        ("13285540", "13285541"),
        require_exact_terminal_exit=True,
        expected_terminal_job_ids=("13285540_853", "13285541"),
    )

    assert observation.sacct.parser == "json"
    assert "missing exact exit evidence" in observation.warnings[-1]
    assert _complete_parent_hierarchy_observation(observation) is False


@pytest.mark.parametrize(
    "expected",
    [("13285540_853", "13285540_853"), ("13285540_[853-900]",), ("99999_853",), ("13285540.batch",)],
)
def test_expected_terminal_endpoints_reject_duplicate_range_foreign_and_step_ids(
    expected: tuple[str, ...],
) -> None:
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=RecordingRunner([]))
    with pytest.raises(ValueError, match=r"expected terminal job ids?"):
        transport.query_observation(
            ("13285540",),
            require_exact_terminal_exit=True,
            expected_terminal_job_ids=expected,
        )


def test_expected_terminal_endpoints_are_illegal_outside_exact_mode() -> None:
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=RecordingRunner([]))
    with pytest.raises(ValueError, match="require exact terminal exit"):
        transport.query_observation(("13285540",), expected_terminal_job_ids=("13285540_853",))


def test_best_effort_sparse_sacct_fallback_failure_preserves_scoped_sacct_json() -> None:
    runner = RecordingRunner(
        [
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps({"jobs": [{"job_id": 13048721, "job_state": "RUNNING"}]}),
                stderr="",
            ),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {
                                "job_id_raw": "13048721",
                                "state": "FAILED",
                                "exit_code": {"return_code": 0, "status": "SUCCESS"},
                            }
                        ]
                    }
                ),
                stderr="",
            ),
            CommandResult(argv=(), returncode=0, stdout="13048721|13048721|FAILED||0|\n", stderr=""),
        ]
    )

    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)
    observation = transport.query_observation_best_effort(("13048721",), require_exact_terminal_exit=True)

    assert observation.sacct.parser == "json"
    assert [(job.job_id, job.state, job.exit_code) for job in observation.sacct_jobs] == [("13048721", "FAILED", None)]
    assert observation.sacct.raw_json == {
        "jobs": [
            {
                "job_id_raw": "13048721",
                "state": "FAILED",
                "exit_code": {"return_code": 0, "status": "SUCCESS"},
            }
        ]
    }
    assert [(state.state, state.source) for state in observation.selected_states] == [("FAILED", "sacct")]
    assert "sacct_json_incomplete" in observation.warnings[-1]
    assert "sacct_unavailable" in observation.warnings[-1]


@pytest.mark.parametrize(
    ("parsable_output", "raises"),
    [
        ("13048721|13048721|FAILED|0|0|\n", True),
        ("", False),
    ],
)
def test_non_best_effort_sparse_sacct_enrichment_is_attempted_once(
    parsable_output: str,
    raises: bool,
) -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {
                                "job_id_raw": "13048721",
                                "state": "FAILED",
                                "exit_code": {"return_code": 0, "status": "SUCCESS"},
                            }
                        ]
                    }
                ),
                stderr="",
            ),
            CommandResult(argv=(), returncode=0, stdout=parsable_output, stderr=""),
        ]
    )

    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)
    if raises:
        with pytest.raises(ValueError, match="malformed sacct identity parsable row"):
            transport.query_observation(("13048721",), require_exact_terminal_exit=True)
    else:
        observation = transport.query_observation(("13048721",), require_exact_terminal_exit=True)
        assert observation.sacct.parser == "json"
        assert [(job.job_id, job.state, job.exit_code) for job in observation.sacct_jobs] == [
            ("13048721", "FAILED", None)
        ]
        assert "sacct_json_incomplete" in observation.warnings[-1]
    assert len([call for call in runner.calls if call[0] == "sacct"]) == 2


def test_command_argv_local_slurm_passes_argv_through() -> None:
    assert command_argv(("squeue", "--version"), transport="local-slurm", ssh_target=None) == ("squeue", "--version")


def test_command_argv_ssh_quotes_remote_command_arguments() -> None:
    remote_command = "bash -lc 'echo '\"'\"'/tmp/with spaces'\"'\"''"
    assert command_argv(
        ("bash", "-lc", "echo '/tmp/with spaces'"),
        transport="ssh",
        ssh_target="example-cluster-login",
    ) == ("ssh", "example-cluster-login", wrap_remote_command(remote_command))


def test_shell_argv_ssh_runs_script_through_login_shell() -> None:
    script = "set -e; echo '/tmp/with spaces'"

    assert shell_argv(script, transport="ssh", ssh_target="example-cluster-login") == (
        "ssh",
        "example-cluster-login",
        wrap_remote_command(script),
    )


def test_shell_argv_local_slurm_runs_script_with_bash_login_shell() -> None:
    script = "set -e; echo ok"

    assert shell_argv(script, transport="local-slurm", ssh_target=None) == ("bash", "-lc", script)


def test_command_argv_ssh_requires_ssh_target() -> None:
    try:
        command_argv(("squeue", "--version"), transport="ssh", ssh_target=None)
    except ValueError as exc:
        assert str(exc) == "ssh transport requires ssh_target"
    else:
        raise AssertionError("expected ValueError")


def test_command_argv_rejects_unknown_transport() -> None:
    try:
        command_argv(("squeue", "--version"), transport="slurmrestd", ssh_target=None)
    except ValueError as exc:
        assert str(exc) == "unsupported transport 'slurmrestd'"
    else:
        raise AssertionError("expected ValueError")


def test_legacy_command_argv_ssh_uses_pre_base64_login_shell_form() -> None:
    """legacy_command_argv must produce the exact legacy ssh wire shape."""
    inner = "sbatch --parsable /tmp/action.sbatch"
    assert legacy_command_argv(
        ("sbatch", "--parsable", "/tmp/action.sbatch"),
        transport="ssh",
        ssh_target="example-cluster-login",
    ) == ("ssh", "example-cluster-login", legacy_remote_command(inner))


def test_legacy_command_argv_local_slurm_passes_argv_through() -> None:
    assert legacy_command_argv(
        ("sbatch", "--parsable", "/tmp/action.sbatch"),
        transport="local-slurm",
        ssh_target=None,
    ) == ("sbatch", "--parsable", "/tmp/action.sbatch")


def test_legacy_command_argv_ssh_requires_ssh_target() -> None:
    try:
        legacy_command_argv(("sbatch", "--version"), transport="ssh", ssh_target=None)
    except ValueError as exc:
        assert str(exc) == "ssh transport requires ssh_target"
    else:
        raise AssertionError("expected ValueError")


def test_legacy_command_argv_rejects_unknown_transport() -> None:
    try:
        legacy_command_argv(("sbatch", "--version"), transport="slurmrestd", ssh_target=None)
    except ValueError as exc:
        assert str(exc) == "unsupported transport 'slurmrestd'"
    else:
        raise AssertionError("expected ValueError")


def test_run_probe_uses_translated_transport_argv() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")

    result = run_probe(("squeue", "--version"), transport="ssh", ssh_target="example-cluster-login", runner=runner)

    assert result.argv == ("ssh", "example-cluster-login", wrap_remote_command("squeue --version"))
    assert calls == [("ssh", "example-cluster-login", wrap_remote_command("squeue --version"))]


def test_default_command_runner_executes_without_shell() -> None:
    result = default_command_runner(("printf", "%s", "a b"))

    assert result.returncode == 0
    assert result.stdout == "a b"


@pytest.mark.parametrize(
    "stdout",
    [
        "12345\n",
        "12345;example-cluster\n",
        "Submitted batch job 12345\n",
        "Submitted batch job 12345 on cluster example-cluster\n",
    ],
)
def test_parse_sbatch_job_id_accepts_common_slurm_outputs(stdout: str) -> None:
    assert parse_sbatch_job_id(stdout) == "12345"


def test_parse_sbatch_job_id_rejects_unstructured_output() -> None:
    with pytest.raises(ValueError, match="could not parse sbatch job id"):
        parse_sbatch_job_id("sbatch failed before returning a job id\n")


def test_remote_slurm_transport_submits_local_script_with_parsable_sbatch() -> None:
    runner = RecordingRunner([CommandResult(argv=(), returncode=0, stdout="Submitted batch job 4242\n", stderr="")])
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    submission = transport.submit_script(Path("/runs/evidence/slurm/preflight.sbatch"))

    assert submission.job_id == "4242"
    assert submission.command == ("sbatch", "--parsable", "/runs/evidence/slurm/preflight.sbatch")
    assert runner.calls == [("sbatch", "--parsable", "/runs/evidence/slurm/preflight.sbatch")]


def test_slurm_action_is_frozen() -> None:
    action = SlurmAction(action_id="preflight", script_path=Path("/runs/evidence/slurm/preflight.sbatch"))

    with pytest.raises(FrozenInstanceError):
        action.action_id = "changed"


def test_remote_slurm_transport_satisfies_action_transport_protocol() -> None:
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=RecordingRunner([]))

    action_transport: SlurmActionTransport = transport

    assert action_transport is transport


def test_remote_slurm_transport_exposes_governed_and_phase_api_union() -> None:
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=RecordingRunner([]))

    for method_name in (
        "stage_verified_artifact",
        "fetch_stable_artifact",
        "submit_script",
        "find_governed_submission_job",
        "submit_action",
        "query_observation",
        "query_observation_best_effort",
        "cancel_jobs",
    ):
        assert callable(getattr(transport, method_name))


def test_remote_slurm_transport_submits_action_with_ordered_dependencies() -> None:
    runner = RecordingRunner([CommandResult(argv=(), returncode=0, stdout="4242\n", stderr="")])
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    submission = transport.submit_action(
        SlurmAction(
            action_id="preprocess",
            script_path=Path("/runs/evidence/slurm/preprocess.sbatch"),
            dependency_job_ids=("1003", "1001", "1002"),
        )
    )

    assert submission.job_id == "4242"
    assert submission.command == (
        "sbatch",
        "--parsable",
        "--dependency=afterok:1003:1001:1002",
        "/runs/evidence/slurm/preprocess.sbatch",
    )
    assert runner.calls == [submission.command]


def test_governed_submission_overrides_job_name_with_exact_token() -> None:
    runner = RecordingRunner([CommandResult(argv=(), returncode=0, stdout="4242\n", stderr="")])
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    transport.submit_script(
        Path("/runs/step.sbatch"),
        job_name="bspp_sub_" + "a" * 64,
        environment=(("BSPP_SUBMISSION_TOKEN", "a" * 64),),
    )

    assert runner.calls == [
        (
            "sbatch",
            "--parsable",
            "--job-name=bspp_sub_" + "a" * 64,
            "--export=ALL,BSPP_SUBMISSION_TOKEN=" + "a" * 64,
            "/runs/step.sbatch",
        )
    ]


def test_governed_submission_reconciles_completed_exact_token() -> None:
    job_name = "bspp_sub_" + "a" * 64
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout="", stderr=""),
            CommandResult(argv=(), returncode=0, stdout=f"4242|{job_name}|tester|COMPLETED\n", stderr=""),
        ]
    )
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    assert transport.find_governed_submission_job(submission_token="a" * 64, owner="tester") == "4242"
    assert runner.calls[0][0] == "squeue"
    assert runner.calls[1][0] == "sacct"


def test_governed_submission_reconciles_completed_array_rows_to_base_job() -> None:
    job_name = "bspp_sub_" + "a" * 64
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout="", stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=(f"4242_1|{job_name}|tester|COMPLETED\n4242_2|{job_name}|tester|COMPLETED\n"),
                stderr="",
            ),
        ]
    )
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    assert transport.find_governed_submission_job(submission_token="a" * 64, owner="tester") == "4242"


def test_governed_submission_reconciliation_fails_closed_on_malformed_accounting() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout="", stderr=""),
            CommandResult(argv=(), returncode=0, stdout="4242.batch|wrong|tester|COMPLETED\n", stderr=""),
        ]
    )
    with pytest.raises(ValueError, match="malformed"):
        RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner).find_governed_submission_job(
            submission_token="a" * 64, owner="tester"
        )


def test_remote_slurm_transport_submits_ssh_script_through_login_shell() -> None:
    runner = RecordingRunner([CommandResult(argv=(), returncode=0, stdout="12345;example-cluster\n", stderr="")])
    transport = RemoteSlurmTransport(kind="ssh", ssh_target="example-cluster-login", runner=runner)

    submission = transport.submit_script(Path("/runs/evidence/slurm/preflight.sbatch"))

    assert submission.job_id == "12345"
    assert runner.calls == [
        command_argv(
            ("sbatch", "--parsable", "/runs/evidence/slurm/preflight.sbatch"),
            transport="ssh",
            ssh_target="example-cluster-login",
        )
    ]


def test_remote_slurm_transport_submits_action_over_ssh_without_exposing_action_id() -> None:
    runner = RecordingRunner([CommandResult(argv=(), returncode=0, stdout="12345;example-cluster\n", stderr="")])
    transport = RemoteSlurmTransport(kind="ssh", ssh_target="example-cluster-login", runner=runner)

    transport.submit_action(
        SlurmAction(
            action_id="caller-only-association",
            script_path=Path("/runs/evidence/slurm/preprocess.sbatch"),
            dependency_job_ids=("1002", "1001"),
        )
    )

    assert runner.calls == [
        command_argv(
            (
                "sbatch",
                "--parsable",
                "--dependency=afterok:1002:1001",
                "/runs/evidence/slurm/preprocess.sbatch",
            ),
            transport="ssh",
            ssh_target="example-cluster-login",
        )
    ]


def test_immutable_staging_accepts_identical_existing_target_and_rejects_collision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    target = tmp_path / "cluster" / "phase-runspec.json"
    source.write_bytes(b'{"exact":true}\n')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None)

    transport.stage_immutable_artifact(
        source,
        str(target),
        expected_sha256=expected,
        staging_token="submission-action",
    )
    transport.stage_immutable_artifact(
        source,
        str(target),
        expected_sha256=expected,
        staging_token="submission-action",
    )

    assert target.read_bytes() == source.read_bytes()
    source.write_bytes(b'{"divergent":true}\n')
    divergent = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="immutable artifact collision"):
        transport.stage_immutable_artifact(
            source,
            str(target),
            expected_sha256=divergent,
            staging_token="submission-action",
        )
    assert target.read_bytes() == b'{"exact":true}\n'


def test_remote_slurm_transport_preserves_submission_failure_detail() -> None:
    runner = RecordingRunner(
        [CommandResult(argv=(), returncode=1, stdout="ignored stdout", stderr="partition unavailable\n")]
    )
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    with pytest.raises(ValueError, match=r"^partition unavailable$"):
        transport.submit_script(Path("/runs/evidence/slurm/preflight.sbatch"))

    assert runner.calls == [("sbatch", "--parsable", "/runs/evidence/slurm/preflight.sbatch")]


def test_local_nonzero_is_definitive_but_ssh_nonzero_is_uncertain() -> None:
    local_result = CommandResult(argv=(), returncode=1, stdout="", stderr="partition unavailable\n")
    local = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=RecordingRunner([local_result]))
    with pytest.raises(SlurmSubmissionRejected, match="partition unavailable") as rejected:
        local.submit_script(Path("/runs/action.sbatch"))
    assert rejected.value.result.returncode == 1

    ssh_result = CommandResult(argv=(), returncode=255, stdout="", stderr="connection lost\n")
    ssh = RemoteSlurmTransport(kind="ssh", ssh_target="example-cluster-login", runner=RecordingRunner([ssh_result]))
    with pytest.raises(SlurmSubmissionUncertain, match="connection lost") as uncertain:
        ssh.submit_script(Path("/runs/action.sbatch"))
    assert uncertain.value.result.returncode == 255


def test_success_without_job_identity_is_uncertain() -> None:
    result = CommandResult(argv=(), returncode=0, stdout="accepted maybe\n", stderr="")
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=RecordingRunner([result]))

    with pytest.raises(SlurmSubmissionUncertain, match="could not parse sbatch job id") as uncertain:
        transport.submit_script(Path("/runs/action.sbatch"))

    assert uncertain.value.result.returncode == 0


def test_submission_correlation_filters_exact_name_comment_and_top_level_ids() -> None:
    squeue = {
        "jobs": [
            {"job_id": 4242, "name": "bspp_phase_abc", "comment": "phase:token", "job_state": "RUNNING"},
            {"job_id": 4243, "name": "bspp_phase_abc", "comment": "other", "job_state": "RUNNING"},
        ]
    }
    sacct = {
        "jobs": [
            {
                "job_id_raw": "4242",
                "job_name": "bspp_phase_abc",
                "comment": "phase:token",
                "state": "COMPLETED",
            },
            {
                "job_id_raw": "4242.batch",
                "job_name": "bspp_phase_abc",
                "comment": "phase:token",
                "state": "COMPLETED",
            },
        ]
    }
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps(squeue), stderr=""),
            CommandResult(argv=(), returncode=0, stdout=json.dumps(sacct), stderr=""),
        ]
    )
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    matches = transport.query_submissions_by_correlation(
        job_name="bspp_phase_abc",
        comment="phase:token",
        submitted_after="2026-08-20T00:00:00.123456Z",
    )

    assert tuple(record.job_id for record in matches) == ("4242",)
    assert matches[0].source == "sacct"
    assert runner.calls == [
        ("squeue", "--json", "--name=bspp_phase_abc"),
        (
            "sacct",
            "--json",
            "-X",
            "--name=bspp_phase_abc",
            "--starttime=2026-08-20T00:00:00",
            "--format=JobIDRaw,JobName,Comment,State,ExitCode,Submit",
        ),
    ]


def test_remote_slurm_transport_fetches_exact_ssh_artifact() -> None:
    runner = RecordingRunner([CommandResult(argv=(), returncode=0, stdout="", stderr="")])
    transport = RemoteSlurmTransport(kind="ssh", ssh_target="example-cluster-login", runner=runner)

    transport.fetch_artifact("/remote/attempt/result.json", Path("/control/fetched/result.json"))

    assert runner.calls == [
        ("scp", "example-cluster-login:/remote/attempt/result.json", "/control/fetched/result.json")
    ]


def test_verified_stage_is_confined_below_remote_root(tmp_path: Path) -> None:
    root = tmp_path / "remote"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "submissions").symlink_to(outside, target_is_directory=True)
    source = tmp_path / "artifact"
    source.write_bytes(b"governed")
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None)

    with pytest.raises(ValueError, match="unsafe remote artifact path"):
        transport.stage_verified_artifact(
            source,
            root / "submissions" / "attempt" / "expectation.json",
            remote_root=root,
            expected_sha256="c86ab019a28d89b95a9ad43a43cb80f6b5fe60c61410aa52748624117a7f7899",
        )

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("transfer_kind", ["symlink", "hardlink"])
def test_streamed_stage_never_touches_preexisting_transfer_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transfer_kind: str
) -> None:
    root = tmp_path / "remote"
    outside = tmp_path / "outside"
    transfer_parent = root / ".bspp-transfer"
    transfer_parent.mkdir(parents=True)
    outside.mkdir()
    nonce = "a" * 32
    if transfer_kind == "symlink":
        (transfer_parent / nonce).symlink_to(outside, target_is_directory=True)
    else:
        outside_file = outside / "file"
        outside_file.write_bytes(b"outside")
        os.link(outside_file, transfer_parent / nonce)
    monkeypatch.setattr("bspp.orchestration.control.transport.secrets.token_hex", lambda _: nonce)
    source = tmp_path / "artifact"
    source.write_bytes(b"governed")

    RemoteSlurmTransport(kind="local-slurm", ssh_target=None).stage_verified_artifact(
        source,
        root / "attempt" / "expectation.json",
        remote_root=root,
        expected_sha256="c86ab019a28d89b95a9ad43a43cb80f6b5fe60c61410aa52748624117a7f7899",
    )

    expected = [] if transfer_kind == "symlink" else [b"outside"]
    assert [path.read_bytes() for path in outside.iterdir() if path.is_file()] == expected


def test_verified_stage_and_stable_fetch_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "remote"
    source = tmp_path / "artifact"
    source.write_bytes(b"governed")
    target = root / "attempt" / "expectation.json"
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None)

    transport.stage_verified_artifact(
        source,
        target,
        remote_root=root,
        expected_sha256="c86ab019a28d89b95a9ad43a43cb80f6b5fe60c61410aa52748624117a7f7899",
    )

    assert transport.fetch_stable_artifact(target, remote_root=root, maximum_bytes=1024) == b"governed"
    assert target.stat().st_nlink == 1


def test_remote_directory_creation_rejects_intermediate_symlink(tmp_path: Path) -> None:
    root = tmp_path / "remote"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "slurm-logs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe remote artifact path"):
        RemoteSlurmTransport(kind="local-slurm", ssh_target=None).ensure_remote_directory(root, Path("slurm-logs/step"))

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("target_kind", ["symlink", "hardlink"])
def test_verified_stage_rejects_unsafe_existing_final_file(tmp_path: Path, target_kind: str) -> None:
    root = tmp_path / "remote"
    target = root / "attempt" / "result.json"
    target.parent.mkdir(parents=True)
    source = tmp_path / "artifact"
    source.write_bytes(b"governed")
    if target_kind == "symlink":
        target.symlink_to(source)
    else:
        os.link(source, target)

    with pytest.raises(ValueError, match="unsafe remote artifact path"):
        RemoteSlurmTransport(kind="local-slurm", ssh_target=None).stage_verified_artifact(
            source,
            target,
            remote_root=root,
            expected_sha256="c86ab019a28d89b95a9ad43a43cb80f6b5fe60c61410aa52748624117a7f7899",
        )


def test_stable_fetch_distinguishes_missing_from_unsafe_remote_path(tmp_path: Path) -> None:
    root = tmp_path / "remote"
    root.mkdir()
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None)

    with pytest.raises(FileNotFoundError):
        transport.fetch_stable_artifact(root / "missing.json", remote_root=root, maximum_bytes=1024)

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "result.json").write_bytes(b"result")
    (root / "attempt").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe remote artifact path"):
        transport.fetch_stable_artifact(root / "attempt" / "result.json", remote_root=root, maximum_bytes=1024)


@pytest.mark.parametrize("target_kind", ["symlink", "hardlink"])
def test_stable_fetch_rejects_unsafe_final_file(tmp_path: Path, target_kind: str) -> None:
    root = tmp_path / "remote"
    root.mkdir()
    source = tmp_path / "source"
    source.write_bytes(b"result")
    target = root / "result.json"
    if target_kind == "symlink":
        target.symlink_to(source)
    else:
        os.link(source, target)

    with pytest.raises(ValueError, match="unsafe remote artifact path"):
        RemoteSlurmTransport(kind="local-slurm", ssh_target=None).fetch_stable_artifact(
            target, remote_root=root, maximum_bytes=1024
        )


def test_streamed_stage_detects_remote_root_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bspp.orchestration.control import transport as transport_module

    root = tmp_path / "remote"
    moved = tmp_path / "moved"
    source = tmp_path / "source"
    source.write_bytes(b"governed")
    injected = transport_module._REMOTE_ARTIFACT_PROGRAM.replace(
        "finally:\n    revalidate_root(args['root'], root_fd)",
        "finally:\n"
        "    if operation == 'publish-input':\n"
        f"        os.rename(args['root'], {str(moved)!r}); os.mkdir(args['root'], 0o700)\n"
        "    revalidate_root(args['root'], root_fd)",
    )
    monkeypatch.setattr(transport_module, "_REMOTE_ARTIFACT_PROGRAM", injected)

    with pytest.raises(ValueError, match="root replaced"):
        RemoteSlurmTransport(kind="local-slurm", ssh_target=None).stage_verified_artifact(
            source,
            root / "attempt/result.json",
            remote_root=root,
            expected_sha256="c86ab019a28d89b95a9ad43a43cb80f6b5fe60c61410aa52748624117a7f7899",
        )

    assert not (root / "attempt/result.json").exists()


def test_runtime_qualification_job_reconciliation_is_exact_and_rejects_ambiguity() -> None:
    runner = RecordingRunner([CommandResult(argv=(), returncode=0, stdout="4242\n", stderr="")])
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)
    assert (
        transport.find_runtime_qualification_job(attempt_token="a" * 32, profile="example-cluster", owner="tester")
        == "4242"
    )
    assert runner.calls == [
        (
            "squeue",
            "--noheader",
            "--user=tester",
            "--name=bspp_rq_example-cluster_" + "a" * 32,
            "--format=%A",
        )
    ]
    ambiguous = RecordingRunner([CommandResult(argv=(), returncode=0, stdout="4242\n4243\n", stderr="")])
    with pytest.raises(ValueError, match="ambiguous"):
        RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=ambiguous).find_runtime_qualification_job(
            attempt_token="a" * 32, profile="example-cluster", owner="tester"
        )


def test_runtime_qualification_reconciliation_finds_completed_base_job_in_accounting() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout="", stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=f"4242|bspp_rq_example-cluster_{'a' * 32}|tester|COMPLETED\n",
                stderr="",
            ),
        ]
    )
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)
    assert (
        transport.find_runtime_qualification_job(attempt_token="a" * 32, profile="example-cluster", owner="tester")
        == "4242"
    )
    assert runner.calls[1][0] == "sacct"


@pytest.mark.parametrize(
    "accounting",
    [
        "unexpected parser output\n",
        f"4242.batch|bspp_rq_example-cluster_{'a' * 32}|tester|COMPLETED\n",
        f"4242_7|bspp_rq_example-cluster_{'a' * 32}|tester|COMPLETED\n",
        "4242|wrong-name|tester|COMPLETED\n",
    ],
)
def test_runtime_qualification_accounting_reconciliation_fails_closed(accounting: str) -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout="", stderr=""),
            CommandResult(argv=(), returncode=0, stdout=accounting, stderr=""),
        ]
    )
    with pytest.raises(ValueError, match=r"malformed|non-base"):
        RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner).find_runtime_qualification_job(
            attempt_token="a" * 32, profile="example-cluster", owner="tester"
        )


def test_remote_slurm_transport_inspects_squeue_and_sacct_and_normalizes_states() -> None:
    runner = RecordingRunner(
        [
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {"job_id": 1001, "job_state": ["RUNNING"]},
                            {"job_id": 1002, "job_state": ["PENDING"]},
                        ]
                    }
                ),
                stderr="",
            ),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {"job_id_raw": "1002", "state": "COMPLETED batch", "exit_code": "0:0"},
                            {"job_id_raw": "1003", "state": "OUT_OF_MEMORY", "exit_code": "0:9"},
                        ]
                    }
                ),
                stderr="",
            ),
        ]
    )
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    observation = transport.query_observation(("1001", "1002", "1003"))
    states = observation.selected_states

    assert [(state.job_id, state.state, state.source) for state in states] == [
        ("1001", "RUNNING", "squeue"),
        ("1002", "COMPLETED", "sacct"),
        ("1003", "OUT_OF_MEMORY", "sacct"),
    ]
    assert observation.squeue.parser == "json"
    assert observation.sacct.parser == "json"
    assert observation.squeue.raw_json == {
        "jobs": [
            {"job_id": 1001, "job_state": ["RUNNING"]},
            {"job_id": 1002, "job_state": ["PENDING"]},
        ]
    }
    assert runner.calls == [
        ("squeue", "--json", "-j", "1001,1002,1003"),
        (
            "sacct",
            "--json",
            "-j",
            "1001,1002,1003",
            "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem,Restarts",
        ),
    ]


def test_remote_slurm_transport_falls_back_when_json_is_unavailable() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=1, stdout="", stderr="unsupported option --json"),
            CommandResult(argv=(), returncode=0, stdout="1001|running\n", stderr=""),
            CommandResult(argv=(), returncode=0, stdout="not-json\n", stderr=""),
            CommandResult(argv=(), returncode=0, stdout="1001|COMPLETED batch\n", stderr=""),
        ]
    )
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    observation = transport.query_observation(("1001",))

    assert [(state.job_id, state.state, state.source) for state in observation.selected_states] == [
        ("1001", "COMPLETED", "sacct")
    ]
    assert observation.squeue.parser == "parsable-fallback"
    assert observation.sacct.parser == "parsable-fallback"
    assert observation.warnings == (
        "squeue_json_unavailable: unsupported option --json",
        "sacct_json_unavailable: sacct --json returned invalid JSON: Expecting value",
    )
    assert runner.calls == [
        ("squeue", "--json", "-j", "1001"),
        ("squeue", "-h", "-j", "1001", "-o", "%i|%T"),
        ("sacct", "--json", "-j", "1001", "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem,Restarts"),
        ("sacct", "-j", "1001", "--format=JobIDRaw,State", "--noheader", "--parsable2"),
    ]


def test_best_effort_observation_retains_sacct_when_squeue_is_unavailable() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=1, stdout="", stderr="json unsupported"),
            CommandResult(argv=(), returncode=2, stdout="", stderr="queue unavailable"),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps({"jobs": [{"job_id_raw": "1001", "state": "COMPLETED", "exit_code": "0:0"}]}),
                stderr="",
            ),
        ]
    )
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    observation = transport.query_observation_best_effort(("1001",))

    assert observation.squeue.parser == "unavailable"
    assert observation.squeue.returncode == 2
    assert observation.squeue.argv == ("squeue", "-h", "-j", "1001", "-o", "%i|%T")
    assert observation.sacct.parser == "json"
    assert [(state.state, state.source, state.exit_code) for state in observation.selected_states] == [
        ("COMPLETED", "sacct", "0:0")
    ]
    assert observation.warnings == (
        "squeue_unavailable: queue unavailable (squeue_json_unavailable: json unsupported)",
    )
    assert [call[0] for call in runner.calls] == ["squeue", "squeue", "sacct"]


def test_best_effort_observation_distinguishes_malformed_fallback_from_empty_success() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout="not-json", stderr=""),
            CommandResult(argv=(), returncode=0, stdout="malformed row\n", stderr=""),
            CommandResult(argv=(), returncode=0, stdout='{"jobs": []}', stderr=""),
        ]
    )
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    observation = transport.query_observation_best_effort(("1001",))

    assert observation.squeue.parser == "unavailable"
    assert observation.squeue.returncode == 0
    assert observation.squeue.raw_text == "malformed row\n"
    assert observation.sacct.parser == "json"
    assert observation.sacct_jobs == ()
    assert observation.selected_states == ()
    assert "malformed row at line 1" in observation.warnings[0]


def test_best_effort_observation_captures_oserror_and_still_queries_other_source() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        if argv[0] == "squeue":
            raise OSError("transport channel failed")
        return CommandResult(argv=argv, returncode=0, stdout='{"jobs": []}', stderr="")

    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    observation = transport.query_observation_best_effort(("1001",))

    assert observation.squeue.parser == "unavailable"
    assert observation.squeue.returncode == -1
    assert observation.squeue.argv == ("squeue", "--json", "-j", "1001")
    assert observation.sacct.parser == "json"
    assert [call[0] for call in calls] == ["squeue", "sacct"]


def test_best_effort_successful_empty_sources_remain_available() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout='{"jobs": []}', stderr=""),
            CommandResult(argv=(), returncode=0, stdout='{"jobs": []}', stderr=""),
        ]
    )
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    observation = transport.query_observation_best_effort(("1001",))

    assert observation.squeue.parser == "json"
    assert observation.sacct.parser == "json"
    assert observation.warnings == ()
    assert observation.selected_states == ()


def test_legacy_observation_remains_fail_closed_on_final_fallback_failure() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=1, stdout="", stderr="json unsupported"),
            CommandResult(argv=(), returncode=2, stdout="", stderr="queue unavailable"),
        ]
    )
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    with pytest.raises(ValueError, match="queue unavailable"):
        transport.query_observation(("1001",))

    assert [call[0] for call in runner.calls] == ["squeue", "squeue"]


def test_remote_slurm_transport_cancels_only_given_job_ids() -> None:
    runner = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout="", stderr=""),
            CommandResult(argv=(), returncode=0, stdout="", stderr=""),
        ]
    )
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    transport.cancel_jobs(("1001", "1003"))

    assert runner.calls == [("scancel", "1001"), ("scancel", "1003")]


def test_normalize_slurm_state_uppercases_and_strips_reasons() -> None:
    assert normalize_slurm_state("cancelled by 1234") == "CANCELLED"
