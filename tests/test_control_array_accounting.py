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

"""Sanitized Slurm array capture; synthetic edge cases remain separate from evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from bspp.orchestration.control.monitoring import parse_sacct_identity_parsable_rows, parse_sacct_json
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore
from bspp.orchestration.control.phase_materialization import materialize_phase
from bspp.orchestration.control.phase_resume import _complete_fold_array_task_set, resume_phase
from bspp.orchestration.control.phase_submission import submit_phase
from bspp.orchestration.control.transport import CommandResult, RemoteSlurmTransport, default_command_runner
from tests.test_phase_folding_cancel_retry import _tree_bytes, _write_packed_profile
from tests.test_phase_folding_lifecycle import FIXED_RUN_ID, FIXED_TIME, _phase_plan

FIXTURES = Path(__file__).parent / "fixtures/slurm/slurm-array-100000001"
PARENT = "100000001"
ENDPOINTS = ("100000001_0", "100000001_1")


def test_sanitized_snapshots_decode_both_actual_children_without_inventing_a_parent() -> None:
    raw_json = (FIXTURES / "sacct.json").read_bytes()
    raw_text = (FIXTURES / "sacct.psv").read_bytes()
    assert hashlib.sha256(raw_json).hexdigest() == "3cfbc68bb69070f2188b01b5e0983da8a8b5bfb4c080dcc14ab5506d5eddd26a"
    assert hashlib.sha256(raw_text).hexdigest() == "58a1439cc3d92c366bf9f635ceac31ac8cea128d1035f818ff877121479542cb"
    records = parse_sacct_json(raw_json.decode(), requested=(PARENT,), strict_rows=True)
    assert [(r.job_id, r.requested_job_id, r.state, r.exit_code) for r in records] == [
        (endpoint, PARENT, "FAILED", None) for endpoint in ENDPOINTS
    ]
    assert [r.raw for r in records] == json.loads(raw_json)["jobs"]
    fallback = parse_sacct_identity_parsable_rows(raw_text.decode(), requested=(PARENT,))
    assert [r.raw for r in fallback] == raw_text.decode().splitlines()
    assert [r.job_id for r in fallback if "." not in r.job_id] == list(ENDPOINTS)
    assert all(r.requested_job_id == PARENT and r.restarts is None for r in fallback)
    tasks = _complete_fold_array_task_set(PARENT, (0, 1), fallback)
    assert tasks is not None
    assert [(t.task_index, t.state, t.exit_code) for t in tasks] == [(0, "FAILED", "15:0"), (1, "FAILED", "15:0")]


class AccountingRunner:
    """Replay the sanitized scheduler fixture; never execute any command."""

    def __init__(self, text: str | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.text = text if text is not None else (FIXTURES / "sacct.psv").read_text()

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        assert argv[0] in {"squeue", "sacct"}, argv
        if argv[0] == "squeue" or argv[argv.index("-j") + 1] != PARENT:
            return CommandResult(argv, 0, '{"jobs": []}', "")
        if any(arg.endswith(",Restarts") for arg in argv):
            return CommandResult(argv, 1, "", (FIXTURES / "unsupported-restarts.stderr").read_text())
        stdout = (FIXTURES / "sacct.json").read_text() if "--json" in argv else self.text
        return CommandResult(argv, 0, stdout, "")


def test_real_transport_recovers_exact_failed_exits_after_unsupported_restarts() -> None:
    runner = AccountingRunner()
    observation = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner).query_observation(
        (PARENT,),
        require_exact_terminal_exit=True,
        expected_terminal_job_ids=ENDPOINTS,
    )
    assert observation.sacct.parser == "parsable-fallback"
    assert observation.sacct.raw_text == (FIXTURES / "sacct.psv").read_text()
    tasks = _complete_fold_array_task_set(PARENT, (0, 1), observation.sacct_jobs)
    assert tasks is not None and [task.exit_code for task in tasks] == ["15:0", "15:0"]
    assert any("lacks an exact exit code" in warning for warning in observation.warnings)
    assert not any("sacct_unavailable" in warning for warning in observation.warnings)
    assert len(runner.calls) == 5
    assert all(record.restarts is None for record in observation.sacct_jobs)


@pytest.mark.parametrize(
    "rows",
    [
        "100000002|100000001_0|COMPLETED|0:0\n",
        "100000002|100000001_0|COMPLETED|0:0\n" * 2 + "100000001|100000001_1|COMPLETED|0:0\n",
    ],
)
def test_synthetic_missing_or_duplicate_endpoint_never_proves_complete_success(rows: str) -> None:
    observation = RemoteSlurmTransport(
        kind="local-slurm", ssh_target=None, runner=AccountingRunner(rows)
    ).query_observation(
        (PARENT,),
        require_exact_terminal_exit=True,
        expected_terminal_job_ids=ENDPOINTS,
    )
    assert _complete_fold_array_task_set(PARENT, (0, 1), observation.sacct_jobs) is None


def test_synthetic_complete_success_has_only_real_explicit_endpoint_rows() -> None:
    rows = "100000002|100000001_0|COMPLETED|0:0\n100000001|100000001_1|COMPLETED|0:0\n"
    records = parse_sacct_identity_parsable_rows(rows, requested=(PARENT,))
    tasks = _complete_fold_array_task_set(PARENT, (0, 1), records)
    assert tasks is not None and all(task.state == "COMPLETED" and task.exit_code == "0:0" for task in tasks)
    assert [record.job_id for record in records] == list(ENDPOINTS)


@pytest.mark.parametrize(
    "row",
    [
        "100000002|999999_0|FAILED|15:0",
        "100000002_0|100000001_0|FAILED|15:0",
        "100000002.batch|100000001_0.extern|FAILED|15:0",
        "100000002|100000001_[0-1]|FAILED|15:0",
        "100000002|100000001_0+|FAILED|15:0",
        "100000002|100000001_0|RUNNING|0:0",
        "100000002|100000001_0|FAILED|15",
        "100000002|100000001_0|FAILED|",
    ],
)
def test_new_numeric_alias_case_keeps_strict_identity_and_exit_guards(row: str) -> None:
    with pytest.raises(ValueError, match="malformed sacct"):
        parse_sacct_identity_parsable_rows(row, requested=(PARENT,))


@pytest.mark.parametrize(
    "second",
    [
        "100000002|100000001_1|FAILED|15:0",
        "18929608|100000001_0|FAILED|15:0",
        "100000002.batch|100000001_1.batch|FAILED|15:0",
    ],
)
def test_conflicting_numeric_aliases_are_rejected(second: str) -> None:
    with pytest.raises(ValueError, match="conflicting sacct array identity"):
        parse_sacct_identity_parsable_rows("100000002|100000001_0|FAILED|15:0\n" + second, requested=(PARENT,))


@pytest.mark.parametrize(
    "value",
    [
        True,
        -1,
        0.5,
        "x",
        "4294967294",
        {"set": True, "infinite": True, "number": 0},
        {"set": True, "number": 0},
        {"set": True, "infinite": "false", "number": 0},
        {"set": True, "infinite": False, "number": True},
        {"number": 0},
    ],
)
def test_json_rejects_nonconcrete_task_identity(value: object) -> None:
    payload = json.loads((FIXTURES / "sacct.json").read_text())
    payload["jobs"][0]["array"]["task_id"] = value
    with pytest.raises(ValueError, match="malformed sacct JSON"):
        parse_sacct_json(json.dumps(payload), requested=(PARENT,), strict_rows=True)


def test_json_unset_task_does_not_invent_zero_or_relabel_foreign_allocation() -> None:
    payload = json.loads((FIXTURES / "sacct.json").read_text())
    for job in payload["jobs"]:
        job["array"]["task_id"]["set"] = False
    records = parse_sacct_json(json.dumps(payload), requested=(PARENT,))
    assert [record.job_id for record in records] == [PARENT]
    assert _complete_fold_array_task_set(PARENT, (0, 1), records) is None


def test_json_explicit_raw_identity_must_agree_with_nested_array() -> None:
    payload = json.loads((FIXTURES / "sacct.json").read_text())
    payload["jobs"][0]["job_id_raw"] = "100000001_1"
    with pytest.raises(ValueError, match="malformed sacct JSON"):
        parse_sacct_json(json.dumps(payload), requested=(PARENT,), strict_rows=True)


def test_json_conflicting_physical_binding_fails_closed() -> None:
    payload = json.loads((FIXTURES / "sacct.json").read_text())
    payload["jobs"][1]["job_id"] = payload["jobs"][0]["job_id"]
    with pytest.raises(ValueError, match="conflicting sacct array identity"):
        parse_sacct_json(json.dumps(payload), requested=(PARENT,), strict_rows=True)


@pytest.mark.parametrize("parent", [True, -1, 1.5, "00", "bad", {"set": False, "number": 100000001}])
def test_json_invalid_nested_parent_cannot_bind_a_child(parent: object) -> None:
    payload = json.loads((FIXTURES / "sacct.json").read_text())
    payload["jobs"][0]["array"]["job_id"] = parent
    with pytest.raises(ValueError, match="malformed sacct JSON"):
        parse_sacct_json(json.dumps(payload), requested=(PARENT,), strict_rows=True)


def test_json_scalar_nonarray_metadata_and_explicit_logical_ids_stay_supported() -> None:
    scalar = {
        "job_id": 1001,
        "array": {"job_id": 0, "task_id": {"set": False, "number": 0}},
        "state": "COMPLETED",
        "exit_code": "0:0",
    }
    records = parse_sacct_json(json.dumps({"jobs": [scalar]}), requested=("1001",), strict_rows=True)
    assert [(r.job_id, r.requested_job_id, r.exit_code) for r in records] == [("1001", "1001", "0:0")]
    payload = json.loads((FIXTURES / "sacct.json").read_text())
    for index, job in enumerate(payload["jobs"]):
        job["job_id_raw"] = ENDPOINTS[index]
    assert [r.job_id for r in parse_sacct_json(json.dumps(payload), requested=(PARENT,), strict_rows=True)] == list(
        ENDPOINTS
    )


class SubmissionRunner:
    """Isolated test submission only; mock scheduler assignments, real local staging."""

    def __init__(self) -> None:
        self.ids = iter(("1001", "1002", "1003", PARENT, "1005"))

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        if argv[0] == "sbatch":
            return CommandResult(argv, 0, next(self.ids) + "\n", "")
        return default_command_runner(argv)


def test_public_resume_records_sanitized_failed_array_once_in_isolated_authority(tmp_path: Path) -> None:
    profile_path = _write_packed_profile(tmp_path)
    profile = yaml.safe_load(profile_path.read_text())
    profile["clusters"]["example-cluster"]["resources"]["gpu_worker"]["nodes"] = 2
    profile_path.write_text(yaml.safe_dump(profile))
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(_phase_plan(tmp_path).to_mapping()))
    authority_root = tmp_path / "authority"
    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    submit_phase(FIXED_RUN_ID, authority_root=authority_root, clock=lambda: FIXED_TIME, runner=SubmissionRunner())
    runner = AccountingRunner()
    result = resume_phase(FIXED_RUN_ID, authority_root=authority_root, clock=lambda: FIXED_TIME, runner=runner)
    assert result.outcome == "failed"
    assert result.terminal_action_ids == ("fold-000001",)
    authority = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    observed = authority.array_terminal_observations
    assert len(observed) == 1 and observed[0].outcome == "failed"
    assert [(task.task_index, task.state, task.exit_code) for task in observed[0].tasks] == [
        (0, "FAILED", "15:0"),
        (1, "FAILED", "15:0"),
    ]
    before = _tree_bytes(authority_root)
    call_count = len(runner.calls)
    with pytest.raises(ValueError, match="cannot cross a failed Phase Attempt boundary"):
        resume_phase(FIXED_RUN_ID, authority_root=authority_root, clock=lambda: FIXED_TIME, runner=runner)
    assert len(runner.calls) == call_count
    assert _tree_bytes(authority_root) == before
    assert all(call[0] in {"sacct", "squeue"} for call in runner.calls)
