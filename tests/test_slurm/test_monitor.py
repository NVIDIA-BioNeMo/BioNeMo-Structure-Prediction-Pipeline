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

"""Tests for direct Slurm monitoring helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pytest

from bspp.orchestration.runtime.slurm.monitor import CommandResult, monitor_job, parse_job_id, submit_sbatch


class FakeRunner:
    """Queue command responses by executable name."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.responses: dict[str, list[CommandResult]] = defaultdict(list)

    def add(self, command: str, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.responses[command].append(CommandResult(returncode=returncode, stdout=stdout, stderr=stderr))

    def __call__(self, argv: Sequence[str]) -> CommandResult:
        call = tuple(argv)
        self.calls.append(call)
        queue = self.responses[call[0]]
        if queue:
            return queue.pop(0)
        return CommandResult(returncode=0, stdout="", stderr="")


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("12345\n", "12345"),
        ("12345;example-cluster\n", "12345"),
        ("12345", "12345"),
    ],
)
def test_parse_job_id(stdout: str, expected: str) -> None:
    assert parse_job_id(stdout) == expected


def test_parse_job_id_rejects_unparsable_output() -> None:
    with pytest.raises(ValueError, match="Could not parse"):
        parse_job_id("Submitted batch job 12345\n")


def test_submit_sbatch_uses_parsable_flag() -> None:
    runner = FakeRunner()
    runner.add("sbatch", stdout="12345;example-cluster\n")

    evidence = submit_sbatch(Path("run_archive.sbatch"), runner=runner)

    assert evidence.success is True
    assert evidence.job_id == "12345"
    assert runner.calls == [("sbatch", "--parsable", "run_archive.sbatch")]


def test_monitor_array_success_without_arraytaskid_or_sacct_array() -> None:
    runner = FakeRunner()
    runner.add("squeue", stdout="")
    runner.add(
        "sacct",
        stdout=("12345_0|12345_0|COMPLETED|0:0|00:01:00|node-a\n12345_1|12345_1|COMPLETED|0:0|00:01:01|node-b\n"),
    )

    result = monitor_job(
        "12345",
        is_array=True,
        expected_task_ids=(0, 1),
        runner=runner,
        sleep=lambda _: None,
        max_polls=2,
    )

    assert result.success is True
    assert result.accounting_complete is True
    assert result.completed_task_ids == (0, 1)
    flattened = " ".join(" ".join(call) for call in runner.calls)
    assert "ArrayTaskID" not in flattened
    assert "--array" not in flattened


def test_monitor_array_failure_state_is_not_success() -> None:
    runner = FakeRunner()
    runner.add("squeue", stdout="")
    runner.add(
        "sacct",
        stdout=("12345_0|12345_0|COMPLETED|0:0|00:01:00|node-a\n12345_1|12345_1|FAILED|1:0|00:00:10|node-b\n"),
    )

    result = monitor_job(
        "12345",
        is_array=True,
        expected_task_ids=(0, 1),
        runner=runner,
        sleep=lambda _: None,
    )

    assert result.success is False
    assert result.accounting_complete is True
    assert result.failure_reason == "terminal_failure"


def test_monitor_array_missing_task_fails_accounting_incomplete() -> None:
    runner = FakeRunner()
    runner.add("squeue", stdout="")
    runner.add("sacct", stdout="12345_0|12345_0|COMPLETED|0:0|00:01:00|node-a\n")
    runner.add("sacct", stdout="12345_0|12345_0|COMPLETED|0:0|00:01:00|node-a\n")

    result = monitor_job(
        "12345",
        is_array=True,
        expected_task_ids=(0, 1),
        runner=runner,
        sleep=lambda _: None,
        sacct_retries=1,
    )

    assert result.success is False
    assert result.accounting_complete is False
    assert result.failure_reason == "accounting_incomplete"


def test_monitor_array_unexpanded_row_is_incomplete_accounting() -> None:
    runner = FakeRunner()
    runner.add("squeue", stdout="")
    runner.add("sacct", stdout="12345_[0-1]|12345_[0-1]|PENDING|0:0|00:00:00|\n")
    runner.add("sacct", stdout="12345_[0-1]|12345_[0-1]|PENDING|0:0|00:00:00|\n")

    result = monitor_job(
        "12345",
        is_array=True,
        expected_task_ids=(0, 1),
        runner=runner,
        sleep=lambda _: None,
        sacct_retries=1,
    )

    assert result.success is False
    assert result.accounting_complete is False
    assert result.failure_reason == "accounting_unexpanded_array_row"


def test_monitor_single_success_and_failure() -> None:
    success_runner = FakeRunner()
    success_runner.add("squeue", stdout="")
    success_runner.add("sacct", stdout="12345|12345|COMPLETED|0:0|00:02:00|node-a\n")

    success = monitor_job("12345", is_array=False, runner=success_runner, sleep=lambda _: None)

    failure_runner = FakeRunner()
    failure_runner.add("squeue", stdout="")
    failure_runner.add("sacct", stdout="12346|12346|TIMEOUT|0:0|01:00:00|node-b\n")

    failure = monitor_job("12346", is_array=False, runner=failure_runner, sleep=lambda _: None)

    assert success.success is True
    assert failure.success is False
    assert failure.failure_reason == "terminal_failure"


def test_monitor_retries_sacct_after_queue_drains() -> None:
    runner = FakeRunner()
    runner.add("squeue", stdout="")
    runner.add("sacct", stdout="")
    runner.add("sacct", stdout="12345|12345|COMPLETED|0:0|00:02:00|node-a\n")

    result = monitor_job(
        "12345",
        is_array=False,
        runner=runner,
        sleep=lambda _: None,
        sacct_retries=2,
    )

    assert result.success is True
    assert [call[0] for call in runner.calls] == ["squeue", "sacct", "sacct"]


def test_monitor_log_tails_are_best_effort(tmp_path: Path) -> None:
    runner = FakeRunner()
    runner.add("squeue", stdout="")
    runner.add("sacct", stdout="12345|12345|COMPLETED|0:0|00:02:00|node-a\n")
    log_path = tmp_path / "native_12345_0.out"
    log_path.write_text("first\nsecond\nthird\n")

    result = monitor_job(
        "12345",
        is_array=False,
        runner=runner,
        sleep=lambda _: None,
        log_patterns=(str(tmp_path / "native_%A_%a.out"), str(tmp_path / "missing_%j.err")),
    )

    assert result.success is True
    assert result.log_tails == {str(log_path): "first\nsecond\nthird\n"}
