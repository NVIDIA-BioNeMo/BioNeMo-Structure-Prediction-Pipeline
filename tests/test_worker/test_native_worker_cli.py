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

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.worker import TaskContext, WorkerDependencies, WorkerResult


def test_worker_archive_task_cli_builds_task_context_from_slurm_env(monkeypatch, tmp_path: Path) -> None:
    spec_path = tmp_path / "runspec.yaml"
    spec_path.write_text("dataset: {}\n")
    loaded_specs: list[Path] = []
    seen: list[tuple[object, TaskContext, WorkerDependencies]] = []
    spec = object()

    def fake_load_runspec(path: Path) -> object:
        loaded_specs.append(path)
        return spec

    def fake_run_archive_task(got_spec: object, task: TaskContext, deps: WorkerDependencies) -> WorkerResult:
        seen.append((got_spec, task, deps))
        return WorkerResult(
            archive_name="archive.tar.lz4",
            logical_shards=(4, 5),
            processed_models=12,
            failed_models=("AF-1",),
            uploaded_files=6,
            exit_code=0,
        )

    monkeypatch.setattr("bspp.orchestration.contract.runspec.load_runspec", fake_load_runspec)
    monkeypatch.setattr("bspp.orchestration.runtime.worker.run_archive_task", fake_run_archive_task)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["worker", "archive-task", "--spec", str(spec_path), "--dry-run"],
        env={
            "SLURM_ARRAY_TASK_ID": "7",
            "SLURM_ARRAY_TASK_COUNT": "20",
            "SLURM_JOB_ID": "job-123",
            "SLURMD_NODENAME": "node-a",
        },
    )

    assert result.exit_code == 0
    assert loaded_specs == [spec_path]
    assert len(seen) == 1
    assert seen[0][0] is spec
    assert seen[0][1] == TaskContext(
        job_id="job-123",
        array_task_id=7,
        array_task_count=20,
        node_name="node-a",
        dry_run=True,
    )
    assert isinstance(seen[0][2], WorkerDependencies)
    assert "archive=archive.tar.lz4 shards=4,5 processed=12 failed=1 uploaded=6" in result.output


def test_worker_archive_task_cli_propagates_worker_exit_code(monkeypatch, tmp_path: Path) -> None:
    spec_path = tmp_path / "runspec.yaml"
    spec_path.write_text("dataset: {}\n")

    monkeypatch.setattr("bspp.orchestration.contract.runspec.load_runspec", lambda _path: object())
    monkeypatch.setattr(
        "bspp.orchestration.runtime.worker.run_archive_task",
        lambda _spec, _task, _deps: WorkerResult(
            archive_name="archive.tar.lz4",
            logical_shards=(),
            processed_models=0,
            failed_models=(),
            uploaded_files=0,
            exit_code=3,
        ),
    )

    result = CliRunner().invoke(
        cli,
        ["worker", "archive-task", "--runspec", str(spec_path)],
        env={"SLURM_ARRAY_TASK_ID": "0"},
    )

    assert result.exit_code == 3


def test_worker_archive_task_cli_rejects_missing_or_invalid_task_id(tmp_path: Path) -> None:
    spec_path = tmp_path / "runspec.yaml"
    spec_path.write_text("dataset: {}\n")
    runner = CliRunner()

    missing = runner.invoke(cli, ["worker", "archive-task", "--spec", str(spec_path)])
    invalid = runner.invoke(
        cli,
        ["worker", "archive-task", "--spec", str(spec_path)],
        env={"SLURM_ARRAY_TASK_ID": "not-an-int"},
    )

    assert missing.exit_code != 0
    assert "SLURM_ARRAY_TASK_ID is required" in missing.output
    assert invalid.exit_code != 0
    assert "SLURM_ARRAY_TASK_ID must be an integer" in invalid.output
