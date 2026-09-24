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

"""Tests for slurm/submit and the resubmit wrapper."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from bspp.orchestration.runtime.slurm.resubmit import resubmit_failed_shards
from bspp.orchestration.runtime.slurm.submit import run_shard, submit_array


def _write_recipe(tmp_path: Path) -> Path:
    recipe_dir = tmp_path / "ds"
    recipe_dir.mkdir()
    config = {
        "job_name": "bspp_ds",
        "run_name": "ds",
        "slurm": {
            "partition": "batch_singlenode",
            "account": "example-account",
            "cpus_per_task": 4,
            "memory": "16G",
            "gres": "gpu:1",
            "time": "01:00:00",
        },
        "paths": {"log_dir": str(tmp_path / "slurm_logs")},
    }
    (recipe_dir / "config.yaml").write_text(yaml.safe_dump(config))
    return recipe_dir


def test_submit_array_dry_run_returns_preview(tmp_path: Path) -> None:
    recipe_dir = _write_recipe(tmp_path)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    summary = submit_array(
        recipe_dir,
        [0, 1, 2],
        input_dir=input_dir,
        output_dir=output_dir,
        dry_run=True,
    )

    assert summary.dry_run is True
    assert summary.shard_ids == (0, 1, 2)
    assert summary.argv_preview[0] == "bspp-orchestration-runtime"
    assert "--shard-id" in summary.argv_preview
    assert "0" in summary.argv_preview


def test_submit_array_rejects_empty_shards(tmp_path: Path) -> None:
    recipe_dir = _write_recipe(tmp_path)
    with pytest.raises(ValueError):
        submit_array(
            recipe_dir,
            [],
            input_dir=tmp_path,
            output_dir=tmp_path,
            dry_run=True,
        )


def test_run_shard_invokes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def fake_run(argv, check):  # type: ignore[no-untyped-def]
        captured.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = run_shard(
        3,
        "/r",
        "/in",
        "/out",
        "/man.csv",
        "/up.duckdb",
        "ipsae dssp",
        8,
        "bspp-orchestration-runtime",
    )

    assert rc == 0
    assert captured == [
        [
            "bspp-orchestration-runtime",
            "process",
            "--shard-id",
            "3",
            "--input-dir",
            "/in",
            "--output-dir",
            "/out",
            "--stages",
            "ipsae dssp",
            "--workers",
            "8",
            "--manifest-csv",
            "/man.csv",
            "--uniprot-db",
            "/up.duckdb",
        ]
    ]


def test_run_shard_callable_positionally_via_delayed_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard against the keyword-only regression.

    submitit's map_array wraps each tuple of iterables in a
    DelayedSubmission that invokes ``fn(*args)`` at shard-run time. If
    run_shard's parameters become keyword-only, this call raises
    TypeError. Exercise that call path directly so the regression is
    caught without a live SLURM cluster.
    """
    from submitit.core.utils import DelayedSubmission

    def fake_run(argv, check):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    delayed = DelayedSubmission(
        run_shard,
        7,
        "/recipe",
        "/in",
        "/out",
        "/m.csv",
        None,
        "stage_a",
        4,
        "bspp-orchestration-runtime",
    )
    # result() forces DelayedSubmission to call run_shard(*args). If the
    # signature regresses to keyword-only, this raises TypeError.
    assert delayed.result() == 0


def _write_shard_config(output_dir: Path, total_models: int, required_shards: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shard_config.json").write_text(
        json.dumps({"total_models": total_models, "required_shards": required_shards})
    )


def test_resubmit_failed_shards_empty_case(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "out" / "ds"
    _write_shard_config(dataset_dir, 2, 2)
    for sid in (0, 1):
        success = dataset_dir / f"shard_{sid}" / "success_outputs"
        success.mkdir(parents=True)
        (success / "m.pdb").write_text("x")

    recipe_dir = _write_recipe(tmp_path)

    failed, summary = resubmit_failed_shards(
        recipe_dir,
        dataset="ds",
        output_base=tmp_path / "out",
        input_dir=tmp_path,
        dry_run=True,
    )

    assert failed == []
    assert summary is None


def test_resubmit_failed_shards_detects_and_dry_runs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "out" / "ds"
    _write_shard_config(dataset_dir, 2, 2)
    # shard 0: OK. shard 1: no outputs + no uploaded marker -> failed.
    success0 = dataset_dir / "shard_0" / "success_outputs"
    success0.mkdir(parents=True)
    (success0 / "m.pdb").write_text("x")
    (dataset_dir / "shard_1" / "success_outputs").mkdir(parents=True)

    recipe_dir = _write_recipe(tmp_path)

    failed, summary = resubmit_failed_shards(
        recipe_dir,
        dataset="ds",
        output_base=tmp_path / "out",
        input_dir=tmp_path,
        dry_run=True,
    )

    assert failed == [1]
    assert summary is not None
    assert summary.dry_run is True
    assert summary.shard_ids == (1,)
