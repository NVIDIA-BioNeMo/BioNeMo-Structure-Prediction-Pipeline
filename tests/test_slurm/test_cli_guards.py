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

"""CLI regression guards for the SLURM submission commands.

These tests exist because the adversarial review flagged two silent
failure modes in the earlier revision:

- ``slurm submit`` / ``slurm resubmit-failed`` accepted ``--manifest-csv``
  as optional, but the ``process`` worker each array task runs exits 1
  when no manifest is given. That enqueued arrays which then failed at
  runtime — far more expensive than failing at submission.
- ``make_executor`` could silently fall back to ``LocalExecutor`` when
  the host had no ``sbatch`` on PATH, running SLURM-tagged jobs on the
  caller's machine without the declared resources.

The first is caught here via Click's CliRunner. The second is caught in
``test_executor.py::test_make_executor_defaults_to_slurm_cluster``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from bspp.orchestration.runtime.cli import cli


def _write_recipe(tmp_path: Path) -> Path:
    recipe = tmp_path / "ds"
    recipe.mkdir()
    config = {
        "job_name": "bspp_ds",
        "run_name": "ds",
        "slurm": {
            "partition": "batch_singlenode",
            "account": "hc",
            "cpus_per_task": 4,
            "memory": "16G",
            "gres": "gpu:1",
            "time": "01:00:00",
        },
        "paths": {"log_dir": str(tmp_path / "slurm_logs")},
    }
    (recipe / "config.yaml").write_text(yaml.safe_dump(config))
    return recipe


def _write_shard_config(dataset_dir: Path, total: int, required: int) -> None:
    import json

    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "shard_config.json").write_text(json.dumps({"total_models": total, "required_shards": required}))


def test_slurm_submit_rejects_missing_manifest(tmp_path: Path) -> None:
    runner = CliRunner()
    recipe = _write_recipe(tmp_path)
    input_dir = tmp_path / "in"
    input_dir.mkdir()

    result = runner.invoke(
        cli,
        [
            "slurm",
            "submit",
            "--recipe-dir",
            str(recipe),
            "--array",
            "0-2",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "manifest-csv" in result.output.lower() or "manifest_csv" in result.output.lower()


def test_slurm_resubmit_failed_rejects_missing_manifest(tmp_path: Path) -> None:
    runner = CliRunner()
    recipe = _write_recipe(tmp_path)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_base = tmp_path / "out"
    _write_shard_config(output_base / "ds", total=4, required=2)

    result = runner.invoke(
        cli,
        [
            "slurm",
            "resubmit-failed",
            "--recipe-dir",
            str(recipe),
            "--dataset",
            "ds",
            "--output-base",
            str(output_base),
            "--input-dir",
            str(input_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "manifest-csv" in result.output.lower() or "manifest_csv" in result.output.lower()
