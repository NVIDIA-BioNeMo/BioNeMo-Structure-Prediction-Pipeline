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

"""Tests for slurm/executor config mapping.

The ``_fake_slurm_affinity`` fixture in ``conftest.py`` bypasses
submitit's SlurmExecutor construction guard so we can exercise the
SLURM path without installing SLURM locally.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from bspp.orchestration.runtime.slurm.arrays import array_parallelism
from bspp.orchestration.runtime.slurm.executor import _time_to_minutes, make_executor


def test_time_to_minutes_hours_minutes_seconds() -> None:
    assert _time_to_minutes("04:00:00") == 240
    assert _time_to_minutes("00:15:30") == 15
    assert _time_to_minutes("2-00:00:00") == 2 * 24 * 60


def test_time_to_minutes_accepts_int() -> None:
    assert _time_to_minutes(120) == 120


def test_time_to_minutes_none_defaults_to_60() -> None:
    assert _time_to_minutes(None) == 60


def test_parse_array_parallelism_extracts_percent() -> None:
    assert array_parallelism("0-99%16") == 16
    assert array_parallelism("0-99") == 0
    assert array_parallelism(None) == 0


def _write_recipe(tmp_path: Path, overrides: dict) -> Path:
    recipe_dir = tmp_path / "ds"
    recipe_dir.mkdir()
    config = {
        "job_name": "bspp_ds",
        "run_name": "ds",
        "slurm": {
            "partition": "batch_singlenode",
            "account": "example-account",
            "cpus_per_task": 30,
            "memory": "128G",
            "gres": "gpu:1",
            "time": "04:00:00",
            "array_range": "0-99%8",
        },
        "paths": {},
    }
    for k, v in overrides.items():
        if isinstance(v, dict):
            config[k] = {**config.get(k, {}), **v}
        else:
            config[k] = v
    (recipe_dir / "config.yaml").write_text(yaml.safe_dump(config))
    return recipe_dir


def test_make_executor_maps_recipe_fields(tmp_path: Path) -> None:
    recipe_dir = _write_recipe(tmp_path, {})

    executor = make_executor(recipe_dir, cluster="slurm")

    # submitit's SlurmExecutor translates the canonical params to its own keys
    # ("name" -> "job_name", "timeout_min" -> "time", etc.); assert post-translation.
    params = executor._executor.parameters
    assert params["job_name"] == "bspp_ds"
    assert params["time"] == 240
    assert params["partition"] == "batch_singlenode"
    assert params["gres"] == "gpu:1"
    assert params["mem"] == "128G"
    assert params["cpus_per_task"] == 30
    assert params["array_parallelism"] == 8
    assert params["additional_parameters"] == {"account": "example-account"}


def test_make_executor_respects_explicit_log_folder(tmp_path: Path) -> None:
    recipe_dir = _write_recipe(tmp_path, {})
    log_dir = tmp_path / "custom_logs"

    executor = make_executor(recipe_dir, log_folder=log_dir, cluster="slurm")

    assert Path(executor.folder).resolve() == log_dir.resolve()


def test_make_executor_without_array_parallelism(tmp_path: Path) -> None:
    recipe_dir = _write_recipe(tmp_path, {"slurm": {"array_range": "0-99"}})

    executor = make_executor(recipe_dir, cluster="slurm")

    assert "array_parallelism" not in executor._executor.parameters


def test_make_executor_defaults_to_slurm_cluster(tmp_path: Path) -> None:
    """Guard: submitting a 'slurm' command must not silently downgrade to local.

    Regression guard for the silent-LocalExecutor-fallback path: if someone
    changes the default to ``cluster=None``/auto-detect, this test breaks
    because calling make_executor without a SLURM install should surface
    SlurmExecutor (via our monkeypatched affinity fixture) rather than
    LocalExecutor.
    """
    import submitit

    recipe_dir = _write_recipe(tmp_path, {})

    executor = make_executor(recipe_dir)  # no explicit cluster= argument

    assert isinstance(executor._executor, submitit.SlurmExecutor)
