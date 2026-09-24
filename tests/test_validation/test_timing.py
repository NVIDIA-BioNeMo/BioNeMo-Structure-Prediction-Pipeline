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

"""Tests for validation/timing."""

from __future__ import annotations

import json
from pathlib import Path

from bspp.orchestration.runtime.validation.timing import (
    aggregate,
    load_shard_results,
    render_timing_summary,
)


def _write_pipeline_results(output_dir: Path, shard_id: int, payload: dict) -> None:
    shard = output_dir / f"shard_{shard_id}"
    shard.mkdir(parents=True, exist_ok=True)
    (shard / "pipeline_results.json").write_text(json.dumps(payload))


def test_load_shard_results_skips_bad_json(tmp_path: Path) -> None:
    _write_pipeline_results(tmp_path, 0, {"stage_timings": {"stage_01": 1.0}})
    bad = tmp_path / "shard_1"
    bad.mkdir()
    (bad / "pipeline_results.json").write_text("{not json")

    results = load_shard_results(tmp_path)

    assert set(results) == {0}


def test_aggregate_computes_stats_and_bottleneck() -> None:
    results = {
        0: {"stage_timings": {"alpha": 10.0, "beta": 2.0}, "total_duration": 12.0},
        1: {"stage_timings": {"alpha": 30.0, "beta": 4.0}, "total_duration": 34.0},
    }

    summary = aggregate(results)

    assert summary.num_shards == 2
    assert summary.bottleneck_stage == "alpha"
    alpha = next(s for s in summary.stage_stats if s.name == "alpha")
    assert alpha.min_s == 10.0
    assert alpha.max_s == 30.0
    assert alpha.avg_s == 20.0
    assert summary.max_wall_time_s == 34.0
    assert summary.min_wall_time_s == 12.0
    assert summary.imbalance_pct > 0


def test_aggregate_handles_top_level_stage_keys() -> None:
    results = {
        0: {
            "stage_01_prep": {"duration": 5.0},
            "stage_02_run": 15.0,
            "total_duration": 20.0,
        }
    }

    summary = aggregate(results)

    stage_names = {s.name for s in summary.stage_stats}
    assert stage_names == {"stage_01_prep", "stage_02_run"}


def test_render_timing_summary_empty() -> None:
    assert render_timing_summary(aggregate({})) == "No shard pipeline_results.json files found."
