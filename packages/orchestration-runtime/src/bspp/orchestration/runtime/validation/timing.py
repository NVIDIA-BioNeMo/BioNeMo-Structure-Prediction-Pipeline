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

"""Aggregate per-stage pipeline timing across shards.

Port of ``slurm-scaling/tools/aggregate_timing.py``. Reads
``shard_N/pipeline_results.json`` files and computes per-stage
min/avg/max plus shard-level wall-time imbalance.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageTiming:
    """Aggregate timing for one pipeline stage across shards."""

    name: str
    min_s: float
    avg_s: float
    max_s: float
    shards: int


@dataclass(frozen=True)
class TimingSummary:
    """Aggregate timing across all shards."""

    num_shards: int
    stage_stats: tuple[StageTiming, ...]
    bottleneck_stage: str | None
    shard_wall_times: dict[int, float]
    max_wall_time_s: float
    min_wall_time_s: float
    imbalance_pct: float


def load_shard_results(dataset_output_dir: Path) -> dict[int, dict[str, Any]]:
    """Load ``pipeline_results.json`` from each ``shard_N/`` subdirectory."""
    results: dict[int, dict[str, Any]] = {}
    for shard_dir in sorted(dataset_output_dir.glob("shard_*")):
        results_file = shard_dir / "pipeline_results.json"
        if not results_file.is_file():
            continue
        suffix = shard_dir.name.removeprefix("shard_")
        if not suffix.isdigit():
            continue
        shard_id = int(suffix)
        try:
            results[shard_id] = json.loads(results_file.read_text())
        except json.JSONDecodeError:
            continue
    return results


def _extract_stages(data: dict[str, Any]) -> dict[str, float]:
    """Pull stage → duration mapping from a pipeline_results payload.

    Handles two historical shapes:
    - Nested under ``stage_timings`` or ``stages`` (value can be a dict
      with ``duration`` or a bare number).
    - Top-level keys starting with ``stage_`` (same two inner shapes).
    """
    nested = data.get("stage_timings") or data.get("stages")
    stages: dict[str, float] = {}
    if isinstance(nested, dict):
        for name, info in nested.items():
            if isinstance(info, (int, float)):
                stages[name] = float(info)
            elif isinstance(info, dict) and "duration" in info:
                stages[name] = float(info["duration"])
        return stages

    for key, val in data.items():
        if not key.startswith("stage_"):
            continue
        if isinstance(val, dict) and "duration" in val:
            stages[key] = float(val["duration"])
        elif isinstance(val, (int, float)):
            stages[key] = float(val)
    return stages


def aggregate(shard_results: dict[int, dict[str, Any]]) -> TimingSummary:
    """Compute per-stage stats + wall-time imbalance from shard results."""
    per_stage: dict[str, list[float]] = defaultdict(list)
    wall_times: dict[int, float] = {}

    for shard_id, data in shard_results.items():
        stages = _extract_stages(data)
        total = sum(stages.values())
        for stage, duration in stages.items():
            per_stage[stage].append(duration)
        wall_times[shard_id] = float(data.get("total_duration", data.get("total_wall_time", total)))

    stage_stats = [
        StageTiming(
            name=name,
            min_s=round(min(durations), 2),
            avg_s=round(sum(durations) / len(durations), 2),
            max_s=round(max(durations), 2),
            shards=len(durations),
        )
        for name, durations in per_stage.items()
    ]
    stage_stats.sort(key=lambda s: s.avg_s, reverse=True)
    bottleneck = stage_stats[0].name if stage_stats else None

    values = list(wall_times.values())
    max_wall = max(values) if values else 0.0
    min_wall = min(values) if values else 0.0
    imbalance_pct = round(((max_wall - min_wall) / max_wall * 100) if max_wall > 0 else 0.0, 1)

    return TimingSummary(
        num_shards=len(shard_results),
        stage_stats=tuple(stage_stats),
        bottleneck_stage=bottleneck,
        shard_wall_times={k: round(v, 2) for k, v in wall_times.items()},
        max_wall_time_s=round(max_wall, 2),
        min_wall_time_s=round(min_wall, 2),
        imbalance_pct=imbalance_pct,
    )


def render_timing_summary(summary: TimingSummary) -> str:
    """Format a timing summary for CLI output."""
    if summary.num_shards == 0:
        return "No shard pipeline_results.json files found."

    lines: list[str] = []
    lines.append(f"Timing Summary ({summary.num_shards} shards)")
    lines.append(f"{'Stage':<40} {'Min':>10} {'Avg':>10} {'Max':>10} {'Shards':>8}")
    lines.append("-" * 82)
    for s in summary.stage_stats:
        lines.append(f"{s.name:<40} {s.min_s:>9.1f}s {s.avg_s:>9.1f}s {s.max_s:>9.1f}s {s.shards:>8}")
    lines.append("")
    if summary.bottleneck_stage:
        lines.append(f"Bottleneck stage: {summary.bottleneck_stage}")
    lines.append(
        f"Wall time — min {summary.min_wall_time_s:.1f}s / max {summary.max_wall_time_s:.1f}s / "
        f"imbalance {summary.imbalance_pct}%"
    )
    return "\n".join(lines)


__all__ = [
    "StageTiming",
    "TimingSummary",
    "aggregate",
    "load_shard_results",
    "render_timing_summary",
]
