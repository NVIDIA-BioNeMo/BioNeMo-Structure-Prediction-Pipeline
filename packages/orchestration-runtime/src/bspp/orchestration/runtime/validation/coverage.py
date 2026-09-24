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

"""Validate shard coverage — no gaps, no duplicates, manifest consistent.

Port of the coverage checks from ``slurm-scaling/tools/validate_multinode.py``.
Produces a :class:`CoverageReport` that can be rendered to text or
consumed programmatically (e.g. to decide whether to resubmit).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from bspp.orchestration.runtime.constants import KNOWN_SUFFIXES
from bspp.orchestration.runtime.postprocessing.sharding import (
    iter_in_range_shard_dirs,
    read_required_shards,
)

logger = logging.getLogger(__name__)


EXPECTED_ARTIFACTS = (
    "shard_manifest.csv",
    "input",
    "pipeline_results.json",
)


def _extract_model_id(filename: str) -> str | None:
    for suffix in KNOWN_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return None


def _collect_shard_model_ids(shard_input_dir: Path) -> set[str]:
    if not shard_input_dir.is_dir():
        return set()
    ids: set[str] = set()
    for entry in shard_input_dir.iterdir():
        mid = _extract_model_id(entry.name)
        if mid is not None:
            ids.add(mid)
    return ids


def _read_manifest_model_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.is_file():
        return set()
    ids: set[str] = set()
    with manifest_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mid = row.get("model_entity_id")
            if mid:
                ids.add(mid)
    return ids


@dataclass(frozen=True)
class ShardCoverage:
    """Coverage view of one shard directory."""

    shard_id: int
    input_ids: frozenset[str]
    manifest_ids: frozenset[str]
    missing_artifacts: tuple[str, ...]

    @property
    def manifest_matches_input(self) -> bool:
        return self.manifest_ids == self.input_ids


@dataclass(frozen=True)
class CoverageReport:
    """Per-shard coverage + global gap/duplicate diagnosis."""

    output_dir: Path
    expected_ids: frozenset[str]
    shards: tuple[ShardCoverage, ...]
    missing_ids: frozenset[str] = field(default_factory=frozenset)
    extra_ids: frozenset[str] = field(default_factory=frozenset)
    duplicate_ids: frozenset[str] = field(default_factory=frozenset)

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    @property
    def coverage_ok(self) -> bool:
        return not self.missing_ids and not self.extra_ids and not self.duplicate_ids


def validate_coverage(
    dataset_output_dir: Path,
    *,
    model_ids_file: Path,
) -> CoverageReport:
    """Check shard coverage for a dataset.

    Stale ``shard_N/`` directories whose id is ``>= required_shards``
    (from ``shard_config.json``) are excluded from the coverage set —
    their leftover input symlinks would otherwise bloat the "seen"
    union and mask genuine gaps. If ``shard_config.json`` is absent the
    helper logs a warning and falls back to the pre-guard behavior.

    Args:
        dataset_output_dir: Directory containing ``shard_N/`` subdirs.
        model_ids_file: File with one model ID per line (the expected
            universe produced by ``preprocess``).

    Returns:
        A :class:`CoverageReport` describing per-shard model sets,
        missing/extra/duplicate IDs, and artifact presence.
    """
    expected = frozenset(line.strip() for line in model_ids_file.read_text().splitlines() if line.strip())

    shards: list[ShardCoverage] = []
    seen_union: set[str] = set()
    duplicates: set[str] = set()

    required_shards = read_required_shards(dataset_output_dir)
    for shard_id, shard_dir in iter_in_range_shard_dirs(dataset_output_dir, required_shards):
        input_ids = _collect_shard_model_ids(shard_dir / "input")
        manifest_ids = _read_manifest_model_ids(shard_dir / "shard_manifest.csv")

        duplicates |= seen_union & input_ids
        seen_union |= input_ids

        missing_artifacts = tuple(a for a in EXPECTED_ARTIFACTS if not (shard_dir / a).exists())

        shards.append(
            ShardCoverage(
                shard_id=shard_id,
                input_ids=frozenset(input_ids),
                manifest_ids=frozenset(manifest_ids),
                missing_artifacts=missing_artifacts,
            )
        )

    missing_ids = expected - seen_union
    extra_ids = seen_union - expected

    return CoverageReport(
        output_dir=dataset_output_dir,
        expected_ids=expected,
        shards=tuple(shards),
        missing_ids=frozenset(missing_ids),
        extra_ids=frozenset(extra_ids),
        duplicate_ids=frozenset(duplicates),
    )


def render_coverage_report(report: CoverageReport) -> str:
    """Format a coverage report for CLI output."""
    lines: list[str] = []
    lines.append("Shard Coverage")
    lines.append(f"  Output dir:    {report.output_dir}")
    lines.append(f"  Expected IDs:  {len(report.expected_ids):,}")
    lines.append(f"  Shards found:  {report.num_shards}")
    lines.append("")

    for s in report.shards:
        marker = "OK" if s.manifest_matches_input else "MISMATCH"
        lines.append(f"  shard_{s.shard_id}: input={len(s.input_ids):,} manifest={len(s.manifest_ids):,} [{marker}]")
        if s.missing_artifacts:
            lines.append(f"    missing artifacts: {', '.join(s.missing_artifacts)}")

    lines.append("")
    if report.missing_ids:
        lines.append(f"Missing from shards: {len(report.missing_ids):,} (sample: {sorted(report.missing_ids)[:5]})")
    if report.extra_ids:
        lines.append(f"Unexpected IDs: {len(report.extra_ids):,} (sample: {sorted(report.extra_ids)[:5]})")
    if report.duplicate_ids:
        lines.append(f"Duplicate IDs across shards: {len(report.duplicate_ids):,}")
    if report.coverage_ok:
        lines.append("Coverage OK: all expected models covered exactly once.")
    return "\n".join(lines)


__all__ = [
    "CoverageReport",
    "ShardCoverage",
    "render_coverage_report",
    "validate_coverage",
]
