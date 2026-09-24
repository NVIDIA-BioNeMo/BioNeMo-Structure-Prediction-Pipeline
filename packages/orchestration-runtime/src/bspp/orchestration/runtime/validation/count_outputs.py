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

"""Count per-shard success outputs and report expected-vs-actual mismatches.

Port of ``slurm-scaling/tools/count_success_outputs.py`` from the legacy
target-cluster pipeline. Preserves the two data sources it relies on:

- ``shard_N/success_outputs/*.pdb``          — files produced by a shard
  that did not self-upload (or whose self-upload failed).
- ``shard_N/.uploaded`` JSON (``model_count``) — fallback marker left by
  shards that **did** self-upload their outputs directly to S3; the
  marker records how many models were uploaded.

Returns a :class:`CountReport` with per-shard detail plus aggregate
totals. :func:`failed_shard_ids` is the thin helper that the SLURM
``resubmit-failed`` command consumes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.runtime.postprocessing.sharding import compute_shard_slice

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShardCount:
    """Per-shard expected-vs-actual output counts.

    ``source`` values:
      - ``"success_outputs"`` — shard produced ``.pdb`` files on Lustre.
      - ``"uploaded_marker"`` — self-upload run; count read from
        ``.uploaded`` JSON.
      - ``"missing"`` — shard_id is in ``range(required_shards)`` but
        has no outputs and no ``.uploaded`` marker (either the
        directory doesn't exist, or exists but is empty).
      - ``"stale"`` — shard_id is **outside** ``range(required_shards)``
        but a directory exists on disk (e.g. ``shard_99/`` left over
        from a prior, larger run). Reported as an anomaly, **not** as
        a failure — resubmit would be meaningless because the id
        doesn't map to a valid slice.
    """

    shard_id: int
    expected: int
    actual: int
    source: str

    @property
    def diff(self) -> int:
        return self.actual - self.expected

    @property
    def ok(self) -> bool:
        # "stale" ids are anomalies, not failures: they don't belong in
        # the expected set so resubmit shouldn't try to recreate them.
        if self.source == "stale":
            return True
        return self.diff == 0 and self.source != "missing"


@dataclass(frozen=True)
class CountReport:
    """Aggregate count report for a dataset's output directory."""

    output_dir: Path
    total_models: int
    required_shards: int
    shards: tuple[ShardCount, ...]

    @property
    def total_expected(self) -> int:
        return sum(s.expected for s in self.shards)

    @property
    def total_actual(self) -> int:
        return sum(s.actual for s in self.shards)

    @property
    def used_uploaded_fallback(self) -> bool:
        return any(s.source == "uploaded_marker" for s in self.shards)

    @property
    def stale_shards(self) -> tuple[ShardCount, ...]:
        return tuple(s for s in self.shards if s.source == "stale")

    @property
    def mismatches(self) -> tuple[ShardCount, ...]:
        return tuple(s for s in self.shards if not s.ok)

    @property
    def failed_ids(self) -> list[int]:
        # Stale shards are deliberately excluded via ShardCount.ok above,
        # so they won't appear here. Resubmit-failed callers get only
        # in-range shards that need to be re-run.
        return sorted(s.shard_id for s in self.mismatches)

    @property
    def valid(self) -> bool:
        """True iff the dataset is healthy: no mismatches **and** no stale dirs.

        Stale shards are not resubmission candidates (their ids are
        outside ``range(required_shards)``) but they are still an
        anomaly — leftover content on disk from a prior, larger run
        indicates the dataset has not been properly cleaned, and
        downstream consumers now skip them. Surfacing the anomaly
        through a non-zero ``validate count`` exit forces an operator
        to notice and delete the stale directories before the next
        pipeline run.
        """
        return not self.mismatches and not self.stale_shards


def _count_pdb_files(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.glob("*.pdb") if p.is_file())


def _count_from_uploaded_marker(shard_dir: Path) -> int | None:
    """Return the ``model_count`` recorded in ``shard_N/.uploaded``, if any."""
    uploaded = shard_dir / ".uploaded"
    if not uploaded.exists():
        return None
    try:
        data = json.loads(uploaded.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse %s: %s", uploaded, exc)
        return None
    if not isinstance(data, dict) or data.get("status") != "uploaded":
        return None
    value = data.get("model_count")
    if isinstance(value, int):
        return value
    return None


def count_shard_outputs(dataset_output_dir: Path) -> CountReport:
    """Scan *dataset_output_dir* for shard dirs and tally outputs.

    Args:
        dataset_output_dir: Directory containing ``shard_config.json`` and
            ``shard_N/`` subdirectories.

    Returns:
        A populated :class:`CountReport` covering **every** shard ID in
        ``range(required_shards)``, not just the directories that happen
        to exist on disk.  Shards whose directory is entirely absent are
        still reported with ``source="missing"`` so recovery commands
        (``slurm resubmit-failed``) can schedule them — scheduler crashes
        and manual cleanups that leave no trace under ``shard_*/`` must
        not drop out of the failure set.  Shards that exist but produced
        no ``.pdb`` outputs and left no ``.uploaded`` marker are also
        reported as ``"missing"``.

    Raises:
        FileNotFoundError: If ``shard_config.json`` is not present.
        ValueError: If ``required_shards`` is zero (preprocess not run).
    """
    shard_config_path = dataset_output_dir / "shard_config.json"
    if not shard_config_path.exists():
        msg = f"shard_config.json not found in {dataset_output_dir}"
        raise FileNotFoundError(msg)

    config = json.loads(shard_config_path.read_text())
    total_models = int(config["total_models"])
    required_shards = int(config["required_shards"])
    if required_shards <= 0:
        msg = f"required_shards must be > 0 in {shard_config_path}"
        raise ValueError(msg)

    # Map shard_id -> shard directory for every on-disk shard whose name
    # parses as an integer.  Extra shards above required_shards (rare but
    # possible from a stale run) are still surfaced so the operator sees
    # the anomaly.
    on_disk: dict[int, Path] = {}
    for entry in sorted(dataset_output_dir.glob("shard_*")):
        if not entry.is_dir():
            continue
        suffix = entry.name.removeprefix("shard_")
        if not suffix.isdigit():
            continue
        on_disk[int(suffix)] = entry

    expected_ids = set(range(required_shards))
    all_ids = sorted(expected_ids | on_disk.keys())

    counts: list[ShardCount] = []
    for shard_id in all_ids:
        shard_dir = on_disk.get(shard_id)

        if shard_id >= required_shards:
            # Stale/extra shard directory from a prior larger run.
            # compute_shard_slice raises ValueError for out-of-range
            # ids, so skip that call entirely and record the shard as
            # an anomaly — it is not part of the current expected set
            # and must not be resubmitted.
            assert shard_dir is not None  # id came from on_disk.keys()
            success_dir = shard_dir / "success_outputs"
            actual = _count_pdb_files(success_dir)
            if actual == 0:
                fallback = _count_from_uploaded_marker(shard_dir)
                if fallback is not None:
                    actual = fallback
            counts.append(ShardCount(shard_id=shard_id, expected=0, actual=actual, source="stale"))
            continue

        start, end = compute_shard_slice(shard_id, total_models, required_shards)
        expected = end - start

        if shard_dir is None:
            counts.append(ShardCount(shard_id=shard_id, expected=expected, actual=0, source="missing"))
            continue

        success_dir = shard_dir / "success_outputs"
        actual = _count_pdb_files(success_dir)
        if actual > 0:
            source = "success_outputs"
        else:
            fallback = _count_from_uploaded_marker(shard_dir)
            if fallback is not None:
                actual = fallback
                source = "uploaded_marker"
            else:
                source = "missing"

        counts.append(ShardCount(shard_id=shard_id, expected=expected, actual=actual, source=source))

    return CountReport(
        output_dir=dataset_output_dir,
        total_models=total_models,
        required_shards=required_shards,
        shards=tuple(counts),
    )


def failed_shard_ids(report: CountReport) -> list[int]:
    """Return the sorted list of shard IDs that did not match expected counts."""
    return report.failed_ids


def render_count_report(report: CountReport, *, compact: bool = False) -> str:
    """Format a :class:`CountReport` for CLI output."""
    lines: list[str] = []
    lines.append(f"Output dir: {report.output_dir}")
    if not compact:
        for s in report.shards:
            tag = "OK" if s.ok else "MISMATCH"
            note = f" ({s.source})" if s.source != "success_outputs" else ""
            lines.append(f"  shard_{s.shard_id}: expected={s.expected:,} actual={s.actual:,} [{tag}]{note}")
    lines.append("")
    lines.append(f"Total expected (from shard_config): {report.total_expected:,}")
    lines.append(f"Total actual  (pdb / uploaded marker): {report.total_actual:,}")
    if report.used_uploaded_fallback:
        lines.append("  (used .uploaded model_count fallback for self-upload shards)")
    lines.append(f"Difference: {report.total_actual - report.total_expected:+,}")
    if report.mismatches:
        lines.append("")
        lines.append(f"Mismatched shards: {len(report.mismatches)}")
        for m in report.mismatches[:10]:
            lines.append(f"  shard_{m.shard_id}: expected {m.expected:,}, got {m.actual:,} (diff {m.diff:+,})")
        if len(report.mismatches) > 10:
            lines.append(f"  ... and {len(report.mismatches) - 10} more")
    stale = report.stale_shards
    if stale:
        lines.append("")
        lines.append(f"Stale shard directories (shard_id >= required_shards={report.required_shards}): {len(stale)}")
        for s in stale[:10]:
            lines.append(f"  shard_{s.shard_id}: actual={s.actual:,} (not resubmitted)")
        if len(stale) > 10:
            lines.append(f"  ... and {len(stale) - 10} more")
    return "\n".join(lines)


__all__ = [
    "CountReport",
    "ShardCount",
    "count_shard_outputs",
    "failed_shard_ids",
    "render_count_report",
]
