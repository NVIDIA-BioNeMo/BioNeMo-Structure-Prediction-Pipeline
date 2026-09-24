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

"""Tests for validation/count_outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bspp.orchestration.runtime.validation import count_shard_outputs, failed_shard_ids
from bspp.orchestration.runtime.validation.count_outputs import render_count_report


def _write_shard_config(output_dir: Path, total_models: int, required_shards: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shard_config.json").write_text(
        json.dumps({"total_models": total_models, "required_shards": required_shards})
    )


def _make_shard(
    output_dir: Path,
    shard_id: int,
    *,
    pdb_count: int = 0,
    uploaded_model_count: int | None = None,
    uploaded_status: str = "uploaded",
) -> None:
    shard = output_dir / f"shard_{shard_id}"
    success = shard / "success_outputs"
    success.mkdir(parents=True, exist_ok=True)
    for i in range(pdb_count):
        (success / f"model_{i}.pdb").write_text("x")
    if uploaded_model_count is not None:
        (shard / ".uploaded").write_text(json.dumps({"model_count": uploaded_model_count, "status": uploaded_status}))


def test_count_shard_outputs_happy_path(tmp_path: Path) -> None:
    # 10 models across 2 shards -> shard 0 = 5, shard 1 = 5
    _write_shard_config(tmp_path, total_models=10, required_shards=2)
    _make_shard(tmp_path, 0, pdb_count=5)
    _make_shard(tmp_path, 1, pdb_count=5)

    report = count_shard_outputs(tmp_path)

    assert report.total_expected == 10
    assert report.total_actual == 10
    assert all(s.ok for s in report.shards)
    assert failed_shard_ids(report) == []


def test_count_shard_outputs_detects_mismatch(tmp_path: Path) -> None:
    _write_shard_config(tmp_path, total_models=10, required_shards=2)
    _make_shard(tmp_path, 0, pdb_count=5)
    _make_shard(tmp_path, 1, pdb_count=3)

    report = count_shard_outputs(tmp_path)

    assert failed_shard_ids(report) == [1]
    mismatches = report.mismatches
    assert len(mismatches) == 1
    assert mismatches[0].shard_id == 1
    assert mismatches[0].diff == -2


def test_count_shard_outputs_falls_back_to_uploaded_marker(tmp_path: Path) -> None:
    _write_shard_config(tmp_path, total_models=10, required_shards=2)
    _make_shard(tmp_path, 0, pdb_count=5)
    _make_shard(tmp_path, 1, pdb_count=0, uploaded_model_count=5)

    report = count_shard_outputs(tmp_path)

    assert failed_shard_ids(report) == []
    assert report.used_uploaded_fallback is True


def test_count_shard_outputs_rejects_partial_uploaded_marker_as_complete(tmp_path: Path) -> None:
    _write_shard_config(tmp_path, total_models=10, required_shards=2)
    _make_shard(tmp_path, 0, pdb_count=5)
    _make_shard(tmp_path, 1, pdb_count=0, uploaded_model_count=5, uploaded_status="partial_uploaded")

    report = count_shard_outputs(tmp_path)

    assert failed_shard_ids(report) == [1]
    assert report.shards[1].source == "missing"
    assert report.valid is False


def test_count_shard_outputs_flags_missing_shard(tmp_path: Path) -> None:
    _write_shard_config(tmp_path, total_models=10, required_shards=2)
    _make_shard(tmp_path, 0, pdb_count=5)
    _make_shard(tmp_path, 1, pdb_count=0)  # no pdbs, no uploaded marker

    report = count_shard_outputs(tmp_path)

    assert failed_shard_ids(report) == [1]
    assert report.shards[1].source == "missing"


def test_count_shard_outputs_synthesizes_absent_shard_directory(tmp_path: Path) -> None:
    """Regression guard: shards whose directory never existed still count as failures.

    If a shard was never created (scheduler crash, manual cleanup), its
    shard_N/ directory is absent. The earlier implementation only
    iterated over shard_* globs, so the absent shard never appeared in
    the report — ``slurm resubmit-failed`` would then think the dataset
    was complete. Verify the new behavior: any shard_id in
    range(required_shards) without an on-disk directory shows up with
    source='missing' and is included in failed_ids.
    """
    _write_shard_config(tmp_path, total_models=10, required_shards=2)
    _make_shard(tmp_path, 0, pdb_count=5)
    # Deliberately never create shard_1/.

    report = count_shard_outputs(tmp_path)

    assert len(report.shards) == 2
    shard1 = next(s for s in report.shards if s.shard_id == 1)
    assert shard1.source == "missing"
    assert shard1.expected == 5
    assert shard1.actual == 0
    assert 1 in failed_shard_ids(report)


def test_count_shard_outputs_tolerates_stale_extra_shard_dir(tmp_path: Path) -> None:
    """Regression guard: stale shard_id >= required_shards must not crash.

    If a previous run left a ``shard_99/`` directory behind and the
    current shard_config declares only 2 shards, the earlier
    implementation called compute_shard_slice(99, total, 2) which
    raises ValueError and aborts the entire report. That turned a
    recoverable anomaly into a hard failure that blocked identifying
    genuinely-failed shards. The new behavior reports it with
    source='stale', expected=0, and keeps the report usable; stale
    shards are explicitly excluded from failed_ids so resubmit-failed
    does not try to recreate them.
    """
    _write_shard_config(tmp_path, total_models=10, required_shards=2)
    _make_shard(tmp_path, 0, pdb_count=5)
    _make_shard(tmp_path, 1, pdb_count=5)
    # shard_99/ is the anomaly: a stale leftover.
    _make_shard(tmp_path, 99, pdb_count=42)

    report = count_shard_outputs(tmp_path)

    # Did not crash; all three shards present in the report.
    shard_ids = {s.shard_id for s in report.shards}
    assert shard_ids == {0, 1, 99}

    stale = report.stale_shards
    assert len(stale) == 1
    assert stale[0].shard_id == 99
    assert stale[0].source == "stale"
    assert stale[0].expected == 0
    assert stale[0].actual == 42

    # Stale shards are not failures (resubmit would be meaningless).
    assert failed_shard_ids(report) == []
    assert stale[0].ok is True

    # Render path must also stay operational and surface the anomaly.
    from bspp.orchestration.runtime.validation.count_outputs import render_count_report

    rendered = render_count_report(report)
    assert "Stale shard directories" in rendered
    assert "shard_99" in rendered

    # Regression guard for the silent-OK-on-stale bug: even though no
    # shard needs resubmission, the report is not "valid" because stale
    # content exists on disk and would contaminate downstream steps.
    assert not report.valid


def test_validate_count_cli_exits_nonzero_on_stale_shard(tmp_path: Path) -> None:
    """`validate count --failed-only` must fail when only stale dirs are present.

    Regression guard: stale shards are excluded from failed_ids (they
    are not resubmit candidates) but leaving them on disk contaminates
    uploads/aggregation/coverage. The CLI reports the situation via a
    non-zero exit.
    """
    from click.testing import CliRunner

    from bspp.orchestration.runtime.cli import cli

    _write_shard_config(tmp_path, total_models=4, required_shards=2)
    _make_shard(tmp_path, 0, pdb_count=2)
    _make_shard(tmp_path, 1, pdb_count=2)
    _make_shard(tmp_path, 99, pdb_count=5)  # stale

    runner = CliRunner()
    output_base = tmp_path.parent
    dataset = tmp_path.name

    result = runner.invoke(
        cli,
        [
            "validate",
            "count",
            "--dataset",
            dataset,
            "--output-base",
            str(output_base),
        ],
    )

    assert result.exit_code == 1, result.output
    assert "shard_99" in result.output


def test_count_shard_outputs_uneven_distribution(tmp_path: Path) -> None:
    # 11 models, 3 shards -> slices: 4, 4, 3 (remainder-first distribution)
    _write_shard_config(tmp_path, total_models=11, required_shards=3)
    _make_shard(tmp_path, 0, pdb_count=4)
    _make_shard(tmp_path, 1, pdb_count=4)
    _make_shard(tmp_path, 2, pdb_count=3)

    report = count_shard_outputs(tmp_path)

    assert [s.expected for s in report.shards] == [4, 4, 3]
    assert failed_shard_ids(report) == []


def test_count_shard_outputs_missing_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        count_shard_outputs(tmp_path)


def test_render_count_report_contains_summary(tmp_path: Path) -> None:
    _write_shard_config(tmp_path, total_models=10, required_shards=2)
    _make_shard(tmp_path, 0, pdb_count=5)
    _make_shard(tmp_path, 1, pdb_count=3)

    rendered = render_count_report(count_shard_outputs(tmp_path))

    assert "Total expected" in rendered
    assert "Mismatched shards: 1" in rendered
    assert "shard_1" in rendered
