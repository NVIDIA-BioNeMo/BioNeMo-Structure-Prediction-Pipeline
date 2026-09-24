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

"""Tests for validation/coverage."""

from __future__ import annotations

import csv
from pathlib import Path

from bspp.orchestration.runtime.validation.coverage import validate_coverage


def _write_model_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n")


def _write_manifest(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model_entity_id", "chain"])
        writer.writeheader()
        for mid in ids:
            writer.writerow({"model_entity_id": mid, "chain": "A"})


def _make_shard_input(output_dir: Path, shard_id: int, model_ids: list[str]) -> None:
    input_dir = output_dir / f"shard_{shard_id}" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for mid in model_ids:
        (input_dir / f"{mid}-model_v1.pdb").write_text("x")
        (input_dir / f"{mid}-meta_v1.json").write_text("{}")


def test_validate_coverage_happy_path(tmp_path: Path) -> None:
    ids = ["A1", "A2", "A3", "A4"]
    _write_model_ids(tmp_path / "model_ids.txt", ids)
    out = tmp_path / "out"
    _make_shard_input(out, 0, ids[:2])
    _make_shard_input(out, 1, ids[2:])
    _write_manifest(out / "shard_0" / "shard_manifest.csv", ids[:2])
    _write_manifest(out / "shard_1" / "shard_manifest.csv", ids[2:])
    for sid in (0, 1):
        (out / f"shard_{sid}" / "pipeline_results.json").write_text("{}")

    report = validate_coverage(out, model_ids_file=tmp_path / "model_ids.txt")

    assert report.coverage_ok
    assert report.num_shards == 2
    assert not report.missing_ids
    assert not report.duplicate_ids
    assert all(s.manifest_matches_input for s in report.shards)


def test_validate_coverage_detects_gap_and_duplicate(tmp_path: Path) -> None:
    ids = ["A1", "A2", "A3", "A4"]
    _write_model_ids(tmp_path / "model_ids.txt", ids)
    out = tmp_path / "out"
    # shard 0 covers A1, A2; shard 1 covers A2 (duplicate) and A3; A4 is missing
    _make_shard_input(out, 0, ["A1", "A2"])
    _make_shard_input(out, 1, ["A2", "A3"])
    _write_manifest(out / "shard_0" / "shard_manifest.csv", ["A1", "A2"])
    _write_manifest(out / "shard_1" / "shard_manifest.csv", ["A2", "A3"])

    report = validate_coverage(out, model_ids_file=tmp_path / "model_ids.txt")

    assert "A4" in report.missing_ids
    assert "A2" in report.duplicate_ids
    assert not report.coverage_ok


def test_validate_coverage_flags_missing_artifacts(tmp_path: Path) -> None:
    ids = ["A1"]
    _write_model_ids(tmp_path / "model_ids.txt", ids)
    out = tmp_path / "out"
    _make_shard_input(out, 0, ids)
    # no shard_manifest.csv, no pipeline_results.json

    report = validate_coverage(out, model_ids_file=tmp_path / "model_ids.txt")

    shard = report.shards[0]
    assert "shard_manifest.csv" in shard.missing_artifacts
    assert "pipeline_results.json" in shard.missing_artifacts
