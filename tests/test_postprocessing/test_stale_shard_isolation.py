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

"""Regression guard: stale shard directories are isolated from downstream consumers.

A ``shard_N/`` directory whose id is ``>= required_shards`` (from
``shard_config.json``) is a leftover from a prior, larger run. Earlier
revisions globbed ``shard_*`` indiscriminately in the upload, aggregate,
and coverage paths, so stale outputs leaked into all three. The helpers
in :mod:`bspp.orchestration.runtime.postprocessing.sharding`
(``iter_in_range_shard_dirs`` + ``read_required_shards``) gate every
consumer; these tests prove the gate is active.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bspp.orchestration.runtime.postprocessing.aggregate import discover_shard_parquets
from bspp.orchestration.runtime.postprocessing.sharding import (
    InvalidShardConfigError,
    read_required_shards,
)
from bspp.orchestration.runtime.postprocessing.upload_and_track import find_success_outputs
from bspp.orchestration.runtime.validation.coverage import validate_coverage


def _write_shard_config(dataset_dir: Path, total: int, required: int) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "shard_config.json").write_text(json.dumps({"total_models": total, "required_shards": required}))


def _make_success_shard(dataset_dir: Path, shard_id: int, models: list[str]) -> None:
    success = dataset_dir / f"shard_{shard_id}" / "success_outputs"
    success.mkdir(parents=True, exist_ok=True)
    for m in models:
        (success / f"{m}.pdb").write_text("x")


def _make_shard_manifest_parquet(dataset_dir: Path, shard_id: int, models: list[str]) -> None:
    shard_dir = dataset_dir / f"shard_{shard_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table({"model_entity_id": pa.array(models, pa.string())})
    pq.write_table(table, shard_dir / "shard_manifest.parquet")


def _write_manifest_csv(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model_entity_id", "chain"])
        writer.writeheader()
        for mid in ids:
            writer.writerow({"model_entity_id": mid, "chain": "A"})


def _make_shard_input(dataset_dir: Path, shard_id: int, ids: list[str]) -> None:
    inp = dataset_dir / f"shard_{shard_id}" / "input"
    inp.mkdir(parents=True, exist_ok=True)
    for mid in ids:
        (inp / f"{mid}-model_v1.pdb").write_text("x")
        (inp / f"{mid}-meta_v1.json").write_text("{}")


def test_find_success_outputs_skips_stale_shard_dir(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _write_shard_config(dataset, total=4, required=2)
    _make_success_shard(dataset, 0, ["m0"])
    _make_success_shard(dataset, 1, ["m1"])
    _make_success_shard(dataset, 99, ["stale_model"])  # leftover

    dirs, _skipped = find_success_outputs(dataset)
    shard_names = {p.parent.name for p in dirs}

    assert shard_names == {"shard_0", "shard_1"}
    assert "shard_99" not in shard_names


def test_discover_shard_parquets_skips_stale_shard_dir(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _write_shard_config(dataset, total=4, required=2)
    _make_shard_manifest_parquet(dataset, 0, ["m0"])
    _make_shard_manifest_parquet(dataset, 1, ["m1"])
    _make_shard_manifest_parquet(dataset, 99, ["stale_model"])

    parquets = discover_shard_parquets(dataset)
    shard_names = {p.parent.name for p in parquets}

    assert shard_names == {"shard_0", "shard_1"}
    assert "shard_99" not in shard_names


def test_validate_coverage_skips_stale_shard_dir(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _write_shard_config(dataset, total=4, required=2)

    ids = ["A1", "A2", "A3", "A4"]
    _write_manifest_csv(dataset / "shard_0" / "shard_manifest.csv", ids[:2])
    _write_manifest_csv(dataset / "shard_1" / "shard_manifest.csv", ids[2:])
    _make_shard_input(dataset, 0, ids[:2])
    _make_shard_input(dataset, 1, ids[2:])
    (dataset / "shard_0" / "pipeline_results.json").write_text("{}")
    (dataset / "shard_1" / "pipeline_results.json").write_text("{}")

    # Stale shard: both an old model AND a model from the expected set —
    # if coverage processed it the duplicate check would fail.
    _make_shard_input(dataset, 99, ["A1", "stale_model"])
    _write_manifest_csv(dataset / "shard_99" / "shard_manifest.csv", ["A1", "stale_model"])
    (dataset / "shard_99" / "pipeline_results.json").write_text("{}")

    ids_file = tmp_path / "model_ids.txt"
    ids_file.write_text("\n".join(ids) + "\n")

    report = validate_coverage(dataset, model_ids_file=ids_file)

    shard_ids = {s.shard_id for s in report.shards}
    assert shard_ids == {0, 1}, "stale shard_99 must not be included"
    assert report.coverage_ok
    assert not report.duplicate_ids


# ---------------------------------------------------------------------------
# Regression guards: corrupted shard_config.json must fail closed.
#
# A truncated / unreadable / semantically-invalid shard_config.json was
# previously logged-and-ignored, which silently reverted every consumer
# to the unguarded shard_* glob — reopening the contamination path this
# commit is supposed to block. read_required_shards() now raises
# InvalidShardConfigError and the consumers let it propagate.
# ---------------------------------------------------------------------------


def test_read_required_shards_absent_returns_none(tmp_path: Path) -> None:
    """Legacy back-compat: truly absent config still returns None (warning path)."""
    assert read_required_shards(tmp_path) is None


def test_read_required_shards_raises_on_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "shard_config.json").write_text("{not: valid json")

    with pytest.raises(InvalidShardConfigError):
        read_required_shards(tmp_path)


def test_read_required_shards_raises_when_field_missing(tmp_path: Path) -> None:
    (tmp_path / "shard_config.json").write_text(json.dumps({"total_models": 10}))

    with pytest.raises(InvalidShardConfigError):
        read_required_shards(tmp_path)


def test_read_required_shards_raises_on_non_positive(tmp_path: Path) -> None:
    (tmp_path / "shard_config.json").write_text(json.dumps({"total_models": 10, "required_shards": 0}))

    with pytest.raises(InvalidShardConfigError):
        read_required_shards(tmp_path)


def test_read_required_shards_raises_on_wrong_type(tmp_path: Path) -> None:
    (tmp_path / "shard_config.json").write_text(json.dumps({"total_models": 10, "required_shards": "two"}))

    with pytest.raises(InvalidShardConfigError):
        read_required_shards(tmp_path)


def test_find_success_outputs_fails_closed_on_corrupted_config(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "shard_config.json").write_text("garbage{")
    _make_success_shard(dataset, 0, ["m0"])
    _make_success_shard(dataset, 99, ["stale"])

    with pytest.raises(InvalidShardConfigError):
        find_success_outputs(dataset)


def test_discover_shard_parquets_fails_closed_on_corrupted_config(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "shard_config.json").write_text(json.dumps({"required_shards": -1}))
    _make_shard_manifest_parquet(dataset, 0, ["m0"])

    with pytest.raises(InvalidShardConfigError):
        discover_shard_parquets(dataset)


def test_validate_coverage_fails_closed_on_corrupted_config(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "shard_config.json").write_text("{}")  # missing required_shards

    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("A1\n")

    with pytest.raises(InvalidShardConfigError):
        validate_coverage(dataset, model_ids_file=ids_file)


# ---------------------------------------------------------------------------
# Semantic cross-checks: a positive-int required_shards that disagrees with
# the rest of the config is also rejected, so a stale or tampered file
# can't quietly drop real shards as "stale".
# ---------------------------------------------------------------------------


def test_read_required_shards_rejects_array_range_mismatch(tmp_path: Path) -> None:
    (tmp_path / "shard_config.json").write_text(json.dumps({"required_shards": 2, "array_range": "0-9"}))

    with pytest.raises(InvalidShardConfigError, match="array_range"):
        read_required_shards(tmp_path)


def test_read_required_shards_accepts_comma_array_range(tmp_path: Path) -> None:
    (tmp_path / "shard_config.json").write_text(json.dumps({"required_shards": 4, "array_range": "0,2-3,9%2"}))

    assert read_required_shards(tmp_path) == 4


def test_read_required_shards_rejects_reverse_array_range(tmp_path: Path) -> None:
    (tmp_path / "shard_config.json").write_text(json.dumps({"required_shards": 2, "array_range": "3-0"}))

    with pytest.raises(InvalidShardConfigError, match="array_range is invalid"):
        read_required_shards(tmp_path)


def test_read_required_shards_rejects_archive_mode_count_mismatch(tmp_path: Path) -> None:
    (tmp_path / "shard_config.json").write_text(
        json.dumps(
            {
                "archive_mode": True,
                "required_shards": 2,
                "total_archives": 10,
                "shards_per_archive": 2,
            }
        )
    )

    with pytest.raises(InvalidShardConfigError, match="total_archives"):
        read_required_shards(tmp_path)


def test_read_required_shards_rejects_flat_mode_count_mismatch(tmp_path: Path) -> None:
    # ceil(100 / 5) == 20, so required_shards=2 is internally inconsistent.
    (tmp_path / "shard_config.json").write_text(
        json.dumps({"total_models": 100, "max_per_shard": 5, "required_shards": 2})
    )

    with pytest.raises(InvalidShardConfigError, match="max_per_shard"):
        read_required_shards(tmp_path)


def test_read_required_shards_accepts_compute_shard_config_output(tmp_path: Path) -> None:
    """Round-trip: the computed configs we ship must satisfy the validator."""
    from bspp.orchestration.runtime.postprocessing.shard_config import (
        compute_archive_shard_config,
        compute_shard_config,
        write_shard_config,
    )

    flat = tmp_path / "flat"
    write_shard_config(compute_shard_config(total_models=137, max_per_shard=10), flat / "shard_config.json")
    assert read_required_shards(flat) == 14  # ceil(137/10)

    archive = tmp_path / "arch"
    write_shard_config(
        compute_archive_shard_config(["a.tar.lz4"] * 4, shards_per_archive=2),
        archive / "shard_config.json",
    )
    assert read_required_shards(archive) == 4
