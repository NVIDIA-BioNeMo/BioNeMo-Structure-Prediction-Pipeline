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

"""Tests for bspp.orchestration.runtime.postprocessing.aggregate."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bspp.orchestration.runtime.postprocessing.aggregate import (
    aggregate_dataset,
    build_aggregated_table,
    discover_shard_parquets,
    load_failed_model_ids,
    load_upload_status,
)


def _write_shard_parquet(path: Path, model_ids: list[str], shard_id: int) -> None:
    """Helper to create a minimal shard_manifest.parquet."""
    table = pa.table(
        {
            "model_entity_id": model_ids,
            "shard_id": [shard_id] * len(model_ids),
            "pdb_path": [f"shard_{shard_id}/{mid}-model_v1.pdb" for mid in model_ids],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_uploaded_marker(path: Path, shard_id: int, s3_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"shard_id": shard_id, "s3_prefix": s3_prefix}) + "\n")


def test_discover_shard_parquets(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    _write_shard_parquet(output_dir / "shard_0" / "shard_manifest.parquet", ["AF-001"], 0)
    _write_shard_parquet(output_dir / "shard_1" / "shard_manifest.parquet", ["AF-002"], 1)

    result = discover_shard_parquets(output_dir)
    assert len(result) == 2
    assert all(p.name == "shard_manifest.parquet" for p in result)
    # Should be sorted
    assert "shard_0" in str(result[0])
    assert "shard_1" in str(result[1])


def test_discover_shard_parquets_empty(tmp_path: Path) -> None:
    output_dir = tmp_path / "empty_dataset"
    output_dir.mkdir()

    result = discover_shard_parquets(output_dir)
    assert result == []


def test_load_failed_model_ids(tmp_path: Path) -> None:
    failed_file = tmp_path / "failed_models.tsv"
    failed_file.write_text("AF-001\textract\tnull meta json\nAF-003\tvalidation\tbad pLDDT\n")

    result = load_failed_model_ids(tmp_path)
    assert result == {"AF-001", "AF-003"}


def test_load_failed_model_ids_missing(tmp_path: Path) -> None:
    result = load_failed_model_ids(tmp_path)
    assert result == set()


def test_load_upload_status_reads_uploaded_marker_payloads(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    _write_uploaded_marker(
        output_dir / "shard_0" / ".uploaded",
        0,
        "s3://example-bucket/users/test/postprocessed/dataset/",
    )
    (output_dir / "shard_1").mkdir(parents=True)
    (output_dir / "shard_1" / ".uploaded").write_text("")

    result = load_upload_status(output_dir)

    assert result == {0: {"shard_id": 0, "s3_prefix": "s3://example-bucket/users/test/postprocessed/dataset/"}}


def test_build_aggregated_table(tmp_path: Path) -> None:
    shard_0 = tmp_path / "shard_0" / "shard_manifest.parquet"
    shard_1 = tmp_path / "shard_1" / "shard_manifest.parquet"
    _write_shard_parquet(shard_0, ["AF-001", "AF-002"], 0)
    _write_shard_parquet(shard_1, ["AF-003", "AF-004"], 1)

    failed_ids = {"AF-002", "AF-004"}
    table = build_aggregated_table([shard_0, shard_1], failed_ids)

    assert table.num_rows == 4
    assert "status" in table.schema.names
    assert "s3_uploaded" in table.schema.names
    assert "s3_prefix" in table.schema.names

    statuses = table.column("status").to_pylist()
    model_ids = table.column("model_entity_id").to_pylist()

    # AF-001 and AF-003 should be success; AF-002 and AF-004 should be failed
    for mid, status in zip(model_ids, statuses, strict=True):
        if mid in failed_ids:
            assert status == "failed", f"{mid} should be failed"
        else:
            assert status == "success", f"{mid} should be success"

    assert table.column("s3_uploaded").to_pylist() == [False, False, False, False]
    assert table.column("s3_prefix").to_pylist() == ["", "", "", ""]


def test_build_aggregated_table_enriches_from_uploaded_markers(tmp_path: Path) -> None:
    shard_0 = tmp_path / "shard_0" / "shard_manifest.parquet"
    shard_1 = tmp_path / "shard_1" / "shard_manifest.parquet"
    _write_shard_parquet(shard_0, ["AF-001", "AF-002"], 0)
    _write_shard_parquet(shard_1, ["AF-003"], 1)

    prefix = "s3://example-bucket/users/test/postprocessed/dsA/"
    table = build_aggregated_table(
        [shard_0, shard_1],
        failed_ids=set(),
        upload_map={0: {"shard_id": 0, "s3_prefix": prefix}},
    )

    assert table.column("s3_uploaded").to_pylist() == [True, True, False]
    assert table.column("s3_prefix").to_pylist() == [prefix, prefix, ""]


def test_aggregate_dataset(tmp_path: Path) -> None:
    dataset = "test_dataset"
    output_base = tmp_path / "output"
    output_dir = output_base / dataset
    output_dir.mkdir(parents=True)

    # Create shard parquets
    _write_shard_parquet(output_dir / "shard_0" / "shard_manifest.parquet", ["AF-001", "AF-002"], 0)
    _write_shard_parquet(output_dir / "shard_1" / "shard_manifest.parquet", ["AF-003"], 1)
    prefix = "s3://example-bucket/users/test/postprocessed/test_dataset/"
    _write_uploaded_marker(output_dir / "shard_0" / ".uploaded", 0, prefix)

    # Create failed_models.tsv
    (output_dir / "failed_models.tsv").write_text("AF-002\textract\tnull meta\n")

    result_path = aggregate_dataset(dataset, output_base)
    assert result_path.exists()
    assert result_path == output_base / f"{dataset}_manifest.parquet"

    table = pq.read_table(result_path)
    assert table.num_rows == 3
    assert "status" in table.schema.names

    statuses = dict(zip(table.column("model_entity_id").to_pylist(), table.column("status").to_pylist(), strict=True))
    assert statuses["AF-001"] == "success"
    assert statuses["AF-002"] == "failed"
    assert statuses["AF-003"] == "success"

    s3_uploaded = dict(
        zip(table.column("model_entity_id").to_pylist(), table.column("s3_uploaded").to_pylist(), strict=True)
    )
    s3_prefix = dict(
        zip(table.column("model_entity_id").to_pylist(), table.column("s3_prefix").to_pylist(), strict=True)
    )
    assert s3_uploaded == {"AF-001": True, "AF-002": True, "AF-003": False}
    assert s3_prefix == {"AF-001": prefix, "AF-002": prefix, "AF-003": ""}
