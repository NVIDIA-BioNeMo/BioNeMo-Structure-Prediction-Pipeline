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

"""Tests for bspp.orchestration.runtime.postprocessing.sharding."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from bspp.orchestration.runtime.postprocessing.sharding import (
    compute_shard_slice,
    create_symlink_shard,
    filter_manifest_for_shard,
    persist_shard_manifest,
)

# --- compute_shard_slice ---


def test_compute_shard_slice_single() -> None:
    """A single shard gets all items."""
    start, end = compute_shard_slice(0, 100, 1)
    assert start == 0
    assert end == 100


def test_compute_shard_slice_multiple() -> None:
    """10 items across 3 shards: 4+3+3."""
    s0 = compute_shard_slice(0, 10, 3)
    s1 = compute_shard_slice(1, 10, 3)
    s2 = compute_shard_slice(2, 10, 3)

    assert s0 == (0, 4)
    assert s1 == (4, 7)
    assert s2 == (7, 10)

    # All items covered, no overlap
    all_indices = list(range(*s0)) + list(range(*s1)) + list(range(*s2))
    assert all_indices == list(range(10))


def test_compute_shard_slice_uneven() -> None:
    """7 items across 3 shards: 3+2+2."""
    s0 = compute_shard_slice(0, 7, 3)
    s1 = compute_shard_slice(1, 7, 3)
    s2 = compute_shard_slice(2, 7, 3)

    assert s0 == (0, 3)
    assert s1 == (3, 5)
    assert s2 == (5, 7)


def test_compute_shard_slice_edge_cases() -> None:
    """Edge cases: zero items and single item."""
    # Zero items, single shard
    start, end = compute_shard_slice(0, 0, 1)
    assert start == 0
    assert end == 0

    # Single item, single shard
    start, end = compute_shard_slice(0, 1, 1)
    assert start == 0
    assert end == 1

    # Single item, multiple shards -- only first shard gets it
    start, end = compute_shard_slice(0, 1, 3)
    assert start == 0
    assert end == 1

    start, end = compute_shard_slice(1, 1, 3)
    assert start == 1
    assert end == 1  # empty slice

    # Invalid inputs
    with pytest.raises(ValueError, match="num_shards must be positive"):
        compute_shard_slice(0, 10, 0)

    with pytest.raises(ValueError, match=r"shard_id.*out of range"):
        compute_shard_slice(3, 10, 3)

    with pytest.raises(ValueError, match=r"shard_id.*out of range"):
        compute_shard_slice(-1, 10, 3)


def test_compute_shard_slice_even_distribution() -> None:
    """Even distribution: 12 items across 4 shards = 3 each."""
    for i in range(4):
        start, end = compute_shard_slice(i, 12, 4)
        assert end - start == 3


# --- create_symlink_shard ---


def test_create_symlink_shard(tmp_path: Path) -> None:
    """Create real files, symlink them, verify symlinks are correct."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shard_dir = tmp_path / "shard_0"

    # Create real input files
    model_ids = ["AF-001", "AF-002", "AF-003"]
    file_index: dict[str, list[str]] = {}
    for mid in model_ids:
        pdb = f"{mid}-model_v1.pdb"
        meta = f"{mid}-meta_v1.json"
        (input_dir / pdb).write_text(f"pdb content for {mid}")
        (input_dir / meta).write_text(f"meta content for {mid}")
        file_index[mid] = [pdb, meta]

    # Symlink only first two models
    count = create_symlink_shard(["AF-001", "AF-002"], file_index, input_dir, shard_dir)
    assert count == 4  # 2 files per model
    assert shard_dir.exists()

    # Verify symlinks
    assert (shard_dir / "AF-001-model_v1.pdb").is_symlink()
    assert (shard_dir / "AF-001-model_v1.pdb").resolve() == (input_dir / "AF-001-model_v1.pdb").resolve()
    assert (shard_dir / "AF-002-meta_v1.json").read_text() == "meta content for AF-002"

    # AF-003 should not be present
    assert not (shard_dir / "AF-003-model_v1.pdb").exists()


def test_create_symlink_shard_skips_existing(tmp_path: Path) -> None:
    """Existing symlinks are silently skipped."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()

    (input_dir / "AF-001-model_v1.pdb").write_text("content")
    file_index = {"AF-001": ["AF-001-model_v1.pdb"]}

    # First call creates symlink
    count1 = create_symlink_shard(["AF-001"], file_index, input_dir, shard_dir)
    assert count1 == 1

    # Second call skips
    count2 = create_symlink_shard(["AF-001"], file_index, input_dir, shard_dir)
    assert count2 == 0


def test_create_symlink_shard_missing_model(tmp_path: Path) -> None:
    """Models not in file_index are silently skipped, count is 0."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shard_dir = tmp_path / "shard"

    count = create_symlink_shard(["AF-MISSING"], {}, input_dir, shard_dir)
    assert count == 0


# --- filter_manifest_for_shard ---


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a test manifest CSV."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_entity_id", "uniprot_ac", "chain_id"])
        writer.writeheader()
        writer.writerows(rows)


def test_filter_manifest_for_shard(tmp_path: Path) -> None:
    """Filtering keeps only rows for shard's model IDs."""
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"model_entity_id": "AF-001", "uniprot_ac": "P111", "chain_id": "A"},
            {"model_entity_id": "AF-002", "uniprot_ac": "P222", "chain_id": "A"},
            {"model_entity_id": "AF-003", "uniprot_ac": "P333", "chain_id": "A"},
            {"model_entity_id": "AF-004", "uniprot_ac": "P444", "chain_id": "A"},
        ],
    )

    output = tmp_path / "shard_manifest.csv"
    count = filter_manifest_for_shard(manifest, {"AF-001", "AF-003"}, output)

    assert count == 2
    with output.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    ids = {r["model_entity_id"] for r in rows}
    assert ids == {"AF-001", "AF-003"}


def test_filter_manifest_for_shard_empty(tmp_path: Path) -> None:
    """No matching models returns 0 rows."""
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [{"model_entity_id": "AF-001", "uniprot_ac": "P111", "chain_id": "A"}],
    )

    output = tmp_path / "shard.csv"
    count = filter_manifest_for_shard(manifest, {"AF-999"}, output)
    assert count == 0


# --- persist_shard_manifest ---


def test_persist_shard_manifest(tmp_path: Path) -> None:
    """Verify parquet output has shard_id and dataset_tag columns."""
    import pyarrow.parquet as pq

    # Write a CSV manifest first
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"model_entity_id": "AF-001", "uniprot_ac": "P111", "chain_id": "A"},
            {"model_entity_id": "AF-002", "uniprot_ac": "P222", "chain_id": "A"},
        ],
    )

    output = tmp_path / "out" / "shard_manifest.parquet"
    persist_shard_manifest(manifest, shard_id=42, dataset_tag="test_run", output_path=output)

    assert output.exists()
    table = pq.read_table(output)
    assert table.num_rows == 2
    assert "shard_id" in table.column_names
    assert "dataset_tag" in table.column_names
    assert table.column("shard_id").to_pylist() == [42, 42]
    assert table.column("dataset_tag").to_pylist() == ["test_run", "test_run"]
    assert table.column("model_entity_id").to_pylist() == ["AF-001", "AF-002"]
