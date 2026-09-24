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

"""Tests for bspp.orchestration.runtime.postprocessing.cleanup."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bspp.orchestration.runtime.postprocessing.cleanup import SAFE_STATUSES, cleanup_shard_outputs


def _create_shard_dirs(base: Path, n_shards: int) -> None:
    """Create shard directories with success_outputs/ and metadata."""
    for i in range(n_shards):
        shard_dir = base / f"shard_{i}"
        success_dir = shard_dir / "success_outputs"
        success_dir.mkdir(parents=True)
        (success_dir / f"model_{i}.pdb").write_text("pdb data")
        (success_dir / f"model_{i}.cif").write_text("cif data")
        # Also create metadata that should be preserved
        (shard_dir / "failed_models.tsv").write_text("")
        (shard_dir / "shard_manifest.parquet").write_text("parquet placeholder")


def _create_tracking(path: Path, status: str, n_rows: int = 5) -> None:
    """Create a minimal tracking parquet with a given status."""
    table = pa.table(
        {
            "model_entity_id": [f"AF-{i:016d}" for i in range(n_rows)],
            "dataset_name": ["test_dataset"] * n_rows,
            "postprocess_status": [status] * n_rows,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_cleanup_shard_outputs_force(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    _create_shard_dirs(output_dir, 3)

    # Verify success_outputs exist
    assert (output_dir / "shard_0" / "success_outputs").exists()
    assert (output_dir / "shard_1" / "success_outputs").exists()
    assert (output_dir / "shard_2" / "success_outputs").exists()

    count = cleanup_shard_outputs(output_dir, force=True)
    assert count == 3

    # success_outputs removed
    assert not (output_dir / "shard_0" / "success_outputs").exists()
    assert not (output_dir / "shard_1" / "success_outputs").exists()
    assert not (output_dir / "shard_2" / "success_outputs").exists()


def test_cleanup_shard_outputs_refuses_without_force(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    _create_shard_dirs(output_dir, 2)

    # No tracking path provided and not force -> ValueError
    with pytest.raises(ValueError, match="tracking_path is required"):
        cleanup_shard_outputs(output_dir, force=False)

    # Tracking with unsafe status -> RuntimeError
    tracking = tmp_path / "tracking.parquet"
    _create_tracking(tracking, "pending")

    with pytest.raises(RuntimeError, match="Cannot clean up"):
        cleanup_shard_outputs(
            output_dir,
            force=False,
            tracking_path=tracking,
            dataset="test_dataset",
        )

    # success_outputs should still exist
    assert (output_dir / "shard_0" / "success_outputs").exists()


def test_cleanup_preserves_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    _create_shard_dirs(output_dir, 2)

    cleanup_shard_outputs(output_dir, force=True)

    # Metadata files should still exist
    assert (output_dir / "shard_0" / "failed_models.tsv").exists()
    assert (output_dir / "shard_0" / "shard_manifest.parquet").exists()
    assert (output_dir / "shard_1" / "failed_models.tsv").exists()
    assert (output_dir / "shard_1" / "shard_manifest.parquet").exists()

    # But success_outputs should be gone
    assert not (output_dir / "shard_0" / "success_outputs").exists()
    assert not (output_dir / "shard_1" / "success_outputs").exists()


def test_safe_statuses() -> None:
    assert isinstance(SAFE_STATUSES, frozenset)
    assert "uploaded_s3" in SAFE_STATUSES
    assert "uploaded_gcs" in SAFE_STATUSES
    assert "done" in SAFE_STATUSES
    assert "pending" not in SAFE_STATUSES
