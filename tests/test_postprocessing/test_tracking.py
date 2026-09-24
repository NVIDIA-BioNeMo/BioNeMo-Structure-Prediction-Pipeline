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

"""Tests for bspp.orchestration.runtime.postprocessing.tracking."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bspp.orchestration.runtime.postprocessing.tracking import (
    LIFECYCLE_STATUSES,
    create_tracking_parquet,
    query_status,
    update_status,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
S3_OUTPUT_PREFIX = "s3://example-bucket/postprocessed/test_dataset/"
GCS_DESTINATION_PREFIX = "gs://example-gcs-bucket/postprocessed/test_dataset/"


def _create_master_parquet(path: Path, n_rows: int = 5) -> None:
    """Create a minimal master parquet for testing."""
    table = pa.table(
        {
            "pdb_path": [f"input/AF-{i:016d}-model_v1.pdb" for i in range(n_rows)],
            "model_entity_id": [f"AF-{i:016d}" for i in range(n_rows)],
            "source_run": ["test_dataset"] * n_rows,
            "swiftstack_archive": ["archive.tar.lz4"] * n_rows,
            "predictions_path": [f"output/AF-{i:016d}-conf.json" for i in range(n_rows)],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_lifecycle_statuses() -> None:
    assert isinstance(LIFECYCLE_STATUSES, tuple)
    assert LIFECYCLE_STATUSES[0] == "pending"
    assert LIFECYCLE_STATUSES[-1] == "done"
    assert "uploaded_s3" in LIFECYCLE_STATUSES
    assert "uploaded_gcs" in LIFECYCLE_STATUSES
    assert "recipe_ready" in LIFECYCLE_STATUSES
    assert len(LIFECYCLE_STATUSES) == 10


def test_create_tracking_parquet(tmp_path: Path) -> None:
    master = FIXTURES_DIR / "master.parquet"
    output = tmp_path / "tracking.parquet"

    n = create_tracking_parquet(master, output, s3_output_prefix=S3_OUTPUT_PREFIX)
    assert n == 5
    assert output.exists()

    table = pq.read_table(output)
    assert "postprocess_status" in table.schema.names
    assert "dataset_name" in table.schema.names
    assert "extract_timestamp" in table.schema.names
    assert "s3_destination" in table.schema.names
    assert "gcs_destination" in table.schema.names
    assert "needs_archive_resolution" in table.schema.names

    # All statuses should be pending
    statuses = set(table.column("postprocess_status").to_pylist())
    assert statuses == {"pending"}

    # Explicit S3 prefix propagates verbatim; absent GCS prefix writes nulls.
    assert set(table.column("s3_destination").to_pylist()) == {S3_OUTPUT_PREFIX}
    assert all(value is None for value in table.column("gcs_destination").to_pylist())


def test_create_tracking_parquet_writes_explicit_gcs_destination(tmp_path: Path) -> None:
    master = FIXTURES_DIR / "master.parquet"
    output = tmp_path / "tracking.parquet"

    create_tracking_parquet(
        master,
        output,
        s3_output_prefix=S3_OUTPUT_PREFIX,
        gcs_destination_prefix=GCS_DESTINATION_PREFIX,
    )

    table = pq.read_table(output)
    assert set(table.column("s3_destination").to_pylist()) == {S3_OUTPUT_PREFIX}
    assert set(table.column("gcs_destination").to_pylist()) == {GCS_DESTINATION_PREFIX}


def test_create_tracking_parquet_requires_s3_prefix(tmp_path: Path) -> None:
    master = FIXTURES_DIR / "master.parquet"
    output = tmp_path / "tracking.parquet"

    with pytest.raises(ValueError, match="s3_output_prefix"):
        create_tracking_parquet(master, output, s3_output_prefix="")
    assert not output.exists()


def test_create_tracking_parquet_rejects_malformed_gcs_prefix(tmp_path: Path) -> None:
    master = FIXTURES_DIR / "master.parquet"
    output = tmp_path / "tracking.parquet"

    with pytest.raises(ValueError, match="gcs_destination_prefix"):
        create_tracking_parquet(
            master,
            output,
            s3_output_prefix=S3_OUTPUT_PREFIX,
            gcs_destination_prefix="not-a-uri",
        )
    assert not output.exists()


def test_create_tracking_parquet_no_overwrite(tmp_path: Path) -> None:
    master = FIXTURES_DIR / "master.parquet"
    output = tmp_path / "tracking.parquet"

    create_tracking_parquet(master, output, s3_output_prefix=S3_OUTPUT_PREFIX)

    with pytest.raises(FileExistsError):
        create_tracking_parquet(master, output, s3_output_prefix=S3_OUTPUT_PREFIX)


def test_create_tracking_parquet_force(tmp_path: Path) -> None:
    master = FIXTURES_DIR / "master.parquet"
    output = tmp_path / "tracking.parquet"

    create_tracking_parquet(master, output, s3_output_prefix=S3_OUTPUT_PREFIX)
    # Should not raise with force=True
    n = create_tracking_parquet(master, output, s3_output_prefix=S3_OUTPUT_PREFIX, force=True)
    assert n == 5


def test_update_status(tmp_path: Path) -> None:
    master = FIXTURES_DIR / "master.parquet"
    tracking = tmp_path / "tracking.parquet"
    create_tracking_parquet(master, tracking, s3_output_prefix=S3_OUTPUT_PREFIX)

    count = update_status(
        tracking,
        match_column="predictions_path",
        match_substring="AF-0000000000000001",
        new_status="downloaded",
    )
    assert count == 1

    table = pq.read_table(tracking)
    statuses = dict(
        zip(
            table.column("model_entity_id").to_pylist(),
            table.column("postprocess_status").to_pylist(),
            strict=True,
        )
    )
    assert statuses["AF-0000000000000001"] == "downloaded"
    # Others remain pending
    assert statuses["AF-0000000000000002"] == "pending"


def test_update_status_dry_run(tmp_path: Path) -> None:
    master = FIXTURES_DIR / "master.parquet"
    tracking = tmp_path / "tracking.parquet"
    create_tracking_parquet(master, tracking, s3_output_prefix=S3_OUTPUT_PREFIX)

    count = update_status(
        tracking,
        match_column="source_run",
        match_substring="test_dataset",
        new_status="downloaded",
        dry_run=True,
    )
    assert count == 5

    # Verify nothing changed
    table = pq.read_table(tracking)
    statuses = set(table.column("postprocess_status").to_pylist())
    assert statuses == {"pending"}


def test_query_status(tmp_path: Path) -> None:
    master = FIXTURES_DIR / "master.parquet"
    tracking = tmp_path / "tracking.parquet"
    create_tracking_parquet(master, tracking, s3_output_prefix=S3_OUTPUT_PREFIX)

    counts = query_status(tracking)
    assert counts == {"pending": 5}

    # Update some and re-query
    update_status(
        tracking,
        match_column="model_entity_id",
        match_substring="AF-0000000000000001",
        new_status="downloaded",
    )
    counts = query_status(tracking, dataset="test_dataset")
    assert counts["pending"] == 4
    assert counts["downloaded"] == 1
