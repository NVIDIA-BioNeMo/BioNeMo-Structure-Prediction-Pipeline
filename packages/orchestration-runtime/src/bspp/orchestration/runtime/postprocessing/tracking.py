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

"""Tracking parquet lifecycle management.

Creates and updates a tracking parquet that records each model's progress
through the post-processing pipeline:

    pending -> downloaded -> extracted -> recipe_ready -> preprocessed
    -> processing -> processed -> uploaded_s3 -> uploaded_gcs -> done

Uses pyarrow exclusively (no pandas) for memory-efficient handling of
large datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bspp.orchestration.runtime.constants import locked_parquet

logger = logging.getLogger(__name__)

LIFECYCLE_STATUSES: tuple[str, ...] = (
    "pending",
    "downloaded",
    "extracted",
    "recipe_ready",
    "preprocessed",
    "processing",
    "processed",
    "uploaded_s3",
    "uploaded_gcs",
    "done",
)

__all__ = [
    "LIFECYCLE_STATUSES",
    "create_tracking_parquet",
    "query_status",
    "update_status",
]


def create_tracking_parquet(
    master_path: Path,
    output_path: Path,
    *,
    s3_output_prefix: str,
    gcs_destination_prefix: str | None = None,
    force: bool = False,
) -> int:
    """Build a tracking parquet from the master parquet.

    Reads the master parquet and appends lifecycle columns:
    ``postprocess_status`` (set to ``"pending"``), timestamp columns
    (null), ``s3_destination``, ``gcs_destination``, and
    ``needs_archive_resolution``.

    Args:
        master_path: Path to the master parquet file.
        output_path: Where to write the tracking parquet.
        s3_output_prefix: Explicit ``s3://`` destination prefix;
            there is no legacy internal default.
        gcs_destination_prefix: Optional explicit ``gs://`` destination
            prefix. When ``None`` the GCS destination column is written as
            null rather than synthesizing a bucket.
        force: If ``True``, overwrite existing output. Otherwise raise
            :class:`FileExistsError`.

    Returns:
        Number of rows in the tracking parquet.

    Raises:
        FileNotFoundError: If *master_path* does not exist.
        FileExistsError: If *output_path* exists and *force* is ``False``.
        ValueError: If *s3_output_prefix* is not a non-empty ``s3://`` URI.
    """
    if not master_path.exists():
        msg = f"Master parquet not found: {master_path}"
        raise FileNotFoundError(msg)

    if output_path.exists() and not force:
        msg = f"Output file already exists: {output_path} (use force=True to overwrite)"
        raise FileExistsError(msg)

    s3_output_prefix = _require_s3_output_prefix(s3_output_prefix)
    if gcs_destination_prefix is not None:
        gcs_destination_prefix = _require_gcs_destination_prefix(gcs_destination_prefix)

    master_table = pq.read_table(master_path)
    n_rows = master_table.num_rows

    source_run = master_table.column("source_run")

    # Determine archive resolution needs
    s3_archive_col = master_table.column("swiftstack_archive")
    has_archive = pc.and_(
        pc.is_valid(s3_archive_col),
        pc.not_equal(s3_archive_col, ""),
    )
    needs_resolution = pc.invert(has_archive)

    # Build s3 destination from the explicit prefix.
    s3_destinations: list[str] = [s3_output_prefix] * n_rows

    null_ts = pa.array([None] * n_rows, type=pa.timestamp("us", tz="UTC"))

    tracking_table = master_table.append_column("dataset_name", source_run)
    tracking_table = tracking_table.append_column(
        "postprocess_status",
        pa.array(["pending"] * n_rows),
    )
    tracking_table = tracking_table.append_column("extract_timestamp", null_ts)
    tracking_table = tracking_table.append_column("postprocess_timestamp", null_ts)
    tracking_table = tracking_table.append_column("s3_upload_timestamp", null_ts)
    tracking_table = tracking_table.append_column("gcs_upload_timestamp", null_ts)
    tracking_table = tracking_table.append_column(
        "s3_destination",
        pa.array(s3_destinations),
    )
    tracking_table = tracking_table.append_column(
        "gcs_destination",
        pa.array([gcs_destination_prefix] * n_rows, type=pa.string()),
    )
    tracking_table = tracking_table.append_column(
        "needs_archive_resolution",
        needs_resolution,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tracking_table, output_path)

    logger.info("Wrote tracking parquet: %s (%d rows)", output_path, n_rows)
    return int(n_rows)


def _require_s3_output_prefix(value: str) -> str:
    if not value or not value.startswith("s3://"):
        msg = f"s3_output_prefix must be a non-empty s3:// URI, got {value!r}"
        raise ValueError(msg)
    return value


def _require_gcs_destination_prefix(value: str) -> str:
    if not value or not value.startswith("gs://"):
        msg = f"gcs_destination_prefix must be a non-empty gs:// URI, got {value!r}"
        raise ValueError(msg)
    return value


def update_status(
    tracking_path: Path,
    *,
    match_column: str,
    match_substring: str,
    new_status: str,
    dry_run: bool = False,
) -> int:
    """Bulk-update status in the tracking parquet by column substring match.

    Reads the tracking parquet, finds rows where *match_column* contains
    *match_substring*, and sets their ``postprocess_status`` to *new_status*.

    Args:
        tracking_path: Path to the tracking parquet.
        match_column: Column name to match against.
        match_substring: Substring to search for in the match column.
        new_status: New ``postprocess_status`` value for matched rows.
        dry_run: If ``True``, count matches but do not write changes.

    Returns:
        Number of rows that matched.

    Raises:
        FileNotFoundError: If *tracking_path* does not exist.
        ValueError: If *new_status* is not in :data:`LIFECYCLE_STATUSES`.
    """
    if not tracking_path.exists():
        msg = f"Tracking parquet not found: {tracking_path}"
        raise FileNotFoundError(msg)

    if new_status not in LIFECYCLE_STATUSES:
        msg = f"Invalid status {new_status!r}; must be one of {LIFECYCLE_STATUSES}"
        raise ValueError(msg)

    with locked_parquet(tracking_path) as p:
        table = pq.read_table(p)

        col = table.column(match_column)
        mask = pc.match_substring(col, match_substring)
        matched_count = int(pc.sum(mask.cast("int64")).as_py())

        if matched_count == 0 or dry_run:
            return matched_count

        current_status = table.column("postprocess_status")
        new_status_col = pc.if_else(mask, new_status, current_status)
        idx = table.schema.get_field_index("postprocess_status")
        table = table.set_column(idx, "postprocess_status", new_status_col)

        pq.write_table(table, p, compression="snappy")

    logger.info("Updated %d rows to status=%r", matched_count, new_status)
    return matched_count


def query_status(
    tracking_path: Path,
    dataset: str | None = None,
) -> dict[str, int]:
    """Query status counts from the tracking parquet.

    Args:
        tracking_path: Path to the tracking parquet.
        dataset: If provided, filter to rows matching this dataset_name.
            If ``None``, count all rows.

    Returns:
        Dict mapping status values to their counts.

    Raises:
        FileNotFoundError: If *tracking_path* does not exist.
    """
    if not tracking_path.exists():
        msg = f"Tracking parquet not found: {tracking_path}"
        raise FileNotFoundError(msg)

    table = pq.read_table(tracking_path)

    if dataset is not None:
        ds_col = table.column("dataset_name")
        mask = pc.equal(ds_col, dataset)
        table = table.filter(mask)

    status_col = table.column("postprocess_status")
    counts: dict[str, int] = {}
    for status in pc.unique(status_col).to_pylist():
        n = int(pc.sum(pc.equal(status_col, status).cast("int64")).as_py())
        counts[status] = n

    return counts
