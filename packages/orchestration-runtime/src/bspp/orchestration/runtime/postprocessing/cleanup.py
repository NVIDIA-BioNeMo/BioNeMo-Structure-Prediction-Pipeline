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

"""Local cleanup of shard output directories.

Removes shard_*/success_outputs/ directories after successful upload,
preserving shard metadata (failed_models.tsv, batch logs, etc.).

Safety: refuses to run unless every row for the dataset in the tracking
parquet has a safe status (uploaded_s3, uploaded_gcs, done), unless
``force=True``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SAFE_STATUSES: frozenset[str] = frozenset({"uploaded_s3", "uploaded_gcs", "done"})

__all__ = [
    "SAFE_STATUSES",
    "cleanup_shard_outputs",
]


def _verify_tracking_safety(tracking_path: Path, dataset: str | None) -> None:
    """Verify all relevant rows in the tracking parquet have safe statuses.

    Raises:
        FileNotFoundError: If *tracking_path* does not exist.
        RuntimeError: If any rows have unsafe statuses.
    """
    if not tracking_path.exists():
        msg = f"Tracking parquet not found: {tracking_path}"
        raise FileNotFoundError(msg)

    table = pq.read_table(tracking_path)
    if dataset is not None:
        ds_col = table.column("dataset_name")
        mask = pc.equal(ds_col, dataset)
        table = table.filter(mask)

    if table.num_rows == 0:
        msg = f"No rows found for dataset {dataset!r} in tracking parquet"
        raise RuntimeError(msg)

    status_col = table.column("postprocess_status")
    unique_statuses = set(pc.unique(status_col).to_pylist())
    unsafe = unique_statuses - SAFE_STATUSES
    if unsafe:
        msg = f"Cannot clean up: {len(unsafe)} unsafe status(es) found: {sorted(unsafe)}. Use force=True to bypass."
        raise RuntimeError(msg)


def cleanup_shard_outputs(
    output_dir: Path,
    *,
    force: bool = False,
    tracking_path: Path | None = None,
    dataset: str | None = None,
    parallel: int = 1,
) -> int:
    """Remove shard_*/success_outputs/ directories.

    If ``force`` is ``False``, verifies all rows in the tracking parquet
    have safe status before removing anything.

    Args:
        output_dir: Directory containing ``shard_*/`` subdirectories.
        force: Skip tracking safety check.
        tracking_path: Path to tracking parquet (required if not force).
        dataset: Dataset name for tracking filter.
        parallel: Number of parallel delete workers (reserved for future
            use; currently sequential).

    Returns:
        Number of shard directories cleaned.

    Raises:
        RuntimeError: If not force and tracking check fails.
        ValueError: If not force and tracking_path is not provided.
    """
    if not force:
        if tracking_path is None:
            msg = "tracking_path is required when force=False"
            raise ValueError(msg)
        _verify_tracking_safety(tracking_path, dataset)

    cleaned = 0
    for shard_dir in sorted(output_dir.glob("shard_*")):
        if not shard_dir.is_dir():
            continue
        success_dir = shard_dir / "success_outputs"
        if success_dir.is_dir():
            shutil.rmtree(success_dir)
            logger.info("Removed %s", success_dir)
            cleaned += 1

    logger.info("Cleaned %d shard directories in %s", cleaned, output_dir)
    return cleaned
