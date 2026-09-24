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

"""Aggregate per-shard manifest parquets into a single per-dataset manifest.

Reads shard_*/shard_manifest.parquet files, enriches them with processing
status (success / failed) from failed_models.tsv and S3 upload state from
.uploaded markers, then writes a single dataset-level parquet.

Uses pyarrow exclusively (no pandas) so it handles 10M+ rows with minimal
memory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bspp.orchestration.runtime.postprocessing.sharding import (
    iter_in_range_shard_dirs,
    read_required_shards,
)

logger = logging.getLogger(__name__)

MODEL_ID_COLUMN = "model_entity_id"
UploadedSummary = dict[str, object]

__all__ = [
    "aggregate_dataset",
    "build_aggregated_table",
    "discover_shard_parquets",
    "load_failed_model_ids",
    "load_upload_status",
]


def discover_shard_parquets(output_dir: Path) -> list[Path]:
    """Find all shard_manifest.parquet files under shard_*/ directories.

    Honors ``shard_config.json``'s ``required_shards`` so a stale
    ``shard_N/`` directory left by a prior larger run does not leak into
    the aggregated dataset parquet. If ``shard_config.json`` is absent
    the pre-guard behavior is preserved and a warning is logged.

    Args:
        output_dir: Base directory containing shard_*/ subdirectories.

    Returns:
        Sorted list of paths to shard_manifest.parquet files, one per
        in-range shard that actually carries a manifest.
    """
    required_shards = read_required_shards(output_dir)
    parquets: list[Path] = []
    for _shard_id, shard_dir in iter_in_range_shard_dirs(output_dir, required_shards):
        manifest = shard_dir / "shard_manifest.parquet"
        if manifest.exists():
            parquets.append(manifest)
    return parquets


def load_failed_model_ids(output_dir: Path) -> set[str]:
    """Load model IDs from the global failed_models.tsv.

    The file format is tab-separated: ``model_id\\tstage\\treason``.
    Only the first column (model_id) is extracted.

    Args:
        output_dir: Directory containing ``failed_models.tsv``.

    Returns:
        Set of failed model IDs, or empty set if file does not exist.
    """
    failed_file = output_dir / "failed_models.tsv"
    if not failed_file.exists():
        return set()

    failed_ids: set[str] = set()
    with open(failed_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                failed_ids.add(parts[0])
    return failed_ids


def load_upload_status(output_dir: Path) -> dict[int, UploadedSummary]:
    """Read self-upload marker JSON from in-range shard directories.

    Returns a mapping of ``shard_id -> uploaded_summary``. Malformed or
    incomplete markers are ignored, matching the legacy aggregate script's
    best-effort enrichment behavior.
    """
    required_shards = read_required_shards(output_dir)
    upload_map: dict[int, UploadedSummary] = {}
    for _shard_id, shard_dir in iter_in_range_shard_dirs(output_dir, required_shards):
        uploaded_file = shard_dir / ".uploaded"
        if not uploaded_file.exists():
            continue
        try:
            parsed = json.loads(uploaded_file.read_text())
            if not isinstance(parsed, dict):
                continue
            summary: UploadedSummary = {str(key): value for key, value in parsed.items()}
            marker_shard_id = summary.get("shard_id")
            if isinstance(marker_shard_id, int | str):
                upload_map[int(marker_shard_id)] = summary
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse %s: %s", uploaded_file, exc)
            continue
    return upload_map


def build_aggregated_table(
    shard_parquets: list[Path],
    failed_ids: set[str],
    upload_map: dict[int, UploadedSummary] | None = None,
) -> pa.Table:
    """Read all shard parquets, concatenate, and add enrichment columns.

    Each row is marked ``"failed"`` if its model_entity_id appears in
    *failed_ids*, otherwise ``"success"``. Rows whose ``shard_id`` has a
    ``.uploaded`` marker are also annotated with ``s3_uploaded`` and
    ``s3_prefix``.

    Args:
        shard_parquets: Paths to shard_manifest.parquet files.
        failed_ids: Set of model IDs that failed processing.
        upload_map: Optional mapping from shard ID to parsed ``.uploaded``
            marker payload.

    Returns:
        Combined PyArrow table with added ``status``, ``s3_uploaded``, and
        ``s3_prefix`` columns.

    Raises:
        ValueError: If *shard_parquets* is empty.
    """
    if not shard_parquets:
        msg = "No shard parquets provided"
        raise ValueError(msg)

    tables = [pq.read_table(p) for p in shard_parquets]
    combined = pa.concat_tables(tables, promote_options="default")

    model_ids = combined.column(MODEL_ID_COLUMN)
    is_failed = pc.is_in(model_ids, pa.array(list(failed_ids), pa.string()))
    status_col = pc.if_else(is_failed, "failed", "success")
    combined = combined.append_column("status", status_col)

    s3_uploaded, s3_prefix = _upload_columns(combined, upload_map or {})
    combined = combined.append_column("s3_uploaded", s3_uploaded)
    combined = combined.append_column("s3_prefix", s3_prefix)

    return combined


def _upload_columns(
    table: pa.Table,
    upload_map: dict[int, UploadedSummary],
) -> tuple[pa.Array | pa.ChunkedArray, pa.Array | pa.ChunkedArray]:
    if "shard_id" not in table.schema.names:
        return (
            pa.repeat(pa.scalar(False, pa.bool_()), table.num_rows),
            pa.repeat(pa.scalar("", pa.string()), table.num_rows),
        )

    shard_ids = table.column("shard_id")
    s3_uploaded = pc.is_in(shard_ids, value_set=pa.array(list(upload_map), pa.int64()))
    s3_prefix: pa.Array | pa.ChunkedArray = pa.repeat(pa.scalar("", pa.string()), table.num_rows)
    for shard_id, summary in upload_map.items():
        prefix = str(summary.get("s3_prefix", ""))
        s3_prefix = pc.if_else(
            pc.equal(shard_ids, shard_id),
            pa.scalar(prefix, pa.string()),
            s3_prefix,
        )
    return s3_uploaded, s3_prefix


def aggregate_dataset(
    dataset: str,
    output_base: Path,
    output_path: Path | None = None,
) -> Path:
    """Orchestrate aggregation of shard manifests for a dataset.

    Discovers shard parquets, loads failed model IDs, builds the
    aggregated table, and writes the result.

    Args:
        dataset: Dataset name (subdirectory under *output_base*).
        output_base: Base directory containing ``<dataset>/shard_*/``.
        output_path: Where to write the aggregated parquet. Defaults to
            ``<output_base>/<dataset>_manifest.parquet``.

    Returns:
        Path to the written aggregated parquet file.

    Raises:
        FileNotFoundError: If the dataset directory does not exist.
        FileNotFoundError: If no shard manifests are found.
    """
    output_dir = output_base / dataset
    if not output_dir.exists():
        msg = f"Dataset directory not found: {output_dir}"
        raise FileNotFoundError(msg)

    shard_parquets = discover_shard_parquets(output_dir)
    if not shard_parquets:
        msg = f"No shard_manifest.parquet files found in {output_dir}/shard_*/"
        raise FileNotFoundError(msg)

    failed_ids = load_failed_model_ids(output_dir)
    upload_map = load_upload_status(output_dir)

    logger.info(
        "Aggregating %d shard parquets for dataset %s (%d failed IDs, %d uploaded markers)",
        len(shard_parquets),
        dataset,
        len(failed_ids),
        len(upload_map),
    )

    table = build_aggregated_table(shard_parquets, failed_ids, upload_map)

    if output_path is None:
        output_path = output_base / f"{dataset}_manifest.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")

    n_success = pc.sum(pc.equal(table.column("status"), "success").cast(pa.int64())).as_py()
    n_failed = pc.sum(pc.equal(table.column("status"), "failed").cast(pa.int64())).as_py()
    logger.info(
        "Wrote %s: %d rows (%d success, %d failed)",
        output_path,
        table.num_rows,
        n_success,
        n_failed,
    )

    return output_path
