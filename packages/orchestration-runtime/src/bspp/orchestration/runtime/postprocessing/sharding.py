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

"""Shard distribution and per-shard input staging.

Provides contiguous shard slice computation, symlink-based shard directory
creation, manifest filtering per shard, and shard manifest persistence as
Parquet.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections.abc import Iterator
from pathlib import Path

from bspp.orchestration.runtime.slurm.arrays import array_task_count

logger = logging.getLogger(__name__)

__all__ = [
    "InvalidShardConfigError",
    "compute_shard_slice",
    "create_symlink_shard",
    "filter_manifest_for_shard",
    "iter_in_range_shard_dirs",
    "persist_shard_manifest",
    "read_required_shards",
]


class InvalidShardConfigError(RuntimeError):
    """``shard_config.json`` is present but unreadable or semantically invalid.

    Raised by :func:`read_required_shards` so downstream consumers
    (upload, aggregate, coverage) fail closed instead of silently
    reverting to the unguarded ``shard_*`` glob. A truncated or
    corrupted config must not reopen the stale-dir contamination path
    the helper is there to prevent.
    """


def _parse_array_range_count(array_range: object) -> int | None:
    """Return how many shard ids ``"0-99"`` or ``"0-99%16"`` covers, or ``None``."""
    if not isinstance(array_range, str):
        return None
    try:
        return array_task_count(array_range)
    except ValueError:
        return None


def _validate_shard_config_semantics(payload: dict[str, object], path: Path, required: int) -> None:
    """Cross-check ``required_shards`` against the rest of the shard config.

    The earlier guard only verified type/range, so a stale config like
    ``{"required_shards": 1}`` on a 10-shard dataset still passed and the
    consumers silently dropped real shards as "stale". Here we tie the
    field to the rest of the config so a tampered or out-of-date file
    cannot be used as the stale-shard authority.
    """
    array_count = _parse_array_range_count(payload.get("array_range"))
    if isinstance(payload.get("array_range"), str) and array_count is None:
        msg = f"{path}: array_range is invalid: {payload.get('array_range')}"
        raise InvalidShardConfigError(msg)
    if array_count is not None and array_count != required:
        msg = f"{path}: array_range covers {array_count} shards but required_shards={required}"
        raise InvalidShardConfigError(msg)

    archive_mode = payload.get("archive_mode") is True
    if archive_mode:
        total_archives = payload.get("total_archives")
        if isinstance(total_archives, int) and total_archives != required:
            msg = f"{path}: archive mode total_archives={total_archives} does not match required_shards={required}"
            raise InvalidShardConfigError(msg)
    else:
        total_models = payload.get("total_models")
        max_per_shard = payload.get("max_per_shard")
        if (
            isinstance(total_models, int)
            and isinstance(max_per_shard, int)
            and not isinstance(total_models, bool)
            and not isinstance(max_per_shard, bool)
            and max_per_shard > 0
            and total_models >= 0
        ):
            expected = max(1, math.ceil(total_models / max_per_shard))
            if expected != required:
                msg = (
                    f"{path}: ceil(total_models={total_models} / "
                    f"max_per_shard={max_per_shard}) = {expected} but "
                    f"required_shards={required}"
                )
                raise InvalidShardConfigError(msg)


def read_required_shards(dataset_output_dir: Path) -> int | None:
    """Read ``required_shards`` from ``dataset_output_dir/shard_config.json``.

    Returns:
        - ``int`` when the config is present, parseable, and **internally
          consistent** (semantic cross-checks below all hold).
        - ``None`` **only** when the config file is absent — this is the
          legacy back-compat case (a dataset directory that pre-dates the
          stale-shard guard). Callers that receive ``None`` fall back to
          the unguarded ``shard_*`` glob with a warning.

    Raises:
        InvalidShardConfigError: The config file exists but is unreadable,
            not valid JSON, missing ``required_shards``, has a
            non-positive / non-integer ``required_shards`` value, **or**
            disagrees with itself (``array_range`` width, ``total_archives``
            in archive mode, or ``ceil(total_models / max_per_shard)`` in
            flat mode all must match ``required_shards``). We fail closed
            here so a tampered or stale config cannot quietly drop real
            shards as "out of range".
    """
    shard_config = dataset_output_dir / "shard_config.json"
    if not shard_config.exists():
        return None
    try:
        raw = shard_config.read_text()
    except OSError as exc:
        msg = f"Cannot read {shard_config}: {exc}"
        raise InvalidShardConfigError(msg) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"Cannot parse {shard_config}: {exc}"
        raise InvalidShardConfigError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"{shard_config} is not a JSON object (got {type(payload).__name__})"
        raise InvalidShardConfigError(msg)
    value = payload.get("required_shards")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"Invalid required_shards in {shard_config}: {value!r}"
        raise InvalidShardConfigError(msg)
    _validate_shard_config_semantics(payload, shard_config, value)
    return value


def iter_in_range_shard_dirs(
    dataset_output_dir: Path,
    required_shards: int | None = None,
) -> Iterator[tuple[int, Path]]:
    """Yield ``(shard_id, shard_dir)`` pairs for on-disk shards in the expected range.

    ``shard_*`` directories whose parsed id is ``>= required_shards`` are
    **skipped** because they are stale leftovers from a prior, larger
    run (see :func:`bspp.orchestration.runtime.validation.count_outputs.count_shard_outputs`
    for the anomaly bookkeeping). All downstream consumers that glob
    ``shard_*`` must use this helper — otherwise contaminated datasets
    would leak into uploads, aggregation, or coverage analysis.

    If *required_shards* is ``None``, the caller is opting into the
    pre-guard behavior: every on-disk shard dir is yielded. A warning
    log makes that case visible.
    """
    if required_shards is None:
        required_shards_effective = None
        logger.warning(
            "iter_in_range_shard_dirs called without required_shards for %s; "
            "stale-dir guard is disabled for this invocation",
            dataset_output_dir,
        )
    else:
        required_shards_effective = required_shards

    for entry in sorted(dataset_output_dir.glob("shard_*")):
        if not entry.is_dir():
            continue
        suffix = entry.name.removeprefix("shard_")
        if not suffix.isdigit():
            continue
        shard_id = int(suffix)
        if required_shards_effective is not None and shard_id >= required_shards_effective:
            logger.debug(
                "Skipping stale %s (shard_id %d >= required_shards %d)",
                entry,
                shard_id,
                required_shards_effective,
            )
            continue
        yield shard_id, entry


def compute_shard_slice(
    shard_id: int,
    total_items: int,
    num_shards: int,
) -> tuple[int, int]:
    """Return ``(start, end)`` indices for a contiguous shard slice.

    Distributes *total_items* across *num_shards* so that the first
    ``total_items % num_shards`` shards each get one extra item.  This
    matches the distribution algorithm used by colabfold-slurm's
    ``shard_master_queue``.

    Args:
        shard_id: Zero-based shard index.
        total_items: Total number of items to distribute.
        num_shards: Number of shards.

    Returns:
        A ``(start, end)`` tuple suitable for slicing: ``items[start:end]``.

    Raises:
        ValueError: If *shard_id* is out of range or inputs are invalid.
    """
    if num_shards <= 0:
        msg = f"num_shards must be positive, got {num_shards}"
        raise ValueError(msg)
    if total_items < 0:
        msg = f"total_items must be non-negative, got {total_items}"
        raise ValueError(msg)
    if shard_id < 0 or shard_id >= num_shards:
        msg = f"shard_id {shard_id} out of range [0, {num_shards})"
        raise ValueError(msg)

    items_per_shard = total_items // num_shards
    remainder = total_items % num_shards

    if shard_id < remainder:
        start = shard_id * (items_per_shard + 1)
        end = start + items_per_shard + 1
    else:
        start = shard_id * items_per_shard + remainder
        end = start + items_per_shard

    return start, end


def create_symlink_shard(
    model_ids: list[str],
    file_index: dict[str, list[str]],
    input_dir: Path,
    shard_dir: Path,
) -> int:
    """Create symlinks in *shard_dir* pointing to files in *input_dir*.

    For each model ID in *model_ids*, looks up filenames in *file_index*
    and creates symlinks ``shard_dir/<filename> -> input_dir/<filename>``.
    Existing symlinks are silently skipped.

    Args:
        model_ids: Model IDs whose files should be symlinked.
        file_index: Mapping from model ID to list of filenames.
        input_dir: Directory containing the original files.
        shard_dir: Destination directory for symlinks.

    Returns:
        Number of symlinks created.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    linked = 0

    for mid in model_ids:
        filenames = file_index.get(mid)
        if not filenames:
            continue
        for fname in filenames:
            target = shard_dir / fname
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(input_dir / fname)
            linked += 1

    return linked


def filter_manifest_for_shard(
    manifest_path: Path,
    shard_model_ids: set[str],
    output_path: Path,
) -> int:
    """Filter a manifest CSV to only rows for the shard's models.

    Reads the CSV at *manifest_path*, keeps rows whose ``model_entity_id``
    is in *shard_model_ids*, and writes the result to *output_path*.

    Args:
        manifest_path: Path to the full manifest CSV.
        shard_model_ids: Set of model IDs belonging to this shard.
        output_path: Where to write the filtered CSV.

    Returns:
        Number of data rows written (excluding header).

    Raises:
        ValueError: If the manifest CSV has no header.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with manifest_path.open(newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            msg = f"Manifest CSV has no header: {manifest_path}"
            raise ValueError(msg)

        with output_path.open("w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row.get("model_entity_id") in shard_model_ids:
                    writer.writerow(row)
                    rows_written += 1

    return rows_written


def persist_shard_manifest(
    manifest_path: Path,
    shard_id: int,
    dataset_tag: str,
    output_path: Path,
) -> None:
    """Read a CSV manifest, add shard metadata columns, and write as Parquet.

    Appends ``shard_id`` (int32) and ``dataset_tag`` (string) columns to the
    table read from *manifest_path*, then writes the result as a Snappy-
    compressed Parquet file to *output_path*.

    Args:
        manifest_path: Path to the shard's CSV manifest.
        shard_id: Numeric shard identifier to embed.
        dataset_tag: Dataset name/tag to embed.
        output_path: Destination Parquet file path.
    """
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    import pyarrow.parquet as pq

    table = pa_csv.read_csv(str(manifest_path))
    n = table.num_rows
    table = table.append_column(
        "shard_id",
        pa.array([shard_id] * n, pa.int32()),
    )
    table = table.append_column(
        "dataset_tag",
        pa.array([dataset_tag] * n, pa.string()),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")
    logger.info("Persisted shard manifest: %s (%d rows)", output_path, n)
