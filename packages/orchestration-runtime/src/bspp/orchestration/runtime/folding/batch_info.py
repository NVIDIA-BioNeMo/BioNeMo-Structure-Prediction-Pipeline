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

"""Historical folding batch-info normalization.

Port Baseline:
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/scripts/preprocess_trt_bionemo.py:186-249``.
Only aliases supported by that code are accepted. Table-library I/O remains
outside this mapping seam. The port requires an explicit input-row identity for
scalar rows instead of taking the baseline's derive-if-missing branch. Retained
list or array-shaped rows derive identity per path and are flattened per row
rather than according to the first row's shape; their shared scheduling length
is retained while each A3M supplies its own parsed query and chain facts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Integral
from pathlib import Path

from bspp.orchestration.contract.folding_index import FoldingIndex, FoldingIndexRecord, make_folding_index
from bspp.orchestration.runtime.folding.a3m import ParsedA3m, parse_a3m

_PATH_FIELDS = ("msa_path", "a3m_path", "path")
_ALLOWED_FIELDS = frozenset(
    {
        "protein_id",
        *_PATH_FIELDS,
        "sequence",
        "seq_length",
        "chain_count",
        "msa_depth",
        "total_length",
    }
)


def normalize_batch_info(rows: Iterable[Mapping[str, object]], *, sort_by_length: bool = False) -> FoldingIndex:
    """Normalize supported current and historical rows into one index shape."""
    records: list[FoldingIndexRecord] = []
    seen_physical_paths: set[Path] = set()
    for row_index, row in enumerate(rows):
        normalized = _normalize_row(
            row,
            row_index=row_index,
            first_source_ordinal=len(records),
            seen_physical_paths=seen_physical_paths,
        )
        records.extend(normalized)
    if not records:
        msg = "batch-info requires at least one row"
        raise ValueError(msg)
    return make_folding_index(tuple(records), sort_by_length=sort_by_length)


def _normalize_row(
    row: Mapping[str, object],
    *,
    row_index: int,
    first_source_ordinal: int,
    seen_physical_paths: set[Path],
) -> tuple[FoldingIndexRecord, ...]:
    unknown = sorted(set(row) - _ALLOWED_FIELDS)
    if unknown:
        msg = f"batch-info row {row_index} has unknown fields: {', '.join(unknown)}"
        raise ValueError(msg)

    raw_paths = _select_path_value(row, row_index=row_index)
    retained_list = isinstance(raw_paths, tuple)
    scalar_protein_id = ""
    if not retained_list:
        scalar_protein_id = _required_nonempty_str(
            row.get("protein_id"),
            field_name="protein_id",
            row_index=row_index,
        )

    seq_length = _optional_positive_int(row.get("seq_length"), field_name="seq_length", row_index=row_index)
    total_length = _optional_positive_int(row.get("total_length"), field_name="total_length", row_index=row_index)
    if seq_length is None and total_length is None:
        msg = f"batch-info row {row_index} requires seq_length or total_length"
        raise ValueError(msg)
    canonical_length = seq_length if seq_length is not None else total_length
    if canonical_length is None:
        msg = f"batch-info row {row_index} requires a usable scheduling length"
        raise ValueError(msg)

    if retained_list:
        # The baseline flattening path drops every other column before adding
        # the default, so even a declared group depth is not per-protein depth.
        msa_depth = 1
    else:
        declared_msa_depth = _optional_positive_int(row.get("msa_depth"), field_name="msa_depth", row_index=row_index)
        msa_depth = 1 if declared_msa_depth is None else declared_msa_depth
    paths = _normalize_paths(
        raw_paths,
        row_index=row_index,
        seen_physical_paths=seen_physical_paths,
    )

    records: list[FoldingIndexRecord] = []
    for path_index, path in enumerate(paths):
        parsed = parse_a3m(path)
        if not retained_list:
            _validate_declared_metadata(
                row,
                parsed=parsed,
                row_index=row_index,
                seq_length=seq_length,
                total_length=total_length,
            )
        canonical_protein_id = path.stem if retained_list else scalar_protein_id
        records.append(
            FoldingIndexRecord(
                source_ordinal=first_source_ordinal + path_index,
                protein_id=canonical_protein_id,
                msa_path=str(path),
                query_sequence=parsed.query_sequence,
                sequence_length=canonical_length,
                chain_lengths=parsed.chain_lengths,
                chain_count=parsed.chain_count,
                chain_cardinalities=parsed.chain_cardinalities,
                msa_depth=msa_depth,
                total_length=parsed.total_length,
            )
        )
    return tuple(records)


def _select_path_value(row: Mapping[str, object], *, row_index: int) -> str | tuple[str, ...]:
    for field_name in _PATH_FIELDS:
        normalized = _coerce_path_value(row.get(field_name), field_name=field_name, row_index=row_index)
        if normalized is not None:
            return normalized
    msg = f"batch-info row {row_index} requires a usable path column: {', '.join(_PATH_FIELDS)}"
    raise ValueError(msg)


def _coerce_path_value(
    value: object,
    *,
    field_name: str,
    row_index: int,
) -> str | tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, str | list | tuple):
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, list | tuple):
        if not value:
            return None
        if not all(isinstance(item, str) and bool(item.strip()) for item in value):
            msg = f"batch-info row {row_index} {field_name} must contain only non-empty paths"
            raise ValueError(msg)
        return tuple(value)
    msg = f"batch-info row {row_index} {field_name} must be a string or non-empty path list"
    raise ValueError(msg)


def _normalize_paths(
    value: object,
    *,
    row_index: int,
    seen_physical_paths: set[Path],
) -> tuple[Path, ...]:
    if isinstance(value, str):
        raw_paths = (value,)
    elif isinstance(value, list | tuple):
        raw_paths = tuple(value)
    else:
        msg = f"batch-info row {row_index} path must be a string or non-empty path list"
        raise ValueError(msg)

    paths: list[Path] = []
    for path_index, raw_path in enumerate(raw_paths):
        if not isinstance(raw_path, str) or not raw_path.strip():
            msg = f"batch-info row {row_index} path {path_index} must be a non-empty string"
            raise ValueError(msg)
        path = Path(raw_path)
        try:
            exists = path.exists()
            is_file = path.is_file()
        except OSError as exc:
            msg = f"Cannot inspect batch-info row {row_index} A3M path {path}: {exc}"
            raise ValueError(msg) from exc
        if not exists:
            msg = f"batch-info row {row_index} A3M path does not exist: {path}"
            raise ValueError(msg)
        if not is_file:
            msg = f"batch-info row {row_index} A3M path is not a readable file: {path}"
            raise ValueError(msg)
        try:
            physical_path = path.resolve(strict=True)
        except OSError as exc:
            msg = f"Cannot resolve batch-info row {row_index} A3M path {path}: {exc}"
            raise ValueError(msg) from exc
        if physical_path in seen_physical_paths:
            msg = f"batch-info row {row_index} repeats the same physical A3M: {path}"
            raise ValueError(msg)
        seen_physical_paths.add(physical_path)
        paths.append(path)
    return tuple(paths)


def _validate_declared_metadata(
    row: Mapping[str, object],
    *,
    parsed: ParsedA3m,
    row_index: int,
    seq_length: int | None,
    total_length: int | None,
) -> None:
    if seq_length is not None and seq_length != parsed.sequence_length:
        msg = (
            f"batch-info row {row_index} seq_length {seq_length} does not match "
            f"A3M header length {parsed.sequence_length}"
        )
        raise ValueError(msg)
    if total_length is not None and total_length != parsed.total_length:
        msg = (
            f"batch-info row {row_index} total_length {total_length} does not match "
            f"A3M header total {parsed.total_length}"
        )
        raise ValueError(msg)

    sequence = row.get("sequence")
    if sequence is not None and (not isinstance(sequence, str) or sequence != parsed.query_sequence):
        msg = f"batch-info row {row_index} sequence does not match the A3M query sequence"
        raise ValueError(msg)

    chain_count = row.get("chain_count")
    if chain_count is not None:
        parsed_chain_count = _optional_positive_int(chain_count, field_name="chain_count", row_index=row_index)
        if parsed_chain_count != parsed.chain_count:
            msg = f"batch-info row {row_index} chain_count does not match the A3M header"
            raise ValueError(msg)


def _required_nonempty_str(value: object, *, field_name: str, row_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"batch-info row {row_index} requires non-empty {field_name}"
        raise ValueError(msg)
    return value


def _optional_positive_int(value: object, *, field_name: str, row_index: int) -> int | None:
    if value is None:
        return None
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        msg = f"batch-info row {row_index} {field_name} must be a positive integer"
        raise ValueError(msg)
    return int(value)


__all__ = ["normalize_batch_info"]
