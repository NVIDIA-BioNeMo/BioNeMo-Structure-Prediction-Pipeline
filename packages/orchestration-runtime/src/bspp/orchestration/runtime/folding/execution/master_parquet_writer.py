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

"""Master-parquet writer for the folding companion-engine track.

This producer builds one master-parquet row against the frozen Phase-0
``MASTER_PARQUET_PROJECTION`` registry and publishes a row only when all 14
required fields are materially populated. The row is the full
35-column union derived from the registry: required columns carry
the caller's resolved value, and optional columns carry the caller's override
or the registry default.

Serialization is pyarrow-only; no pandas/cudf is introduced.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bspp.orchestration.contract.master_parquet_projection import (
    MASTER_PARQUET_PROJECTION,
    validate_projection_columns,
)

__all__ = ["build_master_parquet_row", "write_master_parquet"]

_REQUIRED_FIELDS = tuple(column.name for column in MASTER_PARQUET_PROJECTION if column.required)
_OPTIONAL_FIELDS = frozenset(column.name for column in MASTER_PARQUET_PROJECTION if not column.required)

# Engine-optional scores: these required projection columns may be a RESOLVED
# null (the engine produces no such score — BioIR) rather than a
# PENDING value. A ``None`` for any other required column is pending and
# withholds the row.
_ENGINE_OPTIONAL_NULLABLE_SCORES = frozenset({"ptm", "iptm"})

_DTYPE_MAP: dict[str, object] = {
    "string": pa.string(),
    "Int64": pa.int64(),
    "Float64": pa.float64(),
}


def build_master_parquet_row(
    *,
    msa_path: str | None,
    source_run: str | None,
    pdb_path: str | None,
    json_path: str | None,
    pdb_residue_count: int | None,
    mean_plddt: float | None,
    plddt_above_70: float | None,
    ptm: float | None,
    iptm: float | None,
    max_pae: float | None,
    output_has_nan: str | None,
    pdb_json_match: str | None,
    swiftstack_archive: str | None,
    uploaded_to_gcp: str | None,
    **optional: object,
) -> dict[str, object] | None:
    """Build one master-parquet row, withholding it when any required field is pending.

    A required field is pending iff the caller passes ``None`` and the field is
    not engine-optional. ``ptm`` and ``iptm`` are engine-optional scores: a
    ``None`` there is a RESOLVED null (the engine produces no such score —
    BioIR) and the row is emitted with a Float64 null for that
    column. Every other required field treats ``None`` as pending (withheld,
    returns ``None``). Any non-``None`` value (including the legal defaults
    ``''`` and ``'no'``) is a resolved answer. Registry defaults are never used
    to resolve a required field.

    Optional registry fields are defaulted from the registry and may be
    overridden via ``**optional``; an unknown optional key fails closed.
    """
    required_values: dict[str, object] = {
        "msa_path": msa_path,
        "source_run": source_run,
        "pdb_path": pdb_path,
        "json_path": json_path,
        "pdb_residue_count": pdb_residue_count,
        "mean_plddt": mean_plddt,
        "plddt_above_70": plddt_above_70,
        "ptm": ptm,
        "iptm": iptm,
        "max_pae": max_pae,
        "output_has_nan": output_has_nan,
        "pdb_json_match": pdb_json_match,
        "swiftstack_archive": swiftstack_archive,
        "uploaded_to_gcp": uploaded_to_gcp,
    }

    if any(value is None and name not in _ENGINE_OPTIONAL_NULLABLE_SCORES for name, value in required_values.items()):
        return None

    unknown = sorted(set(optional) - _OPTIONAL_FIELDS)
    if unknown:
        raise ValueError(f"unknown master parquet projection column(s): {', '.join(unknown)}")

    row: dict[str, object] = {}
    for column in MASTER_PARQUET_PROJECTION:
        if column.required:
            row[column.name] = required_values[column.name]
        elif column.name in optional:
            row[column.name] = optional[column.name]
        else:
            row[column.name] = column.default
    return row


def _projection_schema() -> pa.Schema:
    """Build the full-union pyarrow schema in registry order with mapped dtypes."""
    return pa.schema([pa.field(column.name, _DTYPE_MAP[column.dtype]) for column in MASTER_PARQUET_PROJECTION])


def write_master_parquet(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Serialize complete master-parquet rows via pyarrow.

    Every row must already have passed the complete-row gate. The produced
    column set is the full registry union, so ``validate_projection_columns``
    passes (all 14 required present, no unknown or duplicate columns). A row
    whose required registry field is ``None`` is rejected before the output
    directory or parquet file is created, except that ``ptm``/``iptm`` may be
    ``None`` (engine-optional resolved null). Empty ``rows`` writes a zero-row
    table with the full schema.
    """
    for row in rows:
        validate_projection_columns(row.keys())
        pending = [
            name for name in _REQUIRED_FIELDS if row.get(name) is None and name not in _ENGINE_OPTIONAL_NULLABLE_SCORES
        ]
        if pending:
            msg = f"master parquet row has pending required field(s): {', '.join(pending)}"
            raise ValueError(msg)

    table = pa.Table.from_pylist([dict(row) for row in rows], schema=_projection_schema())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
