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

"""Typed registry for the complete harvested master-parquet projection.

The master parquet is produced by two harvested producers:

* ColabFold ``create_master_parquet.py`` (key + core + provenance columns);
* OpenFold-TRT ``archive_openfold_results.py`` (``STRING_COLUMNS`` and
  ``NUMERIC_COLUMNS``, the archive/transfer detail surface).

This module records their complete union as an immutable, typed column
registry: every known column carries its harvested logical dtype and default.
``required=True`` marks the frozen postprocessing compatibility read surface
consumed by ``converter.py`` and
``validate_parquet_outputs.py::VALIDATION_COLUMNS``; all producer detail
columns remain optional so producer rows written before upload round-trip
faithfully.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

MasterParquetDtype = Literal["string", "Int64", "Float64"]

_VALID_DTYPES = ("string", "Int64", "Float64")
_VALID_DTYPES_TEXT = ", ".join(_VALID_DTYPES)


@dataclass(frozen=True)
class MasterParquetColumn:
    """One typed column of the harvested master-parquet projection."""

    name: str
    dtype: MasterParquetDtype
    default: str | int | float | None
    required: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("master parquet column name must be non-empty")
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(f"unsupported master parquet dtype {self.dtype!r}; expected one of {_VALID_DTYPES_TEXT}")
        if not isinstance(self.required, bool):
            raise ValueError("master parquet column required must be a boolean")
        _validate_default(self.default, self.dtype, self.name)

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "default": self.default,
            "required": self.required,
        }


def master_parquet_column_from_mapping(payload: Mapping[str, object]) -> MasterParquetColumn:
    """Load a strict master-parquet column record from a serialized mapping."""
    _strict(payload, {"name", "dtype", "default", "required"}, "MasterParquetColumn")
    dtype_value = payload.get("dtype")
    if dtype_value not in _VALID_DTYPES:
        raise ValueError(f"unsupported master parquet dtype {dtype_value!r}; expected one of {_VALID_DTYPES_TEXT}")
    required = payload.get("required")
    if not isinstance(required, bool):
        raise ValueError("required must be a boolean")
    return MasterParquetColumn(
        name=_str(payload, "name"),
        dtype=cast(MasterParquetDtype, dtype_value),
        default=_scalar_default(payload.get("default")),
        required=required,
    )


def _strict(payload: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")
    missing = sorted(allowed - set(payload))
    if missing:
        raise ValueError(f"Missing {name} field(s): {', '.join(missing)}")


def _str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _scalar_default(value: object) -> str | int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError("default must be a string, number, or null")
    return value


def _validate_default(default: str | int | float | None, dtype: MasterParquetDtype, name: str) -> None:
    if default is None:
        return
    if isinstance(default, bool):
        raise ValueError(f"{name} default must not be a boolean")
    if dtype == "string" and not isinstance(default, str):
        raise ValueError(f"{name} default must be a string or null for dtype 'string'")
    if dtype == "Int64" and not isinstance(default, int):
        raise ValueError(f"{name} default must be an integer or null for dtype 'Int64'")
    if dtype == "Float64" and not isinstance(default, int | float):
        raise ValueError(f"{name} default must be numeric or null for dtype 'Float64'")


MASTER_PARQUET_PROJECTION: tuple[MasterParquetColumn, ...] = (
    # Key
    MasterParquetColumn(name="msa_path", dtype="string", default=None, required=True),
    # Core
    MasterParquetColumn(name="seq_length", dtype="Int64", default=None, required=False),
    MasterParquetColumn(name="seq_count", dtype="Int64", default=None, required=False),
    MasterParquetColumn(name="sequence", dtype="string", default=None, required=False),
    # Provenance
    MasterParquetColumn(name="source_run", dtype="string", default="", required=True),
    MasterParquetColumn(name="source_parquet", dtype="string", default="", required=False),
    MasterParquetColumn(name="predictions_path", dtype="string", default="", required=False),
    # Payload
    MasterParquetColumn(name="pdb_path", dtype="string", default="", required=True),
    MasterParquetColumn(name="json_path", dtype="string", default="", required=True),
    # Quality
    MasterParquetColumn(name="pdb_residue_count", dtype="Int64", default=None, required=True),
    MasterParquetColumn(name="mean_plddt", dtype="Float64", default=None, required=True),
    MasterParquetColumn(name="plddt_above_70", dtype="Float64", default=None, required=True),
    MasterParquetColumn(name="ptm", dtype="Float64", default=None, required=True),
    MasterParquetColumn(name="iptm", dtype="Float64", default=None, required=True),
    MasterParquetColumn(name="max_pae", dtype="Float64", default=None, required=True),
    MasterParquetColumn(name="output_has_nan", dtype="string", default="not_checked", required=True),
    MasterParquetColumn(name="pdb_json_match", dtype="string", default="not_checked", required=True),
    # Archive detail
    MasterParquetColumn(name="archive_status", dtype="string", default="", required=False),
    MasterParquetColumn(name="archive_batch_id", dtype="string", default="", required=False),
    MasterParquetColumn(name="archive_file", dtype="string", default="", required=False),
    MasterParquetColumn(name="archive_created_at", dtype="string", default="", required=False),
    MasterParquetColumn(name="archive_sha256", dtype="string", default="", required=False),
    MasterParquetColumn(name="local_archive_path", dtype="string", default="", required=False),
    MasterParquetColumn(name="archive_member_count", dtype="Int64", default=None, required=False),
    MasterParquetColumn(name="archive_size_bytes", dtype="Int64", default=None, required=False),
    # Transfer detail
    MasterParquetColumn(name="swiftstack_archive", dtype="string", default="", required=True),
    MasterParquetColumn(name="swiftstack_backup_status", dtype="string", default="", required=False),
    MasterParquetColumn(name="swiftstack_uri", dtype="string", default="", required=False),
    MasterParquetColumn(name="swiftstack_uploaded_at", dtype="string", default="", required=False),
    MasterParquetColumn(name="swiftstack_upload_error", dtype="string", default="", required=False),
    MasterParquetColumn(name="uploaded_to_gcp", dtype="string", default="no", required=True),
    MasterParquetColumn(name="gcp_backup_status", dtype="string", default="", required=False),
    MasterParquetColumn(name="gcp_uri", dtype="string", default="", required=False),
    MasterParquetColumn(name="gcp_uploaded_at", dtype="string", default="", required=False),
    MasterParquetColumn(name="gcp_upload_error", dtype="string", default="", required=False),
)

_KNOWN_COLUMNS = frozenset(column.name for column in MASTER_PARQUET_PROJECTION)
_REQUIRED_COLUMNS = frozenset(column.name for column in MASTER_PARQUET_PROJECTION if column.required)


def validate_projection_columns(names: Iterable[str]) -> None:
    """Validate a requested master-parquet projection column list.

    A valid projection must contain every required compatibility column and
    must not contain unknown or duplicate columns.  The iterable is materialized
    once so generators are supported.  This helper is pure: it performs no I/O
    and mutates nothing.
    """
    materialized = tuple(names)
    for name in materialized:
        if not isinstance(name, str) or not name:
            raise ValueError("master parquet projection columns must be non-empty strings")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in materialized:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"duplicate master parquet projection column(s): {', '.join(sorted(duplicates))}")
    unknown = sorted(set(materialized) - _KNOWN_COLUMNS)
    if unknown:
        raise ValueError(f"unknown master parquet projection column(s): {', '.join(unknown)}")
    missing = sorted(_REQUIRED_COLUMNS - set(materialized))
    if missing:
        raise ValueError(f"missing required master parquet projection column(s): {', '.join(missing)}")


__all__ = [
    "MASTER_PARQUET_PROJECTION",
    "MasterParquetColumn",
    "MasterParquetDtype",
    "master_parquet_column_from_mapping",
    "validate_projection_columns",
]
