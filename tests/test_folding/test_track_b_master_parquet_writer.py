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

"""Master-parquet writer tests for the folding companion-engine track (e05s05)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bspp.orchestration.contract.master_parquet_projection import (
    MASTER_PARQUET_PROJECTION,
    MasterParquetColumn,
    validate_projection_columns,
)
from bspp.orchestration.runtime.folding.execution.master_parquet_writer import (
    build_master_parquet_row,
    write_master_parquet,
)

_REQUIRED_NAMES = tuple(column.name for column in MASTER_PARQUET_PROJECTION if column.required)
_ENGINE_OPTIONAL_NULLABLE_SCORES = frozenset({"ptm", "iptm"})
_PENDING_REQUIRED_NAMES = tuple(name for name in _REQUIRED_NAMES if name not in _ENGINE_OPTIONAL_NULLABLE_SCORES)


def _complete_kwargs() -> dict[str, object]:
    return {
        "msa_path": "AFDB_AF-0000000000000001",
        "source_run": "r1",
        "pdb_path": "p.pdb",
        "json_path": "p.json",
        "pdb_residue_count": 100,
        "mean_plddt": 90.0,
        "plddt_above_70": 80.0,
        "ptm": 0.9,
        "iptm": 0.8,
        "max_pae": 5.0,
        "output_has_nan": "no",
        "pdb_json_match": "yes",
        "swiftstack_archive": "bspp_260909_1430_a00001.tar.lz4",
        "uploaded_to_gcp": "no",
    }


def test_complete_row_happy_path() -> None:
    kwargs = _complete_kwargs()
    row = build_master_parquet_row(**kwargs)
    assert row is not None
    for name in _REQUIRED_NAMES:
        assert row[name] == kwargs[name]


@pytest.mark.parametrize("pending_field", _PENDING_REQUIRED_NAMES)
def test_required_field_gate_withholds_row(pending_field: str) -> None:
    kwargs = _complete_kwargs()
    kwargs[pending_field] = None
    assert build_master_parquet_row(**kwargs) is None


def test_resolved_vs_pending_swiftstack_archive() -> None:
    kwargs = _complete_kwargs()

    resolved = build_master_parquet_row(**{**kwargs, "swiftstack_archive": ""})
    assert resolved is not None
    assert resolved["swiftstack_archive"] == ""

    pending = build_master_parquet_row(**{**kwargs, "swiftstack_archive": None})
    assert pending is None


def test_resolved_vs_pending_uploaded_to_gcp() -> None:
    kwargs = _complete_kwargs()

    resolved = build_master_parquet_row(**{**kwargs, "uploaded_to_gcp": "no"})
    assert resolved is not None
    assert resolved["uploaded_to_gcp"] == "no"

    pending = build_master_parquet_row(**{**kwargs, "uploaded_to_gcp": None})
    assert pending is None


def test_full_union_vocabulary_in_registry_order() -> None:
    row = build_master_parquet_row(**_complete_kwargs())
    assert row is not None
    assert list(row.keys()) == [column.name for column in MASTER_PARQUET_PROJECTION]
    assert len(row) == len(MASTER_PARQUET_PROJECTION) == 35


def test_dtype_correctness() -> None:
    row = build_master_parquet_row(**_complete_kwargs())
    assert row is not None
    for column in MASTER_PARQUET_PROJECTION:
        value = row[column.name]
        if value is None:
            assert column.required is False
            continue
        if column.dtype == "string":
            assert isinstance(value, str)
        elif column.dtype == "Int64":
            assert isinstance(value, int)
        elif column.dtype == "Float64":
            assert isinstance(value, float)


def test_optional_defaults() -> None:
    row = build_master_parquet_row(**_complete_kwargs())
    assert row is not None
    for column in MASTER_PARQUET_PROJECTION:
        if not column.required:
            assert row[column.name] == column.default


def test_optional_override() -> None:
    row = build_master_parquet_row(**_complete_kwargs(), archive_status="done")
    assert row is not None
    assert row["archive_status"] == "done"
    for column in MASTER_PARQUET_PROJECTION:
        if not column.required and column.name != "archive_status":
            assert row[column.name] == column.default


def test_engine_optional_null_scores_are_resolved_nulls() -> None:
    kwargs = _complete_kwargs()
    kwargs["ptm"] = None
    kwargs["iptm"] = None

    row = build_master_parquet_row(**kwargs)
    assert row is not None
    assert row["ptm"] is None
    assert row["iptm"] is None
    for name in _PENDING_REQUIRED_NAMES:
        assert row[name] == kwargs[name]


def test_engine_optional_null_scores_round_trip(tmp_path: Path) -> None:
    kwargs = _complete_kwargs()
    kwargs["ptm"] = None
    kwargs["iptm"] = None
    row = build_master_parquet_row(**kwargs)
    assert row is not None

    path = tmp_path / "master.parquet"
    write_master_parquet([row], path)

    table = pq.read_table(path)
    first = table.to_pylist()[0]
    assert first["ptm"] is None
    assert first["iptm"] is None
    assert first["mean_plddt"] == 90.0


def test_unknown_optional_key_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown master parquet projection column"):
        build_master_parquet_row(**_complete_kwargs(), bogus="x")


def test_validate_projection_columns_passes_on_produced_set() -> None:
    row = build_master_parquet_row(**_complete_kwargs())
    assert row is not None
    validate_projection_columns(row.keys())


def _arrow_dtype(column: MasterParquetColumn) -> pa.DataType:
    if column.dtype == "string":
        return pa.string()
    if column.dtype == "Int64":
        return pa.int64()
    return pa.float64()


def test_write_master_parquet_round_trips(tmp_path: Path) -> None:
    rows = [
        build_master_parquet_row(**_complete_kwargs()),
        build_master_parquet_row(**{**_complete_kwargs(), "source_run": "r2", "msa_path": "AFDB_AF-0000000000000002"}),
    ]
    assert all(row is not None for row in rows)

    path = tmp_path / "nested" / "master.parquet"
    write_master_parquet([row for row in rows if row is not None], path)
    assert path.exists()

    table = pq.read_table(path)
    assert table.column_names == [column.name for column in MASTER_PARQUET_PROJECTION]
    assert table.schema.types == [_arrow_dtype(column) for column in MASTER_PARQUET_PROJECTION]

    first = table.to_pylist()[0]
    assert first["msa_path"] == "AFDB_AF-0000000000000001"
    assert first["source_run"] == "r1"
    assert first["mean_plddt"] == 90.0
    assert first["swiftstack_archive"] == "bspp_260909_1430_a00001.tar.lz4"
    assert first["uploaded_to_gcp"] == "no"

    second = table.to_pylist()[1]
    assert second["source_run"] == "r2"


def test_write_master_parquet_empty_preserves_schema(tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    write_master_parquet([], path)
    assert path.exists()

    table = pq.read_table(path)
    assert table.num_rows == 0
    assert table.column_names == [column.name for column in MASTER_PARQUET_PROJECTION]
    assert table.schema.types == [_arrow_dtype(column) for column in MASTER_PARQUET_PROJECTION]


@pytest.mark.parametrize("pending_field", _PENDING_REQUIRED_NAMES)
def test_write_master_parquet_rejects_pending_required_field(pending_field: str, tmp_path: Path) -> None:
    row = build_master_parquet_row(**_complete_kwargs())
    assert row is not None
    bad_row = dict(row)
    bad_row[pending_field] = None

    path = tmp_path / "nested" / "master.parquet"
    with pytest.raises(ValueError, match="pending required field"):
        write_master_parquet([bad_row], path)

    assert not path.exists()
    assert not path.parent.exists()
