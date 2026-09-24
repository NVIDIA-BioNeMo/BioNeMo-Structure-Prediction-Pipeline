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

"""Master-parquet projection registry contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from bspp.orchestration.contract.master_parquet_projection import (
    MASTER_PARQUET_PROJECTION,
    MasterParquetColumn,
    master_parquet_column_from_mapping,
    validate_projection_columns,
)

# Harvested full-union column list, embedded as an independent literal so the
# implementation and test cannot drift together.
EXPECTED_COLUMNS: list[tuple[str, str, object]] = [
    # Key
    ("msa_path", "string", None),
    # Core
    ("seq_length", "Int64", None),
    ("seq_count", "Int64", None),
    ("sequence", "string", None),
    # Provenance
    ("source_run", "string", ""),
    ("source_parquet", "string", ""),
    ("predictions_path", "string", ""),
    # Payload
    ("pdb_path", "string", ""),
    ("json_path", "string", ""),
    # Quality
    ("pdb_residue_count", "Int64", None),
    ("mean_plddt", "Float64", None),
    ("plddt_above_70", "Float64", None),
    ("ptm", "Float64", None),
    ("iptm", "Float64", None),
    ("max_pae", "Float64", None),
    ("output_has_nan", "string", "not_checked"),
    ("pdb_json_match", "string", "not_checked"),
    # Archive detail
    ("archive_status", "string", ""),
    ("archive_batch_id", "string", ""),
    ("archive_file", "string", ""),
    ("archive_created_at", "string", ""),
    ("archive_sha256", "string", ""),
    ("local_archive_path", "string", ""),
    ("archive_member_count", "Int64", None),
    ("archive_size_bytes", "Int64", None),
    # Transfer detail
    ("swiftstack_archive", "string", ""),
    ("swiftstack_backup_status", "string", ""),
    ("swiftstack_uri", "string", ""),
    ("swiftstack_uploaded_at", "string", ""),
    ("swiftstack_upload_error", "string", ""),
    ("uploaded_to_gcp", "string", "no"),
    ("gcp_backup_status", "string", ""),
    ("gcp_uri", "string", ""),
    ("gcp_uploaded_at", "string", ""),
    ("gcp_upload_error", "string", ""),
]

# The frozen postprocessing compatibility read surface.
REQUIRED_COLUMN_NAMES = {
    "msa_path",
    "source_run",
    "pdb_path",
    "json_path",
    "pdb_residue_count",
    "output_has_nan",
    "pdb_json_match",
    "mean_plddt",
    "plddt_above_70",
    "ptm",
    "iptm",
    "max_pae",
    "swiftstack_archive",
    "uploaded_to_gcp",
}


def test_projection_is_a_tuple() -> None:
    assert isinstance(MASTER_PARQUET_PROJECTION, tuple)


def test_names_match_harvested_list_in_order() -> None:
    assert [column.name for column in MASTER_PARQUET_PROJECTION] == [name for name, _, _ in EXPECTED_COLUMNS]


def test_names_are_unique() -> None:
    names = [column.name for column in MASTER_PARQUET_PROJECTION]

    assert len(names) == len(set(names))


def test_every_entry_is_a_frozen_master_parquet_column() -> None:
    for column in MASTER_PARQUET_PROJECTION:
        assert isinstance(column, MasterParquetColumn)


def test_every_dtype_is_within_the_harvested_vocabulary() -> None:
    for column in MASTER_PARQUET_PROJECTION:
        assert column.dtype in {"string", "Int64", "Float64"}


def test_dtype_and_default_metadata_match_harvested_literal() -> None:
    assert [(column.name, column.dtype, column.default) for column in MASTER_PARQUET_PROJECTION] == EXPECTED_COLUMNS


def test_defaults_are_compatible_with_declared_dtype() -> None:
    for column in MASTER_PARQUET_PROJECTION:
        if column.dtype == "string":
            assert column.default is None or isinstance(column.default, str)
        elif column.dtype == "Int64":
            assert column.default is None or (isinstance(column.default, int) and not isinstance(column.default, bool))
        elif column.dtype == "Float64":
            assert column.default is None or (
                isinstance(column.default, (int, float)) and not isinstance(column.default, bool)
            )


def test_records_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        MASTER_PARQUET_PROJECTION[0].name = "changed"  # type: ignore[misc]


def test_required_subset_matches_literal_exactly() -> None:
    required = {column.name for column in MASTER_PARQUET_PROJECTION if column.required}

    assert required == REQUIRED_COLUMN_NAMES


def test_validate_accepts_exact_required_subset() -> None:
    validate_projection_columns(sorted(REQUIRED_COLUMN_NAMES))


def test_validate_accepts_full_canonical_union() -> None:
    validate_projection_columns(column.name for column in MASTER_PARQUET_PROJECTION)


def test_validate_accepts_required_subset_plus_optional_columns() -> None:
    validate_projection_columns(sorted(REQUIRED_COLUMN_NAMES | {"archive_status", "gcp_uri"}))


def test_validate_accepts_alternate_ordering() -> None:
    validate_projection_columns(reversed([column.name for column in MASTER_PARQUET_PROJECTION]))


def test_validate_accepts_generator_input() -> None:
    validate_projection_columns(name for name in sorted(REQUIRED_COLUMN_NAMES))


def test_validate_rejects_one_required_column_omitted() -> None:
    with pytest.raises(ValueError, match="missing required"):
        validate_projection_columns(sorted(REQUIRED_COLUMN_NAMES - {"msa_path"}))


def test_validate_rejects_multiple_required_columns_omitted() -> None:
    with pytest.raises(ValueError, match="missing required"):
        validate_projection_columns(sorted(REQUIRED_COLUMN_NAMES - {"msa_path", "pdb_path"}))


def test_validate_rejects_unknown_column() -> None:
    with pytest.raises(ValueError, match="unknown master parquet projection column"):
        validate_projection_columns(sorted(REQUIRED_COLUMN_NAMES | {"not_a_column"}))


def test_validate_rejects_duplicate_column() -> None:
    with pytest.raises(ValueError, match="duplicate master parquet projection column"):
        validate_projection_columns([*sorted(REQUIRED_COLUMN_NAMES), "msa_path"])


def test_validate_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        validate_projection_columns([*sorted(REQUIRED_COLUMN_NAMES - {"msa_path"}), ""])


def test_validate_rejects_non_string_entry() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        validate_projection_columns([*sorted(REQUIRED_COLUMN_NAMES - {"msa_path"}), 42])  # type: ignore[list-item]


def test_column_mapping_round_trip() -> None:
    column = MASTER_PARQUET_PROJECTION[0]

    assert master_parquet_column_from_mapping(column.to_mapping()) == column


def test_column_mapping_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="Unknown MasterParquetColumn field"):
        master_parquet_column_from_mapping(
            {"name": "x", "dtype": "string", "default": None, "required": False, "extra": 1}
        )


def test_column_mapping_rejects_missing_field() -> None:
    with pytest.raises(ValueError, match="Missing MasterParquetColumn field"):
        master_parquet_column_from_mapping({"name": "x", "dtype": "string", "default": None})


def test_column_mapping_rejects_invalid_dtype() -> None:
    with pytest.raises(ValueError, match="unsupported master parquet dtype"):
        master_parquet_column_from_mapping({"name": "x", "dtype": "object", "default": None, "required": False})


def test_column_mapping_rejects_invalid_default_dtype_combination() -> None:
    with pytest.raises(ValueError, match="default"):
        master_parquet_column_from_mapping({"name": "x", "dtype": "Int64", "default": "oops", "required": False})


def test_column_mapping_rejects_boolean_default() -> None:
    with pytest.raises(ValueError, match="default"):
        master_parquet_column_from_mapping({"name": "x", "dtype": "Float64", "default": True, "required": False})
