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

"""Immutable folding-index records.

The records support behavior derived from Port Baseline sources
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/scripts/preprocessing/generate_batch_info.py:27-187``
and
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/scripts/preprocess_trt_bionemo.py:186-249``.
Filesystem and A3M parsing deliberately remain in the runtime distribution.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "source_ordinal",
        "protein_id",
        "msa_path",
        "query_sequence",
        "sequence_length",
        "chain_lengths",
        "chain_count",
        "chain_cardinalities",
        "msa_depth",
        "total_length",
    }
)
_INDEX_FIELDS = frozenset({"schema_version", "records", "length_sorted_records"})


@dataclass(frozen=True)
class FoldingIndexRecord:
    """Canonical metadata for one explicitly identified folding model.

    ``sequence_length`` is the baseline scheduling column after the exact
    ``seq_length``/``total_length`` alias normalization. ``chain_lengths`` and
    ``total_length`` retain the independently parsed A3M-header structure, so
    historical rows remain interpretable even when their scheduling column was
    named ``total_length``. For the baseline's two-unique-chain header,
    ``chain_count`` is the number of unique chains and ``total_length`` is the
    sum of their lengths; neither expands retained chain cardinalities. A
    retained-list row's shared ``sequence_length`` is scheduling metadata and
    need not equal the parsed query length.
    """

    source_ordinal: int
    protein_id: str
    msa_path: str
    query_sequence: str
    sequence_length: int
    chain_lengths: tuple[int, ...]
    chain_count: int
    chain_cardinalities: tuple[int, ...]
    msa_depth: int
    total_length: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject records whose immutable fields cannot form a coherent index."""
        validate_schema_version(self.schema_version, record_name="FoldingIndexRecord")
        if not isinstance(self.source_ordinal, int) or isinstance(self.source_ordinal, bool) or self.source_ordinal < 0:
            msg = "source_ordinal must be a non-negative contiguous index"
            raise ValueError(msg)
        _validate_nonempty_str(self.protein_id, "protein_id")
        _validate_nonempty_str(self.msa_path, "msa_path")
        _validate_nonempty_str(self.query_sequence, "query_sequence")
        _validate_positive_int(self.sequence_length, "sequence_length")
        _validate_positive_int_tuple(self.chain_lengths, "chain_lengths")
        _validate_positive_int(self.chain_count, "chain_count")
        _validate_positive_int_tuple(self.chain_cardinalities, "chain_cardinalities")
        _validate_positive_int(self.msa_depth, "msa_depth")
        _validate_positive_int(self.total_length, "total_length")

        if len(self.chain_lengths) != len(self.chain_cardinalities):
            msg = "chain_lengths and chain_cardinalities must have the same number of entries"
            raise ValueError(msg)
        if len(self.query_sequence) != sum(self.chain_lengths):
            msg = "query_sequence length must equal the sum of chain_lengths"
            raise ValueError(msg)

        expected_chain_count = self.chain_cardinalities[0] if len(self.chain_lengths) == 1 else len(self.chain_lengths)
        if self.chain_count != expected_chain_count:
            msg = f"chain_count must be {expected_chain_count} for the declared chain structure"
            raise ValueError(msg)

        # This deliberately follows generate_batch_info.py: homomers multiply
        # by the single cardinality, whereas heterodimers use both chain
        # lengths and retain the header cardinalities as separate evidence.
        expected_total_length = (
            self.chain_lengths[0] * self.chain_count if len(self.chain_lengths) == 1 else sum(self.chain_lengths)
        )
        if self.total_length != expected_total_length:
            msg = f"total_length must be {expected_total_length} for the declared chain structure"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/table-ready data."""
        return {
            "schema_version": self.schema_version,
            "source_ordinal": self.source_ordinal,
            "protein_id": self.protein_id,
            "msa_path": self.msa_path,
            "query_sequence": self.query_sequence,
            "sequence_length": self.sequence_length,
            "chain_lengths": list(self.chain_lengths),
            "chain_count": self.chain_count,
            "chain_cardinalities": list(self.chain_cardinalities),
            "msa_depth": self.msa_depth,
            "total_length": self.total_length,
        }


@dataclass(frozen=True)
class FoldingIndex:
    """Primary index order and its deterministic length-sorted companion."""

    records: tuple[FoldingIndexRecord, ...]
    length_sorted_records: tuple[FoldingIndexRecord, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject mutable, duplicate, or inconsistently sorted index views."""
        validate_schema_version(self.schema_version, record_name="FoldingIndex")
        if not isinstance(self.records, tuple) or not isinstance(self.length_sorted_records, tuple):
            msg = "FoldingIndex record collections must be tuples"
            raise ValueError(msg)
        if not self.records:
            msg = "FoldingIndex requires at least one record"
            raise ValueError(msg)
        if not all(isinstance(record, FoldingIndexRecord) for record in self.records):
            msg = "records must contain FoldingIndexRecord values"
            raise ValueError(msg)
        if not all(isinstance(record, FoldingIndexRecord) for record in self.length_sorted_records):
            msg = "length_sorted_records must contain FoldingIndexRecord values"
            raise ValueError(msg)

        ordinals = tuple(sorted(record.source_ordinal for record in self.records))
        if ordinals != tuple(range(len(self.records))):
            msg = "source_ordinal values must form a contiguous zero-based sequence"
            raise ValueError(msg)
        _reject_duplicates((record.protein_id for record in self.records), field_name="protein_id")
        _reject_duplicates((record.msa_path for record in self.records), field_name="msa_path")

        expected_sorted = tuple(sorted(self.records, key=_length_sort_key))
        if self.length_sorted_records != expected_sorted:
            msg = "length_sorted_records must be the deterministic length-sorted view of records"
            raise ValueError(msg)
        if any(record.schema_version != self.schema_version for record in self.records):
            msg = "record schema versions must match the FoldingIndex schema version"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-ready data, including the companion view."""
        return {
            "schema_version": self.schema_version,
            "records": [record.to_mapping() for record in self.records],
            "length_sorted_records": [record.to_mapping() for record in self.length_sorted_records],
        }


def folding_index_record_from_mapping(payload: Mapping[str, object]) -> FoldingIndexRecord:
    """Load one folding record, rejecting unsupported versions and fields."""
    _reject_unknown_fields(payload, _RECORD_FIELDS, record_name="FoldingIndexRecord")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingIndexRecord")
    return FoldingIndexRecord(
        schema_version=schema_version,
        source_ordinal=_required_int(payload, "source_ordinal"),
        protein_id=_required_str(payload, "protein_id"),
        msa_path=_required_str(payload, "msa_path"),
        query_sequence=_required_str(payload, "query_sequence"),
        sequence_length=_required_int(payload, "sequence_length"),
        chain_lengths=_required_int_tuple(payload, "chain_lengths"),
        chain_count=_required_int(payload, "chain_count"),
        chain_cardinalities=_required_int_tuple(payload, "chain_cardinalities"),
        msa_depth=_required_int(payload, "msa_depth"),
        total_length=_required_int(payload, "total_length"),
    )


def folding_index_from_mapping(payload: Mapping[str, object]) -> FoldingIndex:
    """Load a complete folding index and fail closed on an altered companion."""
    _reject_unknown_fields(payload, _INDEX_FIELDS, record_name="FoldingIndex")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingIndex")
    records = _required_record_tuple(payload, "records")
    length_sorted_records = _required_record_tuple(payload, "length_sorted_records")
    return FoldingIndex(
        schema_version=schema_version,
        records=records,
        length_sorted_records=length_sorted_records,
    )


def make_folding_index(records: tuple[FoldingIndexRecord, ...], *, sort_by_length: bool = False) -> FoldingIndex:
    """Build both deterministic index views from source-ordered records."""
    length_sorted_records = tuple(sorted(records, key=_length_sort_key))
    primary_records = length_sorted_records if sort_by_length else records
    return FoldingIndex(records=primary_records, length_sorted_records=length_sorted_records)


def _length_sort_key(record: FoldingIndexRecord) -> tuple[int, int, str, str]:
    return (record.sequence_length, record.source_ordinal, record.protein_id, record.msa_path)


def _reject_duplicates(values: Iterable[str], *, field_name: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            msg = f"Duplicate {field_name} {value!r} in FoldingIndex"
            raise ValueError(msg)
        seen.add(value)


def _reject_unknown_fields(payload: Mapping[str, object], allowed: frozenset[str], *, record_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        msg = f"{record_name} has unknown fields: {', '.join(unknown)}"
        raise ValueError(msg)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


def _required_int_tuple(payload: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list of integers"
        raise ValueError(msg)
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        msg = f"{key} must be a list of integers"
        raise ValueError(msg)
    return tuple(value)


def _required_record_tuple(payload: Mapping[str, object], key: str) -> tuple[FoldingIndexRecord, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list of folding-index records"
        raise ValueError(msg)
    records: list[FoldingIndexRecord] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"{key}[{index}] must be an object"
            raise ValueError(msg)
        records.append(folding_index_record_from_mapping(item))
    return tuple(records)


def _validate_nonempty_str(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)


def _validate_positive_int(value: object, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{field_name} must be a positive integer"
        raise ValueError(msg)


def _validate_positive_int_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple) or not value:
        msg = f"{field_name} must be a non-empty tuple of positive integers"
        raise ValueError(msg)
    if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value):
        msg = f"{field_name} must contain only positive integers"
        raise ValueError(msg)


__all__ = [
    "FoldingIndex",
    "FoldingIndexRecord",
    "folding_index_from_mapping",
    "folding_index_record_from_mapping",
    "make_folding_index",
]
