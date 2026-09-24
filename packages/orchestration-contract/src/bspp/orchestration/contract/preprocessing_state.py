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

"""Immutable preprocessing lifecycle evidence and state contracts.

Port Baseline:
419813dbb5a3949e5e16f289f974d9f95e94bf01:utils/examine_results.sh:6-13,
419813dbb5a3949e5e16f289f974d9f95e94bf01:utils/get_remaining.sh:4-10, and
419813dbb5a3949e5e16f289f974d9f95e94bf01:scripts/examine_results_and_replace_input.py:29-184.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.preprocessing import (
    PreprocessingChunk,
    PreprocessingChunkAssignment,
    PreprocessingWorkPlan,
    preprocessing_chunk_assignment_from_mapping,
    preprocessing_chunk_from_mapping,
    preprocessing_work_plan_from_mapping,
)
from bspp.orchestration.contract.preprocessing_execution import ExpectedA3M, expected_a3m_from_mapping
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

PreprocessingChunkStateKind = Literal["planned", "completed", "retryable", "missing", "invalid"]
PreprocessingStateReason = Literal[
    "awaiting_execution",
    "complete",
    "ghosted_source",
    "pristine_source_missing",
    "contradictory_split_and_finished",
    "missing_record_evidence",
    "missing_log_evidence",
    "empty_record_evidence",
    "empty_log_evidence",
    "record_reports_no_such_file",
    "search_skipped",
    "record_count_mismatch",
    "record_first_line_missing_afdb",
    "malformed_record_entry",
    "record_members_mismatch",
    "a3m_count_not_above_two",
    "tar_missing",
    "tar_empty",
    "tar_lz4_missing",
    "tar_lz4_empty",
    "tar_member_missing",
    "finished_source_missing",
]
PreprocessingRetryOperation = Literal["copy", "remove"]

_CHUNK_STATE_KINDS = frozenset({"planned", "completed", "retryable", "missing", "invalid"})
_STATE_REASONS_BY_KIND: dict[PreprocessingChunkStateKind, frozenset[str]] = {
    "planned": frozenset({"awaiting_execution"}),
    "completed": frozenset({"complete"}),
    "retryable": frozenset(
        {
            "missing_record_evidence",
            "missing_log_evidence",
            "empty_record_evidence",
            "empty_log_evidence",
            "record_reports_no_such_file",
            "search_skipped",
        }
    ),
    "missing": frozenset({"ghosted_source"}),
    "invalid": frozenset(
        {
            "pristine_source_missing",
            "contradictory_split_and_finished",
            "record_count_mismatch",
            "record_first_line_missing_afdb",
            "malformed_record_entry",
            "record_members_mismatch",
            "a3m_count_not_above_two",
            "tar_missing",
            "tar_empty",
            "tar_lz4_missing",
            "tar_lz4_empty",
            "tar_member_missing",
            "finished_source_missing",
        }
    ),
}
_STATE_REASONS = frozenset(reason for reasons in _STATE_REASONS_BY_KIND.values() for reason in reasons)


@dataclass(frozen=True)
class PreprocessingSourceMembership:
    """Explicit source and local/shared completion membership for one chunk."""

    chunk_name: str
    pristine_path: str
    pristine_present: bool
    pristine_record_count: int | None
    split_path: str
    split_present: bool
    finished_input_path: str
    finished_input_present: bool
    shared_finished_read_tar_path: str
    shared_finished_read_present: bool
    shared_finished_write_tar_path: str
    shared_finished_write_present: bool
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="PreprocessingSourceMembership")
        _validate_chunk_paths(
            self.chunk_name,
            self.pristine_path,
            self.split_path,
            self.finished_input_path,
        )
        chunk_stem = self.chunk_name.removesuffix(".fa")
        for field_name, path in (
            ("shared_finished_read_tar_path", self.shared_finished_read_tar_path),
            ("shared_finished_write_tar_path", self.shared_finished_write_tar_path),
        ):
            if not isinstance(path, str) or not path.endswith(f"/{chunk_stem}.tar"):
                msg = f"{field_name} must identify the chunk's flat tar path"
                raise ValueError(msg)
        for field_name, value in (
            ("pristine_present", self.pristine_present),
            ("split_present", self.split_present),
            ("finished_input_present", self.finished_input_present),
            ("shared_finished_read_present", self.shared_finished_read_present),
            ("shared_finished_write_present", self.shared_finished_write_present),
        ):
            _validate_bool(value, field_name)
        if self.pristine_present:
            if (
                not isinstance(self.pristine_record_count, int)
                or isinstance(self.pristine_record_count, bool)
                or self.pristine_record_count <= 0
            ):
                msg = "present pristine input requires a positive pristine_record_count"
                raise ValueError(msg)
        elif self.pristine_record_count is not None:
            msg = "absent pristine input cannot declare pristine_record_count"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready source membership."""
        return {
            "schema_version": self.schema_version,
            "chunk_name": self.chunk_name,
            "pristine_path": self.pristine_path,
            "pristine_present": self.pristine_present,
            "pristine_record_count": self.pristine_record_count,
            "split_path": self.split_path,
            "split_present": self.split_present,
            "finished_input_path": self.finished_input_path,
            "finished_input_present": self.finished_input_present,
            "shared_finished_read_tar_path": self.shared_finished_read_tar_path,
            "shared_finished_read_present": self.shared_finished_read_present,
            "shared_finished_write_tar_path": self.shared_finished_write_tar_path,
            "shared_finished_write_present": self.shared_finished_write_present,
        }


@dataclass(frozen=True)
class PreprocessingPairedEvidence:
    """Explicitly associated durable `.record` and merged-output `.log` evidence."""

    chunk_name: str
    durable_record_path: str
    durable_log_path: str
    record_lines: tuple[str, ...] | None
    log_lines: tuple[str, ...] | None
    record_streams_merged: bool = True
    log_streams_merged: bool = True
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="PreprocessingPairedEvidence")
        chunk_stem = _validate_chunk_name(self.chunk_name)
        if not self.durable_record_path.endswith(f"/{chunk_stem}.record"):
            msg = "durable_record_path must be flat and keyed by chunk stem"
            raise ValueError(msg)
        if not self.durable_log_path.endswith(f"/{chunk_stem}.log"):
            msg = "durable_log_path must be flat and keyed by chunk stem"
            raise ValueError(msg)
        _validate_optional_lines(self.record_lines, "record_lines")
        _validate_optional_lines(self.log_lines, "log_lines")
        if not self.record_streams_merged or not self.log_streams_merged:
            msg = "durable record and log observations must preserve merged stdout/stderr"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready paired durable evidence."""
        return {
            "schema_version": self.schema_version,
            "chunk_name": self.chunk_name,
            "durable_record_path": self.durable_record_path,
            "durable_log_path": self.durable_log_path,
            "record_lines": list(self.record_lines) if self.record_lines is not None else None,
            "log_lines": list(self.log_lines) if self.log_lines is not None else None,
            "record_streams_merged": self.record_streams_merged,
            "log_streams_merged": self.log_streams_merged,
        }


@dataclass(frozen=True)
class PreprocessingArchiveEvidence:
    """Supplied durable tar, compression, and tar-member observations."""

    chunk_name: str
    durable_tar_path: str
    durable_lz4_path: str
    tar_size_bytes: int | None
    lz4_size_bytes: int | None
    tar_members: tuple[str, ...] | None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="PreprocessingArchiveEvidence")
        chunk_stem = _validate_chunk_name(self.chunk_name)
        if not self.durable_tar_path.endswith(f"/{chunk_stem}.tar"):
            msg = "durable_tar_path must be flat and keyed by chunk stem"
            raise ValueError(msg)
        if self.durable_lz4_path != f"{self.durable_tar_path}.lz4":
            msg = "durable_lz4_path must append .lz4 to durable_tar_path"
            raise ValueError(msg)
        _validate_optional_size(self.tar_size_bytes, "tar_size_bytes")
        _validate_optional_size(self.lz4_size_bytes, "lz4_size_bytes")
        if (self.tar_size_bytes is None) != (self.tar_members is None):
            msg = "tar_members must be supplied exactly when the tar is present"
            raise ValueError(msg)
        if self.tar_members is not None and (
            not isinstance(self.tar_members, tuple)
            or any(not isinstance(member, str) or not member for member in self.tar_members)
        ):
            msg = "tar_members must be an immutable tuple of non-empty names"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready archive observations."""
        return {
            "schema_version": self.schema_version,
            "chunk_name": self.chunk_name,
            "durable_tar_path": self.durable_tar_path,
            "durable_lz4_path": self.durable_lz4_path,
            "tar_size_bytes": self.tar_size_bytes,
            "lz4_size_bytes": self.lz4_size_bytes,
            "tar_members": list(self.tar_members) if self.tar_members is not None else None,
        }


@dataclass(frozen=True)
class PreprocessingRetryAction:
    """One declarative source restore or bad-archive removal."""

    chunk_name: str
    operation: PreprocessingRetryOperation
    source_path: str | None
    target_path: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="PreprocessingRetryAction")
        chunk_stem = _validate_chunk_name(self.chunk_name)
        if self.operation == "copy":
            if not self.source_path:
                msg = "copy retry actions require a source_path"
                raise ValueError(msg)
            if not self.source_path.endswith(f"/{self.chunk_name}") or not self.target_path.endswith(
                f"/{self.chunk_name}"
            ):
                msg = "copy retry action paths must be keyed by chunk name"
                raise ValueError(msg)
        elif self.operation == "remove":
            if self.source_path is not None:
                msg = "remove retry actions cannot declare a source_path"
                raise ValueError(msg)
            if not self.target_path.endswith((f"/{chunk_stem}.tar", f"/{chunk_stem}.tar.lz4")):
                msg = "remove retry action target_path must be keyed by chunk archive"
                raise ValueError(msg)
        else:
            msg = f"unsupported retry operation: {self.operation!r}"
            raise ValueError(msg)
        if not self.target_path:
            msg = "retry action target_path must be non-empty"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready retry action data."""
        return {
            "schema_version": self.schema_version,
            "chunk_name": self.chunk_name,
            "operation": self.operation,
            "source_path": self.source_path,
            "target_path": self.target_path,
        }


def derive_preprocessing_retry_actions(
    *,
    chunk_name: str,
    eligible_for_retry: bool,
    pristine_present: bool,
    split_present: bool,
    pristine_path: str,
    split_path: str,
    durable_tar_path: str,
    durable_lz4_path: str,
    tar_present: bool,
    lz4_present: bool,
    record_line_count: int | None,
    record_first_line_contains_afdb: bool | None,
    expected_num_records: int,
) -> tuple[PreprocessingRetryAction, ...]:
    """Derive ordered, non-executing retry actions using baseline split guards."""
    for field_name, value in (
        ("eligible_for_retry", eligible_for_retry),
        ("pristine_present", pristine_present),
        ("split_present", split_present),
        ("tar_present", tar_present),
        ("lz4_present", lz4_present),
    ):
        _validate_bool(value, field_name)
    _validate_optional_size(record_line_count, "record_line_count")
    _validate_optional_bool(record_first_line_contains_afdb, "record_first_line_contains_afdb")
    if not isinstance(expected_num_records, int) or isinstance(expected_num_records, bool) or expected_num_records <= 0:
        msg = "expected_num_records must be a positive integer"
        raise ValueError(msg)
    unconditional_bad_record = (
        record_line_count == expected_num_records and record_first_line_contains_afdb is False
    ) or (record_line_count is not None and record_line_count > expected_num_records)
    may_restore_and_clean = eligible_for_retry and pristine_present and (not split_present or unconditional_bad_record)
    actions: list[PreprocessingRetryAction] = []
    if may_restore_and_clean:
        actions.append(
            PreprocessingRetryAction(
                chunk_name=chunk_name,
                operation="copy",
                source_path=pristine_path,
                target_path=split_path,
            )
        )
    if may_restore_and_clean and tar_present:
        actions.append(
            PreprocessingRetryAction(
                chunk_name=chunk_name,
                operation="remove",
                source_path=None,
                target_path=durable_tar_path,
            )
        )
    if may_restore_and_clean and lz4_present:
        actions.append(
            PreprocessingRetryAction(
                chunk_name=chunk_name,
                operation="remove",
                source_path=None,
                target_path=durable_lz4_path,
            )
        )
    return tuple(actions)


@dataclass(frozen=True)
class PreprocessingChunkState:
    """Deterministic lifecycle interpretation for one planned chunk."""

    chunk_name: str
    tranche_name: str
    chunk_ordinal: int
    state: PreprocessingChunkStateKind
    reason_codes: tuple[PreprocessingStateReason, ...]
    eligible_for_retry: bool
    pristine_present: bool
    split_present: bool
    finished_input_present: bool
    pristine_path: str
    split_path: str
    finished_input_path: str
    shared_finished_read_present: bool
    shared_finished_write_present: bool
    shared_finished_read_tar_path: str
    shared_finished_write_tar_path: str
    durable_record_path: str
    durable_log_path: str
    durable_tar_path: str
    durable_lz4_path: str
    tar_size_bytes: int | None
    lz4_size_bytes: int | None
    expected_a3ms: tuple[ExpectedA3M, ...]
    expected_num_records: int
    pristine_record_count: int | None
    record_line_count: int | None
    record_entries_well_formed: bool | None
    record_reports_no_such_file: bool
    record_first_line_contains_afdb: bool | None
    log_line_count: int | None
    log_reports_skipping: bool
    record_a3m_members: tuple[str, ...]
    missing_tar_members: tuple[str, ...]
    extra_tar_members: tuple[str, ...]
    shared_finished_paths_differ: bool
    retry_actions: tuple[PreprocessingRetryAction, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="PreprocessingChunkState")
        _validate_chunk_name(self.chunk_name)
        if not re.fullmatch(r"tranche\d{2}", self.tranche_name) or not re.fullmatch(
            rf"[^/]+_{re.escape(self.tranche_name)}_\d{{5}}\.fa", self.chunk_name
        ):
            msg = "chunk state name must match its exact tranche"
            raise ValueError(msg)
        if not isinstance(self.chunk_ordinal, int) or isinstance(self.chunk_ordinal, bool) or self.chunk_ordinal < 0:
            msg = "chunk_ordinal must be a non-negative integer"
            raise ValueError(msg)
        if self.state not in _CHUNK_STATE_KINDS:
            msg = f"unsupported preprocessing chunk state: {self.state!r}"
            raise ValueError(msg)
        if not isinstance(self.reason_codes, tuple) or not self.reason_codes:
            msg = "reason_codes must be a non-empty immutable tuple"
            raise ValueError(msg)
        if any(reason not in _STATE_REASONS for reason in self.reason_codes):
            msg = "reason_codes contain an unsupported preprocessing state reason"
            raise ValueError(msg)
        if len(set(self.reason_codes)) != len(self.reason_codes):
            msg = "reason_codes must not contain duplicates"
            raise ValueError(msg)
        if any(reason not in _STATE_REASONS_BY_KIND[self.state] for reason in self.reason_codes):
            msg = f"reason_codes are incompatible with {self.state} state"
            raise ValueError(msg)
        for field_name, value in (
            ("eligible_for_retry", self.eligible_for_retry),
            ("pristine_present", self.pristine_present),
            ("split_present", self.split_present),
            ("finished_input_present", self.finished_input_present),
            ("shared_finished_read_present", self.shared_finished_read_present),
            ("shared_finished_write_present", self.shared_finished_write_present),
            ("shared_finished_paths_differ", self.shared_finished_paths_differ),
        ):
            _validate_bool(value, field_name)
        _validate_chunk_paths(
            self.chunk_name,
            self.pristine_path,
            self.split_path,
            self.finished_input_path,
        )
        chunk_stem = self.chunk_name.removesuffix(".fa")
        expected_paths = (
            (self.shared_finished_read_tar_path, f"/{chunk_stem}.tar"),
            (self.shared_finished_write_tar_path, f"/{chunk_stem}.tar"),
            (self.durable_record_path, f"/{chunk_stem}.record"),
            (self.durable_log_path, f"/{chunk_stem}.log"),
            (self.durable_tar_path, f"/{chunk_stem}.tar"),
            (self.durable_lz4_path, f"/{chunk_stem}.tar.lz4"),
        )
        if any(not path.endswith(suffix) for path, suffix in expected_paths):
            msg = "chunk state evidence paths must be flat and keyed by chunk stem"
            raise ValueError(msg)
        _validate_optional_size(self.tar_size_bytes, "tar_size_bytes")
        _validate_optional_size(self.lz4_size_bytes, "lz4_size_bytes")
        if not isinstance(self.expected_a3ms, tuple) or not self.expected_a3ms:
            msg = "expected_a3ms must be a non-empty immutable tuple"
            raise ValueError(msg)
        if any(expected.chunk_name != self.chunk_name for expected in self.expected_a3ms):
            msg = "expected_a3ms must reference their state chunk"
            raise ValueError(msg)
        expected_members = tuple(expected.member_name for expected in self.expected_a3ms)
        expected_ordinals = tuple(expected.source_ordinal for expected in self.expected_a3ms)
        expected_identities = tuple(expected.record_identity for expected in self.expected_a3ms)
        if (
            len(set(expected_members)) != len(expected_members)
            or len(set(expected_ordinals)) != len(expected_ordinals)
            or len(set(expected_identities)) != len(expected_identities)
        ):
            msg = "expected_a3ms must preserve unique members and source associations"
            raise ValueError(msg)
        if (
            not isinstance(self.expected_num_records, int)
            or isinstance(self.expected_num_records, bool)
            or self.expected_num_records <= 0
        ):
            msg = "expected_num_records must be a positive integer"
            raise ValueError(msg)
        _validate_optional_size(self.pristine_record_count, "pristine_record_count")
        if self.pristine_present:
            if self.pristine_record_count != len(self.expected_a3ms):
                msg = "present pristine source count must match expected_a3ms"
                raise ValueError(msg)
        elif self.pristine_record_count is not None:
            msg = "absent pristine source cannot declare a record count"
            raise ValueError(msg)
        _validate_optional_size(self.record_line_count, "record_line_count")
        _validate_optional_size(self.log_line_count, "log_line_count")
        _validate_optional_bool(self.record_entries_well_formed, "record_entries_well_formed")
        _validate_bool(self.record_reports_no_such_file, "record_reports_no_such_file")
        _validate_optional_bool(
            self.record_first_line_contains_afdb,
            "record_first_line_contains_afdb",
        )
        _validate_bool(self.log_reports_skipping, "log_reports_skipping")
        if self.shared_finished_paths_differ != (
            self.shared_finished_read_tar_path != self.shared_finished_write_tar_path
        ):
            msg = "shared_finished_paths_differ must record the explicit shared read/write path mismatch"
            raise ValueError(msg)
        for field_name, member_tuple in (
            ("record_a3m_members", self.record_a3m_members),
            ("missing_tar_members", self.missing_tar_members),
            ("extra_tar_members", self.extra_tar_members),
        ):
            if not isinstance(member_tuple, tuple) or any(
                not isinstance(member, str) or not member for member in member_tuple
            ):
                msg = f"{field_name} must be an immutable tuple of non-empty names"
                raise ValueError(msg)
        if not set(self.missing_tar_members).issubset(self.record_a3m_members):
            msg = "missing_tar_members must be a subset of record_a3m_members"
            raise ValueError(msg)
        if set(self.extra_tar_members).intersection(self.record_a3m_members):
            msg = "extra_tar_members cannot also be record-listed A3Ms"
            raise ValueError(msg)
        if self.record_line_count is None or self.record_line_count == 0:
            if (
                self.record_entries_well_formed is not None
                or self.record_first_line_contains_afdb is not None
                or self.record_reports_no_such_file
                or self.record_a3m_members
            ):
                msg = "absent or empty record evidence cannot declare parsed record facts"
                raise ValueError(msg)
        else:
            if self.record_entries_well_formed is None or self.record_first_line_contains_afdb is None:
                msg = "nonempty record evidence requires explicit parsed record facts"
                raise ValueError(msg)
            if self.record_reports_no_such_file and (
                self.record_line_count != 1 or self.record_entries_well_formed or self.record_a3m_members
            ):
                msg = "No-such-file record evidence must be one malformed non-A3M line"
                raise ValueError(msg)
            if self.record_entries_well_formed and len(self.record_a3m_members) != self.record_line_count:
                msg = "well-formed record evidence must expose one A3M per line"
                raise ValueError(msg)
            if not self.record_entries_well_formed and len(self.record_a3m_members) > self.record_line_count:
                msg = "malformed record evidence cannot expose more A3Ms than lines"
                raise ValueError(msg)
        if (self.log_line_count is None or self.log_line_count == 0) and self.log_reports_skipping:
            msg = "absent or empty log evidence cannot report Skipping"
            raise ValueError(msg)
        if not isinstance(self.retry_actions, tuple):
            msg = "retry_actions must be an immutable tuple"
            raise ValueError(msg)
        if any(action.chunk_name != self.chunk_name for action in self.retry_actions):
            msg = "retry actions must reference their state chunk"
            raise ValueError(msg)
        expected_actions = derive_preprocessing_retry_actions(
            chunk_name=self.chunk_name,
            eligible_for_retry=self.eligible_for_retry,
            pristine_present=self.pristine_present,
            split_present=self.split_present,
            pristine_path=self.pristine_path,
            split_path=self.split_path,
            durable_tar_path=self.durable_tar_path,
            durable_lz4_path=self.durable_lz4_path,
            tar_present=self.tar_size_bytes is not None,
            lz4_present=self.lz4_size_bytes is not None,
            record_line_count=self.record_line_count,
            record_first_line_contains_afdb=self.record_first_line_contains_afdb,
            expected_num_records=self.expected_num_records,
        )
        if self.retry_actions != expected_actions:
            msg = "retry actions must exactly reflect state evidence; retry actions must match declared state paths"
            raise ValueError(msg)
        pristine_fallback_count = (
            self.record_line_count is not None
            and self.record_line_count < self.expected_num_records
            and self.pristine_record_count == self.record_line_count
        )
        record_inventory_matches = len(set(self.record_a3m_members)) == len(self.record_a3m_members) and set(
            self.record_a3m_members
        ) == set(expected_members)
        reason_facts = {
            "pristine_source_missing": not self.pristine_present,
            "contradictory_split_and_finished": self.split_present and self.finished_input_present,
            "missing_record_evidence": self.record_line_count is None,
            "missing_log_evidence": self.log_line_count is None,
            "empty_record_evidence": self.record_line_count == 0,
            "empty_log_evidence": self.log_line_count == 0,
            "record_reports_no_such_file": self.record_reports_no_such_file,
            "search_skipped": self.log_reports_skipping,
            "record_count_mismatch": self.record_line_count is not None
            and self.record_line_count != self.expected_num_records
            and not pristine_fallback_count,
            "record_first_line_missing_afdb": self.record_line_count == self.expected_num_records
            and self.record_first_line_contains_afdb is False,
            "malformed_record_entry": self.record_line_count is not None
            and self.record_line_count > 0
            and self.record_entries_well_formed is False
            and not self.record_reports_no_such_file,
            "record_members_mismatch": not record_inventory_matches,
            # dead under n-arity relaxation; retained for old evidence
            "a3m_count_not_above_two": self.record_entries_well_formed is True and len(self.record_a3m_members) <= 2,
            "tar_missing": self.tar_size_bytes is None,
            "tar_empty": self.tar_size_bytes == 0,
            "tar_lz4_missing": self.lz4_size_bytes is None,
            "tar_lz4_empty": self.lz4_size_bytes == 0,
            "tar_member_missing": bool(self.missing_tar_members),
            "finished_source_missing": self.split_present or not self.finished_input_present,
        }
        if self.state in {"retryable", "invalid"} and any(not reason_facts[reason] for reason in self.reason_codes):
            msg = f"{self.state} reason_codes must be proven by retained evidence facts"
            raise ValueError(msg)
        if self.state == "planned":
            planned_facts_are_coherent = (
                self.pristine_present
                and self.split_present
                and not self.finished_input_present
                and not self.shared_finished_read_present
                and not self.shared_finished_write_present
                and self.tar_size_bytes is None
                and self.lz4_size_bytes is None
                and self.record_line_count is None
                and self.log_line_count is None
                and not self.record_a3m_members
                and not self.missing_tar_members
                and not self.extra_tar_members
                and self.eligible_for_retry
            )
            if not planned_facts_are_coherent:
                msg = "planned state facts are incoherent"
                raise ValueError(msg)
        elif self.state == "missing":
            missing_facts_are_coherent = (
                self.pristine_present
                and not self.split_present
                and not self.finished_input_present
                and not self.shared_finished_read_present
                and not self.shared_finished_write_present
                and self.tar_size_bytes is None
                and self.lz4_size_bytes is None
                and self.record_line_count is None
                and self.log_line_count is None
                and not self.record_a3m_members
                and not self.missing_tar_members
                and not self.extra_tar_members
                and self.eligible_for_retry
            )
            if not missing_facts_are_coherent:
                msg = "missing state facts are incoherent"
                raise ValueError(msg)
        elif self.state == "completed":
            completed_evidence_is_coherent = (
                self.reason_codes == ("complete",)
                and not self.eligible_for_retry
                and not self.retry_actions
                and self.pristine_present
                and not self.split_present
                and self.finished_input_present
                and self.tar_size_bytes is not None
                and self.tar_size_bytes > 0
                and self.lz4_size_bytes is not None
                and self.lz4_size_bytes > 0
                and len(self.record_a3m_members) >= 1
                and len(set(self.record_a3m_members)) == len(self.record_a3m_members)
                and len(self.expected_a3ms) >= 1
                and not self.missing_tar_members
                and self.record_entries_well_formed is True
                and self.record_line_count is not None
                and (self.record_line_count == self.expected_num_records or pristine_fallback_count)
                and not self.record_reports_no_such_file
                and self.log_line_count is not None
                and self.log_line_count > 0
                and not self.log_reports_skipping
                and (
                    self.record_line_count != self.expected_num_records or self.record_first_line_contains_afdb is True
                )
            )
            if not completed_evidence_is_coherent:
                msg = "completed state requires coherent source and archive evidence"
                raise ValueError(msg)
            if set(self.record_a3m_members) != set(expected_members):
                msg = "completed state requires exact declared A3M membership"
                raise ValueError(msg)
        elif self.state == "retryable":
            if (
                not self.pristine_present
                or (self.split_present and self.finished_input_present)
                or not self.eligible_for_retry
            ):
                msg = "retryable state facts are incoherent"
                raise ValueError(msg)
        else:
            has_pristine_missing = "pristine_source_missing" in self.reason_codes
            has_source_contradiction = "contradictory_split_and_finished" in self.reason_codes
            if has_pristine_missing:
                invalid_facts_are_coherent = (
                    self.reason_codes == ("pristine_source_missing",)
                    and not self.pristine_present
                    and not self.eligible_for_retry
                )
            elif has_source_contradiction:
                invalid_facts_are_coherent = (
                    self.reason_codes == ("contradictory_split_and_finished",)
                    and self.pristine_present
                    and self.split_present
                    and self.finished_input_present
                    and not self.eligible_for_retry
                )
            else:
                invalid_facts_are_coherent = (
                    self.pristine_present
                    and not (self.split_present and self.finished_input_present)
                    and self.eligible_for_retry
                )
            if not invalid_facts_are_coherent:
                msg = "invalid state facts are incoherent"
                raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready chunk state data."""
        return {
            "schema_version": self.schema_version,
            "chunk_name": self.chunk_name,
            "tranche_name": self.tranche_name,
            "chunk_ordinal": self.chunk_ordinal,
            "state": self.state,
            "reason_codes": list(self.reason_codes),
            "eligible_for_retry": self.eligible_for_retry,
            "pristine_present": self.pristine_present,
            "split_present": self.split_present,
            "finished_input_present": self.finished_input_present,
            "pristine_path": self.pristine_path,
            "split_path": self.split_path,
            "finished_input_path": self.finished_input_path,
            "shared_finished_read_present": self.shared_finished_read_present,
            "shared_finished_write_present": self.shared_finished_write_present,
            "shared_finished_read_tar_path": self.shared_finished_read_tar_path,
            "shared_finished_write_tar_path": self.shared_finished_write_tar_path,
            "durable_record_path": self.durable_record_path,
            "durable_log_path": self.durable_log_path,
            "durable_tar_path": self.durable_tar_path,
            "durable_lz4_path": self.durable_lz4_path,
            "tar_size_bytes": self.tar_size_bytes,
            "lz4_size_bytes": self.lz4_size_bytes,
            "expected_a3ms": [expected.to_mapping() for expected in self.expected_a3ms],
            "expected_num_records": self.expected_num_records,
            "pristine_record_count": self.pristine_record_count,
            "record_line_count": self.record_line_count,
            "record_entries_well_formed": self.record_entries_well_formed,
            "record_reports_no_such_file": self.record_reports_no_such_file,
            "record_first_line_contains_afdb": self.record_first_line_contains_afdb,
            "log_line_count": self.log_line_count,
            "log_reports_skipping": self.log_reports_skipping,
            "record_a3m_members": list(self.record_a3m_members),
            "missing_tar_members": list(self.missing_tar_members),
            "extra_tar_members": list(self.extra_tar_members),
            "shared_finished_paths_differ": self.shared_finished_paths_differ,
            "retry_actions": [action.to_mapping() for action in self.retry_actions],
        }


@dataclass(frozen=True)
class PreprocessingTrancheProgress:
    """Deterministic progress, with unavailable baseline percent represented by null."""

    tranche_name: str
    pristine_chunks: int
    remaining_split_chunks: int
    finished_chunks: int
    completed_chunks: int
    eligible_chunks: int
    invalid_chunks: int
    total_chunks: int
    completion_percentage: str | None
    validated_completion_percentage: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="PreprocessingTrancheProgress")
        if re.fullmatch(r"tranche\d{2}", self.tranche_name) is None:
            msg = "tranche_name must use an exact two-digit suffix"
            raise ValueError(msg)
        for field_name, value in (
            ("pristine_chunks", self.pristine_chunks),
            ("remaining_split_chunks", self.remaining_split_chunks),
            ("finished_chunks", self.finished_chunks),
            ("completed_chunks", self.completed_chunks),
            ("eligible_chunks", self.eligible_chunks),
            ("invalid_chunks", self.invalid_chunks),
            ("total_chunks", self.total_chunks),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                msg = f"{field_name} must be a non-negative integer"
                raise ValueError(msg)
        if self.completed_chunks > self.total_chunks:
            msg = "completed_chunks cannot exceed total_chunks"
            raise ValueError(msg)
        if any(
            count > self.total_chunks
            for count in (
                self.pristine_chunks,
                self.remaining_split_chunks,
                self.finished_chunks,
                self.eligible_chunks,
                self.invalid_chunks,
            )
        ):
            msg = "progress counts cannot exceed total_chunks"
            raise ValueError(msg)
        if self.completed_chunks > self.pristine_chunks or self.completed_chunks > self.finished_chunks:
            msg = "completed_chunks cannot exceed pristine or finished counts"
            raise ValueError(msg)
        if self.completed_chunks + self.eligible_chunks > self.total_chunks:
            msg = "completed and eligible chunks cannot overlap"
            raise ValueError(msg)
        if self.completion_percentage != _preprocessing_baseline_completion_percentage(
            self.pristine_chunks,
            self.remaining_split_chunks,
        ):
            msg = "completion_percentage must match coherent pristine and remaining split counts"
            raise ValueError(msg)
        if self.validated_completion_percentage != preprocessing_completion_percentage(
            self.completed_chunks,
            self.total_chunks,
        ):
            msg = "validated_completion_percentage must match completed and total chunks"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready tranche progress data."""
        return {
            "schema_version": self.schema_version,
            "tranche_name": self.tranche_name,
            "pristine_chunks": self.pristine_chunks,
            "remaining_split_chunks": self.remaining_split_chunks,
            "finished_chunks": self.finished_chunks,
            "completed_chunks": self.completed_chunks,
            "eligible_chunks": self.eligible_chunks,
            "invalid_chunks": self.invalid_chunks,
            "total_chunks": self.total_chunks,
            "completion_percentage": self.completion_percentage,
            "validated_completion_percentage": self.validated_completion_percentage,
        }


def summarize_preprocessing_tranche_progress(
    states: tuple[PreprocessingChunkState, ...],
) -> tuple[PreprocessingTrancheProgress, ...]:
    """Summarize chunk state using stable first-seen tranche order."""
    if not isinstance(states, tuple):
        msg = "states must be an immutable tuple"
        raise ValueError(msg)
    tranche_names = tuple(dict.fromkeys(state.tranche_name for state in states))
    progress: list[PreprocessingTrancheProgress] = []
    for tranche_name in tranche_names:
        tranche_states = tuple(state for state in states if state.tranche_name == tranche_name)
        completed = sum(state.state == "completed" for state in tranche_states)
        total = len(tranche_states)
        pristine = sum(state.pristine_present for state in tranche_states)
        remaining_split = sum(state.split_present for state in tranche_states)
        progress.append(
            PreprocessingTrancheProgress(
                tranche_name=tranche_name,
                pristine_chunks=pristine,
                remaining_split_chunks=remaining_split,
                finished_chunks=sum(state.finished_input_present for state in tranche_states),
                completed_chunks=completed,
                eligible_chunks=sum(state.eligible_for_retry for state in tranche_states),
                invalid_chunks=sum(state.state == "invalid" for state in tranche_states),
                total_chunks=total,
                completion_percentage=_preprocessing_baseline_completion_percentage(pristine, remaining_split),
                validated_completion_percentage=preprocessing_completion_percentage(completed, total),
            )
        )
    return tuple(progress)


@dataclass(frozen=True)
class PreprocessingRetryPlan:
    """Immutable retry/top-up selection that preserves completed work."""

    work_plan: PreprocessingWorkPlan
    states: tuple[PreprocessingChunkState, ...]
    eligible_chunk_names: tuple[str, ...]
    eligible_chunks: tuple[PreprocessingChunk, ...]
    eligible_assignments: tuple[PreprocessingChunkAssignment, ...]
    invalid_chunk_names: tuple[str, ...]
    actions: tuple[PreprocessingRetryAction, ...]
    tranche_progress: tuple[PreprocessingTrancheProgress, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="PreprocessingRetryPlan")
        if not isinstance(self.work_plan, PreprocessingWorkPlan):
            msg = "work_plan must be an immutable PreprocessingWorkPlan"
            raise ValueError(msg)
        for field_name, value in (
            ("states", self.states),
            ("eligible_chunk_names", self.eligible_chunk_names),
            ("eligible_chunks", self.eligible_chunks),
            ("eligible_assignments", self.eligible_assignments),
            ("invalid_chunk_names", self.invalid_chunk_names),
            ("actions", self.actions),
            ("tranche_progress", self.tranche_progress),
        ):
            if not isinstance(value, tuple):
                msg = f"{field_name} must be an immutable tuple"
                raise ValueError(msg)
        state_names = tuple(state.chunk_name for state in self.states)
        if len(set(state_names)) != len(state_names):
            msg = "states must name each chunk exactly once"
            raise ValueError(msg)
        expected_state_refs = tuple((chunk.name, chunk.tranche_name, chunk.ordinal) for chunk in self.work_plan.chunks)
        state_refs = tuple((state.chunk_name, state.tranche_name, state.chunk_ordinal) for state in self.states)
        if state_refs != expected_state_refs:
            msg = "states must exactly cover source work-plan chunks"
            raise ValueError(msg)
        input_records_by_ordinal = {record.source_ordinal: record for record in self.work_plan.input.records}
        for chunk, state in zip(self.work_plan.chunks, self.states, strict=True):
            source_records = tuple(input_records_by_ordinal[ordinal] for ordinal in chunk.record_ordinals)
            expected_source_associations = tuple(
                (record.identity, record.source_ordinal, record.header) for record in source_records
            )
            state_source_associations = tuple(
                (expected.record_identity, expected.source_ordinal, expected.source_header)
                for expected in state.expected_a3ms
            )
            if state_source_associations != expected_source_associations:
                msg = "state expected_a3ms must exactly match source work-plan records"
                raise ValueError(msg)
        eligible_from_states = tuple(state.chunk_name for state in self.states if state.eligible_for_retry)
        if self.eligible_chunk_names != eligible_from_states:
            msg = "eligible_chunk_names must exactly select eligible states in stable order"
            raise ValueError(msg)
        eligible_name_set = set(self.eligible_chunk_names)
        expected_eligible_chunks = tuple(chunk for chunk in self.work_plan.chunks if chunk.name in eligible_name_set)
        if self.eligible_chunks != expected_eligible_chunks:
            msg = "eligible_chunks must be the exact source-plan selection"
            raise ValueError(msg)
        expected_eligible_assignments = tuple(
            assignment for assignment in self.work_plan.assignments if assignment.chunk_name in eligible_name_set
        )
        if self.eligible_assignments != expected_eligible_assignments:
            msg = "eligible_assignments must be the exact source-plan selection"
            raise ValueError(msg)
        expected_invalid = tuple(state.chunk_name for state in self.states if state.state == "invalid")
        if self.invalid_chunk_names != expected_invalid:
            msg = "invalid_chunk_names must report every invalid state"
            raise ValueError(msg)
        expected_actions = tuple(action for state in self.states for action in state.retry_actions)
        if self.actions != expected_actions:
            msg = "retry plan actions must exactly aggregate state actions"
            raise ValueError(msg)
        if self.tranche_progress != summarize_preprocessing_tranche_progress(self.states):
            msg = "tranche_progress must exactly summarize states"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready retry-plan data."""
        return {
            "schema_version": self.schema_version,
            "work_plan": self.work_plan.to_mapping(),
            "states": [state.to_mapping() for state in self.states],
            "eligible_chunk_names": list(self.eligible_chunk_names),
            "eligible_chunks": [chunk.to_mapping() for chunk in self.eligible_chunks],
            "eligible_assignments": [assignment.to_mapping() for assignment in self.eligible_assignments],
            "invalid_chunk_names": list(self.invalid_chunk_names),
            "actions": [action.to_mapping() for action in self.actions],
            "tranche_progress": [progress.to_mapping() for progress in self.tranche_progress],
        }


def preprocessing_source_membership_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingSourceMembership:
    """Load explicit source membership from a fail-closed mapping."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "chunk_name",
            "pristine_path",
            "pristine_present",
            "pristine_record_count",
            "split_path",
            "split_present",
            "finished_input_path",
            "finished_input_present",
            "shared_finished_read_tar_path",
            "shared_finished_read_present",
            "shared_finished_write_tar_path",
            "shared_finished_write_present",
        },
        "PreprocessingSourceMembership",
    )
    return PreprocessingSourceMembership(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingSourceMembership"
        ),
        chunk_name=_required_str(payload, "chunk_name"),
        pristine_path=_required_str(payload, "pristine_path"),
        pristine_present=_required_bool(payload, "pristine_present"),
        pristine_record_count=_required_optional_int(payload, "pristine_record_count"),
        split_path=_required_str(payload, "split_path"),
        split_present=_required_bool(payload, "split_present"),
        finished_input_path=_required_str(payload, "finished_input_path"),
        finished_input_present=_required_bool(payload, "finished_input_present"),
        shared_finished_read_tar_path=_required_str(payload, "shared_finished_read_tar_path"),
        shared_finished_read_present=_required_bool(payload, "shared_finished_read_present"),
        shared_finished_write_tar_path=_required_str(payload, "shared_finished_write_tar_path"),
        shared_finished_write_present=_required_bool(payload, "shared_finished_write_present"),
    )


def preprocessing_paired_evidence_from_mapping(payload: Mapping[str, object]) -> PreprocessingPairedEvidence:
    """Load paired durable record/log evidence from a fail-closed mapping."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "chunk_name",
            "durable_record_path",
            "durable_log_path",
            "record_lines",
            "log_lines",
            "record_streams_merged",
            "log_streams_merged",
        },
        "PreprocessingPairedEvidence",
    )
    return PreprocessingPairedEvidence(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingPairedEvidence"
        ),
        chunk_name=_required_str(payload, "chunk_name"),
        durable_record_path=_required_str(payload, "durable_record_path"),
        durable_log_path=_required_str(payload, "durable_log_path"),
        record_lines=_required_optional_str_tuple(payload, "record_lines"),
        log_lines=_required_optional_str_tuple(payload, "log_lines"),
        record_streams_merged=_required_bool(payload, "record_streams_merged"),
        log_streams_merged=_required_bool(payload, "log_streams_merged"),
    )


def preprocessing_archive_evidence_from_mapping(payload: Mapping[str, object]) -> PreprocessingArchiveEvidence:
    """Load durable archive observations from a fail-closed mapping."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "chunk_name",
            "durable_tar_path",
            "durable_lz4_path",
            "tar_size_bytes",
            "lz4_size_bytes",
            "tar_members",
        },
        "PreprocessingArchiveEvidence",
    )
    return PreprocessingArchiveEvidence(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingArchiveEvidence"
        ),
        chunk_name=_required_str(payload, "chunk_name"),
        durable_tar_path=_required_str(payload, "durable_tar_path"),
        durable_lz4_path=_required_str(payload, "durable_lz4_path"),
        tar_size_bytes=_required_optional_int(payload, "tar_size_bytes"),
        lz4_size_bytes=_required_optional_int(payload, "lz4_size_bytes"),
        tar_members=_required_optional_str_tuple(payload, "tar_members"),
    )


def preprocessing_retry_action_from_mapping(payload: Mapping[str, object]) -> PreprocessingRetryAction:
    """Load one retry action from a fail-closed mapping."""
    _reject_unknown_fields(
        payload,
        {"schema_version", "chunk_name", "operation", "source_path", "target_path"},
        "PreprocessingRetryAction",
    )
    operation = _required_str(payload, "operation")
    if operation not in {"copy", "remove"}:
        msg = f"unsupported retry operation: {operation!r}"
        raise ValueError(msg)
    return PreprocessingRetryAction(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PreprocessingRetryAction"),
        chunk_name=_required_str(payload, "chunk_name"),
        operation=cast("PreprocessingRetryOperation", operation),
        source_path=_required_optional_str(payload, "source_path"),
        target_path=_required_str(payload, "target_path"),
    )


def preprocessing_chunk_state_from_mapping(payload: Mapping[str, object]) -> PreprocessingChunkState:
    """Load one interpreted chunk state from a fail-closed mapping."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "chunk_name",
            "tranche_name",
            "chunk_ordinal",
            "state",
            "reason_codes",
            "eligible_for_retry",
            "pristine_present",
            "split_present",
            "finished_input_present",
            "pristine_path",
            "split_path",
            "finished_input_path",
            "shared_finished_read_present",
            "shared_finished_write_present",
            "shared_finished_read_tar_path",
            "shared_finished_write_tar_path",
            "durable_record_path",
            "durable_log_path",
            "durable_tar_path",
            "durable_lz4_path",
            "tar_size_bytes",
            "lz4_size_bytes",
            "expected_a3ms",
            "expected_num_records",
            "pristine_record_count",
            "record_line_count",
            "record_entries_well_formed",
            "record_reports_no_such_file",
            "record_first_line_contains_afdb",
            "log_line_count",
            "log_reports_skipping",
            "record_a3m_members",
            "missing_tar_members",
            "extra_tar_members",
            "shared_finished_paths_differ",
            "retry_actions",
        },
        "PreprocessingChunkState",
    )
    state = _required_str(payload, "state")
    if state not in _CHUNK_STATE_KINDS:
        msg = f"unsupported preprocessing chunk state: {state!r}"
        raise ValueError(msg)
    reasons = _required_str_tuple(payload, "reason_codes")
    if not reasons or any(reason not in _STATE_REASONS for reason in reasons):
        msg = "reason_codes contain an unsupported preprocessing state reason"
        raise ValueError(msg)
    return PreprocessingChunkState(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PreprocessingChunkState"),
        chunk_name=_required_str(payload, "chunk_name"),
        tranche_name=_required_str(payload, "tranche_name"),
        chunk_ordinal=_required_int(payload, "chunk_ordinal"),
        state=cast("PreprocessingChunkStateKind", state),
        reason_codes=cast("tuple[PreprocessingStateReason, ...]", reasons),
        eligible_for_retry=_required_bool(payload, "eligible_for_retry"),
        pristine_present=_required_bool(payload, "pristine_present"),
        split_present=_required_bool(payload, "split_present"),
        finished_input_present=_required_bool(payload, "finished_input_present"),
        pristine_path=_required_str(payload, "pristine_path"),
        split_path=_required_str(payload, "split_path"),
        finished_input_path=_required_str(payload, "finished_input_path"),
        shared_finished_read_present=_required_bool(payload, "shared_finished_read_present"),
        shared_finished_write_present=_required_bool(payload, "shared_finished_write_present"),
        shared_finished_read_tar_path=_required_str(payload, "shared_finished_read_tar_path"),
        shared_finished_write_tar_path=_required_str(payload, "shared_finished_write_tar_path"),
        durable_record_path=_required_str(payload, "durable_record_path"),
        durable_log_path=_required_str(payload, "durable_log_path"),
        durable_tar_path=_required_str(payload, "durable_tar_path"),
        durable_lz4_path=_required_str(payload, "durable_lz4_path"),
        tar_size_bytes=_required_optional_int(payload, "tar_size_bytes"),
        lz4_size_bytes=_required_optional_int(payload, "lz4_size_bytes"),
        expected_a3ms=tuple(
            expected_a3m_from_mapping(item) for item in _required_mapping_sequence(payload, "expected_a3ms")
        ),
        expected_num_records=_required_int(payload, "expected_num_records"),
        pristine_record_count=_required_optional_int(payload, "pristine_record_count"),
        record_line_count=_required_optional_int(payload, "record_line_count"),
        record_entries_well_formed=_required_optional_bool(payload, "record_entries_well_formed"),
        record_reports_no_such_file=_required_bool(payload, "record_reports_no_such_file"),
        record_first_line_contains_afdb=_required_optional_bool(
            payload,
            "record_first_line_contains_afdb",
        ),
        log_line_count=_required_optional_int(payload, "log_line_count"),
        log_reports_skipping=_required_bool(payload, "log_reports_skipping"),
        record_a3m_members=_required_str_tuple(payload, "record_a3m_members"),
        missing_tar_members=_required_str_tuple(payload, "missing_tar_members"),
        extra_tar_members=_required_str_tuple(payload, "extra_tar_members"),
        shared_finished_paths_differ=_required_bool(payload, "shared_finished_paths_differ"),
        retry_actions=tuple(
            preprocessing_retry_action_from_mapping(item)
            for item in _required_mapping_sequence(payload, "retry_actions")
        ),
    )


def preprocessing_tranche_progress_from_mapping(payload: Mapping[str, object]) -> PreprocessingTrancheProgress:
    """Load one tranche progress summary from a fail-closed mapping."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "tranche_name",
            "pristine_chunks",
            "remaining_split_chunks",
            "finished_chunks",
            "completed_chunks",
            "eligible_chunks",
            "invalid_chunks",
            "total_chunks",
            "completion_percentage",
            "validated_completion_percentage",
        },
        "PreprocessingTrancheProgress",
    )
    return PreprocessingTrancheProgress(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingTrancheProgress"
        ),
        tranche_name=_required_str(payload, "tranche_name"),
        pristine_chunks=_required_int(payload, "pristine_chunks"),
        remaining_split_chunks=_required_int(payload, "remaining_split_chunks"),
        finished_chunks=_required_int(payload, "finished_chunks"),
        completed_chunks=_required_int(payload, "completed_chunks"),
        eligible_chunks=_required_int(payload, "eligible_chunks"),
        invalid_chunks=_required_int(payload, "invalid_chunks"),
        total_chunks=_required_int(payload, "total_chunks"),
        completion_percentage=_required_optional_str(payload, "completion_percentage"),
        validated_completion_percentage=_required_str(payload, "validated_completion_percentage"),
    )


def preprocessing_retry_plan_from_mapping(payload: Mapping[str, object]) -> PreprocessingRetryPlan:
    """Load a complete retry/top-up plan and reject recursive schema drift."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "work_plan",
            "states",
            "eligible_chunk_names",
            "eligible_chunks",
            "eligible_assignments",
            "invalid_chunk_names",
            "actions",
            "tranche_progress",
        },
        "PreprocessingRetryPlan",
    )
    return PreprocessingRetryPlan(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PreprocessingRetryPlan"),
        work_plan=preprocessing_work_plan_from_mapping(_required_mapping(payload, "work_plan")),
        states=tuple(
            preprocessing_chunk_state_from_mapping(item) for item in _required_mapping_sequence(payload, "states")
        ),
        eligible_chunk_names=_required_str_tuple(payload, "eligible_chunk_names"),
        eligible_chunks=tuple(
            preprocessing_chunk_from_mapping(item) for item in _required_mapping_sequence(payload, "eligible_chunks")
        ),
        eligible_assignments=tuple(
            preprocessing_chunk_assignment_from_mapping(item)
            for item in _required_mapping_sequence(payload, "eligible_assignments")
        ),
        invalid_chunk_names=_required_str_tuple(payload, "invalid_chunk_names"),
        actions=tuple(
            preprocessing_retry_action_from_mapping(item) for item in _required_mapping_sequence(payload, "actions")
        ),
        tranche_progress=tuple(
            preprocessing_tranche_progress_from_mapping(item)
            for item in _required_mapping_sequence(payload, "tranche_progress")
        ),
    )


def _reject_unknown_fields(payload: Mapping[str, object], allowed: set[str], record_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        msg = f"Unknown {record_name} field(s): {', '.join(unknown)}"
        raise ValueError(msg)


def _required_mapping_sequence(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"{key}[{index}] must be a mapping"
            raise ValueError(msg)
        result.append(item)
    return tuple(result)


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be a mapping"
        raise ValueError(msg)
    return value


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{key} must be a non-empty string"
        raise ValueError(msg)
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


def _required_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean"
        raise ValueError(msg)
    return value


def _required_optional_bool(payload: Mapping[str, object], key: str) -> bool | None:
    if key not in payload:
        msg = f"{key} is required"
        raise ValueError(msg)
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, bool):
        msg = f"{key} must be null or a boolean"
        raise ValueError(msg)
    return value


def _required_optional_int(payload: Mapping[str, object], key: str) -> int | None:
    if key not in payload:
        msg = f"{key} is required"
        raise ValueError(msg)
    value = payload[key]
    if value is None:
        return None
    return _required_int(payload, key)


def _required_optional_str(payload: Mapping[str, object], key: str) -> str | None:
    if key not in payload:
        msg = f"{key} is required"
        raise ValueError(msg)
    value = payload[key]
    if value is None:
        return None
    return _required_str(payload, key)


def _required_str_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple) or any(not isinstance(item, str) for item in value):
        msg = f"{key} must be a list of strings"
        raise ValueError(msg)
    return tuple(value)


def _required_optional_str_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...] | None:
    if key not in payload:
        msg = f"{key} is required"
        raise ValueError(msg)
    value = payload[key]
    if value is None:
        return None
    return _required_str_tuple(payload, key)


def preprocessing_completion_percentage(completed: int, total: int) -> str:
    """Return deterministic, truncated preprocessing completion percentage."""
    for field_name, value in (("completed", completed), ("total", total)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            msg = f"{field_name} must be a non-negative integer"
            raise ValueError(msg)
    if completed > total:
        msg = "completed cannot exceed total"
        raise ValueError(msg)
    if total == 0:
        return "0.00"
    hundredths = completed * 10_000 // total
    return f"{hundredths // 100}.{hundredths % 100:02d}"


def _preprocessing_baseline_completion_percentage(
    pristine_chunks: int,
    remaining_split_chunks: int,
) -> str | None:
    """Mirror baseline progress when its inventory relation is coherent."""
    for field_name, value in (
        ("pristine_chunks", pristine_chunks),
        ("remaining_split_chunks", remaining_split_chunks),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            msg = f"{field_name} must be a non-negative integer"
            raise ValueError(msg)
    if remaining_split_chunks > pristine_chunks:
        return None
    return preprocessing_completion_percentage(
        pristine_chunks - remaining_split_chunks,
        pristine_chunks,
    )


def _validate_chunk_name(chunk_name: str) -> str:
    if not isinstance(chunk_name, str) or not chunk_name.endswith(".fa") or "/" in chunk_name:
        msg = "chunk_name must be a top-level .fa name"
        raise ValueError(msg)
    return chunk_name.removesuffix(".fa")


def _validate_chunk_paths(chunk_name: str, *paths: str) -> None:
    _validate_chunk_name(chunk_name)
    if any(not isinstance(path, str) or not path.endswith(f"/{chunk_name}") for path in paths):
        msg = "source membership paths must be flat and keyed by chunk name"
        raise ValueError(msg)


def _validate_bool(value: bool, field_name: str) -> None:
    if not isinstance(value, bool):
        msg = f"{field_name} must be a boolean"
        raise ValueError(msg)


def _validate_optional_bool(value: bool | None, field_name: str) -> None:
    if value is not None and not isinstance(value, bool):
        msg = f"{field_name} must be null or a boolean"
        raise ValueError(msg)


def _validate_optional_lines(value: tuple[str, ...] | None, field_name: str) -> None:
    if value is not None and (not isinstance(value, tuple) or any(not isinstance(line, str) for line in value)):
        msg = f"{field_name} must be null or an immutable tuple of strings"
        raise ValueError(msg)


def _validate_optional_size(value: int | None, field_name: str) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        msg = f"{field_name} must be null or a non-negative integer"
        raise ValueError(msg)


__all__ = [
    "PreprocessingArchiveEvidence",
    "PreprocessingChunkState",
    "PreprocessingChunkStateKind",
    "PreprocessingPairedEvidence",
    "PreprocessingRetryAction",
    "PreprocessingRetryOperation",
    "PreprocessingRetryPlan",
    "PreprocessingSourceMembership",
    "PreprocessingStateReason",
    "PreprocessingTrancheProgress",
    "derive_preprocessing_retry_actions",
    "preprocessing_archive_evidence_from_mapping",
    "preprocessing_chunk_state_from_mapping",
    "preprocessing_completion_percentage",
    "preprocessing_paired_evidence_from_mapping",
    "preprocessing_retry_action_from_mapping",
    "preprocessing_retry_plan_from_mapping",
    "preprocessing_source_membership_from_mapping",
    "preprocessing_tranche_progress_from_mapping",
    "summarize_preprocessing_tranche_progress",
]
