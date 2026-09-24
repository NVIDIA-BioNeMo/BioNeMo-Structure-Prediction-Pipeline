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

"""Pure preprocessing evidence interpretation.

Port Baseline:
419813dbb5a3949e5e16f289f974d9f95e94bf01:scripts/examine_results_and_replace_input.py:29-184.
"""

from __future__ import annotations

import re

from bspp.orchestration.contract.preprocessing import PreprocessingChunk
from bspp.orchestration.contract.preprocessing_execution import (
    ExpectedA3M,
    PreprocessingEvidencePlan,
    PreprocessingPackagePlan,
    is_pdb_assembly_member_name,
)
from bspp.orchestration.contract.preprocessing_state import (
    PreprocessingArchiveEvidence,
    PreprocessingChunkState,
    PreprocessingChunkStateKind,
    PreprocessingPairedEvidence,
    PreprocessingSourceMembership,
    PreprocessingStateReason,
    derive_preprocessing_retry_actions,
)


def interpret_preprocessing_chunk_state(
    *,
    chunk: PreprocessingChunk,
    expected_a3ms: tuple[ExpectedA3M, ...],
    evidence_plan: PreprocessingEvidencePlan,
    package_plan: PreprocessingPackagePlan,
    expected_num_records: int,
    source: PreprocessingSourceMembership,
    paired_evidence: PreprocessingPairedEvidence,
    archive_evidence: PreprocessingArchiveEvidence,
) -> PreprocessingChunkState:
    """Interpret explicit durable evidence without discovering or mutating files."""
    _validate_inputs(
        chunk=chunk,
        expected_a3ms=expected_a3ms,
        evidence_plan=evidence_plan,
        package_plan=package_plan,
        expected_num_records=expected_num_records,
        source=source,
        paired_evidence=paired_evidence,
        archive_evidence=archive_evidence,
    )
    if not source.pristine_present:
        return _state(
            chunk=chunk,
            evidence_plan=evidence_plan,
            package_plan=package_plan,
            archive_evidence=archive_evidence,
            expected_a3ms=expected_a3ms,
            expected_num_records=expected_num_records,
            paired_evidence=paired_evidence,
            source=source,
            state="invalid",
            reason_codes=("pristine_source_missing",),
            eligible_for_retry=False,
        )
    if source.split_present and source.finished_input_present:
        return _state(
            chunk=chunk,
            evidence_plan=evidence_plan,
            package_plan=package_plan,
            archive_evidence=archive_evidence,
            expected_a3ms=expected_a3ms,
            expected_num_records=expected_num_records,
            paired_evidence=paired_evidence,
            source=source,
            state="invalid",
            reason_codes=("contradictory_split_and_finished",),
            eligible_for_retry=False,
        )
    if (
        source.pristine_present
        and source.split_present
        and not source.finished_input_present
        and paired_evidence.record_lines is None
        and paired_evidence.log_lines is None
        and archive_evidence.tar_size_bytes is None
        and archive_evidence.lz4_size_bytes is None
        and not source.shared_finished_read_present
        and not source.shared_finished_write_present
    ):
        return _state(
            chunk=chunk,
            evidence_plan=evidence_plan,
            package_plan=package_plan,
            archive_evidence=archive_evidence,
            expected_a3ms=expected_a3ms,
            expected_num_records=expected_num_records,
            paired_evidence=paired_evidence,
            source=source,
            state="planned",
            reason_codes=("awaiting_execution",),
            eligible_for_retry=True,
        )
    if (
        source.pristine_present
        and not source.split_present
        and not source.finished_input_present
        and paired_evidence.record_lines is None
        and paired_evidence.log_lines is None
        and archive_evidence.tar_size_bytes is None
        and archive_evidence.lz4_size_bytes is None
        and not source.shared_finished_read_present
        and not source.shared_finished_write_present
    ):
        return _state(
            chunk=chunk,
            evidence_plan=evidence_plan,
            package_plan=package_plan,
            archive_evidence=archive_evidence,
            expected_a3ms=expected_a3ms,
            expected_num_records=expected_num_records,
            paired_evidence=paired_evidence,
            source=source,
            state="missing",
            reason_codes=("ghosted_source",),
            eligible_for_retry=True,
        )
    record_lines = paired_evidence.record_lines
    log_lines = paired_evidence.log_lines
    partial_reasons: list[PreprocessingStateReason] = []
    if record_lines is None:
        partial_reasons.append("missing_record_evidence")
    elif not record_lines:
        partial_reasons.append("empty_record_evidence")
    if log_lines is None:
        partial_reasons.append("missing_log_evidence")
    elif not log_lines:
        partial_reasons.append("empty_log_evidence")
    if partial_reasons:
        return _state(
            chunk=chunk,
            evidence_plan=evidence_plan,
            package_plan=package_plan,
            archive_evidence=archive_evidence,
            expected_a3ms=expected_a3ms,
            expected_num_records=expected_num_records,
            paired_evidence=paired_evidence,
            source=source,
            state="retryable",
            reason_codes=tuple(partial_reasons),
            eligible_for_retry=source.pristine_present,
        )
    if record_lines and log_lines:
        record_members = _record_a3m_members(record_lines)
        # A record's first line identifies the source family either by the
        # legacy ``AFDB`` substring or by a pdb-assembly member name.
        # This single predicate is reused by the invalid/completed branches
        # below and by the fact recorder so all three stay consistent.
        record_first_line_identifies_afdb_or_pdb = "AFDB" in record_lines[0] or any(
            is_pdb_assembly_member_name(member) for member in record_members
        )
        expected_members = tuple(expected.member_name for expected in expected_a3ms)
        tar_members = tuple(_normalize_tar_member(member) for member in (archive_evidence.tar_members or ()))
        extra_tar_members = tuple(member for member in tar_members if member not in record_members)
        if len(record_lines) == 1 and "No such file" in record_lines[0]:
            return _state(
                chunk=chunk,
                evidence_plan=evidence_plan,
                package_plan=package_plan,
                archive_evidence=archive_evidence,
                expected_a3ms=expected_a3ms,
                expected_num_records=expected_num_records,
                paired_evidence=paired_evidence,
                source=source,
                state="retryable",
                reason_codes=("record_reports_no_such_file",),
                eligible_for_retry=True,
            )
        if len(record_members) != len(record_lines):
            return _state(
                chunk=chunk,
                evidence_plan=evidence_plan,
                package_plan=package_plan,
                archive_evidence=archive_evidence,
                expected_a3ms=expected_a3ms,
                expected_num_records=expected_num_records,
                paired_evidence=paired_evidence,
                source=source,
                state="invalid",
                reason_codes=("malformed_record_entry",),
                eligible_for_retry=True,
                record_a3m_members=record_members,
                extra_tar_members=extra_tar_members,
            )
        if any("Skipping" in line for line in log_lines):
            return _state(
                chunk=chunk,
                evidence_plan=evidence_plan,
                package_plan=package_plan,
                archive_evidence=archive_evidence,
                expected_a3ms=expected_a3ms,
                expected_num_records=expected_num_records,
                paired_evidence=paired_evidence,
                source=source,
                state="retryable",
                reason_codes=("search_skipped",),
                eligible_for_retry=True,
                record_a3m_members=record_members,
                extra_tar_members=extra_tar_members,
            )
        pristine_fallback_count = (
            len(record_lines) < expected_num_records
            and source.pristine_record_count is not None
            and len(record_lines) == source.pristine_record_count
        )
        record_member_inventory_matches = len(set(record_members)) == len(record_members) and set(
            record_members
        ) == set(expected_members)
        record_reasons: list[PreprocessingStateReason] = []
        if len(record_lines) != expected_num_records and not pristine_fallback_count:
            record_reasons.append("record_count_mismatch")
        if not record_member_inventory_matches:
            record_reasons.append("record_members_mismatch")
        if record_reasons:
            return _state(
                chunk=chunk,
                evidence_plan=evidence_plan,
                package_plan=package_plan,
                archive_evidence=archive_evidence,
                expected_a3ms=expected_a3ms,
                expected_num_records=expected_num_records,
                paired_evidence=paired_evidence,
                source=source,
                state="invalid",
                reason_codes=tuple(record_reasons),
                eligible_for_retry=True,
                record_a3m_members=record_members,
                extra_tar_members=extra_tar_members,
            )
        missing_tar_members = tuple(member for member in record_members if member not in tar_members)
        archive_reasons: list[PreprocessingStateReason] = []
        if archive_evidence.tar_size_bytes is None:
            archive_reasons.append("tar_missing")
        elif archive_evidence.tar_size_bytes == 0:
            archive_reasons.append("tar_empty")
        if archive_evidence.lz4_size_bytes is None:
            archive_reasons.append("tar_lz4_missing")
        elif archive_evidence.lz4_size_bytes == 0:
            archive_reasons.append("tar_lz4_empty")
        if archive_reasons:
            return _state(
                chunk=chunk,
                evidence_plan=evidence_plan,
                package_plan=package_plan,
                archive_evidence=archive_evidence,
                expected_a3ms=expected_a3ms,
                expected_num_records=expected_num_records,
                paired_evidence=paired_evidence,
                source=source,
                state="invalid",
                reason_codes=tuple(archive_reasons),
                eligible_for_retry=True,
                record_a3m_members=record_members,
                missing_tar_members=missing_tar_members,
                extra_tar_members=extra_tar_members,
            )
        if missing_tar_members:
            return _state(
                chunk=chunk,
                evidence_plan=evidence_plan,
                package_plan=package_plan,
                archive_evidence=archive_evidence,
                expected_a3ms=expected_a3ms,
                expected_num_records=expected_num_records,
                paired_evidence=paired_evidence,
                source=source,
                state="invalid",
                reason_codes=("tar_member_missing",),
                eligible_for_retry=True,
                record_a3m_members=record_members,
                missing_tar_members=missing_tar_members,
                extra_tar_members=extra_tar_members,
            )
        if len(record_lines) == expected_num_records and not record_first_line_identifies_afdb_or_pdb:
            return _state(
                chunk=chunk,
                evidence_plan=evidence_plan,
                package_plan=package_plan,
                archive_evidence=archive_evidence,
                expected_a3ms=expected_a3ms,
                expected_num_records=expected_num_records,
                paired_evidence=paired_evidence,
                source=source,
                state="invalid",
                reason_codes=("record_first_line_missing_afdb",),
                eligible_for_retry=True,
                record_a3m_members=record_members,
                extra_tar_members=extra_tar_members,
            )
        full_record_count = len(record_lines) == expected_num_records and record_first_line_identifies_afdb_or_pdb
        if source.split_present or not source.finished_input_present:
            return _state(
                chunk=chunk,
                evidence_plan=evidence_plan,
                package_plan=package_plan,
                archive_evidence=archive_evidence,
                expected_a3ms=expected_a3ms,
                expected_num_records=expected_num_records,
                paired_evidence=paired_evidence,
                source=source,
                state="invalid",
                reason_codes=("finished_source_missing",),
                eligible_for_retry=True,
                record_a3m_members=record_members,
                extra_tar_members=extra_tar_members,
            )
        if (
            source.pristine_present
            and not source.split_present
            and source.finished_input_present
            and (full_record_count or pristine_fallback_count)
            and record_member_inventory_matches
            and archive_evidence.tar_size_bytes is not None
            and archive_evidence.tar_size_bytes > 0
            and archive_evidence.lz4_size_bytes is not None
            and archive_evidence.lz4_size_bytes > 0
            and all(member in tar_members for member in record_members)
            and not any("Skipping" in line for line in log_lines)
        ):
            return _state(
                chunk=chunk,
                evidence_plan=evidence_plan,
                package_plan=package_plan,
                archive_evidence=archive_evidence,
                expected_a3ms=expected_a3ms,
                expected_num_records=expected_num_records,
                paired_evidence=paired_evidence,
                source=source,
                state="completed",
                reason_codes=("complete",),
                eligible_for_retry=False,
                record_a3m_members=record_members,
                extra_tar_members=extra_tar_members,
            )
    msg = "preprocessing state combination is not implemented"
    raise ValueError(msg)


def _validate_inputs(
    *,
    chunk: PreprocessingChunk,
    expected_a3ms: tuple[ExpectedA3M, ...],
    evidence_plan: PreprocessingEvidencePlan,
    package_plan: PreprocessingPackagePlan,
    expected_num_records: int,
    source: PreprocessingSourceMembership,
    paired_evidence: PreprocessingPairedEvidence,
    archive_evidence: PreprocessingArchiveEvidence,
) -> None:
    if not isinstance(expected_num_records, int) or isinstance(expected_num_records, bool) or expected_num_records <= 0:
        msg = "expected_num_records must be a positive explicit input"
        raise ValueError(msg)
    chunk_references = (
        evidence_plan.chunk_name,
        package_plan.chunk_name,
        source.chunk_name,
        paired_evidence.chunk_name,
        archive_evidence.chunk_name,
        *(expected.chunk_name for expected in expected_a3ms),
    )
    if not expected_a3ms or any(reference != chunk.name for reference in chunk_references):
        msg = "all state inputs must explicitly reference the planned chunk"
        raise ValueError(msg)
    if tuple(expected.member_name for expected in expected_a3ms) != package_plan.declared_stage_members:
        msg = "expected A3Ms must exactly match the declared package inventory"
        raise ValueError(msg)
    if tuple(expected.source_ordinal for expected in expected_a3ms) != chunk.record_ordinals:
        msg = "expected A3Ms must retain planned source-record association"
        raise ValueError(msg)
    if source.pristine_record_count is not None and source.pristine_record_count != len(chunk.record_ordinals):
        msg = "pristine_record_count must match the planned chunk records"
        raise ValueError(msg)
    if source.split_path != package_plan.completed_input_source_path:
        msg = "split membership must use the declared durable package source path"
        raise ValueError(msg)
    if source.finished_input_path != package_plan.completed_input_path:
        msg = "finished membership must use the declared durable completed-input path"
        raise ValueError(msg)
    if (
        paired_evidence.durable_record_path != evidence_plan.durable_record_path
        or paired_evidence.durable_log_path != evidence_plan.durable_log_path
    ):
        msg = "paired evidence must use the declared flat durable paths"
        raise ValueError(msg)
    if (
        archive_evidence.durable_tar_path != package_plan.durable_tar_path
        or archive_evidence.durable_lz4_path != package_plan.durable_lz4_path
    ):
        msg = "archive evidence must use the declared flat durable paths"
        raise ValueError(msg)


def _state(
    *,
    chunk: PreprocessingChunk,
    evidence_plan: PreprocessingEvidencePlan,
    package_plan: PreprocessingPackagePlan,
    archive_evidence: PreprocessingArchiveEvidence,
    expected_a3ms: tuple[ExpectedA3M, ...],
    expected_num_records: int,
    paired_evidence: PreprocessingPairedEvidence,
    source: PreprocessingSourceMembership,
    state: PreprocessingChunkStateKind,
    reason_codes: tuple[PreprocessingStateReason, ...],
    eligible_for_retry: bool,
    record_a3m_members: tuple[str, ...] | None = None,
    missing_tar_members: tuple[str, ...] = (),
    extra_tar_members: tuple[str, ...] | None = None,
) -> PreprocessingChunkState:
    record_lines = paired_evidence.record_lines
    log_lines = paired_evidence.log_lines
    record_line_count = None if record_lines is None else len(record_lines)
    log_line_count = None if log_lines is None else len(log_lines)
    parsed_record_members = _record_a3m_members(record_lines or ())
    effective_record_members = parsed_record_members if record_a3m_members is None else record_a3m_members
    normalized_tar_members = tuple(_normalize_tar_member(member) for member in (archive_evidence.tar_members or ()))
    derived_extra_tar_members = tuple(
        member for member in normalized_tar_members if member not in effective_record_members
    )
    record_reports_no_such_file = bool(record_lines and len(record_lines) == 1 and "No such file" in record_lines[0])
    record_entries_well_formed = None if not record_lines else len(parsed_record_members) == len(record_lines)
    record_first_line_contains_afdb = (
        None
        if not record_lines
        else ("AFDB" in record_lines[0] or any(is_pdb_assembly_member_name(member) for member in parsed_record_members))
    )
    retry_actions = derive_preprocessing_retry_actions(
        chunk_name=chunk.name,
        eligible_for_retry=eligible_for_retry,
        pristine_present=source.pristine_present,
        split_present=source.split_present,
        pristine_path=source.pristine_path,
        split_path=source.split_path,
        durable_tar_path=package_plan.durable_tar_path,
        durable_lz4_path=package_plan.durable_lz4_path,
        tar_present=archive_evidence.tar_size_bytes is not None,
        lz4_present=archive_evidence.lz4_size_bytes is not None,
        record_line_count=record_line_count,
        record_first_line_contains_afdb=record_first_line_contains_afdb,
        expected_num_records=expected_num_records,
    )
    return PreprocessingChunkState(
        chunk_name=chunk.name,
        tranche_name=chunk.tranche_name,
        chunk_ordinal=chunk.ordinal,
        state=state,
        reason_codes=reason_codes,
        eligible_for_retry=eligible_for_retry,
        pristine_present=source.pristine_present,
        split_present=source.split_present,
        finished_input_present=source.finished_input_present,
        pristine_path=source.pristine_path,
        split_path=source.split_path,
        finished_input_path=source.finished_input_path,
        shared_finished_read_present=source.shared_finished_read_present,
        shared_finished_write_present=source.shared_finished_write_present,
        shared_finished_read_tar_path=source.shared_finished_read_tar_path,
        shared_finished_write_tar_path=source.shared_finished_write_tar_path,
        durable_record_path=evidence_plan.durable_record_path,
        durable_log_path=evidence_plan.durable_log_path,
        durable_tar_path=package_plan.durable_tar_path,
        durable_lz4_path=package_plan.durable_lz4_path,
        tar_size_bytes=archive_evidence.tar_size_bytes,
        lz4_size_bytes=archive_evidence.lz4_size_bytes,
        expected_a3ms=expected_a3ms,
        expected_num_records=expected_num_records,
        pristine_record_count=source.pristine_record_count,
        record_line_count=record_line_count,
        record_entries_well_formed=record_entries_well_formed,
        record_reports_no_such_file=record_reports_no_such_file,
        record_first_line_contains_afdb=record_first_line_contains_afdb,
        log_line_count=log_line_count,
        log_reports_skipping=any("Skipping" in line for line in (log_lines or ())),
        record_a3m_members=effective_record_members,
        missing_tar_members=missing_tar_members,
        extra_tar_members=derived_extra_tar_members if extra_tar_members is None else extra_tar_members,
        shared_finished_paths_differ=(source.shared_finished_read_tar_path != source.shared_finished_write_tar_path),
        retry_actions=retry_actions,
    )


def parse_preprocessing_record_a3m_members(lines: tuple[str, ...]) -> tuple[str, ...]:
    """Parse A3M basenames from the pinned real ``ls -lh`` record format."""
    # The pinned baseline writes `.record` with `ls -lh <glob>`. Validate only
    # stable mode, link-count, and path columns; size and date fields are locale-dependent.
    members: list[str] = []
    for line in lines:
        fields = line.split(maxsplit=8)
        if len(fields) != 9 or re.fullmatch(r"-[rwxStTs-]{9}[.+@]?", fields[0]) is None or not fields[1].isdigit():
            continue
        candidate = fields[-1].rsplit("/", maxsplit=1)[-1].strip()
        if candidate.endswith(".a3m"):
            members.append(candidate)
    return tuple(members)


def _record_a3m_members(lines: tuple[str, ...]) -> tuple[str, ...]:
    return parse_preprocessing_record_a3m_members(lines)


def _normalize_tar_member(member: str) -> str:
    while member.startswith("./"):
        member = member[2:]
    return member or "."


__all__ = ["interpret_preprocessing_chunk_state", "parse_preprocessing_record_a3m_members"]
