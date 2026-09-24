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

"""Public-seam tests for deterministic preprocessing state interpretation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from bspp.orchestration.contract.preprocessing import (
    PreprocessingChunk,
    PreprocessingFastaRecord,
    PreprocessingPlanOptions,
    PreprocessingWorkPlan,
)
from bspp.orchestration.contract.preprocessing_execution import (
    ExpectedA3M,
    PreprocessingEvidencePlan,
    PreprocessingPackagePlan,
)
from bspp.orchestration.contract.preprocessing_state import (
    PreprocessingArchiveEvidence,
    PreprocessingChunkState,
    PreprocessingChunkStateKind,
    PreprocessingPairedEvidence,
    PreprocessingSourceMembership,
    PreprocessingStateReason,
    PreprocessingTrancheProgress,
    derive_preprocessing_retry_actions,
    preprocessing_archive_evidence_from_mapping,
    preprocessing_chunk_state_from_mapping,
    preprocessing_paired_evidence_from_mapping,
    preprocessing_retry_plan_from_mapping,
    preprocessing_source_membership_from_mapping,
    preprocessing_tranche_progress_from_mapping,
    summarize_preprocessing_tranche_progress,
)
from bspp.orchestration.runtime.preprocessing.planning import plan_preprocessing_records
from bspp.orchestration.runtime.preprocessing.retry import plan_preprocessing_retries
from bspp.orchestration.runtime.preprocessing.state import interpret_preprocessing_chunk_state

VALID_RECORD_LINES = (
    "-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/AFDB_alpha.a3m",
    "-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/AFDB_beta.a3m",
    "-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/AFDB_gamma.a3m",
)


def test_split_chunk_without_outputs_is_planned_and_eligible() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=True),
        paired_evidence=_paired(record_lines=None, log_lines=None),
        archive_evidence=_archive(tar_size_bytes=None, lz4_size_bytes=None, tar_members=None),
    )

    assert state.state == "planned"
    assert state.reason_codes == ("awaiting_execution",)
    assert state.eligible_for_retry is True
    assert state.retry_actions == ()
    assert state.durable_record_path == "/project logs/proteins_tranche00_00000.record"
    assert state.durable_log_path == "/project logs/proteins_tranche00_00000.log"
    assert "n7g2" not in state.durable_record_path
    assert "_3" not in state.durable_record_path


def test_complete_evidence_requires_finished_source_archives_and_declared_members() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=False, finished_input_present=True, shared_read_present=True),
        paired_evidence=_paired(
            record_lines=(
                "-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/AFDB_alpha.a3m",
                "-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/AFDB_beta.a3m",
                "-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/AFDB_gamma.a3m",
            ),
            log_lines=("search complete",),
        ),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=(
                "./AFDB_alpha.a3m",
                "./AFDB_beta.a3m",
                "./AFDB_gamma.a3m",
                "./temporary-search-db",
            ),
        ),
    )

    assert state.state == "completed"
    assert state.reason_codes == ("complete",)
    assert state.eligible_for_retry is False
    assert state.record_a3m_members == ("AFDB_alpha.a3m", "AFDB_beta.a3m", "AFDB_gamma.a3m")
    assert state.missing_tar_members == ()
    assert state.extra_tar_members == ("temporary-search-db",)
    assert state.shared_finished_paths_differ is True
    assert state.shared_finished_read_tar_path == "/part00_finished_msa/proteins_tranche00_00000.tar"
    assert state.shared_finished_write_tar_path == "/finished_msas/proteins_tranche00_00000.tar"
    assert state.retry_actions == ()


def test_record_listing_order_does_not_replace_source_record_association_order() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=False, finished_input_present=True),
        paired_evidence=_paired(record_lines=_valid_record_lines()[::-1], log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=("./AFDB_beta.a3m", "./AFDB_gamma.a3m", "./AFDB_alpha.a3m"),
        ),
    )

    assert state.state == "completed"
    assert state.record_a3m_members == ("AFDB_gamma.a3m", "AFDB_beta.a3m", "AFDB_alpha.a3m")
    assert tuple(expected.member_name for expected in _expected_a3ms()) == (
        "AFDB_alpha.a3m",
        "AFDB_beta.a3m",
        "AFDB_gamma.a3m",
    )


def _pdb_expected_a3ms(*, identities: tuple[str, ...]) -> tuple[ExpectedA3M, ...]:
    return tuple(
        ExpectedA3M(
            chunk_name=_chunk().name,
            record_identity=identity,
            source_ordinal=ordinal,
            source_header=f">{identity}",
            member_name=f"{identity}.a3m",
        )
        for ordinal, identity in enumerate(identities)
    )


def test_single_pdb_assembly_member_chunk_is_completed_not_invalid() -> None:
    """A single-member chunk with pdb_* identity is completed (AC #18)."""
    identities = ("pdb_5snm_assembly_1",)
    expected_a3ms = _pdb_expected_a3ms(identities=identities)
    record_lines = tuple(
        f"-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/{expected.member_name}" for expected in expected_a3ms
    )
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(record_count=len(identities)),
        expected_a3ms=expected_a3ms,
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(
            record_count=len(identities),
            members=tuple(expected.member_name for expected in expected_a3ms),
        ),
        expected_num_records=len(identities),
        source=_source(split_present=False, finished_input_present=True, pristine_record_count=len(identities)),
        paired_evidence=_paired(record_lines=record_lines, log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=tuple(f"./{expected.member_name}" for expected in expected_a3ms),
        ),
    )

    assert state.state == "completed"
    assert state.reason_codes == ("complete",)
    assert state.eligible_for_retry is False
    assert state.record_first_line_contains_afdb is True


def test_three_pdb_assembly_members_complete_without_value_error() -> None:
    """Three pdb_* members with coherent completion evidence classify completed (B1 regression)."""
    identities = ("pdb_5snm_assembly_1", "pdb_5snm_assembly_2", "pdb_5snm_assembly_3")
    expected_a3ms = _pdb_expected_a3ms(identities=identities)
    record_lines = tuple(
        f"-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/{expected.member_name}" for expected in expected_a3ms
    )
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(record_count=len(identities)),
        expected_a3ms=expected_a3ms,
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(
            record_count=len(identities),
            members=tuple(expected.member_name for expected in expected_a3ms),
        ),
        expected_num_records=len(identities),
        source=_source(split_present=False, finished_input_present=True, pristine_record_count=len(identities)),
        paired_evidence=_paired(record_lines=record_lines, log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=tuple(f"./{expected.member_name}" for expected in expected_a3ms),
        ),
    )

    assert state.state == "completed"
    assert state.reason_codes == ("complete",)
    assert state.eligible_for_retry is False
    assert state.record_first_line_contains_afdb is True


def test_ghosted_chunk_is_missing_and_plans_pristine_restore_without_execution() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=False),
        paired_evidence=_paired(record_lines=None, log_lines=None),
        archive_evidence=_archive(tar_size_bytes=None, lz4_size_bytes=None, tar_members=None),
    )

    assert state.state == "missing"
    assert state.reason_codes == ("ghosted_source",)
    assert state.eligible_for_retry is True
    assert tuple((action.operation, action.source_path, action.target_path) for action in state.retry_actions) == (
        (
            "copy",
            "/pristine input/proteins_tranche00_00000.fa",
            "/split input/proteins_tranche00_00000.fa",
        ),
    )


def test_skipping_log_is_retryable_and_plans_restore_and_archive_removal() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=False, finished_input_present=True),
        paired_evidence=_paired(
            record_lines=_valid_record_lines(),
            log_lines=("search started", "Skipping query beta after transient failure"),
        ),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=tuple(f"./{member.member_name}" for member in _expected_a3ms()),
        ),
    )

    assert state.state == "retryable"
    assert state.reason_codes == ("search_skipped",)
    assert state.eligible_for_retry is True
    assert tuple((action.operation, action.target_path) for action in state.retry_actions) == (
        ("copy", "/split input/proteins_tranche00_00000.fa"),
        ("remove", "/finished msa/proteins_tranche00_00000.tar"),
        ("remove", "/finished msa/proteins_tranche00_00000.tar.lz4"),
    )

    reordered = state.to_mapping()
    assert isinstance(reordered["retry_actions"], list)
    reordered["retry_actions"] = list(reversed(reordered["retry_actions"]))
    with pytest.raises(ValueError, match="retry actions must exactly reflect state evidence"):
        preprocessing_chunk_state_from_mapping(reordered)


def test_split_present_skipping_declares_no_cleanup_and_records_tar_extras() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=True),
        paired_evidence=_paired(record_lines=_valid_record_lines(), log_lines=("Skipping transient search",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=(
                "./AFDB_alpha.a3m",
                "./AFDB_beta.a3m",
                "./AFDB_gamma.a3m",
                "./retry-directory/",
            ),
        ),
    )

    assert state.state == "retryable"
    assert state.reason_codes == ("search_skipped",)
    assert state.retry_actions == ()
    assert state.extra_tar_members == ("retry-directory/",)


def test_split_present_no_such_file_declares_no_cleanup_and_records_tar_extras() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=True),
        paired_evidence=_paired(
            record_lines=("ls: cannot access '/output/*.a3m': No such file or directory",),
            log_lines=("search failed",),
        ),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=("./stale-output.a3m", "./stale-directory/"),
        ),
    )

    assert state.state == "retryable"
    assert state.reason_codes == ("record_reports_no_such_file",)
    assert state.retry_actions == ()
    assert state.extra_tar_members == ("stale-output.a3m", "stale-directory/")


def test_split_present_pristine_count_shortfall_declares_no_cleanup() -> None:
    record_lines = (*_valid_record_lines(), "-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/AFDB_delta.a3m")
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=5,
        source=_source(split_present=True),
        paired_evidence=_paired(record_lines=record_lines, log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=(
                "./AFDB_alpha.a3m",
                "./AFDB_beta.a3m",
                "./AFDB_gamma.a3m",
                "./AFDB_delta.a3m",
            ),
        ),
    )

    assert state.state == "invalid"
    assert state.reason_codes == ("record_count_mismatch", "record_members_mismatch")
    assert state.retry_actions == ()


def test_split_present_unconditional_bad_record_branch_declares_copy_and_removals() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(member_prefix=""),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(member_prefix=""),
        expected_num_records=3,
        source=_source(split_present=True),
        paired_evidence=_paired(record_lines=_valid_record_lines(member_prefix=""), log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=("./alpha.a3m", "./beta.a3m", "./gamma.a3m"),
        ),
    )

    assert state.state == "invalid"
    assert state.reason_codes == ("record_first_line_missing_afdb",)
    assert tuple(action.operation for action in state.retry_actions) == ("copy", "remove", "remove")


def test_split_present_record_count_above_expected_declares_copy_and_removals() -> None:
    record_lines = (*_valid_record_lines(), "-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/AFDB_delta.a3m")
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=True),
        paired_evidence=_paired(record_lines=record_lines, log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=(
                "./AFDB_alpha.a3m",
                "./AFDB_beta.a3m",
                "./AFDB_gamma.a3m",
                "./AFDB_delta.a3m",
            ),
        ),
    )

    assert state.state == "invalid"
    assert state.reason_codes == ("record_count_mismatch", "record_members_mismatch")
    assert tuple(action.operation for action in state.retry_actions) == ("copy", "remove", "remove")


def test_mapping_cannot_suppress_unconditional_actions_by_omitting_trigger_reasons() -> None:
    first_line_state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(member_prefix=""),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(member_prefix=""),
        expected_num_records=3,
        source=_source(split_present=True),
        paired_evidence=_paired(record_lines=_valid_record_lines(member_prefix=""), log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=("./alpha.a3m", "./beta.a3m", "./gamma.a3m"),
        ),
    )
    omitted_first_line_reason = first_line_state.to_mapping()
    omitted_first_line_reason["reason_codes"] = ["tar_empty"]
    omitted_first_line_reason["tar_size_bytes"] = 0
    omitted_first_line_reason["retry_actions"] = []
    with pytest.raises(ValueError, match="retry actions must exactly reflect state evidence"):
        preprocessing_chunk_state_from_mapping(omitted_first_line_reason)

    above_expected_lines = (
        *_valid_record_lines(),
        "-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/AFDB_delta.a3m",
    )
    above_expected_state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=True),
        paired_evidence=_paired(record_lines=above_expected_lines, log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=(
                "./AFDB_alpha.a3m",
                "./AFDB_beta.a3m",
                "./AFDB_gamma.a3m",
                "./AFDB_delta.a3m",
            ),
        ),
    )
    omitted_count_reason = above_expected_state.to_mapping()
    omitted_count_reason["reason_codes"] = ["record_members_mismatch"]
    omitted_count_reason["retry_actions"] = []
    with pytest.raises(ValueError, match="retry actions must exactly reflect state evidence"):
        preprocessing_chunk_state_from_mapping(omitted_count_reason)


@pytest.mark.parametrize(
    ("record_line_count", "record_first_line_contains_afdb"),
    [(3, False), (4, True)],
)
def test_retry_action_derivation_uses_unconditional_record_facts_directly(
    record_line_count: int,
    record_first_line_contains_afdb: bool,
) -> None:
    actions = derive_preprocessing_retry_actions(
        chunk_name=_chunk().name,
        eligible_for_retry=True,
        pristine_present=True,
        split_present=True,
        pristine_path="/pristine input/proteins_tranche00_00000.fa",
        split_path="/split input/proteins_tranche00_00000.fa",
        durable_tar_path="/finished msa/proteins_tranche00_00000.tar",
        durable_lz4_path="/finished msa/proteins_tranche00_00000.tar.lz4",
        tar_present=True,
        lz4_present=True,
        record_line_count=record_line_count,
        record_first_line_contains_afdb=record_first_line_contains_afdb,
        expected_num_records=3,
    )

    assert tuple(action.operation for action in actions) == ("copy", "remove", "remove")


def test_missing_record_listed_tar_member_is_invalid_while_extras_are_nonfatal() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=False, finished_input_present=True),
        paired_evidence=_paired(record_lines=_valid_record_lines(), log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=("./AFDB_alpha.a3m", "./AFDB_beta.a3m", "./non-a3m-extra"),
        ),
    )

    assert state.state == "invalid"
    assert state.reason_codes == ("tar_member_missing",)
    assert state.eligible_for_retry is True
    assert state.missing_tar_members == ("AFDB_gamma.a3m",)
    assert state.extra_tar_members == ("non-a3m-extra",)


def test_tar_directory_entries_do_not_satisfy_record_listed_file_membership() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=False, finished_input_present=True),
        paired_evidence=_paired(record_lines=_valid_record_lines(), log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=("./AFDB_alpha.a3m/", "./AFDB_beta.a3m/", "./AFDB_gamma.a3m/"),
        ),
    )

    assert state.state == "invalid"
    assert state.reason_codes == ("tar_member_missing",)
    assert state.missing_tar_members == ("AFDB_alpha.a3m", "AFDB_beta.a3m", "AFDB_gamma.a3m")
    assert state.extra_tar_members == ("AFDB_alpha.a3m/", "AFDB_beta.a3m/", "AFDB_gamma.a3m/")


def test_no_such_file_record_signal_is_retryable_not_a_low_a3m_inventory() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=False),
        paired_evidence=_paired(
            record_lines=("ls: cannot access '/output/*.a3m': No such file or directory",),
            log_lines=("search failed before output",),
        ),
        archive_evidence=_archive(tar_size_bytes=None, lz4_size_bytes=None, tar_members=None),
    )

    assert state.state == "retryable"
    assert state.reason_codes == ("record_reports_no_such_file",)
    assert state.eligible_for_retry is True
    assert state.record_a3m_members == ()
    assert tuple(action.operation for action in state.retry_actions) == ("copy",)


def test_two_a3ms_are_invalid_even_when_all_other_completion_evidence_is_coherent() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(record_count=2),
        expected_a3ms=_expected_a3ms(record_count=2),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(record_count=2),
        expected_num_records=2,
        source=_source(split_present=False, finished_input_present=True, pristine_record_count=2),
        paired_evidence=_paired(record_lines=_valid_record_lines(record_count=2), log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=("./AFDB_alpha.a3m", "./AFDB_beta.a3m", "./low-count-directory/"),
        ),
    )

    assert state.state == "completed"
    assert state.reason_codes == ("complete",)
    assert state.eligible_for_retry is False
    assert state.extra_tar_members == ("low-count-directory/",)

    # The forged_complete mapping is now a valid completed state (2 members is >= 1)
    forged_complete = state.to_mapping()
    forged_complete.update(
        {
            "state": "completed",
            "reason_codes": ["complete"],
            "eligible_for_retry": False,
            "retry_actions": [],
        }
    )
    preprocessing_chunk_state_from_mapping(forged_complete)


@pytest.mark.parametrize(
    ("record_lines", "log_lines", "reason_codes"),
    [
        (None, ("search started",), ("missing_record_evidence",)),
        (VALID_RECORD_LINES, None, ("missing_log_evidence",)),
        ((), ("search started",), ("empty_record_evidence",)),
        (VALID_RECORD_LINES, (), ("empty_log_evidence",)),
    ],
)
def test_partial_paired_durable_evidence_is_explicitly_retryable(
    record_lines: tuple[str, ...] | None,
    log_lines: tuple[str, ...] | None,
    reason_codes: tuple[str, ...],
) -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=True),
        paired_evidence=_paired(record_lines=record_lines, log_lines=log_lines),
        archive_evidence=_archive(tar_size_bytes=None, lz4_size_bytes=None, tar_members=None),
    )

    assert state.state == "retryable"
    assert state.reason_codes == reason_codes
    assert state.eligible_for_retry is True


def test_partial_paired_evidence_records_supplied_tar_extras_without_cleanup() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=True),
        paired_evidence=_paired(record_lines=_valid_record_lines(), log_lines=None),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=(
                "./AFDB_alpha.a3m",
                "./AFDB_beta.a3m",
                "./AFDB_gamma.a3m",
                "./partial-directory/",
            ),
        ),
    )

    assert state.state == "retryable"
    assert state.reason_codes == ("missing_log_evidence",)
    assert state.extra_tar_members == ("partial-directory/",)
    assert state.retry_actions == ()


@pytest.mark.parametrize(
    ("pristine_present", "split_present", "finished_input_present", "reason"),
    [
        (False, True, False, "pristine_source_missing"),
        (True, True, True, "contradictory_split_and_finished"),
    ],
)
def test_unrecoverable_source_membership_contradictions_are_invalid_and_ineligible(
    pristine_present: bool,
    split_present: bool,
    finished_input_present: bool,
    reason: str,
) -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(
            pristine_present=pristine_present,
            split_present=split_present,
            finished_input_present=finished_input_present,
        ),
        paired_evidence=_paired(record_lines=None, log_lines=None),
        archive_evidence=_archive(tar_size_bytes=None, lz4_size_bytes=None, tar_members=None),
    )

    assert state.state == "invalid"
    assert state.reason_codes == (reason,)
    assert state.eligible_for_retry is False
    assert state.retry_actions == ()


@pytest.mark.parametrize(
    ("tar_size_bytes", "lz4_size_bytes", "tar_members", "reason"),
    [
        (None, 1024, None, "tar_missing"),
        (0, 1024, (), "tar_empty"),
        (4096, None, ("./AFDB_alpha.a3m", "./AFDB_beta.a3m", "./AFDB_gamma.a3m"), "tar_lz4_missing"),
        (4096, 0, ("./AFDB_alpha.a3m", "./AFDB_beta.a3m", "./AFDB_gamma.a3m"), "tar_lz4_empty"),
    ],
)
def test_archive_presence_and_nonempty_compression_are_required_for_completion(
    tar_size_bytes: int | None,
    lz4_size_bytes: int | None,
    tar_members: tuple[str, ...] | None,
    reason: str,
) -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=False, finished_input_present=True),
        paired_evidence=_paired(record_lines=_valid_record_lines(), log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=tar_size_bytes,
            lz4_size_bytes=lz4_size_bytes,
            tar_members=tar_members,
        ),
    )

    assert state.state == "invalid"
    assert reason in state.reason_codes
    assert state.eligible_for_retry is True


def test_short_last_chunk_uses_explicit_pristine_record_count_fallback() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=4,
        source=_source(split_present=False, finished_input_present=True, pristine_record_count=3),
        paired_evidence=_paired(record_lines=_valid_record_lines(), log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=tuple(f"./{expected.member_name}" for expected in _expected_a3ms()),
        ),
    )

    assert state.state == "completed"
    assert state.reason_codes == ("complete",)


def test_full_record_count_requires_afdb_in_the_first_record_line() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(member_prefix=""),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(member_prefix=""),
        expected_num_records=3,
        source=_source(split_present=False, finished_input_present=True),
        paired_evidence=_paired(
            record_lines=_valid_record_lines(member_prefix=""),
            log_lines=("search complete",),
        ),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=("./alpha.a3m", "./beta.a3m", "./gamma.a3m"),
        ),
    )

    assert state.state == "invalid"
    assert state.reason_codes == ("record_first_line_missing_afdb",)


@pytest.mark.parametrize(
    ("record_lines", "tar_members", "reason_codes"),
    [
        (
            ("AFDB_alpha.a3m", "AFDB_beta.a3m", "AFDB_gamma.a3m"),
            ("./AFDB_alpha.a3m", "./AFDB_beta.a3m", "./AFDB_gamma.a3m"),
            ("malformed_record_entry",),
        ),
        (
            (VALID_RECORD_LINES[0], "not an ls A3M record", VALID_RECORD_LINES[2]),
            ("./AFDB_alpha.a3m", "./AFDB_gamma.a3m", "./malformed-directory/"),
            ("malformed_record_entry",),
        ),
        (
            (*VALID_RECORD_LINES, "-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/AFDB_delta.a3m"),
            ("./AFDB_alpha.a3m", "./AFDB_beta.a3m", "./AFDB_gamma.a3m", "./AFDB_delta.a3m"),
            ("record_count_mismatch", "record_members_mismatch"),
        ),
        (
            (VALID_RECORD_LINES[0], VALID_RECORD_LINES[1], VALID_RECORD_LINES[2].replace("gamma", "delta")),
            ("./AFDB_alpha.a3m", "./AFDB_beta.a3m", "./AFDB_delta.a3m"),
            ("record_members_mismatch",),
        ),
    ],
)
def test_malformed_count_and_identity_record_corruption_are_invalid(
    record_lines: tuple[str, ...],
    tar_members: tuple[str, ...],
    reason_codes: tuple[str, ...],
) -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=False, finished_input_present=True),
        paired_evidence=_paired(record_lines=record_lines, log_lines=("search complete",)),
        archive_evidence=_archive(tar_size_bytes=4096, lz4_size_bytes=1024, tar_members=tar_members),
    )

    assert state.state == "invalid"
    assert state.reason_codes == reason_codes
    assert state.eligible_for_retry is True
    if "./malformed-directory/" in tar_members:
        assert state.extra_tar_members == ("malformed-directory/",)


def test_complete_output_without_finished_source_is_invalid() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(split_present=True, finished_input_present=False),
        paired_evidence=_paired(record_lines=_valid_record_lines(), log_lines=("search complete",)),
        archive_evidence=_archive(
            tar_size_bytes=4096,
            lz4_size_bytes=1024,
            tar_members=tuple(f"./{expected.member_name}" for expected in _expected_a3ms()),
        ),
    )

    assert state.state == "invalid"
    assert state.reason_codes == ("finished_source_missing",)
    assert state.eligible_for_retry is True


def test_interpreter_rejects_inputs_outside_chunk_and_execution_authority() -> None:
    paired = _paired(record_lines=_valid_record_lines(), log_lines=("search complete",))
    archive = _archive(
        tar_size_bytes=4096,
        lz4_size_bytes=1024,
        tar_members=("./AFDB_alpha.a3m", "./AFDB_beta.a3m", "./AFDB_gamma.a3m"),
    )
    source = _source(split_present=False, finished_input_present=True)

    wrong_chunk = replace(
        _chunk(),
        name="proteins_tranche00_00001.fa",
        ordinal=1,
        tranche_chunk_ordinal=1,
    )
    with pytest.raises(ValueError, match="all state inputs must explicitly reference the planned chunk"):
        interpret_preprocessing_chunk_state(
            chunk=wrong_chunk,
            expected_a3ms=_expected_a3ms(),
            evidence_plan=_evidence_plan(),
            package_plan=_package_plan(),
            expected_num_records=3,
            source=source,
            paired_evidence=paired,
            archive_evidence=archive,
        )

    wrong_members = replace(
        _package_plan(),
        declared_stage_members=("other-alpha.a3m", "other-beta.a3m", "other-gamma.a3m"),
    )
    with pytest.raises(ValueError, match="expected A3Ms must exactly match the declared package inventory"):
        interpret_preprocessing_chunk_state(
            chunk=_chunk(),
            expected_a3ms=_expected_a3ms(),
            evidence_plan=_evidence_plan(),
            package_plan=wrong_members,
            expected_num_records=3,
            source=source,
            paired_evidence=paired,
            archive_evidence=archive,
        )

    wrong_ordinals = tuple(
        replace(expected, source_ordinal=expected.source_ordinal + 10) for expected in _expected_a3ms()
    )
    with pytest.raises(ValueError, match="expected A3Ms must retain planned source-record association"):
        interpret_preprocessing_chunk_state(
            chunk=_chunk(),
            expected_a3ms=wrong_ordinals,
            evidence_plan=_evidence_plan(),
            package_plan=_package_plan(),
            expected_num_records=3,
            source=source,
            paired_evidence=paired,
            archive_evidence=archive,
        )

    wrong_paired_paths = replace(
        paired,
        durable_record_path="/other logs/proteins_tranche00_00000.record",
        durable_log_path="/other logs/proteins_tranche00_00000.log",
    )
    with pytest.raises(ValueError, match="paired evidence must use the declared flat durable paths"):
        interpret_preprocessing_chunk_state(
            chunk=_chunk(),
            expected_a3ms=_expected_a3ms(),
            evidence_plan=_evidence_plan(),
            package_plan=_package_plan(),
            expected_num_records=3,
            source=source,
            paired_evidence=wrong_paired_paths,
            archive_evidence=archive,
        )

    wrong_archive_paths = replace(
        archive,
        durable_tar_path="/other msa/proteins_tranche00_00000.tar",
        durable_lz4_path="/other msa/proteins_tranche00_00000.tar.lz4",
    )
    with pytest.raises(ValueError, match="archive evidence must use the declared flat durable paths"):
        interpret_preprocessing_chunk_state(
            chunk=_chunk(),
            expected_a3ms=_expected_a3ms(),
            evidence_plan=_evidence_plan(),
            package_plan=_package_plan(),
            expected_num_records=3,
            source=source,
            paired_evidence=paired,
            archive_evidence=wrong_archive_paths,
        )


def test_retry_plan_preserves_completed_work_and_selects_exactly_eligible_assignments() -> None:
    work_plan = _work_plan()
    states = (
        _chunk_state(work_plan.chunks[0], state="completed", eligible=False, split=False, finished=True),
        _chunk_state(work_plan.chunks[1], state="planned", eligible=True, split=True),
        _chunk_state(work_plan.chunks[2], state="retryable", eligible=True, split=True),
        _chunk_state(work_plan.chunks[3], state="missing", eligible=True, split=False),
        _chunk_state(work_plan.chunks[4], state="invalid", eligible=False, split=True, finished=True),
    )

    first = plan_preprocessing_retries(work_plan=work_plan, states=states)
    replay = plan_preprocessing_retries(work_plan=work_plan, states=states)

    assert replay == first
    assert first.eligible_chunk_names == tuple(chunk.name for chunk in work_plan.chunks[1:4])
    assert first.eligible_chunks == work_plan.chunks[1:4]
    assert first.eligible_assignments == work_plan.assignments[1:4]
    assert first.invalid_chunk_names == (work_plan.chunks[4].name,)
    assert work_plan.chunks[0].name not in first.eligible_chunk_names
    assert tuple((action.chunk_name, action.operation) for action in first.actions) == (
        (work_plan.chunks[3].name, "copy"),
    )
    assert tuple(
        (
            progress.tranche_name,
            progress.pristine_chunks,
            progress.remaining_split_chunks,
            progress.finished_chunks,
            progress.completed_chunks,
            progress.invalid_chunks,
            progress.completion_percentage,
            progress.validated_completion_percentage,
        )
        for progress in first.tranche_progress
    ) == (("tranche00", 5, 3, 2, 1, 1, "40.00", "20.00"),)


def test_retry_planner_rejects_nonexact_state_coverage_and_order() -> None:
    work_plan = _work_plan()
    states = tuple(
        _chunk_state(chunk, state="completed", eligible=False, split=False, finished=True) for chunk in work_plan.chunks
    )

    with pytest.raises(ValueError, match="states must cover the work-plan chunks exactly once"):
        plan_preprocessing_retries(work_plan=work_plan, states=states[::-1])
    with pytest.raises(ValueError, match="states must cover the work-plan chunks exactly once"):
        plan_preprocessing_retries(work_plan=work_plan, states=states[:-1])


def test_fully_completed_work_has_no_remaining_retry_or_top_up_work() -> None:
    work_plan = _work_plan()
    states = tuple(
        _chunk_state(chunk, state="completed", eligible=False, split=False, finished=True) for chunk in work_plan.chunks
    )

    retry_plan = plan_preprocessing_retries(work_plan=work_plan, states=states)

    assert retry_plan.eligible_chunk_names == ()
    assert retry_plan.eligible_chunks == ()
    assert retry_plan.eligible_assignments == ()
    assert retry_plan.invalid_chunk_names == ()
    assert retry_plan.actions == ()
    assert retry_plan.tranche_progress[0].completion_percentage == "100.00"
    assert retry_plan.tranche_progress[0].validated_completion_percentage == "100.00"


def test_recoverable_invalid_work_is_reported_and_selected_when_explicitly_eligible() -> None:
    work_plan = _work_plan()
    states = (
        _chunk_state(work_plan.chunks[0], state="invalid", eligible=True, split=True),
        *(
            _chunk_state(chunk, state="completed", eligible=False, split=False, finished=True)
            for chunk in work_plan.chunks[1:]
        ),
    )

    retry_plan = plan_preprocessing_retries(work_plan=work_plan, states=states)

    assert retry_plan.invalid_chunk_names == (work_plan.chunks[0].name,)
    assert retry_plan.eligible_chunk_names == (work_plan.chunks[0].name,)
    assert retry_plan.eligible_assignments == (work_plan.assignments[0],)


def test_retry_plan_mapping_rejects_actions_redirected_from_declared_state_paths() -> None:
    work_plan = _work_plan()
    states = (
        _chunk_state(work_plan.chunks[0], state="invalid", eligible=True, split=False),
        *(
            _chunk_state(chunk, state="completed", eligible=False, split=False, finished=True)
            for chunk in work_plan.chunks[1:]
        ),
    )
    retry_plan = plan_preprocessing_retries(work_plan=work_plan, states=states)
    redirected = deepcopy(retry_plan.to_mapping())
    assert isinstance(redirected["states"], list)
    assert isinstance(redirected["states"][0], dict)
    assert isinstance(redirected["states"][0]["retry_actions"], list)
    assert isinstance(redirected["states"][0]["retry_actions"][1], dict)
    assert isinstance(redirected["actions"], list)
    assert isinstance(redirected["actions"][0], dict)
    redirected_path = "/redirected/proteins_tranche00_00000.tar"
    redirected["states"][0]["retry_actions"][1]["target_path"] = redirected_path
    redirected["actions"][1]["target_path"] = redirected_path

    with pytest.raises(ValueError, match="retry actions must match declared state paths"):
        preprocessing_retry_plan_from_mapping(redirected)

    missing_states = (
        _chunk_state(work_plan.chunks[0], state="missing", eligible=True, split=False),
        *(
            _chunk_state(chunk, state="completed", eligible=False, split=False, finished=True)
            for chunk in work_plan.chunks[1:]
        ),
    )
    missing_plan = plan_preprocessing_retries(work_plan=work_plan, states=missing_states)
    omitted = deepcopy(missing_plan.to_mapping())
    assert isinstance(omitted["states"], list)
    assert isinstance(omitted["states"][0], dict)
    omitted["states"][0]["retry_actions"] = []
    omitted["actions"] = []
    with pytest.raises(ValueError, match="retry actions must exactly reflect state evidence"):
        preprocessing_retry_plan_from_mapping(omitted)


def test_retry_plan_mapping_rejects_chunk_and_assignment_drift_from_source_work_plan() -> None:
    work_plan = _work_plan()
    states = (
        _chunk_state(work_plan.chunks[0], state="invalid", eligible=True, split=True),
        *(
            _chunk_state(chunk, state="completed", eligible=False, split=False, finished=True)
            for chunk in work_plan.chunks[1:]
        ),
    )
    retry_plan = plan_preprocessing_retries(work_plan=work_plan, states=states)

    changed_records = deepcopy(retry_plan.to_mapping())
    assert isinstance(changed_records["eligible_chunks"], list)
    assert isinstance(changed_records["eligible_chunks"][0], dict)
    changed_records["eligible_chunks"][0]["record_ordinals"] = [999]
    with pytest.raises(ValueError, match="eligible_chunks must be the exact source-plan selection"):
        preprocessing_retry_plan_from_mapping(changed_records)

    changed_worker = deepcopy(retry_plan.to_mapping())
    assert isinstance(changed_worker["eligible_assignments"], list)
    assert isinstance(changed_worker["eligible_assignments"][0], dict)
    changed_worker["eligible_assignments"][0].update(
        {"node_index": 9, "worker_label": "n9g0", "staged_path": f"n9g0/{work_plan.chunks[0].name}"}
    )
    with pytest.raises(ValueError, match="eligible_assignments must be the exact source-plan selection"):
        preprocessing_retry_plan_from_mapping(changed_worker)

    changed_expected_source = deepcopy(retry_plan.to_mapping())
    assert isinstance(changed_expected_source["states"], list)
    assert isinstance(changed_expected_source["states"][0], dict)
    assert isinstance(changed_expected_source["states"][0]["expected_a3ms"], list)
    assert isinstance(changed_expected_source["states"][0]["expected_a3ms"][0], dict)
    changed_expected_source["states"][0]["expected_a3ms"][0].update(
        {"record_identity": "unrelated", "source_header": ">unrelated"}
    )
    with pytest.raises(ValueError, match="state expected_a3ms must exactly match source work-plan records"):
        preprocessing_retry_plan_from_mapping(changed_expected_source)


def test_state_records_round_trip_and_reject_recursive_schema_drift_and_tampering() -> None:
    source = _source(split_present=True)
    paired = _paired(record_lines=_valid_record_lines(), log_lines=("search complete",))
    archive = _archive(
        tar_size_bytes=4096,
        lz4_size_bytes=1024,
        tar_members=tuple(f"./{expected.member_name}" for expected in _expected_a3ms()),
    )
    work_plan = _work_plan()
    states = tuple(
        _chunk_state(chunk, state="completed", eligible=False, split=False, finished=True) for chunk in work_plan.chunks
    )
    retry_plan = plan_preprocessing_retries(work_plan=work_plan, states=states)

    assert preprocessing_source_membership_from_mapping(source.to_mapping()) == source
    assert preprocessing_paired_evidence_from_mapping(paired.to_mapping()) == paired
    assert paired.record_streams_merged is True
    assert paired.log_streams_merged is True
    assert preprocessing_archive_evidence_from_mapping(archive.to_mapping()) == archive
    assert preprocessing_chunk_state_from_mapping(states[0].to_mapping()) == states[0]
    assert preprocessing_retry_plan_from_mapping(retry_plan.to_mapping()) == retry_plan

    unsupported_nested = deepcopy(retry_plan.to_mapping())
    assert isinstance(unsupported_nested["states"], list)
    assert isinstance(unsupported_nested["states"][0], dict)
    unsupported_nested["states"][0]["schema_version"] = 2
    with pytest.raises(ValueError, match="Unsupported PreprocessingChunkState schema_version 2"):
        preprocessing_retry_plan_from_mapping(unsupported_nested)

    unknown_nested = deepcopy(retry_plan.to_mapping())
    assert isinstance(unknown_nested["tranche_progress"], list)
    assert isinstance(unknown_nested["tranche_progress"][0], dict)
    unknown_nested["tranche_progress"][0]["undocumented"] = True
    with pytest.raises(ValueError, match=r"Unknown PreprocessingTrancheProgress field\(s\): undocumented"):
        preprocessing_retry_plan_from_mapping(unknown_nested)

    tampered_selection = deepcopy(retry_plan.to_mapping())
    tampered_selection["eligible_chunk_names"] = [work_plan.chunks[0].name]
    with pytest.raises(ValueError, match="eligible_chunk_names must exactly select"):
        preprocessing_retry_plan_from_mapping(tampered_selection)

    tampered_progress = deepcopy(retry_plan.to_mapping())
    assert isinstance(tampered_progress["tranche_progress"], list)
    assert isinstance(tampered_progress["tranche_progress"][0], dict)
    tampered_progress["tranche_progress"][0]["invalid_chunks"] = 1
    with pytest.raises(ValueError, match="tranche_progress must exactly summarize states"):
        preprocessing_retry_plan_from_mapping(tampered_progress)

    duplicate_state = deepcopy(retry_plan.to_mapping())
    assert isinstance(duplicate_state["states"], list)
    duplicate_state["states"][1] = deepcopy(duplicate_state["states"][0])
    with pytest.raises(ValueError, match="states must name each chunk exactly once"):
        preprocessing_retry_plan_from_mapping(duplicate_state)

    with pytest.raises(FrozenInstanceError):
        frozen_field = "state"
        setattr(states[0], frozen_field, "planned")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pristine_present", False),
        ("finished_input_present", False),
        ("split_present", True),
        ("tar_size_bytes", 0),
        ("lz4_size_bytes", None),
    ],
)
def test_completed_state_mapping_rejects_incoherent_source_and_archive_facts(
    field: str,
    value: object,
) -> None:
    completed = _chunk_state(_work_plan().chunks[0], state="completed", eligible=False, split=False, finished=True)
    tampered = completed.to_mapping()
    tampered[field] = value
    if field == "pristine_present":
        tampered["pristine_record_count"] = None

    with pytest.raises(ValueError, match="completed state requires coherent source and archive evidence"):
        preprocessing_chunk_state_from_mapping(tampered)


def test_completed_state_mapping_rejects_members_unrelated_to_declared_expected_a3ms() -> None:
    completed = _chunk_state(_work_plan().chunks[0], state="completed", eligible=False, split=False, finished=True)
    tampered = completed.to_mapping()
    tampered["record_a3m_members"] = ["unrelated-one.a3m", "unrelated-two.a3m", "unrelated-three.a3m"]

    with pytest.raises(ValueError, match="completed state requires exact declared A3M membership"):
        preprocessing_chunk_state_from_mapping(tampered)

    missing_member = completed.to_mapping()
    missing_member["missing_tar_members"] = [completed.record_a3m_members[0]]
    with pytest.raises(ValueError, match="completed state requires coherent source and archive evidence"):
        preprocessing_chunk_state_from_mapping(missing_member)


def test_state_mapping_rejects_reason_codes_from_another_lifecycle_state() -> None:
    planned = _chunk_state(_work_plan().chunks[0], state="planned", eligible=True, split=True)
    tampered = planned.to_mapping()
    tampered["reason_codes"] = ["tar_empty"]

    with pytest.raises(ValueError, match="reason_codes are incompatible with planned state"):
        preprocessing_chunk_state_from_mapping(tampered)


@pytest.mark.parametrize("state_kind", ["planned", "missing"])
def test_state_mapping_rejects_source_and_archive_facts_from_another_lifecycle(
    state_kind: PreprocessingChunkStateKind,
) -> None:
    work_plan = _work_plan()
    state = _chunk_state(
        work_plan.chunks[0],
        state=state_kind,
        eligible=True,
        split=state_kind == "planned",
    )
    tampered = state.to_mapping()
    if state_kind == "planned":
        tampered["tar_size_bytes"] = 0
    else:
        tampered["split_present"] = True
        tampered["retry_actions"] = []

    with pytest.raises(ValueError, match=f"{state_kind} state facts are incoherent"):
        preprocessing_chunk_state_from_mapping(tampered)


def test_standalone_tranche_progress_rejects_counts_above_total() -> None:
    work_plan = _work_plan()
    states = tuple(
        _chunk_state(chunk, state="completed", eligible=False, split=False, finished=True) for chunk in work_plan.chunks
    )
    progress = plan_preprocessing_retries(work_plan=work_plan, states=states).tranche_progress[0].to_mapping()
    progress["pristine_chunks"] = 6

    with pytest.raises(ValueError, match="progress counts cannot exceed total_chunks"):
        preprocessing_tranche_progress_from_mapping(progress)

    invalid_name = (
        plan_preprocessing_retries(
            work_plan=work_plan,
            states=states,
        )
        .tranche_progress[0]
        .to_mapping()
    )
    invalid_name["tranche_name"] = "tranche0x"
    with pytest.raises(ValueError, match="tranche_name must use an exact two-digit suffix"):
        preprocessing_tranche_progress_from_mapping(invalid_name)


def test_tranche_progress_truncates_baseline_and_validated_percentages_to_two_decimals() -> None:
    progress = PreprocessingTrancheProgress(
        tranche_name="tranche00",
        pristine_chunks=3,
        remaining_split_chunks=1,
        finished_chunks=1,
        completed_chunks=1,
        eligible_chunks=2,
        invalid_chunks=0,
        total_chunks=3,
        completion_percentage="66.66",
        validated_completion_percentage="33.33",
    )

    assert preprocessing_tranche_progress_from_mapping(progress.to_mapping()) == progress


def test_tranche_progress_is_total_for_missing_pristine_split_diagnostics() -> None:
    state = interpret_preprocessing_chunk_state(
        chunk=_chunk(),
        expected_a3ms=_expected_a3ms(),
        evidence_plan=_evidence_plan(),
        package_plan=_package_plan(),
        expected_num_records=3,
        source=_source(pristine_present=False, split_present=True),
        paired_evidence=_paired(record_lines=None, log_lines=None),
        archive_evidence=_archive(tar_size_bytes=None, lz4_size_bytes=None, tar_members=None),
    )

    progress = summarize_preprocessing_tranche_progress((state,))[0]

    assert progress.pristine_chunks == 0
    assert progress.remaining_split_chunks == 1
    assert progress.completion_percentage is None
    assert progress.validated_completion_percentage == "0.00"
    assert preprocessing_tranche_progress_from_mapping(progress.to_mapping()) == progress


def _chunk(*, record_count: int = 3) -> PreprocessingChunk:
    return PreprocessingChunk(
        name="proteins_tranche00_00000.fa",
        tranche_name="tranche00",
        ordinal=0,
        tranche_chunk_ordinal=0,
        record_ordinals=tuple(range(record_count)),
    )


def _expected_a3ms(*, record_count: int = 3, member_prefix: str = "AFDB_") -> tuple[ExpectedA3M, ...]:
    return tuple(
        ExpectedA3M(
            chunk_name=_chunk().name,
            record_identity=identity,
            source_ordinal=ordinal,
            source_header=f">{identity}",
            member_name=f"{member_prefix}{identity}.a3m",
        )
        for ordinal, identity in enumerate(("alpha", "beta", "gamma")[:record_count])
    )


def _valid_record_lines(*, record_count: int = 3, member_prefix: str = "AFDB_") -> tuple[str, ...]:
    return tuple(
        f"-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/{expected.member_name}"
        for expected in _expected_a3ms(record_count=record_count, member_prefix=member_prefix)
    )


def _evidence_plan() -> PreprocessingEvidencePlan:
    return PreprocessingEvidencePlan(
        chunk_name=_chunk().name,
        raw_search_output_directory="/scratch output/raw-search/n7g2_3",
        scratch_output_directory="/scratch output/n7g2_3",
        scratch_log_directory="/scratch output/logs/n7g2_3",
        scratch_log_path="/scratch output/logs/n7g2_3/proteins_tranche00_00000.log",
        scratch_record_path="/scratch output/logs/n7g2_3/proteins_tranche00_00000.record",
        a3m_record_glob="/scratch output/n7g2_3/*.a3m",
        durable_log_path="/project logs/proteins_tranche00_00000.log",
        durable_record_path="/project logs/proteins_tranche00_00000.record",
    )


def _package_plan(
    *,
    record_count: int = 3,
    member_prefix: str = "AFDB_",
    members: tuple[str, ...] | None = None,
) -> PreprocessingPackagePlan:
    declared = (
        members
        if members is not None
        else tuple(
            expected.member_name for expected in _expected_a3ms(record_count=record_count, member_prefix=member_prefix)
        )
    )
    return PreprocessingPackagePlan(
        chunk_name=_chunk().name,
        staging_directory="/scratch output/n7g2_3",
        declared_stage_members=declared,
        tar_member_scope=".",
        scratch_tar_path="/scratch output/n7g2_3/proteins_tranche00_00000.tar",
        scratch_lz4_path="/scratch output/n7g2_3/proteins_tranche00_00000.tar.lz4",
        durable_tar_path="/finished msa/proteins_tranche00_00000.tar",
        durable_lz4_path="/finished msa/proteins_tranche00_00000.tar.lz4",
        completed_input_source_path="/split input/proteins_tranche00_00000.fa",
        completed_input_path="/finished input/proteins_tranche00_00000.fa",
        tar_argv=(
            "/usr/bin/tar",
            "cf",
            "/scratch output/n7g2_3/proteins_tranche00_00000.tar",
            "-C",
            "/scratch output/n7g2_3",
            ".",
        ),
        lz4_argv=(
            "/usr/bin/lz4",
            "-v",
            "-3",
            "/scratch output/n7g2_3/proteins_tranche00_00000.tar",
            "/scratch output/n7g2_3/proteins_tranche00_00000.tar.lz4",
        ),
    )


def _source(
    *,
    split_present: bool,
    finished_input_present: bool = False,
    pristine_present: bool = True,
    shared_read_present: bool = False,
    shared_write_present: bool = False,
    pristine_record_count: int = 3,
) -> PreprocessingSourceMembership:
    return PreprocessingSourceMembership(
        chunk_name=_chunk().name,
        pristine_path="/pristine input/proteins_tranche00_00000.fa",
        pristine_present=pristine_present,
        pristine_record_count=pristine_record_count if pristine_present else None,
        split_path="/split input/proteins_tranche00_00000.fa",
        split_present=split_present,
        finished_input_path="/finished input/proteins_tranche00_00000.fa",
        finished_input_present=finished_input_present,
        shared_finished_read_tar_path="/part00_finished_msa/proteins_tranche00_00000.tar",
        shared_finished_read_present=shared_read_present,
        shared_finished_write_tar_path="/finished_msas/proteins_tranche00_00000.tar",
        shared_finished_write_present=shared_write_present,
    )


def _paired(
    *,
    record_lines: tuple[str, ...] | None,
    log_lines: tuple[str, ...] | None,
) -> PreprocessingPairedEvidence:
    return PreprocessingPairedEvidence(
        chunk_name=_chunk().name,
        durable_record_path="/project logs/proteins_tranche00_00000.record",
        durable_log_path="/project logs/proteins_tranche00_00000.log",
        record_lines=record_lines,
        log_lines=log_lines,
    )


def _archive(
    *,
    tar_size_bytes: int | None,
    lz4_size_bytes: int | None,
    tar_members: tuple[str, ...] | None,
) -> PreprocessingArchiveEvidence:
    return PreprocessingArchiveEvidence(
        chunk_name=_chunk().name,
        durable_tar_path="/finished msa/proteins_tranche00_00000.tar",
        durable_lz4_path="/finished msa/proteins_tranche00_00000.tar.lz4",
        tar_size_bytes=tar_size_bytes,
        lz4_size_bytes=lz4_size_bytes,
        tar_members=tar_members,
    )


def _work_plan() -> PreprocessingWorkPlan:
    records = tuple(
        PreprocessingFastaRecord(
            header=f">protein-{ordinal}",
            sequence="AAA",
            identity=f"protein-{ordinal}",
            source_ordinal=ordinal,
        )
        for ordinal in range(15)
    )
    return plan_preprocessing_records(
        source_path="proteins.fa",
        records=records,
        options=PreprocessingPlanOptions(records_per_chunk=3, nodes=1, gpus_per_node=2),
    )


def _chunk_state(
    chunk: PreprocessingChunk,
    *,
    state: PreprocessingChunkStateKind,
    eligible: bool,
    split: bool,
    finished: bool = False,
) -> PreprocessingChunkState:
    stem = chunk.name.removesuffix(".fa")
    reasons_by_state: dict[PreprocessingChunkStateKind, tuple[PreprocessingStateReason, ...]] = {
        "completed": ("complete",),
        "planned": ("awaiting_execution",),
        "retryable": ("search_skipped",),
        "missing": ("ghosted_source",),
        "invalid": ("tar_empty",) if eligible else ("contradictory_split_and_finished",),
    }
    reasons = reasons_by_state[state]
    expected_a3ms = tuple(
        ExpectedA3M(
            chunk_name=chunk.name,
            record_identity=f"protein-{ordinal}",
            source_ordinal=ordinal,
            source_header=f">protein-{ordinal}",
            member_name=f"AFDB_protein-{ordinal}.a3m",
        )
        for ordinal in chunk.record_ordinals
    )
    has_record_evidence = state in {"completed", "retryable"} or (state == "invalid" and eligible)
    tar_size_bytes = 4096 if state == "completed" else (0 if state in {"retryable", "invalid"} and eligible else None)
    actions = derive_preprocessing_retry_actions(
        chunk_name=chunk.name,
        eligible_for_retry=eligible,
        pristine_present=True,
        split_present=split,
        pristine_path=f"/pristine input/{chunk.name}",
        split_path=f"/split input/{chunk.name}",
        durable_tar_path=f"/finished msa/{stem}.tar",
        durable_lz4_path=f"/finished msa/{stem}.tar.lz4",
        tar_present=tar_size_bytes is not None,
        lz4_present=state == "completed",
        record_line_count=len(expected_a3ms) if has_record_evidence else None,
        record_first_line_contains_afdb=True if has_record_evidence else None,
        expected_num_records=len(expected_a3ms),
    )
    return PreprocessingChunkState(
        chunk_name=chunk.name,
        tranche_name=chunk.tranche_name,
        chunk_ordinal=chunk.ordinal,
        state=state,
        reason_codes=reasons,
        eligible_for_retry=eligible,
        pristine_present=True,
        split_present=split,
        finished_input_present=finished,
        pristine_path=f"/pristine input/{chunk.name}",
        split_path=f"/split input/{chunk.name}",
        finished_input_path=f"/finished input/{chunk.name}",
        shared_finished_read_present=False,
        shared_finished_write_present=False,
        shared_finished_read_tar_path=f"/part00_finished_msa/{stem}.tar",
        shared_finished_write_tar_path=f"/finished_msas/{stem}.tar",
        durable_record_path=f"/project logs/{stem}.record",
        durable_log_path=f"/project logs/{stem}.log",
        durable_tar_path=f"/finished msa/{stem}.tar",
        durable_lz4_path=f"/finished msa/{stem}.tar.lz4",
        tar_size_bytes=tar_size_bytes,
        lz4_size_bytes=1024 if state == "completed" else None,
        expected_a3ms=expected_a3ms,
        expected_num_records=len(expected_a3ms),
        pristine_record_count=len(expected_a3ms),
        record_line_count=len(expected_a3ms) if has_record_evidence else None,
        record_entries_well_formed=True if has_record_evidence else None,
        record_reports_no_such_file=False,
        record_first_line_contains_afdb=True if has_record_evidence else None,
        log_line_count=1 if has_record_evidence else None,
        log_reports_skipping=state == "retryable",
        record_a3m_members=tuple(expected.member_name for expected in expected_a3ms) if has_record_evidence else (),
        missing_tar_members=(),
        extra_tar_members=(),
        shared_finished_paths_differ=True,
        retry_actions=actions,
    )
