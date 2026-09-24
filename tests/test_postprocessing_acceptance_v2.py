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

"""V2 postprocessing residual and comparator-coherence tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from bspp.orchestration.contract.postprocessing_acceptance import (
    PostprocessingAcceptancePolicySnapshot,
    PostprocessingBaselineReportBinding,
    PostprocessingCompletionExitContract,
    PostprocessingCrossReportReconciliation,
    PostprocessingRawExitReportOutcome,
    PostprocessingResidualAllowance,
    evaluate_postprocessing_acceptance_reports,
    project_acceptance_evidence_issues,
)

PARITY = "acceptance/tar_payload_parity/tar_payload_parity_report.json"
SEMANTIC = "acceptance/semantic_acceptance/semantic_acceptance_summary.json"
VERIFY = "acceptance/verify_evidence/acceptance_evidence_report.json"
PAYLOAD_MEMBERS = (
    "AF-0000000210662999-model_v1.bcif.zst",
    "AF-0000000205043519-confidence_v1.json.zst",
    "AF-0000000205043519-model_v1.bcif.zst",
    "metadata/clashes_and_interfaces_granular/AF-0000000205034195-model_v1_clashes.json.zst",
)
EXTRA_MEMBER = "metadata/clashes_and_interfaces_granular/AF-0000000210662999-model_v1_interface.json.zst"
SEMANTIC_RESIDUAL = "local_tars.csv semantic rows differ"


@pytest.mark.parametrize(
    ("payload_members", "extra_members", "semantic_errors"),
    (
        ((), (), ()),
        (PAYLOAD_MEMBERS[:3], (), ()),
        (PAYLOAD_MEMBERS[3:], (EXTRA_MEMBER,), (SEMANTIC_RESIDUAL,)),
    ),
    ids=("perfect", "current", "prior-r2"),
)
def test_v2_accepts_perfect_current_and_prior_task853_forms(
    payload_members: tuple[str, ...],
    extra_members: tuple[str, ...],
    semantic_errors: tuple[str, ...],
) -> None:
    evaluation = evaluate_postprocessing_acceptance_reports(
        _policy(),
        _reports(
            payload_members=payload_members,
            extra_members=extra_members,
            semantic_errors=semantic_errors,
        ),
    )

    assert evaluation.unallowlisted_occurrences == frozenset()
    assert all(item.accepted for item in evaluation.residual_cardinalities)
    assert all(item.matched for item in evaluation.reconciliation_results)


def test_v2_rejects_outsiders_duplicates_and_malformed_or_other_items() -> None:
    outsider = _reports(payload_members=("not-allowlisted.zst",))
    outsider_evaluation = evaluate_postprocessing_acceptance_reports(_policy(), outsider)
    assert len(outsider_evaluation.unallowlisted_occurrences) == 1

    duplicate = _reports(payload_members=(PAYLOAD_MEMBERS[0], PAYLOAD_MEMBERS[0]))
    duplicate_evaluation = evaluate_postprocessing_acceptance_reports(_policy(), duplicate)
    duplicated_allowance = next(
        item for item in duplicate_evaluation.residual_cardinalities if PAYLOAD_MEMBERS[0] in item.allowance_id
    )
    assert duplicated_allowance.observed == 2
    assert duplicated_allowance.accepted is False

    malformed = _reports()
    malformed[PARITY]["files"][0]["payload_mismatch_sample"] = [123]
    malformed[PARITY]["files"][0]["duplicate_candidate_normalized_names"] = ["other.zst"]
    malformed_evaluation = evaluate_postprocessing_acceptance_reports(_policy(), malformed)
    assert len(malformed_evaluation.unallowlisted_occurrences) >= 2


def test_v2_rejects_truncated_allowlistable_difference_arrays() -> None:
    reports = _reports(payload_members=(PAYLOAD_MEMBERS[0],))
    reports[PARITY]["files"][0]["payload_mismatch_count"] = 2
    reports[PARITY]["payload_mismatch_count"] = 2
    reports[VERIFY]["issues"] = list(
        project_acceptance_evidence_issues(
            parity_report=reports[PARITY],
            parity_report_path=PARITY,
            semantic_report=reports[SEMANTIC],
            semantic_report_path=SEMANTIC,
        )
    )

    evaluation = evaluate_postprocessing_acceptance_reports(_policy(), reports)

    assert evaluation.unallowlisted_occurrences
    assert any("payload_mismatch_count" in pointer for _, pointer, _ in evaluation.unallowlisted_occurrences)


def test_v2_rejects_inconsistent_member_counts() -> None:
    reports = _reports(extra_members=(EXTRA_MEMBER,))
    reports[PARITY]["files"][0]["candidate_member_count"] = 10

    evaluation = evaluate_postprocessing_acceptance_reports(_policy(), reports)

    assert evaluation.unallowlisted_occurrences
    assert any("candidate_member_count" in pointer for _, pointer, _ in evaluation.unallowlisted_occurrences)


def test_v2_rejects_possibly_truncated_inventory_difference_array() -> None:
    reports = _reports(extra_members=(EXTRA_MEMBER,))
    reports[PARITY]["sample_limit"] = 1

    evaluation = evaluate_postprocessing_acceptance_reports(_policy(), reports)

    assert evaluation.unallowlisted_occurrences
    assert any("extra_in_candidate" in pointer for _, pointer, _ in evaluation.unallowlisted_occurrences)


def test_v2_rejects_payload_sample_larger_than_recorded_limit() -> None:
    reports = _reports(payload_members=PAYLOAD_MEMBERS[:3])
    reports[PARITY]["sample_limit"] = 1

    evaluation = evaluate_postprocessing_acceptance_reports(_policy(), reports)

    assert evaluation.unallowlisted_occurrences
    assert any("payload_mismatch_sample" in pointer for _, pointer, _ in evaluation.unallowlisted_occurrences)


def test_v2_requires_its_exact_policy_and_report_schema_identities() -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="policy version 2"):
        replace(policy, policy_version="1")
    with pytest.raises(ValueError, match="supported schema identity"):
        replace(
            policy,
            completion_exit_contracts=(
                replace(policy.completion_exit_contracts[0], report_schema="other"),
                *policy.completion_exit_contracts[1:],
            ),
        )


@pytest.mark.parametrize(
    ("report", "pointer", "replacement"),
    (
        (PARITY, "/files/0/member_names_ok", False),
        (PARITY, "/files/0/payload_ok", False),
        (PARITY, "/files/0/ok", False),
        (PARITY, "/files/0/compressed_size_mismatch_count", 1),
        (PARITY, "/files/0/compressed_size_mismatch_sample", ["member.zst"]),
        (PARITY, "/tar_file_list_ok", False),
        (PARITY, "/inventory_errors", ["forged"]),
        (PARITY, "/baseline_tar_count", 2),
        (PARITY, "/candidate_tar_count", 2),
        (PARITY, "/compared_tar_count", 2),
        (PARITY, "/compared_members", 11),
        (PARITY, "/duplicate_normalized_member_count", 1),
        (PARITY, "/sampled_member_count", 11),
        (PARITY, "/sample_limit", 0),
        (PARITY, "/payload_hash_scope", "all"),
        (PARITY, "/inventory_only", True),
        (PARITY, "/payload_mismatch_count", 1),
        (PARITY, "/error_count", 1),
        (PARITY, "/ok", False),
        (SEMANTIC, "/ok", False),
        (VERIFY, "/parity_report_path", "wrong.json"),
        (VERIFY, "/semantic_report_path", "wrong.json"),
        (VERIFY, "/schema_version", 2),
        (VERIFY, "/issues", [{"check": "forged"}]),
        (VERIFY, "/ok", False),
    ),
)
def test_v2_rejects_each_derived_comparator_inconsistency(
    report: str,
    pointer: str,
    replacement: object,
) -> None:
    reports = _reports()
    _replace(reports[report], pointer, replacement)

    evaluation = evaluate_postprocessing_acceptance_reports(_policy(), reports)

    assert evaluation.unallowlisted_occurrences
    assert any(pointer_value.startswith("/$coherence") for _, pointer_value, _ in evaluation.error_occurrences)


def _policy() -> PostprocessingAcceptancePolicySnapshot:
    allowances = [
        PostprocessingResidualAllowance(
            report=PARITY,
            json_pointer="/files/payload_mismatch_sample",
            match_kind="exact-tar-member",
            expected_value=member,
            cardinality_kind="permitted-range",
            required_count=0,
            permitted_min=0,
            permitted_max=1,
        )
        for member in PAYLOAD_MEMBERS
    ]
    allowances.extend(
        (
            PostprocessingResidualAllowance(
                report=PARITY,
                json_pointer="/files/extra_in_candidate",
                match_kind="exact-tar-member",
                expected_value=EXTRA_MEMBER,
                cardinality_kind="permitted-range",
                required_count=0,
                permitted_min=0,
                permitted_max=1,
            ),
            PostprocessingResidualAllowance(
                report=SEMANTIC,
                json_pointer="/errors",
                match_kind="json-pointer-count",
                expected_value=SEMANTIC_RESIDUAL,
                cardinality_kind="permitted-range",
                required_count=0,
                permitted_min=0,
                permitted_max=1,
            ),
        )
    )
    return PostprocessingAcceptancePolicySnapshot(
        policy_kind="postprocessing-sealable-v2",
        baseline_id="task853-fixed-fork",
        baseline_version="c8f824d",
        policy_schema="bspp-postprocessing-acceptance",
        policy_version="2",
        residual_allowances=tuple(allowances),
        completion_exit_contracts=tuple(
            PostprocessingCompletionExitContract(
                step_name=step,
                allowed_raw_exit_codes=(0, 1),
                report_paths=(report,),
                report_schema=schema,
                report_schema_version="1",
                outcome_report_path=report,
                outcome_json_pointer="/ok",
                raw_exit_report_outcomes=(
                    PostprocessingRawExitReportOutcome(raw_exit_code=0, report_ok=True),
                    PostprocessingRawExitReportOutcome(raw_exit_code=1, report_ok=False),
                ),
            )
            for step, report, schema in (
                ("acceptance-tar-payload-parity", PARITY, "tar-payload-parity-report"),
                ("acceptance-semantic", SEMANTIC, "semantic-acceptance-summary"),
                ("acceptance-verify-evidence", VERIFY, "acceptance-evidence-report"),
            )
        ),
        baseline_report_bindings=(
            PostprocessingBaselineReportBinding(report=PARITY, baseline_locator_json_pointer="/baseline_dir"),
            PostprocessingBaselineReportBinding(report=SEMANTIC, baseline_locator_json_pointer="/baseline_dir"),
        ),
        cross_report_reconciliations=(
            PostprocessingCrossReportReconciliation(
                left_report=PARITY,
                left_json_pointer="/candidate_dir",
                right_report=SEMANTIC,
                right_json_pointer="/candidate_dir",
            ),
        ),
    )


def _reports(
    *,
    payload_members: tuple[str, ...] = (),
    extra_members: tuple[str, ...] = (),
    semantic_errors: tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    member_names_ok = not extra_members
    payload_ok = not payload_members
    file_ok = member_names_ok and payload_ok
    parity: dict[str, Any] = {
        "baseline_dir": "/baseline",
        "candidate_dir": "/candidate",
        "relative_dir": "local_tars",
        "match_mode": "by-tar",
        "workers": 4,
        "payload_hash_scope": "sampled",
        "sample_limit": 20,
        "payload_sample_count": 5,
        "inventory_only": False,
        "ok": file_ok,
        "tar_file_list_ok": True,
        "inventory_errors": [],
        "baseline_tar_count": 1,
        "candidate_tar_count": 1,
        "baseline_only_tars": [],
        "candidate_only_tars": [],
        "compared_tar_count": 1,
        "compared_members": 10,
        "sampled_member_count": 10,
        "duplicate_normalized_member_count": 0,
        "payload_mismatch_count": len(payload_members),
        "error_count": 0,
        "files": [
            {
                "relative_path": "batch_0001.tar.zst",
                "baseline_member_count": 10,
                "candidate_member_count": 10 + len(extra_members),
                "compared_members": 10,
                "missing_in_candidate": [],
                "extra_in_candidate": list(extra_members),
                "duplicate_baseline_normalized_names": [],
                "duplicate_candidate_normalized_names": [],
                "compressed_size_mismatch_count": 0,
                "compressed_size_mismatch_sample": [],
                "payload_mismatch_count": len(payload_members),
                "payload_mismatch_sample": list(payload_members),
                "errors": [],
                "member_names_ok": member_names_ok,
                "payload_ok": payload_ok,
                "ok": file_ok,
            }
        ],
    }
    semantic: dict[str, Any] = {
        "baseline_dir": "/baseline",
        "candidate_dir": "/candidate",
        "ok": not semantic_errors,
        "errors": list(semantic_errors),
    }
    issues = list(
        project_acceptance_evidence_issues(
            parity_report=parity,
            parity_report_path=PARITY,
            semantic_report=semantic,
            semantic_report_path=SEMANTIC,
        )
    )
    return {
        PARITY: parity,
        SEMANTIC: semantic,
        VERIFY: {
            "schema_version": 1,
            "ok": not issues,
            "parity_report_path": PARITY,
            "semantic_report_path": SEMANTIC,
            "issues": issues,
        },
    }


def _replace(payload: dict[str, Any], pointer: str, replacement: object) -> None:
    current: Any = payload
    tokens = pointer[1:].split("/")
    for token in tokens[:-1]:
        current = current[int(token)] if isinstance(current, list) else current[token]
    current[tokens[-1]] = deepcopy(replacement)


def test_policy_report_path_raises_value_error_on_missing_schema() -> None:
    from bspp.orchestration.contract.postprocessing_acceptance_diagnostics import _policy_report_path

    with pytest.raises(ValueError, match="report_schema"):
        _policy_report_path(_policy(), "nonexistent-report-schema")
