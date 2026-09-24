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

"""Pure acceptance-error evaluation and safe diagnostic projection.

The evaluator deliberately retains complete canonical values.  Its projection is
for operator diagnostics only and must never be used to calculate a verdict.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from bspp.orchestration.contract.postprocessing_acceptance_adjudication import (
    PostprocessingReconciliationResult,
    PostprocessingResidualCardinality,
)
from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    PostprocessingAcceptancePolicySnapshot,
    PostprocessingResidualAllowance,
)

_URI_USERINFO = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^/@\s]+@")
_URI_QUERY_VALUE = re.compile(r"([?&][^=&#\s]+=)[^&#\s]*")


@dataclass(frozen=True)
class PostprocessingAcceptanceEvaluation:
    """Unredacted, unbounded report facts used by the Runtime adjudicator."""

    error_occurrences: frozenset[tuple[str, str, str]]
    claimed_occurrences: frozenset[tuple[str, str, str]]
    residual_cardinalities: tuple[PostprocessingResidualCardinality, ...]
    reconciliation_results: tuple[PostprocessingReconciliationResult, ...]

    @property
    def unallowlisted_occurrences(self) -> frozenset[tuple[str, str, str]]:
        return self.error_occurrences - self.claimed_occurrences


def evaluate_postprocessing_acceptance_reports(
    policy: PostprocessingAcceptancePolicySnapshot,
    reports: Mapping[str, Mapping[str, Any]],
) -> PostprocessingAcceptanceEvaluation:
    """Evaluate all report occurrences without redaction or output limits."""
    if policy.policy_kind == "postprocessing-sealable-v2":
        return _evaluate_v2_acceptance_reports(policy, reports)
    errors = _report_error_occurrences(reports)
    claimed: set[tuple[str, str, str]] = set()
    cardinalities: list[PostprocessingResidualCardinality] = []
    for allowance in policy.residual_allowances:
        allowance_id = canonical_allowance_id(allowance)
        occurrences = _allowance_occurrences(allowance, reports)
        claimed.update(occurrences & errors)
        cardinalities.append(
            PostprocessingResidualCardinality(
                allowance_id=allowance_id,
                observed=len(occurrences),
                permitted_min=allowance.permitted_min,
                permitted_max=allowance.permitted_max,
            )
        )
    reconciliations: list[PostprocessingReconciliationResult] = []
    for reconciliation in policy.cross_report_reconciliations:
        reconciliation_id = (
            f"{reconciliation.left_report}#{reconciliation.left_json_pointer}=="
            f"{reconciliation.right_report}#{reconciliation.right_json_pointer}"
        )
        try:
            left = _json_pointer(reports[reconciliation.left_report], reconciliation.left_json_pointer)
            right = _json_pointer(reports[reconciliation.right_report], reconciliation.right_json_pointer)
            matched = left == right
        except (KeyError, IndexError, TypeError, ValueError):
            matched = False
        reconciliations.append(PostprocessingReconciliationResult(reconciliation_id=reconciliation_id, matched=matched))
    return PostprocessingAcceptanceEvaluation(
        error_occurrences=frozenset(errors),
        claimed_occurrences=frozenset(claimed),
        residual_cardinalities=tuple(sorted(cardinalities, key=lambda item: item.allowance_id)),
        reconciliation_results=tuple(sorted(reconciliations, key=lambda item: item.reconciliation_id)),
    )


def project_acceptance_evidence_issues(
    *,
    parity_report: Mapping[str, Any] | None = None,
    parity_report_path: str | None = None,
    semantic_report: Mapping[str, Any] | None = None,
    semantic_report_path: str | None = None,
) -> tuple[dict[str, object], ...]:
    """Project valid comparator payloads into the verifier's frozen issue order."""
    issues: list[dict[str, object]] = []
    if parity_report is not None:
        if parity_report.get("ok") is not True:
            issues.append(_projected_issue("tar-payload-parity", parity_report_path, "ok is not true"))
        for key in ("payload_mismatch_count", "error_count"):
            if _int_value(parity_report.get(key)) != 0:
                issues.append(
                    _projected_issue(
                        "tar-payload-parity",
                        parity_report_path,
                        f"{key} is not 0: {parity_report.get(key)!r}",
                    )
                )
        for key in ("compared_tar_count", "compared_members"):
            value = _int_value(parity_report.get(key))
            if value is None or value <= 0:
                issues.append(
                    _projected_issue(
                        "tar-payload-parity",
                        parity_report_path,
                        f"{key} is not positive: {parity_report.get(key)!r}",
                    )
                )
        if parity_report.get("payload_sample_count") == 0:
            issues.append(
                _projected_issue(
                    "tar-payload-parity",
                    parity_report_path,
                    "payload_sample_count is 0; inventory-only parity is not acceptance",
                )
            )
    if semantic_report is not None:
        if semantic_report.get("ok") is not True:
            issues.append(_projected_issue("semantic-acceptance", semantic_report_path, "ok is not true"))
        errors = semantic_report.get("errors")
        if errors not in ([], None):
            issues.append(
                _projected_issue(
                    "semantic-acceptance",
                    semantic_report_path,
                    f"errors is not empty: {errors!r}",
                )
            )
    return tuple(issues)


def _projected_issue(check: str, report_path: str | None, message: str) -> dict[str, object]:
    return {"check": check, "report_path": report_path, "message": message}


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _evaluate_v2_acceptance_reports(
    policy: PostprocessingAcceptancePolicySnapshot,
    reports: Mapping[str, Mapping[str, Any]],
) -> PostprocessingAcceptanceEvaluation:
    errors = _v2_report_error_occurrences(policy, reports)
    errors.update(_v2_coherence_occurrences(policy, reports))
    claimed: set[tuple[str, str, str]] = set()
    cardinalities: list[PostprocessingResidualCardinality] = []
    for allowance in policy.residual_allowances:
        allowance_id = canonical_allowance_id(allowance)
        occurrences = _v2_allowance_occurrences(allowance, reports)
        claimed.update(occurrences & errors)
        cardinalities.append(
            PostprocessingResidualCardinality(
                allowance_id=allowance_id,
                observed=len(occurrences),
                permitted_min=allowance.permitted_min,
                permitted_max=allowance.permitted_max,
            )
        )
    reconciliations: list[PostprocessingReconciliationResult] = []
    for reconciliation in policy.cross_report_reconciliations:
        reconciliation_id = (
            f"{reconciliation.left_report}#{reconciliation.left_json_pointer}=="
            f"{reconciliation.right_report}#{reconciliation.right_json_pointer}"
        )
        try:
            left = _json_pointer(reports[reconciliation.left_report], reconciliation.left_json_pointer)
            right = _json_pointer(reports[reconciliation.right_report], reconciliation.right_json_pointer)
            matched = left == right
        except (KeyError, IndexError, TypeError, ValueError):
            matched = False
        reconciliations.append(PostprocessingReconciliationResult(reconciliation_id=reconciliation_id, matched=matched))
    return PostprocessingAcceptanceEvaluation(
        error_occurrences=frozenset(errors),
        claimed_occurrences=frozenset(claimed),
        residual_cardinalities=tuple(sorted(cardinalities, key=lambda item: item.allowance_id)),
        reconciliation_results=tuple(sorted(reconciliations, key=lambda item: item.reconciliation_id)),
    )


def _v2_report_error_occurrences(
    policy: PostprocessingAcceptancePolicySnapshot,
    reports: Mapping[str, Mapping[str, Any]],
) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    parity_path = _policy_report_path(policy, "tar-payload-parity-report")
    semantic_path = _policy_report_path(policy, "semantic-acceptance-summary")
    parity = reports.get(parity_path)
    if parity is not None:
        for pointer in ("/inventory_errors", "/baseline_only_tars", "/candidate_only_tars"):
            _add_error_values(result, parity_path, pointer, _optional_json_pointer(parity, pointer))
        files = parity.get("files")
        if isinstance(files, list):
            for index, item in enumerate(files):
                if not isinstance(item, Mapping):
                    result.add((parity_path, f"/files/{index}", _stable_value(item)))
                    continue
                for name in (
                    "missing_in_candidate",
                    "extra_in_candidate",
                    "duplicate_baseline_normalized_names",
                    "duplicate_candidate_normalized_names",
                    "payload_mismatch_sample",
                    "errors",
                ):
                    _add_error_values(result, parity_path, f"/files/{index}/{name}", item.get(name))
        elif files is not None:
            result.add((parity_path, "/files", _stable_value(files)))
    semantic = reports.get(semantic_path)
    if semantic is not None:
        _add_error_values(result, semantic_path, "/errors", semantic.get("errors"))
    return result


def _v2_allowance_occurrences(
    allowance: PostprocessingResidualAllowance,
    reports: Mapping[str, Mapping[str, Any]],
) -> set[tuple[str, str, str]]:
    if allowance.match_kind != "exact-tar-member":
        return _allowance_occurrences(allowance, reports)
    payload = reports.get(allowance.report)
    if payload is None:
        return set()
    files = payload.get("files")
    if not isinstance(files, list):
        return set()
    difference_kind = allowance.json_pointer.rsplit("/", 1)[-1]
    expected = allowance.expected_value
    occurrences: set[tuple[str, str, str]] = set()
    for file_index, file_payload in enumerate(files):
        if not isinstance(file_payload, Mapping):
            continue
        values = file_payload.get(difference_kind)
        if not isinstance(values, list):
            continue
        occurrences.update(
            (
                allowance.report,
                f"/files/{file_index}/{difference_kind}/{item_index}",
                _stable_value(item),
            )
            for item_index, item in enumerate(values)
            if isinstance(item, str) and item == expected
        )
    return occurrences


def _policy_report_path(policy: PostprocessingAcceptancePolicySnapshot, report_schema: str) -> str:
    result = next(
        (
            report
            for contract in policy.completion_exit_contracts
            if contract.report_schema == report_schema
            for report in contract.report_paths
        ),
        None,
    )
    if result is None:
        raise ValueError(f"V2 policy missing contract with report_schema={report_schema!r}")
    return result


_MISSING = object()


def _v2_coherence_occurrences(
    policy: PostprocessingAcceptancePolicySnapshot,
    reports: Mapping[str, Mapping[str, Any]],
) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    parity_path = _policy_report_path(policy, "tar-payload-parity-report")
    semantic_path = _policy_report_path(policy, "semantic-acceptance-summary")
    verify_path = _policy_report_path(policy, "acceptance-evidence-report")
    parity = reports.get(parity_path)
    semantic = reports.get(semantic_path)
    verify = reports.get(verify_path)
    if parity is None:
        _coherence_failure(result, parity_path, "", "report object", _MISSING)
    else:
        _check_v2_parity_coherence(result, parity_path, parity)
    if semantic is None:
        _coherence_failure(result, semantic_path, "", "report object", _MISSING)
    else:
        _check_v2_semantic_coherence(result, semantic_path, semantic)
    if verify is None:
        _coherence_failure(result, verify_path, "", "report object", _MISSING)
    elif parity is not None and semantic is not None:
        _check_formula(result, verify_path, "/schema_version", 1, verify)
        expected_issues = list(
            project_acceptance_evidence_issues(
                parity_report=parity,
                parity_report_path=parity_path,
                semantic_report=semantic,
                semantic_report_path=semantic_path,
            )
        )
        _check_formula(result, verify_path, "/parity_report_path", parity_path, verify)
        _check_formula(result, verify_path, "/semantic_report_path", semantic_path, verify)
        _check_formula(result, verify_path, "/issues", expected_issues, verify)
        _check_formula(result, verify_path, "/ok", not expected_issues, verify)
    return result


def _check_v2_parity_coherence(
    result: set[tuple[str, str, str]],
    report: str,
    payload: Mapping[str, Any],
) -> None:
    files = _coherent_list(result, report, "/files", payload.get("files", _MISSING))
    baseline_only = _coherent_list(result, report, "/baseline_only_tars", payload.get("baseline_only_tars", _MISSING))
    candidate_only = _coherent_list(
        result, report, "/candidate_only_tars", payload.get("candidate_only_tars", _MISSING)
    )
    inventory_errors = _coherent_list(result, report, "/inventory_errors", payload.get("inventory_errors", _MISSING))
    relative_dir = payload.get("relative_dir", _MISSING)
    if not isinstance(relative_dir, str):
        _coherence_failure(result, report, "/relative_dir", "string", relative_dir)
        relative_dir = None
    payload_sample_count = payload.get("payload_sample_count", _MISSING)
    if payload_sample_count is not None and not _is_non_negative_int(payload_sample_count):
        _coherence_failure(
            result,
            report,
            "/payload_sample_count",
            "null or non-negative integer",
            payload_sample_count,
        )
        payload_sample_count = _MISSING
    sample_limit = payload.get("sample_limit", _MISSING)
    if not _is_positive_int(sample_limit):
        _coherence_failure(result, report, "/sample_limit", "positive integer", sample_limit)
        sample_limit = None

    file_formulas: list[dict[str, object]] = []
    if files is not None:
        for index, item in enumerate(files):
            if not isinstance(item, Mapping):
                _coherence_failure(result, report, f"/files/{index}", "object", item)
                continue
            formulas = _check_v2_parity_file_coherence(
                result,
                report,
                index,
                item,
                sample_limit=sample_limit,
            )
            if formulas is not None:
                file_formulas.append(formulas)

    if files is None or baseline_only is None or candidate_only is None:
        return
    _check_formula(result, report, "/tar_file_list_ok", not baseline_only and not candidate_only, payload)
    _check_formula(result, report, "/baseline_tar_count", len(files) + len(baseline_only), payload)
    _check_formula(result, report, "/candidate_tar_count", len(files) + len(candidate_only), payload)
    _check_formula(result, report, "/compared_tar_count", len(files), payload)
    if relative_dir is not None:
        expected_inventory_errors = (
            []
            if files
            else [
                f"no paired tar files found under relative_dir={relative_dir!r} "
                f"(baseline={len(files) + len(baseline_only)}, candidate={len(files) + len(candidate_only)})"
            ]
        )
        _check_formula(result, report, "/inventory_errors", expected_inventory_errors, payload)
    if len(file_formulas) == len(files):
        compared_members = sum(cast("int", item["compared_members"]) for item in file_formulas)
        payload_mismatches = sum(cast("int", item["payload_mismatch_count"]) for item in file_formulas)
        duplicate_count = sum(cast("int", item["duplicate_count"]) for item in file_formulas)
        file_error_count = sum(cast("int", item["error_count"]) for item in file_formulas)
        _check_formula(result, report, "/compared_members", compared_members, payload)
        _check_formula(result, report, "/payload_mismatch_count", payload_mismatches, payload)
        _check_formula(result, report, "/duplicate_normalized_member_count", duplicate_count, payload)
        if inventory_errors is not None:
            _check_formula(result, report, "/error_count", file_error_count + len(inventory_errors), payload)
            _check_formula(
                result,
                report,
                "/ok",
                not baseline_only
                and not candidate_only
                and not inventory_errors
                and all(bool(item["ok"]) for item in file_formulas),
                payload,
            )
        if payload_sample_count is not _MISSING:
            _check_formula(
                result,
                report,
                "/sampled_member_count",
                compared_members if payload_sample_count is not None else None,
                payload,
            )
    if payload_sample_count is not _MISSING:
        _check_formula(result, report, "/inventory_only", payload_sample_count == 0, payload)
        expected_scope = (
            "all" if payload_sample_count is None else "inventory-only" if payload_sample_count == 0 else "sampled"
        )
        _check_formula(result, report, "/payload_hash_scope", expected_scope, payload)


def _check_v2_parity_file_coherence(
    result: set[tuple[str, str, str]],
    report: str,
    index: int,
    payload: Mapping[str, Any],
    *,
    sample_limit: int | None,
) -> dict[str, object] | None:
    prefix = f"/files/{index}"
    list_fields: dict[str, list[object]] = {}
    for name in (
        "missing_in_candidate",
        "extra_in_candidate",
        "duplicate_baseline_normalized_names",
        "duplicate_candidate_normalized_names",
        "compressed_size_mismatch_sample",
        "payload_mismatch_sample",
        "errors",
    ):
        value = _coherent_list(result, report, f"{prefix}/{name}", payload.get(name, _MISSING))
        if value is not None:
            list_fields[name] = value
    counts: dict[str, int] = {}
    for name in ("compared_members", "compressed_size_mismatch_count", "payload_mismatch_count"):
        value = payload.get(name, _MISSING)
        if not _is_non_negative_int(value):
            _coherence_failure(result, report, f"{prefix}/{name}", "non-negative integer", value)
        else:
            counts[name] = value
    member_counts: dict[str, int | None] = {}
    for name in ("baseline_member_count", "candidate_member_count"):
        value = payload.get(name, _MISSING)
        if value is not None and not _is_non_negative_int(value):
            _coherence_failure(result, report, f"{prefix}/{name}", "null or non-negative integer", value)
        else:
            member_counts[name] = value
    required_lists = {
        "missing_in_candidate",
        "extra_in_candidate",
        "duplicate_baseline_normalized_names",
        "duplicate_candidate_normalized_names",
        "compressed_size_mismatch_sample",
        "payload_mismatch_sample",
        "errors",
    }
    required_counts = {"compared_members", "compressed_size_mismatch_count", "payload_mismatch_count"}
    if set(list_fields) != required_lists or set(counts) != required_counts or len(member_counts) != 2:
        return None
    _check_formula(
        result,
        report,
        f"{prefix}/payload_mismatch_count",
        len(list_fields["payload_mismatch_sample"]),
        payload,
    )
    if sample_limit is not None:
        _check_exact_bounded_sample(
            result,
            report,
            f"{prefix}/compressed_size_mismatch_sample",
            list_fields["compressed_size_mismatch_sample"],
            counts["compressed_size_mismatch_count"],
            sample_limit,
        )
        if len(list_fields["payload_mismatch_sample"]) > sample_limit:
            _coherence_failure(
                result,
                report,
                f"{prefix}/payload_mismatch_sample",
                f"at most sample_limit={sample_limit} items",
                list_fields["payload_mismatch_sample"],
            )
        for name in (
            "missing_in_candidate",
            "extra_in_candidate",
            "duplicate_baseline_normalized_names",
            "duplicate_candidate_normalized_names",
        ):
            values = list_fields[name]
            if len(values) >= sample_limit:
                _coherence_failure(
                    result,
                    report,
                    f"{prefix}/{name}",
                    f"fewer than sample_limit={sample_limit} items (complete array)",
                    values,
                )
    if (
        sample_limit is not None
        and all(
            len(list_fields[name]) < sample_limit
            for name in (
                "missing_in_candidate",
                "extra_in_candidate",
                "duplicate_baseline_normalized_names",
                "duplicate_candidate_normalized_names",
            )
        )
        and not list_fields["duplicate_baseline_normalized_names"]
        and not list_fields["duplicate_candidate_normalized_names"]
        and member_counts["baseline_member_count"] is not None
        and member_counts["candidate_member_count"] is not None
    ):
        expected_candidate_count = (
            member_counts["baseline_member_count"]
            - len(list_fields["missing_in_candidate"])
            + len(list_fields["extra_in_candidate"])
        )
        _check_formula(
            result,
            report,
            f"{prefix}/candidate_member_count",
            expected_candidate_count,
            payload,
        )
    member_names_ok = (
        member_counts["baseline_member_count"] is not None
        and member_counts["candidate_member_count"] is not None
        and not list_fields["missing_in_candidate"]
        and not list_fields["extra_in_candidate"]
        and not list_fields["duplicate_baseline_normalized_names"]
        and not list_fields["duplicate_candidate_normalized_names"]
    )
    payload_ok = counts["payload_mismatch_count"] == 0 and not list_fields["errors"]
    ok = member_names_ok and payload_ok
    _check_formula(result, report, f"{prefix}/member_names_ok", member_names_ok, payload)
    _check_formula(result, report, f"{prefix}/payload_ok", payload_ok, payload)
    _check_formula(result, report, f"{prefix}/ok", ok, payload)
    duplicate_count = len(list_fields["duplicate_baseline_normalized_names"]) + len(
        list_fields["duplicate_candidate_normalized_names"]
    )
    return {
        "compared_members": counts["compared_members"],
        "payload_mismatch_count": counts["payload_mismatch_count"],
        "duplicate_count": duplicate_count,
        "error_count": len(list_fields["errors"]),
        "ok": ok,
    }


def _check_v2_semantic_coherence(
    result: set[tuple[str, str, str]],
    report: str,
    payload: Mapping[str, Any],
) -> None:
    errors = _coherent_list(result, report, "/errors", payload.get("errors", _MISSING))
    if errors is not None:
        _check_formula(result, report, "/ok", not errors, payload)


def _check_exact_bounded_sample(
    result: set[tuple[str, str, str]],
    report: str,
    pointer: str,
    sample: list[object],
    count: int,
    sample_limit: int,
) -> None:
    expected_size = min(count, sample_limit)
    if len(sample) != expected_size:
        _coherence_failure(result, report, pointer, f"exactly min(count, sample_limit)={expected_size} items", sample)


def _coherent_list(
    result: set[tuple[str, str, str]],
    report: str,
    pointer: str,
    value: object,
) -> list[object] | None:
    if not isinstance(value, list):
        _coherence_failure(result, report, pointer, "array", value)
        return None
    return value


def _check_formula(
    result: set[tuple[str, str, str]],
    report: str,
    pointer: str,
    expected: object,
    payload: Mapping[str, Any],
) -> None:
    observed = payload.get(pointer.rsplit("/", 1)[-1], _MISSING)
    if type(observed) is not type(expected) or observed != expected:
        _coherence_failure(result, report, pointer, expected, observed)


def _coherence_failure(
    result: set[tuple[str, str, str]],
    report: str,
    pointer: str,
    expected: object,
    observed: object,
) -> None:
    result.add(
        (
            report,
            f"/$coherence{pointer}",
            _stable_value(
                {
                    "expected": expected,
                    "observed": {"missing": True} if observed is _MISSING else observed,
                }
            ),
        )
    )


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def project_unallowlisted_occurrences(
    occurrences: frozenset[tuple[str, str, str]] | set[tuple[str, str, str]],
    *,
    max_rows: int = 64,
    max_display_value_chars: int = 1024,
) -> dict[str, object]:
    """Return a deterministic redacted, bounded view of canonical occurrences."""
    if max_rows < 0 or max_display_value_chars < 0:
        raise ValueError("acceptance diagnostic projection bounds must be non-negative")
    ordered = tuple(sorted(occurrences))
    rows = [
        {
            "report": report,
            "json_pointer": pointer,
            # Canonical values can contain arbitrary report text, including
            # distinct credentials that happen to render alike.  Keep them in
            # the complete evaluator only; Control receives no raw value.
            "value": safe_canonical_json_projection(value, max_display_value_chars=max_display_value_chars),
        }
        for report, pointer, value in ordered[:max_rows]
    ]
    return {"rows": rows, "omitted_count": len(ordered) - len(rows)}


def _report_error_occurrences(reports: Mapping[str, Mapping[str, Any]]) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    for report, payload in reports.items():
        for pointer in (
            "/inventory_errors",
            "/baseline_only_tars",
            "/candidate_only_tars",
            "/errors",
            "/issues",
            "/payload_mismatch_count",
            "/error_count",
        ):
            _add_error_values(result, report, pointer, _optional_json_pointer(payload, pointer))
        files = payload.get("files")
        if isinstance(files, list):
            for index, item in enumerate(files):
                if not isinstance(item, Mapping):
                    result.add((report, f"/files/{index}", _stable_value(item)))
                    continue
                for name in (
                    "missing_in_candidate",
                    "extra_in_candidate",
                    "duplicate_baseline_normalized_names",
                    "duplicate_candidate_normalized_names",
                    "payload_mismatch_sample",
                    "errors",
                ):
                    pointer = f"/files/{index}/{name}"
                    _add_error_values(result, report, pointer, item.get(name))
    return result


def _add_error_values(destination: set[tuple[str, str, str]], report: str, pointer: str, value: object) -> None:
    if isinstance(value, list):
        destination.update((report, f"{pointer}/{index}", _stable_value(item)) for index, item in enumerate(value))
    elif value not in (None, "", False, 0):
        destination.add((report, pointer, _stable_value(value)))


def _allowance_occurrences(
    allowance: PostprocessingResidualAllowance, reports: Mapping[str, Mapping[str, Any]]
) -> set[tuple[str, str, str]]:
    try:
        value = _json_pointer(reports[allowance.report], allowance.json_pointer)
    except (KeyError, IndexError, TypeError, ValueError):
        return set()
    expected = _stable_value(allowance.expected_value)
    if allowance.match_kind == "json-pointer-equals":
        return {(allowance.report, allowance.json_pointer, expected)} if _stable_value(value) == expected else set()
    if not isinstance(value, list):
        return set()
    return {
        (allowance.report, f"{allowance.json_pointer}/{index}", _stable_value(item))
        for index, item in enumerate(value)
        if _stable_value(item) == expected
    }


def canonical_allowance_id(allowance: PostprocessingResidualAllowance) -> str:
    """Return the frozen byte-for-byte allowance identity used by Runtime."""
    return (
        f"{allowance.report}#{allowance.json_pointer}#{allowance.match_kind}#{_stable_value(allowance.expected_value)}"
    )


def diagnostic_allowance_reference(
    allowance: PostprocessingResidualAllowance,
    *,
    max_chars: int = 1024,
) -> dict[str, str]:
    """Return a safe reference to one frozen allowance without its expected value."""
    if max_chars < 0:
        raise ValueError("acceptance diagnostic reference bound must be non-negative")
    expected = _stable_value(allowance.expected_value)
    return {
        "report": _safe_string_projection(allowance.report, max_chars=max_chars),
        "json_pointer": _safe_string_projection(allowance.json_pointer, max_chars=max_chars),
        "match_kind": _safe_string_projection(allowance.match_kind, max_chars=max_chars),
        "expected_value_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
    }


def safe_canonical_json_projection(value: str, *, max_display_value_chars: int) -> str:
    """Sanitize a canonical JSON value without changing ordinary diagnostic facts."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return _opaque_marker(digest, len(value))
    rendered = json.dumps(_sanitize_json_value(parsed), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return rendered if len(rendered) <= max_display_value_chars else _opaque_marker(digest, len(value))


def _stable_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_pointer(payload: object, pointer: str) -> object:
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be absolute")
    current = payload
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit():
                raise ValueError("JSON pointer list token must be an index")
            current = current[int(token)]
        else:
            raise TypeError("JSON pointer traverses a scalar")
    return current


def _optional_json_pointer(payload: object, pointer: str) -> object:
    try:
        return _json_pointer(payload, pointer)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _sanitize_json_value(value: object) -> object:
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            rendered_key = _sanitize_string(key) if isinstance(key, str) else _sanitize_string(str(key))
            if rendered_key in result:
                rendered_key = f"{rendered_key}#sha256-{hashlib.sha256(str(key).encode()).hexdigest()[:12]}"
            result[rendered_key] = _sanitize_json_value(item)
        return result
    return value


def _sanitize_string(value: str) -> str:
    return _URI_QUERY_VALUE.sub(r"\1<redacted>", _URI_USERINFO.sub(r"\1<redacted>@", value))


def _safe_string_projection(value: str, *, max_chars: int) -> str:
    sanitized = _sanitize_string(value)
    if len(sanitized) <= max_chars:
        return sanitized
    return _opaque_marker(hashlib.sha256(value.encode("utf-8")).hexdigest(), len(value))


def _opaque_marker(digest: str, length: int) -> str:
    return f"<opaque-canonical-json sha256={digest} length={length}>"


__all__ = [
    "PostprocessingAcceptanceEvaluation",
    "canonical_allowance_id",
    "diagnostic_allowance_reference",
    "evaluate_postprocessing_acceptance_reports",
    "project_acceptance_evidence_issues",
    "project_unallowlisted_occurrences",
    "safe_canonical_json_projection",
]
