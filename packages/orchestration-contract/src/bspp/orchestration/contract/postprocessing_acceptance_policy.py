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

"""Immutable postprocessing acceptance policy contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract._postprocessing_validation import _fields, _int, _mapping_list, _schema, _str
from bspp.orchestration.contract.phase import (
    canonical_mapping_digest,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

AcceptanceStepName = Literal["acceptance-tar-payload-parity", "acceptance-semantic", "acceptance-verify-evidence"]
ResidualMatchKind = Literal["json-pointer-equals", "json-pointer-count", "exact-tar-member"]

_V2_TAR_DIFFERENCE_POINTERS = {
    "/files/missing_in_candidate",
    "/files/extra_in_candidate",
    "/files/payload_mismatch_sample",
}


@dataclass(frozen=True)
class PostprocessingRawExitReportOutcome:
    raw_exit_code: int
    report_ok: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.raw_exit_code, int)
            or isinstance(self.raw_exit_code, bool)
            or not 0 <= self.raw_exit_code <= 255
            or not isinstance(self.report_ok, bool)
        ):
            raise ValueError("postprocessing raw exit/report outcome is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {"raw_exit_code": self.raw_exit_code, "report_ok": self.report_ok}


@dataclass(frozen=True)
class PostprocessingResidualAllowance:
    """One exact, cardinality-bounded residual accepted for one fixed baseline."""

    report: str
    json_pointer: str
    match_kind: ResidualMatchKind
    expected_value: object
    cardinality_kind: Literal["exact", "permitted-range"]
    required_count: int
    permitted_min: int
    permitted_max: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not self.report or not self.json_pointer.startswith("/"):
            raise ValueError("acceptance residual report and absolute JSON pointer must be non-empty")
        if self.match_kind not in {"json-pointer-equals", "json-pointer-count", "exact-tar-member"}:
            raise ValueError(f"unsupported residual match kind: {self.match_kind!r}")
        if self.expected_value is None or isinstance(self.expected_value, tuple):
            raise ValueError("acceptance residual expected_value must be a non-null JSON value")
        try:
            json.dumps(self.expected_value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("acceptance residual expected_value must be a non-null JSON value") from exc
        if self.cardinality_kind not in {"exact", "permitted-range"}:
            raise ValueError("acceptance residual cardinality_kind must be exact or permitted-range")
        counts = (self.required_count, self.permitted_min, self.permitted_max)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("acceptance residual cardinalities must be non-negative integers")
        if not self.permitted_min <= self.required_count <= self.permitted_max:
            raise ValueError("required residual count must be within its permitted range")
        if self.cardinality_kind == "exact" and not (self.required_count == self.permitted_min == self.permitted_max):
            raise ValueError("exact residual cardinality requires equal required/min/max counts")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "report": self.report,
            "json_pointer": self.json_pointer,
            "match_kind": self.match_kind,
            "expected_value": self.expected_value,
            "cardinality_kind": self.cardinality_kind,
            "required_count": self.required_count,
            "permitted_min": self.permitted_min,
            "permitted_max": self.permitted_max,
        }


@dataclass(frozen=True)
class PostprocessingCompletionExitContract:
    """Raw-command completion and report requirements for one capture action."""

    step_name: AcceptanceStepName
    allowed_raw_exit_codes: tuple[int, ...]
    report_paths: tuple[str, ...]
    report_schema: str
    report_schema_version: str
    outcome_report_path: str
    outcome_json_pointer: str
    raw_exit_report_outcomes: tuple[PostprocessingRawExitReportOutcome, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.step_name not in {
            "acceptance-tar-payload-parity",
            "acceptance-semantic",
            "acceptance-verify-evidence",
        }:
            raise ValueError(f"unsupported acceptance completion step: {self.step_name!r}")
        if (
            not isinstance(self.allowed_raw_exit_codes, tuple)
            or not self.allowed_raw_exit_codes
            or tuple(sorted(set(self.allowed_raw_exit_codes))) != self.allowed_raw_exit_codes
            or any(
                not isinstance(code, int) or isinstance(code, bool) or not 0 <= code <= 255
                for code in self.allowed_raw_exit_codes
            )
        ):
            raise ValueError("allowed raw exit codes must be a sorted unique non-empty tuple in 0..255")
        if (
            not isinstance(self.report_paths, tuple)
            or not self.report_paths
            or len(set(self.report_paths)) != len(self.report_paths)
            or any(not path or path.startswith("/") or ".." in path.split("/") for path in self.report_paths)
        ):
            raise ValueError("acceptance report paths must be unique authority-relative paths")
        if not self.report_schema or not self.report_schema_version:
            raise ValueError("acceptance report schema and version must be non-empty")
        if self.outcome_report_path not in self.report_paths or not self.outcome_json_pointer.startswith("/"):
            raise ValueError("acceptance completion outcome must select one declared report and absolute pointer")
        if (
            not isinstance(self.raw_exit_report_outcomes, tuple)
            or tuple(item.raw_exit_code for item in self.raw_exit_report_outcomes) != self.allowed_raw_exit_codes
        ):
            raise ValueError("raw exit/report outcomes must map each allowed exit exactly once in sorted order")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "step_name": self.step_name,
            "allowed_raw_exit_codes": list(self.allowed_raw_exit_codes),
            "report_paths": list(self.report_paths),
            "report_schema": self.report_schema,
            "report_schema_version": self.report_schema_version,
            "outcome_report_path": self.outcome_report_path,
            "outcome_json_pointer": self.outcome_json_pointer,
            "raw_exit_report_outcomes": [item.to_mapping() for item in self.raw_exit_report_outcomes],
        }


@dataclass(frozen=True)
class PostprocessingCrossReportReconciliation:
    """One exact equality required between two frozen report identities."""

    left_report: str
    left_json_pointer: str
    right_report: str
    right_json_pointer: str
    comparison: Literal["equal"] = "equal"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if (
            any(
                not value
                for value in (
                    self.left_report,
                    self.left_json_pointer,
                    self.right_report,
                    self.right_json_pointer,
                )
            )
            or not self.left_json_pointer.startswith("/")
            or not self.right_json_pointer.startswith("/")
        ):
            raise ValueError("cross-report reconciliation requires reports and absolute JSON pointers")
        if self.comparison != "equal":
            raise ValueError("cross-report reconciliation comparison must be equal")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "left_report": self.left_report,
            "left_json_pointer": self.left_json_pointer,
            "right_report": self.right_report,
            "right_json_pointer": self.right_json_pointer,
            "comparison": self.comparison,
        }


@dataclass(frozen=True)
class PostprocessingBaselineReportBinding:
    """One raw report pointer that must equal the frozen RunSpec baseline locator."""

    report: str
    baseline_locator_json_pointer: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not self.report or not self.baseline_locator_json_pointer.startswith("/"):
            raise ValueError("baseline report binding requires a report and absolute locator JSON pointer")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "report": self.report,
            "baseline_locator_json_pointer": self.baseline_locator_json_pointer,
        }


@dataclass(frozen=True)
class PostprocessingAcceptancePolicySnapshot:
    """Immutable policy used by capture actions and the sole adjudicator."""

    baseline_id: str
    baseline_version: str
    policy_schema: str
    policy_version: str
    residual_allowances: tuple[PostprocessingResidualAllowance, ...]
    completion_exit_contracts: tuple[PostprocessingCompletionExitContract, ...]
    baseline_report_bindings: tuple[PostprocessingBaselineReportBinding, ...]
    cross_report_reconciliations: tuple[PostprocessingCrossReportReconciliation, ...]
    policy_kind: str = "postprocessing-sealable-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.policy_kind not in {"postprocessing-sealable-v1", "postprocessing-sealable-v2"}:
            raise ValueError(
                "postprocessing acceptance policy_kind must be "
                "'postprocessing-sealable-v1' or 'postprocessing-sealable-v2'"
            )
        if not self.baseline_id or not self.baseline_version or not self.policy_schema or not self.policy_version:
            raise ValueError("postprocessing acceptance baseline and policy identities must be non-empty")
        if not isinstance(self.residual_allowances, tuple):
            raise ValueError("acceptance residual allowances must be an immutable tuple")
        residual_keys = tuple(
            (item.report, item.json_pointer, item.match_kind)
            if self.policy_kind == "postprocessing-sealable-v1"
            else (item.report, item.json_pointer, item.match_kind, _canonical_json(item.expected_value))
            for item in self.residual_allowances
        )
        if len(set(residual_keys)) != len(residual_keys):
            raise ValueError("acceptance residual allowances must be unique")
        if self.policy_kind == "postprocessing-sealable-v1" and any(
            item.match_kind == "exact-tar-member" for item in self.residual_allowances
        ):
            raise ValueError("unsupported residual match kind: 'exact-tar-member'")
        if not isinstance(self.completion_exit_contracts, tuple):
            raise ValueError("acceptance completion contracts must be an immutable tuple")
        observed = tuple(item.step_name for item in self.completion_exit_contracts)
        expected = (
            "acceptance-tar-payload-parity",
            "acceptance-semantic",
            "acceptance-verify-evidence",
        )
        if observed != expected:
            raise ValueError("acceptance completion contracts must contain the three capture steps in order")
        if not isinstance(self.baseline_report_bindings, tuple) or not self.baseline_report_bindings:
            raise ValueError("acceptance policy requires immutable baseline report bindings")
        binding_keys = tuple(
            (item.report, item.baseline_locator_json_pointer) for item in self.baseline_report_bindings
        )
        if len(set(binding_keys)) != len(binding_keys):
            raise ValueError("acceptance policy baseline report bindings must be unique")
        if not isinstance(self.cross_report_reconciliations, tuple) or not self.cross_report_reconciliations:
            raise ValueError("acceptance policy requires cross-report reconciliation rules")
        if self.policy_kind == "postprocessing-sealable-v2":
            self._validate_v2_identity()
            self._validate_v2_allowances()

    def _validate_v2_identity(self) -> None:
        if self.policy_schema != "bspp-postprocessing-acceptance" or self.policy_version != "2":
            raise ValueError("V2 policy kind requires bspp-postprocessing-acceptance policy version 2")
        expected_report_schemas = (
            "tar-payload-parity-report",
            "semantic-acceptance-summary",
            "acceptance-evidence-report",
        )
        for contract, expected_schema in zip(
            self.completion_exit_contracts,
            expected_report_schemas,
            strict=True,
        ):
            if (
                contract.report_schema != expected_schema
                or contract.report_schema_version != "1"
                or len(contract.report_paths) != 1
                or contract.outcome_report_path != contract.report_paths[0]
            ):
                raise ValueError(
                    "V2 completion contracts require one outcome report with the supported schema identity"
                )

    def _validate_v2_allowances(self) -> None:
        report_schema_by_path = {
            report: contract.report_schema
            for contract in self.completion_exit_contracts
            for report in contract.report_paths
        }
        for allowance in self.residual_allowances:
            if allowance.cardinality_kind != "permitted-range" or (
                allowance.required_count,
                allowance.permitted_min,
                allowance.permitted_max,
            ) != (0, 0, 1):
                raise ValueError("V2 residual allowances must be optional with required/min/max counts 0/0/1")
            report_schema = report_schema_by_path.get(allowance.report)
            if allowance.match_kind == "exact-tar-member":
                if (
                    report_schema != "tar-payload-parity-report"
                    or allowance.json_pointer not in _V2_TAR_DIFFERENCE_POINTERS
                    or not isinstance(allowance.expected_value, str)
                    or not allowance.expected_value
                ):
                    raise ValueError(
                        "V2 exact-tar-member allowances require a tar report, a supported /files difference "
                        "pointer, and one non-empty literal member"
                    )
            elif not (
                allowance.match_kind == "json-pointer-count"
                and report_schema == "semantic-acceptance-summary"
                and allowance.json_pointer == "/errors"
                and isinstance(allowance.expected_value, str)
                and allowance.expected_value
            ):
                raise ValueError("V2 non-tar allowances may only match one exact semantic /errors value")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_kind": self.policy_kind,
            "baseline_id": self.baseline_id,
            "baseline_version": self.baseline_version,
            "policy_schema": self.policy_schema,
            "policy_version": self.policy_version,
            "residual_allowances": [item.to_mapping() for item in self.residual_allowances],
            "completion_exit_contracts": [item.to_mapping() for item in self.completion_exit_contracts],
            "baseline_report_bindings": [item.to_mapping() for item in self.baseline_report_bindings],
            "cross_report_reconciliations": [item.to_mapping() for item in self.cross_report_reconciliations],
        }

    @property
    def semantic_digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())

    @property
    def policy_id(self) -> str:
        return f"postprocessing-acceptance-policy-{self.semantic_digest}"


def postprocessing_acceptance_policy_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingAcceptancePolicySnapshot:
    _fields(
        payload,
        {
            "schema_version",
            "policy_kind",
            "baseline_id",
            "baseline_version",
            "policy_schema",
            "policy_version",
            "residual_allowances",
            "completion_exit_contracts",
            "baseline_report_bindings",
            "cross_report_reconciliations",
        },
        "PostprocessingAcceptancePolicySnapshot",
    )
    residuals = _mapping_list(payload, "residual_allowances")
    contracts = _mapping_list(payload, "completion_exit_contracts")
    baseline_bindings = _mapping_list(payload, "baseline_report_bindings")
    reconciliations = _mapping_list(payload, "cross_report_reconciliations")
    return PostprocessingAcceptancePolicySnapshot(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingAcceptancePolicySnapshot"
        ),
        policy_kind=_str(payload, "policy_kind"),
        baseline_id=_str(payload, "baseline_id"),
        baseline_version=_str(payload, "baseline_version"),
        policy_schema=_str(payload, "policy_schema"),
        policy_version=_str(payload, "policy_version"),
        residual_allowances=tuple(
            _residual_from_mapping(item, policy_kind=_str(payload, "policy_kind")) for item in residuals
        ),
        completion_exit_contracts=tuple(_completion_from_mapping(item) for item in contracts),
        baseline_report_bindings=tuple(_baseline_binding_from_mapping(item) for item in baseline_bindings),
        cross_report_reconciliations=tuple(_reconciliation_from_mapping(item) for item in reconciliations),
    )


def _residual_from_mapping(payload: Mapping[str, object], *, policy_kind: str) -> PostprocessingResidualAllowance:
    _fields(
        payload,
        {
            "schema_version",
            "report",
            "json_pointer",
            "match_kind",
            "expected_value",
            "cardinality_kind",
            "required_count",
            "permitted_min",
            "permitted_max",
        },
        "PostprocessingResidualAllowance",
    )
    match_kind = _str(payload, "match_kind")
    supported_match_kinds = {"json-pointer-equals", "json-pointer-count"}
    if policy_kind == "postprocessing-sealable-v2":
        supported_match_kinds.add("exact-tar-member")
    if match_kind not in supported_match_kinds:
        raise ValueError(f"unsupported residual match kind: {match_kind!r}")
    expected = payload.get("expected_value")
    if expected is None:
        raise ValueError("acceptance residual expected_value must be a non-null JSON value")
    return PostprocessingResidualAllowance(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingResidualAllowance"
        ),
        report=_str(payload, "report"),
        json_pointer=_str(payload, "json_pointer"),
        match_kind=cast("ResidualMatchKind", match_kind),
        expected_value=expected,
        cardinality_kind=cast("Literal['exact', 'permitted-range']", _str(payload, "cardinality_kind")),
        required_count=_int(payload, "required_count"),
        permitted_min=_int(payload, "permitted_min"),
        permitted_max=_int(payload, "permitted_max"),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _completion_from_mapping(payload: Mapping[str, object]) -> PostprocessingCompletionExitContract:
    _fields(
        payload,
        {
            "schema_version",
            "step_name",
            "allowed_raw_exit_codes",
            "report_paths",
            "report_schema",
            "report_schema_version",
            "outcome_report_path",
            "outcome_json_pointer",
            "raw_exit_report_outcomes",
        },
        "PostprocessingCompletionExitContract",
    )
    step = _str(payload, "step_name")
    if step not in {
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    }:
        raise ValueError(f"unsupported acceptance completion step: {step!r}")
    codes = payload.get("allowed_raw_exit_codes")
    paths = payload.get("report_paths")
    if not isinstance(codes, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in codes):
        raise ValueError("allowed_raw_exit_codes must be a list of integers")
    if not isinstance(paths, list) or any(not isinstance(item, str) for item in paths):
        raise ValueError("report_paths must be a list of strings")
    outcomes = _mapping_list(payload, "raw_exit_report_outcomes")
    parsed_outcomes: list[PostprocessingRawExitReportOutcome] = []
    for outcome in outcomes:
        _fields(outcome, {"raw_exit_code", "report_ok"}, "PostprocessingRawExitReportOutcome")
        report_ok = outcome.get("report_ok")
        if not isinstance(report_ok, bool):
            raise ValueError("raw exit report_ok must be boolean")
        parsed_outcomes.append(
            PostprocessingRawExitReportOutcome(
                raw_exit_code=_int(outcome, "raw_exit_code"),
                report_ok=report_ok,
            )
        )
    return PostprocessingCompletionExitContract(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingCompletionExitContract"
        ),
        step_name=cast("AcceptanceStepName", step),
        allowed_raw_exit_codes=tuple(codes),
        report_paths=tuple(paths),
        report_schema=_str(payload, "report_schema"),
        report_schema_version=_str(payload, "report_schema_version"),
        outcome_report_path=_str(payload, "outcome_report_path"),
        outcome_json_pointer=_str(payload, "outcome_json_pointer"),
        raw_exit_report_outcomes=tuple(parsed_outcomes),
    )


def _reconciliation_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingCrossReportReconciliation:
    _fields(
        payload,
        {
            "schema_version",
            "left_report",
            "left_json_pointer",
            "right_report",
            "right_json_pointer",
            "comparison",
        },
        "PostprocessingCrossReportReconciliation",
    )
    if _str(payload, "comparison") != "equal":
        raise ValueError("cross-report reconciliation comparison must be equal")
    return PostprocessingCrossReportReconciliation(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingCrossReportReconciliation"
        ),
        left_report=_str(payload, "left_report"),
        left_json_pointer=_str(payload, "left_json_pointer"),
        right_report=_str(payload, "right_report"),
        right_json_pointer=_str(payload, "right_json_pointer"),
    )


def _baseline_binding_from_mapping(payload: Mapping[str, object]) -> PostprocessingBaselineReportBinding:
    _fields(
        payload,
        {
            "schema_version",
            "report",
            "baseline_locator_json_pointer",
        },
        "PostprocessingBaselineReportBinding",
    )
    return PostprocessingBaselineReportBinding(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingBaselineReportBinding"
        ),
        report=_str(payload, "report"),
        baseline_locator_json_pointer=_str(payload, "baseline_locator_json_pointer"),
    )


__all__ = [
    "AcceptanceStepName",
    "PostprocessingAcceptancePolicySnapshot",
    "PostprocessingBaselineReportBinding",
    "PostprocessingCompletionExitContract",
    "PostprocessingCrossReportReconciliation",
    "PostprocessingRawExitReportOutcome",
    "PostprocessingResidualAllowance",
    "ResidualMatchKind",
    "postprocessing_acceptance_policy_from_mapping",
]
