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

"""Final postprocessing acceptance adjudication contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract._postprocessing_validation import _fields, _int, _mapping_list, _schema, _str
from bspp.orchestration.contract.phase import (
    canonical_mapping_digest,
    validate_phase_attempt_id,
    validate_phase_run_id,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")
_POLICY_ID = re.compile(r"postprocessing-acceptance-policy-[0-9a-f]{64}")


@dataclass(frozen=True)
class PostprocessingResidualCardinality:
    allowance_id: str
    observed: int
    permitted_min: int
    permitted_max: int

    def __post_init__(self) -> None:
        if not self.allowance_id or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (self.observed, self.permitted_min, self.permitted_max)
        ):
            raise ValueError("postprocessing adjudicated residual cardinality is invalid")

    @property
    def accepted(self) -> bool:
        return self.permitted_min <= self.observed <= self.permitted_max

    def to_mapping(self) -> dict[str, object]:
        return {
            "allowance_id": self.allowance_id,
            "observed": self.observed,
            "permitted_min": self.permitted_min,
            "permitted_max": self.permitted_max,
        }


@dataclass(frozen=True)
class PostprocessingReconciliationResult:
    reconciliation_id: str
    matched: bool

    def __post_init__(self) -> None:
        if not self.reconciliation_id or not isinstance(self.matched, bool):
            raise ValueError("postprocessing adjudicated reconciliation result is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {"reconciliation_id": self.reconciliation_id, "matched": self.matched}


@dataclass(frozen=True)
class PostprocessingAcceptanceAdjudication:
    """The sole acceptance decision over three coherent raw captures."""

    phase_run_id: str
    attempt_id: str
    policy_id: str
    policy_sha256: str
    capture_digests: tuple[str, ...]
    non_allowlisted_errors: int
    residual_cardinalities: tuple[PostprocessingResidualCardinality, ...]
    reconciliation_results: tuple[PostprocessingReconciliationResult, ...]
    adjudicated_at: str
    result: Literal["passed", "failed"]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        if _POLICY_ID.fullmatch(self.policy_id) is None or _SHA256.fullmatch(self.policy_sha256) is None:
            raise ValueError("acceptance adjudication policy identity is invalid")
        if len(self.capture_digests) != 3 or any(_SHA256.fullmatch(item) is None for item in self.capture_digests):
            raise ValueError("acceptance adjudication must bind exactly three capture digests")
        if (
            not isinstance(self.non_allowlisted_errors, int)
            or isinstance(self.non_allowlisted_errors, bool)
            or self.non_allowlisted_errors < 0
        ):
            raise ValueError("acceptance adjudication non-allowlisted error count must be non-negative")
        residual_ids = tuple(item.allowance_id for item in self.residual_cardinalities)
        if residual_ids != tuple(sorted(residual_ids)) or len(set(residual_ids)) != len(residual_ids):
            raise ValueError("acceptance adjudication residual cardinalities must be identity-sorted")
        reconciliation_ids = tuple(item.reconciliation_id for item in self.reconciliation_results)
        if reconciliation_ids != tuple(sorted(reconciliation_ids)) or len(set(reconciliation_ids)) != len(
            reconciliation_ids
        ):
            raise ValueError("acceptance adjudication reconciliation results must be identity-sorted")
        residuals_ok = all(item.accepted for item in self.residual_cardinalities)
        reconciled = bool(self.reconciliation_results) and all(item.matched for item in self.reconciliation_results)
        passed = self.non_allowlisted_errors == 0 and residuals_ok and reconciled
        if (self.result == "passed") != passed:
            raise ValueError("acceptance adjudication result does not match its exact counts")
        if not self.adjudicated_at:
            raise ValueError("acceptance adjudication timestamp must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "capture_digests": list(self.capture_digests),
            "non_allowlisted_errors": self.non_allowlisted_errors,
            "residual_cardinalities": [item.to_mapping() for item in self.residual_cardinalities],
            "reconciliation_results": [item.to_mapping() for item in self.reconciliation_results],
            "adjudicated_at": self.adjudicated_at,
            "result": self.result,
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


def postprocessing_acceptance_adjudication_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingAcceptanceAdjudication:
    _fields(
        payload,
        {
            "schema_version",
            "phase_run_id",
            "attempt_id",
            "policy_id",
            "policy_sha256",
            "capture_digests",
            "non_allowlisted_errors",
            "residual_cardinalities",
            "reconciliation_results",
            "adjudicated_at",
            "result",
        },
        "PostprocessingAcceptanceAdjudication",
    )
    capture_digests = payload.get("capture_digests")
    if not isinstance(capture_digests, list) or any(not isinstance(item, str) for item in capture_digests):
        raise ValueError("capture_digests must be a list of strings")
    residuals: list[PostprocessingResidualCardinality] = []
    for item in _mapping_list(payload, "residual_cardinalities"):
        _fields(
            item,
            {"allowance_id", "observed", "permitted_min", "permitted_max"},
            "PostprocessingAdjudicatedResidual",
        )
        residuals.append(
            PostprocessingResidualCardinality(
                allowance_id=_str(item, "allowance_id"),
                observed=_int(item, "observed"),
                permitted_min=_int(item, "permitted_min"),
                permitted_max=_int(item, "permitted_max"),
            )
        )
    reconciliations: list[PostprocessingReconciliationResult] = []
    for item in _mapping_list(payload, "reconciliation_results"):
        _fields(item, {"reconciliation_id", "matched"}, "PostprocessingAdjudicatedReconciliation")
        matched = item.get("matched")
        if not isinstance(matched, bool):
            raise ValueError("adjudicated reconciliation matched must be boolean")
        reconciliations.append(
            PostprocessingReconciliationResult(
                reconciliation_id=_str(item, "reconciliation_id"),
                matched=matched,
            )
        )
    result = _str(payload, "result")
    if result not in {"passed", "failed"}:
        raise ValueError("postprocessing acceptance adjudication result must be passed or failed")
    return PostprocessingAcceptanceAdjudication(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingAcceptanceAdjudication"
        ),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        policy_id=_str(payload, "policy_id"),
        policy_sha256=_str(payload, "policy_sha256"),
        capture_digests=tuple(capture_digests),
        non_allowlisted_errors=_int(payload, "non_allowlisted_errors"),
        residual_cardinalities=tuple(residuals),
        reconciliation_results=tuple(reconciliations),
        adjudicated_at=_str(payload, "adjudicated_at"),
        result=cast("Literal['passed', 'failed']", result),
    )


__all__ = [
    "PostprocessingAcceptanceAdjudication",
    "PostprocessingReconciliationResult",
    "PostprocessingResidualCardinality",
    "postprocessing_acceptance_adjudication_from_mapping",
]
