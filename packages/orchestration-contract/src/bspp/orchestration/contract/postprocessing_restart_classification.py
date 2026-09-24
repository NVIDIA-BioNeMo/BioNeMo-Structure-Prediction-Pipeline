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

"""Per-restart postprocessing classification evidence records.

One immutable record is written per failed Runtime incarnation of a group-A
transport failure.  The record is keyed by action/task plus a validated restart
ordinal, binds the frozen RunSpec and action identity, and carries the closed
group-A classification.  It never satisfies the success record.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.phase import validate_phase_attempt_id, validate_phase_run_id
from bspp.orchestration.contract.postprocessing_failure_classification import (
    PostprocessingFailureClassification,
    postprocessing_failure_classification_from_mapping,
)
from bspp.orchestration.contract.postprocessing_phase_ids import POSTPROCESSING_ACTION_IDS
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_KIND: Literal["postprocessing-restart-classification-evidence-v1"] = (
    "postprocessing-restart-classification-evidence-v1"
)


@dataclass(frozen=True)
class PostprocessingRestartClassificationEvidence:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_graph_digest: str
    action_id: str
    runtime_action_digest: str
    task_index: int | None
    restart_ordinal: int
    classification: PostprocessingFailureClassification
    classified_at: str
    evidence_kind: Literal["postprocessing-restart-classification-evidence-v1"] = _EVIDENCE_KIND
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name=type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        for value in (self.phase_runspec_digest, self.action_graph_digest, self.runtime_action_digest):
            _sha(value, "postprocessing restart classification digest")
        if self.evidence_kind != _EVIDENCE_KIND:
            raise ValueError("unsupported postprocessing restart classification evidence discriminator")
        if self.action_id not in tuple(POSTPROCESSING_ACTION_IDS.values())[:-1]:
            raise ValueError("postprocessing restart classification evidence may only describe Actions 01--08")
        if self.task_index is not None and (
            not isinstance(self.task_index, int) or isinstance(self.task_index, bool) or self.task_index < 0
        ):
            raise ValueError("postprocessing restart classification task index must be non-negative or null")
        if (
            not isinstance(self.restart_ordinal, int)
            or isinstance(self.restart_ordinal, bool)
            or self.restart_ordinal < 0
        ):
            raise ValueError("postprocessing restart ordinal must be a non-negative integer")
        if self.classification.group != "A":
            raise ValueError("postprocessing restart classification evidence requires a group-A classification")
        if not self.classified_at:
            raise ValueError("postprocessing restart classification classified_at must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evidence_kind": self.evidence_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_graph_digest": self.action_graph_digest,
            "action_id": self.action_id,
            "runtime_action_digest": self.runtime_action_digest,
            "task_index": self.task_index,
            "restart_ordinal": self.restart_ordinal,
            "classification": self.classification.to_mapping(),
            "classified_at": self.classified_at,
        }


def postprocessing_restart_classification_evidence_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingRestartClassificationEvidence:
    _fields(
        payload,
        {
            "schema_version",
            "evidence_kind",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "action_graph_digest",
            "action_id",
            "runtime_action_digest",
            "task_index",
            "restart_ordinal",
            "classification",
            "classified_at",
        },
        "PostprocessingRestartClassificationEvidence",
    )
    if _string(payload, "evidence_kind") != _EVIDENCE_KIND:
        raise ValueError("unsupported postprocessing restart classification evidence discriminator")
    task_index = payload.get("task_index")
    if task_index is not None and (not isinstance(task_index, int) or isinstance(task_index, bool)):
        raise ValueError("postprocessing restart classification task_index must be an integer or null")
    return PostprocessingRestartClassificationEvidence(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingRestartClassificationEvidence"
        ),
        evidence_kind=_EVIDENCE_KIND,
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        action_graph_digest=_string(payload, "action_graph_digest"),
        action_id=_string(payload, "action_id"),
        runtime_action_digest=_string(payload, "runtime_action_digest"),
        task_index=task_index,
        restart_ordinal=_int(payload, "restart_ordinal"),
        classification=postprocessing_failure_classification_from_mapping(_mapping(payload, "classification")),
        classified_at=_string(payload, "classified_at"),
    )


def _sha(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")


def _fields(payload: Mapping[str, object], allowed: set[str], record: str) -> None:
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown or missing:
        raise ValueError(f"{record} fields differ; missing={missing!r}, unknown={unknown!r}")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return cast("Mapping[str, object]", value)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


__all__ = [
    "PostprocessingRestartClassificationEvidence",
    "postprocessing_restart_classification_evidence_from_mapping",
]
