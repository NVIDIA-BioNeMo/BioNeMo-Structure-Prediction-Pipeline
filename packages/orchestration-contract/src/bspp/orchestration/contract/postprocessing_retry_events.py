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

"""Focused postprocessing contracts extracted from postprocessing_lifecycle.py."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.phase import canonical_mapping_digest, validate_phase_attempt_id
from bspp.orchestration.contract.postprocessing_runspec import (
    ReadablePostprocessingPhaseRunSpec,
    read_postprocessing_phase_runspec_from_mapping,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")


_RETRY_ID = re.compile(r"postprocessing-retry-[0-9a-f]{64}")


@dataclass(frozen=True)
class PostprocessingAttemptRetriedPayload:
    """Complete successor authority; postprocessing Retry never adopts predecessor output."""

    retry_id: str
    predecessor_attempt_id: str
    predecessor_phase_runspec_digest: str
    predecessor_outcome: Literal["failed", "cancelled"]
    phase_plan_digest: str
    logical_input_manifest_digest: str
    scientific_identity_digest: str
    acceptance_policy_semantic_digest: str
    action_graph_digest: str
    action_semantics_digest: str
    successor_phase_runspec: ReadablePostprocessingPhaseRunSpec
    successor_legacy_runspec_yaml: str
    successor_runtime_qualification_json: str
    carry_forward: None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version, type(self).__name__)
        if _RETRY_ID.fullmatch(self.retry_id) is None:
            raise ValueError("postprocessing retry id is invalid")
        validate_phase_attempt_id(self.predecessor_attempt_id)
        _sha(self.predecessor_phase_runspec_digest)
        for value in (
            self.phase_plan_digest,
            self.logical_input_manifest_digest,
            self.scientific_identity_digest,
            self.acceptance_policy_semantic_digest,
            self.action_graph_digest,
            self.action_semantics_digest,
        ):
            _sha(value)
        if self.predecessor_outcome not in {"failed", "cancelled"} or self.carry_forward is not None:
            raise ValueError("postprocessing Retry requires a terminal predecessor and no carry-forward")
        predecessor_ordinal = int(self.predecessor_attempt_id.removeprefix("attempt-"))
        successor_ordinal = int(self.successor_phase_runspec.attempt_id.removeprefix("attempt-"))
        if successor_ordinal != predecessor_ordinal + 1:
            raise ValueError("postprocessing Retry Attempt ordinals must be contiguous")
        if (
            self.successor_phase_runspec.phase_plan_digest != self.phase_plan_digest
            or self.successor_phase_runspec.payload.logical_inputs.digest != self.logical_input_manifest_digest
            or self.successor_phase_runspec.payload.scientific_identity.digest != self.scientific_identity_digest
            or self.successor_phase_runspec.payload.acceptance_policy.semantic_digest
            != self.acceptance_policy_semantic_digest
            or self.successor_phase_runspec.payload.action_graph_digest != self.action_graph_digest
            or self.successor_phase_runspec.payload.action_semantics_digest != self.action_semantics_digest
        ):
            raise ValueError("postprocessing Retry invariants differ from successor authority")
        projection = self.successor_phase_runspec.payload.execution_projection
        rendered = self.successor_legacy_runspec_yaml.encode()
        if (
            not rendered
            or len(rendered) != projection.document_size_bytes
            or hashlib.sha256(rendered).hexdigest() != projection.document_sha256
        ):
            raise ValueError("postprocessing Retry legacy RunSpec bytes differ from successor projection")
        qualification = self.successor_runtime_qualification_json.encode()
        qualified = self.successor_phase_runspec.payload.qualified_runtime
        try:
            qualification_value = json.loads(qualification)
        except json.JSONDecodeError as exc:
            raise ValueError("postprocessing Retry Runtime Qualification is not JSON") from exc
        if (
            not isinstance(qualification_value, Mapping)
            or qualification != (json.dumps(qualification_value, indent=2, sort_keys=True) + "\n").encode()
            or len(qualification) != qualified.qualification_size_bytes
            or hashlib.sha256(qualification).hexdigest() != qualified.qualification_sha256
        ):
            raise ValueError("postprocessing Retry Runtime Qualification bytes differ from successor authority")
        if self.retry_id != postprocessing_retry_id(
            predecessor_attempt_id=self.predecessor_attempt_id,
            predecessor_phase_runspec_digest=self.predecessor_phase_runspec_digest,
            predecessor_outcome=self.predecessor_outcome,
            successor_phase_runspec=self.successor_phase_runspec,
        ):
            raise ValueError("postprocessing retry id differs from its identity preimage")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "retry_id": self.retry_id,
            "predecessor_attempt_id": self.predecessor_attempt_id,
            "predecessor_phase_runspec_digest": self.predecessor_phase_runspec_digest,
            "predecessor_outcome": self.predecessor_outcome,
            "phase_plan_digest": self.phase_plan_digest,
            "logical_input_manifest_digest": self.logical_input_manifest_digest,
            "scientific_identity_digest": self.scientific_identity_digest,
            "acceptance_policy_semantic_digest": self.acceptance_policy_semantic_digest,
            "action_graph_digest": self.action_graph_digest,
            "action_semantics_digest": self.action_semantics_digest,
            "successor_phase_runspec": self.successor_phase_runspec.to_mapping(),
            "successor_legacy_runspec_yaml": self.successor_legacy_runspec_yaml,
            "successor_runtime_qualification_json": self.successor_runtime_qualification_json,
            "carry_forward": self.carry_forward,
        }


def postprocessing_retry_id(
    *,
    predecessor_attempt_id: str,
    predecessor_phase_runspec_digest: str,
    predecessor_outcome: Literal["failed", "cancelled"],
    successor_phase_runspec: ReadablePostprocessingPhaseRunSpec,
) -> str:
    return "postprocessing-retry-" + canonical_mapping_digest(
        {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "phase_kind": "postprocessing",
            "predecessor_attempt_id": predecessor_attempt_id,
            "predecessor_phase_runspec_digest": predecessor_phase_runspec_digest,
            "predecessor_outcome": predecessor_outcome,
            "successor_phase_runspec_digest": successor_phase_runspec.digest,
        }
    )


def _retried(payload: Mapping[str, object]) -> PostprocessingAttemptRetriedPayload:
    _fields(
        payload,
        {
            "schema_version",
            "retry_id",
            "predecessor_attempt_id",
            "predecessor_phase_runspec_digest",
            "predecessor_outcome",
            "phase_plan_digest",
            "logical_input_manifest_digest",
            "scientific_identity_digest",
            "acceptance_policy_semantic_digest",
            "action_graph_digest",
            "action_semantics_digest",
            "successor_phase_runspec",
            "successor_legacy_runspec_yaml",
            "successor_runtime_qualification_json",
            "carry_forward",
        },
        "PostprocessingAttemptRetriedPayload",
    )
    predecessor_outcome = _string(payload, "predecessor_outcome")
    if predecessor_outcome not in {"failed", "cancelled"}:
        raise ValueError("postprocessing Retry predecessor outcome is invalid")
    successor = payload.get("successor_phase_runspec")
    if not isinstance(successor, Mapping):
        raise ValueError("postprocessing Retry successor RunSpec must be a mapping")
    if payload.get("carry_forward") is not None:
        raise ValueError("postprocessing Retry does not support carry-forward")
    return PostprocessingAttemptRetriedPayload(
        schema_version=_schema(payload, "PostprocessingAttemptRetriedPayload"),
        retry_id=_string(payload, "retry_id"),
        predecessor_attempt_id=_string(payload, "predecessor_attempt_id"),
        predecessor_phase_runspec_digest=_string(payload, "predecessor_phase_runspec_digest"),
        predecessor_outcome=cast("Literal['failed', 'cancelled']", predecessor_outcome),
        phase_plan_digest=_string(payload, "phase_plan_digest"),
        logical_input_manifest_digest=_string(payload, "logical_input_manifest_digest"),
        scientific_identity_digest=_string(payload, "scientific_identity_digest"),
        acceptance_policy_semantic_digest=_string(payload, "acceptance_policy_semantic_digest"),
        action_graph_digest=_string(payload, "action_graph_digest"),
        action_semantics_digest=_string(payload, "action_semantics_digest"),
        successor_phase_runspec=read_postprocessing_phase_runspec_from_mapping(successor),
        successor_legacy_runspec_yaml=_text(payload, "successor_legacy_runspec_yaml"),
        successor_runtime_qualification_json=_text(payload, "successor_runtime_qualification_json"),
        carry_forward=None,
    )


def _version(value: int, name: str) -> None:
    validate_schema_version(value, record_name=name)


def _sha(value: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError("postprocessing lifecycle digest must be lowercase SHA-256")


def _fields(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} has missing or extra fields")


def _schema(payload: Mapping[str, object], name: str) -> int:
    return validate_schema_version(payload.get("schema_version"), record_name=name)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


__all__ = ["PostprocessingAttemptRetriedPayload", "postprocessing_retry_id"]
