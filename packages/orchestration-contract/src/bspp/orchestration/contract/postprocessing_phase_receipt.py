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

"""Focused postprocessing contracts extracted from postprocessing_receipt.py."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.phase import (
    canonical_mapping_digest,
    validate_phase_attempt_id,
    validate_phase_run_id,
)
from bspp.orchestration.contract.postprocessing_acceptance_adjudication import (
    PostprocessingAcceptanceAdjudication,
    postprocessing_acceptance_adjudication_from_mapping,
)
from bspp.orchestration.contract.postprocessing_acceptance_capture import (
    PostprocessingAcceptanceCapture,
    postprocessing_acceptance_capture_from_mapping,
)
from bspp.orchestration.contract.postprocessing_artifacts import (
    PostprocessingScientificOutputInventory,
    _scientific_inventory,
)
from bspp.orchestration.contract.postprocessing_attestations import (
    PostprocessingRuntimeInputAttestationSet,
    _attestation_set,
)
from bspp.orchestration.contract.postprocessing_handoff import (
    PostprocessingActionReceiptEvidence,
    PostprocessingOutputHandoff,
    _action,
    _handoff,
)
from bspp.orchestration.contract.postprocessing_phase_ids import POSTPROCESSING_ACTION_IDS
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")


_HANDOFF_ID = re.compile(r"postprocessing-handoff-[0-9a-f]{64}")


_RECEIPT_ID = re.compile(r"postprocessing-phase-receipt-[0-9a-f]{64}")


_ARTIFACT_SET_ID = re.compile(r"postprocessing-artifact-set-[0-9a-f]{64}")


_LOCATION_ID = re.compile(r"postprocessing-artifact-location-[0-9a-f]{64}")


@dataclass(frozen=True)
class PostprocessingPhaseReceipt:
    phase_receipt_id: str
    phase_run_id: str
    attempt_id: str
    phase_plan_digest: str
    phase_runspec_digest: str
    logical_input_manifest_digest: str
    scientific_identity_digest: str
    action_semantics_digest: str
    execution_projection_sha256: str
    qualified_runtime_digest: str
    runtime_input_attestations_digest: str
    scientific_output_inventory_digest: str
    scientific_output_inventory_sha256: str
    scientific_output_inventory_size_bytes: int
    acceptance_policy_id: str
    acceptance_policy_sha256: str
    acceptance_policy_size_bytes: int
    acceptance_policy_semantic_digest: str
    baseline_id: str
    baseline_version: str
    action_graph_digest: str
    terminal_action_evidence_digest: str
    acceptance_capture_digests: tuple[str, ...]
    acceptance_adjudication_digest: str
    output_handoff_id: str
    output_handoff_digest: str
    output_artifact_set_id: str
    physical_artifact_location_ids: tuple[str, ...]
    finalized_at: str
    result: Literal["passed"] = "passed"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        for value in (
            self.phase_plan_digest,
            self.phase_runspec_digest,
            self.logical_input_manifest_digest,
            self.scientific_identity_digest,
            self.action_semantics_digest,
            self.execution_projection_sha256,
            self.qualified_runtime_digest,
            self.runtime_input_attestations_digest,
            self.scientific_output_inventory_digest,
            self.scientific_output_inventory_sha256,
            self.acceptance_policy_sha256,
            self.acceptance_policy_semantic_digest,
            self.action_graph_digest,
            self.terminal_action_evidence_digest,
            self.acceptance_adjudication_digest,
            self.output_handoff_digest,
        ):
            _sha(value)
        if len(self.acceptance_capture_digests) != 3:
            raise ValueError("postprocessing receipt must bind exactly three acceptance captures")
        for value in self.acceptance_capture_digests:
            _sha(value)
        if (
            not isinstance(self.scientific_output_inventory_size_bytes, int)
            or isinstance(self.scientific_output_inventory_size_bytes, bool)
            or self.scientific_output_inventory_size_bytes <= 0
        ):
            raise ValueError("postprocessing receipt scientific inventory size must be positive")
        if (
            self.acceptance_policy_id.removeprefix("postprocessing-acceptance-policy-")
            != self.acceptance_policy_semantic_digest
            or not isinstance(self.acceptance_policy_size_bytes, int)
            or isinstance(self.acceptance_policy_size_bytes, bool)
            or self.acceptance_policy_size_bytes <= 0
            or not self.baseline_id
            or not self.baseline_version
        ):
            raise ValueError("postprocessing receipt policy and baseline binding is invalid")
        if (
            _HANDOFF_ID.fullmatch(self.output_handoff_id) is None
            or _ARTIFACT_SET_ID.fullmatch(self.output_artifact_set_id) is None
            or not self.physical_artifact_location_ids
            or self.physical_artifact_location_ids != tuple(sorted(self.physical_artifact_location_ids))
            or len(set(self.physical_artifact_location_ids)) != len(self.physical_artifact_location_ids)
            or any(_LOCATION_ID.fullmatch(item) is None for item in self.physical_artifact_location_ids)
            or self.result != "passed"
            or not self.finalized_at
        ):
            raise ValueError("postprocessing receipt requires a passed handoff and timestamp")
        if _RECEIPT_ID.fullmatch(self.phase_receipt_id) is None or self.phase_receipt_id != postprocessing_receipt_id(
            self.identity_mapping()
        ):
            raise ValueError("postprocessing Phase Receipt id differs from its identity")

    def identity_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_plan_digest": self.phase_plan_digest,
            "phase_runspec_digest": self.phase_runspec_digest,
            "logical_input_manifest_digest": self.logical_input_manifest_digest,
            "scientific_identity_digest": self.scientific_identity_digest,
            "action_semantics_digest": self.action_semantics_digest,
            "execution_projection_sha256": self.execution_projection_sha256,
            "qualified_runtime_digest": self.qualified_runtime_digest,
            "runtime_input_attestations_digest": self.runtime_input_attestations_digest,
            "scientific_output_inventory_digest": self.scientific_output_inventory_digest,
            "scientific_output_inventory_sha256": self.scientific_output_inventory_sha256,
            "scientific_output_inventory_size_bytes": self.scientific_output_inventory_size_bytes,
            "acceptance_policy_id": self.acceptance_policy_id,
            "acceptance_policy_sha256": self.acceptance_policy_sha256,
            "acceptance_policy_size_bytes": self.acceptance_policy_size_bytes,
            "acceptance_policy_semantic_digest": self.acceptance_policy_semantic_digest,
            "baseline_id": self.baseline_id,
            "baseline_version": self.baseline_version,
            "action_graph_digest": self.action_graph_digest,
            "terminal_action_evidence_digest": self.terminal_action_evidence_digest,
            "acceptance_capture_digests": list(self.acceptance_capture_digests),
            "acceptance_adjudication_digest": self.acceptance_adjudication_digest,
            "output_handoff_id": self.output_handoff_id,
            "output_handoff_digest": self.output_handoff_digest,
            "output_artifact_set_id": self.output_artifact_set_id,
            "physical_artifact_location_ids": list(self.physical_artifact_location_ids),
            "finalized_at": self.finalized_at,
            "result": self.result,
        }

    def to_mapping(self) -> dict[str, object]:
        return {"phase_receipt_id": self.phase_receipt_id, **self.identity_mapping()}


def postprocessing_receipt_id(payload: Mapping[str, object]) -> str:
    return "postprocessing-phase-receipt-" + canonical_mapping_digest(payload)


@dataclass(frozen=True)
class PostprocessingFinalizedPayload:
    terminal_actions: tuple[PostprocessingActionReceiptEvidence, ...]
    runtime_input_attestations: PostprocessingRuntimeInputAttestationSet
    scientific_output_inventory: PostprocessingScientificOutputInventory
    acceptance_captures: tuple[PostprocessingAcceptanceCapture, ...]
    acceptance_adjudication: PostprocessingAcceptanceAdjudication
    output_handoff: PostprocessingOutputHandoff
    receipt: PostprocessingPhaseReceipt
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        receipt = self.receipt
        permanent_ids = tuple(POSTPROCESSING_ACTION_IDS.values())
        action_ids = tuple(item.action_id for item in self.terminal_actions)
        if (
            not action_ids
            or len(set(action_ids)) != len(action_ids)
            or tuple(sorted(action_ids, key=permanent_ids.index)) != action_ids
            or action_ids[-1] != POSTPROCESSING_ACTION_IDS["acceptance-adjudication"]
        ):
            raise ValueError("postprocessing finalized actions must be unique permanent action order")
        capture_steps = tuple(item.step_name for item in self.acceptance_captures)
        if capture_steps != (
            "acceptance-tar-payload-parity",
            "acceptance-semantic",
            "acceptance-verify-evidence",
        ):
            raise ValueError("postprocessing finalized payload requires the exact three capture steps")
        if (
            receipt.phase_run_id != self.acceptance_adjudication.phase_run_id
            or receipt.attempt_id != self.acceptance_adjudication.attempt_id
            or any(
                (capture.phase_run_id, capture.attempt_id) != (receipt.phase_run_id, receipt.attempt_id)
                for capture in self.acceptance_captures
            )
            or (self.output_handoff.phase_run_id, self.output_handoff.attempt_id)
            != (receipt.phase_run_id, receipt.attempt_id)
            or (
                self.runtime_input_attestations.phase_run_id,
                self.runtime_input_attestations.attempt_id,
            )
            != (receipt.phase_run_id, receipt.attempt_id)
            or (
                self.scientific_output_inventory.phase_run_id,
                self.scientific_output_inventory.attempt_id,
            )
            != (receipt.phase_run_id, receipt.attempt_id)
            or self.scientific_output_inventory.phase_runspec_digest != receipt.phase_runspec_digest
        ):
            raise ValueError("postprocessing finalized records must bind one run and Attempt")
        capture_digests = tuple(item.digest for item in self.acceptance_captures)
        action_digest = canonical_mapping_digest(
            {"schema_version": 1, "terminal_actions": [item.to_mapping() for item in self.terminal_actions]}
        )
        handoff_digest = canonical_mapping_digest(self.output_handoff.to_mapping())
        inventory_bytes = _canonical_bytes(self.scientific_output_inventory.to_mapping())
        if (
            self.acceptance_adjudication.result != "passed"
            or self.acceptance_adjudication.capture_digests != capture_digests
            or receipt.acceptance_capture_digests != capture_digests
            or receipt.acceptance_adjudication_digest != self.acceptance_adjudication.digest
            or receipt.runtime_input_attestations_digest != self.runtime_input_attestations.digest
            or receipt.scientific_output_inventory_digest != self.scientific_output_inventory.digest
            or receipt.scientific_output_inventory_sha256 != hashlib.sha256(inventory_bytes).hexdigest()
            or receipt.scientific_output_inventory_size_bytes != len(inventory_bytes)
            or receipt.terminal_action_evidence_digest != action_digest
            or receipt.output_handoff_id != self.output_handoff.handoff_id
            or receipt.output_handoff_digest != handoff_digest
            or receipt.output_artifact_set_id != self.output_handoff.artifact_set.artifact_set_id
            or receipt.physical_artifact_location_ids
            != tuple(item.artifact_location_id for item in self.output_handoff.physical_locations)
        ):
            raise ValueError("postprocessing Phase Receipt does not bind exact successful finalization evidence")
        root_set = self.output_handoff.artifact_set
        children = {item.root_name: item for item in root_set.children}
        if root_set.root_name != "postprocessing-output" or set(children) != {
            "acceptance-evidence",
            "scientific-output",
        }:
            raise ValueError("postprocessing output Artifact Set must separate evidence from scientific output")
        evidence_set = children["acceptance-evidence"]
        scientific_set = children["scientific-output"]
        if not evidence_set.members or not scientific_set.members:
            raise ValueError("postprocessing output Artifact Set leaves must both be non-empty")
        if scientific_set.members != self.scientific_output_inventory.members:
            raise ValueError("scientific output Artifact Set differs from its authoritative inventory")
        artifacts = {item.path: item for item in evidence_set.members}
        required: dict[str, tuple[str, int]] = {}
        for capture in self.acceptance_captures:
            capture_path = f"phase-acceptance/{capture.step_name}-capture.json"
            capture_bytes = _canonical_bytes(capture.to_mapping())
            required[capture_path] = (hashlib.sha256(capture_bytes).hexdigest(), len(capture_bytes))
            for binding in (capture.raw_stdout, capture.raw_stderr, *capture.reports):
                required[binding.path] = (binding.sha256, binding.size_bytes)
        adjudication_bytes = _canonical_bytes(self.acceptance_adjudication.to_mapping())
        required["phase-acceptance/adjudication.json"] = (
            hashlib.sha256(adjudication_bytes).hexdigest(),
            len(adjudication_bytes),
        )
        if set(artifacts) != set(required) or any(
            artifacts[path].verification_kind != "content-sha256-v1"
            or (artifacts[path].sha256, artifacts[path].size_bytes) != identity
            for path, identity in required.items()
        ):
            raise ValueError("postprocessing output handoff does not bind the exact acceptance evidence inventory")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "terminal_actions": [item.to_mapping() for item in self.terminal_actions],
            "runtime_input_attestations": self.runtime_input_attestations.to_mapping(),
            "scientific_output_inventory": self.scientific_output_inventory.to_mapping(),
            "acceptance_captures": [item.to_mapping() for item in self.acceptance_captures],
            "acceptance_adjudication": self.acceptance_adjudication.to_mapping(),
            "output_handoff": self.output_handoff.to_mapping(),
            "receipt": self.receipt.to_mapping(),
        }


def postprocessing_finalized_payload_from_mapping(payload: Mapping[str, object]) -> PostprocessingFinalizedPayload:
    _fields(
        payload,
        {
            "schema_version",
            "terminal_actions",
            "runtime_input_attestations",
            "scientific_output_inventory",
            "acceptance_captures",
            "acceptance_adjudication",
            "output_handoff",
            "receipt",
        },
        "PostprocessingFinalizedPayload",
    )
    return PostprocessingFinalizedPayload(
        schema_version=_schema(payload, "PostprocessingFinalizedPayload"),
        terminal_actions=tuple(_action(item) for item in _mapping_list(payload, "terminal_actions")),
        runtime_input_attestations=_attestation_set(_mapping(payload, "runtime_input_attestations")),
        scientific_output_inventory=_scientific_inventory(_mapping(payload, "scientific_output_inventory")),
        acceptance_captures=tuple(
            postprocessing_acceptance_capture_from_mapping(item)
            for item in _mapping_list(payload, "acceptance_captures")
        ),
        acceptance_adjudication=postprocessing_acceptance_adjudication_from_mapping(
            _mapping(payload, "acceptance_adjudication")
        ),
        output_handoff=_handoff(_mapping(payload, "output_handoff")),
        receipt=_receipt(_mapping(payload, "receipt")),
    )


def _receipt(payload: Mapping[str, object]) -> PostprocessingPhaseReceipt:
    expected = set(PostprocessingPhaseReceipt.__dataclass_fields__) | {"phase_receipt_id"}
    _fields(payload, expected, "PostprocessingPhaseReceipt")
    captures = payload.get("acceptance_capture_digests")
    if not isinstance(captures, list) or any(not isinstance(item, str) for item in captures):
        raise ValueError("postprocessing receipt capture digests must be strings")
    if payload.get("result") != "passed":
        raise ValueError("postprocessing receipt result must be passed")
    return PostprocessingPhaseReceipt(
        schema_version=_schema(payload, "PostprocessingPhaseReceipt"),
        phase_receipt_id=_string(payload, "phase_receipt_id"),
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        phase_plan_digest=_string(payload, "phase_plan_digest"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        logical_input_manifest_digest=_string(payload, "logical_input_manifest_digest"),
        scientific_identity_digest=_string(payload, "scientific_identity_digest"),
        action_semantics_digest=_string(payload, "action_semantics_digest"),
        execution_projection_sha256=_string(payload, "execution_projection_sha256"),
        qualified_runtime_digest=_string(payload, "qualified_runtime_digest"),
        runtime_input_attestations_digest=_string(payload, "runtime_input_attestations_digest"),
        scientific_output_inventory_digest=_string(payload, "scientific_output_inventory_digest"),
        scientific_output_inventory_sha256=_string(payload, "scientific_output_inventory_sha256"),
        scientific_output_inventory_size_bytes=_integer(payload, "scientific_output_inventory_size_bytes"),
        acceptance_policy_id=_string(payload, "acceptance_policy_id"),
        acceptance_policy_sha256=_string(payload, "acceptance_policy_sha256"),
        acceptance_policy_size_bytes=_integer(payload, "acceptance_policy_size_bytes"),
        acceptance_policy_semantic_digest=_string(payload, "acceptance_policy_semantic_digest"),
        baseline_id=_string(payload, "baseline_id"),
        baseline_version=_string(payload, "baseline_version"),
        action_graph_digest=_string(payload, "action_graph_digest"),
        terminal_action_evidence_digest=_string(payload, "terminal_action_evidence_digest"),
        acceptance_capture_digests=tuple(captures),
        acceptance_adjudication_digest=_string(payload, "acceptance_adjudication_digest"),
        output_handoff_id=_string(payload, "output_handoff_id"),
        output_handoff_digest=_string(payload, "output_handoff_digest"),
        output_artifact_set_id=_string(payload, "output_artifact_set_id"),
        physical_artifact_location_ids=_string_list(payload, "physical_artifact_location_ids"),
        finalized_at=_string(payload, "finalized_at"),
        result="passed",
    )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _mapping_list(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of mappings")
    return tuple(cast("Mapping[str, object]", item) for item in value)


def _fields(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} has missing or extra fields")


def _schema(payload_or_version: Mapping[str, object] | int, name: str) -> int:
    value = payload_or_version.get("schema_version") if isinstance(payload_or_version, Mapping) else payload_or_version
    return validate_schema_version(value, record_name=name)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _string_list(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return tuple(value)


def _sha(value: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError("postprocessing receipt digest must be lowercase SHA-256")


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


__all__ = [
    "PostprocessingFinalizedPayload",
    "PostprocessingPhaseReceipt",
    "postprocessing_finalized_payload_from_mapping",
    "postprocessing_receipt_id",
]
