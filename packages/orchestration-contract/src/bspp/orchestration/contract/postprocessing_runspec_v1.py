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

"""Focused postprocessing contracts extracted from phase_postprocessing.py."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract._postprocessing_validation import (
    _fields,
    _int,
    _mapping,
    _mapping_list,
    _sha,
    _str,
)
from bspp.orchestration.contract.phase import (
    canonical_mapping_digest,
    validate_phase_attempt_id,
    validate_phase_run_id,
)
from bspp.orchestration.contract.postprocessing_acceptance_reference import (
    PostprocessingAcceptanceSnapshotReference,
)
from bspp.orchestration.contract.postprocessing_action_contract import (
    PostprocessingRuntimeAction,
    postprocessing_action_graph_digest,
)
from bspp.orchestration.contract.postprocessing_execution import (
    PostprocessingAttemptPaths,
    PostprocessingClusterSnapshot,
    QualifiedPostprocessingRuntimeSelection,
)
from bspp.orchestration.contract.postprocessing_execution_parsing import (
    postprocessing_acceptance_reference_from_mapping,
    postprocessing_attempt_paths_from_mapping,
    qualified_postprocessing_runtime_from_mapping,
)
from bspp.orchestration.contract.postprocessing_logical_identity import (
    LogicalInputManifest,
    PhysicalInputLocator,
    PostprocessingScientificIdentityV1,
)
from bspp.orchestration.contract.postprocessing_runspec_parsing import (
    logical_input_entry_from_mapping,
    physical_input_locator_from_mapping,
    postprocessing_runtime_action_from_mapping,
)
from bspp.orchestration.contract.versioning import validate_schema_version


@dataclass(frozen=True)
class HistoricalPostprocessingPhaseExecutionIdentityV1:
    phase_plan_digest: str
    logical_input_manifest_digest: str
    scientific_identity_digest: str
    phase_run_id: str
    attempt_id: str
    qualified_runtime_digest: str
    output_namespace: str
    substitutions: PostprocessingAttemptPaths
    identity_kind: Literal["postprocessing-execution-projection-v1"] = "postprocessing-execution-projection-v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name=type(self).__name__)
        for value in (
            self.phase_plan_digest,
            self.logical_input_manifest_digest,
            self.scientific_identity_digest,
            self.qualified_runtime_digest,
        ):
            _sha(value, "historical postprocessing execution identity digest")
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        if self.identity_kind != "postprocessing-execution-projection-v1" or not self.output_namespace:
            raise ValueError("historical postprocessing V1 execution identity is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity_kind": self.identity_kind,
            "phase_plan_digest": self.phase_plan_digest,
            "logical_input_manifest_digest": self.logical_input_manifest_digest,
            "scientific_identity_digest": self.scientific_identity_digest,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "qualified_runtime_digest": self.qualified_runtime_digest,
            "output_namespace": self.output_namespace,
            "substitutions": self.substitutions.to_mapping(),
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class HistoricalPostprocessingExecutionProjectionV1:
    document_location: str
    document_sha256: str
    document_size_bytes: int
    legacy_schema_version: int
    phase_identity: HistoricalPostprocessingPhaseExecutionIdentityV1
    phase_identity_digest: str
    projection_kind: Literal["legacy-runspec-yaml-v1"] = "legacy-runspec-yaml-v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name=type(self).__name__)
        if (
            self.projection_kind != "legacy-runspec-yaml-v1"
            or self.legacy_schema_version != 1
            or not self.document_location.startswith("attempts/")
            or not self.document_location.endswith("/legacy-runspec.yaml")
            or self.document_size_bytes <= 0
        ):
            raise ValueError("historical postprocessing V1 execution projection is invalid")
        _sha(self.document_sha256, "historical postprocessing legacy RunSpec SHA-256")
        _sha(self.phase_identity_digest, "historical postprocessing phase identity digest")
        if self.phase_identity_digest != self.phase_identity.digest:
            raise ValueError("historical postprocessing V1 execution identity digest differs")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "projection_kind": self.projection_kind,
            "legacy_schema_version": self.legacy_schema_version,
            "document_location": self.document_location,
            "document_sha256": self.document_sha256,
            "document_size_bytes": self.document_size_bytes,
            "phase_identity_digest": self.phase_identity_digest,
            "phase_identity": self.phase_identity.to_mapping(),
        }


@dataclass(frozen=True)
class HistoricalPostprocessingPhaseRunSpecPayloadV1:
    actions: tuple[PostprocessingRuntimeAction, ...]
    action_graph_digest: str
    action_semantics_digest: str
    scientific_identity: PostprocessingScientificIdentityV1
    logical_inputs: LogicalInputManifest
    physical_inputs: tuple[PhysicalInputLocator, ...]
    execution_projection: HistoricalPostprocessingExecutionProjectionV1
    acceptance_policy: PostprocessingAcceptanceSnapshotReference
    qualified_runtime: QualifiedPostprocessingRuntimeSelection
    attempt_paths: PostprocessingAttemptPaths
    phase_kind: Literal["postprocessing"] = "postprocessing"
    schema_version: int = 1

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name=type(self).__name__)
        if self.phase_kind != "postprocessing" or self.action_graph_digest != postprocessing_action_graph_digest(
            self.actions
        ):
            raise ValueError("historical postprocessing V1 action graph differs")
        _sha(self.action_semantics_digest, "historical postprocessing action semantics")
        if (
            self.scientific_identity.logical_input_identity_digest != self.logical_inputs.digest
            or self.scientific_identity.acceptance_semantic_digest != self.acceptance_policy.semantic_digest
        ):
            raise ValueError("historical postprocessing V1 scientific identity bindings differ")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "actions": [item.to_mapping() for item in self.actions],
            "action_graph_digest": self.action_graph_digest,
            "action_semantics_digest": self.action_semantics_digest,
            "scientific_identity": self.scientific_identity.to_mapping(),
            "logical_inputs": self.logical_inputs.to_mapping(),
            "physical_inputs": [item.to_mapping() for item in self.physical_inputs],
            "execution_projection": self.execution_projection.to_mapping(),
            "acceptance_policy": self.acceptance_policy.to_mapping(),
            "qualified_runtime": self.qualified_runtime.to_mapping(),
            "attempt_paths": self.attempt_paths.to_mapping(),
        }


@dataclass(frozen=True)
class HistoricalPostprocessingPhaseRunSpecV1:
    """Fully typed read-only authority for the superseded V1 identity family."""

    phase_run_id: str
    attempt_id: str
    phase_plan_digest: str
    materialized_at: str
    cluster: PostprocessingClusterSnapshot
    payload: HistoricalPostprocessingPhaseRunSpecPayloadV1
    phase_kind: Literal["postprocessing"] = "postprocessing"
    schema_version: int = 1
    contract_family: Literal["postprocessing-runspec-v1"] = "postprocessing-runspec-v1"

    def __post_init__(self) -> None:
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        _sha(self.phase_plan_digest, "historical postprocessing Phase Plan digest")
        validate_schema_version(self.schema_version, record_name=type(self).__name__)
        if (
            not self.materialized_at
            or self.phase_kind != "postprocessing"
            or self.contract_family != "postprocessing-runspec-v1"
            or self.cluster.runtime_image != self.payload.qualified_runtime.image_path
        ):
            raise ValueError("historical postprocessing V1 RunSpec envelope is invalid")
        identity = self.payload.execution_projection.phase_identity
        if (
            identity.phase_plan_digest != self.phase_plan_digest
            or identity.logical_input_manifest_digest != self.payload.logical_inputs.digest
            or identity.scientific_identity_digest != self.payload.scientific_identity.digest
            or identity.phase_run_id != self.phase_run_id
            or identity.attempt_id != self.attempt_id
            or identity.qualified_runtime_digest != self.payload.qualified_runtime.digest
            or identity.substitutions != self.payload.attempt_paths
        ):
            raise ValueError("historical postprocessing V1 projection bindings differ")
        attempt_prefix = f"attempts/{self.attempt_id}/"
        if not self.payload.execution_projection.document_location.startswith(
            attempt_prefix
        ) or not self.payload.acceptance_policy.location.startswith(attempt_prefix):
            raise ValueError("historical postprocessing V1 Attempt projections are mismatched")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_plan_digest": self.phase_plan_digest,
            "materialized_at": self.materialized_at,
            "cluster": self.cluster.to_mapping(),
            "payload": self.payload.to_mapping(),
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


def _historical_v1_payload_from_mapping(
    payload: Mapping[str, object],
) -> HistoricalPostprocessingPhaseRunSpecPayloadV1:
    """Parse every persisted V1 field into immutable historical authority."""
    _fields(
        payload,
        {
            "schema_version",
            "phase_kind",
            "actions",
            "action_graph_digest",
            "action_semantics_digest",
            "scientific_identity",
            "logical_inputs",
            "physical_inputs",
            "execution_projection",
            "acceptance_policy",
            "qualified_runtime",
            "attempt_paths",
        },
        "HistoricalPostprocessingPhaseRunSpecPayloadV1",
    )
    if _str(payload, "phase_kind") != "postprocessing":
        raise ValueError("historical postprocessing V1 payload has an invalid phase kind")
    actions = tuple(postprocessing_runtime_action_from_mapping(item) for item in _mapping_list(payload, "actions"))
    logical_mapping = _mapping(payload, "logical_inputs")
    _fields(logical_mapping, {"schema_version", "manifest_kind", "entries"}, "LogicalInputManifestV1")
    logical = LogicalInputManifest(
        schema_version=validate_schema_version(
            logical_mapping.get("schema_version"), record_name="LogicalInputManifestV1"
        ),
        manifest_kind=_str(logical_mapping, "manifest_kind"),
        entries=tuple(logical_input_entry_from_mapping(item) for item in _mapping_list(logical_mapping, "entries")),
    )
    scientific_mapping = _mapping(payload, "scientific_identity")
    _fields(
        scientific_mapping,
        {
            "schema_version",
            "identity_kind",
            "dataset_scope_digest",
            "scientific_parameters_digest",
            "logical_input_identity_digest",
            "acceptance_semantic_digest",
        },
        "PostprocessingScientificIdentityV1",
    )
    scientific = PostprocessingScientificIdentityV1(
        schema_version=validate_schema_version(
            scientific_mapping.get("schema_version"), record_name="PostprocessingScientificIdentityV1"
        ),
        identity_kind=cast(
            "Literal['postprocessing-scientific-identity-v1']",
            _str(scientific_mapping, "identity_kind"),
        ),
        dataset_scope_digest=_str(scientific_mapping, "dataset_scope_digest"),
        scientific_parameters_digest=_str(scientific_mapping, "scientific_parameters_digest"),
        logical_input_identity_digest=_str(scientific_mapping, "logical_input_identity_digest"),
        acceptance_semantic_digest=_str(scientific_mapping, "acceptance_semantic_digest"),
    )
    acceptance = postprocessing_acceptance_reference_from_mapping(_mapping(payload, "acceptance_policy"))
    return HistoricalPostprocessingPhaseRunSpecPayloadV1(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="HistoricalPostprocessingPhaseRunSpecPayloadV1"
        ),
        actions=actions,
        action_graph_digest=_str(payload, "action_graph_digest"),
        action_semantics_digest=_str(payload, "action_semantics_digest"),
        scientific_identity=scientific,
        logical_inputs=logical,
        physical_inputs=tuple(
            physical_input_locator_from_mapping(item) for item in _mapping_list(payload, "physical_inputs")
        ),
        execution_projection=_historical_projection(_mapping(payload, "execution_projection")),
        acceptance_policy=acceptance,
        qualified_runtime=qualified_postprocessing_runtime_from_mapping(_mapping(payload, "qualified_runtime")),
        attempt_paths=postprocessing_attempt_paths_from_mapping(_mapping(payload, "attempt_paths")),
    )


def _historical_projection(payload: Mapping[str, object]) -> HistoricalPostprocessingExecutionProjectionV1:
    _fields(
        payload,
        {
            "schema_version",
            "projection_kind",
            "legacy_schema_version",
            "document_location",
            "document_sha256",
            "document_size_bytes",
            "phase_identity_digest",
            "phase_identity",
        },
        "HistoricalPostprocessingExecutionProjectionV1",
    )
    return HistoricalPostprocessingExecutionProjectionV1(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="HistoricalPostprocessingExecutionProjectionV1"
        ),
        projection_kind=cast("Literal['legacy-runspec-yaml-v1']", _str(payload, "projection_kind")),
        legacy_schema_version=_int(payload, "legacy_schema_version"),
        document_location=_str(payload, "document_location"),
        document_sha256=_str(payload, "document_sha256"),
        document_size_bytes=_int(payload, "document_size_bytes"),
        phase_identity_digest=_str(payload, "phase_identity_digest"),
        phase_identity=_historical_execution_identity(_mapping(payload, "phase_identity")),
    )


def _historical_execution_identity(
    payload: Mapping[str, object],
) -> HistoricalPostprocessingPhaseExecutionIdentityV1:
    _fields(
        payload,
        {
            "schema_version",
            "identity_kind",
            "phase_plan_digest",
            "logical_input_manifest_digest",
            "scientific_identity_digest",
            "phase_run_id",
            "attempt_id",
            "qualified_runtime_digest",
            "output_namespace",
            "substitutions",
        },
        "HistoricalPostprocessingPhaseExecutionIdentityV1",
    )
    return HistoricalPostprocessingPhaseExecutionIdentityV1(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="HistoricalPostprocessingPhaseExecutionIdentityV1"
        ),
        identity_kind=cast(
            "Literal['postprocessing-execution-projection-v1']",
            _str(payload, "identity_kind"),
        ),
        phase_plan_digest=_str(payload, "phase_plan_digest"),
        logical_input_manifest_digest=_str(payload, "logical_input_manifest_digest"),
        scientific_identity_digest=_str(payload, "scientific_identity_digest"),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        qualified_runtime_digest=_str(payload, "qualified_runtime_digest"),
        output_namespace=_str(payload, "output_namespace"),
        substitutions=postprocessing_attempt_paths_from_mapping(_mapping(payload, "substitutions")),
    )


__all__ = [
    "HistoricalPostprocessingExecutionProjectionV1",
    "HistoricalPostprocessingPhaseExecutionIdentityV1",
    "HistoricalPostprocessingPhaseRunSpecPayloadV1",
    "HistoricalPostprocessingPhaseRunSpecV1",
]
