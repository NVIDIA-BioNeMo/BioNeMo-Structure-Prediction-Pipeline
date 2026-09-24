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

"""Executable postprocessing Phase RunSpec V3 contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract._postprocessing_validation import (
    _fields,
    _mapping,
    _mapping_list,
    _schema,
    _sha,
    _str,
    _str_list,
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
    validate_postprocessing_action_graph,
)
from bspp.orchestration.contract.postprocessing_autorequeue_policy import (
    POSTPROCESSING_AUTOREQUEUE_DISABLED,
    PostprocessingAutorequeuePolicy,
    postprocessing_autorequeue_policy_from_mapping,
)
from bspp.orchestration.contract.postprocessing_execution import (
    PostprocessingAttemptPaths,
    PostprocessingClusterSnapshot,
    PostprocessingCredentialMountSnapshot,
    PostprocessingExecutionProjection,
    QualifiedPostprocessingRuntimeSelection,
)
from bspp.orchestration.contract.postprocessing_execution_parsing import (
    postprocessing_acceptance_reference_from_mapping,
    postprocessing_attempt_paths_from_mapping,
    postprocessing_cluster_snapshot_from_mapping,
    postprocessing_credential_mount_snapshot_from_mapping,
    postprocessing_execution_projection_from_mapping,
    qualified_postprocessing_runtime_from_mapping,
)
from bspp.orchestration.contract.postprocessing_identity_v3 import (
    PostprocessingActionSemanticsV3,
    PostprocessingActionSemanticV3,
    PostprocessingScientificIdentityV3,
)
from bspp.orchestration.contract.postprocessing_logical_identity import (
    PhysicalInputLocator,
    PostprocessingLogicalInputIdentityManifestV2,
    PostprocessingSemanticField,
)
from bspp.orchestration.contract.postprocessing_plan import PostprocessingPhaseKind
from bspp.orchestration.contract.postprocessing_runspec_parsing import (
    physical_input_locator_from_mapping,
    postprocessing_runtime_action_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runspec_v2 import _logical_manifest
from bspp.orchestration.contract.versioning import validate_schema_version

POSTPROCESSING_RUNSPEC_V3_KIND: Literal["postprocessing-phase-runspec-v3"] = "postprocessing-phase-runspec-v3"


@dataclass(frozen=True)
class PostprocessingPhaseRunSpecPayloadV3:
    actions: tuple[PostprocessingRuntimeAction, ...]
    action_graph_digest: str
    action_semantics_digest: str
    action_semantics: PostprocessingActionSemanticsV3
    scientific_identity: PostprocessingScientificIdentityV3
    logical_inputs: PostprocessingLogicalInputIdentityManifestV2
    physical_inputs: tuple[PhysicalInputLocator, ...]
    execution_projection: PostprocessingExecutionProjection
    acceptance_policy: PostprocessingAcceptanceSnapshotReference
    qualified_runtime: QualifiedPostprocessingRuntimeSelection
    attempt_paths: PostprocessingAttemptPaths
    phase_kind: PostprocessingPhaseKind = "postprocessing"
    schema_version: int = 1
    autorequeue_policy: PostprocessingAutorequeuePolicy = POSTPROCESSING_AUTOREQUEUE_DISABLED

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.phase_kind != "postprocessing":
            raise ValueError("postprocessing RunSpec payload must declare phase_kind 'postprocessing'")
        validate_postprocessing_action_graph(self.actions)
        expected_graph = canonical_mapping_digest(
            {
                "schema_version": 1,
                "phase_kind": self.phase_kind,
                "actions": [
                    {
                        "action_id": item.action_id,
                        "dependencies": list(item.dependencies),
                        "expected_task_indexes": list(item.expected_task_indexes),
                    }
                    for item in self.actions
                ],
            }
        )
        if self.action_graph_digest != expected_graph:
            raise ValueError("postprocessing action graph digest does not match the frozen graph")
        _sha(self.action_semantics_digest, "postprocessing action semantics digest")
        if self.action_semantics_digest != self.action_semantics.digest:
            raise ValueError("postprocessing action semantics digest does not match its stored preimage")
        if self.scientific_identity.logical_input_identity_digest != self.logical_inputs.digest:
            raise ValueError("postprocessing scientific identity does not bind the logical input identity")
        if (
            self.action_semantics.scientific_identity_digest != self.scientific_identity.digest
            or self.action_semantics.logical_input_identity_digest != self.logical_inputs.digest
            or self.action_semantics.acceptance_semantic_digest != self.acceptance_policy.semantic_digest
        ):
            raise ValueError("postprocessing action semantics does not bind its independent identities")
        names = tuple(item.name for item in self.physical_inputs)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("postprocessing physical input locators must be unique and name-sorted")

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "actions": [item.to_mapping() for item in self.actions],
            "action_graph_digest": self.action_graph_digest,
            "action_semantics_digest": self.action_semantics_digest,
            "action_semantics": self.action_semantics.to_mapping(),
            "scientific_identity": self.scientific_identity.to_mapping(),
            "logical_inputs": self.logical_inputs.to_mapping(),
            "physical_inputs": [item.to_mapping() for item in self.physical_inputs],
            "execution_projection": self.execution_projection.to_mapping(),
            "acceptance_policy": self.acceptance_policy.to_mapping(),
            "qualified_runtime": self.qualified_runtime.to_mapping(),
            "attempt_paths": self.attempt_paths.to_mapping(),
        }
        if self.autorequeue_policy != POSTPROCESSING_AUTOREQUEUE_DISABLED:
            mapping["autorequeue_policy"] = self.autorequeue_policy.to_mapping()
        return mapping


@dataclass(frozen=True)
class PostprocessingPhaseRunSpecV3:
    phase_run_id: str
    attempt_id: str
    phase_plan_digest: str
    materialized_at: str
    cluster: PostprocessingClusterSnapshot
    payload: PostprocessingPhaseRunSpecPayloadV3
    credential_mounts: PostprocessingCredentialMountSnapshot | None = None
    cluster_output_root: str | None = None
    object_output_base_prefix: str | None = None
    phase_kind: PostprocessingPhaseKind = "postprocessing"
    schema_version: int = 1
    runspec_kind: Literal["postprocessing-phase-runspec-v3"] = POSTPROCESSING_RUNSPEC_V3_KIND

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        _sha(self.phase_plan_digest, "postprocessing Phase Plan digest")
        if not self.materialized_at:
            raise ValueError("postprocessing Phase RunSpec materialized_at must be non-empty")
        if self.cluster_output_root == "" or self.object_output_base_prefix == "":
            raise ValueError("postprocessing V3 output namespace roots must be non-empty when present")
        if self.phase_kind != "postprocessing" or self.payload.phase_kind != self.phase_kind:
            raise ValueError("postprocessing Phase RunSpec envelope and payload discriminator must match")
        attempt_prefix = f"attempts/{self.attempt_id}/"
        if not self.payload.execution_projection.document_location.startswith(attempt_prefix):
            raise ValueError("postprocessing execution projection must match its Attempt")
        if not self.payload.acceptance_policy.location.startswith(attempt_prefix):
            raise ValueError("postprocessing acceptance snapshot must match its Attempt")
        identity = self.payload.execution_projection.phase_identity
        if (
            identity.phase_plan_digest != self.phase_plan_digest
            or identity.logical_input_manifest_digest != self.payload.logical_inputs.digest
            or identity.scientific_identity_digest != self.payload.scientific_identity.digest
            or identity.action_semantics_digest != self.payload.action_semantics_digest
            or identity.acceptance_semantic_digest != self.payload.acceptance_policy.semantic_digest
            or identity.phase_run_id != self.phase_run_id
            or identity.attempt_id != self.attempt_id
            or identity.qualified_runtime_digest != self.payload.qualified_runtime.digest
            or identity.substitutions != self.payload.attempt_paths
        ):
            raise ValueError("postprocessing projection identity does not bind the complete RunSpec authority")
        if self.cluster.runtime_image != self.payload.qualified_runtime.image_path:
            raise ValueError("postprocessing cluster image must equal the qualified runtime image")
        if self.runspec_kind != POSTPROCESSING_RUNSPEC_V3_KIND:
            raise ValueError("unsupported postprocessing V3 RunSpec discriminator")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "runspec_kind": self.runspec_kind,
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_plan_digest": self.phase_plan_digest,
            "materialized_at": self.materialized_at,
            "cluster": self.cluster.to_mapping(),
            "payload": self.payload.to_mapping(),
        }
        if self.credential_mounts is not None:
            result["credential_mounts"] = self.credential_mounts.to_mapping()
        if self.cluster_output_root is not None:
            result["cluster_output_root"] = self.cluster_output_root
        if self.object_output_base_prefix is not None:
            result["object_output_base_prefix"] = self.object_output_base_prefix
        return result

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


def postprocessing_phase_runspec_v3_from_mapping(payload: Mapping[str, object]) -> PostprocessingPhaseRunSpecV3:
    _fields(
        payload,
        {
            "runspec_kind",
            "schema_version",
            "phase_kind",
            "phase_run_id",
            "attempt_id",
            "phase_plan_digest",
            "materialized_at",
            "cluster",
            "payload",
            "credential_mounts",
            "cluster_output_root",
            "object_output_base_prefix",
        },
        "PostprocessingPhaseRunSpecV3",
    )
    if _str(payload, "runspec_kind") != POSTPROCESSING_RUNSPEC_V3_KIND:
        raise ValueError("unsupported postprocessing V3 RunSpec discriminator")
    if _str(payload, "phase_kind") != "postprocessing":
        raise ValueError("postprocessing Phase RunSpec must declare phase_kind 'postprocessing'")
    return PostprocessingPhaseRunSpecV3(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingPhaseRunSpecV3"
        ),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        phase_plan_digest=_str(payload, "phase_plan_digest"),
        materialized_at=_str(payload, "materialized_at"),
        cluster=postprocessing_cluster_snapshot_from_mapping(_mapping(payload, "cluster")),
        payload=_runspec_payload_v3(_mapping(payload, "payload")),
        credential_mounts=(
            postprocessing_credential_mount_snapshot_from_mapping(_mapping(payload, "credential_mounts"))
            if payload.get("credential_mounts") is not None
            else None
        ),
        cluster_output_root=_optional_non_empty_string(payload, "cluster_output_root"),
        object_output_base_prefix=_optional_non_empty_string(payload, "object_output_base_prefix"),
    )


def _optional_non_empty_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be absent or a non-empty string")
    return value


def _runspec_payload_v3(payload: Mapping[str, object]) -> PostprocessingPhaseRunSpecPayloadV3:
    _fields(
        payload,
        {
            "schema_version",
            "phase_kind",
            "actions",
            "action_graph_digest",
            "action_semantics_digest",
            "action_semantics",
            "scientific_identity",
            "logical_inputs",
            "physical_inputs",
            "execution_projection",
            "acceptance_policy",
            "qualified_runtime",
            "attempt_paths",
            "autorequeue_policy",
        },
        "PostprocessingPhaseRunSpecPayloadV3",
    )
    if _str(payload, "phase_kind") != "postprocessing":
        raise ValueError("postprocessing RunSpec payload must declare phase_kind 'postprocessing'")
    return PostprocessingPhaseRunSpecPayloadV3(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingPhaseRunSpecPayloadV3"
        ),
        actions=tuple(postprocessing_runtime_action_from_mapping(item) for item in _mapping_list(payload, "actions")),
        action_graph_digest=_str(payload, "action_graph_digest"),
        action_semantics_digest=_str(payload, "action_semantics_digest"),
        action_semantics=_action_semantics_v3(_mapping(payload, "action_semantics")),
        scientific_identity=_scientific_identity_v3(_mapping(payload, "scientific_identity")),
        logical_inputs=_logical_manifest(_mapping(payload, "logical_inputs")),
        physical_inputs=tuple(
            physical_input_locator_from_mapping(item) for item in _mapping_list(payload, "physical_inputs")
        ),
        execution_projection=postprocessing_execution_projection_from_mapping(
            _mapping(payload, "execution_projection")
        ),
        acceptance_policy=postprocessing_acceptance_reference_from_mapping(_mapping(payload, "acceptance_policy")),
        qualified_runtime=qualified_postprocessing_runtime_from_mapping(_mapping(payload, "qualified_runtime")),
        attempt_paths=postprocessing_attempt_paths_from_mapping(_mapping(payload, "attempt_paths")),
        autorequeue_policy=(
            postprocessing_autorequeue_policy_from_mapping(_mapping(payload, "autorequeue_policy"))
            if payload.get("autorequeue_policy") is not None
            else POSTPROCESSING_AUTOREQUEUE_DISABLED
        ),
    )


def _scientific_identity_v3(payload: Mapping[str, object]) -> PostprocessingScientificIdentityV3:
    _fields(
        payload,
        {
            "schema_version",
            "identity_kind",
            "dataset_scope_digest",
            "scientific_parameters_digest",
            "logical_input_identity_digest",
        },
        "PostprocessingScientificIdentityV3",
    )
    return PostprocessingScientificIdentityV3(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingScientificIdentityV3"
        ),
        identity_kind=cast("Literal['postprocessing-scientific-identity-v3']", _str(payload, "identity_kind")),
        dataset_scope_digest=_str(payload, "dataset_scope_digest"),
        scientific_parameters_digest=_str(payload, "scientific_parameters_digest"),
        logical_input_identity_digest=_str(payload, "logical_input_identity_digest"),
    )


def _semantic_field(payload: Mapping[str, object]) -> PostprocessingSemanticField:
    _fields(payload, {"schema_version", "path", "value"}, "PostprocessingSemanticField")
    if "value" not in payload:
        raise ValueError("PostprocessingSemanticField is missing value")
    return PostprocessingSemanticField(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingSemanticField"
        ),
        path=_str(payload, "path"),
        value=payload["value"],
    )


def _action_semantic_v3(payload: Mapping[str, object]) -> PostprocessingActionSemanticV3:
    _fields(
        payload,
        {
            "schema_version",
            "semantic_kind",
            "action_id",
            "step_name",
            "mode",
            "dependencies",
            "normalized_task_scope",
            "command_contract",
            "normalized_command_sha256",
            "normalized_arguments",
        },
        "PostprocessingActionSemanticV3",
    )
    mode = payload.get("mode")
    if mode is not None and (not isinstance(mode, str) or not mode):
        raise ValueError("postprocessing action semantic mode must be null or non-empty")
    scope = payload.get("normalized_task_scope")
    if not isinstance(scope, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in scope):
        raise ValueError("postprocessing normalized task scope must be a list of integers")
    return PostprocessingActionSemanticV3(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingActionSemanticV3"
        ),
        semantic_kind=cast("Literal['postprocessing-action-semantic-v3']", _str(payload, "semantic_kind")),
        action_id=_str(payload, "action_id"),
        step_name=_str(payload, "step_name"),
        mode=mode,
        dependencies=_str_list(payload, "dependencies"),
        normalized_task_scope=tuple(scope),
        command_contract=_str(payload, "command_contract"),
        normalized_command_sha256=_str(payload, "normalized_command_sha256"),
        normalized_arguments=tuple(_semantic_field(item) for item in _mapping_list(payload, "normalized_arguments")),
    )


def _action_semantics_v3(payload: Mapping[str, object]) -> PostprocessingActionSemanticsV3:
    _fields(
        payload,
        {
            "schema_version",
            "semantics_kind",
            "scientific_identity_digest",
            "logical_input_identity_digest",
            "acceptance_semantic_digest",
            "semantic_fields",
            "actions",
        },
        "PostprocessingActionSemanticsV3",
    )
    return PostprocessingActionSemanticsV3(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingActionSemanticsV3"
        ),
        semantics_kind=cast("Literal['postprocessing-action-semantics-v3']", _str(payload, "semantics_kind")),
        scientific_identity_digest=_str(payload, "scientific_identity_digest"),
        logical_input_identity_digest=_str(payload, "logical_input_identity_digest"),
        acceptance_semantic_digest=_str(payload, "acceptance_semantic_digest"),
        semantic_fields=tuple(_semantic_field(item) for item in _mapping_list(payload, "semantic_fields")),
        actions=tuple(_action_semantic_v3(item) for item in _mapping_list(payload, "actions")),
    )


__all__ = [
    "POSTPROCESSING_RUNSPEC_V3_KIND",
    "PostprocessingPhaseRunSpecPayloadV3",
    "PostprocessingPhaseRunSpecV3",
    "postprocessing_phase_runspec_v3_from_mapping",
]
