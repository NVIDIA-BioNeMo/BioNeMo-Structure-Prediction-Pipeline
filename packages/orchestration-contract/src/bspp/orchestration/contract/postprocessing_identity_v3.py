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

"""Version 3 postprocessing scientific and action-semantic identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bspp.orchestration.contract._postprocessing_validation import _schema, _sha
from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_logical_identity import PostprocessingSemanticField
from bspp.orchestration.contract.postprocessing_phase_ids import postprocessing_action_id
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION


@dataclass(frozen=True)
class PostprocessingDatasetScopeV3:
    """Scientific dataset scope including the authored tracking selector."""

    dataset_name: str
    mode: str
    array: str
    archive_source: str
    scope_kind: Literal["postprocessing-dataset-scope-v3"] = "postprocessing-dataset-scope-v3"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.scope_kind != "postprocessing-dataset-scope-v3" or not all(
            (self.dataset_name, self.mode, self.array, self.archive_source)
        ):
            raise ValueError("postprocessing dataset scope v3 fields must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope_kind": self.scope_kind,
            "dataset_name": self.dataset_name,
            "mode": self.mode,
            "array": self.array,
            "archive_source": self.archive_source,
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class PostprocessingScientificIdentityV3:
    dataset_scope_digest: str
    scientific_parameters_digest: str
    logical_input_identity_digest: str
    identity_kind: Literal["postprocessing-scientific-identity-v3"] = "postprocessing-scientific-identity-v3"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.identity_kind != "postprocessing-scientific-identity-v3":
            raise ValueError("unsupported postprocessing scientific identity kind")
        for value in (
            self.dataset_scope_digest,
            self.scientific_parameters_digest,
            self.logical_input_identity_digest,
        ):
            _sha(value, "postprocessing scientific identity component")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity_kind": self.identity_kind,
            "dataset_scope_digest": self.dataset_scope_digest,
            "scientific_parameters_digest": self.scientific_parameters_digest,
            "logical_input_identity_digest": self.logical_input_identity_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class PostprocessingActionSemanticV3:
    action_id: str
    step_name: str
    mode: str | None
    dependencies: tuple[str, ...]
    normalized_task_scope: tuple[int, ...]
    command_contract: str
    normalized_command_sha256: str
    normalized_arguments: tuple[PostprocessingSemanticField, ...]
    semantic_kind: Literal["postprocessing-action-semantic-v3"] = "postprocessing-action-semantic-v3"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if (
            self.semantic_kind != "postprocessing-action-semantic-v3"
            or self.action_id != postprocessing_action_id(self.step_name)
            or not self.command_contract.endswith("-v3")
        ):
            raise ValueError("postprocessing action semantic v3 identity is invalid")
        _sha(self.normalized_command_sha256, "postprocessing normalized command")
        paths = tuple(item.path for item in self.normalized_arguments)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("postprocessing normalized action arguments must be unique and sorted")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "semantic_kind": self.semantic_kind,
            "action_id": self.action_id,
            "step_name": self.step_name,
            "mode": self.mode,
            "dependencies": list(self.dependencies),
            "normalized_task_scope": list(self.normalized_task_scope),
            "command_contract": self.command_contract,
            "normalized_command_sha256": self.normalized_command_sha256,
            "normalized_arguments": [item.to_mapping() for item in self.normalized_arguments],
        }


@dataclass(frozen=True)
class PostprocessingActionSemanticsV3:
    scientific_identity_digest: str
    logical_input_identity_digest: str
    acceptance_semantic_digest: str
    semantic_fields: tuple[PostprocessingSemanticField, ...]
    actions: tuple[PostprocessingActionSemanticV3, ...]
    semantics_kind: Literal["postprocessing-action-semantics-v3"] = "postprocessing-action-semantics-v3"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.semantics_kind != "postprocessing-action-semantics-v3":
            raise ValueError("unsupported postprocessing action semantics kind")
        for value in (
            self.scientific_identity_digest,
            self.logical_input_identity_digest,
            self.acceptance_semantic_digest,
        ):
            _sha(value, "postprocessing action semantics component")
        field_paths = tuple(item.path for item in self.semantic_fields)
        action_ids = tuple(item.action_id for item in self.actions)
        if (
            not field_paths
            or field_paths != tuple(sorted(field_paths))
            or len(set(field_paths)) != len(field_paths)
            or not action_ids
            or len(set(action_ids)) != len(action_ids)
        ):
            raise ValueError("postprocessing action semantics fields and actions must be unique and ordered")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "semantics_kind": self.semantics_kind,
            "scientific_identity_digest": self.scientific_identity_digest,
            "logical_input_identity_digest": self.logical_input_identity_digest,
            "acceptance_semantic_digest": self.acceptance_semantic_digest,
            "semantic_fields": [item.to_mapping() for item in self.semantic_fields],
            "actions": [item.to_mapping() for item in self.actions],
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


__all__ = [
    "PostprocessingActionSemanticV3",
    "PostprocessingActionSemanticsV3",
    "PostprocessingDatasetScopeV3",
    "PostprocessingScientificIdentityV3",
]
