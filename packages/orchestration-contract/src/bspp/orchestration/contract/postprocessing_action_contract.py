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

import itertools
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from bspp.orchestration.contract._postprocessing_validation import _schema, _sha
from bspp.orchestration.contract.phase import PhaseSlurmResources, canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_logical_identity import (
    PostprocessingScientificIdentityV1,
    PostprocessingSemanticField,
)
from bspp.orchestration.contract.postprocessing_phase_ids import POSTPROCESSING_STEP_ORDINALS, postprocessing_action_id
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION

PostprocessingActionKind = Literal["postprocessing-step"]


@dataclass(frozen=True)
class PostprocessingRuntimeAction:
    action_id: str
    step_name: str
    dependencies: tuple[str, ...]
    resources: PhaseSlurmResources
    normalized_array: str | None
    expected_task_indexes: tuple[int, ...]
    action_kind: PostprocessingActionKind = "postprocessing-step"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.action_kind != "postprocessing-step" or self.action_id != postprocessing_action_id(self.step_name):
            raise ValueError("postprocessing Runtime Action id/kind does not match its permanent step ordinal")
        if len(set(self.dependencies)) != len(self.dependencies) or any(not item for item in self.dependencies):
            raise ValueError("postprocessing Runtime Action dependencies must be unique non-empty ids")
        normalized, indexes = normalize_slurm_array(self.resources.array)
        if self.normalized_array != normalized or self.expected_task_indexes != indexes:
            raise ValueError("postprocessing Runtime Action array snapshot is not normalized")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_kind": self.action_kind,
            "action_id": self.action_id,
            "step_name": self.step_name,
            "dependencies": list(self.dependencies),
            "resources": self.resources.to_mapping(),
            "normalized_array": self.normalized_array,
            "expected_task_indexes": list(self.expected_task_indexes),
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class PostprocessingActionSemanticV2:
    action_id: str
    step_name: str
    mode: str | None
    dependencies: tuple[str, ...]
    normalized_task_scope: tuple[int, ...]
    command_contract: str
    normalized_command_sha256: str
    normalized_arguments: tuple[PostprocessingSemanticField, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.action_id != postprocessing_action_id(self.step_name) or not self.command_contract:
            raise ValueError("postprocessing action semantic identity is invalid")
        _sha(self.normalized_command_sha256, "postprocessing normalized command")
        paths = tuple(item.path for item in self.normalized_arguments)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("postprocessing normalized action arguments must be unique and sorted")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
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
class PostprocessingActionSemanticsV2:
    scientific_identity_digest: str
    logical_input_identity_digest: str
    acceptance_semantic_digest: str
    semantic_fields: tuple[PostprocessingSemanticField, ...]
    actions: tuple[PostprocessingActionSemanticV2, ...]
    semantics_kind: Literal["postprocessing-action-semantics-v2"] = "postprocessing-action-semantics-v2"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.semantics_kind != "postprocessing-action-semantics-v2":
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


def normalize_slurm_array(expression: str | None) -> tuple[str | None, tuple[int, ...]]:
    """Strictly normalize a bounded Slurm array and enumerate its expected tasks."""
    if expression is None:
        return None, ()
    if not isinstance(expression, str) or not expression or any(character.isspace() for character in expression):
        raise ValueError("Slurm array expression must be a non-empty whitespace-free string")
    base, separator, throttle_text = expression.partition("%")
    if separator:
        if "%" in throttle_text or not throttle_text.isdigit() or int(throttle_text) < 1:
            raise ValueError("Slurm array throttle must be one positive integer")
        throttle = str(int(throttle_text))
    else:
        throttle = ""
    indexes: list[int] = []
    normalized_parts: list[str] = []
    for part in base.split(","):
        match = re.fullmatch(r"([0-9]+)(?:-([0-9]+)(?::([0-9]+))?)?", part)
        if match is None:
            raise ValueError(f"unsupported Slurm array component: {part!r}")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        step = int(match.group(3)) if match.group(3) is not None else 1
        if end < start or step < 1:
            raise ValueError("Slurm array ranges must be ascending with positive steps")
        values = list(range(start, end + 1, step))
        if len(indexes) + len(values) > 100_000:
            raise ValueError("Slurm array expands beyond the 100000-task contract bound")
        indexes.extend(values)
        if match.group(2) is None:
            normalized_parts.append(str(start))
        elif match.group(3) is None:
            normalized_parts.append(f"{start}-{end}")
        else:
            normalized_parts.append(f"{start}-{end}:{step}")
    if len(set(indexes)) != len(indexes):
        raise ValueError("Slurm array expression contains duplicate task indexes")
    normalized = ",".join(normalized_parts) + (f"%{throttle}" if separator else "")
    return normalized, tuple(sorted(indexes))


def postprocessing_action_graph_digest(actions: tuple[PostprocessingRuntimeAction, ...]) -> str:
    return canonical_mapping_digest(
        {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "phase_kind": "postprocessing",
            "actions": [
                {
                    "action_id": item.action_id,
                    "dependencies": list(item.dependencies),
                    "expected_task_indexes": list(item.expected_task_indexes),
                }
                for item in actions
            ],
        }
    )


def postprocessing_action_semantics_digest(
    actions: tuple[PostprocessingRuntimeAction, ...],
    *,
    step_modes: Mapping[str, str | None],
    scientific_identity: PostprocessingScientificIdentityV1,
) -> str:
    """Bind normalized command semantics without operational paths or resources."""
    validate_postprocessing_action_graph(actions)
    if set(step_modes) != {item.step_name for item in actions}:
        raise ValueError("postprocessing action semantics modes must cover every frozen action exactly")
    components = {
        "dataset_scope_digest": scientific_identity.dataset_scope_digest,
        "scientific_parameters_digest": scientific_identity.scientific_parameters_digest,
        "logical_input_identity_digest": scientific_identity.logical_input_identity_digest,
        "acceptance_semantic_digest": scientific_identity.acceptance_semantic_digest,
    }
    return canonical_mapping_digest(
        {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "semantics_kind": "postprocessing-action-semantics-v1",
            "actions": [
                {
                    "action_id": item.action_id,
                    "step_name": item.step_name,
                    "mode": step_modes[item.step_name],
                    "dependencies": list(item.dependencies),
                    "expected_task_indexes": list(item.expected_task_indexes),
                    "command_contract": f"postprocessing-{item.step_name}-v1",
                    "normalized_scientific_arguments": components,
                }
                for item in actions
            ],
        }
    )


def validate_postprocessing_action_graph(actions: tuple[PostprocessingRuntimeAction, ...]) -> None:
    if not isinstance(actions, tuple):
        raise ValueError("postprocessing actions must be an immutable tuple")
    steps = tuple(action.step_name for action in actions)
    if not steps or steps[0] != "preflight" or steps[-1] != "acceptance-adjudication":
        raise ValueError("postprocessing graph must start at preflight and end at acceptance adjudication")
    ordinals = tuple(POSTPROCESSING_STEP_ORDINALS[step] for step in steps)
    if ordinals != tuple(sorted(ordinals)) or len(set(ordinals)) != len(ordinals):
        raise ValueError("postprocessing graph steps must use unique permanent ordinal order")
    required = {
        "analysis-finalize",
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
        "acceptance-adjudication",
    }
    if not required.issubset(steps):
        raise ValueError("postprocessing sealable-v1 graph omits a mandatory finalization/acceptance step")
    present = set(steps)
    chain = [step for step in ("preflight", "recipe", "preprocess", "slurm", "analysis-finalize") if step in present]
    expected_dependencies: dict[str, tuple[str, ...]] = {chain[0]: ()}
    for previous, current in itertools.pairwise(chain):
        expected_dependencies[current] = (postprocessing_action_id(previous),)
    finalizer = postprocessing_action_id("analysis-finalize")
    expected_dependencies["acceptance-tar-payload-parity"] = (finalizer,)
    expected_dependencies["acceptance-semantic"] = (finalizer,)
    expected_dependencies["acceptance-verify-evidence"] = (
        postprocessing_action_id("acceptance-tar-payload-parity"),
        postprocessing_action_id("acceptance-semantic"),
    )
    expected_dependencies["acceptance-adjudication"] = (
        postprocessing_action_id("acceptance-tar-payload-parity"),
        postprocessing_action_id("acceptance-semantic"),
        postprocessing_action_id("acceptance-verify-evidence"),
    )
    for action in actions:
        if action.dependencies != expected_dependencies[action.step_name]:
            raise ValueError(f"postprocessing action {action.action_id!r} has non-canonical dependencies")


__all__ = [
    "PostprocessingActionKind",
    "PostprocessingActionSemanticV2",
    "PostprocessingActionSemanticsV2",
    "PostprocessingRuntimeAction",
    "normalize_slurm_array",
    "postprocessing_action_graph_digest",
    "postprocessing_action_semantics_digest",
    "validate_postprocessing_action_graph",
]
