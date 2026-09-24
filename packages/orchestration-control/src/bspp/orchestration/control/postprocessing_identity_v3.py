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

"""V3 postprocessing identity builders with an authored dataset selector."""

from __future__ import annotations

import hashlib

from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    PostprocessingAcceptancePolicySnapshot,
)
from bspp.orchestration.contract.postprocessing_action_contract import PostprocessingRuntimeAction
from bspp.orchestration.contract.postprocessing_identity_v3 import (
    PostprocessingActionSemanticsV3,
    PostprocessingActionSemanticV3,
    PostprocessingDatasetScopeV3,
    PostprocessingScientificIdentityV3,
)
from bspp.orchestration.contract.postprocessing_logical_identity import (
    PostprocessingLogicalInputIdentityManifestV2,
    PostprocessingScientificParametersV2,
    PostprocessingSemanticField,
)
from bspp.orchestration.contract.runplan import RunPlan
from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.control.postprocessing_action_commands import resolve_postprocessing_action_command
from bspp.orchestration.control.postprocessing_identity import project_run_plan_fields, project_workflow_fields
from bspp.orchestration.control.postprocessing_identity_registry import identity_normalized_runspec


def build_scientific_identity_v3(
    legacy_plan: RunPlan,
    *,
    logical_inputs: PostprocessingLogicalInputIdentityManifestV2,
) -> PostprocessingScientificIdentityV3:
    scope = PostprocessingDatasetScopeV3(
        dataset_name=legacy_plan.dataset.name,
        mode=legacy_plan.dataset.mode,
        array=legacy_plan.dataset.array,
        archive_source=legacy_plan.dataset.archive_source,
    )
    parameters = PostprocessingScientificParametersV2(
        fields=project_run_plan_fields(legacy_plan, roles=frozenset({"scientific"}))
    )
    return PostprocessingScientificIdentityV3(
        dataset_scope_digest=scope.digest,
        scientific_parameters_digest=parameters.digest,
        logical_input_identity_digest=logical_inputs.digest,
    )


def build_action_semantics_v3(
    actions: tuple[PostprocessingRuntimeAction, ...],
    *,
    legacy_plan: RunPlan,
    legacy_runspec: RunSpec,
    scientific_identity: PostprocessingScientificIdentityV3,
    logical_inputs: PostprocessingLogicalInputIdentityManifestV2,
    acceptance_policy: PostprocessingAcceptancePolicySnapshot,
) -> PostprocessingActionSemanticsV3:
    semantic_fields = (
        *project_run_plan_fields(
            legacy_plan,
            roles=frozenset({"scientific", "action-semantic", "encoding", "acceptance"}),
        ),
        PostprocessingSemanticField(path="dataset.name", value=legacy_plan.dataset.name),
        *project_workflow_fields(legacy_runspec),
    )
    deduplicated = {item.path: item for item in semantic_fields}
    if len(deduplicated) != len(semantic_fields):
        raise ValueError("postprocessing V3 semantic projection contains duplicate paths")
    ordered = tuple(deduplicated[path] for path in sorted(deduplicated))
    return PostprocessingActionSemanticsV3(
        scientific_identity_digest=scientific_identity.digest,
        logical_input_identity_digest=logical_inputs.digest,
        acceptance_semantic_digest=acceptance_policy.semantic_digest,
        semantic_fields=ordered,
        actions=tuple(
            resolve_action_semantic_v3(
                action,
                legacy_runspec=legacy_runspec,
                normalized_arguments=ordered,
            )
            for action in actions
        ),
    )


def resolve_action_semantic_v3(
    action: PostprocessingRuntimeAction,
    *,
    legacy_runspec: RunSpec,
    normalized_arguments: tuple[PostprocessingSemanticField, ...],
) -> PostprocessingActionSemanticV3:
    if legacy_runspec.workflow is None:
        raise ValueError("postprocessing action semantics require a materialized workflow")
    step = next((item for item in legacy_runspec.workflow.steps if item.name == action.step_name), None)
    if action.step_name == "acceptance-adjudication":
        mode = None
        raw_command = "bspp-postprocessing-runtime finalization publish-action09"
    else:
        if step is None:
            raise ValueError(f"postprocessing action {action.action_id!r} has no workflow step")
        mode = step.mode
        normalized_runspec = identity_normalized_runspec(legacy_runspec)
        normalized_runspec = normalized_runspec.model_copy(
            update={"dataset": normalized_runspec.dataset.model_copy(update={"name": legacy_runspec.dataset.name})}
        )
        raw_command = resolve_postprocessing_action_command(normalized_runspec, action)
    return PostprocessingActionSemanticV3(
        action_id=action.action_id,
        step_name=action.step_name,
        mode=mode,
        dependencies=action.dependencies,
        normalized_task_scope=action.expected_task_indexes,
        command_contract=f"postprocessing-{action.step_name}-v3",
        normalized_command_sha256=hashlib.sha256(raw_command.encode()).hexdigest(),
        normalized_arguments=normalized_arguments,
    )


__all__ = [
    "build_action_semantics_v3",
    "build_scientific_identity_v3",
    "resolve_action_semantic_v3",
]
