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

"""Locator-free scientific and command-semantic identity for postprocessing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import yaml

from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    PostprocessingAcceptancePolicySnapshot,
)
from bspp.orchestration.contract.postprocessing_action_contract import (
    PostprocessingActionSemanticsV2,
    PostprocessingActionSemanticV2,
    PostprocessingRuntimeAction,
)
from bspp.orchestration.contract.postprocessing_logical_identity import (
    PhysicalInputLocator,
    PostprocessingDatasetScopeV2,
    PostprocessingLogicalInputIdentityManifestV2,
    PostprocessingLogicalInputIdentityV2,
    PostprocessingScientificIdentityV2,
    PostprocessingScientificParametersV2,
    PostprocessingSemanticField,
)
from bspp.orchestration.contract.runplan import RunPlan
from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.control.postprocessing_action_commands import resolve_postprocessing_action_command
from bspp.orchestration.control.postprocessing_identity_registry import (
    POSTPROCESSING_FIELD_RULES,
    PostprocessingClassifiedField,
    PostprocessingFieldRole,
    PostprocessingFieldRule,
    identity_normalized_runspec,
    rule_paths_for_component,
    run_plan_field_walk,
    workflow_field_walk,
)


def build_logical_input_identities(
    inventory_document: bytes,
    *,
    physical_inputs: tuple[PhysicalInputLocator, ...],
) -> PostprocessingLogicalInputIdentityManifestV2:
    inventory = _logical_inventory(inventory_document)
    expected_names = {item.name for item in physical_inputs}
    if set(inventory) != expected_names:
        missing = sorted(expected_names - set(inventory))
        extra = sorted(set(inventory) - expected_names)
        raise ValueError(
            f"logical input inventory names must exactly match physical inputs; missing={missing!r}, extra={extra!r}"
        )
    entries = tuple(
        PostprocessingLogicalInputIdentityV2(
            name=name,
            member_identity=identity,
            expected_content_sha256=digest,
            expected_size_bytes=size,
        )
        for name, (identity, digest, size) in sorted(inventory.items())
    )
    return PostprocessingLogicalInputIdentityManifestV2(entries=entries)


def build_scientific_identity(
    legacy_plan: RunPlan,
    *,
    logical_inputs: PostprocessingLogicalInputIdentityManifestV2,
) -> PostprocessingScientificIdentityV2:
    scope = PostprocessingDatasetScopeV2(
        mode=legacy_plan.dataset.mode,
        array=legacy_plan.dataset.array,
        archive_source=legacy_plan.dataset.archive_source,
    )
    parameters = PostprocessingScientificParametersV2(
        fields=project_run_plan_fields(legacy_plan, roles=frozenset({"scientific"}))
    )
    return PostprocessingScientificIdentityV2(
        dataset_scope_digest=scope.digest,
        scientific_parameters_digest=parameters.digest,
        logical_input_identity_digest=logical_inputs.digest,
    )


def build_action_semantics(
    actions: tuple[PostprocessingRuntimeAction, ...],
    *,
    legacy_plan: RunPlan,
    legacy_runspec: RunSpec,
    scientific_identity: PostprocessingScientificIdentityV2,
    logical_inputs: PostprocessingLogicalInputIdentityManifestV2,
    acceptance_policy: PostprocessingAcceptancePolicySnapshot,
) -> PostprocessingActionSemanticsV2:
    semantic_fields = (
        *project_run_plan_fields(
            legacy_plan,
            roles=frozenset({"scientific", "action-semantic", "encoding", "acceptance"}),
        ),
        *project_workflow_fields(legacy_runspec),
    )
    deduplicated = {item.path: item for item in semantic_fields}
    if len(deduplicated) != len(semantic_fields):
        raise ValueError("postprocessing semantic projection contains duplicate paths")
    ordered = tuple(deduplicated[path] for path in sorted(deduplicated))
    if legacy_runspec.workflow is None:
        raise ValueError("postprocessing action semantics require a materialized workflow")
    action_records = tuple(
        resolve_action_semantic(
            action,
            legacy_runspec=legacy_runspec,
            normalized_arguments=ordered,
        )
        for action in actions
    )
    return PostprocessingActionSemanticsV2(
        scientific_identity_digest=scientific_identity.digest,
        logical_input_identity_digest=logical_inputs.digest,
        acceptance_semantic_digest=acceptance_policy.semantic_digest,
        semantic_fields=ordered,
        actions=action_records,
    )


def project_run_plan_fields(
    plan: RunPlan,
    *,
    roles: frozenset[PostprocessingFieldRole],
) -> tuple[PostprocessingSemanticField, ...]:
    return _project_classified_fields(run_plan_field_walk(plan), roles=roles)


def project_workflow_fields(runspec: RunSpec) -> tuple[PostprocessingSemanticField, ...]:
    return _project_classified_fields(
        workflow_field_walk(runspec),
        roles=frozenset({"scientific", "action-semantic", "encoding", "acceptance"}),
    )


def _project_classified_fields(
    fields: tuple[PostprocessingClassifiedField, ...],
    *,
    roles: frozenset[PostprocessingFieldRole],
) -> tuple[PostprocessingSemanticField, ...]:
    projected: list[PostprocessingSemanticField] = []
    sorted_records: dict[tuple[tuple[str, ...], str], dict[str, object]] = {}
    for field in fields:
        if field.rule.kind != "leaf" or not field.rule.roles & roles:
            continue
        value = field.value
        if field.scope == "workflow-step" and field.rule.pattern[-1] == "array_range" and value is not None:
            if not isinstance(value, str):
                raise ValueError("postprocessing workflow array_range must be a string")
            value = action_task_scope(value)
        if field.sorted_record_pattern is not None and field.sorted_record_key is not None:
            key = (field.sorted_record_pattern, field.sorted_record_key)
            record = sorted_records.setdefault(key, {"name": field.sorted_record_key})
            record[field.rule.pattern[-1]] = value
            continue
        projected.append(PostprocessingSemanticField(path=field.dotted_path, value=value))
    records_by_pattern: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for (pattern, _key), record in sorted_records.items():
        records_by_pattern.setdefault(pattern, []).append(record)
    for pattern, records in records_by_pattern.items():
        records.sort(key=lambda item: json.dumps(item, separators=(",", ":"), sort_keys=True))
        prefix = pattern[: pattern.index("*")]
        projected.extend(
            PostprocessingSemanticField(path=f"{'.'.join(prefix)}.{index:04d}", value=record)
            for index, record in enumerate(records)
        )
    return tuple(sorted(projected, key=lambda item: item.path))


def action_task_scope(array_range: str) -> list[int]:
    from bspp.orchestration.contract.postprocessing_action_contract import (
        normalize_slurm_array,
    )

    return list(normalize_slurm_array(array_range)[1])


def resolve_action_semantic(
    action: PostprocessingRuntimeAction,
    *,
    legacy_runspec: RunSpec,
    normalized_arguments: tuple[PostprocessingSemanticField, ...],
) -> PostprocessingActionSemanticV2:
    """Resolve the command preimage shared by materialization and rendering."""
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
        if normalized_runspec.workflow is None:
            raise ValueError("identity-normalized RunSpec lost its workflow")
        raw_command = resolve_postprocessing_action_command(normalized_runspec, action)
    return PostprocessingActionSemanticV2(
        action_id=action.action_id,
        step_name=action.step_name,
        mode=mode,
        dependencies=action.dependencies,
        normalized_task_scope=action.expected_task_indexes,
        command_contract=f"postprocessing-{action.step_name}-v2",
        normalized_command_sha256=hashlib.sha256(raw_command.encode()).hexdigest(),
        normalized_arguments=normalized_arguments,
    )


def _logical_inventory(document: bytes) -> dict[str, tuple[str, str, int]]:
    payload = yaml.safe_load(document)
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "inventory_kind", "members"}:
        raise ValueError("postprocessing logical input inventory has missing or extra fields")
    if (
        payload.get("schema_version") != 1
        or payload.get("inventory_kind") != "postprocessing-logical-input-inventory-v1"
    ):
        raise ValueError("postprocessing logical input inventory discriminator is unsupported")
    members = payload.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("postprocessing logical input inventory members must be a non-empty list")
    result: dict[str, tuple[str, str, int]] = {}
    for member in members:
        if not isinstance(member, Mapping) or set(member) != {
            "schema_version",
            "name",
            "member_identity",
            "expected_content_sha256",
            "expected_size_bytes",
        }:
            raise ValueError("postprocessing logical input inventory member has invalid fields")
        name = member.get("name")
        identity = member.get("member_identity")
        digest = member.get("expected_content_sha256")
        size = member.get("expected_size_bytes")
        if (
            member.get("schema_version") != 1
            or not isinstance(name, str)
            or not name
            or not isinstance(identity, str)
            or not identity
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or name in result
        ):
            raise ValueError("postprocessing logical input inventory member identity is invalid")
        result[name] = (identity, digest, size)
    return result


__all__ = [
    "POSTPROCESSING_FIELD_RULES",
    "PostprocessingFieldRole",
    "PostprocessingFieldRule",
    "action_task_scope",
    "build_action_semantics",
    "build_logical_input_identities",
    "build_scientific_identity",
    "project_run_plan_fields",
    "project_workflow_fields",
    "rule_paths_for_component",
]
