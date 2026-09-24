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

"""Exhaustive field classification and traversal for postprocessing identity."""

from __future__ import annotations

import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Annotated, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from bspp.orchestration.contract.phase import PhaseMountSnapshot
from bspp.orchestration.contract.postprocessing_execution import (
    PostprocessingAttemptPaths,
    PostprocessingClusterSnapshot,
    QualifiedPostprocessingRuntimeSelection,
)
from bspp.orchestration.contract.runplan import RunPlan
from bspp.orchestration.contract.runspec import (
    ClusterSpec,
    ContainerSpec,
    PathSpec,
    RunSpec,
    SlurmResources,
    SubmissionSpec,
    WorkflowSpec,
)

PostprocessingFieldScope = Literal["run-plan", "workflow-step", "legacy-execution"]
PostprocessingFieldKind = Literal["branch", "leaf"]
PostprocessingWildcardProjection = Literal["concrete-key", "sorted-record"]
PostprocessingFieldRole = Literal[
    "scientific",
    "logical-input",
    "action-semantic",
    "acceptance",
    "encoding",
    "operational",
    "resource",
    "physical-locator",
    "placement",
    "generated",
    "secret",
]

_POSITIVE_ROLES = frozenset({"scientific", "action-semantic", "acceptance", "encoding"})
_EXCLUSION_ROLES = frozenset({"operational", "resource", "physical-locator", "placement", "generated", "secret"})
_GENERATED_LEAVES = frozenset(
    {
        ("run-plan", ("dataset", "name")),
        ("run-plan", ("dataset", "run_id")),
        ("run-plan", ("acceptance", "candidate_run_name")),
    }
)


@dataclass(frozen=True)
class PostprocessingFieldRule:
    """One schema field classification consumed by the production walker."""

    scope: PostprocessingFieldScope
    pattern: tuple[str, ...]
    kind: PostprocessingFieldKind
    roles: frozenset[PostprocessingFieldRole]
    wildcard_projection: PostprocessingWildcardProjection | None = None
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.pattern or any(not item for item in self.pattern) or not self.roles:
            raise ValueError("postprocessing field rules require a pattern and at least one role")
        if self.wildcard_projection is not None and (self.kind != "branch" or "*" not in self.pattern):
            raise ValueError("postprocessing wildcard projection belongs on a wildcard branch")
        if self.kind == "branch" and self.roles != {"generated"}:
            raise ValueError("postprocessing branch rules must use only the generated structural role")
        if "logical-input" in self.roles:
            raise ValueError("logical-input is reserved for the external inventory")
        positive = self.roles & _POSITIVE_ROLES
        exclusions = self.roles & _EXCLUSION_ROLES
        if self.roles & {"physical-locator", "secret"} and positive:
            raise ValueError("physical locators and secrets cannot enter postprocessing semantic identity")
        if self.roles & {"scientific", "encoding", "acceptance"} and exclusions:
            raise ValueError("scientific, encoding, and acceptance roles cannot be exclusions")
        if "generated" in self.roles and self.kind == "leaf" and (self.scope, self.pattern) not in _GENERATED_LEAVES:
            raise ValueError("generated leaves are limited to attempt substitutions")

    @property
    def path(self) -> str:
        """Compatibility spelling for diagnostics and existing consumers."""
        return ".".join(self.pattern)


@dataclass(frozen=True)
class PostprocessingClassifiedField:
    """One concrete branch or leaf produced by registry traversal."""

    scope: PostprocessingFieldScope
    path: tuple[str, ...]
    value: object
    rule: PostprocessingFieldRule
    sorted_record_key: str | None = None
    sorted_record_pattern: tuple[str, ...] | None = None

    @property
    def dotted_path(self) -> str:
        return ".".join(self.path)


def _rule(
    scope: PostprocessingFieldScope,
    path: str,
    kind: PostprocessingFieldKind,
    *roles: PostprocessingFieldRole,
    wildcard_projection: PostprocessingWildcardProjection | None = None,
    optional: bool = False,
) -> PostprocessingFieldRule:
    return PostprocessingFieldRule(
        scope=scope,
        pattern=tuple(path.split(".")),
        kind=kind,
        roles=frozenset(roles),
        wildcard_projection=wildcard_projection,
        optional=optional,
    )


def _branch(
    scope: PostprocessingFieldScope,
    path: str,
    *,
    wildcard_projection: PostprocessingWildcardProjection | None = None,
) -> PostprocessingFieldRule:
    return _rule(scope, path, "branch", "generated", wildcard_projection=wildcard_projection)


def _leaf(
    scope: PostprocessingFieldScope,
    path: str,
    *roles: PostprocessingFieldRole,
    optional: bool = False,
) -> PostprocessingFieldRule:
    return _rule(scope, path, "leaf", *roles, optional=optional)


_RUN_PLAN_RULES: tuple[PostprocessingFieldRule, ...] = (
    _leaf("run-plan", "schema_version", "operational"),
    _leaf("run-plan", "run_kind", "operational"),
    _leaf("run-plan", "target_cluster", "operational"),
    _leaf("run-plan", "workflow_template", "operational"),
    _branch("run-plan", "dataset"),
    _leaf("run-plan", "dataset.name", "generated"),
    _leaf("run-plan", "dataset.run_id", "generated"),
    _leaf("run-plan", "dataset.mode", "scientific", "action-semantic"),
    _leaf("run-plan", "dataset.array", "scientific", "action-semantic"),
    _leaf("run-plan", "dataset.archive_source", "scientific", "action-semantic"),
    _branch("run-plan", "references"),
    _leaf("run-plan", "references.master_parquet", "physical-locator"),
    _leaf("run-plan", "references.tracking_parquet", "physical-locator"),
    _leaf("run-plan", "references.manifest_csv", "physical-locator"),
    _leaf("run-plan", "references.uniprot_duckdb", "physical-locator"),
    _leaf("run-plan", "references.heterodimer_id_manifest", "physical-locator"),
    _branch("run-plan", "references.artifacts"),
    _branch("run-plan", "references.artifacts.*", wildcard_projection="sorted-record"),
    _leaf("run-plan", "references.artifacts.*.path", "physical-locator"),
    _leaf("run-plan", "references.artifacts.*.source_uri", "physical-locator"),
    _leaf("run-plan", "references.artifacts.*.sha256", "action-semantic"),
    _leaf("run-plan", "references.artifacts.*.min_size_bytes", "action-semantic"),
    _leaf("run-plan", "references.artifacts.*.version", "action-semantic"),
    _branch("run-plan", "worker"),
    _leaf("run-plan", "worker.stages", "scientific", "action-semantic"),
    _leaf("run-plan", "worker.workers", "operational", "action-semantic"),
    _leaf("run-plan", "worker.batch_size", "encoding"),
    _leaf("run-plan", "worker.shards_per_archive", "encoding"),
    _leaf("run-plan", "worker.self_upload", "placement", "action-semantic"),
    _leaf("run-plan", "worker.local_scratch", "operational"),
    _leaf("run-plan", "worker.scratch_dir", "operational"),
    _leaf("run-plan", "worker.s5cmd_path", "operational"),
    _leaf("run-plan", "worker.upload_slots", "operational", "action-semantic"),
    _leaf("run-plan", "worker.s5cmd_numworkers", "operational", "action-semantic"),
    _leaf("run-plan", "worker.duckdb_memory_limit", "operational", "action-semantic"),
    _leaf("run-plan", "worker.heterodimers", "scientific", "action-semantic"),
    _leaf("run-plan", "worker.clash_device", "operational", "action-semantic"),
    _leaf("run-plan", "worker.clash_batch_size", "operational", "action-semantic"),
    _leaf("run-plan", "worker.dssp_algorithm", "scientific", "action-semantic"),
    _leaf("run-plan", "worker.parallel_stages", "operational", "action-semantic"),
    _leaf("run-plan", "worker.retry_failed_only", "scientific", "action-semantic"),
    _leaf("run-plan", "worker.retry_metadata_delta_tag", "scientific", "action-semantic"),
    _leaf("run-plan", "worker.tool_used", "scientific", "action-semantic"),
    _leaf("run-plan", "worker.homodimer_tool_used", "scientific", "action-semantic"),
    _leaf("run-plan", "worker.provider_id", "scientific", "action-semantic"),
    _leaf("run-plan", "worker.provider_name", "scientific", "action-semantic"),
    _leaf("run-plan", "worker.provider_url", "scientific", "action-semantic"),
    _leaf("run-plan", "worker.provider_copyrights", "scientific", "action-semantic"),
    _branch("run-plan", "storage"),
    _leaf("run-plan", "storage.s3_archive_prefix", "physical-locator"),
    _leaf("run-plan", "storage.s3_output_prefix", "placement"),
    _leaf("run-plan", "storage.gcs_destination_prefix", "placement"),
    _leaf("run-plan", "storage.upload_mode", "encoding"),
    _leaf("run-plan", "storage.s3_tar_prefix", "placement"),
    _leaf("run-plan", "storage.s3_tar_manifest_csv", "operational"),
    _leaf("run-plan", "storage.local_tar_dir", "operational"),
    _leaf("run-plan", "storage.local_tar_manifest_csv", "operational"),
    _leaf("run-plan", "storage.tar_compression", "encoding"),
    _leaf("run-plan", "storage.allow_production_prefixes", "operational", "action-semantic"),
    _branch("run-plan", "data_placement"),
    _branch("run-plan", "data_placement.*", wildcard_projection="concrete-key"),
    _leaf("run-plan", "data_placement.*.tool", "placement", "action-semantic"),
    _leaf("run-plan", "data_placement.*.required", "placement", "action-semantic"),
    _leaf("run-plan", "data_placement.*.source", "physical-locator"),
    _leaf("run-plan", "data_placement.*.destination", "placement"),
    _branch("run-plan", "analysis_metadata"),
    _leaf("run-plan", "analysis_metadata.enabled", "scientific", "action-semantic"),
    _leaf("run-plan", "analysis_metadata.csv_path", "operational"),
    _leaf("run-plan", "analysis_metadata.ipsae_threshold", "scientific", "action-semantic"),
    _leaf("run-plan", "analysis_metadata.pdockq2_threshold", "scientific", "action-semantic"),
    _leaf("run-plan", "analysis_metadata.finalize_after_gpu", "scientific", "action-semantic"),
    _leaf("run-plan", "analysis_metadata.parquet_path", "operational"),
    _leaf("run-plan", "analysis_metadata.selected_ids_path", "operational"),
    _leaf("run-plan", "analysis_metadata.chunk_size", "encoding"),
    _leaf("run-plan", "analysis_metadata.finalize_partition", "operational"),
    _leaf("run-plan", "analysis_metadata.finalize_cpus_per_task", "operational"),
    _leaf("run-plan", "analysis_metadata.finalize_memory", "operational"),
    _leaf("run-plan", "analysis_metadata.finalize_time", "operational"),
    _branch("run-plan", "analysis_metadata.high_quality_from_tars"),
    _leaf("run-plan", "analysis_metadata.high_quality_from_tars.enabled", "scientific", "action-semantic"),
    _leaf("run-plan", "analysis_metadata.high_quality_from_tars.s3_prefix", "physical-locator"),
    _leaf("run-plan", "analysis_metadata.high_quality_from_tars.work_dir", "operational"),
    _branch("run-plan", "analysis_metadata.high_quality_from_tars.publication"),
    _leaf("run-plan", "analysis_metadata.high_quality_from_tars.publication.enabled", "placement", "action-semantic"),
    _leaf("run-plan", "analysis_metadata.high_quality_from_tars.publication.target_prefix", "placement"),
    _leaf(
        "run-plan",
        "analysis_metadata.high_quality_from_tars.publication.default_target_prefix_from_recipe",
        "placement",
        "action-semantic",
    ),
    _leaf(
        "run-plan",
        "analysis_metadata.high_quality_from_tars.publication.collision_policy",
        "placement",
        "action-semantic",
    ),
    _leaf("run-plan", "analysis_metadata.high_quality_from_tars.publication.overwrite", "placement", "action-semantic"),
    _leaf("run-plan", "analysis_metadata.high_quality_from_tars.publication.chunk_size", "encoding"),
    _leaf("run-plan", "analysis_metadata.high_quality_from_tars.publication.manifest_dir", "operational"),
    _leaf("run-plan", "analysis_metadata.high_quality_from_tars.publication.evidence_dir", "operational"),
    _leaf("run-plan", "analysis_metadata.high_quality_from_tars.publication.sample_download_count", "acceptance"),
    _leaf("run-plan", "analysis_metadata.high_quality_from_tars.publication.s5cmd_numworkers", "operational"),
    _branch("run-plan", "validation"),
    _leaf("run-plan", "validation.expected_archives", "acceptance"),
    _leaf("run-plan", "validation.expected_allowed_ids", "acceptance"),
    _leaf("run-plan", "validation.expected_one_archive_models", "acceptance"),
    _leaf("run-plan", "validation.expected_one_archive_objects", "acceptance"),
    _leaf("run-plan", "validation.expected_one_archive_aggregate_rows", "acceptance"),
    _leaf("run-plan", "validation.expected_tar_count", "acceptance"),
    _leaf("run-plan", "validation.expected_local_tars_rows", "acceptance"),
    _leaf("run-plan", "validation.expected_failed_rows", "acceptance"),
    _leaf("run-plan", "validation.expected_analysis_rows", "acceptance"),
    _leaf("run-plan", "validation.expected_selected_ids", "acceptance"),
    _leaf("run-plan", "validation.gcs_dry_run_only", "acceptance"),
    _branch("run-plan", "acceptance"),
    _leaf("run-plan", "acceptance.baseline_output_dir", "physical-locator"),
    _leaf("run-plan", "acceptance.baseline_run_name", "acceptance"),
    _leaf("run-plan", "acceptance.candidate_run_name", "generated"),
    _leaf("run-plan", "acceptance.tar_payload_match_mode", "acceptance"),
    _leaf("run-plan", "acceptance.payload_sample_count", "acceptance"),
    _leaf("run-plan", "acceptance.candidate_parquet_required", "acceptance"),
    _leaf("run-plan", "acceptance.compare_failed_sets", "acceptance"),
    _leaf("run-plan", "acceptance.compare_tar_manifest_rows", "acceptance"),
    _leaf("run-plan", "acceptance.compare_analysis_model_rows", "acceptance"),
    _branch("run-plan", "secrets"),
    _branch("run-plan", "secrets.*", wildcard_projection="concrete-key"),
    _leaf("run-plan", "secrets.*.scheme", "secret"),
    _leaf("run-plan", "secrets.*.target", "secret"),
)

_WORKFLOW_RULES: tuple[PostprocessingFieldRule, ...] = (
    _branch("workflow-step", "workflow"),
    _branch("workflow-step", "workflow.steps"),
    _branch("workflow-step", "workflow.steps.*", wildcard_projection="concrete-key"),
    _leaf("workflow-step", "workflow.steps.*.name", "action-semantic"),
    _leaf("workflow-step", "workflow.steps.*.run", "action-semantic"),
    _leaf("workflow-step", "workflow.steps.*.mode", "action-semantic"),
    _leaf("workflow-step", "workflow.steps.*.job_id", "operational"),
    _leaf("workflow-step", "workflow.steps.*.array_range", "action-semantic"),
    _leaf("workflow-step", "workflow.steps.*.rendered_script", "operational"),
)

_LEGACY_EXECUTION_RULES: tuple[PostprocessingFieldRule, ...] = (
    _branch("legacy-execution", "cluster"),
    _leaf("legacy-execution", "cluster.name", "resource"),
    _leaf("legacy-execution", "cluster.account", "resource"),
    _leaf("legacy-execution", "cluster.owner", "resource"),
    _branch("legacy-execution", "paths"),
    _leaf("legacy-execution", "paths.project_root", "physical-locator"),
    _leaf("legacy-execution", "paths.staging_dir", "physical-locator"),
    _leaf("legacy-execution", "paths.output_dir", "physical-locator"),
    _leaf("legacy-execution", "paths.log_dir", "physical-locator"),
    _leaf("legacy-execution", "paths.legacy_repo", "physical-locator"),
    _leaf("legacy-execution", "paths.afdb_toolkit_repo", "physical-locator"),
    _leaf("legacy-execution", "paths.orchestration_repo", "physical-locator"),
    _leaf("legacy-execution", "paths.recipe_dir", "physical-locator"),
    _branch("legacy-execution", "container"),
    _leaf("legacy-execution", "container.image", "physical-locator"),
    _leaf("legacy-execution", "container.workdir", "operational"),
    _branch("legacy-execution", "container.mounts"),
    _branch("legacy-execution", "container.mounts.*", wildcard_projection="concrete-key"),
    _leaf("legacy-execution", "container.mounts.*.source", "physical-locator"),
    _leaf("legacy-execution", "container.mounts.*.target", "operational"),
    _leaf("legacy-execution", "container.mounts.*.read_only", "operational"),
    _branch("legacy-execution", "resources"),
    _branch("legacy-execution", "resources.*", wildcard_projection="sorted-record"),
    _leaf("legacy-execution", "resources.*.partition", "resource"),
    _leaf("legacy-execution", "resources.*.cpus_per_task", "resource"),
    _leaf("legacy-execution", "resources.*.memory", "resource"),
    _leaf("legacy-execution", "resources.*.time", "resource"),
    _leaf("legacy-execution", "resources.*.gres", "resource"),
    _leaf("legacy-execution", "resources.*.array", "resource"),
    _leaf("legacy-execution", "resources.*.nodelist", "resource"),
    _leaf("legacy-execution", "resources.*.nodes", "resource", optional=True),
    _leaf("legacy-execution", "resources.*.tasks_per_node", "resource", optional=True),
    _leaf("legacy-execution", "resources.*.gpus_per_task", "resource", optional=True),
    _leaf("legacy-execution", "resources.*.max_parallel", "resource", optional=True),
    _branch("legacy-execution", "submission"),
    _leaf("legacy-execution", "submission.evidence_dir", "physical-locator"),
    _leaf("legacy-execution", "submission.report_path", "physical-locator"),
    _leaf("legacy-execution", "submission.controller_runtime", "operational"),
    _leaf("legacy-execution", "source_path", "physical-locator"),
    _leaf("legacy-execution", "source_hash", "operational"),
    _branch("legacy-execution", "phase_cluster"),
    _leaf("legacy-execution", "phase_cluster.schema_version", "operational"),
    _leaf("legacy-execution", "phase_cluster.profile_name", "resource"),
    _leaf("legacy-execution", "phase_cluster.owner", "resource"),
    _leaf("legacy-execution", "phase_cluster.transport", "resource"),
    _leaf("legacy-execution", "phase_cluster.ssh_target", "physical-locator"),
    _leaf("legacy-execution", "phase_cluster.account", "resource"),
    _leaf("legacy-execution", "phase_cluster.project_root", "physical-locator"),
    _leaf("legacy-execution", "phase_cluster.staging_root", "physical-locator"),
    _leaf("legacy-execution", "phase_cluster.orchestration_repo", "physical-locator"),
    _leaf("legacy-execution", "phase_cluster.runtime_image", "physical-locator"),
    _branch("legacy-execution", "phase_cluster.extra_mounts"),
    _branch("legacy-execution", "phase_cluster.extra_mounts.*", wildcard_projection="concrete-key"),
    _leaf("legacy-execution", "phase_cluster.extra_mounts.*.schema_version", "operational"),
    _leaf("legacy-execution", "phase_cluster.extra_mounts.*.source", "physical-locator"),
    _leaf("legacy-execution", "phase_cluster.extra_mounts.*.target", "operational"),
    _leaf("legacy-execution", "phase_cluster.extra_mounts.*.read_only", "operational"),
    _branch("legacy-execution", "qualified_runtime"),
    _leaf("legacy-execution", "qualified_runtime.schema_version", "operational"),
    _leaf("legacy-execution", "qualified_runtime.selection_kind", "operational"),
    _leaf("legacy-execution", "qualified_runtime.tuple_id", "operational"),
    _leaf("legacy-execution", "qualified_runtime.qualification_location", "physical-locator"),
    _leaf("legacy-execution", "qualified_runtime.qualification_sha256", "operational"),
    _leaf("legacy-execution", "qualified_runtime.qualification_size_bytes", "operational"),
    _leaf("legacy-execution", "qualified_runtime.qualified_at", "operational"),
    _leaf("legacy-execution", "qualified_runtime.expires_at", "operational"),
    _leaf("legacy-execution", "qualified_runtime.image_path", "physical-locator"),
    _leaf("legacy-execution", "qualified_runtime.image_sha256", "operational"),
    _leaf("legacy-execution", "qualified_runtime.image_size_bytes", "operational"),
    _leaf("legacy-execution", "qualified_runtime.image_policy", "operational"),
    _leaf("legacy-execution", "qualified_runtime.source_kind", "operational"),
    _leaf("legacy-execution", "qualified_runtime.source_revision", "operational"),
    _leaf("legacy-execution", "qualified_runtime.source_package_path", "physical-locator"),
    _leaf("legacy-execution", "qualified_runtime.toolkit_package_path", "physical-locator"),
    _leaf("legacy-execution", "qualified_runtime.runtime_ipsae_binary_path", "physical-locator"),
    _leaf("legacy-execution", "qualified_runtime.runtime_ipsae_binary_sha256", "operational"),
    _leaf("legacy-execution", "qualified_runtime.runtime_ipsae_binary_size_bytes", "operational"),
    _leaf("legacy-execution", "qualified_runtime.source_identity_digest", "operational"),
    _leaf("legacy-execution", "qualified_runtime.source_package_identity_digest", "operational"),
    _leaf("legacy-execution", "qualified_runtime.toolkit_identity_digest", "operational"),
    _leaf("legacy-execution", "qualified_runtime.bootstrap_sha256", "operational"),
    _leaf("legacy-execution", "qualified_runtime.runtime_component_identity_digest", "operational"),
    _leaf("legacy-execution", "qualified_runtime.requeue_exit", "operational", optional=True),
    _leaf("legacy-execution", "qualified_runtime.max_batch_requeue", "operational", optional=True),
    _branch("legacy-execution", "attempt_paths"),
    _leaf("legacy-execution", "attempt_paths.schema_version", "operational"),
    _leaf("legacy-execution", "attempt_paths.legacy_run_id", "operational"),
    _leaf("legacy-execution", "attempt_paths.output_dir", "placement"),
    _leaf("legacy-execution", "attempt_paths.evidence_dir", "placement"),
    _leaf("legacy-execution", "attempt_paths.staging_dir", "physical-locator"),
    _leaf("legacy-execution", "attempt_paths.object_prefix", "placement"),
)

POSTPROCESSING_FIELD_RULES: tuple[PostprocessingFieldRule, ...] = (
    *_RUN_PLAN_RULES,
    *_WORKFLOW_RULES,
    *_LEGACY_EXECUTION_RULES,
)

_RULES_BY_SCOPE = {
    scope: tuple(rule for rule in POSTPROCESSING_FIELD_RULES if rule.scope == scope)
    for scope in ("run-plan", "workflow-step", "legacy-execution")
}


def walk_postprocessing_fields(
    values: Mapping[str, object],
    *,
    scope: PostprocessingFieldScope,
) -> tuple[PostprocessingClassifiedField, ...]:
    """Classify every concrete branch and leaf in one registered scope."""
    result: list[PostprocessingClassifiedField] = []
    _walk_value(values, scope=scope, path=(), result=result, sorted_record=None)
    return tuple(result)


def run_plan_field_walk(plan: RunPlan | Mapping[str, object]) -> tuple[PostprocessingClassifiedField, ...]:
    values = plan.model_dump(mode="json") if isinstance(plan, RunPlan) else plan
    return walk_postprocessing_fields(values, scope="run-plan")


def workflow_field_walk(spec: RunSpec) -> tuple[PostprocessingClassifiedField, ...]:
    if spec.workflow is None:
        raise ValueError("postprocessing semantic projection requires a workflow")
    return walk_postprocessing_fields(
        {"workflow": spec.workflow.model_dump(mode="json")},
        scope="workflow-step",
    )


def legacy_execution_field_walk(
    spec: RunSpec,
    *,
    phase_cluster: PostprocessingClusterSnapshot | None = None,
    qualified_runtime: QualifiedPostprocessingRuntimeSelection | None = None,
    attempt_paths: PostprocessingAttemptPaths | None = None,
) -> tuple[PostprocessingClassifiedField, ...]:
    values = spec.model_dump(mode="json")
    execution = {
        key: values[key]
        for key in ("cluster", "paths", "container", "resources", "submission", "source_path", "source_hash")
    }
    execution.update(
        {
            "phase_cluster": _phase_cluster_registry_mapping(phase_cluster)
            if phase_cluster is not None
            else _null_dataclass_mapping(PostprocessingClusterSnapshot, sequence_fields={"extra_mounts"}),
            "qualified_runtime": qualified_runtime.to_mapping()
            if qualified_runtime is not None
            else _null_dataclass_mapping(QualifiedPostprocessingRuntimeSelection),
            "attempt_paths": attempt_paths.to_mapping()
            if attempt_paths is not None
            else _null_dataclass_mapping(PostprocessingAttemptPaths),
        }
    )
    return walk_postprocessing_fields(execution, scope="legacy-execution")


def _phase_cluster_registry_mapping(cluster: PostprocessingClusterSnapshot) -> dict[str, object]:
    """Return the registry-facing phase_cluster mapping with default read_only materialized.

    ``PhaseMountSnapshot.to_mapping()`` omits ``read_only`` when false so historical
    authority stays byte-stable, while the registry declares
    ``phase_cluster.extra_mounts.*.read_only`` as a required fixed child — the same
    posture as the legacy ``container.mounts`` model, whose dump always carries
    ``read_only: false``. Materialize the default here so the walk classifies the
    leaf for every mount instead of rejecting default (read-write) mounts.
    """
    mapping = cluster.to_mapping()
    mounts = mapping.get("extra_mounts")
    if isinstance(mounts, list):
        for mount in mounts:
            if isinstance(mount, dict):
                mount.setdefault("read_only", False)
    return mapping


def _null_dataclass_mapping(
    value_type: type[object],
    *,
    sequence_fields: frozenset[str] | set[str] = frozenset(),
) -> dict[str, object]:
    return {
        field.name: [] if field.name in sequence_fields else None
        for field in dataclass_fields(value_type)  # type: ignore[arg-type]
    }


def rule_paths_for_component(component: str) -> frozenset[str]:
    """Return declared direct Run Plan children for compatibility tests."""
    prefix = (component,)
    return frozenset(
        rule.pattern[len(prefix)]
        for rule in _RUN_PLAN_RULES
        if rule.pattern[: len(prefix)] == prefix
        and len(rule.pattern) > len(prefix)
        and rule.pattern[len(prefix)] != "*"
    )


def has_positive_identity_role(rule: PostprocessingFieldRule) -> bool:
    return bool(rule.roles & _POSITIVE_ROLES)


def is_pure_exclusion(rule: PostprocessingFieldRule) -> bool:
    return bool(rule.roles & _EXCLUSION_ROLES) and not has_positive_identity_role(rule)


def identity_normalized_runspec(spec: RunSpec) -> RunSpec:
    """Return a typed copy with pure-exclusion values replaced before command resolution."""
    plan_values = spec.model_dump(mode="json")
    synthetic_plan = {
        "schema_version": plan_values["schema_version"],
        "run_kind": plan_values["run_kind"],
        "target_cluster": plan_values["cluster"]["name"],
        "workflow_template": "",
        **{
            component: plan_values[component]
            for component in (
                "dataset",
                "references",
                "worker",
                "storage",
                "data_placement",
                "analysis_metadata",
                "validation",
                "acceptance",
                "secrets",
            )
        },
    }
    replacements: dict[tuple[str | int, ...], object] = {}
    run_plan_roots = frozenset(synthetic_plan) - {"target_cluster", "workflow_template"}
    for field in run_plan_field_walk(synthetic_plan):
        if field.path[0] in run_plan_roots:
            _register_identity_replacement(replacements, spec, field)
    for field in workflow_field_walk(spec):
        _register_identity_replacement(replacements, spec, field)
    legacy_roots = frozenset({"cluster", "paths", "container", "resources", "submission", "source_path", "source_hash"})
    for field in legacy_execution_field_walk(spec):
        if field.path[0] in legacy_roots:
            _register_identity_replacement(replacements, spec, field)

    normalized: object = spec
    for path, replacement in sorted(replacements.items(), key=lambda item: repr(item[0])):
        normalized = _replace_typed_path(normalized, path, replacement)
    if not isinstance(normalized, RunSpec):
        raise TypeError("identity normalization did not preserve the RunSpec type")
    return normalized


def _register_identity_replacement(
    replacements: dict[tuple[str | int, ...], object],
    spec: RunSpec,
    field: PostprocessingClassifiedField,
) -> None:
    if field.rule.kind != "leaf" or not is_pure_exclusion(field.rule):
        return
    path = _concrete_typed_path(spec, field.path)
    if path is None:
        return
    replacements[path] = _identity_placeholder(
        _typed_path_value(spec, path),
        annotation=_typed_leaf_annotation(spec, path),
        scope=field.scope,
        path=field.dotted_path,
    )


def _concrete_typed_path(value: object, path: tuple[str, ...]) -> tuple[str | int, ...] | None:
    current = value
    result: list[str | int] = []
    for raw_part in path:
        if current is None:
            return None
        part: str | int = int(raw_part) if isinstance(current, list | tuple) else raw_part
        result.append(part)
        if isinstance(current, BaseModel):
            current = getattr(current, raw_part)
        elif isinstance(current, Mapping):
            current = current[raw_part]
        elif isinstance(current, list | tuple):
            assert isinstance(part, int)
            current = current[part]
        else:
            raise TypeError(f"cannot descend through {type(current).__name__} in identity path")
    return tuple(result)


def _typed_leaf_annotation(value: object, path: tuple[str | int, ...]) -> object | None:
    if not path:
        return None
    parent = _typed_path_value(value, path[:-1])
    leaf = path[-1]
    if isinstance(parent, BaseModel) and isinstance(leaf, str):
        return type(parent).model_fields[leaf].annotation
    return None


def _typed_path_value(value: object, path: tuple[str | int, ...]) -> object:
    current = value
    for part in path:
        if isinstance(current, BaseModel):
            if not isinstance(part, str):
                raise TypeError("model identity paths require string field names")
            current = getattr(current, part)
        elif isinstance(current, Mapping):
            current = current[part]
        elif isinstance(current, list | tuple):
            if not isinstance(part, int):
                raise TypeError("sequence identity paths require integer indexes")
            current = current[part]
        else:
            raise TypeError(f"cannot descend through {type(current).__name__} in identity path")
    return current


def _replace_typed_path(value: object, path: tuple[str | int, ...], replacement: object) -> object:
    if not path:
        return replacement
    head, *tail = path
    remaining = tuple(tail)
    if isinstance(value, BaseModel):
        if not isinstance(head, str):
            raise TypeError("model identity paths require string field names")
        nested = getattr(value, head)
        return value.model_copy(update={head: _replace_typed_path(nested, remaining, replacement)})
    if isinstance(value, Mapping):
        nested = dict(value)
        nested[head] = _replace_typed_path(nested[head], remaining, replacement)
        return nested
    if isinstance(value, tuple):
        if not isinstance(head, int):
            raise TypeError("sequence identity paths require integer indexes")
        nested_tuple = list(value)
        nested_tuple[head] = _replace_typed_path(nested_tuple[head], remaining, replacement)
        return tuple(nested_tuple)
    if isinstance(value, list):
        if not isinstance(head, int):
            raise TypeError("sequence identity paths require integer indexes")
        nested_list = list(value)
        nested_list[head] = _replace_typed_path(nested_list[head], remaining, replacement)
        return nested_list
    raise TypeError(f"cannot descend through {type(value).__name__} in identity path")


def _identity_placeholder(
    value: object,
    *,
    annotation: object | None,
    scope: str,
    path: str,
) -> object:
    marker = f"{scope}/{path.replace('.', '/')}"
    concrete_annotation = _non_none_annotation(annotation)
    if annotation is not None and _annotation_allows_none(annotation):
        return _placeholder_from_annotation(concrete_annotation, marker=marker, path=path)
    if isinstance(value, BaseModel):
        return value.model_copy(
            update={
                name: _identity_placeholder(
                    getattr(value, name),
                    annotation=type(value).model_fields[name].annotation,
                    scope=scope,
                    path=f"{path}.{name}",
                )
                for name in type(value).model_fields
            }
        )
    if isinstance(value, Path):
        return Path(f"/__bspp_identity__/{marker}")
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 1
    if isinstance(value, float):
        return 1.0
    if isinstance(value, str):
        if len(value) in {40, 64} and all(character in "0123456789abcdef" for character in value):
            return "0" * len(value)
        return f"<RUNSPEC:{scope}:{path}>"
    if isinstance(value, Mapping):
        return {
            key: _identity_placeholder(item, annotation=None, scope=scope, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _identity_placeholder(item, annotation=None, scope=scope, path=f"{path}.{index:04d}")
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _identity_placeholder(item, annotation=None, scope=scope, path=f"{path}.{index:04d}")
            for index, item in enumerate(value)
        ]
    if value is None:
        return _placeholder_from_annotation(concrete_annotation, marker=marker, path=path)
    raise TypeError(f"unsupported identity placeholder type: {type(value).__name__}")


def _annotation_allows_none(annotation: object) -> bool:
    annotation = _unwrap_annotated(annotation)
    origin = get_origin(annotation)
    return origin in (Union, types.UnionType) and type(None) in get_args(annotation)


def _non_none_annotation(annotation: object | None) -> object | None:
    if annotation is None:
        return None
    annotation = _unwrap_annotated(annotation)
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        members = tuple(item for item in get_args(annotation) if item is not type(None))
        return _unwrap_annotated(members[0]) if len(members) == 1 else annotation
    return annotation


def _unwrap_annotated(annotation: object) -> object:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _placeholder_from_annotation(annotation: object | None, *, marker: str, path: str) -> object:
    if annotation is not None:
        annotation = _unwrap_annotated(annotation)
    origin = get_origin(annotation)
    if origin is Literal:
        members = tuple(item for item in get_args(annotation) if item is not None)
        if members:
            return members[0]
    if annotation is Path or (isinstance(annotation, type) and issubclass(annotation, Path)):
        return Path(f"/__bspp_identity__/{marker}")
    if annotation is str:
        if path.endswith(("sha256", "digest", "hash")):
            return "0" * 64
        if path.endswith("revision"):
            return "0" * 40
        return f"<RUNSPEC:{marker}>"
    if annotation is bool:
        return False
    if annotation is int:
        return 1
    if annotation is float:
        return 1.0
    raise TypeError(f"cannot construct an identity placeholder for annotation {annotation!r} at {path}")


def _walk_value(
    value: object,
    *,
    scope: PostprocessingFieldScope,
    path: tuple[str, ...],
    result: list[PostprocessingClassifiedField],
    sorted_record: tuple[tuple[str, ...], str] | None,
) -> None:
    matched_rule: PostprocessingFieldRule | None = None
    if path:
        rule = _matching_rule(scope, path)
        matched_rule = rule
        if sorted_record is not None:
            record_pattern, record_key = sorted_record
        else:
            record_pattern = None
            record_key = None
        result.append(
            PostprocessingClassifiedField(
                scope=scope,
                path=path,
                value=value,
                rule=rule,
                sorted_record_key=record_key,
                sorted_record_pattern=record_pattern,
            )
        )
        if rule.kind == "leaf":
            return
        if rule.wildcard_projection == "sorted-record":
            wildcard_index = rule.pattern.index("*")
            sorted_record = (rule.pattern, path[wildcard_index])
    if value is None:
        if matched_rule is not None and matched_rule.wildcard_projection is not None:
            return
        _walk_null_children(scope=scope, path=path, result=result, sorted_record=sorted_record)
        return
    if isinstance(value, Mapping):
        _validate_fixed_children(value, scope=scope, path=path)
        for key, nested in value.items():
            _walk_value(
                nested,
                scope=scope,
                path=(*path, str(key)),
                result=result,
                sorted_record=sorted_record,
            )
        for child in sorted(_optional_children(scope=scope, path=path) - {str(key) for key in value}):
            _walk_value(
                None,
                scope=scope,
                path=(*path, child),
                result=result,
                sorted_record=sorted_record,
            )
        return
    if isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            component = f"{index:02d}" if scope == "workflow-step" else f"{index:04d}"
            _walk_value(
                nested,
                scope=scope,
                path=(*path, component),
                result=result,
                sorted_record=sorted_record,
            )
        return
    if not path:
        raise ValueError("postprocessing registry scope root must be a mapping")


def _walk_null_children(
    *,
    scope: PostprocessingFieldScope,
    path: tuple[str, ...],
    result: list[PostprocessingClassifiedField],
    sorted_record: tuple[tuple[str, ...], str] | None,
) -> None:
    children = sorted(
        {
            rule.pattern[len(path)]
            for rule in _RULES_BY_SCOPE[scope]
            if len(rule.pattern) > len(path)
            and _pattern_matches(rule.pattern[: len(path)], path)
            and rule.pattern[len(path)] != "*"
        }
    )
    for child in children:
        _walk_value(None, scope=scope, path=(*path, child), result=result, sorted_record=sorted_record)


def _optional_children(
    *,
    scope: PostprocessingFieldScope,
    path: tuple[str, ...],
) -> frozenset[str]:
    return frozenset(
        rule.pattern[len(path)]
        for rule in _RULES_BY_SCOPE[scope]
        if len(rule.pattern) > len(path)
        and _pattern_matches(rule.pattern[: len(path)], path)
        and rule.pattern[len(path)] != "*"
        and rule.optional
    )


def _validate_fixed_children(
    value: Mapping[object, object],
    *,
    scope: PostprocessingFieldScope,
    path: tuple[str, ...],
) -> None:
    declared = {
        rule.pattern[len(path)]
        for rule in _RULES_BY_SCOPE[scope]
        if len(rule.pattern) > len(path)
        and _pattern_matches(rule.pattern[: len(path)], path)
        and rule.pattern[len(path)] != "*"
        and not rule.optional
    }
    actual = {str(key) for key in value}
    missing = declared - actual
    if missing:
        raise ValueError(
            f"postprocessing field registry mapping is missing declared children at {'.'.join(path) or '<root>'}: "
            f"{sorted(missing)!r}"
        )


def _matching_rule(scope: PostprocessingFieldScope, path: tuple[str, ...]) -> PostprocessingFieldRule:
    matches = tuple(rule for rule in _RULES_BY_SCOPE[scope] if _pattern_matches(rule.pattern, path))
    if len(matches) != 1:
        raise ValueError(
            f"postprocessing field registry path must match exactly one rule: {scope}:{'.'.join(path)} "
            f"matched {len(matches)}"
        )
    return matches[0]


def _pattern_matches(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    return len(pattern) == len(path) and all(
        expected == "*" or expected == actual for expected, actual in zip(pattern, path, strict=True)
    )


def _validate_registry_schema() -> None:
    matched: set[tuple[PostprocessingFieldScope, tuple[str, ...]]] = set()
    _validate_model_schema(RunPlan, scope="run-plan", prefix=(), matched=matched)
    _require_schema_rule("workflow-step", ("workflow",), "branch", matched)
    _validate_model_schema(WorkflowSpec, scope="workflow-step", prefix=("workflow",), matched=matched)
    for name, model in (
        ("cluster", ClusterSpec),
        ("paths", PathSpec),
        ("container", ContainerSpec),
        ("submission", SubmissionSpec),
    ):
        _require_schema_rule("legacy-execution", (name,), "branch", matched)
        _validate_model_schema(model, scope="legacy-execution", prefix=(name,), matched=matched)
    _require_schema_rule("legacy-execution", ("resources",), "branch", matched)
    _require_schema_rule("legacy-execution", ("resources", "*"), "branch", matched)
    _validate_model_schema(
        SlurmResources,
        scope="legacy-execution",
        prefix=("resources", "*"),
        matched=matched,
    )
    _require_schema_rule("legacy-execution", ("source_path",), "leaf", matched)
    _require_schema_rule("legacy-execution", ("source_hash",), "leaf", matched)
    _validate_dataclass_schema(
        PostprocessingClusterSnapshot,
        scope="legacy-execution",
        prefix=("phase_cluster",),
        matched=matched,
        nested_sequences={"extra_mounts": PhaseMountSnapshot},
    )
    _validate_dataclass_schema(
        QualifiedPostprocessingRuntimeSelection,
        scope="legacy-execution",
        prefix=("qualified_runtime",),
        matched=matched,
    )
    _validate_dataclass_schema(
        PostprocessingAttemptPaths,
        scope="legacy-execution",
        prefix=("attempt_paths",),
        matched=matched,
    )
    declared = {(rule.scope, rule.pattern) for rule in POSTPROCESSING_FIELD_RULES}
    if declared != matched:
        missing = sorted(f"{scope}:{'.'.join(path)}" for scope, path in declared - matched)
        extra = sorted(f"{scope}:{'.'.join(path)}" for scope, path in matched - declared)
        raise RuntimeError(f"postprocessing registry schema mismatch; unreachable={missing!r}, unknown={extra!r}")


def _validate_model_schema(
    model: type[BaseModel],
    *,
    scope: PostprocessingFieldScope,
    prefix: tuple[str, ...],
    matched: set[tuple[PostprocessingFieldScope, tuple[str, ...]]],
) -> None:
    for name, field in model.model_fields.items():
        path = (*prefix, name)
        shape, nested = _annotation_shape(field.annotation)
        if shape == "leaf":
            _require_schema_rule(scope, path, "leaf", matched)
            continue
        _require_schema_rule(scope, path, "branch", matched)
        assert nested is not None
        if shape in {"mapping", "sequence"}:
            path = (*path, "*")
            _require_schema_rule(scope, path, "branch", matched)
        _validate_model_schema(nested, scope=scope, prefix=path, matched=matched)


def _validate_dataclass_schema(
    model: type[object],
    *,
    scope: PostprocessingFieldScope,
    prefix: tuple[str, ...],
    matched: set[tuple[PostprocessingFieldScope, tuple[str, ...]]],
    nested_sequences: Mapping[str, type[object]] | None = None,
) -> None:
    _require_schema_rule(scope, prefix, "branch", matched)
    nested_sequences = nested_sequences or {}
    for field in dataclass_fields(model):  # type: ignore[arg-type]
        path = (*prefix, field.name)
        nested = nested_sequences.get(field.name)
        if nested is None:
            _require_schema_rule(scope, path, "leaf", matched)
            continue
        _require_schema_rule(scope, path, "branch", matched)
        wildcard_path = (*path, "*")
        _require_schema_rule(scope, wildcard_path, "branch", matched)
        _validate_dataclass_schema(nested, scope=scope, prefix=wildcard_path, matched=matched)


def _annotation_shape(
    annotation: object,
) -> tuple[Literal["leaf", "model", "mapping", "sequence"], type[BaseModel] | None]:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        members = tuple(item for item in get_args(annotation) if item is not type(None))
        if len(members) == 1:
            return _annotation_shape(members[0])
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return "model", annotation
    if origin is not None:
        arguments = get_args(annotation)
        if (origin in (dict, Mapping) or (isinstance(origin, type) and issubclass(origin, Mapping))) and len(
            arguments
        ) == 2:
            nested_shape, nested = _annotation_shape(arguments[1])
            if nested_shape == "model":
                return "mapping", nested
        if (
            origin in (list, tuple, Sequence) or (isinstance(origin, type) and issubclass(origin, Sequence))
        ) and arguments:
            nested_shape, nested = _annotation_shape(arguments[0])
            if nested_shape == "model":
                return "sequence", nested
    return "leaf", None


def _require_schema_rule(
    scope: PostprocessingFieldScope,
    path: tuple[str, ...],
    kind: PostprocessingFieldKind,
    matched: set[tuple[PostprocessingFieldScope, tuple[str, ...]]],
) -> None:
    matches = tuple(rule for rule in _RULES_BY_SCOPE[scope] if _pattern_matches(rule.pattern, path))
    if len(matches) != 1 or matches[0].kind != kind:
        raise RuntimeError(f"postprocessing schema path must have exactly one {kind} rule: {scope}:{'.'.join(path)}")
    matched.add((scope, matches[0].pattern))


_validate_registry_schema()


__all__ = [
    "POSTPROCESSING_FIELD_RULES",
    "PostprocessingClassifiedField",
    "PostprocessingFieldKind",
    "PostprocessingFieldRole",
    "PostprocessingFieldRule",
    "PostprocessingFieldScope",
    "PostprocessingWildcardProjection",
    "has_positive_identity_role",
    "identity_normalized_runspec",
    "is_pure_exclusion",
    "legacy_execution_field_walk",
    "rule_paths_for_component",
    "run_plan_field_walk",
    "walk_postprocessing_fields",
    "workflow_field_walk",
]
