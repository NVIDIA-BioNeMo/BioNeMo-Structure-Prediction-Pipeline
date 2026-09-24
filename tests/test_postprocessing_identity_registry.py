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

"""Exhaustive mutation locks for the postprocessing identity registry."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import is_dataclass
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from bspp.orchestration.contract.phase import PhaseMountSnapshot, canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    PostprocessingAcceptancePolicySnapshot,
    postprocessing_acceptance_policy_from_mapping,
)
from bspp.orchestration.contract.postprocessing_acceptance_reference import (
    PostprocessingAcceptanceSnapshotReference,
)
from bspp.orchestration.contract.postprocessing_action_contract import (
    postprocessing_action_graph_digest,
)
from bspp.orchestration.contract.postprocessing_execution import (
    PostprocessingAttemptPaths,
    PostprocessingClusterSnapshot,
    PostprocessingExecutionProjection,
    PostprocessingPhaseExecutionIdentity,
    QualifiedPostprocessingRuntimeSelection,
)
from bspp.orchestration.contract.postprocessing_execution_parsing import (
    postprocessing_attempt_paths_from_mapping,
    postprocessing_cluster_snapshot_from_mapping,
    qualified_postprocessing_runtime_from_mapping,
)
from bspp.orchestration.contract.postprocessing_logical_identity import (
    PhysicalInputLocator,
    PostprocessingLogicalInputIdentityManifestV2,
    PostprocessingLogicalInputIdentityV2,
)
from bspp.orchestration.contract.postprocessing_runspec_v2 import (
    PostprocessingPhaseRunSpec,
    PostprocessingPhaseRunSpecPayload,
)
from bspp.orchestration.contract.runplan import RunPlan
from bspp.orchestration.contract.runspec import RunSpec, WorkflowSpec, WorkflowStepSpec, runspec_from_mapping
from bspp.orchestration.control.plan import materialize_runspec_mapping, render_runspec_yaml
from bspp.orchestration.control.postprocessing_identity import (
    build_action_semantics,
    build_scientific_identity,
    project_run_plan_fields,
    project_workflow_fields,
)
from bspp.orchestration.control.postprocessing_identity_registry import (
    POSTPROCESSING_FIELD_RULES,
    PostprocessingFieldRule,
    identity_normalized_runspec,
    legacy_execution_field_walk,
    run_plan_field_walk,
    walk_postprocessing_fields,
    workflow_field_walk,
)
from bspp.orchestration.control.postprocessing_phase_materialization import postprocessing_runtime_actions
from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.control.workflow_rendering import resolve_workflow_step_command
from bspp.orchestration.control.workflows import load_workflow_template

EXAMPLE = Path(__file__).parents[1] / "tests" / "fixtures" / "postprocessing_phase" / "neutral_v3"
POSITIVE_ACTION_ROLES = frozenset({"scientific", "action-semantic", "encoding", "acceptance"})

RUN_PLAN_MUTATION_PATHS = (
    "schema_version",
    "run_kind",
    "target_cluster",
    "workflow_template",
    "dataset.name",
    "dataset.run_id",
    "dataset.mode",
    "dataset.array",
    "dataset.archive_source",
    "references.master_parquet",
    "references.tracking_parquet",
    "references.manifest_csv",
    "references.uniprot_duckdb",
    "references.heterodimer_id_manifest",
    "references.artifacts.extra_reference.path",
    "references.artifacts.extra_reference.source_uri",
    "references.artifacts.extra_reference.sha256",
    "references.artifacts.extra_reference.min_size_bytes",
    "references.artifacts.extra_reference.version",
    "references.artifacts.master_parquet.path",
    "references.artifacts.master_parquet.source_uri",
    "references.artifacts.master_parquet.sha256",
    "references.artifacts.master_parquet.min_size_bytes",
    "references.artifacts.master_parquet.version",
    "references.artifacts.tracking_parquet.path",
    "references.artifacts.tracking_parquet.source_uri",
    "references.artifacts.tracking_parquet.sha256",
    "references.artifacts.tracking_parquet.min_size_bytes",
    "references.artifacts.tracking_parquet.version",
    "references.artifacts.manifest_csv.path",
    "references.artifacts.manifest_csv.source_uri",
    "references.artifacts.manifest_csv.sha256",
    "references.artifacts.manifest_csv.min_size_bytes",
    "references.artifacts.manifest_csv.version",
    "references.artifacts.uniprot_duckdb.path",
    "references.artifacts.uniprot_duckdb.source_uri",
    "references.artifacts.uniprot_duckdb.sha256",
    "references.artifacts.uniprot_duckdb.min_size_bytes",
    "references.artifacts.uniprot_duckdb.version",
    "references.artifacts.heterodimer_id_manifest.path",
    "references.artifacts.heterodimer_id_manifest.source_uri",
    "references.artifacts.heterodimer_id_manifest.sha256",
    "references.artifacts.heterodimer_id_manifest.min_size_bytes",
    "references.artifacts.heterodimer_id_manifest.version",
    "worker.stages",
    "worker.workers",
    "worker.batch_size",
    "worker.shards_per_archive",
    "worker.self_upload",
    "worker.local_scratch",
    "worker.scratch_dir",
    "worker.s5cmd_path",
    "worker.upload_slots",
    "worker.s5cmd_numworkers",
    "worker.duckdb_memory_limit",
    "worker.heterodimers",
    "worker.clash_device",
    "worker.clash_batch_size",
    "worker.dssp_algorithm",
    "worker.parallel_stages",
    "worker.retry_failed_only",
    "worker.retry_metadata_delta_tag",
    "worker.tool_used",
    "worker.homodimer_tool_used",
    "worker.provider_id",
    "worker.provider_name",
    "worker.provider_url",
    "worker.provider_copyrights",
    "storage.s3_archive_prefix",
    "storage.s3_output_prefix",
    "storage.gcs_destination_prefix",
    "storage.upload_mode",
    "storage.s3_tar_prefix",
    "storage.s3_tar_manifest_csv",
    "storage.local_tar_dir",
    "storage.local_tar_manifest_csv",
    "storage.tar_compression",
    "storage.allow_production_prefixes",
    "data_placement.inputs.tool",
    "data_placement.inputs.required",
    "data_placement.inputs.source",
    "data_placement.inputs.destination",
    "data_placement.baselines.tool",
    "data_placement.baselines.required",
    "data_placement.baselines.source",
    "data_placement.baselines.destination",
    "data_placement.s3.tool",
    "data_placement.s3.required",
    "data_placement.s3.source",
    "data_placement.s3.destination",
    "data_placement.gcs.tool",
    "data_placement.gcs.required",
    "data_placement.gcs.source",
    "data_placement.gcs.destination",
    "data_placement.gcs_direct.tool",
    "data_placement.gcs_direct.required",
    "data_placement.gcs_direct.source",
    "data_placement.gcs_direct.destination",
    "data_placement.publication.tool",
    "data_placement.publication.required",
    "data_placement.publication.source",
    "data_placement.publication.destination",
    "analysis_metadata.enabled",
    "analysis_metadata.csv_path",
    "analysis_metadata.ipsae_threshold",
    "analysis_metadata.pdockq2_threshold",
    "analysis_metadata.finalize_after_gpu",
    "analysis_metadata.parquet_path",
    "analysis_metadata.selected_ids_path",
    "analysis_metadata.chunk_size",
    "analysis_metadata.finalize_partition",
    "analysis_metadata.finalize_cpus_per_task",
    "analysis_metadata.finalize_memory",
    "analysis_metadata.finalize_time",
    "analysis_metadata.high_quality_from_tars.enabled",
    "analysis_metadata.high_quality_from_tars.s3_prefix",
    "analysis_metadata.high_quality_from_tars.work_dir",
    "analysis_metadata.high_quality_from_tars.publication.enabled",
    "analysis_metadata.high_quality_from_tars.publication.target_prefix",
    "analysis_metadata.high_quality_from_tars.publication.default_target_prefix_from_recipe",
    "analysis_metadata.high_quality_from_tars.publication.collision_policy",
    "analysis_metadata.high_quality_from_tars.publication.overwrite",
    "analysis_metadata.high_quality_from_tars.publication.chunk_size",
    "analysis_metadata.high_quality_from_tars.publication.manifest_dir",
    "analysis_metadata.high_quality_from_tars.publication.evidence_dir",
    "analysis_metadata.high_quality_from_tars.publication.sample_download_count",
    "analysis_metadata.high_quality_from_tars.publication.s5cmd_numworkers",
    "validation.expected_archives",
    "validation.expected_allowed_ids",
    "validation.expected_one_archive_models",
    "validation.expected_one_archive_objects",
    "validation.expected_one_archive_aggregate_rows",
    "validation.expected_tar_count",
    "validation.expected_local_tars_rows",
    "validation.expected_failed_rows",
    "validation.expected_analysis_rows",
    "validation.expected_selected_ids",
    "validation.gcs_dry_run_only",
    "acceptance.baseline_output_dir",
    "acceptance.baseline_run_name",
    "acceptance.candidate_run_name",
    "acceptance.tar_payload_match_mode",
    "acceptance.payload_sample_count",
    "acceptance.candidate_parquet_required",
    "acceptance.compare_failed_sets",
    "acceptance.compare_tar_manifest_rows",
    "acceptance.compare_analysis_model_rows",
    "secrets.s3_credentials_ref.scheme",
    "secrets.s3_credentials_ref.target",
    "secrets.gcs_credentials_ref.scheme",
    "secrets.gcs_credentials_ref.target",
)

WORKFLOW_MUTATIONS = {
    "workflow.steps.00.name": "changed-step",
    "workflow.steps.00.run": False,
    "workflow.steps.00.mode": "submit-and-monitor",
    "workflow.steps.00.job_id": "job-456",
    "workflow.steps.00.array_range": "1-4%2",
    "workflow.steps.00.rendered_script": "/scripts/changed.sbatch",
}

RUN_PLAN_SCIENCE_PATHS = frozenset(
    {
        "dataset.mode",
        "dataset.array",
        "dataset.archive_source",
        "worker.stages",
        "worker.heterodimers",
        "worker.dssp_algorithm",
        "worker.retry_failed_only",
        "worker.retry_metadata_delta_tag",
        "worker.tool_used",
        "worker.homodimer_tool_used",
        "worker.provider_id",
        "worker.provider_name",
        "worker.provider_url",
        "worker.provider_copyrights",
        "analysis_metadata.enabled",
        "analysis_metadata.ipsae_threshold",
        "analysis_metadata.pdockq2_threshold",
        "analysis_metadata.finalize_after_gpu",
        "analysis_metadata.high_quality_from_tars.enabled",
    }
)

RUN_PLAN_ACTION_PATHS = frozenset(
    {
        *RUN_PLAN_SCIENCE_PATHS,
        *(
            f"references.artifacts.{name}.{field}"
            for name in (
                "extra_reference",
                "master_parquet",
                "tracking_parquet",
                "manifest_csv",
                "uniprot_duckdb",
                "heterodimer_id_manifest",
            )
            for field in ("sha256", "min_size_bytes", "version")
        ),
        "worker.workers",
        "worker.batch_size",
        "worker.shards_per_archive",
        "worker.self_upload",
        "worker.upload_slots",
        "worker.s5cmd_numworkers",
        "worker.duckdb_memory_limit",
        "worker.clash_device",
        "worker.clash_batch_size",
        "worker.parallel_stages",
        "storage.upload_mode",
        "storage.tar_compression",
        "storage.allow_production_prefixes",
        *(
            f"data_placement.{name}.{field}"
            for name in ("inputs", "baselines", "s3", "gcs", "gcs_direct", "publication")
            for field in ("tool", "required")
        ),
        "analysis_metadata.chunk_size",
        "analysis_metadata.high_quality_from_tars.publication.enabled",
        "analysis_metadata.high_quality_from_tars.publication.default_target_prefix_from_recipe",
        "analysis_metadata.high_quality_from_tars.publication.collision_policy",
        "analysis_metadata.high_quality_from_tars.publication.overwrite",
        "analysis_metadata.high_quality_from_tars.publication.chunk_size",
        "analysis_metadata.high_quality_from_tars.publication.sample_download_count",
        "validation.expected_archives",
        "validation.expected_allowed_ids",
        "validation.expected_one_archive_models",
        "validation.expected_one_archive_objects",
        "validation.expected_one_archive_aggregate_rows",
        "validation.expected_tar_count",
        "validation.expected_local_tars_rows",
        "validation.expected_failed_rows",
        "validation.expected_analysis_rows",
        "validation.expected_selected_ids",
        "validation.gcs_dry_run_only",
        "acceptance.baseline_run_name",
        "acceptance.tar_payload_match_mode",
        "acceptance.payload_sample_count",
        "acceptance.candidate_parquet_required",
        "acceptance.compare_failed_sets",
        "acceptance.compare_tar_manifest_rows",
        "acceptance.compare_analysis_model_rows",
    }
)

WORKFLOW_MUTATION_PATHS = tuple(
    f"workflow.steps.{index:02d}.{field}"
    for index in range(8)
    for field in ("name", "run", "mode", "job_id", "array_range", "rendered_script")
)
WORKFLOW_ACTION_FIELDS = frozenset({"name", "run", "mode", "array_range"})
RUN_PLAN_REJECTED_MUTATION_PATHS = frozenset(
    {
        "schema_version",
        "dataset.mode",
        "worker.self_upload",
        "data_placement.gcs_direct.tool",
        "analysis_metadata.high_quality_from_tars.publication.collision_policy",
    }
)
WORKFLOW_REJECTED_MUTATION_PATHS = frozenset(
    {
        "workflow.steps.00.name",
        "workflow.steps.00.run",
        "workflow.steps.00.mode",
        "workflow.steps.01.name",
        "workflow.steps.01.mode",
        "workflow.steps.02.name",
        "workflow.steps.02.mode",
        "workflow.steps.03.name",
        "workflow.steps.04.name",
        "workflow.steps.04.run",
        "workflow.steps.05.name",
        "workflow.steps.05.run",
        "workflow.steps.06.name",
        "workflow.steps.06.run",
        "workflow.steps.07.name",
        "workflow.steps.07.run",
        "workflow.steps.07.mode",
    }
)
LEGACY_REJECTED_MUTATION_PATHS = frozenset(
    {
        "paths.legacy_repo",
        "container.workdir",
        "container.mounts.0000.read_only",
        "container.mounts.0001.read_only",
        "container.mounts.0002.read_only",
        "container.mounts.0003.read_only",
        "container.mounts.0004.read_only",
        "phase_cluster.schema_version",
        "phase_cluster.extra_mounts.0000.schema_version",
        "qualified_runtime.schema_version",
        "qualified_runtime.selection_kind",
        "qualified_runtime.requeue_exit",
        "qualified_runtime.max_batch_requeue",
        "attempt_paths.schema_version",
        *(
            f"resources.{resource}.{field}"
            for resource in (
                "control_cpu",
                "gpu_worker",
                "analysis_finalize",
                "acceptance_tar_payload_parity",
                "acceptance_semantic",
            )
            for field in ("nodes", "tasks_per_node", "gpus_per_task", "max_parallel")
        ),
    }
)

LEGACY_EXECUTION_MUTATION_PATHS = (
    *(f"cluster.{field}" for field in ("name", "account", "owner")),
    *(
        f"paths.{field}"
        for field in (
            "project_root",
            "staging_dir",
            "output_dir",
            "log_dir",
            "legacy_repo",
            "afdb_toolkit_repo",
            "orchestration_repo",
            "recipe_dir",
        )
    ),
    "container.image",
    "container.workdir",
    *(f"container.mounts.{index:04d}.{field}" for index in range(5) for field in ("source", "target", "read_only")),
    *(
        f"resources.{resource}.{field}"
        for resource in (
            "control_cpu",
            "gpu_worker",
            "analysis_finalize",
            "acceptance_tar_payload_parity",
            "acceptance_semantic",
        )
        for field in (
            "partition",
            "cpus_per_task",
            "memory",
            "time",
            "gres",
            "array",
            "nodelist",
            "nodes",
            "tasks_per_node",
            "gpus_per_task",
            "max_parallel",
        )
    ),
    "submission.evidence_dir",
    "submission.report_path",
    "submission.controller_runtime",
    "source_path",
    "source_hash",
    *(
        f"phase_cluster.{field}"
        for field in (
            "schema_version",
            "profile_name",
            "owner",
            "transport",
            "ssh_target",
            "account",
            "project_root",
            "staging_root",
            "orchestration_repo",
            "runtime_image",
        )
    ),
    *(f"phase_cluster.extra_mounts.0000.{field}" for field in ("schema_version", "source", "target", "read_only")),
    *(
        f"qualified_runtime.{field}"
        for field in (
            "schema_version",
            "selection_kind",
            "tuple_id",
            "qualification_location",
            "qualification_sha256",
            "qualification_size_bytes",
            "qualified_at",
            "expires_at",
            "image_path",
            "image_sha256",
            "image_size_bytes",
            "image_policy",
            "source_kind",
            "source_revision",
            "source_package_path",
            "toolkit_package_path",
            "runtime_ipsae_binary_path",
            "runtime_ipsae_binary_sha256",
            "runtime_ipsae_binary_size_bytes",
            "source_identity_digest",
            "source_package_identity_digest",
            "toolkit_identity_digest",
            "bootstrap_sha256",
            "runtime_component_identity_digest",
            "requeue_exit",
            "max_batch_requeue",
        )
    ),
    *(
        f"attempt_paths.{field}"
        for field in (
            "schema_version",
            "legacy_run_id",
            "output_dir",
            "evidence_dir",
            "staging_dir",
            "object_prefix",
        )
    ),
)


def test_saturated_registry_walk_reaches_every_rule_and_builds_a_typed_normalized_copy() -> None:
    plan = _saturated_plan()
    spec = _saturated_runspec(plan)
    phase_cluster, qualified_runtime, attempt_paths = _phase_execution_fixture()
    walks = {
        "run-plan": run_plan_field_walk(plan),
        "workflow-step": workflow_field_walk(spec),
        "legacy-execution": legacy_execution_field_walk(
            spec,
            phase_cluster=phase_cluster,
            qualified_runtime=qualified_runtime,
            attempt_paths=attempt_paths,
        ),
    }
    for scope, fields in walks.items():
        reached = {field.rule.pattern for field in fields}
        declared = {rule.pattern for rule in POSTPROCESSING_FIELD_RULES if rule.scope == scope}
        assert reached == declared

    normalized = identity_normalized_runspec(spec)
    assert isinstance(normalized, RunSpec)
    assert normalized is not spec
    assert normalized.data_placement is not None
    assert spec.data_placement is not None
    assert normalized.data_placement.inputs.tool == spec.data_placement.inputs.tool
    assert normalized.worker.s5cmd_path != spec.worker.s5cmd_path


def test_attempt_owned_output_leaves_are_operational_or_placement_not_physical_inputs() -> None:
    roles = {
        rule.path: rule.roles for rule in POSTPROCESSING_FIELD_RULES if rule.scope == "run-plan" and rule.kind == "leaf"
    }
    for path in (
        "storage.s3_tar_manifest_csv",
        "storage.local_tar_dir",
        "storage.local_tar_manifest_csv",
        "analysis_metadata.csv_path",
        "analysis_metadata.parquet_path",
        "analysis_metadata.selected_ids_path",
        "analysis_metadata.high_quality_from_tars.work_dir",
        "analysis_metadata.high_quality_from_tars.publication.manifest_dir",
        "analysis_metadata.high_quality_from_tars.publication.evidence_dir",
    ):
        assert roles[path] == {"operational"}
    for path in (
        "storage.s3_output_prefix",
        "storage.gcs_destination_prefix",
        "storage.s3_tar_prefix",
        "analysis_metadata.high_quality_from_tars.publication.target_prefix",
    ):
        assert roles[path] == {"placement"}

    # These are not current V3 output consumers. Activating one requires adding
    # an explicit projection rule rather than silently treating it as an input.
    assert roles["data_placement.*.source"] == {"physical-locator"}
    assert roles["data_placement.*.destination"] == {"placement"}


def test_run_plan_mutation_matrix_is_exhaustive_and_obeys_every_role() -> None:
    plan = _saturated_plan()
    spec = _saturated_runspec(plan)
    cluster, qualified, attempt_paths = _phase_execution_fixture()
    leaves = {field.dotted_path: field for field in run_plan_field_walk(plan) if field.rule.kind == "leaf"}
    assert set(RUN_PLAN_MUTATION_PATHS) == set(leaves)
    assert set(RUN_PLAN_MUTATION_PATHS) >= RUN_PLAN_SCIENCE_PATHS
    assert set(RUN_PLAN_MUTATION_PATHS) >= RUN_PLAN_ACTION_PATHS
    baseline_science, baseline_action, baseline_execution = _actual_identity_digests(
        plan, spec, cluster, qualified, attempt_paths
    )
    rejected: set[str] = set()

    for path in RUN_PLAN_MUTATION_PATHS:
        changed = _validated_run_plan_mutation(plan, path)
        if changed is None:
            rejected.add(path)
            continue
        try:
            changed_spec = _saturated_runspec(changed)
            changed_science, changed_action, changed_execution = _actual_identity_digests(
                changed, changed_spec, cluster, qualified, attempt_paths
            )
        except ValueError:
            rejected.add(path)
            continue
        assert (changed_science != baseline_science) is (path in RUN_PLAN_SCIENCE_PATHS), path
        assert (changed_action != baseline_action) is (path in RUN_PLAN_ACTION_PATHS), path
        assert changed_execution != baseline_execution, path
    assert rejected == RUN_PLAN_REJECTED_MUTATION_PATHS


def test_saturated_workflow_mutation_matrix_uses_an_independent_oracle() -> None:
    plan = _saturated_plan()
    spec = _saturated_runspec(plan)
    cluster, qualified, attempt_paths = _phase_execution_fixture()
    leaves = {field.dotted_path: field for field in workflow_field_walk(spec) if field.rule.kind == "leaf"}
    assert set(WORKFLOW_MUTATION_PATHS) == set(leaves)
    _, baseline_action, baseline_execution = _actual_identity_digests(plan, spec, cluster, qualified, attempt_paths)
    rejected: set[str] = set()
    for path in WORKFLOW_MUTATION_PATHS:
        field_name = path.rsplit(".", 1)[1]
        changed = _validated_workflow_mutation(spec, path)
        if changed is None:
            rejected.add(path)
            continue
        expected_change = field_name in WORKFLOW_ACTION_FIELDS
        try:
            _, changed_action, changed_execution = _actual_identity_digests(
                plan, changed, cluster, qualified, attempt_paths
            )
        except ValueError:
            rejected.add(path)
            continue
        assert (changed_action != baseline_action) is expected_change, path
        assert changed_execution != baseline_execution, path
    assert rejected == WORKFLOW_REJECTED_MUTATION_PATHS


def test_workflow_array_throttle_is_not_semantic() -> None:
    spec = _single_step_runspec()
    step = spec.workflow.steps[0] if spec.workflow is not None else None
    assert step is not None
    baseline = _workflow_projection_digest(spec)
    throttle_only = step.model_copy(update={"array_range": "1-3%1"})
    throttle_spec = spec.model_copy(update={"workflow": spec.workflow.model_copy(update={"steps": (throttle_only,)})})
    assert _workflow_projection_digest(throttle_spec) == baseline


def test_legacy_execution_walk_accepts_default_read_write_phase_mounts() -> None:
    """Regression: omit-when-false read_only must not break the registry walk.

    PhaseMountSnapshot.to_mapping() omits read_only when false (legacy
    byte-stability), while the registry declares phase_cluster.extra_mounts.*.read_only
    a required fixed child. The walk materializes the default so default (read-write)
    mounts — every mount the package itself builds — classify instead of raising.
    """
    import dataclasses

    spec = _single_step_runspec()
    cluster, qualified, attempt_paths = _phase_execution_fixture()
    default_cluster = dataclasses.replace(
        cluster,
        extra_mounts=(PhaseMountSnapshot(source="/cluster/source", target="/container/target"),),
    )
    fields = legacy_execution_field_walk(
        spec,
        phase_cluster=default_cluster,
        qualified_runtime=qualified,
        attempt_paths=attempt_paths,
    )
    leaf = next(field for field in fields if field.dotted_path == "phase_cluster.extra_mounts.0000.read_only")
    assert leaf.value is False
    assert "operational" in leaf.rule.roles


def test_legacy_execution_mutation_matrix_is_exhaustive_and_changes_execution_digest() -> None:
    plan = _saturated_plan()
    spec = _saturated_runspec(plan)
    cluster, qualified, attempt_paths = _phase_execution_fixture()
    leaves = {
        field.dotted_path: field
        for field in legacy_execution_field_walk(
            spec,
            phase_cluster=cluster,
            qualified_runtime=qualified,
            attempt_paths=attempt_paths,
        )
        if field.rule.kind == "leaf"
    }
    assert set(LEGACY_EXECUTION_MUTATION_PATHS) == set(leaves)
    baseline_science, baseline_action, baseline_execution = _actual_identity_digests(
        plan, spec, cluster, qualified, attempt_paths
    )
    rejected: set[str] = set()

    for path in LEGACY_EXECUTION_MUTATION_PATHS:
        changed = _validated_legacy_mutation(spec, cluster, qualified, attempt_paths, path)
        if changed is None:
            rejected.add(path)
            continue
        changed_spec, changed_cluster, changed_qualified, changed_attempt_paths = changed
        if path == "phase_cluster.runtime_image":
            changed_qualified = qualified_postprocessing_runtime_from_mapping(
                {**changed_qualified.to_mapping(), "image_path": changed_cluster.runtime_image}
            )
        elif path == "qualified_runtime.image_path":
            changed_cluster = postprocessing_cluster_snapshot_from_mapping(
                {**changed_cluster.to_mapping(), "runtime_image": changed_qualified.image_path}
            )
        try:
            changed_science, changed_action, changed_execution = _actual_identity_digests(
                plan,
                changed_spec,
                changed_cluster,
                changed_qualified,
                changed_attempt_paths,
            )
        except ValueError:
            rejected.add(path)
            continue
        assert changed_science == baseline_science, path
        assert changed_action == baseline_action, path
        expected_execution_change = path not in {"source_path", "source_hash"}
        assert (changed_execution != baseline_execution) is expected_execution_change, path
    assert rejected == LEGACY_REJECTED_MUTATION_PATHS


def test_typed_command_normalization_does_not_replace_equal_semantic_text() -> None:
    spec = _saturated_runspec(_saturated_plan())
    assert spec.data_placement is not None
    assert spec.worker.s5cmd_path == spec.data_placement.s3.tool == "s5cmd"
    upload = WorkflowStepSpec(name="upload-s3", run=True)
    upload_spec = spec.model_copy(update={"workflow": WorkflowSpec(steps=(upload,))})

    normalized = identity_normalized_runspec(upload_spec)
    normalized_step = normalized.workflow.steps[0] if normalized.workflow is not None else None
    assert normalized_step is not None
    command = resolve_workflow_step_command(normalized, normalized_step)
    assert "--tool s5cmd" in command
    assert normalized.worker.s5cmd_path != "s5cmd"

    changed_s3 = spec.data_placement.s3.model_copy(update={"tool": "rclone"})
    changed_placement = spec.data_placement.model_copy(update={"s3": changed_s3})
    changed = upload_spec.model_copy(update={"data_placement": changed_placement})
    changed_normalized = identity_normalized_runspec(changed)
    changed_step = changed_normalized.workflow.steps[0] if changed_normalized.workflow is not None else None
    assert changed_step is not None
    assert resolve_workflow_step_command(changed_normalized, changed_step) != command


def test_optional_generated_candidate_name_presence_does_not_change_complete_action_identity() -> None:
    present_plan = _saturated_plan()
    present_spec = _saturated_runspec(present_plan)
    cluster, qualified, attempt_paths = _phase_execution_fixture()
    present = _actual_identity_digests(present_plan, present_spec, cluster, qualified, attempt_paths)

    payload = _run_plan_validation_mapping(present_plan)
    acceptance = payload.get("acceptance")
    assert isinstance(acceptance, dict)
    acceptance["candidate_run_name"] = None
    absent_plan = RunPlan.model_validate(payload)
    absent_spec = _saturated_runspec(absent_plan)
    absent = _actual_identity_digests(absent_plan, absent_spec, cluster, qualified, attempt_paths)

    assert absent[0] == present[0]
    assert absent[1] == present[1]
    assert absent[2] != present[2]


def test_sorted_reference_projection_and_inactive_concrete_keys_are_stable() -> None:
    plan = _saturated_plan()
    baseline = project_run_plan_fields(plan, roles=POSITIVE_ACTION_ROLES)
    reversed_artifacts = dict(reversed(tuple(plan.references.artifacts.items())))
    reordered_references = plan.references.model_copy(update={"artifacts": reversed_artifacts})
    reordered = plan.model_copy(update={"references": reordered_references})
    assert project_run_plan_fields(reordered, roles=POSITIVE_ACTION_ROLES) == baseline

    inactive = plan.model_copy(update={"data_placement": None})
    paths = {field.path for field in project_run_plan_fields(inactive, roles=POSITIVE_ACTION_ROLES)}
    assert not any(path.startswith("data_placement.") for path in paths)


def test_registry_rejects_unknown_missing_and_forbidden_role_combinations() -> None:
    mapping = _saturated_plan().model_dump(mode="json")
    mapping["unknown"] = "unclassified"
    with pytest.raises(ValueError, match="match exactly one rule"):
        walk_postprocessing_fields(mapping, scope="run-plan")
    del mapping["unknown"]
    dataset = mapping["dataset"]
    assert isinstance(dataset, dict)
    del dataset["mode"]
    with pytest.raises(ValueError, match="missing declared children"):
        walk_postprocessing_fields(mapping, scope="run-plan")

    PostprocessingFieldRule(
        scope="run-plan",
        pattern=("worker", "workers"),
        kind="leaf",
        roles=frozenset({"operational", "action-semantic"}),
    )
    PostprocessingFieldRule(
        scope="run-plan",
        pattern=("data_placement", "*", "tool"),
        kind="leaf",
        roles=frozenset({"placement", "action-semantic"}),
    )
    for roles in (
        frozenset({"physical-locator", "action-semantic"}),
        frozenset({"secret", "action-semantic"}),
        frozenset({"scientific", "operational"}),
        frozenset({"encoding", "placement"}),
        frozenset({"acceptance", "resource"}),
        frozenset({"logical-input"}),
    ):
        with pytest.raises(ValueError):
            PostprocessingFieldRule(
                scope="run-plan",
                pattern=("invalid",),
                kind="leaf",
                roles=roles,
            )


def _saturated_plan() -> RunPlan:
    payload = yaml.safe_load((EXAMPLE / "run-plan.yaml").read_bytes())
    assert isinstance(payload, dict)
    references = payload["references"]
    assert isinstance(references, dict)
    names = (
        "master_parquet",
        "tracking_parquet",
        "manifest_csv",
        "uniprot_duckdb",
        "heterodimer_id_manifest",
    )
    for index, name in enumerate(names, start=1):
        raw = references[name]
        path = raw["path"] if isinstance(raw, dict) else raw
        references[name] = {
            "path": path,
            "source_uri": f"s3://references/{name}",
            "sha256": str(index) * 64,
            "min_size_bytes": index,
            "version": f"v{index}",
        }
    references["artifacts"] = {
        "extra_reference": {
            "path": "/references/extra",
            "source_uri": "s3://references/extra",
            "sha256": "a" * 64,
            "min_size_bytes": 99,
            "version": "extra-v1",
        }
    }
    storage = payload["storage"]
    assert isinstance(storage, dict)
    storage.update(
        {
            "gcs_destination_prefix": "gs://candidate",
            "s3_tar_prefix": "s3://candidate/tars",
            "s3_tar_manifest_csv": "/output/s3.csv",
        }
    )
    payload["data_placement"] = {
        name: {
            "tool": tool,
            "required": True,
            "source": f"source-{name}",
            "destination": f"destination-{name}",
        }
        for name, tool in (
            ("inputs", "dm"),
            ("baselines", "dm"),
            ("s3", "s5cmd"),
            ("gcs", "gcloud"),
            ("gcs_direct", "gcloud"),
            ("publication", "s5cmd"),
        )
    }
    analysis = payload["analysis_metadata"]
    assert isinstance(analysis, dict)
    analysis["high_quality_from_tars"] = {
        "enabled": True,
        "s3_prefix": "s3://hq",
        "work_dir": "/hq",
        "publication": {
            "enabled": True,
            "target_prefix": "s3://hq-out",
            "default_target_prefix_from_recipe": False,
            "collision_policy": "fail_if_non_empty",
            "overwrite": False,
            "chunk_size": 777,
            "manifest_dir": "/manifest",
            "evidence_dir": "/evidence",
            "sample_download_count": 3,
            "s5cmd_numworkers": 4,
        },
    }
    validation = payload["validation"]
    assert isinstance(validation, dict)
    validation.update(
        {
            "expected_allowed_ids": 7,
            "expected_one_archive_objects": 7,
            "expected_one_archive_aggregate_rows": 7,
        }
    )
    acceptance = payload["acceptance"]
    assert isinstance(acceptance, dict)
    acceptance.update(
        {
            "candidate_parquet_required": True,
            "compare_failed_sets": True,
            "compare_tar_manifest_rows": True,
            "compare_analysis_model_rows": True,
        }
    )
    secrets = payload["secrets"]
    assert isinstance(secrets, dict)
    secrets["gcs_credentials_ref"] = "env:GOOGLE_APPLICATION_CREDENTIALS"
    return RunPlan.model_validate(payload)


def _saturated_runspec(plan: RunPlan) -> RunSpec:
    profile = resolve_cluster_profile("example-cluster", config_path=EXAMPLE / "profiles.yaml")
    workflow = load_workflow_template(EXAMPLE / "run-plan.yaml", "workflow.yaml")
    steps = workflow["steps"]
    assert isinstance(steps, list)
    for index, raw in enumerate(steps):
        assert isinstance(raw, dict)
        raw["rendered_script"] = f"/scripts/{index:02d}-{raw['name']}.sbatch"
        if raw["name"] == "slurm":
            raw["mode"] = "monitor-existing"
            raw["job_id"] = "12345"
            raw["array_range"] = "853-853%1"
    mapping = materialize_runspec_mapping(plan, profile, workflow)
    return runspec_from_mapping(
        mapping,
        source_path=EXAMPLE / "saturated-runspec.yaml",
        source_hash="f" * 64,
    )


def _single_step_runspec() -> RunSpec:
    spec = _saturated_runspec(_saturated_plan())
    step = WorkflowStepSpec(
        name="slurm",
        run=True,
        mode="monitor-existing",
        job_id="job-123",
        array_range="1-3%2",
        rendered_script=Path("/scripts/slurm.sbatch"),
    )
    return spec.model_copy(update={"workflow": WorkflowSpec(steps=(step,))})


def _phase_execution_fixture() -> tuple[
    PostprocessingClusterSnapshot,
    QualifiedPostprocessingRuntimeSelection,
    PostprocessingAttemptPaths,
]:
    cluster = PostprocessingClusterSnapshot(
        profile_name="example-cluster",
        owner="operator",
        transport="ssh",
        ssh_target="cluster.example",
        account="project",
        project_root="/cluster/project",
        staging_root="/cluster/staging",
        orchestration_repo="/cluster/orchestration",
        runtime_image="/cluster/runtime.sqsh",
        extra_mounts=(PhaseMountSnapshot(source="/cluster/source", target="/container/target", read_only=True),),
    )
    qualified = QualifiedPostprocessingRuntimeSelection(
        tuple_id="1" * 64,
        qualification_location="attempts/attempt-0001/runtime-qualification.json",
        qualification_sha256="2" * 64,
        qualification_size_bytes=100,
        qualified_at="2026-09-03T00:00:00Z",
        expires_at="2026-09-04T00:00:00Z",
        image_path="/cluster/runtime.sqsh",
        image_sha256="3" * 64,
        image_size_bytes=200,
        image_policy="digest-checked",
        source_kind="override",
        source_revision="4" * 40,
        source_package_path="/cluster/orchestration.tar",
        toolkit_package_path="/cluster/toolkit.tar",
        runtime_ipsae_binary_path="/cluster/ipsae_cpp",
        runtime_ipsae_binary_sha256="5" * 64,
        runtime_ipsae_binary_size_bytes=300,
        source_identity_digest="6" * 64,
        source_package_identity_digest="7" * 64,
        toolkit_identity_digest="8" * 64,
        bootstrap_sha256="9" * 64,
        runtime_component_identity_digest="a" * 64,
    )
    paths = PostprocessingAttemptPaths(
        legacy_run_id="task853-attempt-0001",
        output_dir="/cluster/output",
        evidence_dir="/cluster/output/evidence",
        staging_dir="/cluster/staging/task853",
        object_prefix="s3://candidate/task853/",
    )
    return cluster, qualified, paths


def _projection_digest(plan: RunPlan, roles: frozenset[str]) -> str:
    fields = project_run_plan_fields(plan, roles=roles)  # type: ignore[arg-type]
    return canonical_mapping_digest({"fields": [field.to_mapping() for field in fields]})


def _workflow_projection_digest(spec: RunSpec) -> str:
    fields = project_workflow_fields(spec)
    return canonical_mapping_digest({"fields": [field.to_mapping() for field in fields]})


def _actual_identity_digests(
    plan: RunPlan,
    spec: RunSpec,
    cluster: PostprocessingClusterSnapshot,
    qualified: QualifiedPostprocessingRuntimeSelection,
    attempt_paths: PostprocessingAttemptPaths,
) -> tuple[str, str, str]:
    logical_inputs = PostprocessingLogicalInputIdentityManifestV2(
        entries=(
            PostprocessingLogicalInputIdentityV2(
                name="fixture-input",
                member_identity="fixture-member-v1",
                expected_content_sha256="1" * 64,
                expected_size_bytes=1,
            ),
        )
    )
    policy = _acceptance_policy()
    science = build_scientific_identity(plan, logical_inputs=logical_inputs)
    actions = postprocessing_runtime_actions(spec)
    semantics = build_action_semantics(
        actions,
        legacy_plan=plan,
        legacy_runspec=spec,
        scientific_identity=science,
        logical_inputs=logical_inputs,
        acceptance_policy=policy,
    )
    phase_plan_digest = canonical_mapping_digest({"legacy_run_plan": plan.model_dump(mode="json")})
    legacy_mapping = spec.model_dump(
        mode="json",
        exclude_none=True,
        exclude={"source_path", "source_hash"},
    )
    legacy_bytes = render_runspec_yaml(legacy_mapping).encode()
    policy_bytes = (json.dumps(policy.to_mapping(), indent=2, sort_keys=True) + "\n").encode()
    policy_reference = PostprocessingAcceptanceSnapshotReference(
        location="attempts/attempt-0001/acceptance-policy.json",
        sha256=hashlib.sha256(policy_bytes).hexdigest(),
        size_bytes=len(policy_bytes),
        semantic_digest=policy.semantic_digest,
        policy_id=policy.policy_id,
    )
    phase_identity = PostprocessingPhaseExecutionIdentity(
        phase_plan_digest=phase_plan_digest,
        logical_input_manifest_digest=logical_inputs.digest,
        scientific_identity_digest=science.digest,
        action_semantics_digest=semantics.digest,
        acceptance_semantic_digest=policy.semantic_digest,
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        qualified_runtime_digest=qualified.digest,
        output_namespace="identity-oracle",
        substitutions=attempt_paths,
    )
    projection = PostprocessingExecutionProjection(
        document_location="attempts/attempt-0001/legacy-runspec.yaml",
        document_sha256=hashlib.sha256(legacy_bytes).hexdigest(),
        document_size_bytes=len(legacy_bytes),
        legacy_schema_version=spec.schema_version,
        phase_identity=phase_identity,
        phase_identity_digest=phase_identity.digest,
    )
    final_runspec = PostprocessingPhaseRunSpec(
        phase_run_id=phase_identity.phase_run_id,
        attempt_id=phase_identity.attempt_id,
        phase_plan_digest=phase_plan_digest,
        materialized_at="2026-09-03T00:00:00Z",
        cluster=cluster,
        payload=PostprocessingPhaseRunSpecPayload(
            actions=actions,
            action_graph_digest=postprocessing_action_graph_digest(actions),
            action_semantics_digest=semantics.digest,
            action_semantics=semantics,
            scientific_identity=science,
            logical_inputs=logical_inputs,
            physical_inputs=(
                PhysicalInputLocator(
                    name="fixture-input",
                    locator="/physical/fixture-input",
                    purpose="identity-oracle",
                ),
            ),
            execution_projection=projection,
            acceptance_policy=policy_reference,
            qualified_runtime=qualified,
            attempt_paths=attempt_paths,
        ),
    )
    return science.digest, semantics.digest, final_runspec.digest


def _acceptance_policy() -> PostprocessingAcceptancePolicySnapshot:
    payload = yaml.safe_load((EXAMPLE / "acceptance-policy.yaml").read_bytes())
    if not isinstance(payload, Mapping):
        raise TypeError("acceptance policy fixture must be a mapping")
    return postprocessing_acceptance_policy_from_mapping(payload)


def _legacy_execution_digest(
    spec: RunSpec,
    cluster: PostprocessingClusterSnapshot,
    qualified: QualifiedPostprocessingRuntimeSelection,
    attempt_paths: PostprocessingAttemptPaths,
) -> str:
    fields = legacy_execution_field_walk(
        spec,
        phase_cluster=cluster,
        qualified_runtime=qualified,
        attempt_paths=attempt_paths,
    )
    return canonical_mapping_digest(
        {"fields": [{"path": field.dotted_path, "value": field.value} for field in fields if field.rule.kind == "leaf"]}
    )


def _workflow_replacement(field_name: str, value: object) -> object:
    if field_name == "name":
        return f"{value}-changed"
    if field_name == "run":
        return not value
    if field_name == "mode":
        return "submit-and-monitor" if value != "submit-and-monitor" else "monitor-existing"
    if field_name == "job_id":
        return "job-999"
    if field_name == "array_range":
        return "1-4%2"
    if field_name == "rendered_script":
        return Path("/scripts/changed.sbatch")
    raise AssertionError(f"unsupported workflow mutation field: {field_name}")


def _validated_run_plan_mutation(plan: RunPlan, path: str) -> RunPlan | None:
    validation_mapping = _run_plan_validation_mapping(plan)
    validation_path = _run_plan_validation_path(path)
    original = _mapping_path_value(plan.model_dump(mode="json"), path)
    for candidate in _mutation_candidates(path, original):
        payload = copy.deepcopy(validation_mapping)
        _set_mapping_path(payload, validation_path, candidate)
        try:
            return RunPlan.model_validate(payload)
        except ValueError:
            continue
    return None


def _validated_workflow_mutation(spec: RunSpec, path: str) -> RunSpec | None:
    if spec.workflow is None:
        raise AssertionError("workflow mutation requires a workflow")
    relative = path.removeprefix("workflow.")
    payload = spec.workflow.model_dump(mode="json")
    original = _mapping_path_value(payload, relative)
    for candidate in _mutation_candidates(path, original):
        changed = copy.deepcopy(payload)
        _set_mapping_path(changed, relative, candidate)
        if path.endswith(".mode") and candidate == "monitor-existing":
            step_index = int(path.split(".")[2])
            changed_steps = changed.get("steps")
            if isinstance(changed_steps, list) and isinstance(changed_steps[step_index], dict):
                changed_steps[step_index]["job_id"] = "identity-mutation-job"
        try:
            workflow = WorkflowSpec.model_validate(changed)
        except ValueError:
            continue
        return spec.model_copy(update={"workflow": workflow})
    return None


def _validated_legacy_mutation(
    spec: RunSpec,
    cluster: PostprocessingClusterSnapshot,
    qualified: QualifiedPostprocessingRuntimeSelection,
    attempt_paths: PostprocessingAttemptPaths,
    path: str,
) -> (
    tuple[RunSpec, PostprocessingClusterSnapshot, QualifiedPostprocessingRuntimeSelection, PostprocessingAttemptPaths]
    | None
):
    root, *tail = path.split(".")
    if root == "phase_cluster":
        payload = cluster.to_mapping()
        parser = postprocessing_cluster_snapshot_from_mapping
        relative = ".".join(tail)
        original = _mapping_path_value(payload, relative)
    elif root == "qualified_runtime":
        payload = qualified.to_mapping()
        parser = qualified_postprocessing_runtime_from_mapping
        relative = ".".join(tail)
        original = payload.get(relative)
    elif root == "attempt_paths":
        payload = attempt_paths.to_mapping()
        parser = postprocessing_attempt_paths_from_mapping
        relative = ".".join(tail)
        original = _mapping_path_value(payload, relative)
    else:
        if (
            path == "paths.legacy_repo"
            or path == "container.workdir"
            or (path.startswith("container.mounts.") and path.endswith(".read_only"))
        ):
            return None
        payload = _runspec_validation_mapping(spec)
        parser = RunSpec.model_validate
        relative = path
        original = _legacy_original_value(spec, relative)
    for candidate in _mutation_candidates(path, original):
        changed = copy.deepcopy(payload)
        _set_mapping_path(changed, relative, candidate)
        if path == "paths.output_dir" and isinstance(candidate, str):
            submission = changed.get("submission")
            if isinstance(submission, dict):
                submission["evidence_dir"] = f"{candidate}/evidence"
                submission["report_path"] = f"{candidate}/RUN_REPORT.md"
        if root == "phase_cluster" and relative == "transport" and candidate == "local-slurm":
            changed["ssh_target"] = None
        if root == "qualified_runtime" and relative == "source_kind" and candidate == "baked":
            changed["toolkit_package_path"] = None
        try:
            parsed = parser(changed)
        except ValueError:
            continue
        if root == "phase_cluster":
            assert isinstance(parsed, PostprocessingClusterSnapshot)
            return spec, parsed, qualified, attempt_paths
        if root == "qualified_runtime":
            assert isinstance(parsed, QualifiedPostprocessingRuntimeSelection)
            return spec, cluster, parsed, attempt_paths
        if root == "attempt_paths":
            assert isinstance(parsed, PostprocessingAttemptPaths)
            return spec, cluster, qualified, parsed
        assert isinstance(parsed, RunSpec)
        return parsed, cluster, qualified, attempt_paths
    return None


def _mapping_path_value(payload: object, path: str) -> object:
    current = payload
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise AssertionError(f"cannot read mapping mutation path {path}")
    return current


def _legacy_original_value(spec: RunSpec, path: str) -> object:
    """Read a legacy-execution mutation path's pre-mutation value.

    The typed packed-topology resource fields (``nodes``, ``tasks_per_node``,
    ``gpus_per_task``, ``max_parallel``) are omitted from ``model_dump`` when they
    hold their scalar defaults (byte-stability), so a mapping read would
    KeyError. Read them from the typed model instead, which returns the actual
    default (``None`` or ``1``) so the mutation matrix still produces a real
    value change for every walkable leaf.
    """
    parts = path.split(".")
    if (
        len(parts) == 3
        and parts[0] == "resources"
        and parts[2]
        in {
            "nodes",
            "tasks_per_node",
            "gpus_per_task",
            "max_parallel",
        }
    ):
        return getattr(spec.resources[parts[1]], parts[2])
    return _mapping_path_value(spec.model_dump(mode="json"), path)


def _run_plan_validation_mapping(plan: RunPlan) -> dict[str, object]:
    payload = plan.model_dump(mode="json")
    _restore_reference_artifact_inputs(payload)
    return payload


def _run_plan_validation_path(path: str) -> str:
    parts = path.split(".")
    reference_names = {
        "master_parquet",
        "tracking_parquet",
        "manifest_csv",
        "uniprot_duckdb",
        "heterodimer_id_manifest",
    }
    if len(parts) == 2 and parts[0] == "references" and parts[1] in reference_names:
        return f"{path}.path"
    if len(parts) >= 4 and parts[:2] == ["references", "artifacts"] and parts[2] in reference_names:
        return ".".join(("references", *parts[2:]))
    return path


def _runspec_validation_mapping(spec: RunSpec) -> dict[str, object]:
    payload = spec.model_dump(mode="json")
    _restore_reference_artifact_inputs(payload)
    paths = payload.get("paths")
    if isinstance(paths, dict):
        paths.pop("legacy_repo", None)
    container = payload.get("container")
    if isinstance(container, dict):
        container.pop("workdir", None)
        mounts = container.get("mounts")
        if isinstance(mounts, list):
            for mount in mounts:
                if isinstance(mount, dict):
                    mount.pop("read_only", None)
    return payload


def _restore_reference_artifact_inputs(payload: dict[str, object]) -> None:
    references = payload.get("references")
    if not isinstance(references, dict):
        raise AssertionError("identity fixture references must be a mapping")
    artifacts = references.get("artifacts")
    if not isinstance(artifacts, dict):
        raise AssertionError("identity fixture reference artifacts must be a mapping")
    for name in (
        "master_parquet",
        "tracking_parquet",
        "manifest_csv",
        "uniprot_duckdb",
        "heterodimer_id_manifest",
    ):
        if name in artifacts:
            references[name] = artifacts.pop(name)


def _set_mapping_path(payload: object, path: str, value: object) -> None:
    parts = path.split(".")
    current = payload
    for part in parts[:-1]:
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise AssertionError(f"cannot descend through mapping mutation path {path}")
    final = parts[-1]
    if isinstance(current, dict):
        current[final] = value
    elif isinstance(current, list):
        current[int(final)] = value
    else:
        raise AssertionError(f"cannot assign mapping mutation path {path}")


def _mutation_candidates(path: str, value: object) -> tuple[object, ...]:
    constrained: dict[str, tuple[object, ...]] = {
        "run_kind": ("dev", "canary", "production"),
        "dataset.archive_source": ("tracking", "staging_dir"),
        "storage.upload_mode": ("files", "tar"),
        "storage.tar_compression": ("none", "gz", "zstd-members"),
        "acceptance.tar_payload_match_mode": ("aggregate", "by-tar"),
        "submission.controller_runtime": ("enroot", "srun-pyxis"),
        "phase_cluster.transport": ("ssh", "local-slurm"),
        "qualified_runtime.image_policy": ("digest-checked", "trusted-cache"),
        "qualified_runtime.source_kind": ("baked", "override"),
        "worker.tool_used": ("OpenFold / AlphaFold-Multimer",),
        **{f"workflow.steps.{index:02d}.array_range": ("854-854%1",) for index in range(8)},
        "workflow.steps.00.mode": (None, "submit-and-monitor", "monitor-existing"),
        "workflow.steps.01.mode": (None, "submit-and-monitor", "monitor-existing"),
        "workflow.steps.02.mode": (None, "submit-and-monitor", "monitor-existing"),
        "workflow.steps.03.mode": (None, "submit-and-monitor", "monitor-existing"),
        "workflow.steps.04.mode": (None, "submit-and-monitor", "monitor-existing"),
        "workflow.steps.05.mode": (None, "submit-and-monitor", "monitor-existing"),
        "workflow.steps.06.mode": (None, "submit-and-monitor", "monitor-existing"),
        "workflow.steps.07.mode": (None, "submit-and-monitor", "monitor-existing"),
    }
    candidates: list[object] = list(constrained.get(path, ()))
    if isinstance(value, bool):
        candidates.append(not value)
    elif isinstance(value, int):
        candidates.append(value + 1)
    elif isinstance(value, float):
        candidates.append(value + 0.125)
    elif isinstance(value, list):
        candidates.append([*value, "identity-mutation"])
    elif isinstance(value, str):
        if len(value) in {40, 64} and all(character in "0123456789abcdef" for character in value):
            candidates.append(("0" if value[0] != "0" else "1") + value[1:])
        elif value.startswith("/"):
            candidates.append(f"{value}-identity-mutation")
        elif "://" in value:
            candidates.append(f"{value.rstrip('/')}/identity-mutation")
        else:
            candidates.extend(
                (
                    f"{value.removesuffix('.json')}-identity-mutation.json" if value.endswith(".json") else value,
                    f"{value}-identity-mutation",
                    "identity-mutation",
                    "2026-09-05T00:00:00Z",
                    "854-854%1",
                    "00:10:00",
                    "1G",
                    "staging_dir",
                    "aggregate",
                    "files",
                    "gz",
                    "dev",
                    "srun-pyxis",
                    "local-slurm",
                    "trusted-cache",
                    "baked",
                    "dm",
                    "s5cmd",
                    "gcloud",
                    "env",
                    "aws",
                )
            )
    elif value is None:
        candidates.extend(
            (
                "identity-mutation",
                "/identity-mutation",
                "99999",
                "854-854%1",
                "00:10:00",
                "1G",
                1,
                False,
            )
        )
    return tuple(candidate for candidate in candidates if candidate != value)


def _replace_model_path(value: object, path: tuple[str, ...], replacement: object) -> object:
    if not path:
        return replacement
    head, *tail = path
    if isinstance(value, BaseModel):
        nested = getattr(value, head)
        return value.model_copy(update={head: _replace_model_path(nested, tuple(tail), replacement)})
    if isinstance(value, Mapping):
        nested = dict(value)
        nested[head] = _replace_model_path(nested[head], tuple(tail), replacement)
        return nested
    if isinstance(value, tuple):
        index = int(head)
        nested_tuple = list(value)
        nested_tuple[index] = _replace_model_path(nested_tuple[index], tuple(tail), replacement)
        return tuple(nested_tuple)
    if isinstance(value, list):
        index = int(head)
        nested_list = list(value)
        nested_list[index] = _replace_model_path(nested_list[index], tuple(tail), replacement)
        return nested_list
    if is_dataclass(value) and not isinstance(value, type):
        nested_record = copy.copy(value)
        object.__setattr__(
            nested_record,
            head,
            _replace_model_path(getattr(value, head), tuple(tail), replacement),
        )
        return nested_record
    raise AssertionError(f"cannot descend through {type(value).__name__} at {'.'.join(path)}")


def _different(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if isinstance(value, Path):
        return Path(f"{value}.mutated")
    if isinstance(value, str):
        if len(value) in {40, 64} and all(character in "0123456789abcdef" for character in value):
            replacement = "0" if value[0] != "0" else "1"
            return f"{replacement}{value[1:]}"
        return f"{value}::mutated"
    if isinstance(value, tuple | list):
        return (*value, "mutated")
    if value is None:
        return "mutated"
    raise AssertionError(f"no mutation for {type(value).__name__}")
