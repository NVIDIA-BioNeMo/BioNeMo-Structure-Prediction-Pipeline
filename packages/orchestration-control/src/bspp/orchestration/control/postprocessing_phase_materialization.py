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

"""Scheduler-free materialization of immutable postprocessing Attempts."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

import yaml

from bspp.orchestration.contract.phase import PhaseMountSnapshot, PhaseSlurmResources
from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    PostprocessingAcceptancePolicySnapshot,
    postprocessing_acceptance_policy_from_mapping,
)
from bspp.orchestration.contract.postprocessing_acceptance_reference import (
    PostprocessingAcceptanceSnapshotReference,
)
from bspp.orchestration.contract.postprocessing_action_contract import (
    PostprocessingRuntimeAction,
    normalize_slurm_array,
    postprocessing_action_graph_digest,
)
from bspp.orchestration.contract.postprocessing_autorequeue_policy import (
    PostprocessingAutorequeuePolicy,
)
from bspp.orchestration.contract.postprocessing_event import (
    PostprocessingPhaseEvent,
)
from bspp.orchestration.contract.postprocessing_execution import (
    PostprocessingAttemptPaths,
    PostprocessingClusterSnapshot,
    PostprocessingExecutionProjection,
    PostprocessingPhaseExecutionIdentity,
    QualifiedPostprocessingRuntimeSelection,
)
from bspp.orchestration.contract.postprocessing_identity_v3 import (
    PostprocessingActionSemanticsV3,
    PostprocessingScientificIdentityV3,
)
from bspp.orchestration.contract.postprocessing_logical_identity import (
    PhysicalInputLocator,
    PostprocessingLogicalInputIdentityManifestV2,
)
from bspp.orchestration.contract.postprocessing_phase_ids import (
    postprocessing_action_id,
)
from bspp.orchestration.contract.postprocessing_plan import (
    LocalAuthorityDocument,
    PostprocessingPhasePlan,
    postprocessing_phase_plan_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runspec_v3 import (
    PostprocessingPhaseRunSpecPayloadV3,
    PostprocessingPhaseRunSpecV3,
)
from bspp.orchestration.contract.postprocessing_submission_events import (
    PostprocessingMaterializedPayload,
)
from bspp.orchestration.contract.runplan import RunPlan, load_runplan
from bspp.orchestration.contract.runspec import RunSpec, WorkflowStepSpec, runspec_from_mapping
from bspp.orchestration.contract.runspec_policies import RunKindPolicyContext, enforce_run_kind_policies
from bspp.orchestration.contract.runspec_validation import validate_active_workflow_static
from bspp.orchestration.control.plan import materialize_runspec_mapping, render_runspec_yaml
from bspp.orchestration.control.postprocessing_attempt_projection import (
    derive_v3_attempt_paths,
    postprocessing_physical_input_locators,
    project_v3_attempt_run_plan,
    validate_v3_attempt_execution_projection,
    validate_v3_attempt_projection,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    canonical_json_bytes as _canonical_json_bytes,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    format_timestamp as _format_timestamp,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    fsync_directory as _fsync_directory,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    fsync_tree_directories as _fsync_tree_directories,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    postprocessing_authority_lock as _postprocessing_authority_lock,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    utc_now as _utc_now,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    write as _write,
)
from bspp.orchestration.control.postprocessing_credential_mounts import (
    snapshot_postprocessing_credential_mounts,
)
from bspp.orchestration.control.postprocessing_identity import (
    build_logical_input_identities,
)
from bspp.orchestration.control.postprocessing_identity_v3 import (
    build_action_semantics_v3,
    build_scientific_identity_v3,
)
from bspp.orchestration.control.postprocessing_phase_types import PostprocessingPhaseMaterializationResult
from bspp.orchestration.control.postprocessing_runtime_qualification import (
    select_pinned_postprocessing_runtime,
)
from bspp.orchestration.control.profiles import ResolvedClusterProfile, resolve_cluster_profile
from bspp.orchestration.control.workflows import load_workflow_template

Clock = Callable[[], datetime]
PhaseRunIdFactory = Callable[[], str]
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_MATERIALIZATION_INPUT_BYTES = 16 * 1024 * 1024


def _validate_materialized_authority(authority_root: Path, phase_run_id: str) -> None:
    from bspp.orchestration.control.postprocessing_authority_v2 import (
        validate_postprocessing_authority,
    )

    validate_postprocessing_authority(authority_root, phase_run_id)


def materialize_postprocessing_phase(
    phase_plan_path: Path,
    *,
    authority_root: Path,
    config_path: Path,
    source_repo: Path | None = None,
    clock: Clock | None = None,
    phase_run_id_factory: PhaseRunIdFactory | None = None,
    phase_plan_document: bytes | None = None,
    config_document: bytes | None = None,
) -> PostprocessingPhaseMaterializationResult:
    """Materialize postprocessing authority without scheduler or remote access."""
    observed_phase_plan = _stable_materialization_input_bytes(phase_plan_path, label="Phase Plan")
    if phase_plan_document is not None and observed_phase_plan != phase_plan_document:
        raise ValueError("postprocessing Phase Plan differs from the captured operator input")
    phase_plan_document = observed_phase_plan
    observed_config = _stable_materialization_input_bytes(config_path, label="Cluster Profile")
    if config_document is not None and observed_config != config_document:
        raise ValueError("postprocessing Cluster Profile differs from the captured operator input")
    config_document = observed_config
    phase_plan = _postprocessing_phase_plan_from_document(phase_plan_document, source=phase_plan_path)
    plan_document = _verified_document(phase_plan.legacy_run_plan, relative_to=phase_plan_path.parent)
    policy_document = _verified_document(phase_plan.acceptance_policy, relative_to=phase_plan_path.parent)
    inventory_document = _verified_document(phase_plan.logical_input_inventory, relative_to=phase_plan_path.parent)
    qualification_document = _verified_document(phase_plan.runtime_qualification, relative_to=phase_plan_path.parent)
    legacy_plan_path = _resolve_document_path(phase_plan.legacy_run_plan, relative_to=phase_plan_path.parent)
    legacy_plan = load_runplan(legacy_plan_path)
    if legacy_plan.target_cluster != phase_plan.target_cluster:
        raise ValueError("postprocessing Phase Plan target_cluster must equal the referenced legacy Run Plan")
    profile = resolve_cluster_profile(
        phase_plan.target_cluster,
        config_path=config_path,
        config_document=config_document,
    )
    policy_payload = yaml.safe_load(policy_document)
    if not isinstance(policy_payload, Mapping):
        raise TypeError("postprocessing acceptance policy document must be a mapping")
    policy = postprocessing_acceptance_policy_from_mapping(policy_payload)

    phase_run_id = (phase_run_id_factory or _new_phase_run_id)()
    attempt_id = "attempt-0001"
    observed_at = (clock or _utc_now)()
    materialized_at = _format_timestamp(observed_at)
    runspec, legacy_bytes, policy_bytes, qualification_bytes = _materialize_postprocessing_attempt(
        phase_plan=phase_plan,
        legacy_plan=legacy_plan,
        legacy_plan_path=legacy_plan_path,
        profile=profile,
        policy=policy,
        plan_document=plan_document,
        policy_document=policy_document,
        inventory_document=inventory_document,
        qualification_document=qualification_document,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        materialized_at=materialized_at,
        source_repo=source_repo or Path.cwd(),
        observed_at=observed_at,
    )
    phase_run = _initial_phase_run(phase_plan, runspec)
    materialized_event = PostprocessingPhaseEvent(
        sequence=1,
        event_type="phase-materialized",
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        occurred_at=materialized_at,
        payload=PostprocessingMaterializedPayload(
            phase_plan_digest=phase_plan.digest,
            phase_runspec_digest=runspec.digest,
            action_graph_digest=runspec.payload.action_graph_digest,
        ),
    )
    _publish_postprocessing_authority(
        authority_root,
        phase_plan=phase_plan,
        phase_run=phase_run,
        runspec=runspec,
        legacy_runspec_bytes=legacy_bytes,
        policy_bytes=policy_bytes,
        qualification_bytes=qualification_bytes,
        materialized_event=materialized_event.to_mapping(),
    )
    _validate_materialized_authority(authority_root, phase_run_id)
    return PostprocessingPhaseMaterializationResult(
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        phase_runspec_digest=runspec.digest,
        authority_root=authority_root,
    )


def load_postprocessing_phase_plan(path: Path) -> PostprocessingPhasePlan:
    return _postprocessing_phase_plan_from_document(path.read_bytes(), source=path)


def _postprocessing_phase_plan_from_document(document: bytes, *, source: Path) -> PostprocessingPhasePlan:
    payload = yaml.safe_load(document)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected postprocessing Phase Plan mapping in {source}")
    return postprocessing_phase_plan_from_mapping(payload)


def _stable_materialization_input_bytes(path: Path, *, label: str) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > _MAX_MATERIALIZATION_INPUT_BYTES
        ):
            raise ValueError(f"postprocessing {label} must be one bounded regular file")
        chunks: list[bytes] = []
        remaining = _MAX_MATERIALIZATION_INPUT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        document = b"".join(chunks)
        after = os.fstat(descriptor)
        reached = path.stat(follow_symlinks=False)
    finally:
        os.close(descriptor)
    if (
        len(document) > _MAX_MATERIALIZATION_INPUT_BYTES
        or _input_stat_signature(before) != _input_stat_signature(after)
        or _input_stat_signature(before) != _input_stat_signature(reached)
    ):
        raise ValueError(f"postprocessing {label} changed while being captured")
    return document


def _input_stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _materialize_postprocessing_attempt(
    *,
    phase_plan: PostprocessingPhasePlan,
    legacy_plan: RunPlan,
    legacy_plan_path: Path,
    profile: ResolvedClusterProfile,
    policy: PostprocessingAcceptancePolicySnapshot,
    plan_document: bytes,
    policy_document: bytes,
    inventory_document: bytes,
    qualification_document: bytes,
    phase_run_id: str,
    attempt_id: str,
    materialized_at: str,
    source_repo: Path,
    observed_at: datetime,
) -> tuple[PostprocessingPhaseRunSpecV3, bytes, bytes, bytes]:
    attempt_paths = _attempt_paths(
        phase_plan=phase_plan,
        legacy_plan=legacy_plan,
        profile=profile,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
    )
    attempt_plan = project_v3_attempt_run_plan(
        legacy_plan,
        profile_output_root=profile.output_root,
        attempt_paths=attempt_paths,
    )
    workflow = load_workflow_template(legacy_plan_path, legacy_plan.workflow_template)
    mapping = materialize_runspec_mapping(attempt_plan, profile, workflow)
    # The general RunSpec expander still derives staging from dataset.name.
    # V3 deliberately preserves that authored scientific selector, so bind the
    # operational staging path to the separate Attempt identity here.
    paths_mapping = mapping.get("paths")
    if not isinstance(paths_mapping, dict):
        raise TypeError("materialized legacy RunSpec paths must be a mapping")
    paths_mapping["staging_dir"] = attempt_paths.staging_dir
    legacy_runspec = runspec_from_mapping(mapping)
    validate_v3_attempt_projection(legacy_runspec, attempt_paths)
    validate_active_workflow_static(legacy_runspec).raise_if_invalid()
    enforce_run_kind_policies(legacy_runspec, RunKindPolicyContext(run_kind=attempt_plan.run_kind))
    legacy_bytes = render_runspec_yaml(mapping).encode()
    # Re-load the exact bytes once so the projection never relies on a mutable mapping later.
    exact_mapping = yaml.safe_load(legacy_bytes)
    if not isinstance(exact_mapping, Mapping):
        raise TypeError("materialized legacy RunSpec YAML must contain a mapping")
    runspec_from_mapping(exact_mapping)

    actions = postprocessing_runtime_actions(legacy_runspec)
    physical_inputs = postprocessing_physical_input_locators(legacy_runspec)
    logical_inputs = _logical_input_manifest(
        inventory_document=inventory_document,
        physical_inputs=physical_inputs,
    )
    qualified_runtime = _qualified_runtime_selection(
        qualification_document,
        phase_plan=phase_plan,
        attempt_id=attempt_id,
        profile=profile,
        source_repo=source_repo,
        observed_at=observed_at,
    )
    require_autorequeue_cap_for_policy(
        policy=phase_plan.autorequeue_policy,
        qualified_runtime=qualified_runtime,
    )
    scientific_identity = _postprocessing_scientific_identity(
        legacy_plan,
        logical_input_manifest=logical_inputs,
    )
    action_semantics = _postprocessing_action_semantics(
        actions,
        legacy_plan=legacy_plan,
        legacy_runspec=legacy_runspec,
        scientific_identity=scientific_identity,
        logical_inputs=logical_inputs,
        acceptance_policy=policy,
    )
    phase_identity = PostprocessingPhaseExecutionIdentity(
        phase_plan_digest=phase_plan.digest,
        logical_input_manifest_digest=logical_inputs.digest,
        scientific_identity_digest=scientific_identity.digest,
        action_semantics_digest=action_semantics.digest,
        acceptance_semantic_digest=policy.semantic_digest,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        qualified_runtime_digest=qualified_runtime.digest,
        output_namespace=phase_plan.output_namespace,
        substitutions=attempt_paths,
    )
    projection = PostprocessingExecutionProjection(
        document_location=f"attempts/{attempt_id}/legacy-runspec.yaml",
        document_sha256=hashlib.sha256(legacy_bytes).hexdigest(),
        document_size_bytes=len(legacy_bytes),
        legacy_schema_version=1,
        phase_identity=phase_identity,
        phase_identity_digest=phase_identity.digest,
    )
    policy_bytes = _canonical_json_bytes(policy.to_mapping())
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    policy_reference = PostprocessingAcceptanceSnapshotReference(
        location=f"attempts/{attempt_id}/acceptance-policy.json",
        sha256=policy_sha256,
        size_bytes=len(policy_bytes),
        semantic_digest=policy.semantic_digest,
        policy_id=policy.policy_id,
    )
    cluster = PostprocessingClusterSnapshot(
        profile_name=profile.name,
        owner=profile.owner,
        transport=profile.transport,
        ssh_target=profile.ssh_target,
        account=profile.account,
        project_root=profile.project_root,
        staging_root=profile.staging_root,
        orchestration_repo=profile.orchestration_repo,
        runtime_image=qualified_runtime.image_path,
        extra_mounts=tuple(PhaseMountSnapshot(source=item.source, target=item.target) for item in profile.extra_mounts),
    )
    runspec = PostprocessingPhaseRunSpecV3(
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        phase_plan_digest=phase_plan.digest,
        materialized_at=materialized_at,
        cluster=cluster,
        credential_mounts=snapshot_postprocessing_credential_mounts(legacy_runspec, profile),
        cluster_output_root=profile.output_root,
        object_output_base_prefix=legacy_plan.storage.s3_output_prefix,
        payload=PostprocessingPhaseRunSpecPayloadV3(
            actions=actions,
            action_graph_digest=postprocessing_action_graph_digest(actions),
            action_semantics_digest=action_semantics.digest,
            action_semantics=action_semantics,
            scientific_identity=scientific_identity,
            logical_inputs=logical_inputs,
            physical_inputs=physical_inputs,
            execution_projection=projection,
            acceptance_policy=policy_reference,
            qualified_runtime=qualified_runtime,
            attempt_paths=attempt_paths,
            autorequeue_policy=phase_plan.autorequeue_policy,
        ),
    )
    validate_v3_attempt_execution_projection(
        runspec,
        legacy_runspec,
        phase_plan_output_namespace=phase_plan.output_namespace,
    )
    return runspec, legacy_bytes, policy_bytes, qualification_document


def require_autorequeue_cap_for_policy(
    *,
    policy: PostprocessingAutorequeuePolicy,
    qualified_runtime: QualifiedPostprocessingRuntimeSelection,
) -> None:
    """Fail closed when an enabled autorequeue policy lacks a qualified cap."""
    if policy.mode == "enabled" and (
        qualified_runtime.requeue_exit != 85
        or qualified_runtime.max_batch_requeue is None
        or qualified_runtime.max_batch_requeue < 0
    ):
        raise ValueError(
            "postprocessing autorequeue requires a qualified Runtime cap pinning RequeueExit 85 "
            "and a non-negative MaxBatchRequeue"
        )


def _attempt_paths(
    *,
    phase_plan: PostprocessingPhasePlan,
    legacy_plan: RunPlan,
    profile: ResolvedClusterProfile,
    phase_run_id: str,
    attempt_id: str,
) -> PostprocessingAttemptPaths:
    return derive_v3_attempt_paths(
        output_namespace=phase_plan.output_namespace,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        cluster_output_root=profile.output_root,
        cluster_staging_root=profile.staging_root,
        object_output_base_prefix=legacy_plan.storage.s3_output_prefix,
    )


def postprocessing_runtime_actions(spec: RunSpec) -> tuple[PostprocessingRuntimeAction, ...]:
    if spec.workflow is None:
        raise ValueError("postprocessing phase requires a legacy active workflow")
    by_name = {step.name: step for step in spec.workflow.steps}
    required = (
        "preflight",
        "analysis-finalize",
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    )
    for name in required:
        if name not in by_name or not by_name[name].run:
            raise ValueError(f"postprocessing sealable-v1 requires workflow step {name!r} with run: true")
    supported = {
        "preflight",
        "recipe",
        "preprocess",
        "slurm",
        "analysis-finalize",
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    }
    unsupported = sorted(step.name for step in spec.workflow.steps if step.run and step.name not in supported)
    if unsupported:
        raise ValueError(f"postprocessing sealable-v1 has active unsupported steps: {', '.join(unsupported)}")
    included = [
        name
        for name in (
            "preflight",
            "recipe",
            "preprocess",
            "slurm",
            "analysis-finalize",
            "acceptance-tar-payload-parity",
            "acceptance-semantic",
            "acceptance-verify-evidence",
        )
        if name in by_name and by_name[name].run
    ]
    included.append("acceptance-adjudication")
    dependencies = _canonical_dependencies(tuple(included))
    actions: list[PostprocessingRuntimeAction] = []
    for name in included:
        step = by_name.get(name)
        resources = _resources_for_action(spec, name, step)
        normalized_array, indexes = normalize_slurm_array(resources.array)
        actions.append(
            PostprocessingRuntimeAction(
                action_id=postprocessing_action_id(name),
                step_name=name,
                dependencies=dependencies[name],
                resources=resources,
                normalized_array=normalized_array,
                expected_task_indexes=indexes,
            )
        )
    return tuple(actions)


def _canonical_dependencies(steps: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    present = set(steps)
    chain = [name for name in ("preflight", "recipe", "preprocess", "slurm", "analysis-finalize") if name in present]
    result: dict[str, tuple[str, ...]] = {chain[0]: ()}
    for index, name in enumerate(chain[1:], start=1):
        result[name] = (postprocessing_action_id(chain[index - 1]),)
    result["acceptance-tar-payload-parity"] = (postprocessing_action_id("analysis-finalize"),)
    result["acceptance-semantic"] = (postprocessing_action_id("analysis-finalize"),)
    result["acceptance-verify-evidence"] = (
        postprocessing_action_id("acceptance-tar-payload-parity"),
        postprocessing_action_id("acceptance-semantic"),
    )
    result["acceptance-adjudication"] = (
        postprocessing_action_id("acceptance-tar-payload-parity"),
        postprocessing_action_id("acceptance-semantic"),
        postprocessing_action_id("acceptance-verify-evidence"),
    )
    return result


def _resources_for_action(
    spec: RunSpec,
    step_name: str,
    step: WorkflowStepSpec | None,
) -> PhaseSlurmResources:
    resource_key = {
        "preflight": "control_cpu",
        "recipe": "control_cpu",
        "preprocess": "control_cpu",
        "slurm": "gpu_worker",
        "analysis-finalize": "analysis_finalize",
        "acceptance-tar-payload-parity": "acceptance_tar_payload_parity",
        "acceptance-semantic": "acceptance_semantic",
        "acceptance-verify-evidence": "control_cpu",
        "acceptance-adjudication": "control_cpu",
    }[step_name]
    resource = spec.resources.get(resource_key)
    if resource is None:
        raise ValueError(f"postprocessing action {step_name!r} requires resources {resource_key!r}")
    array: str | None = None
    if step_name == "slurm":
        array = step.array_range if step is not None and step.array_range is not None else resource.array
    return PhaseSlurmResources(
        partition=resource.partition,
        cpus_per_task=resource.cpus_per_task,
        memory=resource.memory,
        time=resource.time,
        gres=resource.gres,
        array=array,
        nodelist=resource.nodelist,
    )


def _logical_input_manifest(
    *,
    inventory_document: bytes,
    physical_inputs: tuple[PhysicalInputLocator, ...],
) -> PostprocessingLogicalInputIdentityManifestV2:
    return build_logical_input_identities(inventory_document, physical_inputs=physical_inputs)


def _logical_inventory(document: bytes) -> dict[str, tuple[str, str, int]]:
    payload = yaml.safe_load(document)
    if not isinstance(payload, Mapping):
        raise TypeError("postprocessing logical input inventory must be a mapping")
    if set(payload) != {"schema_version", "inventory_kind", "members"}:
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
            or _SHA256.fullmatch(digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or name in result
        ):
            raise ValueError("postprocessing logical input inventory member identity is invalid")
        result[name] = (identity, digest, size)
    return result


def _qualified_runtime_selection(
    document: bytes,
    *,
    phase_plan: PostprocessingPhasePlan,
    attempt_id: str,
    profile: ResolvedClusterProfile,
    source_repo: Path,
    observed_at: datetime,
) -> QualifiedPostprocessingRuntimeSelection:
    if hashlib.sha256(document).hexdigest() != phase_plan.runtime_qualification.sha256:
        raise ValueError("postprocessing Runtime Qualification differs from the Phase Plan authority")
    return select_pinned_postprocessing_runtime(
        document,
        attempt_id=attempt_id,
        profile=profile,
        source_repo=source_repo,
        observed_at=observed_at,
    )


def _postprocessing_scientific_identity(
    legacy_plan: RunPlan,
    *,
    logical_input_manifest: PostprocessingLogicalInputIdentityManifestV2,
) -> PostprocessingScientificIdentityV3:
    return build_scientific_identity_v3(legacy_plan, logical_inputs=logical_input_manifest)


def _postprocessing_action_semantics(
    actions: tuple[PostprocessingRuntimeAction, ...],
    *,
    legacy_plan: RunPlan,
    legacy_runspec: RunSpec,
    scientific_identity: PostprocessingScientificIdentityV3,
    logical_inputs: PostprocessingLogicalInputIdentityManifestV2,
    acceptance_policy: PostprocessingAcceptancePolicySnapshot,
) -> PostprocessingActionSemanticsV3:
    return build_action_semantics_v3(
        actions,
        legacy_plan=legacy_plan,
        legacy_runspec=legacy_runspec,
        scientific_identity=scientific_identity,
        logical_inputs=logical_inputs,
        acceptance_policy=acceptance_policy,
    )


def _initial_phase_run(
    phase_plan: PostprocessingPhasePlan,
    runspec: PostprocessingPhaseRunSpecV3,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase_kind": "postprocessing",
        "phase_run_id": runspec.phase_run_id,
        "phase_plan_location": "phase-plan.json",
        "phase_plan_digest": phase_plan.digest,
        "created_at": runspec.materialized_at,
        "current_attempt_id": runspec.attempt_id,
        "attempts": [
            {
                "schema_version": 1,
                "attempt_id": runspec.attempt_id,
                "ordinal": 1,
                "phase_runspec_location": f"attempts/{runspec.attempt_id}/phase-runspec.json",
                "phase_runspec_digest": runspec.digest,
                "created_at": runspec.materialized_at,
                "status": "materialized",
            }
        ],
        "status": "materialized",
        "sealed": False,
    }


def _publish_postprocessing_authority(
    authority_root: Path,
    *,
    phase_plan: PostprocessingPhasePlan,
    phase_run: Mapping[str, object],
    runspec: PostprocessingPhaseRunSpecV3,
    legacy_runspec_bytes: bytes,
    policy_bytes: bytes,
    qualification_bytes: bytes,
    materialized_event: Mapping[str, object],
) -> None:
    authority_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{runspec.phase_run_id}.staging-", dir=authority_root))
    destination = authority_root / runspec.phase_run_id
    try:
        if destination.exists():
            raise FileExistsError(f"Phase Run authority already exists for {runspec.phase_run_id}")
        _write(staging / "phase-plan.json", _canonical_json_bytes(phase_plan.to_mapping()))
        _write(staging / "phase-run.json", _canonical_json_bytes(phase_run))
        _write(staging / "events" / "000001-phase-materialized.json", _canonical_json_bytes(materialized_event))
        attempt_root = staging / "attempts" / runspec.attempt_id
        _write(attempt_root / "phase-runspec.json", _canonical_json_bytes(runspec.to_mapping()))
        _write(attempt_root / "legacy-runspec.yaml", legacy_runspec_bytes)
        _write(attempt_root / "acceptance-policy.json", policy_bytes)
        _write(attempt_root / "runtime-qualification.json", qualification_bytes)
        _fsync_tree_directories(staging)
        with _postprocessing_authority_lock(authority_root):
            if os.path.lexists(destination):
                raise FileExistsError(f"Phase Run authority already exists for {runspec.phase_run_id}")
            os.rename(staging, destination)
            _fsync_directory(authority_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _verified_document(document: LocalAuthorityDocument, *, relative_to: Path) -> bytes:
    path = _resolve_document_path(document, relative_to=relative_to)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"postprocessing authority document must be a regular non-symlink file: {path}")
    payload = path.read_bytes()
    _verify_bytes(payload, expected_sha256=document.sha256, expected_size=document.size_bytes, label=document.path)
    return payload


def _resolve_document_path(document: LocalAuthorityDocument, *, relative_to: Path) -> Path:
    path = Path(document.path)
    return path if path.is_absolute() else relative_to / path


def _verify_bytes(payload: bytes, *, expected_sha256: str, expected_size: int, label: str) -> None:
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{label} bytes differ from the declared immutable identity")


def _new_phase_run_id() -> str:
    return f"phase-run-{uuid.uuid4().hex}"


__all__ = [
    "load_postprocessing_phase_plan",
    "materialize_postprocessing_phase",
    "postprocessing_physical_input_locators",
    "postprocessing_runtime_actions",
    "require_autorequeue_cap_for_policy",
]
