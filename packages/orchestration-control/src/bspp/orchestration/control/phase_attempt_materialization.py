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

"""Shared scheduler-free construction of immutable Phase Attempt RunSpecs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import yaml

from bspp.orchestration.contract.database_placement import (
    SELECTED_DATABASE_ROOT,
    build_preprocessing_database_binding,
    database_source_manifest_names,
)
from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseSourceManifest,
    canonical_database_source_manifest_bytes,
    database_source_manifest_from_mapping,
)
from bspp.orchestration.contract.phase import (
    PhaseMountSnapshot,
    PhasePlan,
    PhaseRunSpec,
    PhaseSlurmResources,
    PreprocessingPhaseRunSpecPayload,
    PreprocessingRuntimeAction,
    ResolvedClusterSnapshot,
)
from bspp.orchestration.contract.preprocessing_execution import (
    PreprocessingChunkExecutionIntent,
    materialize_preprocessing_chunk_execution_plan,
)
from bspp.orchestration.contract.preprocessing_runtime import QualifiedPreprocessingRuntimeSelection
from bspp.orchestration.control.preprocessing_runtime_qualification import (
    check_preprocessing_runtime_qualification,
)
from bspp.orchestration.control.profiles import (
    ResolvedClusterProfile,
    resolve_cluster_profile,
    resolve_database_manifest_path,
)


@dataclass(frozen=True)
class PhaseAttemptOperationalSelection:
    profile: ResolvedClusterProfile
    qualified_runtime: QualifiedPreprocessingRuntimeSelection


def resolve_phase_attempt_operational_selection(
    profile_name: str,
    *,
    config_path: Path,
    source_repo: Path,
    now: datetime,
) -> PhaseAttemptOperationalSelection:
    """Resolve the exact current operational authority without scheduler access."""
    profile = resolve_cluster_profile(profile_name, config_path=config_path)
    return qualify_phase_attempt_operational_selection(
        profile,
        source_repo=source_repo,
        now=now,
    )


def qualify_phase_attempt_operational_selection(
    profile: ResolvedClusterProfile,
    *,
    source_repo: Path,
    now: datetime,
) -> PhaseAttemptOperationalSelection:
    """Qualify one already-resolved profile without changing validation order."""
    qualified_runtime = check_preprocessing_runtime_qualification(
        profile=profile,
        source_repo=source_repo,
        now=now,
    )
    return PhaseAttemptOperationalSelection(
        profile=profile,
        qualified_runtime=qualified_runtime,
    )


def materialize_preprocessing_attempt_runspec(
    *,
    phase_run_id: str,
    attempt_id: str,
    phase_plan: PhasePlan,
    materialized_at: str,
    operational: PhaseAttemptOperationalSelection,
    rematerialize_runtime_image: bool,
) -> PhaseRunSpec:
    """Construct one complete candidate RunSpec from Plan plus named authority."""
    intent = phase_plan.payload.chunk_execution_intent
    if rematerialize_runtime_image:
        selected_image = operational.qualified_runtime.qualification_tuple.cluster_image_path
        intent = replace(intent, site=intent.site.model_copy(update={"container_image": selected_image}))
    else:
        _verify_authored_runtime_image(phase_plan, operational.qualified_runtime)
    _verify_preprocessing_runtime_semantics(intent)
    if intent.scientific.use_env:
        raise ValueError(
            "preprocessing Phase Plan use_env=true is refused: "
            "no adapter-v3 real-kernel characterization exists for the --use-env 1 search argv"
        )
    source_manifest = _load_database_source_manifest(operational.profile, phase_plan)
    primary_name, metagenomic_name = database_source_manifest_names(source_manifest)
    execution = materialize_preprocessing_chunk_execution_plan(
        intent,
        selected_database_root=SELECTED_DATABASE_ROOT,
        primary_database_name=primary_name,
        metagenomic_database_name=metagenomic_name,
    )
    cluster = _cluster_snapshot(operational.profile, operational.qualified_runtime)
    resources = _preprocessing_resources(operational.profile)
    if rematerialize_runtime_image and resources.array is not None:
        raise ValueError("Phase Retry does not support Slurm arrays")
    action = PreprocessingRuntimeAction(
        action_id=f"preprocessing-chunk-{phase_plan.payload.work_plan.chunks[0].ordinal:06d}",
        dependencies=(),
        resources=resources,
        payload=execution,
    )
    database = build_preprocessing_database_binding(
        selection=phase_plan.payload.database,
        source_manifest=source_manifest,
        source_manifest_projection=f"attempts/{attempt_id}/database-source-manifest.json",
        staging=operational.profile.database_staging,
        gpuserver_argv=execution.gpuserver_argv,
        search_argv=execution.search_argv,
    )
    return PhaseRunSpec(
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        phase_plan_digest=phase_plan.digest,
        materialized_at=materialized_at,
        input_location=phase_plan.input_location,
        cluster=cluster,
        payload=PreprocessingPhaseRunSpecPayload(
            work_plan=phase_plan.payload.work_plan,
            actions=(action,),
            database=database,
            transport=phase_plan.payload.transport,
            s3_publish_prefix=phase_plan.payload.s3_publish_prefix,
        ),
    )


def _cluster_snapshot(
    profile: ResolvedClusterProfile,
    qualified_runtime: QualifiedPreprocessingRuntimeSelection,
) -> ResolvedClusterSnapshot:
    image_path = qualified_runtime.qualification_tuple.cluster_image_path
    return ResolvedClusterSnapshot(
        profile_name=profile.name,
        owner=profile.owner,
        transport=profile.transport,
        ssh_target=profile.ssh_target,
        account=profile.account,
        project_root=profile.project_root,
        output_root=profile.output_root,
        staging_root=profile.staging_root,
        orchestration_repo=profile.orchestration_repo,
        runtime_image=image_path,
        preprocessing_runtime=qualified_runtime,
        source_bundle_root=profile.source_bundle_root,
        runtime_image_cache_root=profile.runtime_image_cache_root,
        runtime_qualification_root=profile.runtime_qualification_root,
        runtime_qualification_expires_hours=profile.runtime_qualification_expires_hours,
        extra_mounts=tuple(
            PhaseMountSnapshot(source=mount.source, target=mount.target) for mount in profile.extra_mounts
        ),
    )


def _verify_authored_runtime_image(
    phase_plan: PhasePlan,
    qualified_runtime: QualifiedPreprocessingRuntimeSelection,
) -> None:
    selected_image = qualified_runtime.qualification_tuple.cluster_image_path
    if phase_plan.payload.chunk_execution_intent.site.container_image != selected_image:
        raise ValueError("preprocessing Phase Plan container_image does not match the qualified preprocessing runtime")


def _verify_preprocessing_runtime_semantics(execution: PreprocessingChunkExecutionIntent) -> None:
    site = execution.site
    expected = {
        "mmseqs_executable": "/usr/local/bin/mmseqs",
        "colabfold_search_executable": "/usr/local/bin/colabfold_search",
        "tar_executable": "/usr/bin/tar",
        "lz4_executable": "/usr/bin/lz4",
    }
    mismatches = [
        f"{field_name}={getattr(site, field_name)!r} (expected {expected_path!r})"
        for field_name, expected_path in expected.items()
        if getattr(site, field_name) != expected_path
    ]
    if mismatches:
        detail = "; ".join(mismatches)
        raise ValueError(f"preprocessing Phase Plan executable paths do not match the qualified image: {detail}")


def _load_database_source_manifest(
    profile: ResolvedClusterProfile,
    phase_plan: PhasePlan,
) -> DatabaseSourceManifest:
    path = resolve_database_manifest_path(profile, phase_plan.payload.database)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Database Source Manifest must be a regular non-symlink file")
    document = path.read_bytes()
    payload = yaml.safe_load(document)
    if not isinstance(payload, dict):
        raise ValueError("Database Source Manifest document must be a mapping")
    manifest = database_source_manifest_from_mapping(payload)
    if document != canonical_database_source_manifest_bytes(manifest):
        raise ValueError("Database Source Manifest bytes must use the exact canonical encoding")
    if manifest.database_set != phase_plan.payload.database.database_set:
        raise ValueError("Database Source Manifest identity does not match the Phase Plan selection")
    return manifest


def _preprocessing_resources(profile: ResolvedClusterProfile) -> PhaseSlurmResources:
    try:
        selected = profile.resources["gpu_worker"]
    except KeyError as exc:
        raise ValueError(f"Cluster Profile {profile.name!r} does not define required gpu_worker resources") from exc
    return PhaseSlurmResources(
        partition=selected.partition,
        cpus_per_task=selected.cpus_per_task,
        memory=selected.memory,
        time=selected.time,
        gres=selected.gres,
        array=selected.array,
        nodelist=selected.nodelist,
        nodes=selected.nodes,
        tasks_per_node=selected.tasks_per_node,
        gpus_per_task=selected.gpus_per_task,
        max_parallel=selected.max_parallel,
    )


__all__ = [
    "PhaseAttemptOperationalSelection",
    "materialize_preprocessing_attempt_runspec",
    "qualify_phase_attempt_operational_selection",
    "resolve_phase_attempt_operational_selection",
]
