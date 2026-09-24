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

"""Run Plan expansion into concrete RunSpec input mappings."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from bspp.orchestration.contract.runplan import RunPlan, load_runplan
from bspp.orchestration.contract.runspec import REFERENCE_ARTIFACT_NAMES, runspec_from_mapping
from bspp.orchestration.contract.runspec_policies import RunKindPolicyContext, enforce_run_kind_policies
from bspp.orchestration.contract.runspec_validation import validate_active_workflow_static
from bspp.orchestration.control.profiles import ResolvedClusterProfile, resolve_cluster_profile
from bspp.orchestration.control.workflows import load_workflow_template

ORCHESTRATION_CONTAINER_TARGET = "/workspace/bspp-orchestration"
AFDB_TOOLKIT_CONTAINER_TARGET = "/workspace/AFDB-Integration-Kit"


def expand_run_plan_to_mapping(run_plan_path: Path, *, config_path: Path) -> dict[str, object]:
    """Expand a Run Plan into a validated active RunSpec input mapping."""
    run_plan = load_runplan(run_plan_path)
    profile = resolve_cluster_profile(run_plan.target_cluster, config_path=config_path)
    workflow = load_workflow_template(run_plan_path, run_plan.workflow_template)
    mapping = materialize_runspec_mapping(run_plan, profile, workflow)
    spec = runspec_from_mapping(mapping)
    validate_active_workflow_static(spec).raise_if_invalid()
    enforce_run_kind_policies(spec, RunKindPolicyContext(run_kind=run_plan.run_kind))
    return mapping


def materialize_runspec_mapping(
    run_plan: RunPlan,
    profile: ResolvedClusterProfile,
    workflow: Mapping[str, object],
) -> dict[str, object]:
    """Materialize a RunSpec input mapping from already-loaded Run Plan inputs."""
    output_dir = Path(profile.output_root) / run_plan.dataset.run_id
    mapping: dict[str, object] = {
        "run_kind": run_plan.run_kind,
        "dataset": _dump_model(run_plan.dataset),
        "cluster": {
            "name": profile.name,
            "account": profile.account,
            "owner": profile.owner,
        },
        "paths": {
            "project_root": profile.project_root,
            "staging_dir": str(Path(profile.staging_root) / run_plan.dataset.name / "staging"),
            "output_dir": str(output_dir),
            "log_dir": str(output_dir / "logs"),
            "orchestration_repo": profile.orchestration_repo,
            "recipe_dir": str(output_dir / "rendered_recipe"),
        },
        "references": _reference_mapping(run_plan),
        "container": {
            "image": _runtime_image_cache_path(profile),
            "mounts": _container_mounts(profile),
        },
        "resources": _resources_mapping(run_plan, profile),
        "worker": _dump_model(run_plan.worker),
        "storage": _dump_model(run_plan.storage),
    }
    if run_plan.data_placement is not None:
        mapping["data_placement"] = _dump_model(run_plan.data_placement)
    if run_plan.analysis_metadata is not None:
        mapping["analysis_metadata"] = _dump_model(run_plan.analysis_metadata)
    if run_plan.validation is not None:
        mapping["validation"] = _dump_model(run_plan.validation)
    mapping["workflow"] = dict(workflow)
    mapping["submission"] = {
        "evidence_dir": str(output_dir / "evidence"),
        "report_path": str(output_dir / "RUN_REPORT.md"),
    }
    if run_plan.acceptance is not None:
        mapping["acceptance"] = _dump_model(run_plan.acceptance)
    mapping["secrets"] = _dump_model(run_plan.secrets)
    if profile.afdb_toolkit_repo is not None:
        paths = mapping["paths"]
        assert isinstance(paths, dict)
        paths["afdb_toolkit_repo"] = profile.afdb_toolkit_repo
    return mapping


def render_runspec_yaml(mapping: Mapping[str, object]) -> str:
    """Render an expanded RunSpec mapping as stable YAML."""
    return yaml.safe_dump(dict(mapping), sort_keys=False)


def _dump_model(model: Any) -> dict[str, object]:
    return cast(dict[str, object], model.model_dump(mode="json", exclude_none=True))


def _reference_mapping(run_plan: RunPlan) -> dict[str, object]:
    """Preserve authored reference provenance in the legacy RunSpec projection."""
    references = _dump_model(run_plan.references)
    for name in REFERENCE_ARTIFACT_NAMES:
        artifact = run_plan.references.artifacts.get(name)
        if artifact is not None:
            references[name] = _dump_model(artifact)
    return references


def _container_mounts(profile: ResolvedClusterProfile) -> list[dict[str, str]]:
    mounts = [
        {"source": profile.orchestration_repo, "target": ORCHESTRATION_CONTAINER_TARGET},
        {"source": profile.project_root, "target": profile.project_root},
    ]
    if profile.afdb_toolkit_repo is not None:
        mounts.insert(1, {"source": profile.afdb_toolkit_repo, "target": AFDB_TOOLKIT_CONTAINER_TARGET})
    mounts.extend(mount.model_dump(mode="json") for mount in profile.extra_mounts)
    return mounts


def _runtime_image_cache_path(profile: ResolvedClusterProfile) -> str:
    if profile.runtime_image_cache_root is None:
        return profile.image
    return str(Path(profile.runtime_image_cache_root) / Path(profile.image).name)


def _resources_mapping(run_plan: RunPlan, profile: ResolvedClusterProfile) -> dict[str, dict[str, object]]:
    resources: dict[str, dict[str, object]] = {
        name: resource.model_dump(mode="json", exclude_none=True) for name, resource in profile.resources.items()
    }
    gpu_worker = resources.get("gpu_worker")
    if gpu_worker is not None:
        gpu_worker["array"] = run_plan.dataset.array
    return resources


__all__ = [
    "AFDB_TOOLKIT_CONTAINER_TARGET",
    "ORCHESTRATION_CONTAINER_TARGET",
    "expand_run_plan_to_mapping",
    "materialize_runspec_mapping",
    "render_runspec_yaml",
]
