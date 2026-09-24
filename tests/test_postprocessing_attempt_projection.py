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

"""Exhaustive V3 Attempt output projection policy tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bspp.orchestration.contract.postprocessing_execution import PostprocessingAttemptPaths
from bspp.orchestration.contract.runplan import RunPlan
from bspp.orchestration.contract.runspec import runspec_from_mapping
from bspp.orchestration.control.plan import materialize_runspec_mapping
from bspp.orchestration.control.postprocessing_attempt_projection import (
    derive_v3_attempt_paths,
    project_v3_attempt_run_plan,
    retarget_v3_attempt_runspec_mapping,
    validate_v3_attempt_projection,
)
from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.control.workflows import load_workflow_template

EXAMPLE = Path(__file__).parents[1] / "tests" / "fixtures" / "postprocessing_phase" / "neutral_v3"
PHASE_RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"

LOCAL_OUTPUT_FIELDS = (
    ("storage", "s3_tar_manifest_csv"),
    ("storage", "local_tar_dir"),
    ("storage", "local_tar_manifest_csv"),
    ("analysis_metadata", "csv_path"),
    ("analysis_metadata", "parquet_path"),
    ("analysis_metadata", "selected_ids_path"),
    ("analysis_metadata", "high_quality_from_tars", "work_dir"),
    ("analysis_metadata", "high_quality_from_tars", "publication", "manifest_dir"),
    ("analysis_metadata", "high_quality_from_tars", "publication", "evidence_dir"),
)


def test_saturated_projection_and_retry_retarget_every_attempt_owned_output() -> None:
    plan, authored_root = _saturated_plan()
    first = _attempt_paths("attempt-0001")
    second = _attempt_paths("attempt-0002")

    projected = project_v3_attempt_run_plan(
        plan,
        profile_output_root=str(authored_root.parent),
        attempt_paths=first,
    )

    assert projected.dataset.name == plan.dataset.name
    assert projected.dataset.run_id == first.legacy_run_id
    assert projected.references == plan.references
    assert projected.storage.s3_archive_prefix == plan.storage.s3_archive_prefix
    assert projected.worker.scratch_dir == plan.worker.scratch_dir
    assert projected.data_placement == plan.data_placement
    for field_path in LOCAL_OUTPUT_FIELDS:
        authored = _model_value(plan, field_path)
        observed = _model_value(projected, field_path)
        assert isinstance(authored, Path)
        assert observed == Path(first.output_dir) / authored.relative_to(authored_root)
    assert projected.storage.s3_output_prefix == first.object_prefix
    assert projected.storage.s3_tar_prefix == f"{first.object_prefix}tars/"
    assert projected.storage.gcs_destination_prefix == f"gs://example-delivery/task853/{PHASE_RUN_ID}-attempt-0001/"
    publication = projected.analysis_metadata.high_quality_from_tars.publication
    assert publication.target_prefix == f"s3://example-hq/task853/{PHASE_RUN_ID}-attempt-0001/"

    first_mapping = _runspec_mapping(projected, first)
    first_spec = runspec_from_mapping(first_mapping)
    validate_v3_attempt_projection(first_spec, first)
    second_mapping = retarget_v3_attempt_runspec_mapping(first_mapping, old_paths=first, new_paths=second)
    second_spec = runspec_from_mapping(second_mapping)
    validate_v3_attempt_projection(second_spec, second)
    for field_path in LOCAL_OUTPUT_FIELDS:
        first_value = _model_value(first_spec, field_path)
        second_value = _model_value(second_spec, field_path)
        assert isinstance(first_value, Path)
        assert second_value == Path(second.output_dir) / first_value.relative_to(first.output_dir)
    assert second_spec.storage.s3_tar_prefix == f"{second.object_prefix}tars/"
    assert second_spec.storage.gcs_destination_prefix.endswith(f"/{PHASE_RUN_ID}-attempt-0002/")
    assert second_spec.analysis_metadata.high_quality_from_tars.publication.target_prefix.endswith(
        f"/{PHASE_RUN_ID}-attempt-0002/"
    )
    serialized = yaml.safe_dump(second_mapping)
    assert PHASE_RUN_ID + "-attempt-0001" not in serialized
    assert first.output_dir not in serialized


def test_enabled_hq_publication_requires_explicit_target_prefix() -> None:
    plan, _authored_root = _saturated_plan()
    mapping = plan.model_dump(mode="json", exclude_none=True)
    publication = mapping["analysis_metadata"]["high_quality_from_tars"]["publication"]
    publication["target_prefix"] = None
    publication["default_target_prefix_from_recipe"] = True

    with pytest.raises(ValidationError, match="requires target_prefix"):
        RunPlan.model_validate(mapping)


@pytest.mark.parametrize("field_path", LOCAL_OUTPUT_FIELDS)
@pytest.mark.parametrize("unsafe", ("relative", "outside", "root", "traversal"))
def test_local_outputs_must_be_safe_strict_authored_root_descendants(
    field_path: tuple[str, ...],
    unsafe: str,
) -> None:
    plan, authored_root = _saturated_plan()
    mapping = plan.model_dump(mode="json", exclude_none=True)
    value = {
        "relative": "relative/output.csv",
        "outside": "/other/output.csv",
        "root": str(authored_root),
        "traversal": str(authored_root / "nested" / ".." / "output.csv"),
    }[unsafe]
    _set_mapping_value(mapping, field_path, value)
    changed = RunPlan.model_validate(mapping)

    with pytest.raises(ValueError, match="Attempt output projection is unsafe"):
        project_v3_attempt_run_plan(
            changed,
            profile_output_root=str(authored_root.parent),
            attempt_paths=_attempt_paths("attempt-0001"),
        )


@pytest.mark.parametrize(
    ("field_path", "value"),
    (
        (("storage", "s3_tar_prefix"), "s3://other-bucket/tars/"),
        (("storage", "gcs_destination_prefix"), "s3://wrong-scheme/delivery/"),
        (("storage", "gcs_destination_prefix"), "gs://example-delivery/task853?query=yes"),
        (("storage", "gcs_destination_prefix"), "gs://example-delivery/task853//nested/"),
        (
            ("storage", "gcs_destination_prefix"),
            f"gs://example-delivery/{PHASE_RUN_ID}-attempt-0001/",
        ),
        (
            ("storage", "gcs_destination_prefix"),
            f"gs://example-delivery/{PHASE_RUN_ID}-attempt-0001/nested/",
        ),
        (
            ("analysis_metadata", "high_quality_from_tars", "publication", "target_prefix"),
            f"s3://example-hq/{PHASE_RUN_ID}-attempt-0001/nested/",
        ),
        (
            ("analysis_metadata", "high_quality_from_tars", "publication", "target_prefix"),
            "s3://example-hq/task853/../escape/",
        ),
        (
            ("analysis_metadata", "high_quality_from_tars", "publication", "target_prefix"),
            "s3://example-hq/task853/%2e%2e/escape/",
        ),
    ),
)
def test_remote_output_prefixes_reject_unsafe_or_unprojectable_values(
    field_path: tuple[str, ...],
    value: str,
) -> None:
    plan, authored_root = _saturated_plan()
    mapping = plan.model_dump(mode="json", exclude_none=True)
    _set_mapping_value(mapping, field_path, value)
    changed = RunPlan.model_validate(mapping)

    with pytest.raises(ValueError, match="Attempt output projection is unsafe"):
        project_v3_attempt_run_plan(
            changed,
            profile_output_root=str(authored_root.parent),
            attempt_paths=_attempt_paths("attempt-0001"),
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        (
            "output_dir",
            "/attempt-output/nested/../task853-phase-run-0123456789abcdef0123456789abcdef-attempt-0001",
        ),
        ("evidence_dir", "/split/evidence"),
        ("staging_dir", "/split/staging"),
        ("object_prefix", "s3://example-output/task853/shared/"),
    ),
)
def test_attempt_base_namespaces_are_validated_before_projection(field_name: str, replacement: str) -> None:
    plan, authored_root = _saturated_plan()
    paths = replace(_attempt_paths("attempt-0001"), **{field_name: replacement})

    with pytest.raises(ValueError, match="Attempt output projection is unsafe"):
        project_v3_attempt_run_plan(
            plan,
            profile_output_root=str(authored_root.parent),
            attempt_paths=paths,
        )


@pytest.mark.parametrize(
    "authored_run_id",
    (
        "/absolute-output",
        ".",
        "../outside-output",
        "nested/../../outside-output",
    ),
)
def test_authored_run_id_cannot_escape_or_equal_the_profile_output_root(authored_run_id: str) -> None:
    plan, authored_root = _saturated_plan()
    dataset = plan.dataset.model_copy(update={"run_id": authored_run_id})
    changed = plan.model_copy(update={"dataset": dataset})

    with pytest.raises(ValueError, match=r"Attempt output projection is unsafe at dataset\.run_id"):
        project_v3_attempt_run_plan(
            changed,
            profile_output_root=str(authored_root.parent),
            attempt_paths=_attempt_paths("attempt-0001"),
        )


@pytest.mark.parametrize("profile_output_root", ("relative-output", "/configured/../escape"))
def test_profile_output_root_is_validated_before_joining(profile_output_root: str) -> None:
    plan, _authored_root = _saturated_plan()

    with pytest.raises(ValueError, match="profile_output_root"):
        project_v3_attempt_run_plan(
            plan,
            profile_output_root=profile_output_root,
            attempt_paths=_attempt_paths("attempt-0001"),
        )


@pytest.mark.parametrize(
    "object_base",
    (
        "s3://example-output/task853",
        "s3://example-output/task853//",
        "s3://example-output/task853/?query=yes",
        f"s3://example-output/{PHASE_RUN_ID}-attempt-0001/",
        f"s3://example-output/{PHASE_RUN_ID}-attempt-0001/nested/",
    ),
)
def test_object_base_is_canonical_and_has_no_current_attempt_suffix(object_base: str) -> None:
    with pytest.raises(ValueError, match="Attempt output projection is unsafe"):
        derive_v3_attempt_paths(
            output_namespace="task853",
            phase_run_id=PHASE_RUN_ID,
            attempt_id="attempt-0001",
            cluster_output_root="/attempt-output",
            cluster_staging_root="/attempt-staging",
            object_output_base_prefix=object_base,
        )


@pytest.mark.parametrize("remote_field", ("object", "gcs", "hq"))
def test_remote_attempt_suffix_must_occur_exactly_once_as_the_terminal_segment(remote_field: str) -> None:
    plan, authored_root = _saturated_plan()
    paths = _attempt_paths("attempt-0001")
    projected = project_v3_attempt_run_plan(
        plan,
        profile_output_root=str(authored_root.parent),
        attempt_paths=paths,
    )
    mapping = _runspec_mapping(projected, paths)
    suffix = f"{PHASE_RUN_ID}-attempt-0001"
    if remote_field == "object":
        paths = replace(paths, object_prefix=f"s3://example-output/{suffix}/nested/{suffix}/")
        mapping["storage"]["s3_output_prefix"] = paths.object_prefix
    elif remote_field == "gcs":
        mapping["storage"]["gcs_destination_prefix"] = f"gs://example-delivery/{suffix}/nested/{suffix}/"
    else:
        mapping["analysis_metadata"]["high_quality_from_tars"]["publication"]["target_prefix"] = (
            f"s3://example-hq/{suffix}/nested/{suffix}/"
        )

    with pytest.raises(ValueError, match="exactly once"):
        validate_v3_attempt_projection(runspec_from_mapping(mapping), paths)


def test_validation_rejects_split_output_authority_and_unsafe_collisions() -> None:
    plan, authored_root = _saturated_plan()
    paths = _attempt_paths("attempt-0001")
    projected = project_v3_attempt_run_plan(
        plan,
        profile_output_root=str(authored_root.parent),
        attempt_paths=paths,
    )
    mapping = _runspec_mapping(projected, paths)
    split = yaml.safe_load(yaml.safe_dump(mapping))
    split["analysis_metadata"]["csv_path"] = str(authored_root / "analysis.csv")
    with pytest.raises(ValueError, match=r"analysis_metadata\.csv_path"):
        validate_v3_attempt_projection(runspec_from_mapping(split), paths)

    collision = yaml.safe_load(yaml.safe_dump(mapping))
    collision["analysis_metadata"]["csv_path"] = collision["storage"]["local_tar_manifest_csv"]
    with pytest.raises(ValueError, match="collides"):
        validate_v3_attempt_projection(runspec_from_mapping(collision), paths)


def _saturated_plan() -> tuple[RunPlan, Path]:
    mapping = yaml.safe_load((EXAMPLE / "run-plan.yaml").read_text())
    profile = yaml.safe_load((EXAMPLE / "profiles.yaml").read_text())["clusters"]["example-cluster"]
    authored_root = Path(profile["paths"]["output_root"]) / mapping["dataset"]["run_id"]
    mapping["storage"].update(
        {
            "s3_tar_prefix": f"{mapping['storage']['s3_output_prefix']}tars/",
            "s3_tar_manifest_csv": str(authored_root / "manifests/s3-tars.csv"),
            "gcs_destination_prefix": "gs://example-delivery/task853/",
        }
    )
    mapping["secrets"]["gcs_credentials_ref"] = "env:fixture/gcs"
    mapping["analysis_metadata"]["high_quality_from_tars"] = {
        "enabled": True,
        "s3_prefix": "s3://example-input/hq/",
        "work_dir": str(authored_root / "hq/work"),
        "publication": {
            "enabled": True,
            "target_prefix": "s3://example-hq/task853/",
            "default_target_prefix_from_recipe": False,
            "manifest_dir": str(authored_root / "hq/manifests"),
            "evidence_dir": str(authored_root / "hq/evidence"),
        },
    }
    mapping["data_placement"] = {
        "publication": {
            "tool": "s5cmd",
            "required": False,
            "source": "s3://example-input/hq/",
            "destination": "s3://inactive-destination/must-remain-authored/",
        }
    }
    return RunPlan.model_validate(mapping), authored_root


def _attempt_paths(attempt_id: str) -> PostprocessingAttemptPaths:
    suffix = f"{PHASE_RUN_ID}-{attempt_id}"
    output_dir = f"/attempt-output/task853-{suffix}"
    return PostprocessingAttemptPaths(
        legacy_run_id=f"task853-{suffix}",
        output_dir=output_dir,
        evidence_dir=f"{output_dir}/evidence",
        staging_dir=f"/attempt-staging/task853-{suffix}/staging",
        object_prefix=f"s3://example-output/task853/{suffix}/",
    )


def _runspec_mapping(plan: RunPlan, paths: PostprocessingAttemptPaths) -> dict[str, object]:
    profile = resolve_cluster_profile("example-cluster", config_path=EXAMPLE / "profiles.yaml")
    profile = replace(profile, output_root="/attempt-output")
    workflow = load_workflow_template(EXAMPLE / "run-plan.yaml", plan.workflow_template)
    mapping = materialize_runspec_mapping(plan, profile, workflow)
    mapping["paths"]["staging_dir"] = paths.staging_dir
    return mapping


def _model_value(value: object, field_path: tuple[str, ...]) -> object:
    current = value
    for part in field_path:
        current = getattr(current, part)
    return current


def _set_mapping_value(value: Mapping[str, object], field_path: tuple[str, ...], replacement: object) -> None:
    current = value
    for part in field_path[:-1]:
        nested = current[part]
        assert isinstance(nested, Mapping)
        current = nested
    current[field_path[-1]] = replacement  # type: ignore[index]
