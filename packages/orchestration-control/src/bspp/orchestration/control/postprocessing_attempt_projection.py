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

"""Attempt-owned output projection for postprocessing V3 authority."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn, cast
from urllib.parse import SplitResult, unquote, urlsplit

from bspp.orchestration.contract.phase import validate_phase_attempt_id, validate_phase_run_id
from bspp.orchestration.contract.postprocessing_execution import PostprocessingAttemptPaths
from bspp.orchestration.contract.postprocessing_logical_identity import PhysicalInputLocator
from bspp.orchestration.contract.postprocessing_runspec_v3 import PostprocessingPhaseRunSpecV3
from bspp.orchestration.contract.runplan import RunPlan
from bspp.orchestration.contract.runspec import RunSpec, runspec_from_mapping

_LOCAL_DIRECTORY_FIELDS = (
    ("storage", "local_tar_dir"),
    ("analysis_metadata", "high_quality_from_tars", "work_dir"),
    ("analysis_metadata", "high_quality_from_tars", "publication", "manifest_dir"),
    ("analysis_metadata", "high_quality_from_tars", "publication", "evidence_dir"),
)
_LOCAL_FILE_FIELDS = (
    ("storage", "s3_tar_manifest_csv"),
    ("storage", "local_tar_manifest_csv"),
    ("analysis_metadata", "csv_path"),
    ("analysis_metadata", "parquet_path"),
    ("analysis_metadata", "selected_ids_path"),
)
_GENERATED_ATTEMPT_SUFFIX = re.compile(r"phase-run-[0-9a-f]{32}-attempt-[0-9]{4}")
_OUTPUT_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def derive_v3_attempt_paths(
    *,
    output_namespace: str,
    phase_run_id: str,
    attempt_id: str,
    cluster_output_root: str,
    cluster_staging_root: str,
    object_output_base_prefix: str,
) -> PostprocessingAttemptPaths:
    """Derive exact V3 Attempt paths from independently stored envelope roots."""
    validate_phase_run_id(phase_run_id)
    validate_phase_attempt_id(attempt_id)
    if _OUTPUT_NAMESPACE.fullmatch(output_namespace) is None:
        _projection_error("execution_projection.phase_identity.output_namespace", "is invalid")
    output_root = Path(cluster_output_root)
    staging_root = Path(cluster_staging_root)
    _validate_absolute_lexical_path(output_root, label="cluster_output_root")
    _validate_absolute_lexical_path(staging_root, label="cluster.staging_root")
    suffix = f"{phase_run_id}-{attempt_id}"
    legacy_run_id = f"{output_namespace}-{suffix}"
    output_dir = output_root / legacy_run_id
    staging_dir = staging_root / legacy_run_id / "staging"
    _strict_relative_path(output_dir, root=output_root, label="derived attempt output_dir")
    _strict_relative_path(staging_dir, root=staging_root, label="derived attempt staging_dir")
    object_prefix = _append_generated_attempt_uri(
        object_output_base_prefix,
        generated_suffix=suffix,
        scheme="s3",
        label="object_output_base_prefix",
    )
    return PostprocessingAttemptPaths(
        legacy_run_id=legacy_run_id,
        output_dir=str(output_dir),
        evidence_dir=str(output_dir / "evidence"),
        staging_dir=str(staging_dir),
        object_prefix=object_prefix,
    )


def project_v3_attempt_run_plan(
    plan: RunPlan,
    *,
    profile_output_root: str,
    attempt_paths: PostprocessingAttemptPaths,
) -> RunPlan:
    """Project every active Run Plan output onto one deterministic V3 Attempt."""
    _validate_attempt_paths_base(attempt_paths)
    authored_root = _authored_output_root(profile_output_root, plan.dataset.run_id)
    storage_updates: dict[str, object] = {
        "s3_output_prefix": attempt_paths.object_prefix,
    }
    for field_name in ("s3_tar_manifest_csv", "local_tar_dir", "local_tar_manifest_csv"):
        value = getattr(plan.storage, field_name)
        if value is not None:
            storage_updates[field_name] = _project_local_path(
                value,
                old_root=authored_root,
                new_root=Path(attempt_paths.output_dir),
                label=f"storage.{field_name}",
            )
    if plan.storage.s3_tar_prefix is not None:
        storage_updates["s3_tar_prefix"] = _project_descendant_uri(
            plan.storage.s3_tar_prefix,
            old_root=plan.storage.s3_output_prefix,
            new_root=attempt_paths.object_prefix,
            scheme="s3",
            label="storage.s3_tar_prefix",
        )
    if plan.storage.gcs_destination_prefix is not None:
        storage_updates["gcs_destination_prefix"] = _append_attempt_uri(
            plan.storage.gcs_destination_prefix,
            attempt_paths=attempt_paths,
            scheme="gs",
            label="storage.gcs_destination_prefix",
        )
    storage = plan.storage.model_copy(update=storage_updates)

    analysis = plan.analysis_metadata
    if analysis is not None:
        analysis_updates: dict[str, object] = {}
        for field_name in ("csv_path", "parquet_path", "selected_ids_path"):
            value = getattr(analysis, field_name)
            if value is not None:
                analysis_updates[field_name] = _project_local_path(
                    value,
                    old_root=authored_root,
                    new_root=Path(attempt_paths.output_dir),
                    label=f"analysis_metadata.{field_name}",
                )
        high_quality = analysis.high_quality_from_tars
        high_quality_updates: dict[str, object] = {}
        if high_quality.work_dir is not None:
            high_quality_updates["work_dir"] = _project_local_path(
                high_quality.work_dir,
                old_root=authored_root,
                new_root=Path(attempt_paths.output_dir),
                label="analysis_metadata.high_quality_from_tars.work_dir",
            )
        publication = high_quality.publication
        publication_updates: dict[str, object] = {}
        for field_name in ("manifest_dir", "evidence_dir"):
            value = getattr(publication, field_name)
            if value is not None:
                publication_updates[field_name] = _project_local_path(
                    value,
                    old_root=authored_root,
                    new_root=Path(attempt_paths.output_dir),
                    label=f"analysis_metadata.high_quality_from_tars.publication.{field_name}",
                )
        target = publication.target_prefix
        if target is not None:
            publication_updates["target_prefix"] = _append_attempt_uri(
                target,
                attempt_paths=attempt_paths,
                scheme="s3",
                label="analysis_metadata.high_quality_from_tars.publication.target_prefix",
            )
        if publication_updates:
            high_quality_updates["publication"] = publication.model_copy(update=publication_updates)
        if high_quality_updates:
            analysis_updates["high_quality_from_tars"] = high_quality.model_copy(update=high_quality_updates)
        analysis = analysis.model_copy(update=analysis_updates)

    acceptance = plan.acceptance
    if acceptance is not None:
        acceptance = acceptance.model_copy(update={"candidate_run_name": attempt_paths.legacy_run_id})
    projected = plan.model_copy(
        update={
            "dataset": plan.dataset.model_copy(update={"run_id": attempt_paths.legacy_run_id}),
            "storage": storage,
            "analysis_metadata": analysis,
            "acceptance": acceptance,
        }
    )
    return projected


def retarget_v3_attempt_runspec_mapping(
    value: Mapping[str, object],
    *,
    old_paths: PostprocessingAttemptPaths,
    new_paths: PostprocessingAttemptPaths,
) -> dict[str, object]:
    """Retarget one validated V3 legacy RunSpec without changing authored inputs."""
    old_spec = runspec_from_mapping(value)
    validate_v3_attempt_projection(old_spec, old_paths)
    successor = copy.deepcopy(dict(value))
    replacements = {
        ("dataset", "run_id"): new_paths.legacy_run_id,
        ("paths", "staging_dir"): new_paths.staging_dir,
        ("paths", "output_dir"): new_paths.output_dir,
        ("paths", "log_dir"): str(Path(new_paths.output_dir) / "logs"),
        ("paths", "recipe_dir"): str(Path(new_paths.output_dir) / "rendered_recipe"),
        ("storage", "s3_output_prefix"): new_paths.object_prefix,
        ("submission", "evidence_dir"): new_paths.evidence_dir,
        ("submission", "report_path"): str(Path(new_paths.output_dir) / "RUN_REPORT.md"),
    }
    if old_spec.acceptance is not None:
        replacements[("acceptance", "candidate_run_name")] = new_paths.legacy_run_id
    for field_path, replacement in replacements.items():
        _replace_exact(successor, field_path, replacement)

    for local_field_path in (*_LOCAL_DIRECTORY_FIELDS, *_LOCAL_FILE_FIELDS):
        old_value = _mapping_value(successor, local_field_path)
        if old_value is not None:
            projected = _project_local_path(
                Path(_require_string(old_value, local_field_path)),
                old_root=Path(old_paths.output_dir),
                new_root=Path(new_paths.output_dir),
                label=".".join(local_field_path),
            )
            _set_mapping_value(successor, local_field_path, str(projected))

    old_s3_tar = old_spec.storage.s3_tar_prefix
    if old_s3_tar is not None:
        _set_mapping_value(
            successor,
            ("storage", "s3_tar_prefix"),
            _project_descendant_uri(
                old_s3_tar,
                old_root=old_paths.object_prefix,
                new_root=new_paths.object_prefix,
                scheme="s3",
                label="storage.s3_tar_prefix",
            ),
        )
    for remote_field_path, scheme in (
        (("storage", "gcs_destination_prefix"), "gs"),
        (("analysis_metadata", "high_quality_from_tars", "publication", "target_prefix"), "s3"),
    ):
        old_value = _mapping_value(successor, remote_field_path)
        if old_value is not None:
            _set_mapping_value(
                successor,
                remote_field_path,
                _replace_attempt_uri(
                    _require_string(old_value, remote_field_path),
                    old_paths=old_paths,
                    new_paths=new_paths,
                    scheme=scheme,
                    label=".".join(remote_field_path),
                ),
            )
    validate_v3_attempt_projection(runspec_from_mapping(successor), new_paths)
    return successor


def validate_v3_attempt_projection(spec: RunSpec, paths: PostprocessingAttemptPaths) -> None:
    """Reject a V3 legacy RunSpec whose output authority is split across roots."""
    _validate_attempt_paths_base(paths)
    if spec.paths.recipe_dir is None:
        _projection_error("paths.recipe_dir", "must be materialized for a V3 Attempt")
    if spec.submission is None:
        _projection_error("submission", "must be materialized for a V3 Attempt")
    if spec.submission.report_path is None:
        _projection_error("submission.report_path", "must be materialized for a V3 Attempt")
    expected = {
        "dataset.run_id": paths.legacy_run_id,
        "paths.staging_dir": paths.staging_dir,
        "paths.output_dir": paths.output_dir,
        "paths.log_dir": str(Path(paths.output_dir) / "logs"),
        "paths.recipe_dir": str(Path(paths.output_dir) / "rendered_recipe"),
        "storage.s3_output_prefix": paths.object_prefix,
        "submission.evidence_dir": paths.evidence_dir,
        "submission.report_path": str(Path(paths.output_dir) / "RUN_REPORT.md"),
    }
    observed: dict[str, object] = {
        "dataset.run_id": spec.dataset.run_id,
        "paths.staging_dir": str(spec.paths.staging_dir),
        "paths.output_dir": str(spec.paths.output_dir),
        "paths.log_dir": str(spec.paths.log_dir),
        "paths.recipe_dir": str(spec.paths.recipe_dir),
        "storage.s3_output_prefix": spec.storage.s3_output_prefix,
        "submission.evidence_dir": str(spec.submission.evidence_dir),
        "submission.report_path": str(spec.submission.report_path),
    }
    if spec.acceptance is not None:
        expected["acceptance.candidate_run_name"] = paths.legacy_run_id
        observed["acceptance.candidate_run_name"] = spec.acceptance.candidate_run_name
    for label, expected_value in expected.items():
        if observed[label] != expected_value:
            _projection_error(label, f"expected {expected_value!r}, observed {observed[label]!r}")

    local_values: list[tuple[str, Path, bool]] = [
        ("paths.log_dir", spec.paths.log_dir, True),
        ("paths.recipe_dir", spec.paths.recipe_dir, True),
        ("submission.evidence_dir", spec.submission.evidence_dir, True),
        ("submission.report_path", spec.submission.report_path, False),
    ]
    for field_path in _LOCAL_DIRECTORY_FIELDS:
        value = _model_value(spec, field_path)
        if value is not None:
            local_values.append((".".join(field_path), cast(Path, value), True))
    for field_path in _LOCAL_FILE_FIELDS:
        value = _model_value(spec, field_path)
        if value is not None:
            local_values.append((".".join(field_path), cast(Path, value), False))
    output_root = Path(paths.output_dir)
    for label, value, _is_directory in local_values:
        _strict_relative_path(value, root=output_root, label=label)
    _validate_local_collisions(local_values)

    if spec.storage.s3_tar_prefix is not None:
        _strict_descendant_uri(
            spec.storage.s3_tar_prefix,
            root=paths.object_prefix,
            scheme="s3",
            label="storage.s3_tar_prefix",
        )
    if spec.storage.gcs_destination_prefix is not None:
        _require_attempt_uri(
            spec.storage.gcs_destination_prefix,
            paths=paths,
            scheme="gs",
            label="storage.gcs_destination_prefix",
        )
    publication = spec.analysis_metadata.high_quality_from_tars.publication
    if publication.enabled and publication.target_prefix is None:
        _projection_error(
            "analysis_metadata.high_quality_from_tars.publication.target_prefix",
            "enabled V3 publication must materialize an explicit target_prefix",
        )
    if publication.target_prefix is not None:
        _require_attempt_uri(
            publication.target_prefix,
            paths=paths,
            scheme="s3",
            label="analysis_metadata.high_quality_from_tars.publication.target_prefix",
        )


def validate_v3_attempt_execution_projection(
    phase_runspec: PostprocessingPhaseRunSpecV3,
    legacy_runspec: RunSpec,
    *,
    phase_plan_output_namespace: str,
) -> None:
    """Validate all V3 output substitutions and the preserved physical inputs."""
    credential_mounts = phase_runspec.credential_mounts
    if credential_mounts is not None:
        reference = legacy_runspec.secrets.s3_credentials_ref
        has_aws_mounts = credential_mounts.aws_shared_credentials_file is not None
        if (reference.scheme == "aws") != has_aws_mounts:
            _projection_error(
                "credential_mounts",
                "must contain both AWS locators exactly when the legacy RunSpec uses an AWS profile",
            )
    identity = phase_runspec.payload.execution_projection.phase_identity
    if identity.output_namespace != phase_plan_output_namespace:
        _projection_error(
            "execution_projection.phase_identity.output_namespace",
            "must equal the envelope-derived Phase Plan output namespace",
        )
    if phase_runspec.cluster_output_root is None:
        _projection_error("cluster_output_root", "is missing from executable V3 authority")
    if phase_runspec.object_output_base_prefix is None:
        _projection_error("object_output_base_prefix", "is missing from executable V3 authority")
    expected_paths = derive_v3_attempt_paths(
        output_namespace=phase_plan_output_namespace,
        phase_run_id=phase_runspec.phase_run_id,
        attempt_id=phase_runspec.attempt_id,
        cluster_output_root=phase_runspec.cluster_output_root,
        cluster_staging_root=phase_runspec.cluster.staging_root,
        object_output_base_prefix=phase_runspec.object_output_base_prefix,
    )
    observed_paths = phase_runspec.payload.attempt_paths
    for field_name in ("legacy_run_id", "output_dir", "evidence_dir", "staging_dir", "object_prefix"):
        expected = getattr(expected_paths, field_name)
        observed = getattr(observed_paths, field_name)
        if observed != expected:
            _projection_error(
                f"payload.attempt_paths.{field_name}",
                f"expected envelope-derived {expected!r}, observed {observed!r}",
            )
    validate_v3_attempt_projection(legacy_runspec, expected_paths)
    observed = postprocessing_physical_input_locators(legacy_runspec)
    if observed != phase_runspec.payload.physical_inputs:
        _projection_error(
            "payload.physical_inputs",
            "stored physical inputs differ from the exact projected legacy RunSpec",
        )


def postprocessing_physical_input_locators(spec: RunSpec) -> tuple[PhysicalInputLocator, ...]:
    """Return the physical inputs that output projection must never rewrite."""
    values = {
        "baseline-output": str(spec.acceptance.baseline_output_dir) if spec.acceptance else "",
        "reference-manifest-csv": str(spec.references.manifest_csv),
        "reference-master-parquet": str(spec.references.master_parquet),
        "reference-tracking-parquet": str(spec.references.tracking_parquet),
        "reference-uniprot-duckdb": str(spec.references.uniprot_duckdb),
        "s3-archive-prefix": spec.storage.s3_archive_prefix,
    }
    if spec.references.heterodimer_id_manifest is not None:
        values["reference-heterodimer-id-manifest"] = str(spec.references.heterodimer_id_manifest)
    return tuple(
        PhysicalInputLocator(name=name, locator=locator, purpose="postprocessing-runtime-input")
        for name, locator in sorted(values.items())
        if locator
    )


def _project_local_path(value: Path, *, old_root: Path, new_root: Path, label: str) -> Path:
    relative = _strict_relative_path(value, root=old_root, label=label)
    return new_root / relative


def _authored_output_root(profile_output_root: str, authored_run_id: str) -> Path:
    profile_root = Path(profile_output_root)
    _validate_absolute_lexical_path(profile_root, label="profile_output_root")
    run_id_path = Path(authored_run_id)
    if run_id_path.is_absolute():
        _projection_error("dataset.run_id", "must be relative to profile_output_root")
    authored_root = profile_root / run_id_path
    _strict_relative_path(authored_root, root=profile_root, label="dataset.run_id")
    return authored_root


def _strict_relative_path(value: Path, *, root: Path, label: str) -> Path:
    _validate_absolute_lexical_path(value, label=label)
    _validate_absolute_lexical_path(root, label=f"{label} root")
    try:
        relative = value.relative_to(root)
    except ValueError:
        _projection_error(label, f"must be a lexical descendant of {root}")
    if relative == Path("."):
        _projection_error(label, "must be a strict descendant, not the output root itself")
    return relative


def _validate_absolute_lexical_path(value: Path, *, label: str) -> None:
    if not value.is_absolute():
        _projection_error(label, "must be an absolute path beneath the configured authored output root")
    if ".." in value.parts:
        _projection_error(label, "must not contain '..' traversal")


def _validate_attempt_paths_base(paths: PostprocessingAttemptPaths) -> None:
    output_dir = Path(paths.output_dir)
    staging_dir = Path(paths.staging_dir)
    _validate_absolute_lexical_path(output_dir, label="attempt_paths.output_dir")
    _validate_absolute_lexical_path(staging_dir, label="attempt_paths.staging_dir")
    if output_dir.name != paths.legacy_run_id:
        _projection_error("attempt_paths.output_dir", "must terminate in the generated legacy run id")
    if paths.evidence_dir != str(output_dir / "evidence"):
        _projection_error("attempt_paths.evidence_dir", "must be the Attempt output evidence directory")
    if staging_dir.name != "staging" or staging_dir.parent.name != paths.legacy_run_id:
        _projection_error("attempt_paths.staging_dir", "must terminate in the generated legacy run id and /staging")
    _require_attempt_uri(
        paths.object_prefix,
        paths=paths,
        scheme="s3",
        label="attempt_paths.object_prefix",
    )


def _validate_local_collisions(values: list[tuple[str, Path, bool]]) -> None:
    for index, (label, value, is_directory) in enumerate(values):
        for other_label, other, other_is_directory in values[index + 1 :]:
            if value == other:
                _projection_error(label, f"collides with {other_label} at {value}")
            if not is_directory and _is_descendant(other, value):
                _projection_error(label, f"file path contains {other_label}: {value}")
            if not other_is_directory and _is_descendant(value, other):
                _projection_error(other_label, f"file path contains {label}: {other}")


def _is_descendant(value: Path, root: Path) -> bool:
    try:
        return value.relative_to(root) != Path(".")
    except ValueError:
        return False


def _append_attempt_uri(
    value: str,
    *,
    attempt_paths: PostprocessingAttemptPaths,
    scheme: str,
    label: str,
) -> str:
    return _append_generated_attempt_uri(
        value,
        generated_suffix=_attempt_suffix(attempt_paths),
        scheme=scheme,
        label=label,
    )


def _append_generated_attempt_uri(value: str, *, generated_suffix: str, scheme: str, label: str) -> str:
    parsed = _validated_uri(value, scheme=scheme, label=label)
    if _uri_segment_count(parsed, generated_suffix) != 0:
        _projection_error(label, "authored base must not already contain the generated Attempt suffix")
    return f"{value}{generated_suffix}/"


def _replace_attempt_uri(
    value: str,
    *,
    old_paths: PostprocessingAttemptPaths,
    new_paths: PostprocessingAttemptPaths,
    scheme: str,
    label: str,
) -> str:
    _require_attempt_uri(value, paths=old_paths, scheme=scheme, label=label)
    old_tail = f"{_attempt_suffix(old_paths)}/"
    return f"{value[: -len(old_tail)]}{_attempt_suffix(new_paths)}/"


def _require_attempt_uri(
    value: str,
    *,
    paths: PostprocessingAttemptPaths,
    scheme: str,
    label: str,
) -> None:
    parsed = _validated_uri(value, scheme=scheme, label=label)
    suffix = _attempt_suffix(paths)
    expected_tail = f"/{suffix}/"
    if not value.endswith(expected_tail) or _uri_segment_count(parsed, suffix) != 1:
        _projection_error(label, f"must contain the generated suffix exactly once, terminally as {expected_tail}")


def _project_descendant_uri(value: str, *, old_root: str, new_root: str, scheme: str, label: str) -> str:
    relative = _strict_descendant_uri(value, root=old_root, scheme=scheme, label=label)
    _validated_uri(new_root, scheme=scheme, label="attempt_paths.object_prefix")
    return f"{new_root}{relative}"


def _strict_descendant_uri(value: str, *, root: str, scheme: str, label: str) -> str:
    parsed = _validated_uri(value, scheme=scheme, label=label)
    parsed_root = _validated_uri(root, scheme=scheme, label=f"{label} root")
    if (parsed.scheme, parsed.netloc) != (parsed_root.scheme, parsed_root.netloc) or not parsed.path.startswith(
        parsed_root.path
    ):
        _projection_error(label, f"must be a descendant of {root}")
    relative = parsed.path[len(parsed_root.path) :]
    if not relative:
        _projection_error(label, f"must be a strict descendant of {root}")
    return relative


def _validated_uri(value: str, *, scheme: str, label: str) -> SplitResult:
    parsed = urlsplit(value)
    if (
        parsed.scheme != scheme
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
        or not parsed.path.endswith("/")
    ):
        _projection_error(
            label,
            f"must be a canonical {scheme}:// prefix ending in '/' without credentials/query/fragment",
        )
    interior_segments = parsed.path.split("/")[1:-1]
    decoded_segments = tuple(unquote(segment) for segment in interior_segments)
    if any(
        segment in {"", ".", ".."}
        or "/" in segment
        or "\\" in segment
        or any(ord(character) < 32 for character in segment)
        for segment in decoded_segments
    ):
        _projection_error(label, "must not contain traversal or empty path segments")
    return parsed


def _uri_segment_count(value: SplitResult, segment: str) -> int:
    return sum(unquote(item) == segment for item in value.path.split("/")[1:-1])


def _attempt_suffix(paths: PostprocessingAttemptPaths) -> str:
    match = re.search(rf"({_GENERATED_ATTEMPT_SUFFIX.pattern})$", paths.legacy_run_id)
    if match is None:
        _projection_error("attempt_paths.legacy_run_id", "must end in the generated phase-run Attempt suffix")
    return match.group(1)


def _replace_exact(value: dict[str, object], field_path: tuple[str, ...], replacement: object) -> None:
    parent = _mapping_parent(value, field_path)
    if field_path[-1] not in parent:
        _projection_error(".".join(field_path), "is absent from the stored legacy RunSpec")
    parent[field_path[-1]] = replacement


def _mapping_parent(value: dict[str, object], field_path: tuple[str, ...]) -> dict[str, object]:
    current = value
    for part in field_path[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            _projection_error(".".join(field_path), f"parent section {part!r} is absent")
        current = nested
    return current


def _mapping_value(value: Mapping[str, object], field_path: tuple[str, ...]) -> object:
    current: object = value
    for part in field_path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _set_mapping_value(value: dict[str, object], field_path: tuple[str, ...], replacement: object) -> None:
    _mapping_parent(value, field_path)[field_path[-1]] = replacement


def _model_value(value: object, field_path: tuple[str, ...]) -> object:
    current = value
    for part in field_path:
        current = getattr(current, part)
    return current


def _require_string(value: object, field_path: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        _projection_error(".".join(field_path), "must be a string in the stored legacy RunSpec")
    return value


def _projection_error(label: str, detail: str) -> NoReturn:
    raise ValueError(
        f"postprocessing V3 Attempt output projection is unsafe at {label}: {detail}; "
        "fix the authored Run Plan and materialize a fresh V3 Phase Run"
    )


__all__ = [
    "derive_v3_attempt_paths",
    "postprocessing_physical_input_locators",
    "project_v3_attempt_run_plan",
    "retarget_v3_attempt_runspec_mapping",
    "validate_v3_attempt_execution_projection",
    "validate_v3_attempt_projection",
]
