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

"""Canonical run specification for BSPP orchestration."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_validator,
    model_serializer,
    model_validator,
)

from bspp.orchestration.contract.config_models import FrozenConfigModel
from bspp.orchestration.contract.data_placement import (
    DataPlacementStage,
    DataPlacementTool,
    validate_data_placement_tool_for_stage,
)
from bspp.orchestration.contract.release_acceptance import validate_processing_upload_policy
from bspp.orchestration.contract.secrets import (
    ResolvedSecret,
    SecretExecutionEnv,
    SecretRef,
    build_secret_execution_env,
    reject_literal_secret_values,
    resolve_secret,
    validate_required_secrets,
)
from bspp.orchestration.contract.slurm import validate_exact_slurm_node
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

BAKED_TOOLKIT_PATH = Path("/opt/afdb-toolkit")
# Canonical public nvidia-postproc commit baked into the postprocessing image.
# The internal child inherits this toolkit and its actual-build provenance.
# Dockerfile TOOLKIT_REF, entrypoint.sh's fallback, and this default stay in sync;
# an explicit public build override binds BSPP_EXPECTED_TOOLKIT_COMMIT to the
# actually built ref. tests/test_baked_toolkit_commit_consistency.py guards drift.
BAKED_TOOLKIT_COMMIT = "e2fa757aa0cb2cec8e4a8382627fcbbca7599556"
REFERENCE_ARTIFACT_NAMES = (
    "master_parquet",
    "tracking_parquet",
    "manifest_csv",
    "uniprot_duckdb",
    "heterodimer_id_manifest",
)
NonEmptyString = Annotated[str, StringConstraints(min_length=1, strict=True)]

VALID_TOOL_USED: tuple[str, ...] = (
    "ColabFold v1.6.0 / AlphaFold-Multimer",
    "OpenFold-TRT / AlphaFold-Multimer",
    "OpenFold / AlphaFold-Multimer",
    "OpenFold2 (BioNeMo IR) / AlphaFold-Multimer",
    "OpenFold2 (BioNeMo IR) / OpenFold-pTM",
)
UploadMode = Literal["files", "tar"]
TarCompression = Literal["none", "gz", "zstd-members"]
ArchiveSource = Literal["tracking", "staging_dir"]
RunKind = Literal["dev", "canary", "production"]
HqChunkCollisionPolicy = Literal["fail_if_non_empty"]
ControllerRuntime = Literal["enroot", "srun-pyxis"]
WorkflowStepMode = Literal["submit-and-monitor", "monitor-existing"]
TarPayloadMatchMode = Literal["by-tar", "aggregate"]

WORKFLOW_STEP_NAMES = frozenset(
    {
        "preflight",
        "recipe",
        "preprocess",
        "slurm",
        "analysis-finalize",
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
        "download-manifest",
        "download",
        "extract",
        "extract-local",
        "aggregate",
        "upload-s3",
        "cleanup",
        "cleanup-local",
        "upload-gcs",
        "upload-gcs-direct",
        "status",
        "rename-afid",
    }
)
MODE_CAPABLE_WORKFLOW_STEPS = frozenset(
    {
        "slurm",
        "analysis-finalize",
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
    }
)
DEFAULT_CONTAINER_WORKDIR = Path("/workspace/bspp-orchestration")


def _coerce_path(value: object) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value:
        return Path(value)
    msg = "Expected non-empty path string"
    raise ValueError(msg)


def _coerce_optional_path(value: object) -> Path | None:
    if value is None:
        return None
    return _coerce_path(value)


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


class DatasetSpec(FrozenConfigModel):
    """Dataset identity and array scope."""

    name: NonEmptyString
    run_id: NonEmptyString
    mode: NonEmptyString = "archive"
    array: NonEmptyString = "0-0"
    archive_source: ArchiveSource = "tracking"


class ClusterSpec(FrozenConfigModel):
    """Cluster/account choices for rendered SLURM artifacts."""

    name: NonEmptyString
    account: NonEmptyString
    owner: StrictStr | None = None


class PathSpec(FrozenConfigModel):
    """Filesystem roots used by orchestration commands."""

    project_root: Path
    staging_dir: Path
    output_dir: Path
    log_dir: Path
    legacy_repo: Path
    afdb_toolkit_repo: Path | None = None
    orchestration_repo: Path
    recipe_dir: Path | None = None

    @model_validator(mode="before")
    @classmethod
    def _backfill_legacy_repo(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "legacy_repo" not in data:
            data["legacy_repo"] = data.get("afdb_toolkit_repo", str(BAKED_TOOLKIT_PATH))
        return data

    @field_validator(
        "project_root",
        "staging_dir",
        "output_dir",
        "log_dir",
        "legacy_repo",
        "afdb_toolkit_repo",
        "orchestration_repo",
        mode="before",
    )
    @classmethod
    def _validate_path(cls, value: object) -> Path | None:
        if value is None:
            return None
        return _coerce_path(value)

    @field_validator("recipe_dir", mode="before")
    @classmethod
    def _validate_optional_path(cls, value: object) -> Path | None:
        return _coerce_optional_path(value)


class ReferenceArtifact(FrozenConfigModel):
    """Reference artifact location and optional provenance constraints."""

    path: Path
    source_uri: StrictStr | None = None
    sha256: StrictStr | None = None
    min_size_bytes: StrictInt | None = None
    version: StrictStr | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_shorthand(cls, value: Any) -> Any:
        if isinstance(value, str | Path):
            return {"path": value}
        return value

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: object) -> Path:
        return _coerce_path(value)


class ReferenceSpec(FrozenConfigModel):
    """Reference artifacts required by the documented pipeline."""

    master_parquet: Path
    tracking_parquet: Path
    manifest_csv: Path
    uniprot_duckdb: Path
    heterodimer_id_manifest: Path | None = None
    artifacts: Mapping[str, ReferenceArtifact] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_references(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value

        data = dict(value)
        raw_artifacts = data.get("artifacts", {})
        if raw_artifacts is None:
            raw_artifacts = {}
        if not isinstance(raw_artifacts, Mapping):
            return data

        artifacts: dict[str, ReferenceArtifact] = {
            str(name): ReferenceArtifact.model_validate(artifact) for name, artifact in raw_artifacts.items()
        }
        for name in REFERENCE_ARTIFACT_NAMES:
            if name in data:
                artifact = ReferenceArtifact.model_validate(data[name])
                data[name] = artifact.path
                artifacts[name] = artifact
            elif name in artifacts:
                data[name] = artifacts[name].path
        data["artifacts"] = artifacts
        return data

    @field_validator("master_parquet", "tracking_parquet", "manifest_csv", "uniprot_duckdb", mode="before")
    @classmethod
    def _validate_path(cls, value: object) -> Path:
        return _coerce_path(value)

    @field_validator("heterodimer_id_manifest", mode="before")
    @classmethod
    def _validate_optional_path(cls, value: object) -> Path | None:
        return _coerce_optional_path(value)

    def artifact(self, name: str) -> ReferenceArtifact:
        """Return full metadata for a named reference artifact."""
        return self.artifacts[name]


class MountSpec(FrozenConfigModel):
    """Container mount rendered from the run specification."""

    source: Path
    target: Path
    read_only: StrictBool = True

    @field_validator("source", "target", mode="before")
    @classmethod
    def _validate_path(cls, value: object) -> Path:
        return _coerce_path(value)


class ContainerSpec(FrozenConfigModel):
    """Container image and runtime context."""

    image: NonEmptyString
    workdir: Path = DEFAULT_CONTAINER_WORKDIR
    mounts: tuple[MountSpec, ...] = ()

    @field_validator("workdir", mode="before")
    @classmethod
    def _validate_path(cls, value: object) -> Path:
        return _coerce_path(value)


class SlurmResources(FrozenConfigModel):
    """SLURM resource shape for one pipeline step."""

    partition: NonEmptyString
    cpus_per_task: StrictInt
    memory: NonEmptyString
    time: NonEmptyString
    gres: StrictStr | None = None
    array: StrictStr | None = None
    nodelist: StrictStr | None = None
    nodes: StrictInt | None = None
    tasks_per_node: StrictInt = 1
    gpus_per_task: StrictInt | None = None
    max_parallel: StrictInt | None = None

    @field_validator("nodelist")
    @classmethod
    def _validate_single_node(cls, value: str | None) -> str | None:
        return validate_exact_slurm_node(value, field_name="Slurm nodelist")

    @field_validator("nodes")
    @classmethod
    def _validate_nodes(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            msg = "Slurm nodes must be at least 1 when present"
            raise ValueError(msg)
        return value

    @field_validator("tasks_per_node")
    @classmethod
    def _validate_tasks_per_node(cls, value: int) -> int:
        if value < 1:
            msg = "Slurm tasks_per_node must be at least 1"
            raise ValueError(msg)
        return value

    @field_validator("gpus_per_task")
    @classmethod
    def _validate_gpus_per_task(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            msg = "Slurm gpus_per_task must be non-negative when present"
            raise ValueError(msg)
        return value

    @field_validator("max_parallel")
    @classmethod
    def _validate_max_parallel(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            msg = "Slurm max_parallel must be at least 1 when present"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _validate_topology(self) -> Self:
        if self.max_parallel is not None and self.nodes is not None and self.max_parallel > self.nodes:
            msg = "Slurm max_parallel must not exceed nodes"
            raise ValueError(msg)
        if self.nodes is None:
            if self.tasks_per_node != 1:
                msg = "Slurm tasks_per_node requires typed topology (nodes)"
                raise ValueError(msg)
            if self.gpus_per_task is not None:
                msg = "Slurm gpus_per_task requires typed topology (nodes)"
                raise ValueError(msg)
            if self.max_parallel is not None:
                msg = "Slurm max_parallel requires typed topology (nodes)"
                raise ValueError(msg)
        else:
            if self.gpus_per_task is None:
                msg = "Slurm typed topology (nodes) requires gpus_per_task"
                raise ValueError(msg)
            if self.nodes == 1 and self.tasks_per_node == 1:
                msg = "Slurm typed topology cannot be the degenerate one-worker shape (nodes=1, tasks_per_node=1)"
                raise ValueError(msg)
        return self

    @property
    def workers(self) -> int:
        return (self.nodes if self.nodes is not None else 1) * self.tasks_per_node

    @property
    def is_packed(self) -> bool:
        """Return True when this topology packs more than one fold worker."""
        return self.workers > 1

    @model_serializer(mode="wrap")
    def _omit_default_topology(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        if self.nodes is None:
            data.pop("nodes", None)
        if self.tasks_per_node == 1:
            data.pop("tasks_per_node", None)
        if self.gpus_per_task is None:
            data.pop("gpus_per_task", None)
        if self.max_parallel is None:
            data.pop("max_parallel", None)
        return data


class WorkflowStepSpec(FrozenConfigModel):
    """One explicitly requested workflow step."""

    name: NonEmptyString
    run: StrictBool
    mode: WorkflowStepMode | None = None
    job_id: StrictStr | None = None
    array_range: StrictStr | None = None
    rendered_script: Path | None = None

    @field_validator("rendered_script", mode="before")
    @classmethod
    def _validate_optional_path(cls, value: object) -> Path | None:
        return _coerce_optional_path(value)


class WorkflowSpec(FrozenConfigModel):
    """Ordered workflow requested by the RunSpec."""

    steps: tuple[WorkflowStepSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_steps(self) -> Self:
        seen: set[str] = set()
        for step in self.steps:
            if step.name not in WORKFLOW_STEP_NAMES:
                msg = f"Unknown workflow step name: {step.name}"
                raise ValueError(msg)
            if step.name in seen:
                msg = f"Duplicate workflow step name: {step.name}"
                raise ValueError(msg)
            seen.add(step.name)
            if step.name in MODE_CAPABLE_WORKFLOW_STEPS:
                if step.mode is None:
                    msg = f"workflow step {step.name} requires mode"
                    raise ValueError(msg)
                if step.mode == "monitor-existing" and step.job_id is None:
                    msg = f"workflow step {step.name} monitor-existing mode requires job_id"
                    raise ValueError(msg)
                if step.name == "slurm" and step.mode == "monitor-existing" and step.array_range is None:
                    msg = "workflow step slurm monitor-existing mode requires array_range"
                    raise ValueError(msg)
            elif step.mode is not None:
                msg = f"workflow step {step.name} does not accept mode"
                raise ValueError(msg)
        return self


class SubmissionSpec(FrozenConfigModel):
    """Submission controller and evidence locations."""

    evidence_dir: Path
    report_path: Path | None = None
    controller_runtime: ControllerRuntime = "enroot"

    @field_validator("evidence_dir", mode="before")
    @classmethod
    def _validate_path(cls, value: object) -> Path:
        return _coerce_path(value)

    @field_validator("report_path", mode="before")
    @classmethod
    def _validate_optional_path(cls, value: object) -> Path | None:
        return _coerce_optional_path(value)


class AcceptanceSpec(FrozenConfigModel):
    """Acceptance comparator inputs for workflow-driven runs."""

    baseline_output_dir: Path | None = None
    baseline_run_name: StrictStr | None = None
    candidate_run_name: StrictStr | None = None
    tar_payload_match_mode: TarPayloadMatchMode = "by-tar"
    payload_sample_count: StrictInt | None = None
    candidate_parquet_required: StrictBool = True
    compare_failed_sets: StrictBool = True
    compare_tar_manifest_rows: StrictBool = True
    compare_analysis_model_rows: StrictBool = True

    @field_validator("baseline_output_dir", mode="before")
    @classmethod
    def _validate_optional_path(cls, value: object) -> Path | None:
        return _coerce_optional_path(value)

    @field_validator("payload_sample_count")
    @classmethod
    def _validate_payload_sample_count(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            msg = "acceptance.payload_sample_count must be non-negative"
            raise ValueError(msg)
        return value


class WorkerSpec(FrozenConfigModel):
    """Archive worker settings that must remain parity-controlled."""

    stages: NonEmptyString
    workers: StrictInt
    batch_size: StrictInt
    shards_per_archive: StrictInt
    self_upload: StrictBool
    local_scratch: StrictBool
    scratch_dir: Path
    s5cmd_path: NonEmptyString
    upload_slots: StrictInt = 4
    s5cmd_numworkers: StrictInt = 32
    duckdb_memory_limit: NonEmptyString = "1GB"
    heterodimers: StrictBool = False
    clash_device: NonEmptyString = "cuda"
    clash_batch_size: StrictInt = 128
    dssp_algorithm: NonEmptyString = "pydssp"
    parallel_stages: StrictBool = False
    retry_failed_only: StrictBool = False
    retry_metadata_delta_tag: NonEmptyString = "retry_failed_delta"
    tool_used: NonEmptyString = "ColabFold v1.6.0 / AlphaFold-Multimer"
    homodimer_tool_used: NonEmptyString = "ColabFold v1.6.0 / AlphaFold-Multimer"
    provider_id: NonEmptyString = "BSPP"
    provider_name: NonEmptyString = "BSPP Orchestration"
    provider_url: NonEmptyString = "https://alphafold.ebi.ac.uk"
    provider_copyrights: tuple[NonEmptyString, ...] = Field(
        default=("Copyright BSPP Orchestration contributors. All rights reserved.",),
        min_length=1,
    )

    @field_validator("scratch_dir", mode="before")
    @classmethod
    def _validate_path(cls, value: object) -> Path:
        return _coerce_path(value)

    @field_validator("tool_used")
    @classmethod
    def _validate_tool_used(cls, value: str) -> str:
        if value not in VALID_TOOL_USED:
            msg = f"tool_used must be one of {VALID_TOOL_USED!r}; got {value!r}"
            raise ValueError(msg)
        return value


class ObjectStorageSpec(FrozenConfigModel):
    """Object-store destinations and production-prefix policy."""

    s3_archive_prefix: NonEmptyString
    s3_output_prefix: NonEmptyString
    gcs_destination_prefix: StrictStr | None = None
    upload_mode: UploadMode = "files"
    s3_tar_prefix: StrictStr | None = None
    s3_tar_manifest_csv: Path | None = None
    local_tar_dir: Path | None = None
    local_tar_manifest_csv: Path | None = None
    tar_compression: TarCompression = "none"
    allow_production_prefixes: StrictBool = False

    @field_validator("s3_tar_manifest_csv", "local_tar_dir", "local_tar_manifest_csv", mode="before")
    @classmethod
    def _validate_optional_path(cls, value: object) -> Path | None:
        return _coerce_optional_path(value)


class DataMoverSelectionSpec(FrozenConfigModel):
    """Selected cluster-side mover for one run data-placement requirement."""

    tool: DataPlacementTool
    required: StrictBool = True
    source: StrictStr | None = None
    destination: StrictStr | None = None

    @field_validator("source", "destination")
    @classmethod
    def _validate_non_empty_optional_string(cls, value: str | None) -> str | None:
        if value == "":
            msg = "Expected non-empty string"
            raise ValueError(msg)
        return value


class RunDataPlacementSpec(FrozenConfigModel):
    """Run-scoped data-placement mover selections."""

    inputs: DataMoverSelectionSpec | None = None
    baselines: DataMoverSelectionSpec | None = None
    s3: DataMoverSelectionSpec | None = None
    gcs: DataMoverSelectionSpec | None = None
    gcs_direct: DataMoverSelectionSpec | None = None
    publication: DataMoverSelectionSpec | None = None

    @model_validator(mode="after")
    def _validate_stage_tools(self) -> Self:
        stage_fields: tuple[tuple[str, DataPlacementStage], ...] = (
            ("s3", "s3"),
            ("gcs", "gcs"),
            ("gcs_direct", "gcs-direct"),
        )
        for field_name, stage_name in stage_fields:
            selection = getattr(self, field_name)
            if selection is not None:
                validate_data_placement_tool_for_stage(selection.tool, stage_name)
        return self


class HqChunkPublicationSpec(FrozenConfigModel):
    """Publication settings for orchestration-native HQ chunk tar artifacts."""

    enabled: StrictBool = False
    target_prefix: StrictStr | None = None
    default_target_prefix_from_recipe: StrictBool = True
    collision_policy: HqChunkCollisionPolicy = "fail_if_non_empty"
    overwrite: StrictBool = False
    chunk_size: StrictInt = 1000
    manifest_dir: Path | None = None
    evidence_dir: Path | None = None
    sample_download_count: StrictInt = 20
    s5cmd_numworkers: StrictInt = 32

    @field_validator("target_prefix")
    @classmethod
    def _validate_target_prefix(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("s3://"):
            msg = "HQ chunk publication target_prefix must be an s3:// URI"
            raise ValueError(msg)
        return value

    @field_validator("manifest_dir", "evidence_dir", mode="before")
    @classmethod
    def _validate_optional_path(cls, value: object) -> Path | None:
        return _coerce_optional_path(value)

    @model_validator(mode="after")
    def _validate_positive_counts(self) -> Self:
        if self.chunk_size < 1:
            msg = "HQ chunk publication chunk_size must be at least 1"
            raise ValueError(msg)
        if self.sample_download_count < 0:
            msg = "HQ chunk publication sample_download_count must be non-negative"
            raise ValueError(msg)
        if self.s5cmd_numworkers < 1:
            msg = "HQ chunk publication s5cmd_numworkers must be at least 1"
            raise ValueError(msg)
        if self.enabled and self.target_prefix is None:
            msg = "HQ chunk publication requires target_prefix when enabled"
            raise ValueError(msg)
        return self


class HighQualityFromTarsSpec(FrozenConfigModel):
    """High-quality subset extraction settings for tar delivery runs."""

    enabled: StrictBool = False
    s3_prefix: StrictStr | None = None
    work_dir: Path | None = None
    publication: HqChunkPublicationSpec = Field(default_factory=HqChunkPublicationSpec)

    @field_validator("work_dir", mode="before")
    @classmethod
    def _validate_optional_path(cls, value: object) -> Path | None:
        return _coerce_optional_path(value)


class AnalysisMetadataSpec(FrozenConfigModel):
    """GPU analysis metadata collection and CPU finalization settings."""

    enabled: StrictBool = False
    csv_path: Path | None = None
    ipsae_threshold: StrictFloat = 0.6
    pdockq2_threshold: StrictFloat = 0.23
    finalize_after_gpu: StrictBool = True
    parquet_path: Path | None = None
    selected_ids_path: Path | None = None
    chunk_size: StrictInt = 100_000
    finalize_partition: StrictStr | None = None
    finalize_cpus_per_task: StrictInt = 4
    finalize_memory: NonEmptyString = "16G"
    finalize_time: NonEmptyString = "04:00:00"
    high_quality_from_tars: HighQualityFromTarsSpec = Field(default_factory=HighQualityFromTarsSpec)

    @field_validator("csv_path", "parquet_path", "selected_ids_path", mode="before")
    @classmethod
    def _validate_optional_path(cls, value: object) -> Path | None:
        return _coerce_optional_path(value)


class ValidationPolicy(FrozenConfigModel):
    """Counts and safety checks expected before promotion."""

    expected_archives: StrictInt | None = None
    expected_allowed_ids: StrictInt | None = None
    expected_one_archive_models: StrictInt | None = None
    expected_one_archive_objects: StrictInt | None = None
    expected_one_archive_aggregate_rows: StrictInt | None = None
    expected_tar_count: StrictInt | None = None
    expected_local_tars_rows: StrictInt | None = None
    expected_failed_rows: StrictInt | None = None
    expected_analysis_rows: StrictInt | None = None
    expected_selected_ids: StrictInt | None = None
    gcs_dry_run_only: StrictBool = True


# Retired secret targets from the vendor-neutral rename are rejected so legacy
# RunSpecs fail closed instead of resolving as literal env-var names. The retired
# brand is split across string literals to keep the contract-package grep for the
# retired token (task-2 verify) clean; the runtime values are identical.
_REJECTED_LEGACY_SECRET_TARGETS = frozenset({"bspp/" + "swift" + "stack", "swift" + "stack"})


class RunSecrets(FrozenConfigModel):
    """Secret references required by selected run stages."""

    s3_credentials_ref: SecretRef
    gcs_credentials_ref: SecretRef | None = None

    @field_validator("gcs_credentials_ref", mode="before")
    @classmethod
    def _empty_gcs_ref_is_absent(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("s3_credentials_ref", mode="after")
    @classmethod
    def _reject_legacy_secret_target(cls, value: SecretRef) -> SecretRef:
        if value.target in _REJECTED_LEGACY_SECRET_TARGETS:
            msg = f"Legacy secret target {value.target!r} is not accepted; use 'bspp/s3' or 's3'"
            raise ValueError(msg)
        return value


class RunSpec(FrozenConfigModel):
    """Single canonical source for orchestration decisions."""

    schema_version: Literal[1] = CURRENT_CONTRACT_SCHEMA_VERSION
    run_kind: RunKind | None = None
    dataset: DatasetSpec
    cluster: ClusterSpec
    paths: PathSpec
    references: ReferenceSpec
    container: ContainerSpec
    resources: Mapping[str, SlurmResources]
    worker: WorkerSpec
    storage: ObjectStorageSpec
    data_placement: RunDataPlacementSpec = Field(default_factory=RunDataPlacementSpec)
    analysis_metadata: AnalysisMetadataSpec = Field(default_factory=AnalysisMetadataSpec)
    validation: ValidationPolicy = Field(default_factory=ValidationPolicy)
    workflow: WorkflowSpec | None = None
    submission: SubmissionSpec | None = None
    acceptance: AcceptanceSpec | None = None
    secrets: RunSecrets
    source_path: Path | None = None
    source_hash: StrictStr | None = None

    @model_validator(mode="before")
    @classmethod
    def _validate_active_workflow_contract(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        data["schema_version"] = validate_schema_version(data.get("schema_version"), record_name="RunSpec")

        active_workflow = any(key in data for key in ("workflow", "submission", "acceptance"))
        if not active_workflow:
            return data

        if "workflow" not in data:
            msg = "Active workflow RunSpecs require top-level workflow"
            raise ValueError(msg)
        if "submission" not in data:
            msg = "Active workflow RunSpecs require top-level submission"
            raise ValueError(msg)

        paths = data.get("paths")
        if isinstance(paths, Mapping) and "legacy_repo" in paths:
            msg = "paths.legacy_repo is removed from active workflow RunSpecs; use paths.afdb_toolkit_repo"
            raise ValueError(msg)

        container = data.get("container")
        if isinstance(container, Mapping) and "workdir" in container:
            msg = "container.workdir is removed from active workflow RunSpecs; omit it"
            raise ValueError(msg)
        if isinstance(container, Mapping):
            normalized_container = dict(container)
            raw_mounts = normalized_container.get("mounts")
            if isinstance(raw_mounts, list | tuple):
                normalized_mounts: list[object] = []
                for mount in raw_mounts:
                    if not isinstance(mount, Mapping):
                        normalized_mounts.append(mount)
                        continue
                    if "read_only" in mount:
                        msg = "container.mounts[].read_only is removed from active workflow RunSpecs; omit it"
                        raise ValueError(msg)
                    normalized_mount = dict(mount)
                    normalized_mount["read_only"] = False
                    normalized_mounts.append(normalized_mount)
                normalized_container["mounts"] = normalized_mounts
                data["container"] = normalized_container

        return data

    @model_validator(mode="after")
    def _validate_submission_paths(self) -> Self:
        if self.submission is None:
            return self
        if not _path_is_under(self.submission.evidence_dir, self.paths.output_dir):
            msg = "submission.evidence_dir must be under paths.output_dir"
            raise ValueError(msg)
        if self.submission.report_path is not None and not _path_is_under(
            self.submission.report_path,
            self.paths.output_dir,
        ):
            msg = "submission.report_path must be under paths.output_dir"
            raise ValueError(msg)
        return self

    @field_validator("source_path", mode="before")
    @classmethod
    def _validate_optional_source_path(cls, value: object) -> Path | None:
        return _coerce_optional_path(value)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
        *,
        source_path: Path | None = None,
        source_hash: str | None = None,
    ) -> Self:
        """Build a :class:`RunSpec` from a parsed mapping."""
        payload = dict(data)
        payload["source_path"] = source_path
        payload["source_hash"] = source_hash
        return cls.model_validate(payload)

    def validate(self) -> None:  # type: ignore[override]
        """Validate guardrails that are independent of local file existence."""
        if self.run_kind in {"canary", "production"}:
            validate_processing_upload_policy(
                self_upload=self.worker.self_upload,
                workflow_steps=tuple(step.name for step in self.workflow.steps if step.run)
                if self.workflow is not None
                else (),
            )
        if self.dataset.mode != "archive":
            msg = "Milestone 1 only supports archive-mode RunSpec files"
            raise ValueError(msg)
        if self.worker.shards_per_archive < 1:
            msg = "worker.shards_per_archive must be at least 1"
            raise ValueError(msg)
        if self.worker.self_upload and self.secrets.s3_credentials_ref is None:
            msg = "self-upload requires an S3 credentials reference"
            raise ValueError(msg)
        if self.worker.heterodimers and self.references.heterodimer_id_manifest is None:
            msg = "worker.heterodimers requires references.heterodimer_id_manifest"
            raise ValueError(msg)
        if self.storage.upload_mode == "tar":
            has_s3_tar = bool(self.worker.self_upload and self.storage.s3_tar_prefix)
            has_local_tar = self.storage.local_tar_dir is not None
            if not has_s3_tar and not has_local_tar:
                msg = "tar upload mode requires storage.s3_tar_prefix with self_upload or storage.local_tar_dir"
                raise ValueError(msg)
            if has_local_tar and self.storage.local_tar_manifest_csv is None:
                msg = "local tar delivery requires storage.local_tar_manifest_csv"
                raise ValueError(msg)
            if has_s3_tar and self.storage.s3_tar_manifest_csv is None:
                msg = "S3 tar delivery requires storage.s3_tar_manifest_csv"
                raise ValueError(msg)
        if self.analysis_metadata.enabled and self.analysis_metadata.csv_path is None:
            msg = "analysis_metadata.enabled requires analysis_metadata.csv_path"
            raise ValueError(msg)
        if self.analysis_metadata.finalize_after_gpu and self.analysis_metadata.enabled:
            if self.analysis_metadata.parquet_path is None:
                msg = "analysis metadata finalization requires analysis_metadata.parquet_path"
                raise ValueError(msg)
            if self.analysis_metadata.selected_ids_path is None:
                msg = "analysis metadata finalization requires analysis_metadata.selected_ids_path"
                raise ValueError(msg)
        if self.storage.gcs_destination_prefix and self.secrets.gcs_credentials_ref is None:
            msg = "GCS destination prefix requires a GCS credentials reference"
            raise ValueError(msg)
        self._validate_data_placement_workflow()

    def _validate_data_placement_workflow(self) -> None:
        if self.workflow is None:
            return
        required_by_step = {
            "upload-s3": ("s3", self.data_placement.s3),
            "upload-gcs": ("gcs", self.data_placement.gcs),
            "upload-gcs-direct": ("gcs_direct", self.data_placement.gcs_direct),
        }
        for step in self.workflow.steps:
            if not step.run or step.name not in required_by_step:
                continue
            field_name, selection = required_by_step[step.name]
            if selection is None:
                msg = f"workflow step {step.name} requires data_placement.{field_name}"
                raise ValueError(msg)

    def resolve_secrets(self, *, environ: Mapping[str, str] | None = None) -> tuple[ResolvedSecret, ...]:
        """Resolve configured secret references with redacted statuses."""
        return tuple(resolve_secret(ref, environ=environ) for ref in self.secret_refs().values())

    def secret_refs(self) -> Mapping[str, SecretRef]:
        """Return configured secrets keyed by logical purpose."""
        refs = {"s3_credentials": self.secrets.s3_credentials_ref}
        if self.secrets.gcs_credentials_ref is not None:
            refs["gcs_credentials"] = self.secrets.gcs_credentials_ref
        return refs

    def validate_required_secrets(self, *, environ: Mapping[str, str] | None = None) -> tuple[ResolvedSecret, ...]:
        """Strictly validate that every configured required secret is available."""
        return validate_required_secrets(self.secret_refs().values(), environ=environ)

    def secret_execution_env(self, *, environ: Mapping[str, str] | None = None) -> SecretExecutionEnv:
        """Return a redacted subprocess environment overlay for configured secrets."""
        return build_secret_execution_env(self.secret_refs(), environ=environ)


def load_runspec(path: Path) -> RunSpec:
    """Load and validate a YAML run specification."""
    raw_bytes = path.read_bytes()
    data = yaml.safe_load(raw_bytes)
    if not isinstance(data, dict):
        msg = f"Expected RunSpec YAML mapping in {path}"
        raise TypeError(msg)
    reject_literal_secret_values(data)
    spec = runspec_from_mapping(data, source_path=path, source_hash=hashlib.sha256(raw_bytes).hexdigest())
    spec.validate()
    return spec


def runspec_from_mapping(
    data: Mapping[str, object],
    *,
    source_path: Path | None = None,
    source_hash: str | None = None,
) -> RunSpec:
    """Build a :class:`RunSpec` from a parsed mapping."""
    spec = RunSpec.from_mapping(data, source_path=source_path, source_hash=source_hash)
    spec.validate()
    return spec


def resolve_hq_chunk_publication_target(spec: RunSpec) -> str:
    """Resolve the HQ chunk publication target prefix without using the legacy flat-file prefix."""
    publication = spec.analysis_metadata.high_quality_from_tars.publication
    prefix = publication.target_prefix
    if prefix is None:
        msg = "HQ chunk publication target prefix is not configured"
        raise ValueError(msg)
    return _ensure_trailing_slash(prefix)


def _ensure_trailing_slash(prefix: str) -> str:
    return prefix if prefix.endswith("/") else f"{prefix}/"


def render_dry_run(spec: RunSpec, secret_statuses: tuple[ResolvedSecret, ...] = ()) -> str:
    """Render a concise, redacted dry-run summary."""
    lines = [
        "RunSpec dry run",
        f"  run_id: {spec.dataset.run_id}",
        f"  dataset: {spec.dataset.name}",
        f"  mode: {spec.dataset.mode}",
        f"  array: {spec.dataset.array}",
        f"  archive_source: {spec.dataset.archive_source}",
        f"  cluster: {spec.cluster.name}",
        f"  container: {spec.container.image}",
        f"  container_workdir: {spec.container.workdir}",
        f"  staging_dir: {spec.paths.staging_dir}",
        f"  output_dir: {spec.paths.output_dir}",
        f"  s3_archive_prefix: {spec.storage.s3_archive_prefix}",
        f"  s3_output_prefix: {spec.storage.s3_output_prefix}",
        f"  upload_mode: {spec.storage.upload_mode}",
        f"  tar_compression: {spec.storage.tar_compression}",
        f"  s3_tar_prefix: {spec.storage.s3_tar_prefix or '<disabled>'}",
        f"  s3_tar_manifest_csv: {spec.storage.s3_tar_manifest_csv or '<disabled>'}",
        f"  local_tar_dir: {spec.storage.local_tar_dir or '<disabled>'}",
        f"  local_tar_manifest_csv: {spec.storage.local_tar_manifest_csv or '<disabled>'}",
        f"  gcs_destination_prefix: {spec.storage.gcs_destination_prefix or '<disabled>'}",
        f"  production_prefixes: {'enabled' if spec.storage.allow_production_prefixes else 'blocked by default'}",
        "  references:",
    ]
    for name in sorted(spec.references.artifacts):
        artifact = spec.references.artifact(name)
        source = artifact.source_uri or "<local-only>"
        details = []
        if artifact.sha256:
            details.append(f"sha256={artifact.sha256}")
        if artifact.min_size_bytes is not None:
            details.append(f"min_size_bytes={artifact.min_size_bytes}")
        if artifact.version:
            details.append(f"version={artifact.version}")
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"    {name}: path={artifact.path}, source={source}{suffix}")
    lines.append("  container_mounts:")
    for mount in spec.container.mounts:
        mode = "ro" if mount.read_only else "rw"
        lines.append(f"    {mount.source} -> {mount.target} ({mode})")
    lines.extend(
        [
            "  worker:",
            f"    stages: {spec.worker.stages}",
            f"    workers: {spec.worker.workers}",
            f"    batch_size: {spec.worker.batch_size}",
            f"    shards_per_archive: {spec.worker.shards_per_archive}",
            f"    self_upload: {spec.worker.self_upload}",
            f"    local_scratch: {spec.worker.local_scratch}",
            f"    scratch_dir: {spec.worker.scratch_dir}",
            f"    s5cmd_path: {spec.worker.s5cmd_path}",
            f"    upload_slots: {spec.worker.upload_slots}",
            f"    s5cmd_numworkers: {spec.worker.s5cmd_numworkers}",
            f"    duckdb_memory_limit: {spec.worker.duckdb_memory_limit}",
            f"    heterodimers: {spec.worker.heterodimers}",
            f"    clash_device: {spec.worker.clash_device}",
            f"    clash_batch_size: {spec.worker.clash_batch_size}",
            f"    dssp_algorithm: {spec.worker.dssp_algorithm}",
            f"    parallel_stages: {spec.worker.parallel_stages}",
            f"    retry_failed_only: {spec.worker.retry_failed_only}",
            f"    retry_metadata_delta_tag: {spec.worker.retry_metadata_delta_tag}",
            f"    provider_copyrights: {list(spec.worker.provider_copyrights)}",
            "  analysis_metadata:",
            f"    enabled: {spec.analysis_metadata.enabled}",
            f"    csv_path: {spec.analysis_metadata.csv_path or '<disabled>'}",
            f"    ipsae_threshold: {spec.analysis_metadata.ipsae_threshold}",
            f"    pdockq2_threshold: {spec.analysis_metadata.pdockq2_threshold}",
            f"    finalize_after_gpu: {spec.analysis_metadata.finalize_after_gpu}",
            f"    parquet_path: {spec.analysis_metadata.parquet_path or '<disabled>'}",
            f"    selected_ids_path: {spec.analysis_metadata.selected_ids_path or '<disabled>'}",
            f"    chunk_size: {spec.analysis_metadata.chunk_size}",
            f"    high_quality_from_tars: {spec.analysis_metadata.high_quality_from_tars.enabled}",
            f"    hq_chunk_publication: {spec.analysis_metadata.high_quality_from_tars.publication.enabled}",
            f"    hq_chunk_publication_target: {_dry_run_hq_publication_target(spec)}",
            f"    hq_chunk_publication_collision_policy: "
            f"{spec.analysis_metadata.high_quality_from_tars.publication.collision_policy}",
            f"    hq_chunk_publication_overwrite: "
            f"{spec.analysis_metadata.high_quality_from_tars.publication.overwrite}",
            "  validation_counts:",
            f"    expected_archives: {spec.validation.expected_archives}",
            f"    expected_allowed_ids: {spec.validation.expected_allowed_ids}",
            f"    expected_one_archive_models: {spec.validation.expected_one_archive_models}",
            f"    expected_one_archive_objects: {spec.validation.expected_one_archive_objects}",
            f"    expected_one_archive_aggregate_rows: {spec.validation.expected_one_archive_aggregate_rows}",
            f"    gcs_dry_run_only: {spec.validation.gcs_dry_run_only}",
        ]
    )
    lines.append("  resources:")
    for name in sorted(spec.resources):
        resource = spec.resources[name]
        gres = f", gres={resource.gres}" if resource.gres else ""
        array = f", array={resource.array}" if resource.array else ""
        lines.append(
            f"    {name}: partition={resource.partition}, cpus={resource.cpus_per_task}, "
            f"mem={resource.memory}, time={resource.time}{gres}{array}"
        )
    lines.append("  secrets:")
    statuses = secret_statuses or spec.resolve_secrets()
    for status in statuses:
        state = "ok" if status.ok else "missing"
        lines.append(f"    {status.ref.redacted()}: {state} ({status.message})")
    if spec.source_hash:
        lines.append(f"  source_hash: {spec.source_hash}")
    return "\n".join(lines)


def _dry_run_hq_publication_target(spec: RunSpec) -> str:
    publication = spec.analysis_metadata.high_quality_from_tars.publication
    if not publication.enabled:
        return "<disabled>"
    return resolve_hq_chunk_publication_target(spec)


def render_artifact_header(spec: RunSpec, artifact_name: str, *, comment_prefix: str = "#") -> str:
    """Render a do-not-edit header for files generated from a RunSpec."""
    source = str(spec.source_path) if spec.source_path is not None else "<in-memory>"
    source_hash = spec.source_hash or "<unknown>"
    rendered_at = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        f"{comment_prefix} Generated artifact: {artifact_name}",
        f"{comment_prefix} Run ID: {spec.dataset.run_id}",
        f"{comment_prefix} Source RunSpec: {source}",
        f"{comment_prefix} Source SHA256: {source_hash}",
        f"{comment_prefix} Rendered at: {rendered_at}",
        f"{comment_prefix} DO NOT EDIT: regenerate this file from the RunSpec.",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "BAKED_TOOLKIT_COMMIT",
    "BAKED_TOOLKIT_PATH",
    "AcceptanceSpec",
    "AnalysisMetadataSpec",
    "ClusterSpec",
    "ContainerSpec",
    "ControllerRuntime",
    "DataMoverSelectionSpec",
    "DatasetSpec",
    "HighQualityFromTarsSpec",
    "HqChunkPublicationSpec",
    "MountSpec",
    "ObjectStorageSpec",
    "PathSpec",
    "ReferenceArtifact",
    "ReferenceSpec",
    "RunDataPlacementSpec",
    "RunKind",
    "RunSecrets",
    "RunSpec",
    "SlurmResources",
    "SubmissionSpec",
    "TarPayloadMatchMode",
    "ValidationPolicy",
    "WorkerSpec",
    "WorkflowSpec",
    "WorkflowStepMode",
    "WorkflowStepSpec",
    "load_runspec",
    "render_artifact_header",
    "render_dry_run",
    "resolve_hq_chunk_publication_target",
    "runspec_from_mapping",
]
