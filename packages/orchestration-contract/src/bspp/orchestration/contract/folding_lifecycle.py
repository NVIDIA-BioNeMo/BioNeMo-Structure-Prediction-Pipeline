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

"""Typed configuration and records for the folding phase lifecycle.

Port Baseline: ``3864d0eda67e70979b8e48f00ed6a08f9e71c59e`` from
``folding/openfold-pipeline/docs/openfoldctl.md:1-582``,
``folding/openfold-pipeline/scripts/openfoldctl.py:392-552``, and
``folding/openfold-pipeline/recipes/validation_20260223_heterodimers_smoke_256_3node_batch/config.yaml:1-94``.

Only :class:`FoldingLifecycleConfig` is user-authored configuration. Lifecycle
evidence, requests, and decisions are immutable dataclasses below it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Self

from pydantic import StrictBool, StrictInt, StrictStr, field_validator, model_validator

from bspp.orchestration.contract.config_models import FrozenConfigModel
from bspp.orchestration.contract.folding_archive import (
    ArchiveManifestRecord,
    FoldingArchivePlan,
    FoldingResultInventory,
    archive_manifest_record_from_mapping,
    archive_plan_from_mapping,
    folding_result_inventory_from_mapping,
)
from bspp.orchestration.contract.folding_checkpoint import (
    FoldingCheckpointState,
    FoldingRemainingWorkPlan,
    folding_checkpoint_state_from_mapping,
    folding_remaining_work_plan_from_mapping,
)
from bspp.orchestration.contract.folding_index import FoldingIndex, folding_index_from_mapping
from bspp.orchestration.contract.folding_queue import (
    FoldingQueueArtifact,
    FoldingQueueLayout,
    FoldingQueuePlan,
    FoldingQueueStrategy,
    folding_queue_artifact_from_mapping,
    folding_queue_plan_from_mapping,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_INDEX_REQUEST_FIELDS = frozenset({"schema_version", "config"})
_INDEX_DECISION_FIELDS = frozenset({"schema_version", "index", "index_worker_count", "sort_by_length"})
_QUEUE_WRITE_FIELDS = frozenset({"schema_version", "path", "artifact"})
_SUBMIT_REQUEST_FIELDS = frozenset({"schema_version", "config", "index"})
_SUBMIT_DECISION_FIELDS = frozenset(
    {"schema_version", "queue_plan", "queue_writes", "submission_planned", "execution_authorized"}
)
_CHECKPOINT_INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "node_completion_shards",
        "node_failure_shards",
        "current_gpu_completion_shards",
        "current_gpu_failure_shards",
        "historical_gpu_completion_shards",
        "historical_gpu_failure_shards",
        "merged_completion_view",
        "merged_failure_view",
    }
)
_MERGE_REQUEST_FIELDS = frozenset({"schema_version", "config", "checkpoint_inputs"})
_MERGE_DECISION_FIELDS = frozenset(
    {"schema_version", "state", "completed_view_path", "failed_view_path", "write_planned"}
)
_RESUME_REQUEST_FIELDS = frozenset(
    {"schema_version", "config", "index", "checkpoint_state", "result_inventory", "retry_failed"}
)
_RESUME_DECISION_FIELDS = frozenset({"schema_version", "remaining_work", "recovered_result_ids", "submit_decision"})
_ARCHIVE_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "config",
        "index",
        "result_inventory",
        "scan_predictions_root",
        "selected_protein_ids",
        "prior_manifest",
        "force",
    }
)
_ARCHIVE_DECISION_FIELDS = frozenset(
    {"schema_version", "result_inventory", "archive_plan", "manifest_jsonl", "execution_authorized"}
)
PreflightAction = Literal["index", "submit", "resume", "merge", "archive"]
PreflightName = Literal["config", "index", "checkpoint", "result", "archive"]
PreflightStatus = Literal["pass", "warning", "fail", "not-required"]
FoldingLifecycleState = Literal["initial", "partial", "complete", "invalid"]
FoldingStatusEvidenceKind = Literal["manifest", "checkpoint", "result", "archive"]
_PREFLIGHT_CHECK_FIELDS = frozenset({"schema_version", "name", "status", "detail"})
_PREFLIGHT_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "action",
        "config",
        "index",
        "checkpoint_inputs",
        "checkpoint_state",
        "result_inventory",
        "scan_predictions_root",
    }
)
_PREFLIGHT_DECISION_FIELDS = frozenset({"schema_version", "action", "checks", "can_plan"})
_MANIFEST_FIELDS = frozenset({"schema_version", "total_proteins", "queue_artifact_count", "queue_layout"})
_STATUS_REQUEST_FIELDS = frozenset(
    {"schema_version", "config", "index", "manifest", "checkpoint_state", "result_inventory", "archive_plan"}
)
_STATUS_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "evidence_kinds",
        "planned_model_count",
        "manifest_model_count",
        "checkpoint_completed_count",
        "checkpoint_failed_count",
        "recovered_result_count",
        "completed_model_count",
        "incomplete_result_count",
        "planned_archive_count",
        "invalid_evidence",
    }
)


class FoldingLifecycleConfig(FrozenConfigModel):
    """User-authored folding paths, model choices, resources, and archive intent."""

    schema_version: Literal[1] = CURRENT_CONTRACT_SCHEMA_VERSION
    run_name: StrictStr
    declared_a3m_inputs: tuple[StrictStr, ...]
    batch_info_path: StrictStr
    output_dir: StrictStr
    work_dir: StrictStr
    predictions_root: StrictStr
    checkpoint_dir: StrictStr
    container_image: StrictStr
    trt_bionemo_dir: StrictStr
    trt_engines_dir: StrictStr
    trt_checkpoint_path: StrictStr
    model_preset: StrictStr
    max_recycling_iters: StrictInt = 4
    trt_fallback_threshold: StrictInt = 1536
    preprocessing_enabled: StrictBool = True
    sort_by_length: StrictBool = True
    index_worker_count: StrictInt
    queue_strategy: FoldingQueueStrategy
    queue_layout: FoldingQueueLayout
    node_count: StrictInt
    gpus_per_node: StrictInt
    worker_count: StrictInt
    cpus_per_task: StrictInt
    max_proteins_per_batch: StrictInt = 200
    job_name: StrictStr
    account: StrictStr
    partition: StrictStr
    walltime: StrictStr
    archive_enabled: StrictBool = False
    archive_run_tag: StrictStr | None = None
    archive_stage_root: StrictStr | None = None
    archive_root: StrictStr | None = None
    proteins_per_archive: StrictInt = 5_000
    archive_start_index: StrictInt = 0
    archive_max_archives: StrictInt | None = None
    archive_shuffle: StrictBool = True
    archive_shuffle_seed: StrictInt = 42
    lz4_executable: StrictStr = "lz4"

    @model_validator(mode="before")
    @classmethod
    def _validate_schema(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        data["schema_version"] = validate_schema_version(
            data.get("schema_version"), record_name="FoldingLifecycleConfig"
        )
        return data

    @field_validator(
        "run_name",
        "batch_info_path",
        "output_dir",
        "work_dir",
        "predictions_root",
        "checkpoint_dir",
        "container_image",
        "trt_bionemo_dir",
        "trt_engines_dir",
        "trt_checkpoint_path",
        "model_preset",
        "job_name",
        "account",
        "partition",
        "walltime",
        "lz4_executable",
    )
    @classmethod
    def _nonempty_text(cls, value: str) -> str:
        if not value.strip():
            msg = "folding configuration strings must be non-empty"
            raise ValueError(msg)
        return value

    @field_validator("declared_a3m_inputs")
    @classmethod
    def _declared_inputs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item.strip() for item in value):
            msg = "declared_a3m_inputs must contain non-empty paths"
            raise ValueError(msg)
        if len(value) != len(set(value)):
            msg = "declared_a3m_inputs must be unique"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _coherent_resources_and_archive(self) -> Self:
        for field_name in (
            "max_recycling_iters",
            "trt_fallback_threshold",
            "index_worker_count",
            "node_count",
            "gpus_per_node",
            "worker_count",
            "cpus_per_task",
            "max_proteins_per_batch",
            "proteins_per_archive",
        ):
            if getattr(self, field_name) <= 0:
                msg = f"{field_name} must be positive"
                raise ValueError(msg)
        if self.archive_start_index < 0 or self.archive_shuffle_seed < 0:
            msg = "archive_start_index and archive_shuffle_seed must be non-negative"
            raise ValueError(msg)
        if self.archive_max_archives is not None and self.archive_max_archives < 0:
            msg = "archive_max_archives must be non-negative or null"
            raise ValueError(msg)
        if not re.fullmatch(r"\d{2,3}:\d{2}:\d{2}", self.walltime):
            msg = "walltime must use HH:MM:SS"
            raise ValueError(msg)

        expected_workers = self.node_count if self.queue_layout == "per_node" else self.node_count * self.gpus_per_node
        if self.worker_count != expected_workers:
            relationship = "node_count" if self.queue_layout == "per_node" else "node_count * gpus_per_node"
            msg = f"worker_count must equal {relationship} for {self.queue_layout} layout"
            raise ValueError(msg)

        archive_values = (self.archive_run_tag, self.archive_stage_root, self.archive_root)
        if self.archive_enabled and any(value is None or not value.strip() for value in archive_values):
            msg = "archive_run_tag, archive_stage_root, and archive_root are required when archive is enabled"
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class FoldingIndexLifecycleRequest:
    """Request to interpret explicitly declared local A3Ms as an index."""

    config: FoldingLifecycleConfig
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingIndexLifecycleRequest")
        if not isinstance(self.config, FoldingLifecycleConfig):
            msg = "config must be a FoldingLifecycleConfig"
            raise ValueError(msg)
        _matching_version(self.schema_version, self.config.schema_version, "config")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class FoldingIndexLifecycleDecision:
    """Indexed A3M evidence plus the baseline controller's local worker intent."""

    index: FoldingIndex
    index_worker_count: int
    sort_by_length: bool
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingIndexLifecycleDecision")
        if not isinstance(self.index, FoldingIndex):
            msg = "index must be a FoldingIndex"
            raise ValueError(msg)
        _positive_int(self.index_worker_count, "index_worker_count")
        if not isinstance(self.sort_by_length, bool):
            msg = "sort_by_length must be a boolean"
            raise ValueError(msg)
        _matching_version(self.schema_version, self.index.schema_version, "index")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "index": self.index.to_mapping(),
            "index_worker_count": self.index_worker_count,
            "sort_by_length": self.sort_by_length,
        }


@dataclass(frozen=True)
class FoldingQueueWriteDeclaration:
    """One planned local queue-view write; no filesystem action is performed."""

    path: str
    artifact: FoldingQueueArtifact
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingQueueWriteDeclaration")
        _nonempty_str(self.path, "path")
        if not isinstance(self.artifact, FoldingQueueArtifact):
            msg = "artifact must be a FoldingQueueArtifact"
            raise ValueError(msg)
        _matching_version(self.schema_version, self.artifact.schema_version, "artifact")
        if self.path.rstrip("/").rsplit("/", maxsplit=1)[-1] != self.artifact.file_name:
            msg = "queue write path must end with the rendered artifact file_name"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "artifact": self.artifact.to_mapping(),
        }


@dataclass(frozen=True)
class FoldingSubmitLifecycleRequest:
    """Request to produce queue plans and local write declarations only."""

    config: FoldingLifecycleConfig
    index: FoldingIndex
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingSubmitLifecycleRequest")
        if not isinstance(self.config, FoldingLifecycleConfig) or not isinstance(self.index, FoldingIndex):
            msg = "submit request requires FoldingLifecycleConfig and FoldingIndex values"
            raise ValueError(msg)
        _matching_version(self.schema_version, self.config.schema_version, "config")
        _matching_version(self.schema_version, self.index.schema_version, "index")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.model_dump(mode="json"),
            "index": self.index.to_mapping(),
        }


@dataclass(frozen=True)
class FoldingSubmitLifecycleDecision:
    """Deterministic submit preparation that explicitly cannot execute."""

    queue_plan: FoldingQueuePlan
    queue_writes: tuple[FoldingQueueWriteDeclaration, ...]
    submission_planned: bool
    execution_authorized: Literal[False] = False
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingSubmitLifecycleDecision")
        if not isinstance(self.queue_plan, FoldingQueuePlan):
            msg = "queue_plan must be a FoldingQueuePlan"
            raise ValueError(msg)
        _typed_tuple(self.queue_writes, FoldingQueueWriteDeclaration, "queue_writes")
        if not isinstance(self.submission_planned, bool) or self.submission_planned != bool(
            self.queue_plan.assignments
        ):
            msg = "submission_planned must reflect whether queue assignments exist"
            raise ValueError(msg)
        if self.execution_authorized is not False:
            msg = "folding lifecycle decisions never authorize execution"
            raise ValueError(msg)
        _matching_version(self.schema_version, self.queue_plan.schema_version, "queue_plan")
        if any(item.schema_version != self.schema_version for item in self.queue_writes):
            msg = "queue write schema versions must match their submit decision"
            raise ValueError(msg)
        artifacts = tuple(item.artifact for item in self.queue_writes)
        worker_ids = tuple(artifact.worker_id for artifact in artifacts)
        expected_workers = tuple(sorted({assignment.worker_id for assignment in self.queue_plan.assignments}))
        if worker_ids != expected_workers:
            msg = "queue_writes must cover each nonempty queue worker once in sorted order"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "queue_plan": self.queue_plan.to_mapping(),
            "queue_writes": [item.to_mapping() for item in self.queue_writes],
            "submission_planned": self.submission_planned,
            "execution_authorized": self.execution_authorized,
        }


@dataclass(frozen=True)
class FoldingCheckpointInputs:
    """Explicit checkpoint slots with a highest-precedence merged-view slot.

    Current node rows and current-schema GPU rows share the five-column parser
    but remain distinct lifecycle evidence. Historical GPU rows use the
    retained positional three-column schema. Supplied merged views are always
    interpreted after all shard classes, never through accidental path order.
    """

    node_completion_shards: tuple[str, ...]
    node_failure_shards: tuple[str, ...]
    current_gpu_completion_shards: tuple[str, ...]
    current_gpu_failure_shards: tuple[str, ...]
    historical_gpu_completion_shards: tuple[str, ...]
    historical_gpu_failure_shards: tuple[str, ...]
    merged_completion_view: str | None
    merged_failure_view: str | None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingCheckpointInputs")
        for field_name in (
            "node_completion_shards",
            "node_failure_shards",
            "current_gpu_completion_shards",
            "current_gpu_failure_shards",
            "historical_gpu_completion_shards",
            "historical_gpu_failure_shards",
        ):
            _string_tuple(getattr(self, field_name), field_name)
        for field_name in ("merged_completion_view", "merged_failure_view"):
            value = getattr(self, field_name)
            if value is not None:
                _nonempty_str(value, field_name)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "node_completion_shards": list(self.node_completion_shards),
            "node_failure_shards": list(self.node_failure_shards),
            "current_gpu_completion_shards": list(self.current_gpu_completion_shards),
            "current_gpu_failure_shards": list(self.current_gpu_failure_shards),
            "historical_gpu_completion_shards": list(self.historical_gpu_completion_shards),
            "historical_gpu_failure_shards": list(self.historical_gpu_failure_shards),
            "merged_completion_view": self.merged_completion_view,
            "merged_failure_view": self.merged_failure_view,
        }


@dataclass(frozen=True)
class FoldingMergeLifecycleRequest:
    """Request to interpret checkpoint inputs and declare canonical outputs."""

    config: FoldingLifecycleConfig
    checkpoint_inputs: FoldingCheckpointInputs
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingMergeLifecycleRequest")
        if not isinstance(self.config, FoldingLifecycleConfig) or not isinstance(
            self.checkpoint_inputs, FoldingCheckpointInputs
        ):
            msg = "merge request requires FoldingLifecycleConfig and FoldingCheckpointInputs values"
            raise ValueError(msg)
        _matching_version(self.schema_version, self.config.schema_version, "config")
        _matching_version(self.schema_version, self.checkpoint_inputs.schema_version, "checkpoint_inputs")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.model_dump(mode="json"),
            "checkpoint_inputs": self.checkpoint_inputs.to_mapping(),
        }


@dataclass(frozen=True)
class FoldingMergeLifecycleDecision:
    """Canonical checkpoint state and non-executed aggregate write declaration."""

    state: FoldingCheckpointState
    completed_view_path: str
    failed_view_path: str
    write_planned: bool
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingMergeLifecycleDecision")
        if not isinstance(self.state, FoldingCheckpointState):
            msg = "state must be a FoldingCheckpointState"
            raise ValueError(msg)
        _nonempty_str(self.completed_view_path, "completed_view_path")
        _nonempty_str(self.failed_view_path, "failed_view_path")
        if self.completed_view_path == self.failed_view_path:
            msg = "merged completion and failure view paths must be distinct"
            raise ValueError(msg)
        if not isinstance(self.write_planned, bool):
            msg = "write_planned must be a boolean"
            raise ValueError(msg)
        _matching_version(self.schema_version, self.state.schema_version, "state")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "state": self.state.to_mapping(),
            "completed_view_path": self.completed_view_path,
            "failed_view_path": self.failed_view_path,
            "write_planned": self.write_planned,
        }


@dataclass(frozen=True)
class FoldingResumeLifecycleRequest:
    """Request to combine checkpoint state with already-associated result evidence."""

    config: FoldingLifecycleConfig
    index: FoldingIndex
    checkpoint_state: FoldingCheckpointState
    result_inventory: FoldingResultInventory
    retry_failed: bool
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingResumeLifecycleRequest")
        expected_types = (
            (self.config, FoldingLifecycleConfig, "config"),
            (self.index, FoldingIndex, "index"),
            (self.checkpoint_state, FoldingCheckpointState, "checkpoint_state"),
            (self.result_inventory, FoldingResultInventory, "result_inventory"),
        )
        for value, expected_type, field_name in expected_types:
            if not isinstance(value, expected_type):
                msg = f"{field_name} must be a {expected_type.__name__}"
                raise ValueError(msg)
            _matching_version(self.schema_version, value.schema_version, field_name)
        if not isinstance(self.retry_failed, bool):
            msg = "retry_failed must be a boolean"
            raise ValueError(msg)
        planned_ids = tuple(
            record.protein_id for record in sorted(self.index.records, key=lambda item: item.source_ordinal)
        )
        associated_ids = tuple(item.protein_id for item in self.result_inventory.associations)
        if associated_ids != planned_ids:
            msg = "result_inventory associations must exactly cover the folding index in source order"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.model_dump(mode="json"),
            "index": self.index.to_mapping(),
            "checkpoint_state": self.checkpoint_state.to_mapping(),
            "result_inventory": self.result_inventory.to_mapping(),
            "retry_failed": self.retry_failed,
        }


@dataclass(frozen=True)
class FoldingResumeLifecycleDecision:
    """Recovery-aware remaining work and an optional non-executing submit plan."""

    remaining_work: FoldingRemainingWorkPlan
    recovered_result_ids: tuple[str, ...]
    submit_decision: FoldingSubmitLifecycleDecision
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingResumeLifecycleDecision")
        if not isinstance(self.remaining_work, FoldingRemainingWorkPlan) or not isinstance(
            self.submit_decision, FoldingSubmitLifecycleDecision
        ):
            msg = "resume decision requires remaining-work and submit decisions"
            raise ValueError(msg)
        _string_tuple(self.recovered_result_ids, "recovered_result_ids")
        _matching_version(self.schema_version, self.remaining_work.schema_version, "remaining_work")
        _matching_version(self.schema_version, self.submit_decision.schema_version, "submit_decision")
        completed = set(self.remaining_work.completed_model_ids)
        if not set(self.recovered_result_ids) <= completed:
            msg = "recovered_result_ids must be included in completed_model_ids"
            raise ValueError(msg)
        queued_ids = {assignment.protein_id for assignment in self.submit_decision.queue_plan.assignments}
        if queued_ids != set(self.remaining_work.remaining_model_ids):
            msg = "resume submit queue must contain exactly the remaining model identities"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "remaining_work": self.remaining_work.to_mapping(),
            "recovered_result_ids": list(self.recovered_result_ids),
            "submit_decision": self.submit_decision.to_mapping(),
        }


@dataclass(frozen=True)
class FoldingArchiveLifecycleRequest:
    """Archive request using supplied associations or one explicit #52 scan."""

    config: FoldingLifecycleConfig
    index: FoldingIndex
    result_inventory: FoldingResultInventory | None
    scan_predictions_root: str | None
    selected_protein_ids: tuple[str, ...] | None
    prior_manifest: tuple[ArchiveManifestRecord, ...]
    force: bool
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingArchiveLifecycleRequest")
        if not isinstance(self.config, FoldingLifecycleConfig) or not isinstance(self.index, FoldingIndex):
            msg = "archive request requires FoldingLifecycleConfig and FoldingIndex values"
            raise ValueError(msg)
        if (self.result_inventory is None) == (self.scan_predictions_root is None):
            msg = "archive request must supply exactly one result_inventory or scan_predictions_root"
            raise ValueError(msg)
        if self.result_inventory is not None and not isinstance(self.result_inventory, FoldingResultInventory):
            msg = "result_inventory must be a FoldingResultInventory or null"
            raise ValueError(msg)
        if self.scan_predictions_root is not None:
            _nonempty_str(self.scan_predictions_root, "scan_predictions_root")
        if self.selected_protein_ids is not None:
            _string_tuple(self.selected_protein_ids, "selected_protein_ids")
        _typed_tuple(self.prior_manifest, ArchiveManifestRecord, "prior_manifest")
        if not isinstance(self.force, bool):
            msg = "force must be a boolean"
            raise ValueError(msg)
        for field_name, value in (
            ("config", self.config),
            ("index", self.index),
            ("result_inventory", self.result_inventory),
        ):
            if value is not None:
                _matching_version(self.schema_version, value.schema_version, field_name)
        if any(item.schema_version != self.schema_version for item in self.prior_manifest):
            msg = "prior_manifest schema versions must match the archive request"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.model_dump(mode="json"),
            "index": self.index.to_mapping(),
            "result_inventory": None if self.result_inventory is None else self.result_inventory.to_mapping(),
            "scan_predictions_root": self.scan_predictions_root,
            "selected_protein_ids": (None if self.selected_protein_ids is None else list(self.selected_protein_ids)),
            "prior_manifest": [item.to_mapping() for item in self.prior_manifest],
            "force": self.force,
        }


@dataclass(frozen=True)
class FoldingArchiveLifecycleDecision:
    """Associated result evidence and archive declarations that never execute."""

    result_inventory: FoldingResultInventory
    archive_plan: FoldingArchivePlan
    manifest_jsonl: str
    execution_authorized: Literal[False] = False
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingArchiveLifecycleDecision")
        if not isinstance(self.result_inventory, FoldingResultInventory) or not isinstance(
            self.archive_plan, FoldingArchivePlan
        ):
            msg = "archive decision requires result inventory and archive plan values"
            raise ValueError(msg)
        if not isinstance(self.manifest_jsonl, str):
            msg = "manifest_jsonl must be a string"
            raise ValueError(msg)
        if self.execution_authorized is not False:
            msg = "folding lifecycle decisions never authorize execution"
            raise ValueError(msg)
        _matching_version(self.schema_version, self.result_inventory.schema_version, "result_inventory")
        _matching_version(self.schema_version, self.archive_plan.schema_version, "archive_plan")
        complete_ids = {
            association.protein_id
            for association in self.result_inventory.associations
            if association.status == "complete"
        }
        referenced_ids = {protein_id for batch in self.archive_plan.batches for protein_id in batch.protein_ids} | set(
            self.archive_plan.skipped_prior_protein_ids
        )
        if not referenced_ids <= complete_ids:
            msg = "archive plan can reference only complete result associations"
            raise ValueError(msg)
        _validate_archive_manifest_jsonl(self.manifest_jsonl, self.archive_plan)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "result_inventory": self.result_inventory.to_mapping(),
            "archive_plan": self.archive_plan.to_mapping(),
            "manifest_jsonl": self.manifest_jsonl,
            "execution_authorized": self.execution_authorized,
        }


@dataclass(frozen=True)
class FoldingPreflightCheck:
    """One explicit local prerequisite result."""

    name: PreflightName
    status: PreflightStatus
    detail: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingPreflightCheck")
        if self.name not in {"config", "index", "checkpoint", "result", "archive"}:
            msg = f"unsupported preflight check name: {self.name!r}"
            raise ValueError(msg)
        if self.status not in {"pass", "warning", "fail", "not-required"}:
            msg = f"unsupported preflight check status: {self.status!r}"
            raise ValueError(msg)
        _nonempty_str(self.detail, "detail")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FoldingPreflightLifecycleRequest:
    """Actual local inputs to validate before one lifecycle action is planned."""

    action: PreflightAction
    config: FoldingLifecycleConfig
    index: FoldingIndex | None
    checkpoint_inputs: FoldingCheckpointInputs | None
    checkpoint_state: FoldingCheckpointState | None
    result_inventory: FoldingResultInventory | None
    scan_predictions_root: str | None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingPreflightLifecycleRequest")
        if self.action not in {"index", "submit", "resume", "merge", "archive"}:
            msg = f"unsupported preflight action: {self.action!r}"
            raise ValueError(msg)
        if not isinstance(self.config, FoldingLifecycleConfig):
            msg = "config must be a FoldingLifecycleConfig"
            raise ValueError(msg)
        optional_values = (
            (self.index, FoldingIndex, "index"),
            (self.checkpoint_inputs, FoldingCheckpointInputs, "checkpoint_inputs"),
            (self.checkpoint_state, FoldingCheckpointState, "checkpoint_state"),
            (self.result_inventory, FoldingResultInventory, "result_inventory"),
        )
        _matching_version(self.schema_version, self.config.schema_version, "config")
        for value, expected_type, field_name in optional_values:
            if value is not None:
                if not isinstance(value, expected_type):
                    msg = f"{field_name} must be a {expected_type.__name__} or null"
                    raise ValueError(msg)
                _matching_version(self.schema_version, value.schema_version, field_name)
        if self.scan_predictions_root is not None:
            _nonempty_str(self.scan_predictions_root, "scan_predictions_root")
        if self.action == "merge" and self.checkpoint_state is not None:
            msg = "merge preflight accepts checkpoint_inputs, not an already-merged checkpoint_state"
            raise ValueError(msg)
        if self.action != "merge" and self.checkpoint_inputs is not None:
            msg = "checkpoint_inputs are valid only for merge preflight"
            raise ValueError(msg)
        if self.action != "archive" and self.scan_predictions_root is not None:
            msg = "scan_predictions_root is valid only for archive preflight"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action": self.action,
            "config": self.config.model_dump(mode="json"),
            "index": None if self.index is None else self.index.to_mapping(),
            "checkpoint_inputs": (None if self.checkpoint_inputs is None else self.checkpoint_inputs.to_mapping()),
            "checkpoint_state": None if self.checkpoint_state is None else self.checkpoint_state.to_mapping(),
            "result_inventory": None if self.result_inventory is None else self.result_inventory.to_mapping(),
            "scan_predictions_root": self.scan_predictions_root,
        }


@dataclass(frozen=True)
class FoldingPreflightLifecycleDecision:
    """All prerequisite checks, including non-required categories."""

    action: PreflightAction
    checks: tuple[FoldingPreflightCheck, ...]
    can_plan: bool
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingPreflightLifecycleDecision")
        if self.action not in {"index", "submit", "resume", "merge", "archive"}:
            msg = f"unsupported preflight action: {self.action!r}"
            raise ValueError(msg)
        _typed_tuple(self.checks, FoldingPreflightCheck, "checks")
        expected_names = ("config", "index", "checkpoint", "result", "archive")
        if tuple(check.name for check in self.checks) != expected_names:
            msg = "preflight checks must cover config, index, checkpoint, result, and archive in order"
            raise ValueError(msg)
        if any(check.schema_version != self.schema_version for check in self.checks):
            msg = "preflight check schema versions must match the decision"
            raise ValueError(msg)
        if not isinstance(self.can_plan, bool) or self.can_plan != all(check.status != "fail" for check in self.checks):
            msg = "can_plan must be true exactly when no preflight check failed"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action": self.action,
            "checks": [check.to_mapping() for check in self.checks],
            "can_plan": self.can_plan,
        }


@dataclass(frozen=True)
class FoldingManifestEvidence:
    """Caller-supplied local preprocessing-manifest facts used by status."""

    total_proteins: int
    queue_artifact_count: int
    queue_layout: FoldingQueueLayout
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingManifestEvidence")
        _nonnegative_int(self.total_proteins, "total_proteins")
        _nonnegative_int(self.queue_artifact_count, "queue_artifact_count")
        if self.queue_layout not in {"per_node", "legacy_per_gpu"}:
            msg = f"unsupported queue_layout: {self.queue_layout!r}"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "total_proteins": self.total_proteins,
            "queue_artifact_count": self.queue_artifact_count,
            "queue_layout": self.queue_layout,
        }


@dataclass(frozen=True)
class FoldingStatusLifecycleRequest:
    """Pure status request containing only caller-supplied local evidence."""

    config: FoldingLifecycleConfig
    index: FoldingIndex
    manifest: FoldingManifestEvidence | None
    checkpoint_state: FoldingCheckpointState | None
    result_inventory: FoldingResultInventory | None
    archive_plan: FoldingArchivePlan | None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingStatusLifecycleRequest")
        values = (
            (self.config, FoldingLifecycleConfig, "config"),
            (self.index, FoldingIndex, "index"),
            (self.manifest, FoldingManifestEvidence, "manifest"),
            (self.checkpoint_state, FoldingCheckpointState, "checkpoint_state"),
            (self.result_inventory, FoldingResultInventory, "result_inventory"),
            (self.archive_plan, FoldingArchivePlan, "archive_plan"),
        )
        for value, expected_type, field_name in values:
            if value is None and field_name not in {"manifest", "checkpoint_state", "result_inventory", "archive_plan"}:
                msg = f"{field_name} is required"
                raise ValueError(msg)
            if value is not None:
                if not isinstance(value, expected_type):
                    msg = f"{field_name} must be a {expected_type.__name__}"
                    raise ValueError(msg)
                _matching_version(self.schema_version, value.schema_version, field_name)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.model_dump(mode="json"),
            "index": self.index.to_mapping(),
            "manifest": None if self.manifest is None else self.manifest.to_mapping(),
            "checkpoint_state": None if self.checkpoint_state is None else self.checkpoint_state.to_mapping(),
            "result_inventory": None if self.result_inventory is None else self.result_inventory.to_mapping(),
            "archive_plan": None if self.archive_plan is None else self.archive_plan.to_mapping(),
        }


@dataclass(frozen=True)
class FoldingStatusLifecycleDecision:
    """Deterministic summary with no live scheduler or filesystem lookup."""

    state: FoldingLifecycleState
    evidence_kinds: tuple[FoldingStatusEvidenceKind, ...]
    planned_model_count: int
    manifest_model_count: int | None
    checkpoint_completed_count: int
    checkpoint_failed_count: int
    recovered_result_count: int
    completed_model_count: int
    incomplete_result_count: int
    planned_archive_count: int
    invalid_evidence: tuple[str, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingStatusLifecycleDecision")
        if self.state not in {"initial", "partial", "complete", "invalid"}:
            msg = f"unsupported folding lifecycle state: {self.state!r}"
            raise ValueError(msg)
        evidence_order = ("manifest", "checkpoint", "result", "archive")
        if (
            not isinstance(self.evidence_kinds, tuple)
            or any(kind not in evidence_order for kind in self.evidence_kinds)
            or self.evidence_kinds != tuple(kind for kind in evidence_order if kind in set(self.evidence_kinds))
        ):
            msg = "evidence_kinds must be unique and ordered as manifest, checkpoint, result, archive"
            raise ValueError(msg)
        for field_name in (
            "planned_model_count",
            "checkpoint_completed_count",
            "checkpoint_failed_count",
            "recovered_result_count",
            "completed_model_count",
            "incomplete_result_count",
            "planned_archive_count",
        ):
            _nonnegative_int(getattr(self, field_name), field_name)
        if self.manifest_model_count is not None:
            _nonnegative_int(self.manifest_model_count, "manifest_model_count")
        evidence = set(self.evidence_kinds)
        if ("manifest" in evidence) != (self.manifest_model_count is not None):
            msg = "manifest evidence must correspond exactly to manifest_model_count presence"
            raise ValueError(msg)
        if "checkpoint" not in evidence and (self.checkpoint_completed_count or self.checkpoint_failed_count):
            msg = "checkpoint counts require checkpoint evidence"
            raise ValueError(msg)
        if "result" not in evidence and (self.recovered_result_count or self.incomplete_result_count):
            msg = "result counts require result evidence"
            raise ValueError(msg)
        if "archive" not in evidence and self.planned_archive_count:
            msg = "planned_archive_count requires archive evidence"
            raise ValueError(msg)
        _string_tuple(self.invalid_evidence, "invalid_evidence")
        if (self.state == "invalid") != bool(self.invalid_evidence):
            msg = "invalid state must correspond exactly to nonempty invalid_evidence"
            raise ValueError(msg)
        if self.state == "complete" and self.completed_model_count != self.planned_model_count:
            msg = "complete state requires every planned model to be completed"
            raise ValueError(msg)
        if self.completed_model_count > self.planned_model_count:
            msg = "completed_model_count cannot exceed planned_model_count"
            raise ValueError(msg)
        if self.completed_model_count > self.checkpoint_completed_count + self.recovered_result_count:
            msg = "completed_model_count cannot exceed supplied completion evidence"
            raise ValueError(msg)
        if self.state == "initial" and (self.evidence_kinds or self.completed_model_count):
            msg = "initial state cannot contain lifecycle evidence or completed models"
            raise ValueError(msg)
        if self.state == "partial" and (
            not self.evidence_kinds or self.completed_model_count >= self.planned_model_count
        ):
            msg = "partial state requires evidence and fewer completed than planned models"
            raise ValueError(msg)
        if self.state == "complete" and not self.evidence_kinds:
            msg = "complete state requires explicit lifecycle evidence"
            raise ValueError(msg)
        if self.state == "invalid" and not self.evidence_kinds:
            msg = "invalid state requires contradictory supplied evidence"
            raise ValueError(msg)
        if self.state != "invalid":
            if self.manifest_model_count is not None and self.manifest_model_count != self.planned_model_count:
                msg = "valid manifest evidence must match planned_model_count"
                raise ValueError(msg)
            if self.checkpoint_completed_count + self.checkpoint_failed_count > self.planned_model_count:
                msg = "valid checkpoint evidence cannot exceed planned_model_count"
                raise ValueError(msg)
            if "result" in evidence and (
                self.recovered_result_count + self.incomplete_result_count != self.planned_model_count
            ):
                msg = "valid result evidence must account for every planned model"
                raise ValueError(msg)
            if self.planned_archive_count > self.planned_model_count:
                msg = "valid archive evidence cannot exceed planned_model_count"
                raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "evidence_kinds": list(self.evidence_kinds),
            "planned_model_count": self.planned_model_count,
            "manifest_model_count": self.manifest_model_count,
            "checkpoint_completed_count": self.checkpoint_completed_count,
            "checkpoint_failed_count": self.checkpoint_failed_count,
            "recovered_result_count": self.recovered_result_count,
            "completed_model_count": self.completed_model_count,
            "incomplete_result_count": self.incomplete_result_count,
            "planned_archive_count": self.planned_archive_count,
            "invalid_evidence": list(self.invalid_evidence),
        }


def folding_lifecycle_config_from_mapping(payload: Mapping[str, object]) -> FoldingLifecycleConfig:
    """Load user-authored folding configuration with strict unknown-field rejection."""
    validate_schema_version(payload.get("schema_version"), record_name="FoldingLifecycleConfig")
    return FoldingLifecycleConfig.model_validate(dict(payload))


def folding_index_lifecycle_request_from_mapping(payload: Mapping[str, object]) -> FoldingIndexLifecycleRequest:
    _reject_unknown(payload, _INDEX_REQUEST_FIELDS, "FoldingIndexLifecycleRequest")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingIndexLifecycleRequest")
    return FoldingIndexLifecycleRequest(
        config=folding_lifecycle_config_from_mapping(_required_mapping(payload, "config")),
        schema_version=schema_version,
    )


def folding_index_lifecycle_decision_from_mapping(payload: Mapping[str, object]) -> FoldingIndexLifecycleDecision:
    _reject_unknown(payload, _INDEX_DECISION_FIELDS, "FoldingIndexLifecycleDecision")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingIndexLifecycleDecision")
    sort_by_length = payload.get("sort_by_length")
    if not isinstance(sort_by_length, bool):
        msg = "sort_by_length must be a boolean"
        raise ValueError(msg)
    return FoldingIndexLifecycleDecision(
        index=folding_index_from_mapping(_required_mapping(payload, "index")),
        index_worker_count=_required_int(payload, "index_worker_count"),
        sort_by_length=sort_by_length,
        schema_version=schema_version,
    )


def folding_queue_write_declaration_from_mapping(payload: Mapping[str, object]) -> FoldingQueueWriteDeclaration:
    _reject_unknown(payload, _QUEUE_WRITE_FIELDS, "FoldingQueueWriteDeclaration")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingQueueWriteDeclaration")
    return FoldingQueueWriteDeclaration(
        path=_required_str(payload, "path"),
        artifact=folding_queue_artifact_from_mapping(_required_mapping(payload, "artifact")),
        schema_version=schema_version,
    )


def folding_submit_lifecycle_request_from_mapping(payload: Mapping[str, object]) -> FoldingSubmitLifecycleRequest:
    _reject_unknown(payload, _SUBMIT_REQUEST_FIELDS, "FoldingSubmitLifecycleRequest")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingSubmitLifecycleRequest")
    return FoldingSubmitLifecycleRequest(
        config=folding_lifecycle_config_from_mapping(_required_mapping(payload, "config")),
        index=folding_index_from_mapping(_required_mapping(payload, "index")),
        schema_version=schema_version,
    )


def folding_submit_lifecycle_decision_from_mapping(payload: Mapping[str, object]) -> FoldingSubmitLifecycleDecision:
    _reject_unknown(payload, _SUBMIT_DECISION_FIELDS, "FoldingSubmitLifecycleDecision")
    schema_version = validate_schema_version(
        payload.get("schema_version"), record_name="FoldingSubmitLifecycleDecision"
    )
    submission_planned = _required_bool(payload, "submission_planned")
    execution_authorized = _required_bool(payload, "execution_authorized")
    if execution_authorized:
        msg = "execution_authorized must be false"
        raise ValueError(msg)
    return FoldingSubmitLifecycleDecision(
        queue_plan=folding_queue_plan_from_mapping(_required_mapping(payload, "queue_plan")),
        queue_writes=_mapping_tuple(
            payload,
            "queue_writes",
            FoldingQueueWriteDeclaration,
            folding_queue_write_declaration_from_mapping,
        ),
        submission_planned=submission_planned,
        execution_authorized=False,
        schema_version=schema_version,
    )


def folding_checkpoint_inputs_from_mapping(payload: Mapping[str, object]) -> FoldingCheckpointInputs:
    _reject_unknown(payload, _CHECKPOINT_INPUT_FIELDS, "FoldingCheckpointInputs")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingCheckpointInputs")
    return FoldingCheckpointInputs(
        node_completion_shards=_required_str_tuple(payload, "node_completion_shards"),
        node_failure_shards=_required_str_tuple(payload, "node_failure_shards"),
        current_gpu_completion_shards=_required_str_tuple(payload, "current_gpu_completion_shards"),
        current_gpu_failure_shards=_required_str_tuple(payload, "current_gpu_failure_shards"),
        historical_gpu_completion_shards=_required_str_tuple(payload, "historical_gpu_completion_shards"),
        historical_gpu_failure_shards=_required_str_tuple(payload, "historical_gpu_failure_shards"),
        merged_completion_view=_optional_str(payload, "merged_completion_view"),
        merged_failure_view=_optional_str(payload, "merged_failure_view"),
        schema_version=schema_version,
    )


def folding_merge_lifecycle_request_from_mapping(payload: Mapping[str, object]) -> FoldingMergeLifecycleRequest:
    _reject_unknown(payload, _MERGE_REQUEST_FIELDS, "FoldingMergeLifecycleRequest")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingMergeLifecycleRequest")
    return FoldingMergeLifecycleRequest(
        config=folding_lifecycle_config_from_mapping(_required_mapping(payload, "config")),
        checkpoint_inputs=folding_checkpoint_inputs_from_mapping(_required_mapping(payload, "checkpoint_inputs")),
        schema_version=schema_version,
    )


def folding_merge_lifecycle_decision_from_mapping(payload: Mapping[str, object]) -> FoldingMergeLifecycleDecision:
    _reject_unknown(payload, _MERGE_DECISION_FIELDS, "FoldingMergeLifecycleDecision")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingMergeLifecycleDecision")
    return FoldingMergeLifecycleDecision(
        state=folding_checkpoint_state_from_mapping(_required_mapping(payload, "state")),
        completed_view_path=_required_str(payload, "completed_view_path"),
        failed_view_path=_required_str(payload, "failed_view_path"),
        write_planned=_required_bool(payload, "write_planned"),
        schema_version=schema_version,
    )


def folding_resume_lifecycle_request_from_mapping(payload: Mapping[str, object]) -> FoldingResumeLifecycleRequest:
    _reject_unknown(payload, _RESUME_REQUEST_FIELDS, "FoldingResumeLifecycleRequest")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingResumeLifecycleRequest")
    return FoldingResumeLifecycleRequest(
        config=folding_lifecycle_config_from_mapping(_required_mapping(payload, "config")),
        index=folding_index_from_mapping(_required_mapping(payload, "index")),
        checkpoint_state=folding_checkpoint_state_from_mapping(_required_mapping(payload, "checkpoint_state")),
        result_inventory=folding_result_inventory_from_mapping(_required_mapping(payload, "result_inventory")),
        retry_failed=_required_bool(payload, "retry_failed"),
        schema_version=schema_version,
    )


def folding_resume_lifecycle_decision_from_mapping(payload: Mapping[str, object]) -> FoldingResumeLifecycleDecision:
    _reject_unknown(payload, _RESUME_DECISION_FIELDS, "FoldingResumeLifecycleDecision")
    schema_version = validate_schema_version(
        payload.get("schema_version"), record_name="FoldingResumeLifecycleDecision"
    )
    return FoldingResumeLifecycleDecision(
        remaining_work=folding_remaining_work_plan_from_mapping(_required_mapping(payload, "remaining_work")),
        recovered_result_ids=_required_str_tuple(payload, "recovered_result_ids"),
        submit_decision=folding_submit_lifecycle_decision_from_mapping(_required_mapping(payload, "submit_decision")),
        schema_version=schema_version,
    )


def folding_archive_lifecycle_request_from_mapping(payload: Mapping[str, object]) -> FoldingArchiveLifecycleRequest:
    _reject_unknown(payload, _ARCHIVE_REQUEST_FIELDS, "FoldingArchiveLifecycleRequest")
    schema_version = validate_schema_version(
        payload.get("schema_version"), record_name="FoldingArchiveLifecycleRequest"
    )
    raw_inventory = payload.get("result_inventory")
    if raw_inventory is not None and not isinstance(raw_inventory, Mapping):
        msg = "result_inventory must be an object or null"
        raise ValueError(msg)
    return FoldingArchiveLifecycleRequest(
        config=folding_lifecycle_config_from_mapping(_required_mapping(payload, "config")),
        index=folding_index_from_mapping(_required_mapping(payload, "index")),
        result_inventory=(None if raw_inventory is None else folding_result_inventory_from_mapping(raw_inventory)),
        scan_predictions_root=_optional_str(payload, "scan_predictions_root"),
        selected_protein_ids=_optional_str_tuple(payload, "selected_protein_ids"),
        prior_manifest=_mapping_tuple(
            payload,
            "prior_manifest",
            ArchiveManifestRecord,
            archive_manifest_record_from_mapping,
        ),
        force=_required_bool(payload, "force"),
        schema_version=schema_version,
    )


def folding_archive_lifecycle_decision_from_mapping(payload: Mapping[str, object]) -> FoldingArchiveLifecycleDecision:
    _reject_unknown(payload, _ARCHIVE_DECISION_FIELDS, "FoldingArchiveLifecycleDecision")
    schema_version = validate_schema_version(
        payload.get("schema_version"), record_name="FoldingArchiveLifecycleDecision"
    )
    execution_authorized = _required_bool(payload, "execution_authorized")
    if execution_authorized:
        msg = "execution_authorized must be false"
        raise ValueError(msg)
    return FoldingArchiveLifecycleDecision(
        result_inventory=folding_result_inventory_from_mapping(_required_mapping(payload, "result_inventory")),
        archive_plan=archive_plan_from_mapping(_required_mapping(payload, "archive_plan")),
        manifest_jsonl=_required_str(payload, "manifest_jsonl"),
        execution_authorized=False,
        schema_version=schema_version,
    )


def folding_preflight_check_from_mapping(payload: Mapping[str, object]) -> FoldingPreflightCheck:
    _reject_unknown(payload, _PREFLIGHT_CHECK_FIELDS, "FoldingPreflightCheck")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingPreflightCheck")
    return FoldingPreflightCheck(
        name=_required_preflight_name(payload, "name"),
        status=_required_preflight_status(payload, "status"),
        detail=_required_str(payload, "detail"),
        schema_version=schema_version,
    )


def folding_preflight_lifecycle_request_from_mapping(payload: Mapping[str, object]) -> FoldingPreflightLifecycleRequest:
    _reject_unknown(payload, _PREFLIGHT_REQUEST_FIELDS, "FoldingPreflightLifecycleRequest")
    schema_version = validate_schema_version(
        payload.get("schema_version"), record_name="FoldingPreflightLifecycleRequest"
    )
    return FoldingPreflightLifecycleRequest(
        action=_required_preflight_action(payload, "action"),
        config=folding_lifecycle_config_from_mapping(_required_mapping(payload, "config")),
        index=_optional_mapping_load(payload, "index", folding_index_from_mapping),
        checkpoint_inputs=_optional_mapping_load(
            payload,
            "checkpoint_inputs",
            folding_checkpoint_inputs_from_mapping,
        ),
        checkpoint_state=_optional_mapping_load(payload, "checkpoint_state", folding_checkpoint_state_from_mapping),
        result_inventory=_optional_mapping_load(payload, "result_inventory", folding_result_inventory_from_mapping),
        scan_predictions_root=_optional_str(payload, "scan_predictions_root"),
        schema_version=schema_version,
    )


def folding_preflight_lifecycle_decision_from_mapping(
    payload: Mapping[str, object],
) -> FoldingPreflightLifecycleDecision:
    _reject_unknown(payload, _PREFLIGHT_DECISION_FIELDS, "FoldingPreflightLifecycleDecision")
    schema_version = validate_schema_version(
        payload.get("schema_version"), record_name="FoldingPreflightLifecycleDecision"
    )
    return FoldingPreflightLifecycleDecision(
        action=_required_preflight_action(payload, "action"),
        checks=_mapping_tuple(
            payload,
            "checks",
            FoldingPreflightCheck,
            folding_preflight_check_from_mapping,
        ),
        can_plan=_required_bool(payload, "can_plan"),
        schema_version=schema_version,
    )


def folding_manifest_evidence_from_mapping(payload: Mapping[str, object]) -> FoldingManifestEvidence:
    _reject_unknown(payload, _MANIFEST_FIELDS, "FoldingManifestEvidence")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingManifestEvidence")
    layout = payload.get("queue_layout")
    if layout not in {"per_node", "legacy_per_gpu"}:
        msg = "queue_layout must be per_node or legacy_per_gpu"
        raise ValueError(msg)
    return FoldingManifestEvidence(
        total_proteins=_required_int(payload, "total_proteins"),
        queue_artifact_count=_required_int(payload, "queue_artifact_count"),
        queue_layout=layout,  # type: ignore[arg-type]
        schema_version=schema_version,
    )


def folding_status_lifecycle_request_from_mapping(payload: Mapping[str, object]) -> FoldingStatusLifecycleRequest:
    _reject_unknown(payload, _STATUS_REQUEST_FIELDS, "FoldingStatusLifecycleRequest")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingStatusLifecycleRequest")
    return FoldingStatusLifecycleRequest(
        config=folding_lifecycle_config_from_mapping(_required_mapping(payload, "config")),
        index=folding_index_from_mapping(_required_mapping(payload, "index")),
        manifest=_optional_mapping_load(payload, "manifest", folding_manifest_evidence_from_mapping),
        checkpoint_state=_optional_mapping_load(payload, "checkpoint_state", folding_checkpoint_state_from_mapping),
        result_inventory=_optional_mapping_load(payload, "result_inventory", folding_result_inventory_from_mapping),
        archive_plan=_optional_mapping_load(payload, "archive_plan", archive_plan_from_mapping),
        schema_version=schema_version,
    )


def folding_status_lifecycle_decision_from_mapping(payload: Mapping[str, object]) -> FoldingStatusLifecycleDecision:
    _reject_unknown(payload, _STATUS_DECISION_FIELDS, "FoldingStatusLifecycleDecision")
    schema_version = validate_schema_version(
        payload.get("schema_version"), record_name="FoldingStatusLifecycleDecision"
    )
    state = payload.get("state")
    if state not in {"initial", "partial", "complete", "invalid"}:
        msg = "state must be initial, partial, complete, or invalid"
        raise ValueError(msg)
    manifest_count = payload.get("manifest_model_count")
    if manifest_count is not None and (not isinstance(manifest_count, int) or isinstance(manifest_count, bool)):
        msg = "manifest_model_count must be an integer or null"
        raise ValueError(msg)
    return FoldingStatusLifecycleDecision(
        state=state,  # type: ignore[arg-type]
        evidence_kinds=_required_status_evidence_kinds(payload, "evidence_kinds"),
        planned_model_count=_required_int(payload, "planned_model_count"),
        manifest_model_count=manifest_count,
        checkpoint_completed_count=_required_int(payload, "checkpoint_completed_count"),
        checkpoint_failed_count=_required_int(payload, "checkpoint_failed_count"),
        recovered_result_count=_required_int(payload, "recovered_result_count"),
        completed_model_count=_required_int(payload, "completed_model_count"),
        incomplete_result_count=_required_int(payload, "incomplete_result_count"),
        planned_archive_count=_required_int(payload, "planned_archive_count"),
        invalid_evidence=_required_str_tuple(payload, "invalid_evidence"),
        schema_version=schema_version,
    )


def _validate_version(value: int, name: str) -> None:
    validated = validate_schema_version(value, record_name=name)
    if value != validated:
        msg = f"{name} schema_version must be declared explicitly"
        raise ValueError(msg)


def _validate_archive_manifest_jsonl(contents: str, plan: FoldingArchivePlan) -> None:
    expected_records = tuple(batch.manifest_record.to_mapping() for batch in plan.batches)
    lines = contents.splitlines()
    if len(lines) != len(expected_records):
        msg = "manifest_jsonl must contain exactly one record per archive batch"
        raise ValueError(msg)
    for line_number, (line, expected) in enumerate(zip(lines, expected_records, strict=True), start=1):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"manifest_jsonl line {line_number} is not valid JSON"
            raise ValueError(msg) from exc
        if not isinstance(parsed, Mapping) or parsed != expected:
            msg = f"manifest_jsonl line {line_number} does not match its archive batch"
            raise ValueError(msg)
    canonical = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in expected_records)
    if contents != canonical:
        msg = "manifest_jsonl must exactly render the archive plan manifest records"
        raise ValueError(msg)


def _matching_version(outer: int, inner: int, field_name: str) -> None:
    if outer != inner:
        msg = f"{field_name} schema version must match its lifecycle record"
        raise ValueError(msg)


def _positive_int(value: object, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{field_name} must be a positive integer"
        raise ValueError(msg)


def _nonnegative_int(value: object, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{field_name} must be a non-negative integer"
        raise ValueError(msg)


def _nonempty_str(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)


def _typed_tuple(value: object, item_type: type[object], field_name: str) -> None:
    if not isinstance(value, tuple) or not all(isinstance(item, item_type) for item in value):
        msg = f"{field_name} must be an immutable tuple of {item_type.__name__} values"
        raise ValueError(msg)


def _string_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple) or any(not isinstance(item, str) or not item.strip() for item in value):
        msg = f"{field_name} must be an immutable tuple of non-empty strings"
        raise ValueError(msg)
    if len(value) != len(set(value)):
        msg = f"{field_name} must not contain duplicate values"
        raise ValueError(msg)


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)
    return value


def _required_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean"
        raise ValueError(msg)
    return value


def _required_str_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple) or not all(isinstance(item, str) for item in value):
        msg = f"{key} must be a list of strings"
        raise ValueError(msg)
    return tuple(value)


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        msg = f"{key} must be a string or null"
        raise ValueError(msg)
    return value


def _optional_str_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, list | tuple) or not all(isinstance(item, str) for item in value):
        msg = f"{key} must be a list of strings or null"
        raise ValueError(msg)
    return tuple(value)


def _optional_mapping_load[ItemT](
    payload: Mapping[str, object],
    key: str,
    loader: Callable[[Mapping[str, object]], ItemT],
) -> ItemT | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        msg = f"{key} must be an object or null"
        raise ValueError(msg)
    return loader(value)


def _required_preflight_action(payload: Mapping[str, object], key: str) -> PreflightAction:
    value = payload.get(key)
    if value not in {"index", "submit", "resume", "merge", "archive"}:
        msg = f"{key} must be index, submit, resume, merge, or archive"
        raise ValueError(msg)
    return value  # type: ignore[return-value]


def _required_preflight_name(payload: Mapping[str, object], key: str) -> PreflightName:
    value = payload.get(key)
    if value not in {"config", "index", "checkpoint", "result", "archive"}:
        msg = f"{key} must be an owned preflight prerequisite name"
        raise ValueError(msg)
    return value  # type: ignore[return-value]


def _required_preflight_status(payload: Mapping[str, object], key: str) -> PreflightStatus:
    value = payload.get(key)
    if value not in {"pass", "warning", "fail", "not-required"}:
        msg = f"{key} must be pass, warning, fail, or not-required"
        raise ValueError(msg)
    return value  # type: ignore[return-value]


def _required_status_evidence_kinds(
    payload: Mapping[str, object],
    key: str,
) -> tuple[FoldingStatusEvidenceKind, ...]:
    values = _required_str_tuple(payload, key)
    if any(value not in {"manifest", "checkpoint", "result", "archive"} for value in values):
        msg = f"{key} contains an unsupported status evidence kind"
        raise ValueError(msg)
    return values  # type: ignore[return-value]


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be an object"
        raise ValueError(msg)
    return value


def _mapping_tuple[ItemT](
    payload: Mapping[str, object],
    key: str,
    item_type: type[ItemT],
    loader: Callable[[Mapping[str, object]], ItemT],
) -> tuple[ItemT, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    items: list[ItemT] = []
    for index, raw_item in enumerate(value):
        if not isinstance(raw_item, Mapping):
            msg = f"{key}[{index}] must be an object"
            raise ValueError(msg)
        loaded = loader(raw_item)
        if not isinstance(loaded, item_type):
            msg = f"{key}[{index}] has the wrong record type"
            raise ValueError(msg)
        items.append(loaded)
    return tuple(items)


def _reject_unknown(payload: Mapping[str, object], allowed: frozenset[str], name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        msg = f"{name} has unknown fields: {', '.join(unknown)}"
        raise ValueError(msg)


__all__ = [
    "FoldingArchiveLifecycleDecision",
    "FoldingArchiveLifecycleRequest",
    "FoldingCheckpointInputs",
    "FoldingIndexLifecycleDecision",
    "FoldingIndexLifecycleRequest",
    "FoldingLifecycleConfig",
    "FoldingLifecycleState",
    "FoldingManifestEvidence",
    "FoldingMergeLifecycleDecision",
    "FoldingMergeLifecycleRequest",
    "FoldingPreflightCheck",
    "FoldingPreflightLifecycleDecision",
    "FoldingPreflightLifecycleRequest",
    "FoldingQueueWriteDeclaration",
    "FoldingResumeLifecycleDecision",
    "FoldingResumeLifecycleRequest",
    "FoldingStatusEvidenceKind",
    "FoldingStatusLifecycleDecision",
    "FoldingStatusLifecycleRequest",
    "FoldingSubmitLifecycleDecision",
    "FoldingSubmitLifecycleRequest",
    "PreflightAction",
    "PreflightName",
    "PreflightStatus",
    "folding_archive_lifecycle_decision_from_mapping",
    "folding_archive_lifecycle_request_from_mapping",
    "folding_checkpoint_inputs_from_mapping",
    "folding_index_lifecycle_decision_from_mapping",
    "folding_index_lifecycle_request_from_mapping",
    "folding_lifecycle_config_from_mapping",
    "folding_manifest_evidence_from_mapping",
    "folding_merge_lifecycle_decision_from_mapping",
    "folding_merge_lifecycle_request_from_mapping",
    "folding_preflight_check_from_mapping",
    "folding_preflight_lifecycle_decision_from_mapping",
    "folding_preflight_lifecycle_request_from_mapping",
    "folding_queue_write_declaration_from_mapping",
    "folding_resume_lifecycle_decision_from_mapping",
    "folding_resume_lifecycle_request_from_mapping",
    "folding_status_lifecycle_decision_from_mapping",
    "folding_status_lifecycle_request_from_mapping",
    "folding_submit_lifecycle_decision_from_mapping",
    "folding_submit_lifecycle_request_from_mapping",
]
