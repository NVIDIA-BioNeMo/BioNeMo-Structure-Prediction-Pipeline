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

"""Immutable folding queue plans and render results.

Port Baseline sources:
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/scripts/preprocess_trt_bionemo.py:356-629``
and
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/python/batch_csv_manager.py:27-179``.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

FoldingQueueStrategy = Literal["round_robin", "exact_length", "runtime_balanced"]
FoldingQueueLayout = Literal["per_node", "legacy_per_gpu"]

_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "strategy",
        "layout",
        "worker_count",
        "max_proteins_per_batch",
        "runtime_coefficient",
        "base_overhead_seconds",
    }
)
_ASSIGNMENT_FIELDS = frozenset(
    {
        "schema_version",
        "source_ordinal",
        "worker_id",
        "batch_id",
        "sequence_length",
        "msa_path",
        "protein_id",
        "length_batch",
    }
)
_PLAN_FIELDS = frozenset({"schema_version", "config", "assignments"})
_ARTIFACT_FIELDS = frozenset({"schema_version", "worker_id", "file_name", "content", "total_proteins", "num_batches"})
_RENDER_FIELDS = frozenset({"schema_version", "layout", "artifacts"})


@dataclass(frozen=True)
class FoldingQueueConfig:
    """Explicit strategy, layout, worker, and runtime-model inputs."""

    strategy: FoldingQueueStrategy
    layout: FoldingQueueLayout
    worker_count: int
    max_proteins_per_batch: int = 200
    runtime_coefficient: float = 0.0001
    base_overhead_seconds: float = 30.0
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingQueueConfig")
        if self.strategy not in {"round_robin", "exact_length", "runtime_balanced"}:
            msg = f"unsupported folding queue strategy {self.strategy!r}"
            raise ValueError(msg)
        if self.layout not in {"per_node", "legacy_per_gpu"}:
            msg = f"unsupported folding queue layout {self.layout!r}"
            raise ValueError(msg)
        _validate_positive_int(self.worker_count, "worker_count")
        _validate_positive_int(self.max_proteins_per_batch, "max_proteins_per_batch")
        _validate_nonnegative_number(self.runtime_coefficient, "runtime_coefficient")
        _validate_nonnegative_number(self.base_overhead_seconds, "base_overhead_seconds")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "layout": self.layout,
            "worker_count": self.worker_count,
            "max_proteins_per_batch": self.max_proteins_per_batch,
            "runtime_coefficient": self.runtime_coefficient,
            "base_overhead_seconds": self.base_overhead_seconds,
        }


@dataclass(frozen=True)
class FoldingQueueAssignment:
    """One canonical folding-index record assigned to one queue worker."""

    source_ordinal: int
    worker_id: int
    batch_id: int
    sequence_length: int
    msa_path: str
    protein_id: str
    length_batch: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingQueueAssignment")
        _validate_nonnegative_int(self.source_ordinal, "source_ordinal")
        _validate_nonnegative_int(self.worker_id, "worker_id")
        _validate_nonnegative_int(self.batch_id, "batch_id")
        _validate_positive_int(self.sequence_length, "sequence_length")
        _validate_nonempty_str(self.msa_path, "msa_path")
        _validate_nonempty_str(self.protein_id, "protein_id")
        if self.length_batch is not None:
            _validate_nonempty_str(self.length_batch, "length_batch")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_ordinal": self.source_ordinal,
            "worker_id": self.worker_id,
            "batch_id": self.batch_id,
            "sequence_length": self.sequence_length,
            "msa_path": self.msa_path,
            "protein_id": self.protein_id,
            "length_batch": self.length_batch,
        }


@dataclass(frozen=True)
class FoldingQueuePlan:
    """Validated immutable result from one queue-planning strategy."""

    config: FoldingQueueConfig
    assignments: tuple[FoldingQueueAssignment, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingQueuePlan")
        if not isinstance(self.config, FoldingQueueConfig):
            msg = "config must be a FoldingQueueConfig"
            raise ValueError(msg)
        if not isinstance(self.assignments, tuple) or not all(
            isinstance(assignment, FoldingQueueAssignment) for assignment in self.assignments
        ):
            msg = "assignments must be a tuple of FoldingQueueAssignment values"
            raise ValueError(msg)
        if self.config.schema_version != self.schema_version or any(
            assignment.schema_version != self.schema_version for assignment in self.assignments
        ):
            msg = "nested queue schema versions must match the plan schema version"
            raise ValueError(msg)
        if any(assignment.worker_id >= self.config.worker_count for assignment in self.assignments):
            msg = "assignment worker_id must be smaller than config.worker_count"
            raise ValueError(msg)
        _reject_duplicate_values(
            (assignment.source_ordinal for assignment in self.assignments), field_name="source_ordinal"
        )
        _reject_duplicate_values((assignment.protein_id for assignment in self.assignments), field_name="protein_id")
        _reject_duplicate_values((assignment.msa_path for assignment in self.assignments), field_name="msa_path")

        batch_ids = sorted({assignment.batch_id for assignment in self.assignments})
        if batch_ids != list(range(len(batch_ids))):
            msg = "batch_id values must form a contiguous zero-based sequence"
            raise ValueError(msg)
        if self.config.strategy == "exact_length":
            if any(assignment.length_batch is None for assignment in self.assignments):
                msg = "exact_length assignments require length_batch"
                raise ValueError(msg)
        elif any(assignment.length_batch is not None for assignment in self.assignments):
            msg = f"{self.config.strategy} assignments must not declare length_batch"
            raise ValueError(msg)
        assignments_by_batch: dict[int, list[FoldingQueueAssignment]] = {}
        for assignment in self.assignments:
            assignments_by_batch.setdefault(assignment.batch_id, []).append(assignment)
        for batch_id, assignments in assignments_by_batch.items():
            if len({assignment.worker_id for assignment in assignments}) != 1:
                msg = f"batch_id {batch_id} must belong to exactly one worker"
                raise ValueError(msg)
            if self.config.strategy == "exact_length":
                if (
                    len({assignment.length_batch for assignment in assignments}) != 1
                    or len({assignment.sequence_length for assignment in assignments}) != 1
                ):
                    msg = f"exact_length batch_id {batch_id} must have one length_batch and sequence length"
                    raise ValueError(msg)
            elif len(assignments) != 1:
                msg = f"{self.config.strategy} batch_id {batch_id} must identify exactly one protein"
                raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_mapping(),
            "assignments": [assignment.to_mapping() for assignment in self.assignments],
        }


@dataclass(frozen=True)
class FoldingQueueArtifact:
    """One nonempty per-node or legacy per-GPU CSV render."""

    worker_id: int
    file_name: str
    content: str
    total_proteins: int
    num_batches: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingQueueArtifact")
        _validate_nonnegative_int(self.worker_id, "worker_id")
        _validate_nonempty_str(self.file_name, "file_name")
        _validate_nonempty_str(self.content, "content")
        _validate_positive_int(self.total_proteins, "total_proteins")
        _validate_positive_int(self.num_batches, "num_batches")
        _validate_artifact_content(self)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "worker_id": self.worker_id,
            "file_name": self.file_name,
            "content": self.content,
            "total_proteins": self.total_proteins,
            "num_batches": self.num_batches,
        }


@dataclass(frozen=True)
class FoldingQueueRenderResult:
    """Immutable collection of rendered queue artifacts."""

    layout: FoldingQueueLayout
    artifacts: tuple[FoldingQueueArtifact, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingQueueRenderResult")
        if self.layout not in {"per_node", "legacy_per_gpu"}:
            msg = f"unsupported folding queue layout {self.layout!r}"
            raise ValueError(msg)
        if not isinstance(self.artifacts, tuple) or not all(
            isinstance(artifact, FoldingQueueArtifact) for artifact in self.artifacts
        ):
            msg = "artifacts must be a tuple of FoldingQueueArtifact values"
            raise ValueError(msg)
        worker_ids = tuple(artifact.worker_id for artifact in self.artifacts)
        if worker_ids != tuple(sorted(set(worker_ids))):
            msg = "rendered artifact worker_id values must be unique and sorted"
            raise ValueError(msg)
        prefix = "node" if self.layout == "per_node" else "gpu"
        for artifact in self.artifacts:
            expected_name = f"{prefix}{artifact.worker_id}_batches.csv"
            if artifact.file_name != expected_name:
                msg = f"artifact file_name must be {expected_name!r} for layout {self.layout!r}"
                raise ValueError(msg)
            if artifact.schema_version != self.schema_version:
                msg = "artifact schema versions must match the render-result schema version"
                raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "layout": self.layout,
            "artifacts": [artifact.to_mapping() for artifact in self.artifacts],
        }


def folding_queue_config_from_mapping(payload: Mapping[str, object]) -> FoldingQueueConfig:
    _reject_unknown_fields(payload, _CONFIG_FIELDS, record_name="FoldingQueueConfig")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingQueueConfig")
    return FoldingQueueConfig(
        schema_version=schema_version,
        strategy=_required_strategy(payload, "strategy"),
        layout=_required_layout(payload, "layout"),
        worker_count=_required_int(payload, "worker_count"),
        max_proteins_per_batch=_required_int(payload, "max_proteins_per_batch"),
        runtime_coefficient=_required_number(payload, "runtime_coefficient"),
        base_overhead_seconds=_required_number(payload, "base_overhead_seconds"),
    )


def folding_queue_assignment_from_mapping(payload: Mapping[str, object]) -> FoldingQueueAssignment:
    _reject_unknown_fields(payload, _ASSIGNMENT_FIELDS, record_name="FoldingQueueAssignment")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingQueueAssignment")
    return FoldingQueueAssignment(
        schema_version=schema_version,
        source_ordinal=_required_int(payload, "source_ordinal"),
        worker_id=_required_int(payload, "worker_id"),
        batch_id=_required_int(payload, "batch_id"),
        sequence_length=_required_int(payload, "sequence_length"),
        msa_path=_required_str(payload, "msa_path"),
        protein_id=_required_str(payload, "protein_id"),
        length_batch=_optional_str(payload, "length_batch"),
    )


def folding_queue_plan_from_mapping(payload: Mapping[str, object]) -> FoldingQueuePlan:
    _reject_unknown_fields(payload, _PLAN_FIELDS, record_name="FoldingQueuePlan")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingQueuePlan")
    raw_config = payload.get("config")
    if not isinstance(raw_config, Mapping):
        msg = "config must be an object"
        raise ValueError(msg)
    config = folding_queue_config_from_mapping(raw_config)
    assignments = _required_assignment_tuple(payload, "assignments")
    return FoldingQueuePlan(schema_version=schema_version, config=config, assignments=assignments)


def folding_queue_artifact_from_mapping(payload: Mapping[str, object]) -> FoldingQueueArtifact:
    _reject_unknown_fields(payload, _ARTIFACT_FIELDS, record_name="FoldingQueueArtifact")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingQueueArtifact")
    return FoldingQueueArtifact(
        schema_version=schema_version,
        worker_id=_required_int(payload, "worker_id"),
        file_name=_required_str(payload, "file_name"),
        content=_required_str(payload, "content"),
        total_proteins=_required_int(payload, "total_proteins"),
        num_batches=_required_int(payload, "num_batches"),
    )


def folding_queue_render_result_from_mapping(payload: Mapping[str, object]) -> FoldingQueueRenderResult:
    _reject_unknown_fields(payload, _RENDER_FIELDS, record_name="FoldingQueueRenderResult")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingQueueRenderResult")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list | tuple):
        msg = "artifacts must be a list"
        raise ValueError(msg)
    artifacts: list[FoldingQueueArtifact] = []
    for index, raw_artifact in enumerate(raw_artifacts):
        if not isinstance(raw_artifact, Mapping):
            msg = f"artifacts[{index}] must be an object"
            raise ValueError(msg)
        artifacts.append(folding_queue_artifact_from_mapping(raw_artifact))
    return FoldingQueueRenderResult(
        schema_version=schema_version,
        layout=_required_layout(payload, "layout"),
        artifacts=tuple(artifacts),
    )


def _required_assignment_tuple(payload: Mapping[str, object], key: str) -> tuple[FoldingQueueAssignment, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    assignments: list[FoldingQueueAssignment] = []
    for index, raw_assignment in enumerate(value):
        if not isinstance(raw_assignment, Mapping):
            msg = f"{key}[{index}] must be an object"
            raise ValueError(msg)
        assignments.append(folding_queue_assignment_from_mapping(raw_assignment))
    return tuple(assignments)


def _required_strategy(payload: Mapping[str, object], key: str) -> FoldingQueueStrategy:
    value = _required_str(payload, key)
    if value not in {"round_robin", "exact_length", "runtime_balanced"}:
        msg = f"{key} must be round_robin, exact_length, or runtime_balanced"
        raise ValueError(msg)
    return cast("FoldingQueueStrategy", value)


def _required_layout(payload: Mapping[str, object], key: str) -> FoldingQueueLayout:
    value = _required_str(payload, key)
    if value not in {"per_node", "legacy_per_gpu"}:
        msg = f"{key} must be per_node or legacy_per_gpu"
        raise ValueError(msg)
    return cast("FoldingQueueLayout", value)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None or isinstance(value, str):
        return value
    msg = f"{key} must be a string or null"
    raise ValueError(msg)


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


def _required_number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        msg = f"{key} must be a number"
        raise ValueError(msg)
    return float(value)


def _reject_unknown_fields(payload: Mapping[str, object], allowed: frozenset[str], *, record_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        msg = f"{record_name} has unknown fields: {', '.join(unknown)}"
        raise ValueError(msg)


def _reject_duplicate_values(values: Iterable[object], *, field_name: str) -> None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            msg = f"duplicate {field_name} {value!r} in folding queue plan"
            raise ValueError(msg)
        seen.add(value)


def _validate_nonempty_str(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)


def _validate_nonnegative_int(value: object, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{field_name} must be a non-negative integer"
        raise ValueError(msg)


def _validate_positive_int(value: object, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{field_name} must be a positive integer"
        raise ValueError(msg)


def _validate_nonnegative_number(value: object, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        msg = f"{field_name} must be a finite non-negative number"
        raise ValueError(msg)


def _validate_artifact_content(artifact: FoldingQueueArtifact) -> None:
    try:
        reader = csv.DictReader(io.StringIO(artifact.content, newline=""), strict=True)
        if tuple(reader.fieldnames or ()) != ("batch_id", "seq_length", "msa_path", "protein_id"):
            msg = "queue artifact content must use the exact baseline CSV columns"
            raise ValueError(msg)
        rows = list(reader)
    except csv.Error as exc:
        msg = f"queue artifact content is not valid CSV: {exc}"
        raise ValueError(msg) from exc
    if len(rows) != artifact.total_proteins:
        msg = "queue artifact total_proteins must match its CSV rows"
        raise ValueError(msg)
    batch_ids = {row["batch_id"] for row in rows}
    if "" in batch_ids or len(batch_ids) != artifact.num_batches:
        msg = "queue artifact num_batches must match its distinct CSV batch IDs"
        raise ValueError(msg)


def _validate_direct_schema_version(schema_version: int, record_name: str) -> None:
    validated = validate_schema_version(schema_version, record_name=record_name)
    if validated != schema_version:
        msg = f"{record_name} schema_version must be declared explicitly"
        raise ValueError(msg)


__all__ = [
    "FoldingQueueArtifact",
    "FoldingQueueAssignment",
    "FoldingQueueConfig",
    "FoldingQueueLayout",
    "FoldingQueuePlan",
    "FoldingQueueRenderResult",
    "FoldingQueueStrategy",
    "folding_queue_artifact_from_mapping",
    "folding_queue_assignment_from_mapping",
    "folding_queue_config_from_mapping",
    "folding_queue_plan_from_mapping",
    "folding_queue_render_result_from_mapping",
]
