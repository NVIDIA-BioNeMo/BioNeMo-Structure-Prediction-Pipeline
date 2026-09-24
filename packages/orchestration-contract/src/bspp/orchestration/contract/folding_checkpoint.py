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

"""Immutable folding checkpoint and resume records.

The records support behavior derived from Port Baseline sources
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/python/checkpoint_manager.py:36-250``
and
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/python/checkpoint_manager.py:488-653``.
CSV parsing, merging, and local writes remain in the runtime distribution.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

CheckpointLayout = Literal["current-per-node", "current-per-gpu", "legacy-per-gpu"]
FailureDisposition = Literal["transient", "permanent"]

_COMPLETION_FIELDS = frozenset(
    {
        "schema_version",
        "protein_id",
        "runtime_seconds",
        "timestamp",
        "node_id",
        "gpu_id",
        "source_layout",
        "source_path",
        "source_ordinal",
        "row_number",
    }
)
_FAILURE_FIELDS = frozenset(
    {
        "schema_version",
        "protein_id",
        "error_message",
        "timestamp",
        "node_id",
        "gpu_id",
        "source_layout",
        "source_path",
        "source_ordinal",
        "row_number",
    }
)
_STATE_FIELDS = frozenset({"schema_version", "completions", "failures"})
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "planned_model_ids",
        "completed_model_ids",
        "transient_failure_ids",
        "permanent_failure_ids",
        "remaining_model_ids",
        "unplanned_checkpoint_ids",
        "retry_failed",
    }
)


@dataclass(frozen=True)
class FoldingCompletionRecord:
    """One canonical current or legacy folding completion event."""

    protein_id: str
    runtime_seconds: float
    timestamp: str
    node_id: str | None
    gpu_id: int | None
    source_layout: CheckpointLayout
    source_path: str
    source_ordinal: int
    row_number: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingCompletionRecord")
        _validate_protein_id(self.protein_id)
        _validate_nonnegative_float(self.runtime_seconds, "runtime_seconds")
        _validate_nonempty_text(self.timestamp, "timestamp")
        _validate_layout_scope(self.source_layout, self.node_id, self.gpu_id)
        _validate_nonempty_text(self.source_path, "source_path")
        _validate_nonnegative_int(self.source_ordinal, "source_ordinal")
        _validate_row_number(self.row_number)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/table-ready data."""
        return {
            "schema_version": self.schema_version,
            "protein_id": self.protein_id,
            "runtime_seconds": self.runtime_seconds,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "gpu_id": self.gpu_id,
            "source_layout": self.source_layout,
            "source_path": self.source_path,
            "source_ordinal": self.source_ordinal,
            "row_number": self.row_number,
        }


@dataclass(frozen=True)
class FoldingFailureRecord:
    """One canonical current or legacy folding failure event."""

    protein_id: str
    error_message: str
    timestamp: str
    node_id: str | None
    gpu_id: int | None
    source_layout: CheckpointLayout
    source_path: str
    source_ordinal: int
    row_number: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingFailureRecord")
        _validate_protein_id(self.protein_id)
        _validate_error_message(self.error_message)
        _validate_nonempty_text(self.timestamp, "timestamp")
        _validate_layout_scope(self.source_layout, self.node_id, self.gpu_id)
        _validate_nonempty_text(self.source_path, "source_path")
        _validate_nonnegative_int(self.source_ordinal, "source_ordinal")
        _validate_row_number(self.row_number)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/table-ready data."""
        return {
            "schema_version": self.schema_version,
            "protein_id": self.protein_id,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "gpu_id": self.gpu_id,
            "source_layout": self.source_layout,
            "source_path": self.source_path,
            "source_ordinal": self.source_ordinal,
            "row_number": self.row_number,
        }


@dataclass(frozen=True)
class FoldingCheckpointState:
    """Stable completion-dominant merged checkpoint state."""

    completions: tuple[FoldingCompletionRecord, ...]
    failures: tuple[FoldingFailureRecord, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingCheckpointState")
        _validate_record_tuple(self.completions, FoldingCompletionRecord, "completions")
        _validate_record_tuple(self.failures, FoldingFailureRecord, "failures")
        _validate_strictly_sorted_ids(self.completions, "completions")
        _validate_strictly_sorted_ids(self.failures, "failures")
        if any(record.schema_version != self.schema_version for record in self.completions) or any(
            record.schema_version != self.schema_version for record in self.failures
        ):
            msg = "checkpoint record schema versions must match the state schema version"
            raise ValueError(msg)
        completed_ids = {record.protein_id for record in self.completions}
        conflicting_ids = sorted(completed_ids & {record.protein_id for record in self.failures})
        if conflicting_ids:
            msg = "model identities cannot be both completed and failed: " + ", ".join(conflicting_ids)
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-ready merged state."""
        return {
            "schema_version": self.schema_version,
            "completions": [record.to_mapping() for record in self.completions],
            "failures": [record.to_mapping() for record in self.failures],
        }


@dataclass(frozen=True)
class FoldingRemainingWorkPlan:
    """Checkpoint-based remaining work in declared model order."""

    planned_model_ids: tuple[str, ...]
    completed_model_ids: tuple[str, ...]
    transient_failure_ids: tuple[str, ...]
    permanent_failure_ids: tuple[str, ...]
    remaining_model_ids: tuple[str, ...]
    unplanned_checkpoint_ids: tuple[str, ...]
    retry_failed: bool
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingRemainingWorkPlan")
        _validate_identity_tuple(self.planned_model_ids, "planned_model_ids")
        _validate_identity_tuple(self.completed_model_ids, "completed_model_ids")
        _validate_identity_tuple(self.transient_failure_ids, "transient_failure_ids")
        _validate_identity_tuple(self.permanent_failure_ids, "permanent_failure_ids")
        _validate_identity_tuple(self.remaining_model_ids, "remaining_model_ids")
        _validate_identity_tuple(self.unplanned_checkpoint_ids, "unplanned_checkpoint_ids")
        if not isinstance(self.retry_failed, bool):
            msg = "retry_failed must be a boolean"
            raise ValueError(msg)

        planned_set = set(self.planned_model_ids)
        _validate_planned_subsequence(self.completed_model_ids, self.planned_model_ids, "completed_model_ids")
        _validate_planned_subsequence(self.transient_failure_ids, self.planned_model_ids, "transient_failure_ids")
        _validate_planned_subsequence(self.permanent_failure_ids, self.planned_model_ids, "permanent_failure_ids")
        _validate_planned_subsequence(self.remaining_model_ids, self.planned_model_ids, "remaining_model_ids")
        completed = set(self.completed_model_ids)
        transient = set(self.transient_failure_ids)
        permanent = set(self.permanent_failure_ids)
        if completed & (transient | permanent) or transient & permanent:
            msg = "completed, transient-failure, and permanent-failure identities must be disjoint"
            raise ValueError(msg)
        if set(self.unplanned_checkpoint_ids) & planned_set:
            msg = "unplanned_checkpoint_ids cannot contain planned model identities"
            raise ValueError(msg)
        if self.unplanned_checkpoint_ids != tuple(sorted(self.unplanned_checkpoint_ids)):
            msg = "unplanned_checkpoint_ids must be strictly sorted"
            raise ValueError(msg)

        expected_remaining = tuple(
            protein_id
            for protein_id in self.planned_model_ids
            if protein_id not in completed and (self.retry_failed or protein_id not in permanent)
        )
        if self.remaining_model_ids != expected_remaining:
            msg = "remaining_model_ids do not match completion and retry-failed policy"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-ready resume-plan data."""
        return {
            "schema_version": self.schema_version,
            "planned_model_ids": list(self.planned_model_ids),
            "completed_model_ids": list(self.completed_model_ids),
            "transient_failure_ids": list(self.transient_failure_ids),
            "permanent_failure_ids": list(self.permanent_failure_ids),
            "remaining_model_ids": list(self.remaining_model_ids),
            "unplanned_checkpoint_ids": list(self.unplanned_checkpoint_ids),
            "retry_failed": self.retry_failed,
        }


def folding_completion_record_from_mapping(payload: Mapping[str, object]) -> FoldingCompletionRecord:
    """Load one completion event and reject unknown or malformed fields."""
    _reject_unknown_fields(payload, _COMPLETION_FIELDS, record_name="FoldingCompletionRecord")
    return FoldingCompletionRecord(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="FoldingCompletionRecord"),
        protein_id=_required_str(payload, "protein_id"),
        runtime_seconds=_required_float(payload, "runtime_seconds"),
        timestamp=_required_str(payload, "timestamp"),
        node_id=_optional_str(payload, "node_id"),
        gpu_id=_optional_int(payload, "gpu_id"),
        source_layout=_required_layout(payload, "source_layout"),
        source_path=_required_str(payload, "source_path"),
        source_ordinal=_required_int(payload, "source_ordinal"),
        row_number=_required_int(payload, "row_number"),
    )


def folding_failure_record_from_mapping(payload: Mapping[str, object]) -> FoldingFailureRecord:
    """Load one failure event and reject unknown or malformed fields."""
    _reject_unknown_fields(payload, _FAILURE_FIELDS, record_name="FoldingFailureRecord")
    return FoldingFailureRecord(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="FoldingFailureRecord"),
        protein_id=_required_str(payload, "protein_id"),
        error_message=_required_str(payload, "error_message"),
        timestamp=_required_str(payload, "timestamp"),
        node_id=_optional_str(payload, "node_id"),
        gpu_id=_optional_int(payload, "gpu_id"),
        source_layout=_required_layout(payload, "source_layout"),
        source_path=_required_str(payload, "source_path"),
        source_ordinal=_required_int(payload, "source_ordinal"),
        row_number=_required_int(payload, "row_number"),
    )


def folding_checkpoint_state_from_mapping(payload: Mapping[str, object]) -> FoldingCheckpointState:
    """Load a complete merged state through the fail-closed record loaders."""
    _reject_unknown_fields(payload, _STATE_FIELDS, record_name="FoldingCheckpointState")
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="FoldingCheckpointState")
    return FoldingCheckpointState(
        schema_version=schema_version,
        completions=_required_record_tuple(payload, "completions", folding_completion_record_from_mapping),
        failures=_required_record_tuple(payload, "failures", folding_failure_record_from_mapping),
    )


def folding_remaining_work_plan_from_mapping(payload: Mapping[str, object]) -> FoldingRemainingWorkPlan:
    """Load a complete remaining-work plan and reject altered derived fields."""
    _reject_unknown_fields(payload, _PLAN_FIELDS, record_name="FoldingRemainingWorkPlan")
    return FoldingRemainingWorkPlan(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="FoldingRemainingWorkPlan"),
        planned_model_ids=_required_str_tuple(payload, "planned_model_ids"),
        completed_model_ids=_required_str_tuple(payload, "completed_model_ids"),
        transient_failure_ids=_required_str_tuple(payload, "transient_failure_ids"),
        permanent_failure_ids=_required_str_tuple(payload, "permanent_failure_ids"),
        remaining_model_ids=_required_str_tuple(payload, "remaining_model_ids"),
        unplanned_checkpoint_ids=_required_str_tuple(payload, "unplanned_checkpoint_ids"),
        retry_failed=_required_bool(payload, "retry_failed"),
    )


def _required_record_tuple[RecordT](
    payload: Mapping[str, object],
    key: str,
    loader: Callable[[Mapping[str, object]], RecordT],
) -> tuple[RecordT, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list of checkpoint records"
        raise ValueError(msg)
    records: list[RecordT] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"{key}[{index}] must be an object"
            raise ValueError(msg)
        records.append(loader(item))
    return tuple(records)


def _validate_record_tuple(value: object, record_type: type[object], field_name: str) -> None:
    if not isinstance(value, tuple) or not all(isinstance(record, record_type) for record in value):
        msg = f"{field_name} must be an immutable tuple of {record_type.__name__} values"
        raise ValueError(msg)


def _validate_strictly_sorted_ids(
    records: tuple[FoldingCompletionRecord, ...] | tuple[FoldingFailureRecord, ...],
    field_name: str,
) -> None:
    identities = tuple(record.protein_id for record in records)
    if identities != tuple(sorted(set(identities))):
        msg = f"{field_name} must be strictly sorted by unique protein_id"
        raise ValueError(msg)


def _validate_identity_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        msg = f"{field_name} must be an immutable tuple"
        raise ValueError(msg)
    seen: set[str] = set()
    for protein_id in value:
        _validate_protein_id(protein_id)
        if protein_id in seen:
            msg = f"{field_name} contains duplicate model identity {protein_id!r}"
            raise ValueError(msg)
        seen.add(protein_id)


def _validate_planned_subsequence(value: tuple[str, ...], planned: tuple[str, ...], field_name: str) -> None:
    selected = set(value)
    if value != tuple(protein_id for protein_id in planned if protein_id in selected):
        msg = f"{field_name} must be a declared-order subset of planned_model_ids"
        raise ValueError(msg)


def _validate_layout_scope(layout: object, node_id: object, gpu_id: object) -> None:
    if layout in {"current-per-node", "current-per-gpu"}:
        _validate_nonempty_text(node_id, "node_id")
        _validate_nonnegative_int(gpu_id, "gpu_id")
        return
    if layout == "legacy-per-gpu":
        if node_id is not None or gpu_id is not None:
            msg = "legacy-per-gpu records must use null node_id and gpu_id because those fields are absent"
            raise ValueError(msg)
        return
    msg = f"unsupported checkpoint source_layout: {layout!r}"
    raise ValueError(msg)


def _validate_protein_id(value: object) -> None:
    _validate_nonempty_text(value, "protein_id")
    if not isinstance(value, str) or any(character.isspace() or character == "," for character in value):
        msg = "protein_id must contain no whitespace or commas"
        raise ValueError(msg)


def _validate_error_message(value: object) -> None:
    if not isinstance(value, str) or value != value.strip() or "\n" in value or "\r" in value:
        msg = "error_message must be a trimmed single-line string"
        raise ValueError(msg)


def _validate_nonempty_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip() or "\n" in value or "\r" in value:
        msg = f"{field_name} must be a non-empty trimmed single-line string"
        raise ValueError(msg)


def _validate_nonnegative_float(value: object, field_name: str) -> None:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        msg = f"{field_name} must be a finite non-negative number"
        raise ValueError(msg)


def _validate_nonnegative_int(value: object, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{field_name} must be a non-negative integer"
        raise ValueError(msg)


def _validate_row_number(value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 2:
        msg = "row_number must identify a data row at line 2 or later"
        raise ValueError(msg)


def _validate_direct_schema_version(schema_version: int, record_name: str) -> None:
    validated = validate_schema_version(schema_version, record_name=record_name)
    if validated != schema_version:
        msg = f"{record_name} schema_version must be declared explicitly"
        raise ValueError(msg)


def _reject_unknown_fields(payload: Mapping[str, object], allowed: frozenset[str], *, record_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        msg = f"{record_name} has unknown fields: {', '.join(unknown)}"
        raise ValueError(msg)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)
    return value


def _required_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        msg = f"{key} must be a number"
        raise ValueError(msg)
    return float(value)


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        msg = f"{key} must be a string or null"
        raise ValueError(msg)
    return value


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        msg = f"{key} must be an integer or null"
        raise ValueError(msg)
    return value


def _required_layout(payload: Mapping[str, object], key: str) -> CheckpointLayout:
    value = payload.get(key)
    if value == "current-per-node" or value == "current-per-gpu" or value == "legacy-per-gpu":
        return value
    msg = f"{key} must be current-per-node, current-per-gpu, or legacy-per-gpu"
    raise ValueError(msg)


def _required_str_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple) or not all(isinstance(item, str) for item in value):
        msg = f"{key} must be a list of strings"
        raise ValueError(msg)
    return tuple(value)


def _required_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean"
        raise ValueError(msg)
    return value


__all__ = [
    "CheckpointLayout",
    "FailureDisposition",
    "FoldingCheckpointState",
    "FoldingCompletionRecord",
    "FoldingFailureRecord",
    "FoldingRemainingWorkPlan",
    "folding_checkpoint_state_from_mapping",
    "folding_completion_record_from_mapping",
    "folding_failure_record_from_mapping",
    "folding_remaining_work_plan_from_mapping",
]
