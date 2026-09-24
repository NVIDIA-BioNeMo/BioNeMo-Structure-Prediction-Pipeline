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

"""Immutable preprocessing planning contracts.

Port Baseline:
419813dbb5a3949e5e16f289f974d9f95e94bf01:README.md:17-41,
419813dbb5a3949e5e16f289f974d9f95e94bf01:utils/generate_tranches.sh:77-90,
419813dbb5a3949e5e16f289f974d9f95e94bf01:scripts/split_file.sh:3-10, and
419813dbb5a3949e5e16f289f974d9f95e94bf01:scripts/prepare_input_folder_structure.py:44-66.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

FastaNormalizationMode = Literal["strict-two-line", "normalize-multiline"]


@dataclass(frozen=True)
class PreprocessingFastaRecord:
    """One canonical FASTA record in source order."""

    header: str
    sequence: str
    identity: str
    source_ordinal: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingFastaRecord")
        if not self.header.startswith(">") or len(self.header) == 1 or self.header[1].isspace():
            msg = "header must contain a canonical FASTA identity immediately after '>'"
            raise ValueError(msg)
        if not self.sequence or any(character.isspace() for character in self.sequence):
            msg = "sequence must be non-empty and contain no whitespace"
            raise ValueError(msg)
        canonical_identity = self.header[1:].split(maxsplit=1)[0]
        if self.identity != canonical_identity:
            msg = f"identity {self.identity!r} does not match header identity {canonical_identity!r}"
            raise ValueError(msg)
        _validate_nonnegative_int(self.source_ordinal, "source_ordinal")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready record data."""
        return {
            "schema_version": self.schema_version,
            "header": self.header,
            "sequence": self.sequence,
            "identity": self.identity,
            "source_ordinal": self.source_ordinal,
        }


@dataclass(frozen=True)
class PreprocessingInput:
    """Declared and validated preprocessing FASTA input."""

    source_path: str
    normalization_mode: FastaNormalizationMode
    records: tuple[PreprocessingFastaRecord, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingInput")
        if not self.source_path.endswith(".fa"):
            msg = "preprocessing FASTA input must use the .fa suffix"
            raise ValueError(msg)
        if self.normalization_mode not in {"strict-two-line", "normalize-multiline"}:
            msg = f"unsupported FASTA normalization mode: {self.normalization_mode!r}"
            raise ValueError(msg)
        if not isinstance(self.records, tuple):
            msg = "records must be an immutable tuple"
            raise ValueError(msg)
        identities: set[str] = set()
        previous_ordinal = -1
        for record in self.records:
            if record.identity in identities:
                msg = f"duplicate FASTA identity {record.identity!r}"
                raise ValueError(msg)
            if record.source_ordinal <= previous_ordinal:
                msg = "selected FASTA records must retain strictly increasing source ordinals"
                raise ValueError(msg)
            identities.add(record.identity)
            previous_ordinal = record.source_ordinal

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready input data."""
        return {
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "normalization_mode": self.normalization_mode,
            "records": [record.to_mapping() for record in self.records],
        }


@dataclass(frozen=True)
class PreprocessingPlanOptions:
    """Deterministic preprocessing partition and worker options."""

    requested_tranches: int = 1
    records_per_chunk: int = 300
    nodes: int = 1
    gpus_per_node: int = 1
    normalization_mode: FastaNormalizationMode = "strict-two-line"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingPlanOptions")
        if (
            not isinstance(self.requested_tranches, int)
            or isinstance(self.requested_tranches, bool)
            or not 1 <= self.requested_tranches <= 100
        ):
            msg = "requested_tranches must be between 1 and 100"
            raise ValueError(msg)
        if (
            not isinstance(self.records_per_chunk, int)
            or isinstance(self.records_per_chunk, bool)
            or self.records_per_chunk <= 0
        ):
            msg = "records_per_chunk must be positive"
            raise ValueError(msg)
        if not isinstance(self.nodes, int) or isinstance(self.nodes, bool) or self.nodes <= 0:
            msg = "nodes must be positive"
            raise ValueError(msg)
        if not isinstance(self.gpus_per_node, int) or isinstance(self.gpus_per_node, bool) or self.gpus_per_node <= 0:
            msg = "gpus_per_node must be positive"
            raise ValueError(msg)
        if self.normalization_mode not in {"strict-two-line", "normalize-multiline"}:
            msg = f"unsupported FASTA normalization mode: {self.normalization_mode!r}"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready option data."""
        return {
            "schema_version": self.schema_version,
            "requested_tranches": self.requested_tranches,
            "records_per_chunk": self.records_per_chunk,
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "normalization_mode": self.normalization_mode,
        }


@dataclass(frozen=True)
class PreprocessingTranche:
    """One contiguous record-safe FASTA tranche."""

    name: str
    fasta_name: str
    ordinal: int
    record_ordinals: tuple[int, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingTranche")
        _validate_nonnegative_int(self.ordinal, "ordinal")
        if self.ordinal > 99 or self.name != f"tranche{self.ordinal:02d}":
            msg = "tranche name must match its two-digit ordinal"
            raise ValueError(msg)
        if not re.fullmatch(r"[^/]+_tranche\d{2}\.fa", self.fasta_name) or not self.fasta_name.endswith(
            f"_{self.name}.fa"
        ):
            msg = "tranche FASTA name must end with its exact two-digit tranche name"
            raise ValueError(msg)
        _validate_record_ordinals(self.record_ordinals, record_name="PreprocessingTranche")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready tranche data."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "fasta_name": self.fasta_name,
            "ordinal": self.ordinal,
            "record_ordinals": list(self.record_ordinals),
        }


@dataclass(frozen=True)
class PreprocessingChunk:
    """One record-safe chunk within a tranche."""

    name: str
    tranche_name: str
    ordinal: int
    tranche_chunk_ordinal: int
    record_ordinals: tuple[int, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingChunk")
        _validate_nonnegative_int(self.ordinal, "ordinal")
        _validate_nonnegative_int(self.tranche_chunk_ordinal, "tranche_chunk_ordinal")
        if self.tranche_chunk_ordinal > 99_999:
            msg = "tranche_chunk_ordinal exceeds five-digit suffix capacity"
            raise ValueError(msg)
        if not re.fullmatch(r"tranche\d{2}", self.tranche_name):
            msg = "tranche_name must use an exact two-digit suffix"
            raise ValueError(msg)
        expected_suffix = f"_{self.tranche_name}_{self.tranche_chunk_ordinal:05d}.fa"
        if not re.fullmatch(r"[^/]+_tranche\d{2}_\d{5}\.fa", self.name) or not self.name.endswith(expected_suffix):
            msg = "chunk name must match its tranche and five-digit within-tranche ordinal"
            raise ValueError(msg)
        _validate_record_ordinals(self.record_ordinals, record_name="PreprocessingChunk")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready chunk data."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "tranche_name": self.tranche_name,
            "ordinal": self.ordinal,
            "tranche_chunk_ordinal": self.tranche_chunk_ordinal,
            "record_ordinals": list(self.record_ordinals),
        }


@dataclass(frozen=True)
class PreprocessingChunkAssignment:
    """Explicit non-executing worker placement for one chunk copy."""

    chunk_name: str
    global_worker_index: int
    node_index: int
    gpu_index: int
    worker_label: str
    worker_ordinal: int
    source_path: str
    staged_path: str
    staging_operation: Literal["copy"] = "copy"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingChunkAssignment")
        if not re.fullmatch(r"[^/]+_tranche\d{2}_\d{5}\.fa", self.chunk_name):
            msg = "assignment chunk_name must be a canonical five-digit chunk name"
            raise ValueError(msg)
        for field_name, value in (
            ("global_worker_index", self.global_worker_index),
            ("node_index", self.node_index),
            ("gpu_index", self.gpu_index),
            ("worker_ordinal", self.worker_ordinal),
        ):
            _validate_nonnegative_int(value, field_name)
        expected_label = f"n{self.node_index}g{self.gpu_index}"
        if self.worker_label != expected_label:
            msg = "worker_label must match node_index and gpu_index"
            raise ValueError(msg)
        if self.staging_operation != "copy":
            msg = "staging_operation must be 'copy'"
            raise ValueError(msg)
        if self.source_path != f"splitted/{self.chunk_name}":
            msg = "source_path must identify the chunk under splitted/"
            raise ValueError(msg)
        if self.staged_path != f"{self.worker_label}/{self.chunk_name}":
            msg = "staged_path must identify the chunk under its worker label"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready assignment data."""
        return {
            "schema_version": self.schema_version,
            "chunk_name": self.chunk_name,
            "global_worker_index": self.global_worker_index,
            "node_index": self.node_index,
            "gpu_index": self.gpu_index,
            "worker_label": self.worker_label,
            "worker_ordinal": self.worker_ordinal,
            "source_path": self.source_path,
            "staged_path": self.staged_path,
            "staging_operation": self.staging_operation,
        }


@dataclass(frozen=True)
class PreprocessingWorkPlan:
    """Complete non-executing preprocessing work plan."""

    input: PreprocessingInput
    options: PreprocessingPlanOptions
    tranches: tuple[PreprocessingTranche, ...]
    chunks: tuple[PreprocessingChunk, ...]
    assignments: tuple[PreprocessingChunkAssignment, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingWorkPlan")
        _validate_work_plan(self)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready work-plan data."""
        return {
            "schema_version": self.schema_version,
            "input": self.input.to_mapping(),
            "options": self.options.to_mapping(),
            "tranches": [tranche.to_mapping() for tranche in self.tranches],
            "chunks": [chunk.to_mapping() for chunk in self.chunks],
            "assignments": [assignment.to_mapping() for assignment in self.assignments],
        }


def _validate_work_plan(plan: PreprocessingWorkPlan) -> None:
    for field_name, value in (
        ("tranches", plan.tranches),
        ("chunks", plan.chunks),
        ("assignments", plan.assignments),
    ):
        if not isinstance(value, tuple):
            msg = f"{field_name} must be an immutable tuple"
            raise ValueError(msg)
    if plan.input.normalization_mode != plan.options.normalization_mode:
        msg = "input and planning options must use the same normalization mode"
        raise ValueError(msg)

    input_ordinals = tuple(record.source_ordinal for record in plan.input.records)
    if not input_ordinals:
        if plan.tranches or plan.chunks or plan.assignments:
            msg = "an empty preprocessing input must have an empty work plan"
            raise ValueError(msg)
        return

    source_filename = plan.input.source_path.rsplit("/", maxsplit=1)[-1]
    source_stem = source_filename[:-3]
    records_per_tranche = (len(input_ordinals) + plan.options.requested_tranches - 1) // plan.options.requested_tranches
    expected_tranches: list[tuple[str, str, int, tuple[int, ...]]] = []
    expected_chunks: list[tuple[str, str, int, int, tuple[int, ...]]] = []
    for tranche_ordinal, start in enumerate(range(0, len(input_ordinals), records_per_tranche)):
        tranche_ordinals = input_ordinals[start : start + records_per_tranche]
        tranche_name = f"tranche{tranche_ordinal:02d}"
        expected_tranches.append(
            (
                tranche_name,
                f"{source_stem}_{tranche_name}.fa",
                tranche_ordinal,
                tranche_ordinals,
            )
        )
        chunk_count = (len(tranche_ordinals) + plan.options.records_per_chunk - 1) // plan.options.records_per_chunk
        if chunk_count > 100_000:
            msg = f"{tranche_name} exceeds five-digit chunk suffix capacity"
            raise ValueError(msg)
        for tranche_chunk_ordinal, chunk_start in enumerate(
            range(0, len(tranche_ordinals), plan.options.records_per_chunk)
        ):
            expected_chunks.append(
                (
                    f"{source_stem}_{tranche_name}_{tranche_chunk_ordinal:05d}.fa",
                    tranche_name,
                    len(expected_chunks),
                    tranche_chunk_ordinal,
                    tranche_ordinals[chunk_start : chunk_start + plan.options.records_per_chunk],
                )
            )

    observed_tranches = tuple(
        (tranche.name, tranche.fasta_name, tranche.ordinal, tranche.record_ordinals) for tranche in plan.tranches
    )
    if observed_tranches != tuple(expected_tranches):
        msg = "tranches do not exactly partition input records in deterministic tranche order"
        raise ValueError(msg)
    observed_chunks = tuple(
        (
            chunk.name,
            chunk.tranche_name,
            chunk.ordinal,
            chunk.tranche_chunk_ordinal,
            chunk.record_ordinals,
        )
        for chunk in plan.chunks
    )
    if observed_chunks != tuple(expected_chunks):
        msg = "chunks do not exactly partition tranches in deterministic chunk order"
        raise ValueError(msg)
    if len(plan.assignments) != len(plan.chunks):
        msg = "work plan must contain exactly one assignment per chunk"
        raise ValueError(msg)

    total_workers = plan.options.nodes * plan.options.gpus_per_node
    local_ordinal_by_tranche: dict[str, int] = {}
    for chunk, assignment in zip(plan.chunks, plan.assignments, strict=True):
        if assignment.chunk_name != chunk.name:
            msg = "assignments must reference each chunk exactly once in deterministic chunk order"
            raise ValueError(msg)
        local_ordinal = local_ordinal_by_tranche.get(chunk.tranche_name, 0)
        expected_global_worker = local_ordinal % total_workers
        expected_node = expected_global_worker // plan.options.gpus_per_node
        expected_gpu = expected_global_worker % plan.options.gpus_per_node
        expected_worker_ordinal = local_ordinal // total_workers
        if (
            assignment.global_worker_index != expected_global_worker
            or assignment.node_index != expected_node
            or assignment.gpu_index != expected_gpu
            or assignment.worker_label != f"n{expected_node}g{expected_gpu}"
            or assignment.worker_ordinal != expected_worker_ordinal
        ):
            msg = "assignment worker arithmetic must restart and remain exact within each tranche"
            raise ValueError(msg)
        local_ordinal_by_tranche[chunk.tranche_name] = local_ordinal + 1


def _validate_direct_schema_version(schema_version: int, record_name: str) -> None:
    validated = validate_schema_version(schema_version, record_name=record_name)
    if validated != schema_version:
        msg = f"{record_name} schema_version must be declared explicitly"
        raise ValueError(msg)


def _validate_nonnegative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{field_name} must be a non-negative integer"
        raise ValueError(msg)


def _validate_record_ordinals(record_ordinals: tuple[int, ...], *, record_name: str) -> None:
    if not isinstance(record_ordinals, tuple) or not record_ordinals:
        msg = f"{record_name} record_ordinals must be a non-empty immutable tuple"
        raise ValueError(msg)
    previous = -1
    for ordinal in record_ordinals:
        _validate_nonnegative_int(ordinal, "record ordinal")
        if ordinal <= previous:
            msg = f"{record_name} record_ordinals must be strictly increasing"
            raise ValueError(msg)
        previous = ordinal


def preprocessing_fasta_record_from_mapping(payload: Mapping[str, object]) -> PreprocessingFastaRecord:
    """Load one FASTA record, rejecting schema drift and unknown fields."""
    _reject_unknown_fields(
        payload,
        {"schema_version", "header", "sequence", "identity", "source_ordinal"},
        "PreprocessingFastaRecord",
    )
    return PreprocessingFastaRecord(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PreprocessingFastaRecord"),
        header=_required_str(payload, "header"),
        sequence=_required_str(payload, "sequence"),
        identity=_required_str(payload, "identity"),
        source_ordinal=_required_nonnegative_int(payload, "source_ordinal"),
    )


def preprocessing_input_from_mapping(payload: Mapping[str, object]) -> PreprocessingInput:
    """Load preprocessing input, rejecting schema drift and unknown fields."""
    _reject_unknown_fields(
        payload,
        {"schema_version", "source_path", "normalization_mode", "records"},
        "PreprocessingInput",
    )
    return PreprocessingInput(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PreprocessingInput"),
        source_path=_required_str(payload, "source_path"),
        normalization_mode=_normalization_mode(payload, "normalization_mode"),
        records=tuple(
            preprocessing_fasta_record_from_mapping(record) for record in _required_mapping_sequence(payload, "records")
        ),
    )


def preprocessing_plan_options_from_mapping(payload: Mapping[str, object]) -> PreprocessingPlanOptions:
    """Load planning options, rejecting schema drift and unknown fields."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "requested_tranches",
            "records_per_chunk",
            "nodes",
            "gpus_per_node",
            "normalization_mode",
        },
        "PreprocessingPlanOptions",
    )
    return PreprocessingPlanOptions(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PreprocessingPlanOptions"),
        requested_tranches=_required_int(payload, "requested_tranches"),
        records_per_chunk=_required_int(payload, "records_per_chunk"),
        nodes=_required_int(payload, "nodes"),
        gpus_per_node=_required_int(payload, "gpus_per_node"),
        normalization_mode=_normalization_mode(payload, "normalization_mode"),
    )


def preprocessing_tranche_from_mapping(payload: Mapping[str, object]) -> PreprocessingTranche:
    """Load one tranche, rejecting schema drift and unknown fields."""
    _reject_unknown_fields(
        payload,
        {"schema_version", "name", "fasta_name", "ordinal", "record_ordinals"},
        "PreprocessingTranche",
    )
    return PreprocessingTranche(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PreprocessingTranche"),
        name=_required_str(payload, "name"),
        fasta_name=_required_str(payload, "fasta_name"),
        ordinal=_required_nonnegative_int(payload, "ordinal"),
        record_ordinals=_required_nonnegative_int_tuple(payload, "record_ordinals"),
    )


def preprocessing_chunk_from_mapping(payload: Mapping[str, object]) -> PreprocessingChunk:
    """Load one chunk, rejecting schema drift and unknown fields."""
    _reject_unknown_fields(
        payload,
        {"schema_version", "name", "tranche_name", "ordinal", "tranche_chunk_ordinal", "record_ordinals"},
        "PreprocessingChunk",
    )
    return PreprocessingChunk(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PreprocessingChunk"),
        name=_required_str(payload, "name"),
        tranche_name=_required_str(payload, "tranche_name"),
        ordinal=_required_nonnegative_int(payload, "ordinal"),
        tranche_chunk_ordinal=_required_nonnegative_int(payload, "tranche_chunk_ordinal"),
        record_ordinals=_required_nonnegative_int_tuple(payload, "record_ordinals"),
    )


def preprocessing_chunk_assignment_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingChunkAssignment:
    """Load one chunk assignment, rejecting schema drift and unknown fields."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "chunk_name",
            "global_worker_index",
            "node_index",
            "gpu_index",
            "worker_label",
            "worker_ordinal",
            "source_path",
            "staged_path",
            "staging_operation",
        },
        "PreprocessingChunkAssignment",
    )
    staging_operation = _required_str(payload, "staging_operation")
    if staging_operation != "copy":
        msg = "staging_operation must be 'copy'"
        raise ValueError(msg)
    return PreprocessingChunkAssignment(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingChunkAssignment"
        ),
        chunk_name=_required_str(payload, "chunk_name"),
        global_worker_index=_required_nonnegative_int(payload, "global_worker_index"),
        node_index=_required_nonnegative_int(payload, "node_index"),
        gpu_index=_required_nonnegative_int(payload, "gpu_index"),
        worker_label=_required_str(payload, "worker_label"),
        worker_ordinal=_required_nonnegative_int(payload, "worker_ordinal"),
        source_path=_required_str(payload, "source_path"),
        staged_path=_required_str(payload, "staged_path"),
        staging_operation="copy",
    )


def preprocessing_work_plan_from_mapping(payload: Mapping[str, object]) -> PreprocessingWorkPlan:
    """Load a complete work plan, rejecting schema drift and unknown fields recursively."""
    _reject_unknown_fields(
        payload,
        {"schema_version", "input", "options", "tranches", "chunks", "assignments"},
        "PreprocessingWorkPlan",
    )
    return PreprocessingWorkPlan(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PreprocessingWorkPlan"),
        input=preprocessing_input_from_mapping(_required_mapping(payload, "input")),
        options=preprocessing_plan_options_from_mapping(_required_mapping(payload, "options")),
        tranches=tuple(
            preprocessing_tranche_from_mapping(tranche) for tranche in _required_mapping_sequence(payload, "tranches")
        ),
        chunks=tuple(
            preprocessing_chunk_from_mapping(chunk) for chunk in _required_mapping_sequence(payload, "chunks")
        ),
        assignments=tuple(
            preprocessing_chunk_assignment_from_mapping(assignment)
            for assignment in _required_mapping_sequence(payload, "assignments")
        ),
    )


def _reject_unknown_fields(payload: Mapping[str, object], allowed: set[str], record_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        msg = f"Unknown {record_name} field(s): {', '.join(unknown)}"
        raise ValueError(msg)


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be a mapping"
        raise ValueError(msg)
    return value


def _required_mapping_sequence(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    mappings: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"{key}[{index}] must be a mapping"
            raise ValueError(msg)
        mappings.append(item)
    return tuple(mappings)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{key} must be a non-empty string"
        raise ValueError(msg)
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


def _required_nonnegative_int(payload: Mapping[str, object], key: str) -> int:
    value = _required_int(payload, key)
    if value < 0:
        msg = f"{key} must be non-negative"
        raise ValueError(msg)
    return value


def _required_nonnegative_int_tuple(payload: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    result: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            msg = f"{key}[{index}] must be a non-negative integer"
            raise ValueError(msg)
        result.append(item)
    return tuple(result)


def _normalization_mode(payload: Mapping[str, object], key: str) -> FastaNormalizationMode:
    value = _required_str(payload, key)
    if value not in {"strict-two-line", "normalize-multiline"}:
        msg = f"unsupported FASTA normalization mode: {value!r}"
        raise ValueError(msg)
    return cast("FastaNormalizationMode", value)


__all__ = [
    "FastaNormalizationMode",
    "PreprocessingChunk",
    "PreprocessingChunkAssignment",
    "PreprocessingFastaRecord",
    "PreprocessingInput",
    "PreprocessingPlanOptions",
    "PreprocessingTranche",
    "PreprocessingWorkPlan",
    "preprocessing_chunk_assignment_from_mapping",
    "preprocessing_chunk_from_mapping",
    "preprocessing_fasta_record_from_mapping",
    "preprocessing_input_from_mapping",
    "preprocessing_plan_options_from_mapping",
    "preprocessing_tranche_from_mapping",
    "preprocessing_work_plan_from_mapping",
]
