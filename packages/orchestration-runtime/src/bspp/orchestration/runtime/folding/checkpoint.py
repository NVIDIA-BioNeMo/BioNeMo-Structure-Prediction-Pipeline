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

"""Fail-closed folding checkpoint normalization and deterministic merge.

Port Baseline:
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/python/checkpoint_manager.py:36-250``
and
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/python/checkpoint_manager.py:595-653``.

The baseline traverses lexically sorted current per-node shards before legacy
per-GPU shards and lets the last row for an identity win within each event
kind. This port preserves that precedence, sorts the merged public view by
identity, fails on malformed evidence instead of warning and skipping, and
adds the accepted duplicate-safe guardrail that completion dominates failure.
"""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

from bspp.orchestration.contract.folding_checkpoint import (
    CheckpointLayout,
    FoldingCheckpointState,
    FoldingCompletionRecord,
    FoldingFailureRecord,
)

CURRENT_COMPLETION_FIELDS = ("protein_id", "runtime_seconds", "timestamp", "node_id", "gpu_id")
CURRENT_FAILURE_FIELDS = ("protein_id", "error_message", "timestamp", "node_id", "gpu_id")
LEGACY_COMPLETION_FIELDS = ("protein_id", "runtime_seconds", "timestamp")
LEGACY_FAILURE_FIELDS = ("protein_id", "error_message", "timestamp")


def normalize_checkpoint_protein_id(value: str) -> str:
    """Normalize baseline ``AF_<digit>`` prefixes to parquet-style hyphens."""
    if not isinstance(value, str):
        msg = "checkpoint protein_id must be a string"
        raise ValueError(msg)
    return re.sub(r"AF_(\d)", r"AF-\1", value)


def load_checkpoint_state(
    *,
    current_node_completion_shards: tuple[Path, ...] = (),
    current_gpu_completion_shards: tuple[Path, ...] = (),
    legacy_completion_shards: tuple[Path, ...] = (),
    current_node_failure_shards: tuple[Path, ...] = (),
    current_gpu_failure_shards: tuple[Path, ...] = (),
    legacy_failure_shards: tuple[Path, ...] = (),
) -> FoldingCheckpointState:
    """Load explicitly declared shard paths and return one canonical state."""
    completions = _load_completion_groups(
        current_node_completion_shards,
        current_gpu_completion_shards,
        legacy_completion_shards,
    )
    failures = _load_failure_groups(
        current_node_failure_shards,
        current_gpu_failure_shards,
        legacy_failure_shards,
    )
    return merge_checkpoint_records(completions, failures)


def merge_checkpoint_records(
    completions: Iterable[FoldingCompletionRecord],
    failures: Iterable[FoldingFailureRecord],
) -> FoldingCheckpointState:
    """Resolve duplicate events by deterministic source precedence.

    Within completion and failure events independently, the event with the
    greatest ``(source_ordinal, row_number)`` wins. Source ordinals produced by
    :func:`load_checkpoint_state` preserve lexical per-node traversal followed
    by one schema-independent lexical per-GPU group. Completion removes any
    selected failure for the same identity; this is an explicit port guardrail
    over the baseline's two independently written global CSVs.
    """
    completed_by_id = _select_completion_events(completions)
    failed_by_id = _select_failure_events(failures)
    for protein_id in completed_by_id:
        failed_by_id.pop(protein_id, None)
    return FoldingCheckpointState(
        completions=tuple(completed_by_id[protein_id] for protein_id in sorted(completed_by_id)),
        failures=tuple(failed_by_id[protein_id] for protein_id in sorted(failed_by_id)),
    )


def write_merged_checkpoint_views(
    state: FoldingCheckpointState,
    *,
    completed_path: Path,
    failed_path: Path,
) -> tuple[Path, Path]:
    """Replace deterministic current-schema merged CSV views failure-safely.

    Both temporary files and snapshots of existing destinations are fully
    written and fsynced before replacement begins. Each destination
    replacement is atomic. If a later replacement fails, any destination
    already replaced during this call is restored from its snapshot.
    """
    if not isinstance(state, FoldingCheckpointState):
        msg = "state must be a FoldingCheckpointState"
        raise ValueError(msg)
    if not isinstance(completed_path, Path) or not isinstance(failed_path, Path):
        msg = "merged checkpoint destinations must be pathlib.Path values"
        raise ValueError(msg)
    if completed_path.resolve(strict=False) == failed_path.resolve(strict=False):
        msg = "completed_path and failed_path must be distinct"
        raise ValueError(msg)

    temporary_paths: list[Path] = []
    replaced_destinations: list[tuple[Path, Path | None]] = []
    preserved_recovery_paths: set[Path] = set()
    try:
        completed_tmp = _write_csv_temp(
            completed_path,
            fieldnames=CURRENT_COMPLETION_FIELDS,
            rows=(_completion_csv_row(record) for record in state.completions),
        )
        temporary_paths.append(completed_tmp)
        failed_tmp = _write_csv_temp(
            failed_path,
            fieldnames=CURRENT_FAILURE_FIELDS,
            rows=(_failure_csv_row(record) for record in state.failures),
        )
        temporary_paths.append(failed_tmp)
        completed_backup = _snapshot_destination(completed_path)
        if completed_backup is not None:
            temporary_paths.append(completed_backup)
        failed_backup = _snapshot_destination(failed_path)
        if failed_backup is not None:
            temporary_paths.append(failed_backup)
        os.replace(completed_tmp, completed_path)
        replaced_destinations.append((completed_path, completed_backup))
        os.replace(failed_tmp, failed_path)
        replaced_destinations.append((failed_path, failed_backup))
    except BaseException as replacement_error:
        rollback_errors: list[BaseException] = []
        for destination, backup in reversed(replaced_destinations):
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
                if backup is not None:
                    preserved_recovery_paths.add(backup)
        if rollback_errors:
            recovery_summary = ", ".join(str(path) for path in sorted(preserved_recovery_paths)) or "none available"
            msg = f"Merged checkpoint rollback failed; recovery snapshots preserved: {recovery_summary}"
            raise BaseExceptionGroup(msg, [replacement_error, *rollback_errors]) from replacement_error
        raise
    finally:
        for temporary_path in temporary_paths:
            if temporary_path not in preserved_recovery_paths:
                temporary_path.unlink(missing_ok=True)
    return completed_path, failed_path


def _load_completion_groups(
    current_paths: tuple[Path, ...],
    current_gpu_paths: tuple[Path, ...],
    legacy_paths: tuple[Path, ...],
) -> tuple[FoldingCompletionRecord, ...]:
    records: list[FoldingCompletionRecord] = []
    for source_ordinal, (path, layout) in enumerate(_ordered_sources(current_paths, current_gpu_paths, legacy_paths)):
        records.extend(_read_completion_shard(path, layout=layout, source_ordinal=source_ordinal))
    return tuple(records)


def _load_failure_groups(
    current_paths: tuple[Path, ...],
    current_gpu_paths: tuple[Path, ...],
    legacy_paths: tuple[Path, ...],
) -> tuple[FoldingFailureRecord, ...]:
    records: list[FoldingFailureRecord] = []
    for source_ordinal, (path, layout) in enumerate(_ordered_sources(current_paths, current_gpu_paths, legacy_paths)):
        records.extend(_read_failure_shard(path, layout=layout, source_ordinal=source_ordinal))
    return tuple(records)


def _ordered_sources(
    current_paths: tuple[Path, ...],
    current_gpu_paths: tuple[Path, ...],
    legacy_paths: tuple[Path, ...],
) -> tuple[tuple[Path, CheckpointLayout], ...]:
    if (
        not isinstance(current_paths, tuple)
        or not isinstance(current_gpu_paths, tuple)
        or not isinstance(legacy_paths, tuple)
    ):
        msg = "checkpoint shard collections must be explicit immutable tuples"
        raise ValueError(msg)
    ordered: list[tuple[Path, CheckpointLayout]] = []
    ordered.extend((path, "current-per-node") for path in sorted(current_paths, key=str))
    gpu_sources: list[tuple[Path, CheckpointLayout]] = [
        *((path, "current-per-gpu") for path in current_gpu_paths),
        *((path, "legacy-per-gpu") for path in legacy_paths),
    ]
    ordered.extend(sorted(gpu_sources, key=lambda item: str(item[0])))
    seen: set[Path] = set()
    for path, _layout in ordered:
        if not isinstance(path, Path):
            msg = "checkpoint shard paths must be pathlib.Path values"
            raise ValueError(msg)
        try:
            identity = path.resolve(strict=True)
        except OSError as exc:
            msg = f"Cannot resolve checkpoint shard {path}: {exc}"
            raise ValueError(msg) from exc
        if identity in seen:
            msg = f"Checkpoint shard was declared more than once: {path}"
            raise ValueError(msg)
        seen.add(identity)
    return tuple(ordered)


def _read_completion_shard(
    path: Path, *, layout: CheckpointLayout, source_ordinal: int
) -> tuple[FoldingCompletionRecord, ...]:
    fields = LEGACY_COMPLETION_FIELDS if layout == "legacy-per-gpu" else CURRENT_COMPLETION_FIELDS
    rows = _read_csv_rows(path, expected_fields=fields)
    records: list[FoldingCompletionRecord] = []
    for row_number, row in rows:
        try:
            runtime_seconds = float(row[1])
            record_layout, node_id, gpu_id = _parse_record_scope(row, layout=layout)
            records.append(
                FoldingCompletionRecord(
                    protein_id=normalize_checkpoint_protein_id(row[0].strip()),
                    runtime_seconds=runtime_seconds,
                    timestamp=row[2],
                    node_id=node_id,
                    gpu_id=gpu_id,
                    source_layout=record_layout,
                    source_path=str(path),
                    source_ordinal=source_ordinal,
                    row_number=row_number,
                )
            )
        except ValueError as exc:
            msg = f"Invalid completion checkpoint row {path}:{row_number}: {exc}"
            raise ValueError(msg) from exc
    return tuple(records)


def _read_failure_shard(
    path: Path, *, layout: CheckpointLayout, source_ordinal: int
) -> tuple[FoldingFailureRecord, ...]:
    fields = LEGACY_FAILURE_FIELDS if layout == "legacy-per-gpu" else CURRENT_FAILURE_FIELDS
    rows = _read_csv_rows(path, expected_fields=fields)
    records: list[FoldingFailureRecord] = []
    for row_number, row in rows:
        try:
            record_layout, node_id, gpu_id = _parse_record_scope(row, layout=layout)
            records.append(
                FoldingFailureRecord(
                    protein_id=normalize_checkpoint_protein_id(row[0].strip()),
                    error_message=row[1].strip(),
                    timestamp=row[2],
                    node_id=node_id,
                    gpu_id=gpu_id,
                    source_layout=record_layout,
                    source_path=str(path),
                    source_ordinal=source_ordinal,
                    row_number=row_number,
                )
            )
        except ValueError as exc:
            msg = f"Invalid failure checkpoint row {path}:{row_number}: {exc}"
            raise ValueError(msg) from exc
    return tuple(records)


def _read_csv_rows(path: Path, *, expected_fields: tuple[str, ...]) -> tuple[tuple[int, tuple[str, ...]], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            contents = handle.read()
            if not contents.strip():
                return ()
            has_terminal_newline = contents.endswith(("\n", "\r"))
            # The baseline removes LF from error messages but can leave CR.
            # Preserve CRLF record boundaries and normalize only stray CR.
            normalized = contents.replace("\r\n", "\n").replace("\r", " ")
            reader = csv.reader(io.StringIO(normalized, newline=""), strict=False)
            try:
                header = tuple(next(reader))
            except StopIteration as exc:
                msg = f"Checkpoint shard is empty: {path}"
                raise ValueError(msg) from exc
            if header != expected_fields:
                msg = f"Checkpoint shard {path} has header {header!r}; expected {expected_fields!r}"
                raise ValueError(msg)
            raw_rows = list(reader)
            rows: list[tuple[int, tuple[str, ...]]] = []
            for row_index, raw_row in enumerate(raw_rows):
                row_number = row_index + 2
                row = tuple(raw_row)
                if not row:
                    continue
                if len(row) != len(expected_fields):
                    is_torn_final_append = (
                        row_index == len(raw_rows) - 1 and not has_terminal_newline and len(row) < len(expected_fields)
                    )
                    if is_torn_final_append:
                        continue
                    msg = f"Checkpoint shard {path}:{row_number} has {len(row)} fields; expected {len(expected_fields)}"
                    raise ValueError(msg)
                rows.append((row_number, row))
    except (OSError, UnicodeError, csv.Error) as exc:
        msg = f"Cannot read checkpoint shard {path}: {exc}"
        raise ValueError(msg) from exc
    return tuple(rows)


def _parse_gpu_id(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = "gpu_id must be a non-negative integer"
        raise ValueError(msg) from exc
    if parsed < 0 or str(parsed) != value:
        msg = "gpu_id must be a canonical non-negative integer"
        raise ValueError(msg)
    return parsed


def _parse_record_scope(
    row: tuple[str, ...], *, layout: CheckpointLayout
) -> tuple[CheckpointLayout, str | None, int | None]:
    if layout == "legacy-per-gpu":
        return layout, None, None
    node_id = row[3]
    gpu_id = row[4]
    if not node_id and not gpu_id:
        # Merged current-schema views project legacy records as a blank scope.
        return "legacy-per-gpu", None, None
    if not node_id or not gpu_id:
        msg = "node_id and gpu_id must either both be present or both be blank"
        raise ValueError(msg)
    return layout, node_id, _parse_gpu_id(gpu_id)


def _completion_csv_row(record: FoldingCompletionRecord) -> tuple[str, ...]:
    return (
        record.protein_id,
        repr(record.runtime_seconds),
        record.timestamp,
        record.node_id or "",
        "" if record.gpu_id is None else str(record.gpu_id),
    )


def _failure_csv_row(record: FoldingFailureRecord) -> tuple[str, ...]:
    return (
        record.protein_id,
        record.error_message,
        record.timestamp,
        record.node_id or "",
        "" if record.gpu_id is None else str(record.gpu_id),
    )


def _write_csv_temp(
    destination: Path,
    *,
    fieldnames: tuple[str, ...],
    rows: Iterable[tuple[str, ...]],
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\r\n")
            writer.writerow(fieldnames)
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _snapshot_destination(destination: Path) -> Path | None:
    if not destination.exists():
        return None
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.backup.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as backup, destination.open("rb") as source:
            shutil.copyfileobj(source, backup)
            backup.flush()
            os.fsync(backup.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _select_completion_events(
    records: Iterable[FoldingCompletionRecord],
) -> dict[str, FoldingCompletionRecord]:
    selected: dict[str, FoldingCompletionRecord] = {}
    coordinates: dict[tuple[int, int], FoldingCompletionRecord] = {}
    for record in records:
        _reject_ambiguous_coordinate(record, coordinates)
        previous = selected.get(record.protein_id)
        if previous is None or _event_precedence(record) > _event_precedence(previous):
            selected[record.protein_id] = record
    return selected


def _select_failure_events(records: Iterable[FoldingFailureRecord]) -> dict[str, FoldingFailureRecord]:
    selected: dict[str, FoldingFailureRecord] = {}
    coordinates: dict[tuple[int, int], FoldingFailureRecord] = {}
    for record in records:
        _reject_ambiguous_coordinate(record, coordinates)
        previous = selected.get(record.protein_id)
        if previous is None or _event_precedence(record) > _event_precedence(previous):
            selected[record.protein_id] = record
    return selected


def _reject_ambiguous_coordinate[RecordT: (FoldingCompletionRecord, FoldingFailureRecord)](
    record: RecordT,
    coordinates: dict[tuple[int, int], RecordT],
) -> None:
    coordinate = _event_precedence(record)
    previous = coordinates.get(coordinate)
    if previous is not None and previous != record:
        msg = f"Conflicting checkpoint records claim precedence coordinate {record.source_ordinal}:{record.row_number}"
        raise ValueError(msg)
    coordinates[coordinate] = record


def _event_precedence(
    record: FoldingCompletionRecord | FoldingFailureRecord,
) -> tuple[int, int]:
    return (record.source_ordinal, record.row_number)


__all__ = [
    "CURRENT_COMPLETION_FIELDS",
    "CURRENT_FAILURE_FIELDS",
    "LEGACY_COMPLETION_FIELDS",
    "LEGACY_FAILURE_FIELDS",
    "load_checkpoint_state",
    "merge_checkpoint_records",
    "normalize_checkpoint_protein_id",
    "write_merged_checkpoint_views",
]
