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

"""Batch resume markers and failure-record state for the native worker."""

from __future__ import annotations

import fcntl
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FailedModelRecord:
    """One row in ``failed_models.tsv``."""

    model_id: str
    stage: str
    reason: str

    def to_tsv_line(self) -> str:
        """Return the legacy TSV row representation."""

        return f"{self.model_id}\t{self.stage}\t{self.reason}\n"


@dataclass(frozen=True, slots=True)
class FailedModelWriteResult:
    """Accounting for failure rows appended to global and shard files."""

    global_failed_path: Path
    shard_failed_path: Path
    row_count: int


def batch_marker_name(batch_index: int, *, failed_model_count: int = 0, retry_failed_only: bool = False) -> str:
    """Return the legacy batch marker filename."""

    _require_non_negative_int("batch_index", batch_index)
    prefix = "retry_failed_batch" if retry_failed_only else "batch"
    suffix = "partial_uploaded" if failed_model_count else "done"
    return f".{prefix}_{batch_index}_{suffix}"


def should_skip_batch(
    done_marker: Path,
    *,
    retry_failed_only: bool = False,
    local_tar_restart_requires_rerun: bool = False,
) -> bool:
    """Return whether a batch should be skipped because its done marker exists."""

    return bool(done_marker.exists() and not retry_failed_only and not local_tar_restart_requires_rerun)


def write_batch_marker(
    shard_dir: Path,
    batch_index: int,
    uploaded_files: Iterable[str | Path],
    *,
    failed_model_count: int = 0,
    retry_failed_only: bool = False,
) -> Path:
    """Write a batch completion or partial-upload marker."""

    marker = shard_dir / batch_marker_name(
        batch_index,
        failed_model_count=failed_model_count,
        retry_failed_only=retry_failed_only,
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(path) for path in uploaded_files if str(path)]
    marker.write_text("\n".join(lines) + ("\n" if lines else ""))
    return marker


def parse_failed_model_reasons(content: str) -> dict[str, str]:
    """Parse failed model TSV content into ``model_id -> compact reason``."""

    reasons: dict[str, list[str]] = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        model_id = parts[0].strip()
        if not model_id:
            continue
        if len(parts) >= 3:
            reason = f"{parts[1]}: {parts[2]}"
        elif len(parts) == 2:
            reason = parts[1]
        else:
            reason = "model failure"
        seen = reasons.setdefault(model_id, [])
        if reason not in seen:
            seen.append(reason)
    return {model_id: "; ".join(values) for model_id, values in reasons.items()}


def append_failed_model_records(
    records: Iterable[FailedModelRecord],
    *,
    global_failed_path: Path,
    shard_failed_path: Path,
    lock_path: Path | None = None,
) -> FailedModelWriteResult:
    """Append failure rows to global and shard TSVs.

    The global append is protected with a flock lock so concurrent workers do
    not interleave rows. The shard file is task-owned but is written while the
    same lock is held to keep the two files in step.
    """

    lines = tuple(record.to_tsv_line() for record in records)
    if not lines:
        return FailedModelWriteResult(global_failed_path, shard_failed_path, 0)

    global_failed_path.parent.mkdir(parents=True, exist_ok=True)
    shard_failed_path.parent.mkdir(parents=True, exist_ok=True)
    actual_lock_path = lock_path or global_failed_path.with_suffix(f"{global_failed_path.suffix}.lock")
    actual_lock_path.parent.mkdir(parents=True, exist_ok=True)

    with _exclusive_lock(actual_lock_path):
        with global_failed_path.open("a", encoding="utf-8") as handle:
            handle.writelines(lines)
        with shard_failed_path.open("a", encoding="utf-8") as handle:
            handle.writelines(lines)

    return FailedModelWriteResult(global_failed_path, shard_failed_path, len(lines))


def record_batch_failed(
    model_ids: Iterable[str],
    *,
    reason: str,
    global_failed_path: Path,
    shard_failed_path: Path,
    lock_path: Path | None = None,
) -> FailedModelWriteResult:
    """Record every model in a crashed batch as ``batch_crash``."""

    return append_failed_model_records(
        (FailedModelRecord(model_id=model_id, stage="batch_crash", reason=reason) for model_id in model_ids),
        global_failed_path=global_failed_path,
        shard_failed_path=shard_failed_path,
        lock_path=lock_path,
    )


def record_prefiltered_failed(
    bad_models: Iterable[tuple[str, str]],
    *,
    global_failed_path: Path,
    shard_failed_path: Path,
    lock_path: Path | None = None,
) -> FailedModelWriteResult:
    """Record input-validation failures discovered before pipeline execution."""

    return append_failed_model_records(
        (
            FailedModelRecord(model_id=model_id, stage="input_validation", reason=reason)
            for model_id, reason in bad_models
        ),
        global_failed_path=global_failed_path,
        shard_failed_path=shard_failed_path,
        lock_path=lock_path,
    )


def flush_local_failed_models(
    local_failed_path: Path,
    *,
    global_failed_path: Path,
    shard_failed_path: Path,
    lock_path: Path | None = None,
) -> dict[str, str]:
    """Move a pipeline-local ``failed_models.tsv`` into global and shard files."""

    if not local_failed_path.exists() or local_failed_path.stat().st_size == 0:
        return {}
    content = local_failed_path.read_text(encoding="utf-8")
    records = tuple(_records_from_tsv(content))
    append_failed_model_records(
        records,
        global_failed_path=global_failed_path,
        shard_failed_path=shard_failed_path,
        lock_path=lock_path,
    )
    local_failed_path.unlink()
    return parse_failed_model_reasons(content)


def load_failed_model_ids(failed_path: Path) -> frozenset[str]:
    """Load unique model IDs from a failure TSV."""

    if not failed_path.exists():
        return frozenset()
    return frozenset(
        line.split("\t", 1)[0]
        for line in failed_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.split("\t", 1)[0]
    )


def remove_shard_failed_file_if_clean(shard_failed_path: Path, *, had_failures: bool) -> bool:
    """Remove an obsolete per-shard failure file when all batches succeeded."""

    if had_failures or not shard_failed_path.exists():
        return False
    shard_failed_path.unlink()
    return True


def _records_from_tsv(content: str) -> Iterator[FailedModelRecord]:
    for line in content.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        model_id = parts[0].strip()
        if not model_id:
            continue
        stage = parts[1].strip() if len(parts) >= 2 else "model_failure"
        reason = parts[2].strip() if len(parts) >= 3 else ""
        yield FailedModelRecord(model_id=model_id, stage=stage, reason=reason)


@contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _require_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{name} must be a non-negative integer, got {value!r}"
        raise ValueError(msg)


__all__ = [
    "FailedModelRecord",
    "FailedModelWriteResult",
    "append_failed_model_records",
    "batch_marker_name",
    "flush_local_failed_models",
    "load_failed_model_ids",
    "parse_failed_model_reasons",
    "record_batch_failed",
    "record_prefiltered_failed",
    "remove_shard_failed_file_if_clean",
    "should_skip_batch",
    "write_batch_marker",
]
