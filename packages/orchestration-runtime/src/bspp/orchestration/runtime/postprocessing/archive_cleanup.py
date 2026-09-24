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

"""Archive cleanup planning and execution for archive-mode self-upload runs.

Removes compressed ``.tar.lz4`` archives from the RunSpec staging directory
only after the corresponding archive task has fully processed and self-uploaded
to S3, gated on the phase-era worker's upload-confirmation evidence.

The per-shard ``.uploaded`` marker is the canonical upload record; its
``status == "uploaded"`` semantics are shared with
:func:`bspp.orchestration.runtime.worker.upload.uploaded_marker_is_complete`.
Because a marker is written even when a batch upload fell back to Lustre or a
metadata upload silently failed, the marker alone is not sufficient proof: this
module additionally requires an exact batch-completion count and a non-empty
``metadata_files`` list before any archive is eligible for deletion.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.runtime.inputs.reports import write_json_report
from bspp.orchestration.runtime.postprocessing.shard_config import load_shard_config
from bspp.orchestration.runtime.worker.archive_plan import archive_active_lock, archive_active_marker_name

__all__ = [
    "ArchiveCleanupItem",
    "ArchiveCleanupReport",
    "load_archive_names",
    "logical_shard_ids_for_archive",
    "plan_archive_cleanup",
    "render_archive_cleanup_tsv",
    "write_archive_cleanup_report",
]

_REPORT_JSON_NAME = "archive_cleanup_report.json"
_REPORT_TSV_NAME = "archive_cleanup.tsv"


@dataclass(frozen=True)
class ArchiveCleanupItem:
    """One per-archive row of cleanup evaluation."""

    archive_name: str
    archive_task_id: int
    logical_shards: tuple[int, ...]
    archive_path: Path
    archive_size: int
    expected_model_count: int | None
    uploaded_model_count: int
    expected_s3_prefix: str
    uploaded_s3_prefixes: tuple[str, ...]
    status: str
    action: str
    reason: str

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable item data without secrets."""
        return {
            "archive_name": self.archive_name,
            "archive_task_id": self.archive_task_id,
            "logical_shards": list(self.logical_shards),
            "archive_path": str(self.archive_path),
            "archive_size": self.archive_size,
            "expected_model_count": self.expected_model_count,
            "uploaded_model_count": self.uploaded_model_count,
            "expected_s3_prefix": self.expected_s3_prefix,
            "uploaded_s3_prefixes": list(self.uploaded_s3_prefixes),
            "status": self.status,
            "action": self.action,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ArchiveCleanupReport:
    """Aggregate cleanup plan or result for one archive-mode RunSpec."""

    staging_dir: Path
    output_dir: Path
    dataset: str
    execute: bool
    expected_s3_prefix: str
    items: tuple[ArchiveCleanupItem, ...]

    @property
    def delete_ready_count(self) -> int:
        """Return the number of archives eligible for deletion."""
        return sum(1 for item in self.items if item.status == "delete_ready")

    @property
    def deleted_count(self) -> int:
        """Return the number of archives deleted during this execution."""
        return sum(1 for item in self.items if item.status == "deleted")

    @property
    def kept_count(self) -> int:
        """Return the number of archives conservatively kept."""
        return sum(1 for item in self.items if item.status.startswith("keep_"))

    @property
    def missing_count(self) -> int:
        """Return the number of archives already absent from staging."""
        return sum(1 for item in self.items if item.status == "missing_archive")

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable report data without secrets."""
        return {
            "staging_dir": str(self.staging_dir),
            "output_dir": str(self.output_dir),
            "dataset": self.dataset,
            "execute": self.execute,
            "expected_s3_prefix": self.expected_s3_prefix,
            "delete_ready_count": self.delete_ready_count,
            "deleted_count": self.deleted_count,
            "kept_count": self.kept_count,
            "missing_count": self.missing_count,
            "items": [item.to_redacted_dict() for item in self.items],
        }


def load_archive_names(output_dir: Path) -> tuple[str, ...]:
    """Read and validate ``archive_list.json`` as a JSON list of strings."""
    path = output_dir / "archive_list.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        msg = f"Archive list must be a JSON list of strings: {path}"
        raise ValueError(msg)
    return tuple(data)


def logical_shard_ids_for_archive(archive_idx: int, shards_per_archive: int) -> tuple[int, ...]:
    """Return the global logical shard IDs owned by one archive task."""
    if archive_idx < 0:
        msg = f"archive_idx must be non-negative, got {archive_idx}"
        raise ValueError(msg)
    if shards_per_archive <= 0:
        msg = f"shards_per_archive must be positive, got {shards_per_archive}"
        raise ValueError(msg)
    return tuple(archive_idx * shards_per_archive + sub_shard_idx for sub_shard_idx in range(shards_per_archive))


def plan_archive_cleanup(
    spec: RunSpec,
    *,
    expected_s3_prefix: str | None = None,
    execute: bool = False,
) -> ArchiveCleanupReport:
    """Plan or execute archive deletion for a self-uploaded archive-mode run.

    Raises:
        ValueError: If a precondition is violated (non-self-upload run,
            ``archive_source`` other than ``tracking``, missing or inconsistent
            ``archive_list.json``/``shard_config.json``, empty expected prefix).
    """
    if not spec.worker.self_upload:
        msg = "archive cleanup requires worker.self_upload (no S3-confirmed per-shard marker otherwise)"
        raise ValueError(msg)
    if spec.dataset.archive_source != "tracking":
        msg = f"archive cleanup requires dataset.archive_source 'tracking', got {spec.dataset.archive_source!r}"
        raise ValueError(msg)

    archive_names = load_archive_names(spec.paths.output_dir)
    shard_config = load_shard_config(spec.paths.output_dir / "shard_config.json")
    if shard_config.archive_mode is not True:
        msg = f"archive cleanup requires shard_config.archive_mode=true: {spec.paths.output_dir / 'shard_config.json'}"
        raise ValueError(msg)
    if shard_config.shards_per_archive is None:
        msg = "archive cleanup requires shard_config.shards_per_archive"
        raise ValueError(msg)
    if shard_config.shards_per_archive != spec.worker.shards_per_archive:
        msg = (
            f"shard_config.shards_per_archive={shard_config.shards_per_archive} "
            f"does not match worker.shards_per_archive={spec.worker.shards_per_archive}"
        )
        raise ValueError(msg)
    if shard_config.total_archives is not None and shard_config.total_archives != len(archive_names):
        msg = (
            f"shard_config.total_archives={shard_config.total_archives} "
            f"does not match archive_list.json length={len(archive_names)}"
        )
        raise ValueError(msg)

    resolved_prefix = expected_s3_prefix if expected_s3_prefix is not None else spec.storage.s3_output_prefix
    normalized_prefix = resolved_prefix.strip().rstrip("/")
    if not normalized_prefix:
        msg = "expected S3 prefix must be non-empty"
        raise ValueError(msg)

    shards_per_archive = shard_config.shards_per_archive
    items: list[ArchiveCleanupItem] = []
    for archive_task_id, archive_name in enumerate(archive_names):
        logical_shards = logical_shard_ids_for_archive(archive_task_id, shards_per_archive)
        if execute:
            # Hold the worker-active lock across the marker check and deletion so a
            # worker cannot write its marker and begin extraction between the two.
            with archive_active_lock(spec.paths.output_dir, archive_task_id):
                item = _evaluate_archive(
                    archive_name=archive_name,
                    archive_task_id=archive_task_id,
                    logical_shards=logical_shards,
                    staging_dir=spec.paths.staging_dir,
                    output_dir=spec.paths.output_dir,
                    expected_s3_prefix=normalized_prefix,
                )
                if item.status == "delete_ready":
                    item = _delete_archive(item)
        else:
            item = _evaluate_archive(
                archive_name=archive_name,
                archive_task_id=archive_task_id,
                logical_shards=logical_shards,
                staging_dir=spec.paths.staging_dir,
                output_dir=spec.paths.output_dir,
                expected_s3_prefix=normalized_prefix,
            )
        items.append(item)

    return ArchiveCleanupReport(
        staging_dir=spec.paths.staging_dir,
        output_dir=spec.paths.output_dir,
        dataset=spec.dataset.name,
        execute=execute,
        expected_s3_prefix=normalized_prefix,
        items=tuple(items),
    )


def render_archive_cleanup_tsv(report: ArchiveCleanupReport) -> str:
    """Render the cleanup report as deterministic tab-separated text."""
    fields = (
        "archive_name",
        "archive_task_id",
        "logical_shards",
        "archive_path",
        "archive_size",
        "expected_model_count",
        "uploaded_model_count",
        "expected_s3_prefix",
        "uploaded_s3_prefixes",
        "status",
        "action",
        "reason",
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(fields)
    for item in report.items:
        writer.writerow(
            (
                item.archive_name,
                item.archive_task_id,
                ",".join(str(shard) for shard in item.logical_shards),
                str(item.archive_path),
                item.archive_size,
                "" if item.expected_model_count is None else item.expected_model_count,
                item.uploaded_model_count,
                item.expected_s3_prefix,
                ",".join(item.uploaded_s3_prefixes),
                item.status,
                item.action,
                item.reason,
            )
        )
    return buffer.getvalue()


def write_archive_cleanup_report(report: ArchiveCleanupReport, output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and TSV cleanup reports under *output_dir*."""
    json_path = write_json_report(report, output_dir / _REPORT_JSON_NAME)
    tsv_path = _write_text_atomic(render_archive_cleanup_tsv(report), output_dir / _REPORT_TSV_NAME)
    return json_path, tsv_path


def _evaluate_archive(
    *,
    archive_name: str,
    archive_task_id: int,
    logical_shards: tuple[int, ...],
    staging_dir: Path,
    output_dir: Path,
    expected_s3_prefix: str,
) -> ArchiveCleanupItem:
    """Evaluate one archive against the safety criteria without mutating it."""
    base = ArchiveCleanupItem(
        archive_name=archive_name,
        archive_task_id=archive_task_id,
        logical_shards=logical_shards,
        archive_path=staging_dir / archive_name,
        archive_size=0,
        expected_model_count=_allowlist_count(output_dir, archive_name),
        uploaded_model_count=0,
        expected_s3_prefix=expected_s3_prefix,
        uploaded_s3_prefixes=(),
        status="",
        action="keep",
        reason="",
    )

    # Criterion 1: plain archive-list entry and a real, non-symlinked file.
    if Path(archive_name).name != archive_name or not archive_name.endswith(".tar.lz4"):
        reason = f"archive-list entry {archive_name!r} is not a plain .tar.lz4 filename"
        return _with_status(base, "keep_unsafe_archive_path", reason)
    archive_path = base.archive_path
    if archive_path.is_symlink():
        return _with_status(base, "keep_unsafe_archive_path", f"archive is a symlink: {archive_path}")
    if not archive_path.is_file():
        return _with_status(base, "missing_archive", f"archive not present in staging: {archive_path}", action="skip")
    if archive_path.resolve().parent != staging_dir.resolve():
        return _with_status(base, "keep_unsafe_archive_path", f"archive resolves outside staging: {archive_path}")

    # Criterion: refuse while an archive task worker is in flight. The worker
    # writes this marker for the duration of extraction through final upload,
    # so a concurrent rerun cannot lose its staged input mid-extraction.
    if (output_dir / archive_active_marker_name(archive_task_id)).exists():
        return _with_status(base, "keep_worker_active", f"archive task {archive_task_id} is still active")

    shard_dirs = tuple(output_dir / f"shard_{shard_id}" for shard_id in logical_shards)

    # Criterion 2: all logical shard directories exist.
    missing_shards = tuple(shard_dir for shard_dir in shard_dirs if not shard_dir.is_dir())
    if missing_shards:
        names = ", ".join(shard_dir.name for shard_dir in missing_shards)
        return _with_status(base, "keep_missing_shard", f"missing shard directories: {names}")

    # Criteria 3-8: per-shard marker and completion evidence.
    uploaded_model_count = 0
    uploaded_prefixes: list[str] = []
    for shard_dir in shard_dirs:
        marker = _read_uploaded_marker(shard_dir)
        if marker is None:
            if not (shard_dir / ".uploaded").exists():
                return _with_status(base, "keep_missing_uploaded_marker", f"{shard_dir.name} lacks .uploaded")
            return _with_status(base, "keep_invalid_marker", f"{shard_dir.name} .uploaded is not a valid JSON object")

        if _marker_str(marker, "status") != "uploaded":
            return _with_status(base, "keep_failed_models", f"{shard_dir.name} .uploaded status is not 'uploaded'")

        s3_prefix = (_marker_str(marker, "s3_prefix") or "").strip().rstrip("/")
        if s3_prefix != expected_s3_prefix:
            return _with_status(
                base,
                "keep_prefix_mismatch",
                f"{shard_dir.name} .uploaded s3_prefix {s3_prefix!r} != expected {expected_s3_prefix!r}",
            )
        uploaded_prefixes.append(s3_prefix)

        total_batches = _marker_int(marker, "total_batches")
        done_count, partial_count = _batch_marker_counts(shard_dir)
        if total_batches is None or done_count != total_batches or partial_count != 0:
            reason = (
                f"{shard_dir.name} batch markers done={done_count} "
                f"partial={partial_count} != total_batches={total_batches}"
            )
            return _with_status(base, "keep_missing_batch_marker", reason)

        metadata_files = _marker_str_list(marker, "metadata_files")
        if metadata_files is None or not metadata_files:
            reason = f"{shard_dir.name} .uploaded metadata_files is empty"
            return _with_status(base, "keep_missing_metadata_upload", reason)

        if (shard_dir / "failed_models.tsv").exists() and (shard_dir / "failed_models.tsv").stat().st_size > 0:
            return _with_status(base, "keep_failed_models", f"{shard_dir.name} has non-empty failed_models.tsv")

        if _success_outputs_has_files(shard_dir):
            return _with_status(base, "keep_fallback_upload", f"{shard_dir.name} success_outputs still contains files")

        model_count = _marker_int(marker, "model_count") or 0
        uploaded_model_count += model_count

    # Criterion 9: allowlist model-count equality when an allowlist exists.
    expected_model_count = base.expected_model_count
    if expected_model_count is not None and uploaded_model_count != expected_model_count:
        return _with_status(
            base,
            "keep_model_count_mismatch",
            f"uploaded model count {uploaded_model_count} != allowlist count {expected_model_count}",
        )

    return replace(
        base,
        archive_size=archive_path.stat().st_size,
        uploaded_model_count=uploaded_model_count,
        uploaded_s3_prefixes=tuple(dict.fromkeys(uploaded_prefixes)),
        status="delete_ready",
        action="delete",
        reason="all shards self-uploaded to the expected prefix",
    )


def _delete_archive(item: ArchiveCleanupItem) -> ArchiveCleanupItem:
    """Delete a delete-ready archive, reporting the outcome without claiming success on error."""
    try:
        item.archive_path.unlink()
    except OSError as exc:
        return _with_status(item, "keep_delete_error", f"unlink failed: {exc}")
    return replace(item, status="deleted", action="delete", reason="archive deleted after confirmed self-upload")


def _with_status(item: ArchiveCleanupItem, status: str, reason: str, *, action: str = "keep") -> ArchiveCleanupItem:
    return replace(item, status=status, action=action, reason=reason)


def _read_uploaded_marker(shard_dir: Path) -> dict[str, object] | None:
    """Return the parsed ``.uploaded`` marker payload, or ``None`` if invalid/missing."""
    marker_path = shard_dir / ".uploaded"
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return cast(dict[str, object], payload)


def _allowlist_count(output_dir: Path, archive_name: str) -> int | None:
    """Return the allowlist model count for an archive, or ``None`` if absent."""
    path = output_dir / "allowlists" / f"{archive_name}.txt"
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _batch_marker_counts(shard_dir: Path) -> tuple[int, int]:
    """Return ``(done_markers, partial_markers)`` counts for one shard."""
    done = 0
    partial = 0
    if not shard_dir.is_dir():
        return done, partial
    for entry in shard_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        for prefix in (".batch_", ".retry_failed_batch_"):
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix) :]
            if rest.endswith("_done") and rest[: -len("_done")].isdigit():
                done += 1
            elif rest.endswith("_partial_uploaded") and rest[: -len("_partial_uploaded")].isdigit():
                partial += 1
    return done, partial


def _success_outputs_has_files(shard_dir: Path) -> bool:
    """Return whether a shard's ``success_outputs`` directory contains any file."""
    success_dir = shard_dir / "success_outputs"
    if not success_dir.is_dir():
        return False
    return any(entry.is_file() for entry in success_dir.iterdir())


def _marker_str(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _marker_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _marker_str_list(payload: dict[str, object], key: str) -> tuple[str, ...] | None:
    value = payload.get(key)
    if not isinstance(value, list):
        return None
    return tuple(item for item in value if isinstance(item, str))


def _write_text_atomic(payload: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    return path
