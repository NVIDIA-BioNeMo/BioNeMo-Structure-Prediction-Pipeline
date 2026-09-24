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

"""RunSpec-driven validation and local postprocessing reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary
from bspp.orchestration.runtime.postprocessing.cleanup import cleanup_shard_outputs
from bspp.orchestration.runtime.slurm.arrays import array_task_count


@dataclass(frozen=True)
class ShardArchiveOutputStatus:
    """Observed state for one archive-mode logical shard directory."""

    shard_id: int
    shard_dir: Path
    status: str
    uploaded_marker: bool
    marker_model_count: int | None
    marker_file_count: int | None
    success_output_files: int
    batch_done_markers: int
    manifest_rows: int | None
    marker_error: str | None = None

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable shard status data."""
        return {
            "shard_id": self.shard_id,
            "shard_dir": str(self.shard_dir),
            "status": self.status,
            "uploaded_marker": self.uploaded_marker,
            "marker_model_count": self.marker_model_count,
            "marker_file_count": self.marker_file_count,
            "success_output_files": self.success_output_files,
            "batch_done_markers": self.batch_done_markers,
            "manifest_rows": self.manifest_rows,
            "marker_error": self.marker_error,
        }


@dataclass(frozen=True)
class ExpectedCheck:
    """Actual-vs-expected count check."""

    name: str
    actual: int | None
    expected: int | None

    @property
    def ok(self) -> bool | None:
        """Return true when applicable counts match, or ``None`` when skipped."""
        if self.expected is None or self.actual is None:
            return None
        return self.actual == self.expected

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable check data."""
        return {"name": self.name, "actual": self.actual, "expected": self.expected, "ok": self.ok}


@dataclass(frozen=True)
class ArchiveOutputValidationReport:
    """Aggregate validation report for a RunSpec output directory."""

    output_dir: Path
    dataset: str
    run_id: str
    archive_array: str
    expected_archives_for_array: int
    expected_logical_shards_for_array: int
    aggregate_manifest: Path | None
    aggregate_rows: int | None
    shards: tuple[ShardArchiveOutputStatus, ...]
    checks: tuple[ExpectedCheck, ...]

    @property
    def shard_dirs(self) -> int:
        return len(self.shards)

    @property
    def uploaded_marker_count(self) -> int:
        return sum(1 for shard in self.shards if shard.uploaded_marker)

    @property
    def processed_models(self) -> int:
        return sum(shard.marker_model_count or 0 for shard in self.shards)

    @property
    def uploaded_objects(self) -> int:
        return sum(shard.marker_file_count or 0 for shard in self.shards)

    @property
    def success_output_files(self) -> int:
        return sum(shard.success_output_files for shard in self.shards)

    @property
    def manifest_rows(self) -> int:
        return sum(shard.manifest_rows or 0 for shard in self.shards)

    @property
    def fallback_upload_candidates(self) -> int:
        return sum(1 for shard in self.shards if shard.status == "needs_fallback_upload")

    @property
    def marker_errors(self) -> tuple[ShardArchiveOutputStatus, ...]:
        return tuple(shard for shard in self.shards if shard.marker_error)

    @property
    def valid(self) -> bool:
        applicable = [check.ok for check in self.checks if check.ok is not None]
        return all(applicable) and not self.marker_errors

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable report data."""
        return {
            "output_dir": str(self.output_dir),
            "dataset": self.dataset,
            "run_id": self.run_id,
            "archive_array": self.archive_array,
            "expected_archives_for_array": self.expected_archives_for_array,
            "expected_logical_shards_for_array": self.expected_logical_shards_for_array,
            "aggregate_manifest": str(self.aggregate_manifest) if self.aggregate_manifest is not None else None,
            "aggregate_rows": self.aggregate_rows,
            "shard_dirs": self.shard_dirs,
            "uploaded_marker_count": self.uploaded_marker_count,
            "processed_models": self.processed_models,
            "uploaded_objects": self.uploaded_objects,
            "success_output_files": self.success_output_files,
            "manifest_rows": self.manifest_rows,
            "fallback_upload_candidates": self.fallback_upload_candidates,
            "valid": self.valid,
            "checks": [check.to_redacted_dict() for check in self.checks],
            "shards": [shard.to_redacted_dict() for shard in self.shards],
        }


@dataclass(frozen=True)
class FallbackUploadReport:
    """Dry-run plan for local success-output fallback upload."""

    output_dir: Path
    dataset: str
    destination_prefix: str
    tool: str
    shards_queued: int
    shards_skipped_self_uploaded: int
    queued_success_output_dirs: tuple[Path, ...]

    @property
    def no_op(self) -> bool:
        return self.shards_queued == 0

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable fallback upload plan data."""
        return {
            "output_dir": str(self.output_dir),
            "dataset": self.dataset,
            "destination_prefix": self.destination_prefix,
            "tool": self.tool,
            "shards_queued": self.shards_queued,
            "shards_skipped_self_uploaded": self.shards_skipped_self_uploaded,
            "queued_success_output_dirs": [str(path) for path in self.queued_success_output_dirs],
            "no_op": self.no_op,
        }


@dataclass(frozen=True)
class CleanupPlanReport:
    """Plan or result for local success output cleanup."""

    output_dir: Path
    dataset: str
    execute: bool
    force: bool
    success_output_dirs_before: int
    success_output_dirs_after: int
    cleaned_dirs: int
    unsafe_dirs: tuple[Path, ...]

    @property
    def safe_to_execute(self) -> bool:
        return self.force or not self.unsafe_dirs

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable cleanup plan data."""
        return {
            "output_dir": str(self.output_dir),
            "dataset": self.dataset,
            "execute": self.execute,
            "force": self.force,
            "success_output_dirs_before": self.success_output_dirs_before,
            "success_output_dirs_after": self.success_output_dirs_after,
            "cleaned_dirs": self.cleaned_dirs,
            "unsafe_dirs": [str(path) for path in self.unsafe_dirs],
            "safe_to_execute": self.safe_to_execute,
        }


def build_archive_output_validation_report(
    spec: RunSpec,
    *,
    aggregate_manifest: Path | None = None,
) -> ArchiveOutputValidationReport:
    """Scan archive-mode outputs and compare them with RunSpec validation counts."""
    shards = tuple(_scan_shard(path) for _, path in _numeric_shard_dirs(spec.paths.output_dir))
    aggregate_rows = _parquet_rows(aggregate_manifest) if aggregate_manifest is not None else None
    archive_count = _archive_count_from_array(spec.dataset.array)
    logical_shards = archive_count * spec.worker.shards_per_archive

    report = ArchiveOutputValidationReport(
        output_dir=spec.paths.output_dir,
        dataset=spec.dataset.name,
        run_id=spec.dataset.run_id,
        archive_array=spec.dataset.array,
        expected_archives_for_array=archive_count,
        expected_logical_shards_for_array=logical_shards,
        aggregate_manifest=aggregate_manifest,
        aggregate_rows=aggregate_rows,
        shards=shards,
        checks=(),
    )
    checks = (
        ExpectedCheck("logical_shards_for_array", report.shard_dirs, logical_shards),
        ExpectedCheck(
            "uploaded_markers",
            report.uploaded_marker_count,
            logical_shards if spec.worker.self_upload else None,
        ),
        ExpectedCheck(
            "one_archive_processed_models",
            report.processed_models if archive_count == 1 else None,
            spec.validation.expected_one_archive_models if archive_count == 1 else None,
        ),
        ExpectedCheck(
            "one_archive_uploaded_objects",
            report.uploaded_objects if archive_count == 1 else None,
            spec.validation.expected_one_archive_objects if archive_count == 1 else None,
        ),
        ExpectedCheck(
            "one_archive_aggregate_rows",
            aggregate_rows if archive_count == 1 else None,
            spec.validation.expected_one_archive_aggregate_rows if archive_count == 1 else None,
        ),
    )
    return ArchiveOutputValidationReport(
        output_dir=report.output_dir,
        dataset=report.dataset,
        run_id=report.run_id,
        archive_array=report.archive_array,
        expected_archives_for_array=report.expected_archives_for_array,
        expected_logical_shards_for_array=report.expected_logical_shards_for_array,
        aggregate_manifest=report.aggregate_manifest,
        aggregate_rows=report.aggregate_rows,
        shards=report.shards,
        checks=checks,
    )


def plan_fallback_upload(spec: RunSpec, *, tool: str = "s5cmd") -> FallbackUploadReport:
    """Plan which local success outputs still need an S3 fallback upload."""
    queued: list[Path] = []
    skipped = 0
    for _shard_id, shard_dir in _numeric_shard_dirs(spec.paths.output_dir):
        success_dir = shard_dir / "success_outputs"
        if not success_dir.is_dir() or not any(item.is_file() for item in success_dir.iterdir()):
            continue
        if (shard_dir / ".uploaded").exists():
            skipped += 1
        else:
            queued.append(success_dir)
    return FallbackUploadReport(
        output_dir=spec.paths.output_dir,
        dataset=spec.dataset.name,
        destination_prefix=spec.storage.s3_output_prefix,
        tool=tool,
        shards_queued=len(queued),
        shards_skipped_self_uploaded=skipped,
        queued_success_output_dirs=tuple(queued),
    )


def plan_cleanup(spec: RunSpec, *, execute: bool = False, force: bool = False) -> CleanupPlanReport:
    """Plan or remove local success outputs once shards are self-uploaded."""
    before_dirs = _success_output_dirs(spec.paths.output_dir)
    unsafe = tuple(path for path in before_dirs if not (path.parent / ".uploaded").exists())
    if execute and unsafe and not force:
        msg = "Refusing cleanup because some success_outputs dirs lack .uploaded markers"
        raise RuntimeError(msg)
    cleaned = cleanup_shard_outputs(spec.paths.output_dir, force=True) if execute else 0
    after_dirs = _success_output_dirs(spec.paths.output_dir)
    return CleanupPlanReport(
        output_dir=spec.paths.output_dir,
        dataset=spec.dataset.name,
        execute=execute,
        force=force,
        success_output_dirs_before=len(before_dirs),
        success_output_dirs_after=len(after_dirs),
        cleaned_dirs=cleaned,
        unsafe_dirs=unsafe,
    )


def render_report(report: object) -> str:
    """Render a deterministic JSON report."""
    return report_to_json(report)


def write_archive_output_validation_report(
    report: ArchiveOutputValidationReport, output_dir: Path
) -> tuple[Path, Path]:
    """Write JSON and text archive-output validation reports under *output_dir*."""
    json_path = write_json_report(report, output_dir / "wp6" / "archive_output_validation_report.json")
    text_path = write_text_summary(report, output_dir / "wp6" / "archive_output_validation_report.txt")
    return json_path, text_path


def _scan_shard(shard_dir: Path) -> ShardArchiveOutputStatus:
    shard_id = int(shard_dir.name.removeprefix("shard_"))
    marker = shard_dir / ".uploaded"
    marker_model_count = None
    marker_file_count = None
    marker_error = None
    if marker.exists():
        try:
            payload = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            marker_error = str(exc)
        else:
            if isinstance(payload, dict):
                marker_model_count = _optional_int(payload.get("model_count"))
                marker_file_count = _optional_int(payload.get("total_files"))
            else:
                marker_error = "marker payload is not a JSON object"

    success_files = _count_success_files(shard_dir / "success_outputs")
    manifest_rows = _parquet_rows(shard_dir / "shard_manifest.parquet")
    batch_done = sum(1 for path in shard_dir.glob(".batch_*_done") if path.is_file())
    if marker_error:
        status = "marker_error"
    elif marker.exists():
        status = "self_uploaded"
    elif success_files:
        status = "needs_fallback_upload"
    else:
        status = "empty"

    return ShardArchiveOutputStatus(
        shard_id=shard_id,
        shard_dir=shard_dir,
        status=status,
        uploaded_marker=marker.exists(),
        marker_model_count=marker_model_count,
        marker_file_count=marker_file_count,
        success_output_files=success_files,
        batch_done_markers=batch_done,
        manifest_rows=manifest_rows,
        marker_error=marker_error,
    )


def _numeric_shard_dirs(output_dir: Path) -> tuple[tuple[int, Path], ...]:
    result: list[tuple[int, Path]] = []
    for path in output_dir.glob("shard_*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("shard_")
        if suffix.isdigit():
            result.append((int(suffix), path))
    return tuple(sorted(result))


def _success_output_dirs(output_dir: Path) -> tuple[Path, ...]:
    dirs: list[Path] = []
    for _shard_id, path in _numeric_shard_dirs(output_dir):
        success_dir = path / "success_outputs"
        if success_dir.is_dir():
            dirs.append(success_dir)
    return tuple(dirs)


def _count_success_files(success_dir: Path) -> int:
    if not success_dir.is_dir():
        return 0
    return sum(1 for path in success_dir.iterdir() if path.is_file())


def _parquet_rows(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return int(pq.read_metadata(path).num_rows)


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _archive_count_from_array(value: str) -> int:
    return array_task_count(value)


__all__ = [
    "ArchiveOutputValidationReport",
    "CleanupPlanReport",
    "ExpectedCheck",
    "FallbackUploadReport",
    "ShardArchiveOutputStatus",
    "build_archive_output_validation_report",
    "plan_cleanup",
    "plan_fallback_upload",
    "render_report",
    "write_archive_output_validation_report",
]
