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

"""Archive coverage and staging planning for input preparation."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from bspp.orchestration.runtime.inputs.references import redact_uri
from bspp.orchestration.runtime.slurm.arrays import array_task_ids

ARCHIVE_COLUMN = "swiftstack_archive"
DATASET_COLUMNS: tuple[str, ...] = ("dataset_name", "source_run", "dataset")


@dataclass(frozen=True)
class ArchiveCoverage:
    """Archive coverage for a dataset slice in a tracking parquet."""

    dataset: str
    archives: tuple[str, ...]
    total_rows: int
    missing_archive_rows: int

    @property
    def unique_archive_count(self) -> int:
        """Return the number of unique non-empty archives."""
        return len(self.archives)

    def to_redacted_dict(self) -> dict[str, Any]:
        """Return JSON-serializable coverage data."""
        return {
            "dataset": self.dataset,
            "archives": list(self.archives),
            "unique_archive_count": self.unique_archive_count,
            "total_rows": self.total_rows,
            "missing_archive_rows": self.missing_archive_rows,
        }


def archives_from_staging_dir(staging_dir: Path, dataset: str) -> ArchiveCoverage:
    """Return deterministic archive inventory from already staged ``.tar.lz4`` files."""
    if not staging_dir.exists():
        raise FileNotFoundError(f"Staging directory not found: {staging_dir}")
    if not staging_dir.is_dir():
        raise NotADirectoryError(f"Staging path is not a directory: {staging_dir}")
    archives = tuple(path.name for path in sorted(staging_dir.glob("*.tar.lz4")))
    return ArchiveCoverage(
        dataset=dataset,
        archives=archives,
        total_rows=len(archives),
        missing_archive_rows=0,
    )


def archives_for_runspec(spec: Any) -> ArchiveCoverage:
    """Return archive coverage according to ``dataset.archive_source``."""
    dataset = spec.dataset
    source = getattr(dataset, "archive_source", "tracking")
    if source == "tracking":
        return archives_for_dataset(spec.references.tracking_parquet, dataset.name)
    if source == "staging_dir":
        return archives_from_staging_dir(spec.paths.staging_dir, dataset.name)
    msg = f"Unsupported dataset.archive_source: {source!r}"
    raise ValueError(msg)


def archives_for_runspec_staging(spec: Any, coverage: ArchiveCoverage | None = None) -> ArchiveCoverage:
    """Return the archive subset that must be present before this RunSpec runs.

    Tracking parquet coverage stays full so rendered recipes and archive lists
    preserve legacy task indexing. Staging only needs the archives selected by
    ``dataset.array`` for bounded one-archive or subset runs.
    """
    coverage = coverage or archives_for_runspec(spec)
    dataset = spec.dataset
    if getattr(dataset, "archive_source", "tracking") != "tracking":
        return coverage
    array = getattr(dataset, "array", None)
    if not array:
        return coverage
    return select_archives_for_array(coverage, array)


def select_archives_for_array(coverage: ArchiveCoverage, array: str) -> ArchiveCoverage:
    """Return the archive names selected by a Slurm array expression."""
    task_ids = array_task_ids(array)
    archives = coverage.archives
    out_of_range = [task_id for task_id in task_ids if task_id >= len(archives)]
    if out_of_range:
        msg = (
            f"SLURM array {array!r} selects archive index {out_of_range[0]} "
            f"but tracking coverage has {len(archives)} archives"
        )
        raise ValueError(msg)
    selected = tuple(dict.fromkeys(archives[task_id] for task_id in task_ids))
    return ArchiveCoverage(
        dataset=coverage.dataset,
        archives=selected,
        total_rows=len(selected),
        missing_archive_rows=0,
    )


@dataclass(frozen=True)
class ArchiveStagingItem:
    """Status and planned source for one staged archive."""

    archive: str
    destination: Path
    present: bool
    size_bytes: int = 0
    source: str | None = None
    action: str = "skip"

    def to_redacted_dict(self) -> dict[str, Any]:
        """Return JSON-serializable staging item data without secrets."""
        return {
            "archive": self.archive,
            "destination": str(self.destination),
            "present": self.present,
            "size_bytes": self.size_bytes,
            "source": redact_uri(self.source) if self.source else None,
            "action": self.action,
        }


@dataclass(frozen=True)
class ArchiveCommandFilePlan:
    """Planned batched s5cmd archive downloads without filesystem writes."""

    command_file: Path
    s5cmd_path: str
    argv: tuple[str, ...]
    commands: tuple[str, ...]
    numworkers: int | None = None

    @property
    def command_count(self) -> int:
        """Return the number of planned s5cmd sub-commands."""
        return len(self.commands)

    def to_redacted_dict(self) -> dict[str, Any]:
        """Return JSON-serializable command-file plan data without secrets."""
        return {
            "command_file": str(self.command_file),
            "s5cmd_path": self.s5cmd_path,
            "argv": [redact_uri(arg) for arg in self.argv],
            "numworkers": self.numworkers,
            "command_count": self.command_count,
            "commands": [_redact_command(command) for command in self.commands],
        }


@dataclass(frozen=True)
class ArchiveStagingPlan:
    """Deterministic local staging status for required archives."""

    staging_dir: Path
    items: tuple[ArchiveStagingItem, ...]
    dry_run: bool = True
    download_command_file: ArchiveCommandFilePlan | None = None

    @property
    def present_count(self) -> int:
        """Return the number of archives already staged."""
        return sum(1 for item in self.items if item.present)

    @property
    def planned_count(self) -> int:
        """Return the number of archives needing transfer."""
        return sum(1 for item in self.items if item.action != "skip")

    @property
    def total_staged_bytes(self) -> int:
        """Return bytes for archives already staged locally."""
        return sum(item.size_bytes for item in self.items if item.present)

    @property
    def missing(self) -> tuple[ArchiveStagingItem, ...]:
        """Return archive items that are not present locally."""
        return tuple(item for item in self.items if not item.present)

    def to_redacted_dict(self) -> dict[str, Any]:
        """Return JSON-serializable staging data without secrets."""
        return {
            "staging_dir": str(self.staging_dir),
            "dry_run": self.dry_run,
            "present_count": self.present_count,
            "planned_count": self.planned_count,
            "total_staged_bytes": self.total_staged_bytes,
            "items": [item.to_redacted_dict() for item in self.items],
            "download_command_file": (
                self.download_command_file.to_redacted_dict() if self.download_command_file is not None else None
            ),
        }


def archives_for_dataset(tracking_parquet: Path, dataset: str) -> ArchiveCoverage:
    """Return deterministic unique archives and missing-row counts for *dataset*."""
    if not tracking_parquet.exists():
        raise FileNotFoundError(f"Tracking parquet not found: {tracking_parquet}")

    schema_names = _parquet_schema_names(tracking_parquet)
    if ARCHIVE_COLUMN not in schema_names:
        raise ValueError(f"Tracking parquet missing required column {ARCHIVE_COLUMN!r}")

    dataset_column = _dataset_column(schema_names)
    columns = _unique_columns((dataset_column, ARCHIVE_COLUMN))
    filters = [(dataset_column, "=", dataset)] if dataset_column is not None else None
    filtered = pq.read_table(
        tracking_parquet,
        columns=columns,
        filters=filters,
    )

    archives = filtered.column(ARCHIVE_COLUMN).to_pylist()
    present_archives = sorted({str(value) for value in archives if value is not None and str(value) != ""})
    missing_rows = sum(1 for value in archives if value is None or str(value) == "")
    return ArchiveCoverage(
        dataset=dataset,
        archives=tuple(present_archives),
        total_rows=filtered.num_rows,
        missing_archive_rows=missing_rows,
    )


def plan_archive_staging(
    archives: ArchiveCoverage | tuple[str, ...] | list[str],
    staging_dir: Path,
    *,
    archive_prefix: str | None = None,
    dry_run: bool = True,
    s5cmd_path: str | Path = "s5cmd",
    s5cmd_numworkers: int | None = None,
    command_file: Path | None = None,
) -> ArchiveStagingPlan:
    """Plan local archive staging without performing network transfers."""
    names = archives.archives if isinstance(archives, ArchiveCoverage) else tuple(archives)
    items: list[ArchiveStagingItem] = []
    for archive in sorted(dict.fromkeys(names)):
        destination = staging_dir / Path(archive).name
        size_bytes = destination.stat().st_size if destination.is_file() else 0
        present = size_bytes > 0
        source = _archive_source(archive_prefix, archive) if not present and archive_prefix else None
        action = "skip" if present else ("download" if source else "missing")
        items.append(
            ArchiveStagingItem(
                archive=archive,
                destination=destination,
                present=present,
                size_bytes=size_bytes,
                source=source,
                action=action,
            )
        )
    command_plan = _download_command_file_plan(
        staging_dir,
        tuple(items),
        s5cmd_path=str(s5cmd_path),
        s5cmd_numworkers=s5cmd_numworkers,
        command_file=command_file,
    )
    return ArchiveStagingPlan(
        staging_dir=staging_dir,
        items=tuple(items),
        dry_run=dry_run,
        download_command_file=command_plan,
    )


def _download_command_file_plan(
    staging_dir: Path,
    items: tuple[ArchiveStagingItem, ...],
    *,
    s5cmd_path: str,
    s5cmd_numworkers: int | None,
    command_file: Path | None,
) -> ArchiveCommandFilePlan | None:
    commands = tuple(
        f"cp {item.source} {item.destination}"
        for item in items
        if item.action == "download" and item.source is not None and item.source.startswith("s3://")
    )
    if not commands:
        return None
    command_file = command_file or (staging_dir / "s5cmd_download_archives.txt")
    argv: list[str] = [s5cmd_path]
    if s5cmd_numworkers is not None and s5cmd_numworkers > 0:
        argv.extend(["--numworkers", str(s5cmd_numworkers)])
    argv.extend(["run", str(command_file)])
    return ArchiveCommandFilePlan(
        command_file=command_file,
        s5cmd_path=s5cmd_path,
        argv=tuple(argv),
        commands=commands,
        numworkers=s5cmd_numworkers,
    )


def _parquet_schema_names(path: Path) -> list[str]:
    return [str(name) for name in pq.ParquetFile(path).schema_arrow.names]


def _unique_columns(columns: tuple[str | None, ...]) -> list[str]:
    result: list[str] = []
    for column in columns:
        if column is not None and column not in result:
            result.append(column)
    return result


def _dataset_column(names: list[str]) -> str | None:
    for name in DATASET_COLUMNS:
        if name in names:
            return name
    return None


def _archive_source(prefix: str | None, archive: str) -> str | None:
    if prefix is None:
        return None
    if "://" in archive:
        return archive
    if not prefix.endswith("/"):
        prefix = f"{prefix}/"
    return f"{prefix}{Path(archive).name}"


def _redact_command(command: str) -> str:
    return " ".join(redact_uri(part) for part in shlex.split(command))


__all__ = [
    "ARCHIVE_COLUMN",
    "DATASET_COLUMNS",
    "ArchiveCommandFilePlan",
    "ArchiveCoverage",
    "ArchiveStagingItem",
    "ArchiveStagingPlan",
    "archives_for_dataset",
    "archives_for_runspec",
    "archives_for_runspec_staging",
    "archives_from_staging_dir",
    "plan_archive_staging",
    "select_archives_for_array",
]
