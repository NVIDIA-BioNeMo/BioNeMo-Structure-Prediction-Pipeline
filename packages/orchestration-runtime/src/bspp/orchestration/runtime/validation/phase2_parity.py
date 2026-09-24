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

"""Filesystem parity checks for Phase 2 task 278 delivery artifacts."""

from __future__ import annotations

import csv
import hashlib
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary

REQUIRED_FILES = (
    "local_tars.csv",
    "analysis_metadata.csv",
    "processing_log.csv",
    "failed_models.tsv",
    "high_quality_model_ids.txt",
)
EXACT_REQUIRED_FILES = (*REQUIRED_FILES, "model_tar_index.csv")
REQUIRED_DIRECTORIES = ("local_tars",)
NORMALIZED_ROW_COUNT_FILES = (
    "analysis_metadata.csv",
    "processing_log.csv",
    "failed_models.tsv",
    "high_quality_model_ids.txt",
)
EXACT_NORMALIZED_ROW_COUNT_FILES = (*NORMALIZED_ROW_COUNT_FILES, "model_tar_index.csv")
EXACT_SHA256_FILES = EXACT_REQUIRED_FILES
ParityProfile = Literal["fast", "exact"]
TAR_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".tar.zst",
    ".tzst",
)
TARFILE_READABLE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tbz2", ".tar.xz", ".txz")


@dataclass(frozen=True)
class PresenceComparison:
    """Presence check for a required file or directory."""

    relative_path: str
    expected_kind: str
    baseline_exists: bool
    candidate_exists: bool

    @property
    def ok(self) -> bool:
        return self.baseline_exists and self.candidate_exists

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable comparison data."""
        return {
            "relative_path": self.relative_path,
            "expected_kind": self.expected_kind,
            "baseline_exists": self.baseline_exists,
            "candidate_exists": self.candidate_exists,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class CsvComparison:
    """CSV schema and record-count comparison."""

    relative_path: str
    baseline_columns: tuple[str, ...] | None
    candidate_columns: tuple[str, ...] | None
    baseline_rows: int | None
    candidate_rows: int | None

    @property
    def schema_ok(self) -> bool:
        return self.baseline_columns is not None and self.baseline_columns == self.candidate_columns

    @property
    def rows_ok(self) -> bool:
        return self.baseline_rows is not None and self.baseline_rows == self.candidate_rows

    @property
    def ok(self) -> bool:
        return self.schema_ok and self.rows_ok

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable comparison data."""
        return {
            "relative_path": self.relative_path,
            "baseline_columns": list(self.baseline_columns) if self.baseline_columns is not None else None,
            "candidate_columns": list(self.candidate_columns) if self.candidate_columns is not None else None,
            "baseline_rows": self.baseline_rows,
            "candidate_rows": self.candidate_rows,
            "schema_ok": self.schema_ok,
            "rows_ok": self.rows_ok,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class RowCountComparison:
    """Normalized row-count comparison for line-oriented artifacts."""

    relative_path: str
    baseline_rows: int | None
    candidate_rows: int | None

    @property
    def ok(self) -> bool:
        return self.baseline_rows is not None and self.baseline_rows == self.candidate_rows

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable comparison data."""
        return {
            "relative_path": self.relative_path,
            "baseline_rows": self.baseline_rows,
            "candidate_rows": self.candidate_rows,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class HashComparison:
    """SHA256 comparison for an exact file present in both output dirs."""

    relative_path: str
    baseline_sha256: str | None
    candidate_sha256: str | None

    @property
    def ok(self) -> bool:
        return self.baseline_sha256 is not None and self.baseline_sha256 == self.candidate_sha256

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable comparison data."""
        return {
            "relative_path": self.relative_path,
            "baseline_sha256": self.baseline_sha256,
            "candidate_sha256": self.candidate_sha256,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class TarFileListComparison:
    """Relative tar-file inventory comparison under ``local_tars``."""

    relative_dir: str
    baseline_files: tuple[str, ...] | None
    candidate_files: tuple[str, ...] | None
    baseline_unreadable: tuple[str, ...] = ()
    candidate_unreadable: tuple[str, ...] = ()

    @property
    def file_list_ok(self) -> bool:
        return self.baseline_files is not None and self.baseline_files == self.candidate_files

    @property
    def tar_readability_ok(self) -> bool:
        return not self.baseline_unreadable and not self.candidate_unreadable

    @property
    def ok(self) -> bool:
        return self.file_list_ok and self.tar_readability_ok

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable comparison data."""
        return {
            "relative_dir": self.relative_dir,
            "baseline_files": list(self.baseline_files) if self.baseline_files is not None else None,
            "candidate_files": list(self.candidate_files) if self.candidate_files is not None else None,
            "baseline_unreadable": list(self.baseline_unreadable),
            "candidate_unreadable": list(self.candidate_unreadable),
            "file_list_ok": self.file_list_ok,
            "tar_readability_ok": self.tar_readability_ok,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class TarMemberCountComparison:
    """Per-tar member-count comparison under ``local_tars``."""

    relative_dir: str
    baseline_counts: dict[str, int] | None
    candidate_counts: dict[str, int] | None

    @property
    def ok(self) -> bool:
        return self.baseline_counts is not None and self.baseline_counts == self.candidate_counts

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable comparison data."""
        return {
            "relative_dir": self.relative_dir,
            "baseline_counts": self.baseline_counts,
            "candidate_counts": self.candidate_counts,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class MultisetComparison:
    """Normalized multiset comparison for IDs in line-oriented artifacts."""

    relative_path: str
    key: str
    baseline_count: int | None
    candidate_count: int | None
    only_baseline: tuple[str, ...]
    only_candidate: tuple[str, ...]
    differing_multiplicity: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            self.baseline_count is not None
            and self.candidate_count is not None
            and not self.only_baseline
            and not self.only_candidate
            and not self.differing_multiplicity
        )

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable comparison data."""
        return {
            "relative_path": self.relative_path,
            "key": self.key,
            "baseline_count": self.baseline_count,
            "candidate_count": self.candidate_count,
            "only_baseline": list(self.only_baseline),
            "only_candidate": list(self.only_candidate),
            "differing_multiplicity": list(self.differing_multiplicity),
            "ok": self.ok,
        }


@dataclass(frozen=True)
class Phase2ParityReport:
    """Side-effect-free parity report for Phase 2 task 278 artifacts."""

    profile: ParityProfile
    baseline_dir: Path
    candidate_dir: Path
    required: tuple[PresenceComparison, ...]
    local_tars_csv: CsvComparison
    local_tars: TarFileListComparison
    local_tar_member_counts: TarMemberCountComparison
    sha256: tuple[HashComparison, ...]
    row_counts: tuple[RowCountComparison, ...]
    multisets: tuple[MultisetComparison, ...]

    @property
    def ok(self) -> bool:
        return (
            all(item.ok for item in self.required)
            and self.local_tars_csv.ok
            and self.local_tars.ok
            and self.local_tar_member_counts.ok
            and all(item.ok for item in self.sha256)
            and all(item.ok for item in self.row_counts)
            and all(item.ok for item in self.multisets)
        )

    @property
    def mismatches(self) -> tuple[object, ...]:
        mismatches: list[object] = []
        mismatches.extend(item for item in self.required if not item.ok)
        if not self.local_tars_csv.ok:
            mismatches.append(self.local_tars_csv)
        if not self.local_tars.ok:
            mismatches.append(self.local_tars)
        if not self.local_tar_member_counts.ok:
            mismatches.append(self.local_tar_member_counts)
        mismatches.extend(item for item in self.sha256 if not item.ok)
        mismatches.extend(item for item in self.row_counts if not item.ok)
        mismatches.extend(item for item in self.multisets if not item.ok)
        return tuple(mismatches)

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable report data."""
        return {
            "profile": self.profile,
            "baseline_dir": str(self.baseline_dir),
            "candidate_dir": str(self.candidate_dir),
            "ok": self.ok,
            "required": [item.to_redacted_dict() for item in self.required],
            "local_tars_csv": self.local_tars_csv.to_redacted_dict(),
            "local_tars": self.local_tars.to_redacted_dict(),
            "local_tar_member_counts": self.local_tar_member_counts.to_redacted_dict(),
            "sha256": [item.to_redacted_dict() for item in self.sha256],
            "row_counts": [item.to_redacted_dict() for item in self.row_counts],
            "multisets": [item.to_redacted_dict() for item in self.multisets],
            "mismatches": [_mismatch_summary(item) for item in self.mismatches],
        }


def compare_phase2_parity(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    profile: ParityProfile = "exact",
) -> Phase2ParityReport:
    """Compare baseline and candidate Phase 2 task 278 output directories.

    The function only reads files under the two supplied directories. It
    does not shell out, mutate files, or infer correctness from external
    systems.
    """
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)
    if profile not in {"fast", "exact"}:
        msg = f"unknown Phase 2 parity profile: {profile}"
        raise ValueError(msg)

    required_files = EXACT_REQUIRED_FILES if profile == "exact" else REQUIRED_FILES
    row_count_files = EXACT_NORMALIZED_ROW_COUNT_FILES if profile == "exact" else NORMALIZED_ROW_COUNT_FILES
    sha256_files = EXACT_SHA256_FILES if profile == "exact" else ()

    required = tuple(
        _compare_presence(baseline_dir, candidate_dir, relative_path, "file") for relative_path in required_files
    ) + tuple(
        _compare_presence(baseline_dir, candidate_dir, relative_path, "directory")
        for relative_path in REQUIRED_DIRECTORIES
    )
    local_tars_csv = _compare_csv(baseline_dir, candidate_dir, "local_tars.csv")
    local_tars = _compare_tar_file_list(baseline_dir, candidate_dir, "local_tars")
    local_tar_member_counts = _compare_tar_member_counts(baseline_dir, candidate_dir, "local_tars")
    sha256 = tuple(_compare_sha256(baseline_dir, candidate_dir, relative_path) for relative_path in sha256_files)
    row_counts = tuple(
        _compare_row_count(baseline_dir, candidate_dir, relative_path) for relative_path in row_count_files
    )
    multisets = (
        _compare_csv_key_multiset(baseline_dir, candidate_dir, "analysis_metadata.csv", "model_id"),
        _compare_csv_key_multiset(baseline_dir, candidate_dir, "processing_log.csv", "model_id"),
        _compare_line_multiset(baseline_dir, candidate_dir, "high_quality_model_ids.txt"),
        _compare_tsv_column_multiset(baseline_dir, candidate_dir, "failed_models.tsv", 0),
    )

    return Phase2ParityReport(
        profile=profile,
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        required=required,
        local_tars_csv=local_tars_csv,
        local_tars=local_tars,
        local_tar_member_counts=local_tar_member_counts,
        sha256=sha256,
        row_counts=row_counts,
        multisets=multisets,
    )


def render_phase2_parity_report(report: Phase2ParityReport) -> str:
    """Render a deterministic JSON Phase 2 parity report."""
    return report_to_json(report)


def write_phase2_parity_report(report: Phase2ParityReport, output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and text Phase 2 parity reports under *output_dir*."""
    json_path = write_json_report(report, output_dir / "wp8" / "phase2_parity_report.json")
    text_path = write_text_summary(report, output_dir / "wp8" / "phase2_parity_report.txt")
    return json_path, text_path


def _mismatch_summary(item: object) -> dict[str, object]:
    relative_path = getattr(item, "relative_path", None)
    relative_dir = getattr(item, "relative_dir", None)
    return {
        "type": type(item).__name__,
        "path": relative_path or relative_dir,
    }


def _compare_presence(
    baseline_dir: Path,
    candidate_dir: Path,
    relative_path: str,
    expected_kind: str,
) -> PresenceComparison:
    return PresenceComparison(
        relative_path=relative_path,
        expected_kind=expected_kind,
        baseline_exists=_path_has_kind(baseline_dir / relative_path, expected_kind),
        candidate_exists=_path_has_kind(candidate_dir / relative_path, expected_kind),
    )


def _path_has_kind(path: Path, expected_kind: str) -> bool:
    if expected_kind == "directory":
        return path.is_dir()
    return path.is_file()


def _compare_csv(baseline_dir: Path, candidate_dir: Path, relative_path: str) -> CsvComparison:
    baseline_columns, baseline_rows = _read_csv_summary(baseline_dir / relative_path)
    candidate_columns, candidate_rows = _read_csv_summary(candidate_dir / relative_path)
    return CsvComparison(
        relative_path=relative_path,
        baseline_columns=baseline_columns,
        candidate_columns=candidate_columns,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
    )


def _read_csv_summary(path: Path) -> tuple[tuple[str, ...] | None, int | None]:
    if not path.is_file():
        return None, None
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return (), 0
        row_count = sum(1 for _row in reader)
    return tuple(reader.fieldnames), row_count


def _compare_row_count(baseline_dir: Path, candidate_dir: Path, relative_path: str) -> RowCountComparison:
    return RowCountComparison(
        relative_path=relative_path,
        baseline_rows=_normalized_row_count(baseline_dir / relative_path),
        candidate_rows=_normalized_row_count(candidate_dir / relative_path),
    )


def _normalized_row_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    if path.suffix == ".csv":
        _columns, rows = _read_csv_summary(path)
        return rows
    if path.suffix == ".tsv":
        with path.open(newline="") as fh:
            return sum(1 for row in csv.reader(fh, delimiter="\t") if any(cell.strip() for cell in row))
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _compare_sha256(baseline_dir: Path, candidate_dir: Path, relative_path: str) -> HashComparison:
    return HashComparison(
        relative_path=relative_path,
        baseline_sha256=_sha256_file(baseline_dir / relative_path),
        candidate_sha256=_sha256_file(candidate_dir / relative_path),
    )


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare_tar_file_list(baseline_dir: Path, candidate_dir: Path, relative_dir: str) -> TarFileListComparison:
    baseline_files, baseline_unreadable = _tar_file_inventory(baseline_dir / relative_dir)
    candidate_files, candidate_unreadable = _tar_file_inventory(candidate_dir / relative_dir)
    return TarFileListComparison(
        relative_dir=relative_dir,
        baseline_files=baseline_files,
        candidate_files=candidate_files,
        baseline_unreadable=baseline_unreadable,
        candidate_unreadable=candidate_unreadable,
    )


def _compare_tar_member_counts(baseline_dir: Path, candidate_dir: Path, relative_dir: str) -> TarMemberCountComparison:
    return TarMemberCountComparison(
        relative_dir=relative_dir,
        baseline_counts=_tar_member_counts(baseline_dir / relative_dir),
        candidate_counts=_tar_member_counts(candidate_dir / relative_dir),
    )


def _tar_member_counts(path: Path) -> dict[str, int] | None:
    if not path.is_dir():
        return None
    counts: dict[str, int] = {}
    for relative in _tar_file_inventory(path)[0] or ():
        tar_path = path / relative
        try:
            with tarfile.open(tar_path, mode="r:*") as archive:
                counts[relative] = sum(1 for member in archive.getmembers() if member.isfile())
        except (OSError, tarfile.TarError):
            counts[relative] = -1
    return counts


def _tar_file_inventory(path: Path) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    if not path.is_dir():
        return None, ()

    tar_files = tuple(
        sorted(
            str(entry.relative_to(path)) for entry in path.rglob("*") if entry.is_file() and _has_tar_suffix(entry.name)
        )
    )
    unreadable = tuple(
        relative
        for relative in tar_files
        if _has_tarfile_readable_suffix(relative) and not _tar_is_readable(path / relative)
    )
    return tar_files, unreadable


def _has_tar_suffix(filename: str) -> bool:
    return filename.endswith(TAR_SUFFIXES)


def _has_tarfile_readable_suffix(filename: str) -> bool:
    return filename.endswith(TARFILE_READABLE_SUFFIXES)


def _tar_is_readable(path: Path) -> bool:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            archive.getmembers()
    except (OSError, tarfile.TarError):
        return False
    return True


def _compare_csv_key_multiset(
    baseline_dir: Path,
    candidate_dir: Path,
    relative_path: str,
    key: str,
) -> MultisetComparison:
    return _compare_counter(
        relative_path,
        key,
        _csv_key_counter(baseline_dir / relative_path, key),
        _csv_key_counter(candidate_dir / relative_path, key),
    )


def _csv_key_counter(path: Path, key: str) -> Counter[str] | None:
    if not path.is_file():
        return None
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or key not in reader.fieldnames:
            return None
        return Counter(row[key] for row in reader if row.get(key, "").strip())


def _compare_line_multiset(baseline_dir: Path, candidate_dir: Path, relative_path: str) -> MultisetComparison:
    return _compare_counter(
        relative_path,
        "line",
        _line_counter(baseline_dir / relative_path),
        _line_counter(candidate_dir / relative_path),
    )


def _line_counter(path: Path) -> Counter[str] | None:
    if not path.is_file():
        return None
    return Counter(line.strip() for line in path.read_text().splitlines() if line.strip())


def _compare_tsv_column_multiset(
    baseline_dir: Path,
    candidate_dir: Path,
    relative_path: str,
    column_index: int,
) -> MultisetComparison:
    return _compare_counter(
        relative_path,
        f"column_{column_index}",
        _tsv_column_counter(baseline_dir / relative_path, column_index),
        _tsv_column_counter(candidate_dir / relative_path, column_index),
    )


def _tsv_column_counter(path: Path, column_index: int) -> Counter[str] | None:
    if not path.is_file():
        return None
    counter: Counter[str] = Counter()
    with path.open(newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) > column_index and row[column_index].strip():
                counter[row[column_index]] += 1
    return counter


def _compare_counter(
    relative_path: str,
    key: str,
    baseline: Counter[str] | None,
    candidate: Counter[str] | None,
) -> MultisetComparison:
    baseline_keys = set(baseline or ())
    candidate_keys = set(candidate or ())
    common = baseline_keys & candidate_keys
    return MultisetComparison(
        relative_path=relative_path,
        key=key,
        baseline_count=sum(baseline.values()) if baseline is not None else None,
        candidate_count=sum(candidate.values()) if candidate is not None else None,
        only_baseline=tuple(sorted(baseline_keys - candidate_keys)),
        only_candidate=tuple(sorted(candidate_keys - baseline_keys)),
        differing_multiplicity=(
            tuple(sorted(item for item in common if baseline[item] != candidate[item]))
            if baseline is not None and candidate is not None
            else ()
        ),
    )


__all__ = [
    "REQUIRED_DIRECTORIES",
    "REQUIRED_FILES",
    "CsvComparison",
    "HashComparison",
    "MultisetComparison",
    "ParityProfile",
    "Phase2ParityReport",
    "PresenceComparison",
    "RowCountComparison",
    "TarFileListComparison",
    "TarMemberCountComparison",
    "compare_phase2_parity",
    "render_phase2_parity_report",
    "write_phase2_parity_report",
]
