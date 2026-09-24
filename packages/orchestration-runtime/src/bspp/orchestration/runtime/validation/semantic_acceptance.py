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

"""Semantic acceptance checks for one-archive local-tar output comparisons."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary

TAR_MANIFEST_KEYS = (
    "tar_name",
    "source_archive",
    "shard_id",
    "task_id",
    "batch_id",
    "member_count",
    "size_bytes",
    "compression",
)
ANALYSIS_MODEL_KEYS = (
    "model_id",
    "original_id",
    "upload_status",
    "passes_quality_threshold",
    "failure_reason",
)

# Keys used for the baseline-vs-candidate local_tars.csv row comparison.
# ``size_bytes`` is excluded: compressed tar bytes legitimately differ across
# image builds (zstd output varies while payloads are identical), and the
# tar-payload parity comparator already treats compressed-size-only mismatches
# as non-fatal when payload hashes match. Payload integrity is enforced by the
# tar-payload parity stage; this comparison guards tar identity and structure.
TAR_MANIFEST_COMPARE_KEYS = tuple(key for key in TAR_MANIFEST_KEYS if key != "size_bytes")


@dataclass(frozen=True)
class SemanticAcceptanceReport:
    """Semantic parity report for baseline and candidate local-tar outputs."""

    baseline_dir: Path
    candidate_dir: Path
    ok: bool
    errors: tuple[str, ...]
    baseline_tar_count: int
    candidate_tar_count: int
    baseline_local_tars_csv_rows: int
    candidate_local_tars_csv_rows: int
    baseline_failed_rows: int
    candidate_failed_rows: int
    baseline_analysis_csv_rows: int
    candidate_analysis_csv_rows: int
    baseline_selected_ids_from_csv: int
    candidate_selected_ids_from_csv: int
    candidate_selected_ids_file_rows: int | None
    baseline_parquet_exists: bool
    candidate_parquet_exists: bool
    candidate_parquet_rows: int | None

    def to_redacted_dict(self) -> dict[str, Any]:
        """Return JSON-serializable report data."""
        return {
            "baseline_dir": str(self.baseline_dir),
            "candidate_dir": str(self.candidate_dir),
            "ok": self.ok,
            "errors": list(self.errors),
            "baseline_tar_count": self.baseline_tar_count,
            "candidate_tar_count": self.candidate_tar_count,
            "baseline_local_tars_csv_rows": self.baseline_local_tars_csv_rows,
            "candidate_local_tars_csv_rows": self.candidate_local_tars_csv_rows,
            "baseline_failed_rows": self.baseline_failed_rows,
            "candidate_failed_rows": self.candidate_failed_rows,
            "baseline_analysis_csv_rows": self.baseline_analysis_csv_rows,
            "candidate_analysis_csv_rows": self.candidate_analysis_csv_rows,
            "baseline_selected_ids_from_csv": self.baseline_selected_ids_from_csv,
            "candidate_selected_ids_from_csv": self.candidate_selected_ids_from_csv,
            "candidate_selected_ids_file_rows": self.candidate_selected_ids_file_rows,
            "baseline_parquet_exists": self.baseline_parquet_exists,
            "candidate_parquet_exists": self.candidate_parquet_exists,
            "candidate_parquet_rows": self.candidate_parquet_rows,
        }


def compare_semantic_acceptance(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    expected_tar_count: int | None = None,
    expected_local_tars_rows: int | None = None,
    expected_failed_rows: int | None = None,
    expected_analysis_rows: int | None = None,
    expected_selected_ids: int | None = None,
    require_candidate_parquet: bool = True,
    compare_failed_sets: bool = True,
    compare_tar_manifest_rows: bool = True,
    compare_analysis_model_rows: bool = True,
) -> SemanticAcceptanceReport:
    """Compare semantic outputs that should match after payload parity passes."""
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)
    errors: list[str] = []

    baseline_tar_count = _count_tars(baseline_dir)
    candidate_tar_count = _count_tars(candidate_dir)
    _check_expected("baseline tar count", baseline_tar_count, expected_tar_count, errors)
    _check_expected("candidate tar count", candidate_tar_count, expected_tar_count, errors)
    if baseline_tar_count != candidate_tar_count:
        errors.append(f"tar counts differ: baseline={baseline_tar_count} candidate={candidate_tar_count}")

    baseline_tar_rows = _read_csv_rows(
        baseline_dir / "local_tars.csv",
        "baseline local_tars.csv",
        errors,
        required_columns=TAR_MANIFEST_KEYS,
    )
    candidate_tar_rows = _read_csv_rows(
        candidate_dir / "local_tars.csv",
        "candidate local_tars.csv",
        errors,
        required_columns=TAR_MANIFEST_KEYS,
    )
    baseline_local_tars_rows = len(baseline_tar_rows)
    candidate_local_tars_rows = len(candidate_tar_rows)
    _check_expected("baseline local_tars.csv rows", baseline_local_tars_rows, expected_local_tars_rows, errors)
    _check_expected("candidate local_tars.csv rows", candidate_local_tars_rows, expected_local_tars_rows, errors)
    if baseline_local_tars_rows != candidate_local_tars_rows:
        errors.append(
            f"local_tars.csv row counts differ: baseline={baseline_local_tars_rows} "
            f"candidate={candidate_local_tars_rows}"
        )
    if compare_tar_manifest_rows and _semantic_rows(baseline_tar_rows, TAR_MANIFEST_COMPARE_KEYS) != _semantic_rows(
        candidate_tar_rows,
        TAR_MANIFEST_COMPARE_KEYS,
    ):
        errors.append("local_tars.csv semantic rows differ")

    baseline_failed = _read_failed_models_lines(
        baseline_dir / "failed_models.tsv", "baseline failed_models.tsv", errors
    )
    candidate_failed = _read_failed_models_lines(
        candidate_dir / "failed_models.tsv", "candidate failed_models.tsv", errors
    )
    baseline_failed_rows = len(baseline_failed)
    candidate_failed_rows = len(candidate_failed)
    _check_expected("baseline failed rows", baseline_failed_rows, expected_failed_rows, errors)
    _check_expected("candidate failed rows", candidate_failed_rows, expected_failed_rows, errors)
    if baseline_failed_rows != candidate_failed_rows:
        errors.append(f"failed row counts differ: baseline={baseline_failed_rows} candidate={candidate_failed_rows}")
    if compare_failed_sets and sorted(baseline_failed) != sorted(candidate_failed):
        errors.append("failed model sets differ")

    baseline_analysis_rows = _read_csv_rows(
        baseline_dir / "analysis_metadata.csv",
        "baseline analysis_metadata.csv",
        errors,
        required_columns=ANALYSIS_MODEL_KEYS,
    )
    candidate_analysis_rows = _read_csv_rows(
        candidate_dir / "analysis_metadata.csv",
        "candidate analysis_metadata.csv",
        errors,
        required_columns=ANALYSIS_MODEL_KEYS,
    )
    baseline_analysis_csv_rows = len(baseline_analysis_rows)
    candidate_analysis_csv_rows = len(candidate_analysis_rows)
    _check_expected("baseline analysis rows", baseline_analysis_csv_rows, expected_analysis_rows, errors)
    _check_expected("candidate analysis rows", candidate_analysis_csv_rows, expected_analysis_rows, errors)
    if baseline_analysis_csv_rows != candidate_analysis_csv_rows:
        errors.append(
            f"analysis_metadata.csv row counts differ: baseline={baseline_analysis_csv_rows} "
            f"candidate={candidate_analysis_csv_rows}"
        )
    if compare_analysis_model_rows and _semantic_rows(baseline_analysis_rows, ANALYSIS_MODEL_KEYS) != _semantic_rows(
        candidate_analysis_rows,
        ANALYSIS_MODEL_KEYS,
    ):
        errors.append("analysis metadata semantic model rows differ")

    baseline_selected = _selected_ids_from_analysis(baseline_analysis_rows)
    candidate_selected = _selected_ids_from_analysis(candidate_analysis_rows)
    baseline_selected_ids_from_csv = len(baseline_selected)
    candidate_selected_ids_from_csv = len(candidate_selected)
    _check_expected("baseline selected IDs", baseline_selected_ids_from_csv, expected_selected_ids, errors)
    _check_expected("candidate selected IDs", candidate_selected_ids_from_csv, expected_selected_ids, errors)
    if baseline_selected != candidate_selected:
        errors.append("selected ID sets from analysis CSV differ")

    selected_ids_path = candidate_dir / "high_quality_model_ids.txt"
    candidate_selected_ids_file_rows = None
    if selected_ids_path.exists():
        selected_file_set = set(_read_nonempty_lines(selected_ids_path, "candidate high_quality_model_ids.txt", errors))
        candidate_selected_ids_file_rows = len(selected_file_set)
        if selected_file_set != candidate_selected:
            errors.append("candidate high_quality_model_ids.txt differs from analysis CSV selected IDs")
    else:
        errors.append("candidate high_quality_model_ids.txt is missing")

    baseline_parquet_exists = (baseline_dir / "analysis_metadata.parquet").exists()
    candidate_parquet = candidate_dir / "analysis_metadata.parquet"
    candidate_parquet_exists = candidate_parquet.exists()
    candidate_parquet_rows = None
    if candidate_parquet_exists:
        try:
            candidate_parquet_rows = pq.read_metadata(candidate_parquet).num_rows
        except Exception as exc:
            errors.append(f"candidate analysis parquet could not be read: {exc}")
        else:
            _check_expected("candidate analysis parquet rows", candidate_parquet_rows, expected_analysis_rows, errors)
    elif require_candidate_parquet:
        errors.append("candidate analysis parquet is missing")

    return SemanticAcceptanceReport(
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        ok=not errors,
        errors=tuple(errors),
        baseline_tar_count=baseline_tar_count,
        candidate_tar_count=candidate_tar_count,
        baseline_local_tars_csv_rows=baseline_local_tars_rows,
        candidate_local_tars_csv_rows=candidate_local_tars_rows,
        baseline_failed_rows=baseline_failed_rows,
        candidate_failed_rows=candidate_failed_rows,
        baseline_analysis_csv_rows=baseline_analysis_csv_rows,
        candidate_analysis_csv_rows=candidate_analysis_csv_rows,
        baseline_selected_ids_from_csv=baseline_selected_ids_from_csv,
        candidate_selected_ids_from_csv=candidate_selected_ids_from_csv,
        candidate_selected_ids_file_rows=candidate_selected_ids_file_rows,
        baseline_parquet_exists=baseline_parquet_exists,
        candidate_parquet_exists=candidate_parquet_exists,
        candidate_parquet_rows=candidate_parquet_rows,
    )


def render_semantic_acceptance_report(report: SemanticAcceptanceReport) -> str:
    """Render a deterministic JSON report."""
    return report_to_json(report)


def write_semantic_acceptance_report(report: SemanticAcceptanceReport, output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and text semantic acceptance reports under *output_dir*."""
    json_path = write_json_report(report, output_dir / "semantic_acceptance_summary.json")
    text_path = write_text_summary(report, output_dir / "semantic_acceptance_summary.txt")
    return json_path, text_path


def _count_tars(root: Path) -> int:
    return sum(1 for _ in (root / "local_tars").rglob("*.tar"))


def _read_csv_rows(
    path: Path,
    label: str,
    errors: list[str],
    *,
    required_columns: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    if not path.exists():
        errors.append(f"{label} is missing: {path}")
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            rows = list(reader)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        errors.append(f"{label} could not be read: {exc}")
        return []
    if not reader.fieldnames:
        errors.append(f"{label} has no header")
        return []
    missing_columns = [column for column in required_columns if column not in reader.fieldnames]
    if missing_columns:
        errors.append(f"{label} is missing required columns: {', '.join(missing_columns)}")
    if None in {key for row in rows for key in row}:
        errors.append(f"{label} has malformed rows with too many columns")
        return []
    return rows


def _read_failed_models_lines(path: Path, label: str, errors: list[str]) -> list[str]:
    """Read a failed-models TSV, treating a missing file as zero recorded failures.

    The worker writes ``failed_models.tsv`` only when models fail (and removes
    clean per-shard failure files), so an absent file is the zero-failure case,
    not a structural defect. Unreadable content remains an error. A run that
    does record failures is still caught by the expected-row count and the
    baseline-vs-candidate count/set comparisons.
    """
    if not path.exists():
        return []
    return _read_nonempty_lines(path, label, errors)


def _read_nonempty_lines(path: Path, label: str, errors: list[str]) -> list[str]:
    if not path.exists():
        errors.append(f"{label} is missing: {path}")
        return []
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"{label} could not be read: {exc}")
        return []


def _selected_ids_from_analysis(rows: list[dict[str, str]]) -> set[str]:
    selected: set[str] = set()
    for row in rows:
        if row.get("passes_quality_threshold", "").strip().lower() == "true":
            model_id = row.get("model_id", "").strip()
            if model_id:
                selected.add(model_id)
    return selected


def _semantic_rows(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[tuple[str, ...]]:
    return sorted(tuple(row.get(key, "") for key in keys) for row in rows)


def _check_expected(name: str, actual: int, expected: int | None, errors: list[str]) -> None:
    if expected is not None and actual != expected:
        errors.append(f"{name} expected {expected}, got {actual}")


__all__ = [
    "SemanticAcceptanceReport",
    "compare_semantic_acceptance",
    "render_semantic_acceptance_report",
    "write_semantic_acceptance_report",
]
