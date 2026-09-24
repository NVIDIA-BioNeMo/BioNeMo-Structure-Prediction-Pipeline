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

"""Tests for Phase 2 task 278 parity checks."""

from __future__ import annotations

import csv
import tarfile
from pathlib import Path

from click.testing import CliRunner

from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.validation.phase2_parity import compare_phase2_parity


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_tsv(path: Path, rows: list[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerows(rows)


def _write_tar(path: Path, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.stem}-staging"
    staging.mkdir()
    for relative_path, content in files.items():
        source = staging / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(content)
    with tarfile.open(path, "w") as archive:
        for source in sorted(p for p in staging.rglob("*") if p.is_file()):
            archive.add(source, arcname=str(source.relative_to(staging)))


def _make_phase2_dir(root: Path, *, local_rows: int = 2) -> None:
    _write_csv(
        root / "local_tars.csv",
        ["tar_path", "model_count", "sha256"],
        [
            {"tar_path": f"local_tars/batch_{idx}.tar", "model_count": "1", "sha256": f"hash-{idx}"}
            for idx in range(local_rows)
        ],
    )
    _write_csv(
        root / "analysis_metadata.csv",
        ["model_id", "ipsae", "pdockq2"],
        [
            {"model_id": "AF-0000000000000001", "ipsae": "0.7", "pdockq2": "0.3"},
            {"model_id": "AF-0000000000000002", "ipsae": "0.5", "pdockq2": "0.2"},
        ],
    )
    _write_csv(
        root / "processing_log.csv",
        ["model_id", "status"],
        [{"model_id": "AF-0000000000000001", "status": "ok"}],
    )
    _write_tsv(root / "failed_models.tsv", [("AF-0000000000000003", "extract", "missing meta")])
    (root / "high_quality_model_ids.txt").write_text("AF-0000000000000001\n\nAF-0000000000000004\n")
    _write_csv(
        root / "model_tar_index.csv",
        ["model_id", "tar_path"],
        [{"model_id": "AF-0000000000000001", "tar_path": "local_tars/batch_0.tar"}],
    )
    _write_tar(root / "local_tars" / "batch_0.tar", {"AF-0000000000000001.pdb": "MODEL\n"})
    _write_tar(root / "local_tars" / "batch_1.tar", {"AF-0000000000000002.pdb": "MODEL\n"})


def test_compare_phase2_parity_happy_path(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _make_phase2_dir(baseline)
    _make_phase2_dir(candidate)

    report = compare_phase2_parity(baseline, candidate)

    assert report.ok
    assert report.mismatches == ()
    assert report.local_tars_csv.baseline_columns == ("tar_path", "model_count", "sha256")
    assert report.local_tars_csv.baseline_rows == 2
    assert report.local_tars.baseline_files == ("batch_0.tar", "batch_1.tar")
    assert report.local_tar_member_counts.baseline_counts == {"batch_0.tar": 1, "batch_1.tar": 1}
    assert all(item.ok for item in report.sha256)
    assert all(item.ok for item in report.multisets)
    assert {item.relative_path: item.baseline_rows for item in report.row_counts} == {
        "analysis_metadata.csv": 2,
        "processing_log.csv": 1,
        "failed_models.tsv": 1,
        "high_quality_model_ids.txt": 2,
        "model_tar_index.csv": 1,
    }
    rendered = report.to_redacted_dict()
    assert rendered["ok"] is True
    assert rendered["profile"] == "exact"
    assert rendered["mismatches"] == []


def test_compare_phase2_parity_fast_profile_skips_optional_index_and_volatile_hashes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _make_phase2_dir(baseline)
    _make_phase2_dir(candidate)
    (candidate / "model_tar_index.csv").unlink()
    _write_csv(
        candidate / "processing_log.csv",
        ["model_id", "status"],
        [{"model_id": "AF-0000000000000001", "status": "timestamp-only-drift"}],
    )

    report = compare_phase2_parity(baseline, candidate, profile="fast")

    assert report.ok
    assert report.sha256 == ()
    assert "model_tar_index.csv" not in {item.relative_path for item in report.required}
    assert "model_tar_index.csv" not in {item.relative_path for item in report.row_counts}
    assert report.to_redacted_dict()["profile"] == "fast"


def test_compare_phase2_parity_fast_profile_detects_model_set_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _make_phase2_dir(baseline)
    _make_phase2_dir(candidate)
    _write_csv(
        candidate / "analysis_metadata.csv",
        ["model_id", "ipsae", "pdockq2"],
        [
            {"model_id": "AF-0000000000000001", "ipsae": "0.7", "pdockq2": "0.3"},
            {"model_id": "AF-CHANGED", "ipsae": "0.5", "pdockq2": "0.2"},
        ],
    )

    report = compare_phase2_parity(baseline, candidate, profile="fast")

    assert not report.ok
    analysis_keys = next(item for item in report.multisets if item.relative_path == "analysis_metadata.csv")
    assert analysis_keys.only_baseline == ("AF-0000000000000002",)
    assert analysis_keys.only_candidate == ("AF-CHANGED",)


def test_compare_phase2_parity_reports_missing_required_file(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _make_phase2_dir(baseline)
    _make_phase2_dir(candidate)
    (candidate / "processing_log.csv").unlink()

    report = compare_phase2_parity(baseline, candidate)

    assert not report.ok
    missing = next(item for item in report.required if item.relative_path == "processing_log.csv")
    assert missing.baseline_exists is True
    assert missing.candidate_exists is False
    assert missing in report.mismatches


def test_compare_phase2_parity_detects_local_tars_csv_schema_and_row_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _make_phase2_dir(baseline)
    _make_phase2_dir(candidate)
    _write_csv(
        candidate / "local_tars.csv",
        ["tar_path", "model_count", "extra"],
        [{"tar_path": "local_tars/batch_0.tar", "model_count": "1", "extra": "x"}],
    )

    report = compare_phase2_parity(baseline, candidate)

    assert not report.local_tars_csv.ok
    assert not report.local_tars_csv.schema_ok
    assert not report.local_tars_csv.rows_ok
    assert report.local_tars_csv in report.mismatches


def test_compare_phase2_parity_detects_tar_file_list_and_hash_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _make_phase2_dir(baseline)
    _make_phase2_dir(candidate)
    (candidate / "local_tars" / "batch_1.tar").unlink()
    (candidate / "high_quality_model_ids.txt").write_text("AF-0000000000000001\nAF-CHANGED\n")

    report = compare_phase2_parity(baseline, candidate)

    assert not report.ok
    assert not report.local_tars.file_list_ok
    assert report.local_tars.baseline_files == ("batch_0.tar", "batch_1.tar")
    assert report.local_tars.candidate_files == ("batch_0.tar",)

    hq_hash = next(item for item in report.sha256 if item.relative_path == "high_quality_model_ids.txt")
    assert not hq_hash.ok
    assert hq_hash.baseline_sha256 != hq_hash.candidate_sha256


def test_compare_phase2_parity_detects_normalized_row_count_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _make_phase2_dir(baseline)
    _make_phase2_dir(candidate)
    (candidate / "failed_models.tsv").write_text("AF-1\textract\ta\n\nAF-2\textract\tb\n")

    report = compare_phase2_parity(baseline, candidate)

    failed_count = next(item for item in report.row_counts if item.relative_path == "failed_models.tsv")
    assert not failed_count.ok
    assert failed_count.baseline_rows == 1
    assert failed_count.candidate_rows == 2


def test_validate_phase2_parity_cli_writes_report_and_fails_strict_on_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    report_dir = tmp_path / "report"
    _make_phase2_dir(baseline)
    _make_phase2_dir(candidate)
    (candidate / "failed_models.tsv").write_text("AF-1\textract\ta\n\nAF-2\textract\tb\n")
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "validate",
            "phase2-parity",
            "--baseline-dir",
            str(baseline),
            "--candidate-dir",
            str(candidate),
            "--write-report",
            str(report_dir),
            "--strict",
        ],
    )

    assert result.exit_code == 1
    assert '"ok": false' in result.output
    assert (report_dir / "wp8" / "phase2_parity_report.json").exists()
    assert (report_dir / "wp8" / "phase2_parity_report.txt").exists()
