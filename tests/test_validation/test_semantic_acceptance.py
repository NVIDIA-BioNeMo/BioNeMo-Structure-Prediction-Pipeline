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

"""Tests for semantic acceptance checks."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from click.testing import CliRunner

from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.validation.semantic_acceptance import compare_semantic_acceptance


def test_compare_semantic_acceptance_passes_matching_outputs(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline, run_name="upstream")
    _write_output(candidate, run_name="candidate", with_parquet=True, with_selected_file=True)

    report = compare_semantic_acceptance(
        baseline,
        candidate,
        expected_tar_count=2,
        expected_local_tars_rows=2,
        expected_failed_rows=1,
        expected_analysis_rows=3,
        expected_selected_ids=2,
    )

    assert report.ok
    assert report.errors == ()
    assert report.candidate_parquet_rows == 3
    assert report.candidate_selected_ids_file_rows == 2


def test_compare_semantic_acceptance_zero_failures_without_failed_tsv(tmp_path: Path) -> None:
    """Fixed-fork contract: zero failures means no failed_models.tsv on either side."""
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline, run_name="upstream", with_failed_file=False)
    _write_output(candidate, run_name="candidate", with_parquet=True, with_selected_file=True, with_failed_file=False)

    report = compare_semantic_acceptance(
        baseline,
        candidate,
        expected_tar_count=2,
        expected_local_tars_rows=2,
        expected_failed_rows=0,
        expected_analysis_rows=3,
        expected_selected_ids=2,
    )

    assert report.ok
    assert report.errors == ()
    assert report.baseline_failed_rows == 0
    assert report.candidate_failed_rows == 0


def test_compare_semantic_acceptance_failed_rows_still_flagged_when_zero_expected(tmp_path: Path) -> None:
    """A run that does record failures must still trip the zero-failure gate."""
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline, run_name="upstream", with_failed_file=False)
    _write_output(
        candidate,
        run_name="candidate",
        with_parquet=True,
        with_selected_file=True,
        failed=("AF-1\tfailure", "AF-2\tfailure"),
    )

    report = compare_semantic_acceptance(baseline, candidate, expected_failed_rows=0)

    assert not report.ok
    assert "candidate failed rows expected 0, got 2" in report.errors
    assert "failed row counts differ: baseline=0 candidate=2" in report.errors
    assert "failed model sets differ" in report.errors


def test_compare_semantic_acceptance_tar_manifest_size_bytes_drift_tolerated(tmp_path: Path) -> None:
    """Compressed-size drift across image builds is benign; identity keys still compared."""
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline, run_name="upstream")
    _write_output(candidate, run_name="candidate", with_parquet=True, with_selected_file=True)
    candidate_csv = candidate / "local_tars.csv"
    candidate_csv.write_text(
        candidate_csv.read_text(encoding="utf-8")
        .replace(",12,zstd-members", ",99,zstd-members")
        .replace(",8,zstd-members", ",88,zstd-members"),
        encoding="utf-8",
    )

    report = compare_semantic_acceptance(baseline, candidate, expected_failed_rows=1)

    assert report.ok
    assert "local_tars.csv semantic rows differ" not in report.errors


def test_compare_semantic_acceptance_tar_manifest_structural_drift_still_flagged(tmp_path: Path) -> None:
    """Identity/structure drift (e.g. member_count) must still fail the comparison."""
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline, run_name="upstream")
    _write_output(candidate, run_name="candidate", with_parquet=True, with_selected_file=True)
    candidate_csv = candidate / "local_tars.csv"
    candidate_csv.write_text(
        candidate_csv.read_text(encoding="utf-8").replace(",3,12,zstd-members", ",4,12,zstd-members"),
        encoding="utf-8",
    )

    report = compare_semantic_acceptance(baseline, candidate)

    assert not report.ok
    assert "local_tars.csv semantic rows differ" in report.errors


def test_compare_semantic_acceptance_detects_selected_id_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline, run_name="upstream")
    _write_output(candidate, run_name="candidate", selected_flags=("true", "false", "false"), with_parquet=True)

    report = compare_semantic_acceptance(baseline, candidate, expected_selected_ids=2)

    assert not report.ok
    assert "candidate selected IDs expected 2, got 1" in report.errors
    assert "selected ID sets from analysis CSV differ" in report.errors


def test_validate_semantic_acceptance_cli_writes_report_and_fails_strict(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    report_dir = tmp_path / "report"
    _write_output(baseline, run_name="upstream")
    _write_output(candidate, run_name="candidate", failed=("AF-9\tfailure",), with_parquet=True)

    result = CliRunner().invoke(
        cli,
        [
            "validate",
            "semantic-acceptance",
            "--baseline-dir",
            str(baseline),
            "--candidate-dir",
            str(candidate),
            "--expected-failed-rows",
            "1",
            "--write-report",
            str(report_dir),
            "--strict",
        ],
    )

    assert result.exit_code == 1
    assert '"ok": false' in result.output
    assert "failed model sets differ" in result.output
    assert (report_dir / "semantic_acceptance_summary.json").exists()
    assert (report_dir / "semantic_acceptance_summary.txt").exists()


def test_compare_semantic_acceptance_reports_missing_artifacts_without_traceback(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline, run_name="upstream")
    candidate.mkdir()

    report = compare_semantic_acceptance(baseline, candidate, require_candidate_parquet=True)

    assert not report.ok
    assert report.candidate_local_tars_csv_rows == 0
    assert report.candidate_failed_rows == 0
    assert report.candidate_analysis_csv_rows == 0
    assert any("candidate local_tars.csv is missing" in error for error in report.errors)
    # A missing failed_models.tsv is the zero-failure case (the worker writes
    # it only when models fail), not an error.
    assert not any("failed_models.tsv is missing" in error for error in report.errors)
    assert any("candidate analysis_metadata.csv is missing" in error for error in report.errors)
    assert "candidate high_quality_model_ids.txt is missing" in report.errors
    assert "candidate analysis parquet is missing" in report.errors


def test_compare_semantic_acceptance_reports_malformed_csv_and_parquet_without_traceback(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline, run_name="upstream")
    _write_output(candidate, run_name="candidate", with_selected_file=True)
    (candidate / "local_tars.csv").write_text("tar_name,source_archive\nbatch_0.tar,archive.tar.lz4,extra\n")
    (candidate / "analysis_metadata.csv").write_text('model_id,passes_quality_threshold\nAF-1,"true\n')
    (candidate / "analysis_metadata.parquet").write_text("not parquet")

    report = compare_semantic_acceptance(baseline, candidate, require_candidate_parquet=True)

    assert not report.ok
    assert any("candidate local_tars.csv is missing required columns" in error for error in report.errors)
    assert "candidate local_tars.csv has malformed rows with too many columns" in report.errors
    assert any("candidate analysis_metadata.csv could not be read" in error for error in report.errors)
    assert any("candidate analysis parquet could not be read" in error for error in report.errors)


def test_validate_semantic_acceptance_cli_writes_report_on_malformed_strict_failure(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    report_dir = tmp_path / "report"
    _write_output(baseline, run_name="upstream")
    _write_output(candidate, run_name="candidate", with_selected_file=True)
    (candidate / "analysis_metadata.csv").write_text('model_id,passes_quality_threshold\nAF-1,"true\n')

    result = CliRunner().invoke(
        cli,
        [
            "validate",
            "semantic-acceptance",
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
    assert "candidate analysis_metadata.csv could not be read" in result.output
    assert (report_dir / "semantic_acceptance_summary.json").exists()


def _write_output(
    root: Path,
    *,
    run_name: str,
    selected_flags: tuple[str, str, str] = ("true", "false", "true"),
    failed: tuple[str, ...] = ("AF-2\tfailure",),
    with_parquet: bool = False,
    with_selected_file: bool = False,
    with_failed_file: bool = True,
) -> None:
    (root / "local_tars" / "shard_1").mkdir(parents=True)
    (root / "local_tars" / "metadata").mkdir(parents=True)
    (root / "local_tars" / "shard_1" / "batch_0.tar").write_text("tar")
    (root / "local_tars" / "metadata" / "shard_1_metadata.tar").write_text("tar")
    (root / "local_tars.csv").write_text(
        "\n".join(
            [
                "timestamp,run_name,tar_type,s3_uri,tar_name,source_archive,shard_id,task_id,batch_id,member_count,size_bytes,compression",
                f"t,{run_name},batch,file://x,batch_0.tar,archive.tar.lz4,1,0,0,3,12,zstd-members",
                f"t,{run_name},metadata,file://x,shard_1_metadata.tar,archive.tar.lz4,1,1,metadata,2,8,zstd-members",
            ]
        )
        + "\n"
    )
    if with_failed_file:
        (root / "failed_models.tsv").write_text("\n".join(failed) + "\n")
    rows = [
        ("AF-1", "orig-1", "uploaded", selected_flags[0], "", "0.8", "0.3"),
        ("AF-2", "orig-2", "pipeline_failed", selected_flags[1], "failure", "", ""),
        ("AF-3", "orig-3", "uploaded", selected_flags[2], "", "0.9", "0.4"),
    ]
    analysis_header = (
        "timestamp,run_name,model_id,original_id,upload_status,passes_quality_threshold,"
        "ipsae_max,pdockq2_max,quality_ipsae_threshold,quality_pdockq2_threshold,source_archive,"
        "shard_id,task_id,batch_id,batch_started_at,batch_finished_at,failure_reason,"
        "expected_output_files_json,scores_json"
    )
    analysis_lines = [analysis_header]
    for model_id, original_id, status, selected, reason, ipsae, pdockq2 in rows:
        analysis_lines.append(
            f"t,{run_name},{model_id},{original_id},{status},{selected},{ipsae},{pdockq2},0.6,0.23,"
            f"archive.tar.lz4,1,0,0,start,end,{reason},[],{{}}"
        )
    (root / "analysis_metadata.csv").write_text("\n".join(analysis_lines) + "\n")
    if with_parquet:
        pq.write_table(pa.table({"model_id": [row[0] for row in rows]}), root / "analysis_metadata.parquet")
    if with_selected_file:
        selected_ids = [
            model_id for model_id, *_rest, selected, _reason, _ipsae, _pdockq2 in rows if selected == "true"
        ]
        (root / "high_quality_model_ids.txt").write_text("\n".join(selected_ids) + "\n")
