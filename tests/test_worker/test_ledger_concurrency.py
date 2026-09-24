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

from __future__ import annotations

import csv
import multiprocessing
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bspp.orchestration.runtime.worker import (
    ANALYSIS_METADATA_FIELDS,
    PROCESSING_LOG_FIELDS,
    TAR_MANIFEST_FIELDS,
    FailedModelRecord,
    append_failed_model_records,
    append_tar_manifest_rows,
    native_worker,
)

PROCESS_COUNT = 4
ROWS_PER_PROCESS = 12


def test_failed_model_records_append_safely_across_processes(tmp_path: Path) -> None:
    _run_concurrent_writers(_write_failed_model_records, tmp_path)

    global_failed = tmp_path / "failed_models.tsv"
    shard_failed = tmp_path / "shard_0" / "failed_models.tsv"
    expected_lines = {
        f"AF-{process_idx:04d}-{row_idx:04d}\tmodel\treason {process_idx}-{row_idx}\n"
        for process_idx in range(PROCESS_COUNT)
        for row_idx in range(ROWS_PER_PROCESS)
    }

    assert global_failed.read_text(encoding="utf-8") == shard_failed.read_text(encoding="utf-8")
    assert set(global_failed.read_text(encoding="utf-8").splitlines(keepends=True)) == expected_lines


def test_processing_log_append_writes_one_header_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "processing_log.csv"

    _run_concurrent_writers(_write_processing_log_rows, tmp_path)

    assert _header_count(path, PROCESSING_LOG_FIELDS) == 1
    rows = _csv_rows(path)
    assert len(rows) == PROCESS_COUNT * ROWS_PER_PROCESS
    assert {row["model_id"] for row in rows} == _expected_model_ids()
    assert {row["upload_status"] for row in rows} == {"pipeline_failed"}


def test_local_tars_manifest_append_writes_one_header_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "local_tars.csv"

    _run_concurrent_writers(_write_local_tar_manifest_rows, tmp_path)

    assert _header_count(path, TAR_MANIFEST_FIELDS) == 1
    rows = _csv_rows(path)
    assert len(rows) == PROCESS_COUNT * ROWS_PER_PROCESS
    assert {row["tar_name"] for row in rows} == {
        f"shard_{process_idx}_batch_{row_idx}.tar"
        for process_idx in range(PROCESS_COUNT)
        for row_idx in range(ROWS_PER_PROCESS)
    }
    assert {row["tar_type"] for row in rows} == {"batch"}


def test_analysis_metadata_append_writes_one_header_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "analysis_metadata.csv"

    _run_concurrent_writers(_write_analysis_metadata_rows, tmp_path)

    assert _header_count(path, ANALYSIS_METADATA_FIELDS) == 1
    rows = _csv_rows(path)
    assert len(rows) == PROCESS_COUNT * ROWS_PER_PROCESS
    assert {row["model_id"] for row in rows} == _expected_model_ids()
    assert {row["scores_json"] for row in rows} == {"{}"}


def _run_concurrent_writers(
    target: Callable[[str, int, int, Any], None],
    tmp_path: Path,
    *,
    process_count: int = PROCESS_COUNT,
    rows_per_process: int = ROWS_PER_PROCESS,
) -> None:
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    processes = [
        context.Process(target=target, args=(str(tmp_path), process_idx, rows_per_process, start_event))
        for process_idx in range(process_count)
    ]

    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=15)

    timed_out = [process for process in processes if process.is_alive()]
    for process in timed_out:
        process.terminate()
    for process in timed_out:
        process.join(timeout=5)

    assert not timed_out
    assert all(process.exitcode == 0 for process in processes)


def _write_failed_model_records(
    base_dir: str,
    process_idx: int,
    rows_per_process: int,
    start_event: Any,
) -> None:
    start_event.wait()
    result = append_failed_model_records(
        (
            FailedModelRecord(
                model_id=f"AF-{process_idx:04d}-{row_idx:04d}",
                stage="model",
                reason=f"reason {process_idx}-{row_idx}",
            )
            for row_idx in range(rows_per_process)
        ),
        global_failed_path=Path(base_dir) / "failed_models.tsv",
        shard_failed_path=Path(base_dir) / "shard_0" / "failed_models.tsv",
        lock_path=Path(base_dir) / "failed_models.tsv.lock",
    )
    assert result.row_count == rows_per_process


def _write_processing_log_rows(base_dir: str, process_idx: int, rows_per_process: int, start_event: Any) -> None:
    start_event.wait()
    native_worker._append_processing_log_rows(
        Path(base_dir) / "processing_log.csv",
        (
            {
                "model_id": f"AF-{process_idx:04d}-{row_idx:04d}",
                "original_id": f"AF-{process_idx:04d}-{row_idx:04d}",
                "upload_status": "pipeline_failed",
                "tar_of_origin": f"archive_{process_idx}.tar.lz4",
                "shard_id": process_idx,
                "task_id": process_idx,
                "failure_reason": f"reason {process_idx}-{row_idx}",
                "timestamp": "2026-05-23T00:00:00+00:00",
            }
            for row_idx in range(rows_per_process)
        ),
    )


def _write_local_tar_manifest_rows(base_dir: str, process_idx: int, rows_per_process: int, start_event: Any) -> None:
    start_event.wait()
    append_tar_manifest_rows(
        Path(base_dir) / "local_tars.csv",
        (
            {
                "timestamp": "2026-05-23T00:00:00+00:00",
                "run_name": "run",
                "tar_type": "batch",
                "s3_uri": f"file:///tmp/shard_{process_idx}_batch_{row_idx}.tar",
                "tar_name": f"shard_{process_idx}_batch_{row_idx}.tar",
                "source_archive": f"archive_{process_idx}.tar.lz4",
                "shard_id": process_idx,
                "task_id": process_idx,
                "batch_id": row_idx,
                "member_count": 7,
                "size_bytes": 1024 + row_idx,
                "compression": "zstd-members",
            }
            for row_idx in range(rows_per_process)
        ),
    )


def _write_analysis_metadata_rows(base_dir: str, process_idx: int, rows_per_process: int, start_event: Any) -> None:
    start_event.wait()
    native_worker._append_analysis_metadata_rows(
        Path(base_dir) / "analysis_metadata.csv",
        (
            {
                "timestamp": "2026-05-23T00:00:00+00:00",
                "run_name": "run",
                "model_id": f"AF-{process_idx:04d}-{row_idx:04d}",
                "original_id": f"AF-{process_idx:04d}-{row_idx:04d}",
                "upload_status": "uploaded",
                "passes_quality_threshold": "false",
                "ipsae_max": "",
                "pdockq2_max": "",
                "quality_ipsae_threshold": 0.5,
                "quality_pdockq2_threshold": 0.23,
                "source_archive": f"archive_{process_idx}.tar.lz4",
                "shard_id": process_idx,
                "task_id": process_idx,
                "batch_id": row_idx,
                "batch_started_at": "2026-05-23T00:00:00+00:00",
                "batch_finished_at": "2026-05-23T00:00:01+00:00",
                "failure_reason": "",
                "expected_output_files_json": "[]",
                "scores_json": "{}",
            }
            for row_idx in range(rows_per_process)
        ),
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _header_count(path: Path, fieldnames: tuple[str, ...]) -> int:
    header = ",".join(fieldnames)
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line == header)


def _expected_model_ids() -> set[str]:
    return {
        f"AF-{process_idx:04d}-{row_idx:04d}"
        for process_idx in range(PROCESS_COUNT)
        for row_idx in range(ROWS_PER_PROCESS)
    }
