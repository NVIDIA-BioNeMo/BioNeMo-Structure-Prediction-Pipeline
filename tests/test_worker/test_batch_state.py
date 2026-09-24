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

from pathlib import Path

from bspp.orchestration.runtime.worker import (
    FailedModelRecord,
    append_failed_model_records,
    batch_marker_name,
    flush_local_failed_models,
    load_failed_model_ids,
    parse_failed_model_reasons,
    record_batch_failed,
    record_prefiltered_failed,
    remove_shard_failed_file_if_clean,
    should_skip_batch,
    write_batch_marker,
)


def test_batch_marker_names_and_resume_decision(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shard_7"
    done_marker = write_batch_marker(shard_dir, 3, ["a.cif", "b.json"])

    assert done_marker.name == ".batch_3_done"
    assert done_marker.read_text() == "a.cif\nb.json\n"
    assert batch_marker_name(3, failed_model_count=1) == ".batch_3_partial_uploaded"
    assert batch_marker_name(3, failed_model_count=1, retry_failed_only=True) == (
        ".retry_failed_batch_3_partial_uploaded"
    )
    assert should_skip_batch(done_marker) is True
    assert should_skip_batch(done_marker, retry_failed_only=True) is False
    assert should_skip_batch(done_marker, local_tar_restart_requires_rerun=True) is False


def test_append_failed_model_records_dual_writes_under_lock(tmp_path: Path) -> None:
    global_failed = tmp_path / "failed_models.tsv"
    shard_failed = tmp_path / "shard_1" / "failed_models.tsv"

    result = append_failed_model_records(
        (
            FailedModelRecord("AF-0000000000000001", "model", "bad scores"),
            FailedModelRecord("AF-0000000000000002", "batch_crash", "pipeline_exit_1"),
        ),
        global_failed_path=global_failed,
        shard_failed_path=shard_failed,
        lock_path=tmp_path / "failed.lock",
    )

    assert result.row_count == 2
    assert global_failed.read_text() == shard_failed.read_text()
    assert load_failed_model_ids(global_failed) == frozenset({"AF-0000000000000001", "AF-0000000000000002"})


def test_flush_local_failed_models_consumes_local_file_and_parses_reasons(tmp_path: Path) -> None:
    local_failed = tmp_path / "work" / "failed_models.tsv"
    local_failed.parent.mkdir()
    local_failed.write_text(
        "AF-0000000000000001\tmodel\tbad scores\n"
        "AF-0000000000000001\tmodel\tbad scores\n"
        "AF-0000000000000001\tupload\tfallback\n",
    )

    reasons = flush_local_failed_models(
        local_failed,
        global_failed_path=tmp_path / "failed_models.tsv",
        shard_failed_path=tmp_path / "shard_1" / "failed_models.tsv",
    )

    assert not local_failed.exists()
    assert reasons == {"AF-0000000000000001": "model: bad scores; upload: fallback"}


def test_record_batch_failed_writes_every_model_as_batch_crash(tmp_path: Path) -> None:
    global_failed = tmp_path / "failed_models.tsv"
    shard_failed = tmp_path / "shard_0" / "failed_models.tsv"

    result = record_batch_failed(
        ["AF-0000000000000001", "AF-0000000000000002"],
        reason="pipeline_exit_9",
        global_failed_path=global_failed,
        shard_failed_path=shard_failed,
    )

    expected = "AF-0000000000000001\tbatch_crash\tpipeline_exit_9\nAF-0000000000000002\tbatch_crash\tpipeline_exit_9\n"
    assert result.row_count == 2
    assert global_failed.read_text() == expected
    assert shard_failed.read_text() == expected


def test_record_prefiltered_failed_writes_input_validation_rows(tmp_path: Path) -> None:
    global_failed = tmp_path / "failed_models.tsv"
    shard_failed = tmp_path / "shard_0" / "failed_models.tsv"

    result = record_prefiltered_failed(
        [("AF-0000000000000001", "null in input json")],
        global_failed_path=global_failed,
        shard_failed_path=shard_failed,
    )

    assert result.row_count == 1
    assert global_failed.read_text() == "AF-0000000000000001\tinput_validation\tnull in input json\n"
    assert shard_failed.read_text() == global_failed.read_text()


def test_parse_failed_model_reasons_handles_short_rows() -> None:
    assert parse_failed_model_reasons("AF-1\tstage only\nAF-2\n") == {
        "AF-1": "stage only",
        "AF-2": "model failure",
    }


def test_remove_shard_failed_file_only_when_clean(tmp_path: Path) -> None:
    failed_path = tmp_path / "shard_0" / "failed_models.tsv"
    failed_path.parent.mkdir()
    failed_path.write_text("old\n")

    assert remove_shard_failed_file_if_clean(failed_path, had_failures=True) is False
    assert failed_path.exists()
    assert remove_shard_failed_file_if_clean(failed_path, had_failures=False) is True
    assert not failed_path.exists()
