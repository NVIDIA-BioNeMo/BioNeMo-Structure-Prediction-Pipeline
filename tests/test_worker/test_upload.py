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

import json
from collections.abc import Sequence
from pathlib import Path

from bspp.orchestration.runtime.worker import (
    clean_batch_outputs,
    cleanup_success_outputs,
    collect_flat_upload_files,
    copy_batch_flat_to_success,
    plan_s3_upload_transfers,
    resolve_upload_slot_count,
    try_upload_then_lustre_fallback,
    upload_files_to_s3,
    upload_single_file_to_s3,
    upload_slot,
    write_batch_marker,
    write_s5cmd_command_file,
    write_uploaded_marker,
)


def _write_output_tree(root: Path, model_ids: tuple[str, ...] = ("AF-0000000000000001",)) -> None:
    for model_id in model_ids:
        for dirname, filename in {
            "modelcif": f"{model_id}-model_v1.cif",
            "modelpdb": f"{model_id}-model_v1.pdb",
            "bcif": f"{model_id}-model_v1.bcif",
            "scores": f"{model_id}-confidence_v1.json",
            "clash_interface_analysis": f"{model_id}-model_v1_clashes.json",
        }.items():
            path = root / dirname / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(filename)
    search = root / "metadata" / "search" / "AF-metadata-1-of-1.json"
    search.parent.mkdir(parents=True, exist_ok=True)
    search.write_text("{}")


def test_collect_flat_upload_files_preserves_legacy_destination_layout(tmp_path: Path) -> None:
    _write_output_tree(tmp_path)

    pairs = collect_flat_upload_files(tmp_path, ["AF-0000000000000001"])
    relative_paths = [relative.as_posix() for _source, relative in pairs]

    assert "AF-0000000000000001-model_v1.cif" in relative_paths
    assert "metadata/clashes_and_interfaces_granular/AF-0000000000000001-model_v1_clashes.json" in relative_paths
    assert "metadata/search/AF-metadata-1-of-1.json" in relative_paths


def test_write_s5cmd_command_file_and_upload_retry_cleanup(tmp_path: Path) -> None:
    _write_output_tree(tmp_path)
    transfers = plan_s3_upload_transfers(tmp_path, "s3://bucket/prefix", ["AF-0000000000000001"])
    command_file = tmp_path / "_s3_upload_commands.txt"
    calls: list[Sequence[str]] = []

    assert write_s5cmd_command_file(command_file, transfers) == len(transfers)
    assert command_file.read_text().startswith("cp ")

    def runner(argv: Sequence[str]) -> int:
        calls.append(tuple(argv))
        return 1 if len(calls) == 1 else 0

    result = upload_files_to_s3(
        tmp_path,
        "s3://bucket/prefix",
        s5cmd_path="s5cmd",
        batch_ids=["AF-0000000000000001"],
        runner=runner,
        retry_delay_seconds=0,
        sleep=lambda _seconds: None,
    )

    assert result.success is True
    assert result.attempts == 2
    assert not command_file.exists()
    assert calls == [
        ("s5cmd", "--numworkers", "256", "run", str(command_file)),
        ("s5cmd", "--numworkers", "256", "run", str(command_file)),
    ]


def test_upload_single_file_to_s3_uses_cp_command(tmp_path: Path) -> None:
    tar_path = tmp_path / "shard_0_batch_0.tar"
    tar_path.write_text("tar")
    calls: list[Sequence[str]] = []

    result = upload_single_file_to_s3(
        tar_path,
        "s3://bucket/tars/shard_0_batch_0.tar",
        s5cmd_path="s5cmd",
        runner=lambda argv: calls.append(tuple(argv)) or 0,
    )

    assert result.success is True
    assert calls == [("s5cmd", "--numworkers", "16", "cp", str(tar_path), "s3://bucket/tars/shard_0_batch_0.tar")]


def test_upload_files_to_s3_retry_exhaustion_cleans_command_file(tmp_path: Path) -> None:
    _write_output_tree(tmp_path)
    calls: list[Sequence[str]] = []

    result = upload_files_to_s3(
        tmp_path,
        "s3://bucket/prefix",
        s5cmd_path="s5cmd",
        batch_ids=["AF-0000000000000001"],
        runner=lambda argv: calls.append(tuple(argv)) or 1,
        retry_delay_seconds=0,
        sleep=lambda _seconds: None,
    )

    assert result.success is False
    assert result.attempts == 3
    assert result.uploaded_files == ()
    assert not (tmp_path / "_s3_upload_commands.txt").exists()
    assert len(calls) == 3


def test_upload_single_file_to_s3_retry_exhaustion(tmp_path: Path) -> None:
    tar_path = tmp_path / "shard_0_batch_0.tar"
    tar_path.write_text("tar")
    calls: list[Sequence[str]] = []

    result = upload_single_file_to_s3(
        tar_path,
        "s3://bucket/tars/shard_0_batch_0.tar",
        s5cmd_path="s5cmd",
        runner=lambda argv: calls.append(tuple(argv)) or 1,
        retry_delay_seconds=0,
        sleep=lambda _seconds: None,
    )

    assert result.success is False
    assert result.attempts == 3
    assert result.uploaded_files == ()
    assert len(calls) == 3


def test_copy_flat_fallback_and_clean_batch_outputs(tmp_path: Path) -> None:
    _write_output_tree(tmp_path, ("AF-0000000000000001", "AF-0000000000000002"))

    copied = copy_batch_flat_to_success(tmp_path, tmp_path / "success_outputs", ["AF-0000000000000001"])
    assert tmp_path / "success_outputs" / "AF-0000000000000001-model_v1.cif" in copied
    assert (
        tmp_path
        / "success_outputs"
        / "metadata"
        / "clashes_and_interfaces_granular"
        / "AF-0000000000000001-model_v1_clashes.json"
    ).exists()

    removed = clean_batch_outputs(tmp_path, ["AF-0000000000000001"])
    assert removed == 5
    assert (tmp_path / "modelcif" / "AF-0000000000000002-model_v1.cif").exists()


def test_try_upload_then_lustre_fallback_copies_outputs_and_cleans_scratch(tmp_path: Path) -> None:
    model_id = "AF-0000000000000001"
    _write_output_tree(tmp_path / "work", (model_id,))
    for dirname in ("model_jsons", "chain_jsons"):
        metadata = tmp_path / "work" / dirname / f"{model_id}.json"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text("{}")

    result = try_upload_then_lustre_fallback(
        tmp_path / "work",
        tmp_path / "shard_0" / "success_outputs",
        "s3://bucket/prefix",
        s5cmd_path="s5cmd",
        batch_ids=[model_id],
        runner=lambda _argv: 1,
        retry_delay_seconds=0,
        sleep=lambda _seconds: None,
    )

    assert result.fallback_used is True
    assert result.status == "upload_failed_lustre_fallback"
    assert result.upload_result.success is False
    assert tmp_path / "shard_0" / "success_outputs" / f"{model_id}-model_v1.cif" in result.copied_files
    assert (tmp_path / "shard_0" / "success_outputs" / f"{model_id}-model_v1.cif").exists()
    assert not (tmp_path / "work" / "modelcif" / f"{model_id}-model_v1.cif").exists()
    assert not (tmp_path / "work" / "model_jsons" / f"{model_id}.json").exists()

    marker = write_uploaded_marker(
        tmp_path / "shard_0",
        s3_prefix="s3://bucket/prefix",
        upload_mode="files",
        tar_prefix="",
        total_batches=1,
        metadata_files=[],
        shard_id=0,
        model_count=1,
        timestamp="2026-05-07T00:00:00+00:00",
    )
    payload = json.loads(marker.read_text())
    assert payload["total_files"] == 0
    assert payload["metadata_files"] == []


def test_try_upload_then_lustre_fallback_success_cleans_without_copy(tmp_path: Path) -> None:
    model_id = "AF-0000000000000001"
    _write_output_tree(tmp_path / "work", (model_id,))

    result = try_upload_then_lustre_fallback(
        tmp_path / "work",
        tmp_path / "shard_0" / "success_outputs",
        "s3://bucket/prefix",
        s5cmd_path="s5cmd",
        batch_ids=[model_id],
        runner=lambda _argv: 0,
    )

    assert result.fallback_used is False
    assert result.status == "uploaded"
    assert result.copied_files == ()
    assert not (tmp_path / "shard_0" / "success_outputs").exists()
    assert not (tmp_path / "work" / "modelcif" / f"{model_id}-model_v1.cif").exists()


def test_cleanup_success_outputs_removes_directory(tmp_path: Path) -> None:
    success_dir = tmp_path / "success_outputs"
    success_dir.mkdir()
    (success_dir / "payload").write_text("x")

    assert cleanup_success_outputs(success_dir) is True
    assert not success_dir.exists()
    assert cleanup_success_outputs(success_dir) is False


def test_upload_slot_override_and_disabled_slot(tmp_path: Path) -> None:
    slots_dir = tmp_path / ".upload_slots"
    slots_dir.mkdir()
    (slots_dir / "max_slots").write_text("0")

    assert resolve_upload_slot_count(slots_dir, 50) == 0
    with upload_slot(slots_dir, 50) as slot:
        assert slot is None


def test_write_uploaded_marker_schema_counts_batch_lines_and_failures(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shard_7"
    write_batch_marker(shard_dir, 0, ["a.cif", "b.json"])
    write_batch_marker(shard_dir, 1, ["c.cif"], failed_model_count=1)
    (shard_dir / "failed_models.tsv").write_text("AF-0000000000000002\tmodel\tbad\n")

    marker = write_uploaded_marker(
        shard_dir,
        s3_prefix="s3://bucket/prefix/shard_7",
        upload_mode="files",
        tar_prefix="",
        total_batches=2,
        metadata_files=["AF-metadata-8-of-9.json"],
        shard_id=7,
        model_count=4,
        timestamp="2026-05-07T00:00:00+00:00",
    )

    payload = json.loads(marker.read_text())
    assert set(payload) == {
        "s3_prefix",
        "upload_mode",
        "tar_prefix",
        "status",
        "total_files",
        "total_batches",
        "failed_models",
        "metadata_files",
        "timestamp",
        "shard_id",
        "model_count",
    }
    assert payload["status"] == "partial_uploaded"
    assert payload["total_files"] == 4
    assert payload["failed_models"] == 1
