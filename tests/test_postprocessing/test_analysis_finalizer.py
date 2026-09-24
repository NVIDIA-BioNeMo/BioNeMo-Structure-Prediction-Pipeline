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

"""Tests for native analysis metadata finalization."""

from __future__ import annotations

import csv
import json
import os
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bspp.orchestration.runtime.postprocessing.analysis_finalizer import (
    build_model_tar_index,
    extract_high_quality_from_tars,
    finalize_analysis_metadata,
    main,
    upload_files_with_s5cmd,
    write_analysis_metadata_parquet,
    write_high_quality_model_ids,
)


def test_finalize_analysis_metadata_writes_parquet_and_selected_ids(tmp_path: Path) -> None:
    csv_path = tmp_path / "analysis_metadata.csv"
    _write_csv(
        csv_path,
        [
            "model_id",
            "original_id",
            "upload_status",
            "passes_quality_threshold",
            "failure_reason",
            "ipsae_max",
            "pdockq2_max",
        ],
        [
            {
                "model_id": "AF-0001",
                "original_id": "AF-0001",
                "upload_status": "uploaded",
                "passes_quality_threshold": "false",
                "failure_reason": "",
                "ipsae_max": "0.4",
                "pdockq2_max": "0.3",
            },
            {
                "model_id": "AF-0002",
                "original_id": "AF-0002",
                "upload_status": "uploaded",
                "passes_quality_threshold": "true",
                "failure_reason": "",
                "ipsae_max": "0.7",
                "pdockq2_max": "0.4",
            },
            {
                "model_id": "AF-0003",
                "original_id": "AF-0003",
                "upload_status": "uploaded",
                "passes_quality_threshold": "1",
                "failure_reason": "",
                "ipsae_max": "0.8",
                "pdockq2_max": "0.5",
            },
            {
                "model_id": "AF-0002",
                "original_id": "AF-0002",
                "upload_status": "uploaded",
                "passes_quality_threshold": "true",
                "failure_reason": "",
                "ipsae_max": "0.9",
                "pdockq2_max": "0.6",
            },
        ],
    )

    result = finalize_analysis_metadata(
        csv_path=csv_path,
        parquet_path=tmp_path / "analysis_metadata.parquet",
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
    )

    assert result.analysis_row_count == 4
    assert result.selected_model_count == 1
    assert result.indexed_model_count is None
    assert result.extracted_file_count is None
    assert result.uploaded_file_count is None
    assert result.selected_ids_path.read_text() == "AF-0002\n"
    table = pq.read_table(result.parquet_path)
    assert table.num_rows == 4
    assert table.column("model_id").to_pylist() == ["AF-0001", "AF-0002", "AF-0003", "AF-0002"]


def test_build_model_tar_index_from_tar_path_manifest(tmp_path: Path) -> None:
    _write_tar(
        tmp_path / "local_tars" / "batch_0.tar",
        {
            "AF-0001-model_v1.cif.zst": "model 1",
            "AF-0002-model_v1.cif.zst": "model 2",
        },
    )
    _write_tar(tmp_path / "local_tars" / "batch_1.tar", {"nested/AF-0003-model_v1.pdb": "model 3"})
    _write_csv(
        tmp_path / "local_tars.csv",
        ["tar_path", "model_count", "sha256"],
        [
            {"tar_path": "local_tars/batch_0.tar", "model_count": "2", "sha256": "hash-0"},
            {"tar_path": "local_tars/batch_1.tar", "model_count": "1", "sha256": "hash-1"},
        ],
    )

    row_count = build_model_tar_index(
        local_tars_csv=tmp_path / "local_tars.csv",
        model_ids=("AF-0003", "AF-0001"),
        output_path=tmp_path / "model_tar_index.csv",
    )

    assert row_count == 2
    assert _read_csv(tmp_path / "model_tar_index.csv") == [
        {"model_id": "AF-0001", "tar_path": "local_tars/batch_0.tar"},
        {"model_id": "AF-0003", "tar_path": "local_tars/batch_1.tar"},
    ]


def test_build_model_tar_index_resolves_native_tar_name_manifest(tmp_path: Path) -> None:
    _write_tar(tmp_path / "local_tars" / "shard_7" / "shard_7_batch_0.tar", {"AF-0004-model_v1.bcif": "model"})
    _write_csv(
        tmp_path / "local_tars.csv",
        ["tar_type", "tar_name", "shard_id"],
        [{"tar_type": "batch", "tar_name": "shard_7_batch_0.tar", "shard_id": "7"}],
    )

    row_count = build_model_tar_index(
        local_tars_csv=tmp_path / "local_tars.csv",
        model_ids=("AF-0004",),
        output_path=tmp_path / "model_tar_index.csv",
    )

    assert row_count == 1
    assert _read_csv(tmp_path / "model_tar_index.csv") == [
        {"model_id": "AF-0004", "tar_path": "local_tars/shard_7/shard_7_batch_0.tar"}
    ]


def test_write_high_quality_model_ids_requires_expected_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "analysis_metadata.csv"
    _write_csv(csv_path, ["model_id"], [{"model_id": "AF-0001"}])

    with pytest.raises(ValueError, match="passes_quality_threshold"):
        write_high_quality_model_ids(csv_path, tmp_path / "high_quality_model_ids.txt")


def test_finalize_analysis_metadata_validates_schema_before_writing_parquet(tmp_path: Path) -> None:
    csv_path = tmp_path / "analysis_metadata.csv"
    parquet_path = tmp_path / "analysis_metadata.parquet"
    _write_csv(csv_path, ["model_id"], [{"model_id": "AF-0001"}])

    with pytest.raises(ValueError, match="passes_quality_threshold"):
        finalize_analysis_metadata(
            csv_path=csv_path,
            parquet_path=parquet_path,
            selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        )

    assert not parquet_path.exists()


def test_write_analysis_metadata_parquet_handles_late_failure_reason_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "analysis_metadata.csv"
    _write_csv(
        csv_path,
        ["model_id", "original_id", "upload_status", "passes_quality_threshold", "failure_reason"],
        [
            {
                "model_id": f"AF-{idx:04d}",
                "original_id": f"AF-{idx:04d}",
                "upload_status": "local_tarred",
                "passes_quality_threshold": "true",
                "failure_reason": "",
            }
            for idx in range(40)
        ]
        + [
            {
                "model_id": "AF-FAILED",
                "original_id": "AF-FAILED",
                "upload_status": "model_failed",
                "passes_quality_threshold": "false",
                "failure_reason": (
                    "stage_03_convert_colabfold: DuckDB residue count (1664) does not match pLDDT length (832)."
                ),
            }
        ],
    )

    row_count = write_analysis_metadata_parquet(
        csv_path,
        tmp_path / "analysis_metadata.parquet",
        block_size_bytes=512,
    )

    table = pq.read_table(tmp_path / "analysis_metadata.parquet")
    assert row_count == 41
    assert table.schema.field("failure_reason").type == "string"
    assert table.column("failure_reason").to_pylist()[-1].startswith("stage_03_convert_colabfold:")


def test_build_model_tar_index_fails_when_analysis_model_is_missing(tmp_path: Path) -> None:
    _write_tar(tmp_path / "local_tars" / "batch_0.tar", {"AF-0001-model_v1.cif.zst": "model 1"})
    _write_csv(
        tmp_path / "local_tars.csv",
        ["tar_path", "model_count", "sha256"],
        [{"tar_path": "local_tars/batch_0.tar", "model_count": "1", "sha256": "hash-0"}],
    )

    with pytest.raises(ValueError, match="did not index all analysis models"):
        build_model_tar_index(
            local_tars_csv=tmp_path / "local_tars.csv",
            model_ids=("AF-0001", "AF-0002"),
            output_path=tmp_path / "model_tar_index.csv",
        )


def test_build_model_tar_index_writes_blank_rows_for_allowed_missing_models(tmp_path: Path) -> None:
    _write_tar(tmp_path / "local_tars" / "batch_0.tar", {"AF-0001-model_v1.cif.zst": "model 1"})
    _write_csv(
        tmp_path / "local_tars.csv",
        ["tar_path", "model_count", "sha256"],
        [{"tar_path": "local_tars/batch_0.tar", "model_count": "1", "sha256": "hash-0"}],
    )

    row_count = build_model_tar_index(
        local_tars_csv=tmp_path / "local_tars.csv",
        model_ids=("AF-0001", "AF-FAILED"),
        output_path=tmp_path / "model_tar_index.csv",
        allowed_missing_model_ids={"AF-FAILED"},
    )

    assert row_count == 2
    assert _read_csv(tmp_path / "model_tar_index.csv") == [
        {"model_id": "AF-0001", "tar_path": "local_tars/batch_0.tar"},
        {"model_id": "AF-FAILED", "tar_path": ""},
    ]


def test_finalize_analysis_metadata_allows_failed_models_without_tar_members(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "analysis_metadata.csv",
        ["model_id", "original_id", "upload_status", "passes_quality_threshold", "failure_reason"],
        [
            {
                "model_id": "AF-0001",
                "original_id": "AF-0001",
                "upload_status": "local_tarred",
                "passes_quality_threshold": "true",
                "failure_reason": "",
            },
            {
                "model_id": "AF-FAILED",
                "original_id": "AF-FAILED",
                "upload_status": "model_failed",
                "passes_quality_threshold": "false",
                "failure_reason": "fake_stage: configured failure",
            },
        ],
    )
    _write_tar(tmp_path / "local_tars" / "batch_0.tar", {"AF-0001-model_v1.cif.zst": "model 1"})
    _write_csv(
        tmp_path / "local_tars.csv",
        ["tar_path", "model_count", "sha256"],
        [{"tar_path": "local_tars/batch_0.tar", "model_count": "1", "sha256": "hash-0"}],
    )

    result = finalize_analysis_metadata(
        csv_path=tmp_path / "analysis_metadata.csv",
        parquet_path=tmp_path / "analysis_metadata.parquet",
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        local_tars_csv=tmp_path / "local_tars.csv",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
    )

    assert result.analysis_row_count == 2
    assert result.selected_model_count == 1
    assert result.indexed_model_count == 2
    assert _read_csv(tmp_path / "model_tar_index.csv") == [
        {"model_id": "AF-0001", "tar_path": "local_tars/batch_0.tar"},
        {"model_id": "AF-FAILED", "tar_path": ""},
    ]


def test_build_model_tar_index_prefers_file_uri_manifest(tmp_path: Path) -> None:
    tar_path = tmp_path / "exact" / "batch_0.tar"
    _write_tar(tar_path, {"AF-0005-model_v1.cif.zst": "model 5"})
    _write_csv(
        tmp_path / "local_tars.csv",
        ["tar_type", "s3_uri", "tar_name", "shard_id"],
        [{"tar_type": "batch", "s3_uri": tar_path.resolve().as_uri(), "tar_name": "batch_0.tar", "shard_id": ""}],
    )

    row_count = build_model_tar_index(
        local_tars_csv=tmp_path / "local_tars.csv",
        model_ids=("AF-0005",),
        output_path=tmp_path / "model_tar_index.csv",
    )

    assert row_count == 1
    assert _read_csv(tmp_path / "model_tar_index.csv") == [{"model_id": "AF-0005", "tar_path": "exact/batch_0.tar"}]


def test_extract_high_quality_from_tars_writes_selected_models_and_metadata(tmp_path: Path) -> None:
    _write_tar(
        tmp_path / "local_tars" / "shard_1" / "batch_0.tar",
        {
            "AF-0001-model_v1.cif.zst": "selected",
            "AF-0002-model_v1.cif.zst": "not selected",
        },
    )
    _write_tar(tmp_path / "local_tars" / "metadata" / "shard_1_metadata.tar", {"meta_1.json": "{}"})
    _write_csv(
        tmp_path / "local_tars.csv",
        ["tar_type", "tar_name", "shard_id"],
        [
            {"tar_type": "batch", "tar_name": "batch_0.tar", "shard_id": "1"},
            {"tar_type": "metadata", "tar_name": "shard_1_metadata.tar", "shard_id": "1"},
        ],
    )

    extracted = extract_high_quality_from_tars(
        local_tars_csv=tmp_path / "local_tars.csv",
        selected_model_ids=("AF-0001",),
        work_dir=tmp_path / "hq-work",
    )

    assert [path.name for path in extracted] == ["AF-0001-model_v1.cif.zst", "meta_1.json"]
    assert (tmp_path / "hq-work" / "extracted" / "AF-0001-model_v1.cif.zst").read_text() == "selected"


def test_upload_files_with_s5cmd_writes_run_file_and_invokes_command(tmp_path: Path) -> None:
    upload_source = tmp_path / "AF-0001-model_v1.cif.zst"
    upload_source.write_text("selected", encoding="utf-8")
    fake_s5cmd = tmp_path / "s5cmd"
    fake_s5cmd.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$BSPP_FAKE_S5CMD_ARGS"\n',
        encoding="utf-8",
    )
    fake_s5cmd.chmod(0o755)
    args_path = tmp_path / "s5cmd.args"

    old_env = os.environ.get("BSPP_FAKE_S5CMD_ARGS")
    os.environ["BSPP_FAKE_S5CMD_ARGS"] = str(args_path)
    try:
        uploaded = upload_files_with_s5cmd(
            files=(upload_source,),
            destination_prefix="s3://bucket/hq/",
            s5cmd_path=str(fake_s5cmd),
            command_dir=tmp_path / "hq-work",
            numworkers=3,
        )
    finally:
        if old_env is None:
            os.environ.pop("BSPP_FAKE_S5CMD_ARGS", None)
        else:
            os.environ["BSPP_FAKE_S5CMD_ARGS"] = old_env

    assert uploaded == 1
    assert args_path.read_text().splitlines() == [
        "--numworkers",
        "3",
        "run",
        str(tmp_path / "hq-work" / "s5cmd_high_quality_upload.txt"),
    ]
    assert (
        "s3://bucket/hq/AF-0001-model_v1.cif.zst"
        in (tmp_path / "hq-work" / "s5cmd_high_quality_upload.txt").read_text()
    )


def test_analysis_finalizer_module_cli_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_csv(
        tmp_path / "analysis_metadata.csv",
        ["model_id", "original_id", "upload_status", "passes_quality_threshold", "failure_reason"],
        [
            {
                "model_id": "AF-0001",
                "original_id": "AF-0001",
                "upload_status": "uploaded",
                "passes_quality_threshold": "true",
                "failure_reason": "",
            }
        ],
    )

    rc = main(
        [
            "--csv",
            str(tmp_path / "analysis_metadata.csv"),
            "--parquet",
            str(tmp_path / "analysis_metadata.parquet"),
            "--selected-ids",
            str(tmp_path / "high_quality_model_ids.txt"),
        ]
    )

    assert rc == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["analysis_row_count"] == 1
    assert rendered["selected_model_count"] == 1


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_tar(path: Path, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.stem}-staging"
    staging.mkdir()
    for relative_path, content in files.items():
        source = staging / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(content, encoding="utf-8")
    with tarfile.open(path, "w") as archive:
        for source in sorted(p for p in staging.rglob("*") if p.is_file()):
            archive.add(source, arcname=source.relative_to(staging).as_posix())
