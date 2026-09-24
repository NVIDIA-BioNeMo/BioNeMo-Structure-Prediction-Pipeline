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

"""Tests for high-quality chunk validation."""

from __future__ import annotations

import csv
import json
import tarfile
from io import BytesIO
from pathlib import Path

import pytest
import zstandard
from click.testing import CliRunner

from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.postprocessing.hq_chunks import DEFAULT_HQ_CHUNK_SUFFIXES
from bspp.orchestration.runtime.validation.hq_chunks import validate_hq_chunks


def test_validate_hq_chunks_accepts_complete_chunks(tmp_path: Path) -> None:
    _write_fixture(tmp_path, selected=("AF-0001", "AF-0002", "AF-0003"), chunk_size=2)

    report = validate_hq_chunks(
        chunks_dir=tmp_path / "chunks",
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
        chunk_size=2,
    )

    assert report.ok
    assert report.expected_chunk_count == 2
    assert report.actual_chunk_count == 2
    assert report.payload_hashes_recorded
    assert [chunk.file_count for chunk in report.chunks] == [10, 5]


def test_validate_hq_chunks_fails_missing_suffix(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        selected=("AF-0001",),
        chunk_size=2,
        omit={(0, "-confidence_v1.json.zst")},
    )

    report = validate_hq_chunks(
        chunks_dir=tmp_path / "chunks",
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
        chunk_size=2,
    )

    assert not report.ok
    assert report.missing_member_names == ("AF-0001-confidence_v1.json.zst",)
    assert report.chunks[0].file_count == 4
    assert report.chunks[0].expected_file_count == 5


def test_validate_hq_chunks_fails_unselected_model(tmp_path: Path) -> None:
    _write_fixture(tmp_path, selected=("AF-0001",), chunk_size=2, extra_models=("AF-9999",))

    report = validate_hq_chunks(
        chunks_dir=tmp_path / "chunks",
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
        chunk_size=2,
    )

    assert not report.ok
    assert report.chunks[0].unselected_model_ids == ("AF-9999",)


def test_validate_hq_chunks_fails_duplicate_member_names(tmp_path: Path) -> None:
    _write_lines(tmp_path / "high_quality_model_ids.txt", ("AF-0001",))
    _write_csv(
        tmp_path / "model_tar_index.csv",
        ["model_id", "tar_path"],
        [{"model_id": "AF-0001", "tar_path": "local_tars/batch_0.tar"}],
    )
    entries = (
        *((f"AF-0001{suffix}", _zstd(suffix)) for suffix in DEFAULT_HQ_CHUNK_SUFFIXES),
        ("AF-0001-model_v1.cif.zst", _zstd("duplicate")),
    )
    _write_tar_entries(tmp_path / "chunks" / "chunk_0000.tar", entries)

    report = validate_hq_chunks(
        chunks_dir=tmp_path / "chunks",
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
    )

    assert not report.ok
    assert report.chunks[0].duplicate_member_names == ("AF-0001-model_v1.cif.zst",)


def test_validate_hq_chunks_fails_malformed_tar_without_traceback(tmp_path: Path) -> None:
    _write_lines(tmp_path / "high_quality_model_ids.txt", ("AF-0001",))
    _write_csv(
        tmp_path / "model_tar_index.csv",
        ["model_id", "tar_path"],
        [{"model_id": "AF-0001", "tar_path": "local_tars/batch_0.tar"}],
    )
    (tmp_path / "chunks").mkdir()
    (tmp_path / "chunks" / "chunk_0000.tar").write_text("not a tar", encoding="utf-8")

    report = validate_hq_chunks(
        chunks_dir=tmp_path / "chunks",
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
    )

    assert not report.ok
    assert report.chunks[0].errors


def test_validate_hq_chunks_fails_incomplete_model_tar_index(tmp_path: Path) -> None:
    _write_fixture(tmp_path, selected=("AF-0001", "AF-0002"), chunk_size=2)
    _write_csv(
        tmp_path / "model_tar_index.csv",
        ["model_id", "tar_path"],
        [{"model_id": "AF-0001", "tar_path": "local_tars/batch_0.tar"}],
    )

    report = validate_hq_chunks(
        chunks_dir=tmp_path / "chunks",
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
        chunk_size=2,
    )

    assert not report.ok
    assert report.missing_model_ids == ("AF-0002",)


def test_validate_hq_chunks_fails_blank_index_tar_path(tmp_path: Path) -> None:
    _write_fixture(tmp_path, selected=("AF-0001",), chunk_size=2)
    _write_csv(
        tmp_path / "model_tar_index.csv",
        ["model_id", "tar_path"],
        [{"model_id": "AF-0001", "tar_path": ""}],
    )

    report = validate_hq_chunks(
        chunks_dir=tmp_path / "chunks",
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
        chunk_size=2,
    )

    assert not report.ok
    assert report.blank_index_model_ids == ("AF-0001",)


def test_validate_hq_chunks_fails_missing_and_extra_chunk_tars(tmp_path: Path) -> None:
    _write_fixture(tmp_path, selected=("AF-0001", "AF-0002"), chunk_size=1)
    (tmp_path / "chunks" / "chunk_0001.tar").unlink()
    _write_tar_entries(tmp_path / "chunks" / "chunk_9999.tar", ())

    report = validate_hq_chunks(
        chunks_dir=tmp_path / "chunks",
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
        chunk_size=1,
    )

    assert not report.ok
    assert report.missing_chunk_tars == ("chunk_0001.tar",)
    assert report.extra_chunk_tars == ("chunk_9999.tar",)


def test_validate_hq_chunks_fails_misplaced_model(tmp_path: Path) -> None:
    _write_lines(tmp_path / "high_quality_model_ids.txt", ("AF-0001", "AF-0002"))
    _write_csv(
        tmp_path / "model_tar_index.csv",
        ["model_id", "tar_path"],
        [
            {"model_id": "AF-0001", "tar_path": "local_tars/batch_0.tar"},
            {"model_id": "AF-0002", "tar_path": "local_tars/batch_0.tar"},
        ],
    )
    entries = tuple(
        (f"{model_id}{suffix}", _zstd(f"{model_id}:{suffix}"))
        for model_id in ("AF-0001", "AF-0002")
        for suffix in DEFAULT_HQ_CHUNK_SUFFIXES
    )
    _write_tar_entries(tmp_path / "chunks" / "chunk_0000.tar", entries)
    _write_tar_entries(tmp_path / "chunks" / "chunk_0001.tar", ())

    report = validate_hq_chunks(
        chunks_dir=tmp_path / "chunks",
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
        chunk_size=1,
    )

    assert not report.ok
    assert report.chunks[0].misplaced_model_ids == ("AF-0002",)


def test_validate_hq_chunks_rejects_zero_sample_limit(tmp_path: Path) -> None:
    _write_fixture(tmp_path, selected=("AF-0001",), chunk_size=2)

    with pytest.raises(ValueError, match="sample_limit must be positive"):
        validate_hq_chunks(
            chunks_dir=tmp_path / "chunks",
            selected_ids_path=tmp_path / "high_quality_model_ids.txt",
            model_tar_index_path=tmp_path / "model_tar_index.csv",
            sample_limit=0,
        )


def test_validate_hq_chunks_reports_malformed_model_tar_index_without_traceback(tmp_path: Path) -> None:
    _write_lines(tmp_path / "high_quality_model_ids.txt", ("AF-0001",))
    _write_csv(
        tmp_path / "model_tar_index.csv",
        ["tar_path"],
        [{"tar_path": "local_tars/batch_0.tar"}],
    )
    (tmp_path / "chunks").mkdir()

    report = validate_hq_chunks(
        chunks_dir=tmp_path / "chunks",
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
    )

    assert not report.ok
    assert "missing required column: model_id" in report.errors[0]
    assert report.missing_model_ids == ("AF-0001",)


def test_validate_hq_chunks_cli_writes_report_and_fails_strict(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        selected=("AF-0001",),
        chunk_size=2,
        omit={(0, "-confidence_v1.json.zst")},
    )

    result = CliRunner().invoke(
        cli,
        [
            "validate",
            "hq-chunks",
            "--chunks-dir",
            str(tmp_path / "chunks"),
            "--selected-ids",
            str(tmp_path / "high_quality_model_ids.txt"),
            "--model-tar-index",
            str(tmp_path / "model_tar_index.csv"),
            "--write-report",
            str(tmp_path / "evidence"),
            "--strict",
        ],
    )

    assert result.exit_code == 1
    rendered = json.loads(result.output)
    assert rendered["ok"] is False
    assert (tmp_path / "evidence" / "hq_chunks_validation_report.json").exists()
    assert (tmp_path / "evidence" / "hq_chunks_validation_report.txt").exists()


def _write_fixture(
    root: Path,
    *,
    selected: tuple[str, ...],
    chunk_size: int,
    omit: set[tuple[int, str]] | None = None,
    extra_models: tuple[str, ...] = (),
) -> None:
    _write_lines(root / "high_quality_model_ids.txt", selected)
    _write_csv(
        root / "model_tar_index.csv",
        ["model_id", "tar_path"],
        [{"model_id": model_id, "tar_path": "local_tars/batch_0.tar"} for model_id in selected],
    )
    omit = omit or set()
    entries: list[tuple[str, bytes]] = []
    for model_index, model_id in enumerate((*selected, *extra_models)):
        for suffix in DEFAULT_HQ_CHUNK_SUFFIXES:
            if (model_index, suffix) in omit:
                continue
            chunk_index = min(model_index // chunk_size, max(0, (len(selected) - 1) // chunk_size))
            entries.append((f"chunk_{chunk_index:04d}/{model_id}{suffix}", _zstd(f"{model_id}:{suffix}\n")))
    by_chunk: dict[int, list[tuple[str, bytes]]] = {}
    for name, payload in entries:
        chunk_name, member_name = name.split("/", 1)
        by_chunk.setdefault(int(chunk_name.removeprefix("chunk_")), []).append((member_name, payload))
    for chunk_index, chunk_entries in by_chunk.items():
        _write_tar_entries(root / "chunks" / f"chunk_{chunk_index:04d}.tar", tuple(chunk_entries))


def _write_lines(path: Path, lines: tuple[str, ...]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_tar_entries(path: Path, entries: tuple[tuple[str, bytes], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))


def _zstd(payload: str) -> bytes:
    return zstandard.ZstdCompressor().compress(payload.encode())
