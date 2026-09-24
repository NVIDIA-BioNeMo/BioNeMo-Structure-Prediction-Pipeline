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

"""Tests for orchestration-native high-quality chunk building."""

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
from bspp.orchestration.runtime.postprocessing.hq_chunks import (
    DEFAULT_HQ_CHUNK_SUFFIXES,
    build_hq_chunks,
    read_hq_chunk_manifest,
    read_model_tar_index,
)
from bspp.orchestration.runtime.validation.hq_chunks import validate_hq_chunks


def test_build_hq_chunks_writes_ordered_chunk_tars_and_manifest(tmp_path: Path) -> None:
    selected = ("AF-0001", "AF-0002", "AF-0003")
    _write_lines(tmp_path / "high_quality_model_ids.txt", selected)
    _write_source_tar(
        tmp_path / "local_tars" / "shard_1" / "batch_0.tar",
        model_ids=("AF-0001", "AF-0003", "AF-9999"),
    )
    _write_source_tar(tmp_path / "local_tars" / "shard_2" / "batch_0.tar", model_ids=("AF-0002",))
    _write_csv(
        tmp_path / "model_tar_index.csv",
        ["model_id", "tar_path"],
        [
            {"model_id": "AF-0001", "tar_path": "local_tars/shard_1/batch_0.tar"},
            {"model_id": "AF-0002", "tar_path": "local_tars/shard_2/batch_0.tar"},
            {"model_id": "AF-0003", "tar_path": "local_tars/shard_1/batch_0.tar"},
        ],
    )

    result = build_hq_chunks(
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
        staging_root=tmp_path / "staging",
        chunks_dir=tmp_path / "chunks",
        chunk_size=2,
    )

    assert result.selected_model_count == 3
    assert result.source_tar_count == 2
    assert result.extracted_file_count == 15
    assert [chunk.file_count for chunk in result.chunks] == [10, 5]
    assert _tar_names(tmp_path / "chunks" / "chunk_0000.tar") == sorted(
        f"{model_id}{suffix}" for model_id in ("AF-0001", "AF-0002") for suffix in DEFAULT_HQ_CHUNK_SUFFIXES
    )
    assert _tar_names(tmp_path / "chunks" / "chunk_0001.tar") == sorted(
        f"AF-0003{suffix}" for suffix in DEFAULT_HQ_CHUNK_SUFFIXES
    )
    assert not any("AF-9999" in name for name in _tar_names(tmp_path / "chunks" / "chunk_0000.tar"))
    manifest = _read_csv(tmp_path / "chunks" / "hq_chunks_manifest.csv")
    assert manifest[0]["file_count"] == "10"
    assert manifest[0]["payload_sha256"] == result.chunks[0].payload_sha256
    parsed_manifest = read_hq_chunk_manifest(tmp_path / "chunks" / "hq_chunks_manifest.csv")
    assert parsed_manifest == result.chunks
    repeated = build_hq_chunks(
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
        staging_root=tmp_path / "staging-repeated",
        chunks_dir=tmp_path / "chunks-repeated",
        chunk_size=2,
    )
    assert [chunk.sha256 for chunk in repeated.chunks] == [chunk.sha256 for chunk in result.chunks]

    validation = validate_hq_chunks(
        chunks_dir=tmp_path / "chunks",
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
        chunk_size=2,
    )
    assert validation.ok


def test_build_hq_chunks_supports_legacy_file_uri_index(tmp_path: Path) -> None:
    selected_path = tmp_path / "high_quality_model_ids.txt"
    _write_lines(selected_path, ("AF-0001",))
    tar_path = tmp_path / "source" / "batch_0.tar"
    _write_source_tar(tar_path, model_ids=("AF-0001",))
    _write_csv(
        tmp_path / "model_tar_index.csv",
        ["model_id", "output_tar_s3_uri"],
        [{"model_id": "AF-0001", "output_tar_s3_uri": tar_path.resolve().as_uri()}],
    )

    entries = read_model_tar_index(tmp_path / "model_tar_index.csv")
    assert entries["AF-0001"].tar_path == tar_path.resolve()

    result = build_hq_chunks(
        selected_ids_path=selected_path,
        model_tar_index_path=tmp_path / "model_tar_index.csv",
        staging_root=tmp_path / "staging",
        chunks_dir=tmp_path / "chunks",
    )

    assert result.ok
    assert _tar_names(tmp_path / "chunks" / "chunk_0000.tar") == sorted(
        f"AF-0001{suffix}" for suffix in DEFAULT_HQ_CHUNK_SUFFIXES
    )


def test_build_hq_chunks_supports_local_tar_root_with_native_relative_paths(tmp_path: Path) -> None:
    _write_lines(tmp_path / "high_quality_model_ids.txt", ("AF-0001",))
    local_tar_root = tmp_path / "materialized" / "local_tars"
    _write_source_tar(local_tar_root / "shard_1" / "batch_0.tar", model_ids=("AF-0001",))
    _write_csv(
        tmp_path / "model_tar_index.csv",
        ["model_id", "tar_path"],
        [{"model_id": "AF-0001", "tar_path": "local_tars/shard_1/batch_0.tar"}],
    )

    entries = read_model_tar_index(tmp_path / "model_tar_index.csv", local_tar_root=local_tar_root)
    assert entries["AF-0001"].tar_path == local_tar_root / "shard_1" / "batch_0.tar"

    result = build_hq_chunks(
        selected_ids_path=tmp_path / "high_quality_model_ids.txt",
        model_tar_index_path=tmp_path / "model_tar_index.csv",
        staging_root=tmp_path / "staging",
        chunks_dir=tmp_path / "chunks",
        local_tar_root=local_tar_root,
    )

    assert result.ok


def test_build_hq_chunks_rejects_remote_index_uri(tmp_path: Path) -> None:
    _write_lines(tmp_path / "high_quality_model_ids.txt", ("AF-0001",))
    _write_csv(
        tmp_path / "model_tar_index.csv",
        ["model_id", "tar_path"],
        [{"model_id": "AF-0001", "tar_path": "s3://bucket/batch_0.tar"}],
    )

    with pytest.raises(ValueError, match="requires local/materialized tar paths"):
        build_hq_chunks(
            selected_ids_path=tmp_path / "high_quality_model_ids.txt",
            model_tar_index_path=tmp_path / "model_tar_index.csv",
            staging_root=tmp_path / "staging",
            chunks_dir=tmp_path / "chunks",
        )


def test_build_hq_chunks_rejects_missing_and_blank_index_entries(tmp_path: Path) -> None:
    _write_lines(tmp_path / "high_quality_model_ids.txt", ("AF-0001", "AF-0002"))
    _write_csv(
        tmp_path / "model_tar_index.csv",
        ["model_id", "tar_path"],
        [{"model_id": "AF-0001", "tar_path": ""}],
    )

    with pytest.raises(ValueError, match="missing from model_tar_index"):
        build_hq_chunks(
            selected_ids_path=tmp_path / "high_quality_model_ids.txt",
            model_tar_index_path=tmp_path / "model_tar_index.csv",
            staging_root=tmp_path / "staging",
            chunks_dir=tmp_path / "chunks",
        )

    _write_lines(tmp_path / "high_quality_model_ids.txt", ("AF-0001",))
    with pytest.raises(ValueError, match="blank tar paths"):
        build_hq_chunks(
            selected_ids_path=tmp_path / "high_quality_model_ids.txt",
            model_tar_index_path=tmp_path / "model_tar_index.csv",
            staging_root=tmp_path / "staging",
            chunks_dir=tmp_path / "chunks",
        )


def test_build_hq_chunks_rejects_empty_selected_ids(tmp_path: Path) -> None:
    _write_lines(tmp_path / "high_quality_model_ids.txt", ())
    _write_csv(tmp_path / "model_tar_index.csv", ["model_id", "tar_path"], [])

    with pytest.raises(ValueError, match="contains no selected model IDs"):
        build_hq_chunks(
            selected_ids_path=tmp_path / "high_quality_model_ids.txt",
            model_tar_index_path=tmp_path / "model_tar_index.csv",
            staging_root=tmp_path / "staging",
            chunks_dir=tmp_path / "chunks",
        )


def test_build_hq_chunks_fails_when_selected_suffix_is_missing(tmp_path: Path) -> None:
    _write_lines(tmp_path / "high_quality_model_ids.txt", ("AF-0001",))
    _write_source_tar(
        tmp_path / "local_tars" / "batch_0.tar",
        model_ids=("AF-0001",),
        omit={(0, "-confidence_v1.json.zst")},
    )
    _write_csv(
        tmp_path / "model_tar_index.csv",
        ["model_id", "tar_path"],
        [{"model_id": "AF-0001", "tar_path": "local_tars/batch_0.tar"}],
    )

    with pytest.raises(ValueError, match="missing expected members"):
        build_hq_chunks(
            selected_ids_path=tmp_path / "high_quality_model_ids.txt",
            model_tar_index_path=tmp_path / "model_tar_index.csv",
            staging_root=tmp_path / "staging",
            chunks_dir=tmp_path / "chunks",
        )


def test_build_hq_chunks_removes_stale_chunk_tars(tmp_path: Path) -> None:
    selected_path = tmp_path / "high_quality_model_ids.txt"
    index_path = tmp_path / "model_tar_index.csv"
    chunks_dir = tmp_path / "chunks"
    _write_source_tar(tmp_path / "local_tars" / "batch_0.tar", model_ids=("AF-0001", "AF-0002", "AF-0003"))
    _write_lines(selected_path, ("AF-0001", "AF-0002", "AF-0003"))
    _write_csv(
        index_path,
        ["model_id", "tar_path"],
        [
            {"model_id": "AF-0001", "tar_path": "local_tars/batch_0.tar"},
            {"model_id": "AF-0002", "tar_path": "local_tars/batch_0.tar"},
            {"model_id": "AF-0003", "tar_path": "local_tars/batch_0.tar"},
        ],
    )

    build_hq_chunks(
        selected_ids_path=selected_path,
        model_tar_index_path=index_path,
        staging_root=tmp_path / "staging",
        chunks_dir=chunks_dir,
        chunk_size=2,
    )
    assert (chunks_dir / "chunk_0001.tar").exists()

    _write_lines(selected_path, ("AF-0001",))
    _write_csv(
        index_path,
        ["model_id", "tar_path"],
        [{"model_id": "AF-0001", "tar_path": "local_tars/batch_0.tar"}],
    )
    build_hq_chunks(
        selected_ids_path=selected_path,
        model_tar_index_path=index_path,
        staging_root=tmp_path / "staging",
        chunks_dir=chunks_dir,
        chunk_size=2,
    )

    assert (chunks_dir / "chunk_0000.tar").exists()
    assert not (chunks_dir / "chunk_0001.tar").exists()


def test_build_hq_chunks_cli_writes_report(tmp_path: Path) -> None:
    _write_lines(tmp_path / "high_quality_model_ids.txt", ("AF-0001",))
    _write_source_tar(tmp_path / "local_tars" / "batch_0.tar", model_ids=("AF-0001",))
    _write_csv(
        tmp_path / "model_tar_index.csv",
        ["model_id", "tar_path"],
        [{"model_id": "AF-0001", "tar_path": "local_tars/batch_0.tar"}],
    )

    result = CliRunner().invoke(
        cli,
        [
            "hq-chunks",
            "build",
            "--selected-ids",
            str(tmp_path / "high_quality_model_ids.txt"),
            "--model-tar-index",
            str(tmp_path / "model_tar_index.csv"),
            "--staging-root",
            str(tmp_path / "staging"),
            "--chunks-dir",
            str(tmp_path / "chunks"),
            "--write-report",
            str(tmp_path / "evidence"),
        ],
    )

    assert result.exit_code == 0
    rendered = json.loads(result.output)
    assert rendered["selected_model_count"] == 1
    assert (tmp_path / "evidence" / "hq_chunks_build_report.json").exists()
    assert (tmp_path / "evidence" / "hq_chunks_build_report.txt").exists()


def _write_lines(path: Path, lines: tuple[str, ...]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_source_tar(
    path: Path,
    *,
    model_ids: tuple[str, ...],
    omit: set[tuple[int, str]] | None = None,
) -> None:
    omit = omit or set()
    entries: list[tuple[str, bytes]] = []
    for model_index, model_id in enumerate(model_ids):
        for suffix in DEFAULT_HQ_CHUNK_SUFFIXES:
            if (model_index, suffix) in omit:
                continue
            entries.append((f"nested/{model_id}{suffix}", _zstd(f"{model_id}:{suffix}\n")))
    _write_tar_entries(path, tuple(entries))


def _write_tar_entries(path: Path, entries: tuple[tuple[str, bytes], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))


def _zstd(payload: str) -> bytes:
    return zstandard.ZstdCompressor().compress(payload.encode())


def _tar_names(path: Path) -> list[str]:
    with tarfile.open(path, "r:*") as archive:
        return sorted(member.name for member in archive if member.isfile())
