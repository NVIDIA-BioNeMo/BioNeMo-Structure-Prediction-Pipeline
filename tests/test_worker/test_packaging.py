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

import tarfile
from pathlib import Path

import bspp.orchestration.runtime.worker.packaging as packaging
from bspp.orchestration.runtime.worker import create_outputs_tar, zstd_member_arcname


def test_zstd_member_arcname_appends_zst_once() -> None:
    assert zstd_member_arcname(Path("scores/AF-1-confidence_v1.json")).as_posix() == (
        "scores/AF-1-confidence_v1.json.zst"
    )
    assert zstd_member_arcname(Path("scores/already.zst")).as_posix() == "scores/already.zst"


def test_create_outputs_tar_with_zstd_members_uses_flat_layout_and_fake_compressor(tmp_path: Path) -> None:
    model_id = "AF-0000000000000001"
    for dirname, filename in {
        "modelcif": f"{model_id}-model_v1.cif",
        "scores": f"{model_id}-confidence_v1.json",
        "clash_interface_analysis": f"{model_id}-model_v1_clashes.json",
    }.items():
        path = tmp_path / "work" / dirname / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)

    def fake_zstd(source: Path, destination: Path) -> None:
        destination.write_bytes(b"zstd:" + source.read_bytes())

    result = create_outputs_tar(
        tmp_path / "work",
        tmp_path / "tars" / "shard_0_batch_0.tar",
        batch_ids=[model_id],
        compression="zstd-members",
        zstd_compressor=fake_zstd,
    )

    assert result.member_count == 3
    assert result.size_bytes > 0
    with tarfile.open(result.tar_path) as archive:
        assert sorted(archive.getnames()) == [
            f"{model_id}-confidence_v1.json.zst",
            f"{model_id}-model_v1.cif.zst",
            f"metadata/clashes_and_interfaces_granular/{model_id}-model_v1_clashes.json.zst",
        ]
        member = archive.extractfile(f"{model_id}-model_v1.cif.zst")
        assert member is not None
        assert member.read().startswith(b"zstd:")


def test_create_outputs_tar_with_zstd_members_uses_nvcomp_when_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_id = "AF-0000000000000001"
    path = tmp_path / "work" / "modelcif" / f"{model_id}-model_v1.cif"
    path.parent.mkdir(parents=True)
    path.write_text("model")

    def fake_gpu(files: list[tuple[Path, Path]], *, sub_batch_size: int = 256) -> list[bytes]:
        assert sub_batch_size == 256
        return [b"gpu:" + local_path.read_bytes() for local_path, _ in files]

    monkeypatch.setattr(packaging, "_compress_files_zstd_gpu_batched", fake_gpu)

    result = create_outputs_tar(
        tmp_path / "work",
        tmp_path / "tars" / "shard_0_batch_0.tar",
        batch_ids=[model_id],
        compression="zstd-members",
    )

    assert result.member_count == 1
    with tarfile.open(result.tar_path) as archive:
        member = archive.extractfile(f"{model_id}-model_v1.cif.zst")
        assert member is not None
        assert member.read() == b"gpu:model"


def test_create_outputs_tar_with_zstd_members_falls_back_to_cpu_when_nvcomp_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_id = "AF-0000000000000001"
    path = tmp_path / "work" / "modelcif" / f"{model_id}-model_v1.cif"
    path.parent.mkdir(parents=True)
    path.write_text("model")

    def fail_gpu(files: list[tuple[Path, Path]], *, sub_batch_size: int = 256) -> list[bytes]:
        raise RuntimeError("CUDA error")

    def fake_cpu(zstd_path: str | None):
        def compress(source: Path, destination: Path) -> None:
            destination.write_bytes(b"cpu:" + source.read_bytes())

        return compress

    monkeypatch.setattr(packaging, "_compress_files_zstd_gpu_batched", fail_gpu)
    monkeypatch.setattr(packaging, "_default_zstd_compressor", fake_cpu)

    result = create_outputs_tar(
        tmp_path / "work",
        tmp_path / "tars" / "shard_0_batch_0.tar",
        batch_ids=[model_id],
        compression="zstd-members",
    )

    assert result.member_count == 1
    with tarfile.open(result.tar_path) as archive:
        member = archive.extractfile(f"{model_id}-model_v1.cif.zst")
        assert member is not None
        assert member.read() == b"cpu:model"


def test_gpu_zstd_batch_halves_on_oom_and_preserves_order(tmp_path: Path, monkeypatch) -> None:
    files: list[tuple[Path, Path]] = []
    for index in range(3):
        path = tmp_path / f"member_{index}.txt"
        path.write_text(str(index))
        files.append((path, Path(path.name)))

    calls: list[int] = []

    def fake_batch(codec: object, batch: list[tuple[Path, Path]]) -> list[bytes]:
        calls.append(len(batch))
        if len(batch) > 1:
            raise RuntimeError("CUDA out of memory")
        return [local_path.read_bytes() for local_path, _ in batch]

    monkeypatch.setattr(packaging, "_get_nvcomp_zstd_codec", lambda: object())
    monkeypatch.setattr(packaging, "_compress_file_batch_zstd_gpu", fake_batch)

    assert packaging._compress_files_zstd_gpu_batched(files, sub_batch_size=2) == [b"0", b"1", b"2"]
    assert calls == [2, 1, 2, 1, 1]


def test_create_outputs_tar_returns_zero_for_empty_source(tmp_path: Path) -> None:
    result = create_outputs_tar(tmp_path / "work", tmp_path / "empty.tar")

    assert result.member_count == 0
    assert result.size_bytes == 0
    assert not result.tar_path.exists()
