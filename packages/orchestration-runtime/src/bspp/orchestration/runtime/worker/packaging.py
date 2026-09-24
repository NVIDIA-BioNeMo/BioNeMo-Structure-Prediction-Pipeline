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

"""Tar packaging helpers for native worker delivery artifacts."""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from logging import getLogger
from pathlib import Path
from typing import Any, Protocol, SupportsBytes, cast

from bspp.orchestration.runtime.worker.upload import collect_flat_upload_files

_LOGGER = getLogger(__name__)
_NVCOMP_ZSTD_SUB_BATCH_SIZE = 256


class ZstdCompressor(Protocol):
    """Fakeable zstd compression boundary."""

    def __call__(self, source: Path, destination: Path) -> None:
        """Compress *source* to *destination*."""


class _NvcompEncodedBuffer(Protocol):
    def cpu(self) -> SupportsBytes:
        """Return an object convertible to compressed bytes."""


class _NvcompCodec(Protocol):
    def encode(self, arrays: object) -> Iterable[_NvcompEncodedBuffer]:
        """Encode nvCOMP-compatible byte arrays."""


@dataclass(frozen=True, slots=True)
class OutputTarResult:
    """Accounting for an output tar build."""

    tar_path: Path
    member_count: int
    size_bytes: int


def zstd_member_arcname(relative_path: Path) -> Path:
    """Return the in-tar name for a zstd-member payload."""

    if relative_path.name.endswith(".zst"):
        return relative_path
    return relative_path.with_name(f"{relative_path.name}.zst")


def compress_file_to_zstd(source: Path, destination: Path, *, zstd_path: str = "zstd") -> None:
    """Compress one file with ``zstd -q -f -c``."""

    with destination.open("wb") as output:
        result = subprocess.run(
            [zstd_path, "-q", "-f", "-c", str(source)],
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
        )
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        msg = f"zstd failed for {source}: {result.stderr.strip()[:1000]}"
        raise RuntimeError(msg)


def create_outputs_tar(
    source_dir: Path,
    tar_path: Path,
    *,
    batch_ids: Iterable[str] | None = None,
    compression: str = "none",
    zstd_path: str | None = None,
    zstd_compressor: ZstdCompressor | None = None,
) -> OutputTarResult:
    """Package uploadable outputs into a tar while preserving flat layout."""

    files = list(collect_flat_upload_files(source_dir, batch_ids))
    if not files:
        return OutputTarResult(tar_path=tar_path, member_count=0, size_bytes=0)

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    if compression == "gz":
        with tarfile.open(tar_path, "w:gz") as archive:
            for local_path, relative_path in files:
                archive.add(local_path, arcname=relative_path.as_posix())
    elif compression == "zstd-members":
        if zstd_compressor is None and zstd_path is None:
            try:
                _create_outputs_tar_zstd_members_gpu(files, tar_path)
            except Exception as exc:
                _LOGGER.warning("nvCOMP zstd-member tar creation failed; falling back to CPU zstd: %s", exc)
                _create_outputs_tar_zstd_members_cpu(files, tar_path, compressor=_default_zstd_compressor(zstd_path))
        else:
            _create_outputs_tar_zstd_members_cpu(
                files,
                tar_path,
                compressor=zstd_compressor or _default_zstd_compressor(zstd_path),
            )
    else:
        with tarfile.open(tar_path, "w") as archive:
            for local_path, relative_path in files:
                archive.add(local_path, arcname=relative_path.as_posix())

    return OutputTarResult(
        tar_path=tar_path,
        member_count=len(files),
        size_bytes=tar_path.stat().st_size,
    )


def _create_outputs_tar_zstd_members_cpu(
    files: list[tuple[Path, Path]],
    tar_path: Path,
    *,
    compressor: ZstdCompressor,
) -> None:
    with (
        tarfile.open(tar_path, "w") as archive,
        tempfile.TemporaryDirectory(prefix=".zstd_members_", dir=tar_path.parent) as tmp,
    ):
        tmp_dir = Path(tmp)
        for index, (local_path, relative_path) in enumerate(files):
            _add_zstd_member(
                archive,
                local_path,
                relative_path,
                tmp_dir=tmp_dir,
                compressor=compressor,
                index=index,
            )


def _create_outputs_tar_zstd_members_gpu(files: list[tuple[Path, Path]], tar_path: Path) -> None:
    to_compress = [
        (local_path, relative_path) for local_path, relative_path in files if not relative_path.name.endswith(".zst")
    ]
    compressed_payloads = iter(_compress_files_zstd_gpu_batched(to_compress))
    tmp_tar_path = tar_path.with_name(f".{tar_path.name}.nvcomp.tmp")
    tmp_tar_path.unlink(missing_ok=True)
    try:
        with tarfile.open(tmp_tar_path, "w") as archive:
            for local_path, relative_path in files:
                if relative_path.name.endswith(".zst"):
                    archive.add(local_path, arcname=relative_path.as_posix())
                    continue
                _add_zstd_member_from_bytes(
                    archive,
                    local_path,
                    zstd_member_arcname(relative_path),
                    next(compressed_payloads),
                )
        tmp_tar_path.replace(tar_path)
    except Exception:
        tmp_tar_path.unlink(missing_ok=True)
        raise


def _add_zstd_member(
    archive: tarfile.TarFile,
    local_path: Path,
    relative_path: Path,
    *,
    tmp_dir: Path,
    compressor: ZstdCompressor,
    index: int,
) -> None:
    if relative_path.name.endswith(".zst"):
        archive.add(local_path, arcname=relative_path.as_posix())
        return

    compressed = tmp_dir / f"member_{index}.zst"
    compressor(local_path, compressed)
    arcname = zstd_member_arcname(relative_path).as_posix()
    info = archive.gettarinfo(str(compressed), arcname=arcname)
    source_stat = local_path.stat()
    info.mtime = int(source_stat.st_mtime)
    info.mode = source_stat.st_mode & 0o777
    info.uid = source_stat.st_uid
    info.gid = source_stat.st_gid
    info.uname = ""
    info.gname = ""
    with compressed.open("rb") as data:
        archive.addfile(info, data)
    compressed.unlink(missing_ok=True)


def _add_zstd_member_from_bytes(
    archive: tarfile.TarFile,
    source_path: Path,
    arcname: Path,
    payload: bytes,
) -> None:
    source_stat = source_path.stat()
    info = tarfile.TarInfo(arcname.as_posix())
    info.size = len(payload)
    info.mtime = int(source_stat.st_mtime)
    info.mode = source_stat.st_mode & 0o777
    info.uid = source_stat.st_uid
    info.gid = source_stat.st_gid
    info.uname = ""
    info.gname = ""
    archive.addfile(info, BytesIO(payload))


def _default_zstd_compressor(zstd_path: str | None) -> ZstdCompressor:
    resolved_zstd = zstd_path or shutil.which("zstd")
    if not resolved_zstd:
        msg = "compression='zstd-members' requires zstd on PATH or an injected compressor"
        raise RuntimeError(msg)

    def _compress(source: Path, destination: Path) -> None:
        compress_file_to_zstd(source, destination, zstd_path=resolved_zstd)

    return _compress


def _compress_files_zstd_gpu_batched(
    files: list[tuple[Path, Path]],
    *,
    sub_batch_size: int = _NVCOMP_ZSTD_SUB_BATCH_SIZE,
) -> list[bytes]:
    if not files:
        return []
    if sub_batch_size < 1:
        msg = "sub_batch_size must be at least 1"
        raise ValueError(msg)

    codec = _get_nvcomp_zstd_codec()
    compressed: list[bytes] = []
    index = 0
    current_sub_batch_size = sub_batch_size
    while index < len(files):
        batch = files[index : index + current_sub_batch_size]
        try:
            compressed.extend(_compress_file_batch_zstd_gpu(codec, batch))
        except RuntimeError as exc:
            if current_sub_batch_size <= 1 or not _is_cuda_oom(exc):
                raise
            current_sub_batch_size = max(1, current_sub_batch_size // 2)
            _LOGGER.warning("nvCOMP zstd encode OOM; retrying with sub_batch_size=%s", current_sub_batch_size)
            continue
        index += len(batch)
        current_sub_batch_size = sub_batch_size
    return compressed


def _compress_file_batch_zstd_gpu(codec: _NvcompCodec, files: list[tuple[Path, Path]]) -> list[bytes]:
    import numpy as np

    nvcomp = cast(Any, import_module("nvidia.nvcomp"))
    payloads = [local_path.read_bytes() for local_path, _ in files]
    arrays = [nvcomp.as_array(np.frombuffer(payload, dtype=np.uint8)) for payload in payloads]
    _quiesce_cuda_before_nvcomp()
    encoded = codec.encode(arrays)
    return [bytes(item.cpu()) for item in encoded]


def _get_nvcomp_zstd_codec() -> _NvcompCodec:
    nvcomp = cast(Any, import_module("nvidia.nvcomp"))

    return cast(_NvcompCodec, nvcomp.Codec(algorithm="Zstd", bitstream_kind=nvcomp.BitstreamKind.RAW))


def _quiesce_cuda_before_nvcomp() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _is_cuda_oom(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


__all__ = [
    "OutputTarResult",
    "ZstdCompressor",
    "compress_file_to_zstd",
    "create_outputs_tar",
    "zstd_member_arcname",
]
