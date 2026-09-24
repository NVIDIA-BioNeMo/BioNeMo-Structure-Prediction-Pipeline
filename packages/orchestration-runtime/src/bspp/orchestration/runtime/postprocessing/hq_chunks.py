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

"""Build high-quality chunk tars from accepted local-tar outputs."""

from __future__ import annotations

import csv
import hashlib
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary

DEFAULT_HQ_CHUNK_SIZE = 1000
DEFAULT_HQ_CHUNK_SUFFIXES = (
    "-model_v1.cif.zst",
    "-model_v1.pdb.zst",
    "-model_v1.bcif.zst",
    "-predicted_aligned_error_v1.json.zst",
    "-confidence_v1.json.zst",
)


@dataclass(frozen=True, slots=True)
class ModelTarIndexEntry:
    """One model-to-local-tar mapping from ``model_tar_index.csv``."""

    model_id: str
    tar_path: Path | None
    raw_location: str


@dataclass(frozen=True, slots=True)
class HqChunkBuildChunk:
    """One packed HQ chunk tar."""

    chunk_index: int
    tar_path: Path
    file_count: int
    size_bytes: int
    sha256: str
    payload_sha256: str

    def to_redacted_dict(self) -> dict[str, object]:
        return {
            "chunk_index": self.chunk_index,
            "tar_path": str(self.tar_path),
            "file_count": self.file_count,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class HqChunkBuildResult:
    """Summary of an HQ chunk build."""

    selected_ids_path: Path
    model_tar_index_path: Path
    staging_root: Path
    chunks_dir: Path
    chunk_size: int
    selected_model_count: int
    source_tar_count: int
    extracted_file_count: int
    chunk_count: int
    chunks: tuple[HqChunkBuildChunk, ...]

    @property
    def ok(self) -> bool:
        return self.chunk_count == len(self.chunks)

    def to_redacted_dict(self) -> dict[str, object]:
        return {
            "selected_ids_path": str(self.selected_ids_path),
            "model_tar_index_path": str(self.model_tar_index_path),
            "staging_root": str(self.staging_root),
            "chunks_dir": str(self.chunks_dir),
            "chunk_size": self.chunk_size,
            "selected_model_count": self.selected_model_count,
            "source_tar_count": self.source_tar_count,
            "extracted_file_count": self.extracted_file_count,
            "chunk_count": self.chunk_count,
            "ok": self.ok,
            "chunks": [chunk.to_redacted_dict() for chunk in self.chunks],
        }


def build_hq_chunks(
    *,
    selected_ids_path: Path,
    model_tar_index_path: Path,
    staging_root: Path,
    chunks_dir: Path,
    local_tar_root: Path | None = None,
    chunk_size: int = DEFAULT_HQ_CHUNK_SIZE,
    expected_suffixes: tuple[str, ...] = DEFAULT_HQ_CHUNK_SUFFIXES,
) -> HqChunkBuildResult:
    """Build chunked HQ tars from local-tar outputs.

    The builder intentionally consumes orchestration-owned finalizer artifacts
    rather than invoking ``AFDB-Integration-Kit/slurm-scaling`` tooling.
    """
    if chunk_size < 1:
        msg = f"chunk_size must be positive, got {chunk_size}"
        raise ValueError(msg)
    if not expected_suffixes:
        msg = "expected_suffixes must be non-empty"
        raise ValueError(msg)
    if staging_root.resolve() == chunks_dir.resolve():
        msg = "staging_root and chunks_dir must be different directories"
        raise ValueError(msg)

    selected_ids = read_selected_model_ids(selected_ids_path)
    if not selected_ids:
        msg = f"{selected_ids_path} contains no selected model IDs"
        raise ValueError(msg)
    chunk_by_model = {model_id: index // chunk_size for index, model_id in enumerate(selected_ids)}
    expected_chunk_count = _expected_chunk_count(len(selected_ids), chunk_size)

    index_entries = read_model_tar_index(model_tar_index_path, local_tar_root=local_tar_root)
    source_tars: dict[Path, set[str]] = {}
    missing_index_ids: list[str] = []
    blank_index_ids: list[str] = []
    for model_id in selected_ids:
        entry = index_entries.get(model_id)
        if entry is None:
            missing_index_ids.append(model_id)
        elif entry.tar_path is None:
            blank_index_ids.append(model_id)
        else:
            source_tars.setdefault(entry.tar_path, set()).add(model_id)
    if missing_index_ids:
        msg = f"selected IDs missing from model_tar_index.csv: {_sample_message(missing_index_ids)}"
        raise ValueError(msg)
    if blank_index_ids:
        msg = f"selected IDs have blank tar paths in model_tar_index.csv: {_sample_message(blank_index_ids)}"
        raise ValueError(msg)

    _prepare_staging_root(staging_root)
    _prepare_chunks_dir(chunks_dir)
    extracted_members: set[str] = set()
    extracted_file_count = 0
    for tar_path, target_ids in sorted(source_tars.items(), key=lambda item: str(item[0])):
        extracted_file_count += _extract_selected_members(
            tar_path=tar_path,
            target_ids=frozenset(target_ids),
            chunk_by_model=chunk_by_model,
            staging_root=staging_root,
            expected_suffixes=expected_suffixes,
            extracted_members=extracted_members,
        )

    _assert_staged_outputs_complete(
        selected_ids=selected_ids,
        staging_root=staging_root,
        chunk_size=chunk_size,
        expected_suffixes=expected_suffixes,
    )
    chunks = tuple(
        _pack_chunk(staging_root / f"chunk_{chunk_index:04d}", chunks_dir / f"chunk_{chunk_index:04d}.tar", chunk_index)
        for chunk_index in range(expected_chunk_count)
    )
    write_hq_chunk_manifest(chunks_dir / "hq_chunks_manifest.csv", chunks)

    return HqChunkBuildResult(
        selected_ids_path=selected_ids_path,
        model_tar_index_path=model_tar_index_path,
        staging_root=staging_root,
        chunks_dir=chunks_dir,
        chunk_size=chunk_size,
        selected_model_count=len(selected_ids),
        source_tar_count=len(source_tars),
        extracted_file_count=extracted_file_count,
        chunk_count=expected_chunk_count,
        chunks=chunks,
    )


def read_selected_model_ids(path: Path) -> tuple[str, ...]:
    """Read selected model IDs preserving file order."""
    if not path.exists():
        raise FileNotFoundError(path)
    ids = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    duplicates = _duplicates(ids)
    if duplicates:
        msg = f"{path} contains duplicate selected model IDs: {_sample_message(list(duplicates))}"
        raise ValueError(msg)
    return ids


def read_model_tar_index(
    path: Path,
    *,
    local_tar_root: Path | None = None,
) -> dict[str, ModelTarIndexEntry]:
    """Read a model tar index produced by the native finalizer or legacy tools."""
    if not path.exists():
        raise FileNotFoundError(path)
    entries: dict[str, ModelTarIndexEntry] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            msg = f"CSV has no header: {path}"
            raise ValueError(msg)
        if "model_id" not in reader.fieldnames:
            msg = f"{path} is missing required column: model_id"
            raise ValueError(msg)
        if not {"tar_path", "output_tar_s3_uri", "s3_uri"} & set(reader.fieldnames):
            msg = f"{path} requires one of tar_path, output_tar_s3_uri, or s3_uri"
            raise ValueError(msg)
        for row_number, row in enumerate(reader, start=2):
            model_id = (row.get("model_id") or "").strip()
            if not model_id:
                msg = f"{path} row {row_number} has empty model_id"
                raise ValueError(msg)
            if model_id in entries:
                msg = f"{path} has duplicate model_id: {model_id}"
                raise ValueError(msg)
            raw_location = _first_nonempty(row, ("tar_path", "output_tar_s3_uri", "s3_uri"))
            entries[model_id] = ModelTarIndexEntry(
                model_id=model_id,
                tar_path=resolve_index_tar_path(raw_location, index_path=path, local_tar_root=local_tar_root),
                raw_location=raw_location,
            )
    return entries


def resolve_index_tar_path(
    value: str,
    *,
    index_path: Path,
    local_tar_root: Path | None = None,
) -> Path | None:
    """Resolve a model-tar-index location to a local filesystem path."""
    location = value.strip()
    if not location:
        return None
    file_uri = _path_from_file_uri(location)
    if file_uri is not None:
        return file_uri
    parsed = urlparse(location)
    if parsed.scheme and parsed.scheme != "file":
        msg = f"HQ chunk building requires local/materialized tar paths, got {location!r}"
        raise ValueError(msg)
    raw_path = Path(location)
    if raw_path.is_absolute():
        return raw_path
    root = local_tar_root or index_path.parent
    if local_tar_root is not None and raw_path.parts and raw_path.parts[0] == local_tar_root.name:
        return local_tar_root.parent / raw_path
    return root / raw_path


def model_id_from_member_name(name: str, expected_suffixes: tuple[str, ...] = DEFAULT_HQ_CHUNK_SUFFIXES) -> str | None:
    """Return the selected model ID implied by an HQ member basename."""
    basename = PurePosixPath(name).name
    for suffix in expected_suffixes:
        if basename.endswith(suffix):
            return basename[: -len(suffix)]
    return None


def write_hq_chunk_manifest(path: Path, chunks: tuple[HqChunkBuildChunk, ...]) -> Path:
    """Write a compact manifest for packed chunk tar files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("chunk_index", "tar_path", "file_count", "size_bytes", "sha256", "payload_sha256"),
        )
        writer.writeheader()
        for chunk in chunks:
            writer.writerow(
                {
                    "chunk_index": chunk.chunk_index,
                    "tar_path": str(chunk.tar_path),
                    "file_count": chunk.file_count,
                    "size_bytes": chunk.size_bytes,
                    "sha256": chunk.sha256,
                    "payload_sha256": chunk.payload_sha256,
                }
            )
    return path


def read_hq_chunk_manifest(path: Path) -> tuple[HqChunkBuildChunk, ...]:
    """Read ``hq_chunks_manifest.csv`` written by :func:`write_hq_chunk_manifest`."""
    if not path.exists():
        raise FileNotFoundError(path)
    chunks: list[HqChunkBuildChunk] = []
    required = {"chunk_index", "tar_path", "file_count", "size_bytes", "sha256", "payload_sha256"}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            msg = f"CSV has no header: {path}"
            raise ValueError(msg)
        missing = required - set(reader.fieldnames)
        if missing:
            msg = f"{path} is missing required columns: {', '.join(sorted(missing))}"
            raise ValueError(msg)
        seen: set[int] = set()
        for row_number, row in enumerate(reader, start=2):
            chunk_index = _read_int(row, "chunk_index", path=path, row_number=row_number)
            if chunk_index in seen:
                msg = f"{path} has duplicate chunk_index: {chunk_index}"
                raise ValueError(msg)
            seen.add(chunk_index)
            tar_path = (row.get("tar_path") or "").strip()
            if not tar_path:
                msg = f"{path} row {row_number} has empty tar_path"
                raise ValueError(msg)
            resolved_tar_path = Path(tar_path)
            if not resolved_tar_path.is_absolute():
                resolved_tar_path = path.parent / resolved_tar_path
            chunks.append(
                HqChunkBuildChunk(
                    chunk_index=chunk_index,
                    tar_path=resolved_tar_path,
                    file_count=_read_int(row, "file_count", path=path, row_number=row_number),
                    size_bytes=_read_int(row, "size_bytes", path=path, row_number=row_number),
                    sha256=(row.get("sha256") or "").strip(),
                    payload_sha256=(row.get("payload_sha256") or "").strip(),
                )
            )
    return tuple(sorted(chunks, key=lambda chunk: chunk.chunk_index))


def render_hq_chunk_build_report(report: HqChunkBuildResult) -> str:
    """Render a deterministic JSON HQ chunk build report."""
    return report_to_json(report)


def write_hq_chunk_build_report(report: HqChunkBuildResult, output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and text build reports under *output_dir*."""
    json_path = write_json_report(report, output_dir / "hq_chunks_build_report.json")
    text_path = write_text_summary(report, output_dir / "hq_chunks_build_report.txt")
    return json_path, text_path


def _extract_selected_members(
    *,
    tar_path: Path,
    target_ids: frozenset[str],
    chunk_by_model: dict[str, int],
    staging_root: Path,
    expected_suffixes: tuple[str, ...],
    extracted_members: set[str],
) -> int:
    if not tar_path.exists():
        raise FileNotFoundError(tar_path)
    extracted_count = 0
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            basename = PurePosixPath(member.name).name
            model_id = model_id_from_member_name(basename, expected_suffixes)
            if model_id is None or model_id not in target_ids:
                continue
            if basename in extracted_members:
                msg = f"duplicate HQ member across source tars: {basename}"
                raise ValueError(msg)
            source = archive.extractfile(member)
            if source is None:
                continue
            chunk_dir = staging_root / f"chunk_{chunk_by_model[model_id]:04d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            output_path = chunk_dir / basename
            tmp_path = output_path.with_name(output_path.name + ".tmp")
            with source, tmp_path.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            tmp_path.replace(output_path)
            extracted_members.add(basename)
            extracted_count += 1
    return extracted_count


def _assert_staged_outputs_complete(
    *,
    selected_ids: tuple[str, ...],
    staging_root: Path,
    chunk_size: int,
    expected_suffixes: tuple[str, ...],
) -> None:
    missing: list[str] = []
    for index, model_id in enumerate(selected_ids):
        chunk_dir = staging_root / f"chunk_{index // chunk_size:04d}"
        for suffix in expected_suffixes:
            expected = chunk_dir / f"{model_id}{suffix}"
            if not expected.is_file():
                missing.append(f"{model_id}{suffix}")
    if missing:
        msg = f"HQ chunk staging is missing expected members: {_sample_message(missing)}"
        raise ValueError(msg)


def _pack_chunk(chunk_dir: Path, tar_path: Path, chunk_index: int) -> HqChunkBuildChunk:
    if not chunk_dir.is_dir():
        msg = f"missing chunk staging directory: {chunk_dir}"
        raise FileNotFoundError(msg)
    files = tuple(sorted(path for path in chunk_dir.iterdir() if path.is_file() and not path.name.endswith(".tmp")))
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tar_path.with_name(tar_path.name + ".tmp")
    with tarfile.open(tmp_path, "w") as archive:
        for path in files:
            stat = path.stat()
            info = tarfile.TarInfo(path.name)
            info.size = stat.st_size
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    tmp_path.replace(tar_path)
    return HqChunkBuildChunk(
        chunk_index=chunk_index,
        tar_path=tar_path,
        file_count=len(files),
        size_bytes=tar_path.stat().st_size,
        sha256=_file_sha256(tar_path),
        payload_sha256=_chunk_payload_sha256(tar_path),
    )


def _prepare_staging_root(staging_root: Path) -> None:
    staging_root.mkdir(parents=True, exist_ok=True)
    stale_entries = [
        entry for entry in staging_root.iterdir() if entry.name.startswith("chunk_") or entry.name.endswith(".tmp")
    ]
    for entry in stale_entries:
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _prepare_chunks_dir(chunks_dir: Path) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    stale_entries = [
        entry
        for entry in chunks_dir.iterdir()
        if (entry.is_file() and entry.name.startswith("chunk_") and entry.suffix == ".tar")
        or entry.name in {"hq_chunks_manifest.csv"}
        or entry.name.endswith(".tmp")
    ]
    for entry in stale_entries:
        entry.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _chunk_payload_sha256(tar_path: Path) -> str:
    digest = hashlib.sha256()
    with tarfile.open(tar_path, "r:*") as archive:
        members = sorted((member for member in archive if member.isfile()), key=lambda item: item.name)
        for member in members:
            payload_digest = hashlib.sha256()
            source = archive.extractfile(member)
            if source is None:
                continue
            with source:
                while chunk := source.read(1024 * 1024):
                    payload_digest.update(chunk)
            digest.update(PurePosixPath(member.name).name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(payload_digest.hexdigest().encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _path_from_file_uri(value: str) -> Path | None:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme != "file":
        return None
    if parsed.netloc and parsed.netloc not in {"", "localhost"}:
        msg = f"unsupported file URI host in model tar index: {value}"
        raise ValueError(msg)
    return Path(unquote(parsed.path))


def _first_nonempty(row: dict[str, str], columns: tuple[str, ...]) -> str:
    for column in columns:
        value = (row.get(column) or "").strip()
        if value:
            return value
    return ""


def _read_int(row: dict[str, str], column: str, *, path: Path, row_number: int) -> int:
    value = (row.get(column) or "").strip()
    try:
        return int(value)
    except ValueError as exc:
        msg = f"{path} row {row_number} has invalid integer {column}: {value!r}"
        raise ValueError(msg) from exc


def _expected_chunk_count(selected_count: int, chunk_size: int) -> int:
    if selected_count == 0:
        return 0
    return (selected_count + chunk_size - 1) // chunk_size


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _sample_message(values: list[str], limit: int = 10) -> str:
    sample = ", ".join(values[:limit])
    if len(values) <= limit:
        return sample
    return f"{sample}, ... ({len(values)} total)"


__all__ = [
    "DEFAULT_HQ_CHUNK_SIZE",
    "DEFAULT_HQ_CHUNK_SUFFIXES",
    "HqChunkBuildChunk",
    "HqChunkBuildResult",
    "ModelTarIndexEntry",
    "build_hq_chunks",
    "model_id_from_member_name",
    "read_hq_chunk_manifest",
    "read_model_tar_index",
    "read_selected_model_ids",
    "render_hq_chunk_build_report",
    "resolve_index_tar_path",
    "write_hq_chunk_build_report",
    "write_hq_chunk_manifest",
]
