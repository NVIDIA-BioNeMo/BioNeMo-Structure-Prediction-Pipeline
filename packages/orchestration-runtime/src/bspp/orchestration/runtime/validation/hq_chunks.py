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

"""Validate high-quality chunk tar outputs."""

from __future__ import annotations

import hashlib
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary
from bspp.orchestration.runtime.postprocessing.hq_chunks import (
    DEFAULT_HQ_CHUNK_SIZE,
    DEFAULT_HQ_CHUNK_SUFFIXES,
    model_id_from_member_name,
    read_model_tar_index,
    read_selected_model_ids,
)


@dataclass(frozen=True, slots=True)
class HqChunkTarReport:
    """Validation summary for one HQ chunk tar."""

    chunk_index: int
    tar_path: Path
    file_count: int
    expected_file_count: int
    model_count: int
    expected_model_count: int
    payload_sha256: str
    unknown_members: tuple[str, ...]
    unselected_model_ids: tuple[str, ...]
    misplaced_model_ids: tuple[str, ...]
    duplicate_member_names: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            self.file_count == self.expected_file_count
            and self.model_count == self.expected_model_count
            and not self.unknown_members
            and not self.unselected_model_ids
            and not self.misplaced_model_ids
            and not self.duplicate_member_names
            and not self.errors
        )

    def to_redacted_dict(self) -> dict[str, object]:
        return {
            "chunk_index": self.chunk_index,
            "tar_path": str(self.tar_path),
            "file_count": self.file_count,
            "expected_file_count": self.expected_file_count,
            "model_count": self.model_count,
            "expected_model_count": self.expected_model_count,
            "payload_sha256": self.payload_sha256,
            "unknown_members": list(self.unknown_members),
            "unselected_model_ids": list(self.unselected_model_ids),
            "misplaced_model_ids": list(self.misplaced_model_ids),
            "duplicate_member_names": list(self.duplicate_member_names),
            "errors": list(self.errors),
            "ok": self.ok,
        }


@dataclass(frozen=True, slots=True)
class HqChunkValidationReport:
    """Full HQ chunk validation report."""

    chunks_dir: Path
    selected_ids_path: Path
    model_tar_index_path: Path
    chunk_size: int
    expected_suffixes: tuple[str, ...]
    selected_model_count: int
    indexed_selected_model_count: int
    expected_chunk_count: int
    actual_chunk_count: int
    missing_chunk_tars: tuple[str, ...]
    extra_chunk_tars: tuple[str, ...]
    missing_model_ids: tuple[str, ...]
    missing_member_names: tuple[str, ...]
    duplicate_selected_ids: tuple[str, ...]
    blank_index_model_ids: tuple[str, ...]
    errors: tuple[str, ...]
    chunks: tuple[HqChunkTarReport, ...]

    @property
    def ok(self) -> bool:
        return (
            not self.missing_chunk_tars
            and not self.extra_chunk_tars
            and not self.missing_model_ids
            and not self.missing_member_names
            and not self.duplicate_selected_ids
            and not self.blank_index_model_ids
            and not self.errors
            and all(chunk.ok for chunk in self.chunks)
        )

    @property
    def payload_hashes_recorded(self) -> bool:
        return bool(self.chunks) and all(bool(chunk.payload_sha256) for chunk in self.chunks)

    def to_redacted_dict(self) -> dict[str, object]:
        return {
            "chunks_dir": str(self.chunks_dir),
            "selected_ids_path": str(self.selected_ids_path),
            "model_tar_index_path": str(self.model_tar_index_path),
            "chunk_size": self.chunk_size,
            "expected_suffixes": list(self.expected_suffixes),
            "selected_model_count": self.selected_model_count,
            "indexed_selected_model_count": self.indexed_selected_model_count,
            "expected_chunk_count": self.expected_chunk_count,
            "actual_chunk_count": self.actual_chunk_count,
            "missing_chunk_tars": list(self.missing_chunk_tars),
            "extra_chunk_tars": list(self.extra_chunk_tars),
            "missing_model_ids": list(self.missing_model_ids),
            "missing_member_names": list(self.missing_member_names),
            "duplicate_selected_ids": list(self.duplicate_selected_ids),
            "blank_index_model_ids": list(self.blank_index_model_ids),
            "payload_hashes_recorded": self.payload_hashes_recorded,
            "errors": list(self.errors),
            "chunks": [chunk.to_redacted_dict() for chunk in self.chunks],
            "ok": self.ok,
        }


def validate_hq_chunks(
    *,
    chunks_dir: Path,
    selected_ids_path: Path,
    model_tar_index_path: Path,
    chunk_size: int = DEFAULT_HQ_CHUNK_SIZE,
    expected_suffixes: tuple[str, ...] = DEFAULT_HQ_CHUNK_SUFFIXES,
    sample_limit: int = 20,
) -> HqChunkValidationReport:
    """Validate HQ chunk tar internal consistency against finalizer artifacts."""
    if sample_limit < 1:
        msg = f"sample_limit must be positive, got {sample_limit}"
        raise ValueError(msg)
    errors: list[str] = []
    duplicate_selected_ids: tuple[str, ...] = ()
    try:
        selected_ids = read_selected_model_ids(selected_ids_path)
    except ValueError as exc:
        selected_ids = _read_selected_model_ids_lenient(selected_ids_path)
        duplicate_selected_ids = _duplicates(selected_ids)
        errors.append(str(exc))
    except OSError as exc:
        selected_ids = ()
        errors.append(str(exc))
    selected_set = frozenset(selected_ids)
    model_position = {model_id: index for index, model_id in enumerate(selected_ids)}
    expected_chunk_count = _expected_chunk_count(len(selected_ids), chunk_size) if chunk_size > 0 else 0

    try:
        index_entries = read_model_tar_index(model_tar_index_path)
    except (OSError, ValueError) as exc:
        index_entries = {}
        errors.append(str(exc))
    indexed_selected = tuple(model_id for model_id in selected_ids if model_id in index_entries)
    missing_model_ids = tuple(model_id for model_id in selected_ids if model_id not in index_entries)
    blank_index_model_ids = tuple(model_id for model_id in indexed_selected if index_entries[model_id].tar_path is None)

    actual_tars = _chunk_tar_inventory(chunks_dir)
    expected_tars = {f"chunk_{index:04d}.tar": index for index in range(expected_chunk_count)}
    common_tars = tuple(sorted(set(actual_tars) & set(expected_tars)))
    chunk_reports = tuple(
        _validate_one_chunk(
            chunk_index=expected_tars[name],
            tar_path=actual_tars[name],
            selected_ids=selected_ids,
            selected_set=selected_set,
            model_position=model_position,
            chunk_size=chunk_size,
            expected_suffixes=expected_suffixes,
            sample_limit=sample_limit,
        )
        for name in common_tars
    )
    observed_members: dict[str, set[str]] = {}
    for report in chunk_reports:
        try:
            for model_id, suffixes in _member_suffixes_by_model(
                report.tar_path,
                selected_set=selected_set,
                expected_suffixes=expected_suffixes,
            ).items():
                observed_members.setdefault(model_id, set()).update(suffixes)
        except (OSError, tarfile.TarError) as exc:
            errors.append(f"could not rescan {report.tar_path}: {exc}")

    missing_member_names: list[str] = []
    for model_id in selected_ids:
        suffixes = observed_members.get(model_id, set())
        for suffix in expected_suffixes:
            if suffix not in suffixes:
                missing_member_names.append(f"{model_id}{suffix}")

    return HqChunkValidationReport(
        chunks_dir=chunks_dir,
        selected_ids_path=selected_ids_path,
        model_tar_index_path=model_tar_index_path,
        chunk_size=chunk_size,
        expected_suffixes=expected_suffixes,
        selected_model_count=len(selected_ids),
        indexed_selected_model_count=len(indexed_selected),
        expected_chunk_count=expected_chunk_count,
        actual_chunk_count=len(actual_tars),
        missing_chunk_tars=_limited(tuple(sorted(set(expected_tars) - set(actual_tars))), sample_limit),
        extra_chunk_tars=_limited(tuple(sorted(set(actual_tars) - set(expected_tars))), sample_limit),
        missing_model_ids=_limited(missing_model_ids, sample_limit),
        missing_member_names=_limited(tuple(missing_member_names), sample_limit),
        duplicate_selected_ids=_limited(duplicate_selected_ids, sample_limit),
        blank_index_model_ids=_limited(blank_index_model_ids, sample_limit),
        errors=tuple(errors[:sample_limit]),
        chunks=chunk_reports,
    )


def render_hq_chunk_validation_report(report: HqChunkValidationReport) -> str:
    """Render a deterministic JSON HQ chunk validation report."""
    return report_to_json(report)


def write_hq_chunk_validation_report(report: HqChunkValidationReport, output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and text validation reports under *output_dir*."""
    json_path = write_json_report(report, output_dir / "hq_chunks_validation_report.json")
    text_path = write_text_summary(report, output_dir / "hq_chunks_validation_report.txt")
    return json_path, text_path


def _validate_one_chunk(
    *,
    chunk_index: int,
    tar_path: Path,
    selected_ids: tuple[str, ...],
    selected_set: frozenset[str],
    model_position: dict[str, int],
    chunk_size: int,
    expected_suffixes: tuple[str, ...],
    sample_limit: int,
) -> HqChunkTarReport:
    expected_model_count = _expected_model_count_for_chunk(len(selected_ids), chunk_index, chunk_size)
    expected_file_count = expected_model_count * len(expected_suffixes)
    file_count = 0
    model_ids: set[str] = set()
    unknown_members: list[str] = []
    unselected_model_ids: set[str] = set()
    misplaced_model_ids: set[str] = set()
    duplicate_members: set[str] = set()
    seen_members: set[str] = set()
    errors: list[str] = []
    member_hashes: list[tuple[str, str]] = []

    try:
        with tarfile.open(tar_path, "r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                file_count += 1
                basename = PurePosixPath(member.name).name
                if basename in seen_members:
                    duplicate_members.add(basename)
                    continue
                seen_members.add(basename)
                model_id = model_id_from_member_name(basename, expected_suffixes)
                if model_id is None:
                    unknown_members.append(basename)
                    continue
                if model_id not in selected_set:
                    unselected_model_ids.add(model_id)
                else:
                    model_ids.add(model_id)
                    if model_position[model_id] // chunk_size != chunk_index:
                        misplaced_model_ids.add(model_id)
                source = archive.extractfile(member)
                if source is None:
                    errors.append(f"cannot extract {basename}")
                    continue
                member_digest = hashlib.sha256()
                with source:
                    while chunk := source.read(1024 * 1024):
                        member_digest.update(chunk)
                member_hashes.append((basename, member_digest.hexdigest()))
    except (OSError, tarfile.TarError) as exc:
        errors.append(str(exc))

    return HqChunkTarReport(
        chunk_index=chunk_index,
        tar_path=tar_path,
        file_count=file_count,
        expected_file_count=expected_file_count,
        model_count=len(model_ids),
        expected_model_count=expected_model_count,
        payload_sha256=_payload_evidence_hash(member_hashes),
        unknown_members=_limited(tuple(sorted(unknown_members)), sample_limit),
        unselected_model_ids=_limited(tuple(sorted(unselected_model_ids)), sample_limit),
        misplaced_model_ids=_limited(tuple(sorted(misplaced_model_ids)), sample_limit),
        duplicate_member_names=_limited(tuple(sorted(duplicate_members)), sample_limit),
        errors=tuple(errors[:sample_limit]),
    )


def _payload_evidence_hash(member_hashes: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for name, payload_hash in sorted(member_hashes):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _member_suffixes_by_model(
    tar_path: Path,
    *,
    selected_set: frozenset[str],
    expected_suffixes: tuple[str, ...],
) -> dict[str, set[str]]:
    observed: dict[str, set[str]] = {}
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            basename = PurePosixPath(member.name).name
            model_id = model_id_from_member_name(basename, expected_suffixes)
            if model_id is None or model_id not in selected_set:
                continue
            for suffix in expected_suffixes:
                if basename == f"{model_id}{suffix}":
                    observed.setdefault(model_id, set()).add(suffix)
                    break
    return observed


def _chunk_tar_inventory(chunks_dir: Path) -> dict[str, Path]:
    if not chunks_dir.is_dir():
        return {}
    return {
        path.name: path
        for path in sorted(chunks_dir.iterdir())
        if path.is_file() and path.name.startswith("chunk_") and path.name.endswith(".tar")
    }


def _read_selected_model_ids_lenient(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _expected_chunk_count(selected_count: int, chunk_size: int) -> int:
    if selected_count == 0:
        return 0
    return (selected_count + chunk_size - 1) // chunk_size


def _expected_model_count_for_chunk(selected_count: int, chunk_index: int, chunk_size: int) -> int:
    start = chunk_index * chunk_size
    if start >= selected_count:
        return 0
    return min(chunk_size, selected_count - start)


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _limited(items: tuple[str, ...], limit: int) -> tuple[str, ...]:
    return items[: max(0, limit)]


__all__ = [
    "HqChunkTarReport",
    "HqChunkValidationReport",
    "render_hq_chunk_validation_report",
    "validate_hq_chunks",
    "write_hq_chunk_validation_report",
]
