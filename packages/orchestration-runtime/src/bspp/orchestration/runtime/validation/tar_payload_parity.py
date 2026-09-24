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

"""Tar payload parity checks for local-tar post-processing outputs.

User-facing entry point:

    bspp-orchestration-runtime validate tar-payload-parity

Run full checks inside a CPU Slurm job with ``--workers`` set to the allocated
CPU count. See the tar-payload parity runbook in the project documentation.
"""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal

from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary

TAR_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".tar.zst",
    ".tzst",
)
MatchMode = Literal["by-tar", "aggregate"]


@dataclass(frozen=True)
class TarPayloadFileReport:
    """Payload comparison for one paired tar file."""

    relative_path: str
    baseline_member_count: int | None
    candidate_member_count: int | None
    compared_members: int
    missing_in_candidate: tuple[str, ...]
    extra_in_candidate: tuple[str, ...]
    duplicate_baseline_normalized_names: tuple[str, ...]
    duplicate_candidate_normalized_names: tuple[str, ...]
    compressed_size_mismatch_count: int
    compressed_size_mismatch_sample: tuple[str, ...]
    payload_mismatch_count: int
    payload_mismatch_sample: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def member_names_ok(self) -> bool:
        return (
            self.baseline_member_count is not None
            and self.candidate_member_count is not None
            and not self.missing_in_candidate
            and not self.extra_in_candidate
            and not self.duplicate_baseline_normalized_names
            and not self.duplicate_candidate_normalized_names
        )

    @property
    def payload_ok(self) -> bool:
        return self.payload_mismatch_count == 0 and not self.errors

    @property
    def ok(self) -> bool:
        return self.member_names_ok and self.payload_ok

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable comparison data."""
        return {
            "relative_path": self.relative_path,
            "baseline_member_count": self.baseline_member_count,
            "candidate_member_count": self.candidate_member_count,
            "compared_members": self.compared_members,
            "missing_in_candidate": list(self.missing_in_candidate),
            "extra_in_candidate": list(self.extra_in_candidate),
            "duplicate_baseline_normalized_names": list(self.duplicate_baseline_normalized_names),
            "duplicate_candidate_normalized_names": list(self.duplicate_candidate_normalized_names),
            "compressed_size_mismatch_count": self.compressed_size_mismatch_count,
            "compressed_size_mismatch_sample": list(self.compressed_size_mismatch_sample),
            "payload_mismatch_count": self.payload_mismatch_count,
            "payload_mismatch_sample": list(self.payload_mismatch_sample),
            "errors": list(self.errors),
            "member_names_ok": self.member_names_ok,
            "payload_ok": self.payload_ok,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class TarPayloadParityReport:
    """Full local-tar payload parity report."""

    baseline_dir: Path
    candidate_dir: Path
    relative_dir: str
    match_mode: MatchMode
    workers: int
    baseline_only_tars: tuple[str, ...]
    candidate_only_tars: tuple[str, ...]
    files: tuple[TarPayloadFileReport, ...]
    sample_limit: int = 20
    payload_sample_count: int | None = None

    @property
    def tar_file_list_ok(self) -> bool:
        return not self.baseline_only_tars and not self.candidate_only_tars

    @property
    def ok(self) -> bool:
        return self.tar_file_list_ok and not self.inventory_errors and all(item.ok for item in self.files)

    @property
    def baseline_tar_count(self) -> int:
        return len(self.files) + len(self.baseline_only_tars)

    @property
    def candidate_tar_count(self) -> int:
        return len(self.files) + len(self.candidate_only_tars)

    @property
    def inventory_only(self) -> bool:
        return self.payload_sample_count == 0

    @property
    def inventory_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.files:
            errors.append(
                f"no paired tar files found under relative_dir={self.relative_dir!r} "
                f"(baseline={self.baseline_tar_count}, candidate={self.candidate_tar_count})"
            )
        return tuple(errors)

    @property
    def payload_mismatch_count(self) -> int:
        return sum(item.payload_mismatch_count for item in self.files)

    @property
    def error_count(self) -> int:
        return sum(len(item.errors) for item in self.files) + len(self.inventory_errors)

    @property
    def compared_members(self) -> int:
        return sum(item.compared_members for item in self.files)

    @property
    def duplicate_normalized_member_count(self) -> int:
        return sum(
            len(item.duplicate_baseline_normalized_names) + len(item.duplicate_candidate_normalized_names)
            for item in self.files
        )

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable report data."""
        payload_hash_scope = (
            "all"
            if self.payload_sample_count is None
            else "inventory-only"
            if self.payload_sample_count == 0
            else "sampled"
        )
        return {
            "baseline_dir": str(self.baseline_dir),
            "candidate_dir": str(self.candidate_dir),
            "relative_dir": self.relative_dir,
            "match_mode": self.match_mode,
            "workers": self.workers,
            "payload_hash_scope": payload_hash_scope,
            "sample_limit": self.sample_limit,
            "payload_sample_count": self.payload_sample_count,
            "inventory_only": self.inventory_only,
            "ok": self.ok,
            "tar_file_list_ok": self.tar_file_list_ok,
            "inventory_errors": list(self.inventory_errors),
            "baseline_tar_count": self.baseline_tar_count,
            "candidate_tar_count": self.candidate_tar_count,
            "baseline_only_tars": list(self.baseline_only_tars),
            "candidate_only_tars": list(self.candidate_only_tars),
            "compared_tar_count": len(self.files),
            "compared_members": self.compared_members,
            "sampled_member_count": self.compared_members if self.payload_sample_count is not None else None,
            "duplicate_normalized_member_count": self.duplicate_normalized_member_count,
            "payload_mismatch_count": self.payload_mismatch_count,
            "error_count": self.error_count,
            "files": [item.to_redacted_dict() for item in self.files],
        }


def compare_tar_payload_parity(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    relative_dir: str = "local_tars",
    baseline_run_name: str | None = None,
    candidate_run_name: str | None = None,
    zstd_path: str = "zstd",
    workers: int = 1,
    sample_limit: int = 20,
    payload_sample_count: int | None = None,
    match_mode: MatchMode = "by-tar",
    exclude: tuple[str, ...] = (),
) -> TarPayloadParityReport:
    """Compare local tar payload bytes, decompressing ``.zst`` members.

    Whole-tar bytes and compressed member bytes are not acceptance criteria for
    ``zstd-members`` outputs. This function compares the normalized member
    inventory and then hashes decompressed member payloads.
    """
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)
    baseline_root = baseline_dir / relative_dir
    candidate_root = candidate_dir / relative_dir

    baseline_tars = _tar_inventory(baseline_root, exclude)
    candidate_tars = _tar_inventory(candidate_root, exclude)
    baseline_names = set(baseline_tars)
    candidate_names = set(candidate_tars)
    common = tuple(sorted(baseline_names & candidate_names))
    worker_count = max(1, workers)

    if match_mode == "by-tar":
        jobs = [
            _CompareJob(
                relative_path=relative,
                baseline_path=baseline_tars[relative],
                candidate_path=candidate_tars[relative],
                baseline_run_name=baseline_run_name,
                candidate_run_name=candidate_run_name,
                zstd_path=zstd_path,
                sample_limit=sample_limit,
                payload_sample_count=payload_sample_count,
            )
            for relative in common
        ]

        if worker_count == 1 or len(jobs) <= 1:
            files = tuple(_compare_one_tar(job) for job in jobs)
        else:
            with ProcessPoolExecutor(max_workers=min(worker_count, len(jobs))) as pool:
                files = tuple(pool.map(_compare_one_tar, jobs))
        effective_workers = min(worker_count, max(1, len(jobs)))
    elif match_mode == "aggregate":
        aggregate_report = _compare_aggregate(
            baseline_tars=baseline_tars,
            candidate_tars=candidate_tars,
            baseline_run_name=baseline_run_name,
            candidate_run_name=candidate_run_name,
            zstd_path=zstd_path,
            sample_limit=sample_limit,
            payload_sample_count=payload_sample_count,
            workers=worker_count,
        )
        files = (aggregate_report,)
        effective_workers = min(worker_count, max(1, aggregate_report.compared_members))
    else:
        msg = f"unknown tar payload parity match mode: {match_mode}"
        raise ValueError(msg)

    return TarPayloadParityReport(
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        relative_dir=relative_dir,
        match_mode=match_mode,
        workers=effective_workers,
        baseline_only_tars=tuple(sorted(baseline_names - candidate_names)),
        candidate_only_tars=tuple(sorted(candidate_names - baseline_names)),
        files=files,
        sample_limit=sample_limit,
        payload_sample_count=payload_sample_count,
    )


def render_tar_payload_parity_report(report: TarPayloadParityReport) -> str:
    """Render a deterministic JSON tar payload parity report."""
    return report_to_json(report)


def write_tar_payload_parity_report(report: TarPayloadParityReport, output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and text reports under *output_dir*."""
    json_path = write_json_report(report, output_dir / "tar_payload_parity_report.json")
    text_path = write_text_summary(report, output_dir / "tar_payload_parity_report.txt")
    return json_path, text_path


@dataclass(frozen=True)
class _CompareJob:
    relative_path: str
    baseline_path: Path
    candidate_path: Path
    baseline_run_name: str | None
    candidate_run_name: str | None
    zstd_path: str
    sample_limit: int
    payload_sample_count: int | None = None


@dataclass(frozen=True)
class _MemberInfo:
    tar_relative_path: str
    original_name: str
    normalized_name: str
    size: int
    payload_sha256: str | None


def _tar_inventory(root: Path, exclude: tuple[str, ...] = ()) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    return {
        rel: path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name.endswith(TAR_SUFFIXES)
        and not _is_excluded(rel := str(path.relative_to(root)), exclude)
    }


def _compare_one_tar(job: _CompareJob) -> TarPayloadFileReport:
    hash_normalized_names: frozenset[str] | None = None
    if job.payload_sample_count is not None:
        baseline_inventory = _hash_one_tar(
            _HashTarJob(
                relative_path=job.relative_path,
                tar_path=job.baseline_path,
                baseline_run_name=job.baseline_run_name,
                candidate_run_name=job.candidate_run_name,
                zstd_path=job.zstd_path,
                side="baseline",
                hash_normalized_names=frozenset(),
            )
        )
        candidate_inventory = _hash_one_tar(
            _HashTarJob(
                relative_path=job.relative_path,
                tar_path=job.candidate_path,
                baseline_run_name=job.baseline_run_name,
                candidate_run_name=job.candidate_run_name,
                zstd_path=job.zstd_path,
                side="candidate",
                hash_normalized_names=frozenset(),
            )
        )
        inventory_report = _compare_member_sets(
            job.relative_path,
            baseline_inventory.members,
            candidate_inventory.members,
            errors=list(baseline_inventory.errors + candidate_inventory.errors),
            sample_limit=job.sample_limit,
        )
        if not inventory_report.member_names_ok or inventory_report.errors or job.payload_sample_count == 0:
            return inventory_report

        baseline_map = _unique_member_map(baseline_inventory.members)
        candidate_map = _unique_member_map(candidate_inventory.members)
        common_names = sorted(set(baseline_map) & set(candidate_map))
        hash_normalized_names = frozenset(_select_sample_names(common_names, job.payload_sample_count))

    baseline_result = _hash_one_tar(
        _HashTarJob(
            relative_path=job.relative_path,
            tar_path=job.baseline_path,
            baseline_run_name=job.baseline_run_name,
            candidate_run_name=job.candidate_run_name,
            zstd_path=job.zstd_path,
            side="baseline",
            hash_normalized_names=hash_normalized_names,
        )
    )
    candidate_result = _hash_one_tar(
        _HashTarJob(
            relative_path=job.relative_path,
            tar_path=job.candidate_path,
            baseline_run_name=job.baseline_run_name,
            candidate_run_name=job.candidate_run_name,
            zstd_path=job.zstd_path,
            side="candidate",
            hash_normalized_names=hash_normalized_names,
        )
    )
    return _compare_member_sets(
        job.relative_path,
        baseline_result.members,
        candidate_result.members,
        errors=list(baseline_result.errors + candidate_result.errors),
        sample_limit=job.sample_limit,
    )


@dataclass(frozen=True)
class _HashTarJob:
    relative_path: str
    tar_path: Path
    baseline_run_name: str | None
    candidate_run_name: str | None
    zstd_path: str
    side: str
    hash_normalized_names: frozenset[str] | None = None


@dataclass(frozen=True)
class _HashTarResult:
    side: str
    relative_path: str
    members: tuple[_MemberInfo, ...]
    errors: tuple[str, ...]


def _hash_one_tar(job: _HashTarJob) -> _HashTarResult:
    errors: list[str] = []
    members: list[_MemberInfo] = []
    try:
        with tarfile.open(job.tar_path, mode="r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                try:
                    normalized_name = _normalize_member_name(
                        member.name,
                        job.baseline_run_name,
                        job.candidate_run_name,
                    )
                    if job.hash_normalized_names is None or normalized_name in job.hash_normalized_names:
                        payload_hash = _payload_hash_from_member(archive, member, job.zstd_path)
                    else:
                        payload_hash = None
                except (OSError, RuntimeError, tarfile.TarError) as exc:
                    errors.append(f"{job.side} {job.relative_path}:{member.name}: {exc}")
                    continue
                members.append(
                    _MemberInfo(
                        tar_relative_path=job.relative_path,
                        original_name=member.name,
                        normalized_name=normalized_name,
                        size=member.size,
                        payload_sha256=payload_hash,
                    )
                )
    except (OSError, tarfile.TarError) as exc:
        errors.append(f"{job.side} unreadable tar {job.tar_path}: {exc}")
    return _HashTarResult(
        side=job.side,
        relative_path=job.relative_path,
        members=tuple(members),
        errors=tuple(errors),
    )


def _compare_member_sets(
    relative_path: str,
    baseline_members: tuple[_MemberInfo, ...],
    candidate_members: tuple[_MemberInfo, ...],
    *,
    errors: list[str],
    sample_limit: int,
) -> TarPayloadFileReport:
    baseline_duplicates = _duplicates(baseline_members)
    candidate_duplicates = _duplicates(candidate_members)

    baseline_map = _unique_member_map(baseline_members)
    candidate_map = _unique_member_map(candidate_members)
    baseline_names = set(baseline_map)
    candidate_names = set(candidate_map)
    common_names = tuple(sorted(baseline_names & candidate_names))

    compressed_size_mismatches = [name for name in common_names if baseline_map[name].size != candidate_map[name].size]
    payload_compared_names = [
        name
        for name in common_names
        if baseline_map[name].payload_sha256 is not None and candidate_map[name].payload_sha256 is not None
    ]
    payload_mismatches = [
        name
        for name in payload_compared_names
        if baseline_map[name].payload_sha256 != candidate_map[name].payload_sha256
    ]

    compared_members = len(payload_compared_names) if not baseline_duplicates and not candidate_duplicates else 0
    if baseline_duplicates or candidate_duplicates:
        payload_mismatches = []

    return TarPayloadFileReport(
        relative_path=relative_path,
        baseline_member_count=len(baseline_members),
        candidate_member_count=len(candidate_members),
        compared_members=compared_members,
        missing_in_candidate=_limited(sorted(baseline_names - candidate_names), sample_limit),
        extra_in_candidate=_limited(sorted(candidate_names - baseline_names), sample_limit),
        duplicate_baseline_normalized_names=_limited(baseline_duplicates, sample_limit),
        duplicate_candidate_normalized_names=_limited(candidate_duplicates, sample_limit),
        compressed_size_mismatch_count=len(compressed_size_mismatches),
        compressed_size_mismatch_sample=_limited(compressed_size_mismatches, sample_limit),
        payload_mismatch_count=len(payload_mismatches),
        payload_mismatch_sample=_limited(payload_mismatches, sample_limit),
        errors=tuple(errors[:sample_limit]),
    )


def _normalize_member_name(name: str, baseline_run_name: str | None, candidate_run_name: str | None) -> str:
    normalized = name
    if baseline_run_name:
        normalized = normalized.replace(baseline_run_name, "<RUN_NAME>")
    if candidate_run_name:
        normalized = normalized.replace(candidate_run_name, "<RUN_NAME>")
    return normalized


def _duplicates(members: tuple[_MemberInfo, ...] | None) -> tuple[str, ...]:
    if members is None:
        return ()
    seen: set[str] = set()
    duplicates: set[str] = set()
    for member in members:
        if member.normalized_name in seen:
            duplicates.add(member.normalized_name)
        seen.add(member.normalized_name)
    return tuple(sorted(duplicates))


def _unique_member_map(members: tuple[_MemberInfo, ...] | None) -> dict[str, _MemberInfo]:
    result: dict[str, _MemberInfo] = {}
    if members is None:
        return result
    duplicate_names = set(_duplicates(members))
    for member in members:
        if member.normalized_name not in duplicate_names:
            result[member.normalized_name] = member
    return result


def _compare_aggregate(
    *,
    baseline_tars: dict[str, Path],
    candidate_tars: dict[str, Path],
    baseline_run_name: str | None,
    candidate_run_name: str | None,
    zstd_path: str,
    sample_limit: int,
    payload_sample_count: int | None,
    workers: int,
) -> TarPayloadFileReport:
    hash_normalized_names: frozenset[str] | None = None
    if payload_sample_count is not None:
        inventory_results = _hash_tars(
            tuple(
                _HashTarJob(
                    relative_path=relative_path,
                    tar_path=tar_path,
                    baseline_run_name=baseline_run_name,
                    candidate_run_name=candidate_run_name,
                    zstd_path=zstd_path,
                    side=side,
                    hash_normalized_names=frozenset(),
                )
                for side, tars in (("baseline", baseline_tars), ("candidate", candidate_tars))
                for relative_path, tar_path in sorted(tars.items())
            ),
            workers=workers,
        )
        baseline_inventory = tuple(
            member for result in inventory_results if result.side == "baseline" for member in result.members
        )
        candidate_inventory = tuple(
            member for result in inventory_results if result.side == "candidate" for member in result.members
        )
        inventory_errors = [error for result in inventory_results for error in result.errors]
        inventory_report = _compare_member_sets(
            "<aggregate>",
            baseline_inventory,
            candidate_inventory,
            errors=inventory_errors,
            sample_limit=sample_limit,
        )
        if not inventory_report.member_names_ok or inventory_report.errors or payload_sample_count == 0:
            return inventory_report

        baseline_map = _unique_member_map(baseline_inventory)
        candidate_map = _unique_member_map(candidate_inventory)
        common_names = sorted(set(baseline_map) & set(candidate_map))
        hash_normalized_names = frozenset(_select_sample_names(common_names, payload_sample_count))

    jobs = tuple(
        _HashTarJob(
            relative_path=relative_path,
            tar_path=tar_path,
            baseline_run_name=baseline_run_name,
            candidate_run_name=candidate_run_name,
            zstd_path=zstd_path,
            side=side,
            hash_normalized_names=hash_normalized_names,
        )
        for side, tars in (("baseline", baseline_tars), ("candidate", candidate_tars))
        for relative_path, tar_path in sorted(tars.items())
    )
    results = _hash_tars(jobs, workers=workers)

    baseline_members = tuple(member for result in results if result.side == "baseline" for member in result.members)
    candidate_members = tuple(member for result in results if result.side == "candidate" for member in result.members)
    errors = [error for result in results for error in result.errors]
    return _compare_member_sets(
        "<aggregate>",
        baseline_members,
        candidate_members,
        errors=errors,
        sample_limit=sample_limit,
    )


def _hash_tars(jobs: tuple[_HashTarJob, ...], *, workers: int) -> tuple[_HashTarResult, ...]:
    if workers == 1 or len(jobs) <= 1:
        return tuple(_hash_one_tar(job) for job in jobs)
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return tuple(pool.map(_hash_one_tar, jobs))


def _select_sample_names(names: list[str], count: int) -> tuple[str, ...]:
    if count >= len(names):
        return tuple(names)
    sampled = sorted((_stable_sample_digest(name), name) for name in names)[:count]
    return tuple(sorted(name for _digest, name in sampled))


def _stable_sample_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _payload_hash_from_member(archive: tarfile.TarFile, member: tarfile.TarInfo, zstd_path: str) -> str:
    extracted = archive.extractfile(member)
    if extracted is None:
        msg = f"cannot extract {member.name}"
        raise RuntimeError(msg)
    digest = hashlib.sha256()
    if member.name.endswith(".zst"):
        _hash_zstd_stream(extracted, digest, zstd_path)
    else:
        _hash_stream(extracted, digest)
    return digest.hexdigest()


def _hash_stream(stream: IO[bytes], digest: Any) -> None:
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)


def _hash_zstd_stream(stream: IO[bytes], digest: Any, zstd_path: str) -> None:
    try:
        import zstandard
    except ImportError:
        payload = stream.read()
        digest.update(_zstd_decompress(payload, zstd_path))
        return

    decompressor = zstandard.ZstdDecompressor()
    with decompressor.stream_reader(stream) as reader:
        _hash_stream(reader, digest)


def _zstd_decompress(payload: bytes, zstd_path: str) -> bytes:
    result = subprocess.run(
        [zstd_path, "-dc"],
        input=payload,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        msg = result.stderr.decode(errors="replace").strip() or f"{zstd_path} exited {result.returncode}"
        raise RuntimeError(msg)
    return result.stdout


def _is_excluded(rel: str, exclude: tuple[str, ...]) -> bool:
    return any(rel == p or rel.startswith(p.rstrip("/") + "/") for p in exclude)


def _limited(items: list[str] | tuple[str, ...], limit: int) -> tuple[str, ...]:
    return tuple(items[: max(0, limit)])


__all__ = [
    "MatchMode",
    "TarPayloadFileReport",
    "TarPayloadParityReport",
    "compare_tar_payload_parity",
    "render_tar_payload_parity_report",
    "write_tar_payload_parity_report",
]
