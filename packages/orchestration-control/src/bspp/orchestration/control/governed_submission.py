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

"""Governed create-once submission and immutable evidence integration."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bspp.orchestration.contract.submission_evidence import (
    EvidenceIndex,
    EvidenceIndexEntry,
    EvidenceIndexValidation,
    SubmissionExpectation,
    SubmissionResult,
    SubmissionToken,
    evidence_index_from_bytes,
    validate_submission_result,
)
from bspp.orchestration.control.submission_coordinator import SubmissionClaim, SubmissionCoordinator

MAX_BOUND_ARTIFACT_BYTES = 16 * 1024 * 1024
FINAL_INDEX_NAME = "evidence-index.json"
PROCESSING_INDEX_NAME = "processing-provenance-index.json"


class SubmissionAmbiguousError(ValueError):
    """The scheduler may have accepted a claim whose job identity is unknown."""


@dataclass(frozen=True)
class SchedulerSubmission:
    job_id: str
    status: str


@dataclass(frozen=True)
class ClaimedSubmission:
    job_id: str
    scheduler_status: str
    submitted_now: bool


@dataclass(frozen=True)
class FinalizedEvidenceIndex:
    index_path: Path
    index: EvidenceIndex


class Coordinator(Protocol):
    def claim(self, token: SubmissionToken) -> SubmissionClaim: ...

    def bind_job(self, token: SubmissionToken, *, job_id: str, scheduler_status: str) -> object: ...


def stable_read(path: Path, *, maximum_bytes: int = MAX_BOUND_ARTIFACT_BYTES) -> bytes:
    """Read one regular, exclusive file while proving its identity stayed stable."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"bound artifact must be a regular file: {path}")
    if before.st_nlink != 1:
        raise ValueError(f"bound artifact must not be hard-linked: {path}")
    if before.st_size > maximum_bytes:
        raise ValueError(f"bound artifact exceeds size bound: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if _metadata(opened) != _metadata(before):
            raise ValueError(f"bound artifact changed during open: {path}")
        data = handle.read(maximum_bytes + 1)
        finished = os.fstat(handle.fileno())
    current = path.lstat()
    if len(data) > maximum_bytes:
        raise ValueError(f"bound artifact exceeds size bound: {path}")
    if _metadata(finished) != _metadata(opened) or _metadata(current) != _metadata(opened):
        raise ValueError(f"bound artifact changed during read: {path}")
    return data


def stable_sha256(path: Path) -> str:
    return hashlib.sha256(stable_read(path)).hexdigest()


def build_governed_submission_expectation(
    *,
    run_id: str,
    step_index: int,
    step_name: str,
    slice_id: str,
    attempt: int,
    runspec_path: Path,
    script_path: Path,
    bootstrap_path: Path | None = None,
    bootstrap_sha256: str | None = None,
    control_state_path: Path,
    runtime_qualification_path: Path,
) -> SubmissionExpectation:
    """Bind the exact bytes and qualified iPSAE identity used by one submission."""
    runspec = stable_read(runspec_path)
    script = stable_read(script_path)
    if (bootstrap_path is None) == (bootstrap_sha256 is None):
        raise ValueError("provide exactly one bootstrap identity")
    control_state = stable_read(control_state_path)
    qualification = stable_read(runtime_qualification_path)
    source_revision, binary_sha256 = _runtime_ipsae_identity(qualification)
    digests = {
        "runspec": hashlib.sha256(runspec).hexdigest(),
        "script": hashlib.sha256(script).hexdigest(),
        "bootstrap": bootstrap_sha256 or stable_sha256(bootstrap_path),  # type: ignore[arg-type]
        "control_state": hashlib.sha256(control_state).hexdigest(),
        "runtime_qualification": hashlib.sha256(qualification).hexdigest(),
    }
    token = SubmissionToken.create(
        run_id=run_id,
        runspec_sha256=digests["runspec"],
        step_index=step_index,
        step_name=step_name,
        slice_id=slice_id,
        attempt=attempt,
        script_sha256=digests["script"],
        bootstrap_sha256=digests["bootstrap"],
        control_state_sha256=digests["control_state"],
        runtime_qualification_sha256=digests["runtime_qualification"],
    )
    return SubmissionExpectation(
        token,
        digests["script"],
        digests["control_state"],
        digests["bootstrap"],
        digests["runtime_qualification"],
        source_revision,
        binary_sha256,
    )


def submit_claimed_once(
    *,
    expectation: SubmissionExpectation,
    coordinator: Coordinator,
    submit: Callable[[], SchedulerSubmission],
    reconcile: Callable[[str], str | None] | None = None,
) -> ClaimedSubmission:
    """Submit only a newly-created claim; every restart reuses or fails closed."""
    claim = coordinator.claim(expectation.token)
    if not claim.created:
        if claim.record.job_id is None:
            reconciled_job_id = reconcile(expectation.token.token) if reconcile is not None else None
            if reconciled_job_id is not None:
                try:
                    coordinator.bind_job(
                        expectation.token,
                        job_id=reconciled_job_id,
                        scheduler_status="RECONCILED",
                    )
                except Exception as exc:
                    raise SubmissionAmbiguousError(
                        "scheduler job was reconciled but its durable coordinator binding failed"
                    ) from exc
                return ClaimedSubmission(reconciled_job_id, "RECONCILED", False)
            raise SubmissionAmbiguousError(
                "submission claim exists without a bound scheduler job; reconcile the exact token before retrying"
            )
        if claim.record.scheduler_status is None:
            raise SubmissionAmbiguousError("submission claim has an incomplete scheduler binding")
        return ClaimedSubmission(claim.record.job_id, claim.record.scheduler_status, False)
    try:
        submitted = submit()
    except Exception as exc:
        raise SubmissionAmbiguousError(
            "scheduler submission outcome is uncertain; the create-once claim forbids automatic resubmission"
        ) from exc
    try:
        coordinator.bind_job(expectation.token, job_id=submitted.job_id, scheduler_status=submitted.status)
    except Exception as exc:
        raise SubmissionAmbiguousError(
            "scheduler accepted the submission but its durable coordinator binding failed"
        ) from exc
    return ClaimedSubmission(submitted.job_id, submitted.status, True)


def validate_bound_submission_result(
    expectation: SubmissionExpectation,
    result: SubmissionResult,
    *,
    coordinator_job_id: str,
) -> None:
    validate_submission_result(expectation, result)
    if result.job_id != coordinator_job_id:
        raise ValueError("SubmissionResult job does not match coordinator job")
    observations = dict(result.runtime_observations)
    expected = {
        "runtime_ipsae_source_revision": expectation.runtime_ipsae_source_revision,
        "runtime_ipsae_binary_sha256": expectation.runtime_ipsae_binary_sha256,
    }
    if any(observations.get(name) != value for name, value in expected.items()):
        raise ValueError("SubmissionResult runtime iPSAE observations do not match qualification evidence")


def write_immutable(path: Path, data: bytes) -> Path:
    """Create one durable immutable record without replacing an existing path."""
    return write_immutable_descendant(path.parent, Path(path.name), data)


def write_immutable_descendant(root: Path, relative_path: Path, data: bytes) -> Path:
    """Create a durable record below *root* without following any descendant link."""
    parts = _validated_relative_parts(relative_path)
    parent_descriptor = _open_descendant_directory(root, parts[:-1], create=True)
    try:
        descriptor = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("stalled immutable evidence write")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent_descriptor)
        _revalidate_descendant_directory(root, parts[:-1], parent_descriptor)
    except (NotADirectoryError, OSError) as exc:
        if isinstance(exc, FileExistsError):
            raise
        raise ValueError(f"immutable evidence path is not a safe directory descendant: {relative_path}") from exc
    finally:
        os.close(parent_descriptor)
    return root / relative_path


def stable_read_descendant(
    root: Path,
    relative_path: Path,
    *,
    maximum_bytes: int = MAX_BOUND_ARTIFACT_BYTES,
) -> bytes:
    """Read a stable regular file without following links below *root*."""
    parts = _validated_relative_parts(relative_path)
    parent_descriptor = _open_descendant_directory(root, parts[:-1], create=False)
    try:
        before = os.stat(parts[-1], dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"bound artifact must be an exclusive regular file: {relative_path}")
        if before.st_size > maximum_bytes:
            raise ValueError(f"bound artifact exceeds size bound: {relative_path}")
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        try:
            opened = os.fstat(descriptor)
            if _metadata(opened) != _metadata(before):
                raise ValueError(f"bound artifact changed during open: {relative_path}")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > maximum_bytes:
                    break
            data = b"".join(chunks)
            finished = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = os.stat(parts[-1], dir_fd=parent_descriptor, follow_symlinks=False)
        if len(data) > maximum_bytes:
            raise ValueError(f"bound artifact exceeds size bound: {relative_path}")
        if _metadata(finished) != _metadata(opened) or _metadata(current) != _metadata(opened):
            raise ValueError(f"bound artifact changed during read: {relative_path}")
        return data
    finally:
        os.close(parent_descriptor)


def finalize_evidence_index(root: Path) -> FinalizedEvidenceIndex:
    index_path = root / FINAL_INDEX_NAME
    index = _build_final_index(root, excluded=index_path)
    data = _canonical_json(index.to_mapping())
    if os.path.lexists(index_path):
        existing = stable_read(index_path)
        if existing != data:
            raise ValueError("final evidence index already exists with different bytes")
    else:
        write_immutable(index_path, data)
    return FinalizedEvidenceIndex(index_path, index)


def finalize_selected_evidence_index(root: Path, relative_paths: tuple[Path, ...]) -> FinalizedEvidenceIndex:
    """Reconcile an exact processing-provenance inventory and publish its index."""
    expected = {path.as_posix() for path in relative_paths}
    if len(expected) != len(relative_paths):
        raise ValueError("duplicate processing provenance path")
    index_path = root / PROCESSING_INDEX_NAME
    observed_tree = _build_final_index(root, excluded=index_path)
    observed = {
        entry.path
        for entry in observed_tree.entries
        if entry.path.startswith("submissions/") or entry.path.startswith("submission-coordinator/")
    }
    missing, extra = expected - observed, observed - expected
    if missing or extra:
        raise ValueError(f"processing provenance inventory mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    entries = tuple(
        sorted(
            (
                EvidenceIndexEntry(path, hashlib.sha256(stable_read_descendant(root, Path(path))).hexdigest())
                for path in expected
            ),
            key=lambda entry: entry.path.encode(),
        )
    )
    index = EvidenceIndex(1, entries)
    data = _canonical_json(index.to_mapping())
    if os.path.lexists(index_path):
        if stable_read_descendant(root, Path(PROCESSING_INDEX_NAME)) != data:
            raise ValueError("processing provenance index already exists with different bytes")
    else:
        write_immutable_descendant(root, Path(PROCESSING_INDEX_NAME), data)
    return FinalizedEvidenceIndex(index_path, index)


def read_evidence_index(path: Path) -> EvidenceIndex:
    return evidence_index_from_bytes(stable_read(path))


def verify_final_evidence_index(root: Path, index: EvidenceIndex) -> EvidenceIndexValidation:
    try:
        actual = _build_final_index(root, excluded=root / FINAL_INDEX_NAME)
    except (OSError, ValueError) as exc:
        return EvidenceIndexValidation(False, (str(exc),))
    expected = {entry.path: entry.sha256 for entry in index.entries}
    observed = {entry.path: entry.sha256 for entry in actual.entries}
    issues = [f"missing evidence path: {path}" for path in sorted(set(expected) - set(observed))]
    issues.extend(f"unexpected evidence path: {path}" for path in sorted(set(observed) - set(expected)))
    issues.extend(
        f"evidence digest mismatch: {path}"
        for path in sorted(set(expected) & set(observed))
        if expected[path] != observed[path]
    )
    return EvidenceIndexValidation(not issues, tuple(issues))


def _runtime_ipsae_identity(data: bytes) -> tuple[str, str]:
    try:
        payload = json.loads(data)
        evidence = payload["smoke_evidence"]["runtime_ipsae"]
        source_revision = evidence["source_revision"]
        binary_sha256 = evidence["binary"]["sha256"]
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("runtime qualification lacks runtime iPSAE evidence") from exc
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("runtime qualification has invalid runtime iPSAE source revision")
    if not isinstance(binary_sha256, str) or len(binary_sha256) != 64 or not _is_lower_hex(binary_sha256):
        raise ValueError("runtime qualification has invalid runtime iPSAE binary SHA-256")
    return source_revision, binary_sha256


def _build_final_index(root: Path, *, excluded: Path) -> EvidenceIndex:
    metadata = root.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
        raise ValueError("evidence root must be a real directory")
    excluded_relative = excluded.relative_to(root).as_posix()
    root_descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened_root = os.fstat(root_descriptor)
        if _metadata(opened_root) != _metadata(metadata):
            raise ValueError("evidence root changed during open")
        entries = _index_directory(root_descriptor, Path(), excluded_relative)
        current_root = root.lstat()
        if _metadata(current_root) != _metadata(opened_root):
            raise ValueError("evidence root changed during traversal")
    finally:
        os.close(root_descriptor)
    return EvidenceIndex(1, tuple(sorted(entries, key=lambda entry: entry.path.encode())))


def _index_directory(
    directory_descriptor: int,
    relative_directory: Path,
    excluded_relative: str,
) -> list[EvidenceIndexEntry]:
    entries: list[EvidenceIndexEntry] = []
    for name in sorted(os.listdir(directory_descriptor), key=os.fsencode):
        relative = relative_directory / name
        relative_name = relative.as_posix()
        if relative_name == excluded_relative:
            continue
        item = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if stat.S_ISDIR(item.st_mode):
            try:
                child_descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise ValueError(f"evidence directory changed during traversal: {relative_name}") from exc
            try:
                opened = os.fstat(child_descriptor)
                if _metadata(opened) != _metadata(item):
                    raise ValueError(f"evidence directory changed during traversal: {relative_name}")
                entries.extend(_index_directory(child_descriptor, relative, excluded_relative))
            finally:
                os.close(child_descriptor)
            current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            if _metadata(current) != _metadata(item):
                raise ValueError(f"evidence directory changed during traversal: {relative_name}")
            continue
        if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
            raise ValueError(
                f"evidence must contain only real directories and exclusive regular files: {relative_name}"
            )
        entries.append(EvidenceIndexEntry(relative_name, _stable_sha256_at(directory_descriptor, name, item)))
    return entries


def _stable_sha256_at(directory_descriptor: int, name: str, before: os.stat_result) -> str:
    descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_descriptor)
    try:
        opened = os.fstat(descriptor)
        if _metadata(opened) != _metadata(before):
            raise ValueError(f"evidence file changed during open: {name}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        finished = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if _metadata(finished) != _metadata(opened) or _metadata(current) != _metadata(opened):
        raise ValueError(f"evidence file changed during read: {name}")
    return digest.hexdigest()


def _validated_relative_parts(path: Path) -> tuple[str, ...]:
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"immutable evidence path must be a normalized relative path: {path}")
    return path.parts


def _open_descendant_directory(root: Path, parts: tuple[str, ...], *, create: bool) -> int:
    metadata = root.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
        raise ValueError("immutable evidence root must be a real directory")
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        if _metadata(os.fstat(descriptor)) != _metadata(metadata):
            raise ValueError("immutable evidence root changed during open")
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise ValueError(f"immutable evidence directory is unsafe: {part}") from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _revalidate_descendant_directory(root: Path, parts: tuple[str, ...], opened_descriptor: int) -> None:
    """Prove the held directory is still reachable through the bound root chain."""
    current_descriptor = _open_descendant_directory(root, parts, create=False)
    try:
        if _metadata(os.fstat(current_descriptor)) != _metadata(os.fstat(opened_descriptor)):
            raise ValueError("immutable evidence directory was replaced during operation")
    finally:
        os.close(current_descriptor)


def _metadata(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _is_lower_hex(value: str) -> bool:
    return all(character in "0123456789abcdef" for character in value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ClaimedSubmission",
    "FinalizedEvidenceIndex",
    "SchedulerSubmission",
    "SubmissionAmbiguousError",
    "SubmissionClaim",
    "SubmissionCoordinator",
    "build_governed_submission_expectation",
    "finalize_evidence_index",
    "finalize_selected_evidence_index",
    "read_evidence_index",
    "stable_read",
    "stable_read_descendant",
    "stable_sha256",
    "submit_claimed_once",
    "validate_bound_submission_result",
    "verify_final_evidence_index",
    "write_immutable",
    "write_immutable_descendant",
]
