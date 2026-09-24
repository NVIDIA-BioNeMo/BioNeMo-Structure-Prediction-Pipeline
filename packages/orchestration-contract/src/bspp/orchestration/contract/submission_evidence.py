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

"""Strict v1 submission identity and immutable evidence contracts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

FORMAT_VERSION: Final = 1
MAX_RECORD_BYTES: Final = 1024 * 1024
MAX_INDEX_ENTRIES: Final = 1000


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _hex(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _version(value: object, record: str) -> int:
    if type(value) is not int or value != FORMAT_VERSION:
        raise ValueError(f"Unsupported {record} format_version {value}; supported versions: 1")
    return value


def _fields(payload: dict[str, object], expected: set[str], record: str) -> None:
    missing, extra = expected - set(payload), set(payload) - expected
    if missing or extra:
        raise ValueError(f"Invalid {record} fields; missing={sorted(missing)}, extra={sorted(extra)}")


@dataclass(frozen=True)
class SubmissionToken:
    """Content-derived authorization identity for exactly one submission slot."""

    run_id: str
    runspec_sha256: str
    step_index: int
    step_name: str
    slice_id: str
    attempt: int
    script_sha256: str
    bootstrap_sha256: str
    control_state_sha256: str
    runtime_qualification_sha256: str
    token: str
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        _version(self.format_version, "SubmissionToken")
        _string(self.run_id, "run_id")
        _hex(self.runspec_sha256, "runspec_sha256")
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        _string(self.step_name, "step_name")
        _string(self.slice_id, "slice_id")
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        _hex(self.script_sha256, "script_sha256")
        _hex(self.bootstrap_sha256, "bootstrap_sha256")
        _hex(self.control_state_sha256, "control_state_sha256")
        _hex(self.runtime_qualification_sha256, "runtime_qualification_sha256")
        _hex(self.token, "token")
        if hashlib.sha256(self.canonical_identity_bytes()).hexdigest() != self.token:
            raise ValueError("SubmissionToken token does not match canonical identity")

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        runspec_sha256: str,
        step_index: int,
        step_name: str,
        slice_id: str,
        attempt: int,
        script_sha256: str,
        bootstrap_sha256: str,
        control_state_sha256: str,
        runtime_qualification_sha256: str,
    ) -> SubmissionToken:
        identity = {
            "attempt": attempt,
            "format_version": FORMAT_VERSION,
            "run_id": run_id,
            "runspec_sha256": runspec_sha256,
            "script_sha256": script_sha256,
            "bootstrap_sha256": bootstrap_sha256,
            "control_state_sha256": control_state_sha256,
            "runtime_qualification_sha256": runtime_qualification_sha256,
            "slice_id": slice_id,
            "step_index": step_index,
            "step_name": step_name,
        }
        return cls(
            run_id,
            runspec_sha256,
            step_index,
            step_name,
            slice_id,
            attempt,
            script_sha256,
            bootstrap_sha256,
            control_state_sha256,
            runtime_qualification_sha256,
            hashlib.sha256(_canonical_json(identity)).hexdigest(),
        )

    def canonical_identity_bytes(self) -> bytes:
        value = self.to_mapping()
        del value["token"]
        return _canonical_json(value)

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "run_id": self.run_id,
            "runspec_sha256": self.runspec_sha256,
            "step_index": self.step_index,
            "step_name": self.step_name,
            "slice_id": self.slice_id,
            "attempt": self.attempt,
            "script_sha256": self.script_sha256,
            "bootstrap_sha256": self.bootstrap_sha256,
            "control_state_sha256": self.control_state_sha256,
            "runtime_qualification_sha256": self.runtime_qualification_sha256,
            "token": self.token,
        }


def submission_token_from_mapping(payload: dict[str, object]) -> SubmissionToken:
    expected = {
        "format_version",
        "run_id",
        "runspec_sha256",
        "step_index",
        "step_name",
        "slice_id",
        "attempt",
        "script_sha256",
        "bootstrap_sha256",
        "control_state_sha256",
        "runtime_qualification_sha256",
        "token",
    }
    _fields(payload, expected, "SubmissionToken")
    _version(payload["format_version"], "SubmissionToken")
    if type(payload["step_index"]) is not int or type(payload["attempt"]) is not int:
        raise ValueError("SubmissionToken indices must be integers")
    return SubmissionToken(
        _string(payload["run_id"], "run_id"),
        _hex(payload["runspec_sha256"], "runspec_sha256"),
        int(payload["step_index"]),
        _string(payload["step_name"], "step_name"),
        _string(payload["slice_id"], "slice_id"),
        int(payload["attempt"]),
        _hex(payload["script_sha256"], "script_sha256"),
        _hex(payload["bootstrap_sha256"], "bootstrap_sha256"),
        _hex(payload["control_state_sha256"], "control_state_sha256"),
        _hex(payload["runtime_qualification_sha256"], "runtime_qualification_sha256"),
        _hex(payload["token"], "token"),
    )


@dataclass(frozen=True)
class SubmissionExpectation:
    token: SubmissionToken
    rendered_script_sha256: str
    control_state_sha256: str
    bootstrap_sha256: str
    runtime_qualification_sha256: str
    runtime_ipsae_source_revision: str
    runtime_ipsae_binary_sha256: str
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        _version(self.format_version, "SubmissionExpectation")
        for name in (
            "rendered_script_sha256",
            "control_state_sha256",
            "bootstrap_sha256",
            "runtime_qualification_sha256",
        ):
            _hex(getattr(self, name), name)
        _string(self.runtime_ipsae_source_revision, "runtime_ipsae_source_revision")
        _hex(self.runtime_ipsae_binary_sha256, "runtime_ipsae_binary_sha256")
        if self.rendered_script_sha256 != self.token.script_sha256:
            raise ValueError("SubmissionExpectation rendered script does not match token")
        for name in ("control_state_sha256", "bootstrap_sha256", "runtime_qualification_sha256"):
            if getattr(self, name) != getattr(self.token, name):
                raise ValueError(f"SubmissionExpectation {name} does not match token")

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "token": self.token.to_mapping(),
            "rendered_script_sha256": self.rendered_script_sha256,
            "control_state_sha256": self.control_state_sha256,
            "bootstrap_sha256": self.bootstrap_sha256,
            "runtime_qualification_sha256": self.runtime_qualification_sha256,
            "runtime_ipsae_source_revision": self.runtime_ipsae_source_revision,
            "runtime_ipsae_binary_sha256": self.runtime_ipsae_binary_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_mapping())


def submission_expectation_from_mapping(payload: dict[str, object]) -> SubmissionExpectation:
    expected = {
        "format_version",
        "token",
        "rendered_script_sha256",
        "control_state_sha256",
        "bootstrap_sha256",
        "runtime_qualification_sha256",
        "runtime_ipsae_source_revision",
        "runtime_ipsae_binary_sha256",
    }
    _fields(payload, expected, "SubmissionExpectation")
    _version(payload["format_version"], "SubmissionExpectation")
    if not isinstance(payload["token"], dict):
        raise ValueError("SubmissionExpectation token must be an object")
    return SubmissionExpectation(
        submission_token_from_mapping(payload["token"]),
        _hex(payload["rendered_script_sha256"], "rendered_script_sha256"),
        _hex(payload["control_state_sha256"], "control_state_sha256"),
        _hex(payload["bootstrap_sha256"], "bootstrap_sha256"),
        _hex(payload["runtime_qualification_sha256"], "runtime_qualification_sha256"),
        _string(payload["runtime_ipsae_source_revision"], "runtime_ipsae_source_revision"),
        _hex(payload["runtime_ipsae_binary_sha256"], "runtime_ipsae_binary_sha256"),
    )


def submission_expectation_from_bytes(data: bytes) -> SubmissionExpectation:
    payload = _mapping_from_bytes(data, "SubmissionExpectation")
    record = submission_expectation_from_mapping(payload)
    if record.canonical_bytes() != data:
        raise ValueError("SubmissionExpectation is not canonical compact JSON")
    return record


@dataclass(frozen=True)
class SubmissionResult(SubmissionExpectation):
    job_id: str = ""
    scheduler_status: str = ""
    runtime_observations: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        _string(self.job_id, "job_id")
        _string(self.scheduler_status, "scheduler_status")
        if self.runtime_observations != tuple(sorted(set(self.runtime_observations))):
            raise ValueError("runtime_observations must be canonical sorted unique")
        names = tuple(name for name, _ in self.runtime_observations)
        if len(names) != len(set(names)):
            raise ValueError("runtime observation names must be unique")
        for name, value in self.runtime_observations:
            _string(name, "runtime observation name")
            _string(value, "runtime observation value")

    def to_mapping(self) -> dict[str, object]:
        value = super().to_mapping()
        value.update(
            job_id=self.job_id,
            scheduler_status=self.scheduler_status,
            runtime_observations=[
                {"name": name, "value": observation} for name, observation in self.runtime_observations
            ],
        )
        return value

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_mapping())


def submission_result_from_mapping(payload: dict[str, object]) -> SubmissionResult:
    expected = {
        "format_version",
        "token",
        "rendered_script_sha256",
        "control_state_sha256",
        "bootstrap_sha256",
        "runtime_qualification_sha256",
        "runtime_ipsae_source_revision",
        "runtime_ipsae_binary_sha256",
        "job_id",
        "scheduler_status",
        "runtime_observations",
    }
    _fields(payload, expected, "SubmissionResult")
    base = submission_expectation_from_mapping(
        {
            name: payload[name]
            for name in (
                "format_version",
                "token",
                "rendered_script_sha256",
                "control_state_sha256",
                "bootstrap_sha256",
                "runtime_qualification_sha256",
                "runtime_ipsae_source_revision",
                "runtime_ipsae_binary_sha256",
            )
        }
    )
    raw = payload["runtime_observations"]
    if not isinstance(raw, list) or any(not isinstance(item, dict) or set(item) != {"name", "value"} for item in raw):
        raise ValueError("runtime_observations must be a list of name/value objects")
    observations = tuple((_string(item["name"], "name"), _string(item["value"], "value")) for item in raw)
    return SubmissionResult(
        base.token,
        base.rendered_script_sha256,
        base.control_state_sha256,
        base.bootstrap_sha256,
        base.runtime_qualification_sha256,
        base.runtime_ipsae_source_revision,
        base.runtime_ipsae_binary_sha256,
        FORMAT_VERSION,
        _string(payload["job_id"], "job_id"),
        _string(payload["scheduler_status"], "scheduler_status"),
        observations,
    )


def submission_result_from_bytes(data: bytes) -> SubmissionResult:
    payload = _mapping_from_bytes(data, "SubmissionResult")
    record = submission_result_from_mapping(payload)
    if record.canonical_bytes() != data:
        raise ValueError("SubmissionResult is not canonical compact JSON")
    return record


def validate_submission_result(expectation: SubmissionExpectation, result: SubmissionResult) -> None:
    """Require a result to preserve every immutable expectation binding."""
    for name in (
        "rendered_script_sha256",
        "control_state_sha256",
        "bootstrap_sha256",
        "runtime_qualification_sha256",
        "runtime_ipsae_source_revision",
        "runtime_ipsae_binary_sha256",
        "token",
    ):
        if getattr(result, name) != getattr(expectation, name):
            raise ValueError(f"SubmissionResult {name} does not match expectation")


@dataclass(frozen=True)
class EvidenceIndexEntry:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        pure = PurePosixPath(self.path)
        if (
            not self.path
            or self.path == "."
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != self.path
        ):
            raise ValueError(f"evidence path must be relative canonical POSIX: {self.path!r}")
        _hex(self.sha256, "evidence sha256")

    def to_mapping(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class EvidenceIndex:
    format_version: int
    entries: tuple[EvidenceIndexEntry, ...]

    def __post_init__(self) -> None:
        _version(self.format_version, "EvidenceIndex")
        if len(self.entries) > MAX_INDEX_ENTRIES:
            raise ValueError("EvidenceIndex entry count exceeds bound")
        paths = tuple(entry.path for entry in self.entries)
        if paths != tuple(sorted(set(paths), key=lambda value: value.encode())):
            raise ValueError("EvidenceIndex entries must be canonical sorted unique")

    def to_mapping(self) -> dict[str, object]:
        return {"format_version": self.format_version, "entries": [entry.to_mapping() for entry in self.entries]}


def evidence_index_from_mapping(payload: dict[str, object]) -> EvidenceIndex:
    _fields(payload, {"format_version", "entries"}, "EvidenceIndex")
    version = _version(payload["format_version"], "EvidenceIndex")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("EvidenceIndex entries must be a list")
    if len(raw_entries) > MAX_INDEX_ENTRIES:
        raise ValueError("EvidenceIndex entry count exceeds bound")
    entries: list[EvidenceIndexEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ValueError("EvidenceIndex entry must be an object")
        _fields(raw, {"path", "sha256"}, "EvidenceIndexEntry")
        entries.append(EvidenceIndexEntry(_string(raw["path"], "path"), _hex(raw["sha256"], "sha256")))
    return EvidenceIndex(version, tuple(entries))


def evidence_index_from_bytes(data: bytes) -> EvidenceIndex:
    payload = _mapping_from_bytes(data, "EvidenceIndex")
    record = evidence_index_from_mapping(payload)
    if _canonical_json(record.to_mapping()) != data:
        raise ValueError("EvidenceIndex is not canonical compact JSON")
    return record


@dataclass(frozen=True)
class EvidenceIndexValidation:
    ok: bool
    issues: tuple[str, ...] = ()


def _stable_metadata(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _file_digest_at(parent_descriptor: int, name: str, metadata: os.stat_result, relative: str) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stable_metadata(opened) != _stable_metadata(metadata)
        ):
            raise ValueError(f"evidence file changed during open: {relative}")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        finished = os.fstat(handle.fileno())
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        finished.st_nlink != 1
        or current.st_nlink != 1
        or _stable_metadata(finished) != _stable_metadata(opened)
        or _stable_metadata(current) != _stable_metadata(opened)
    ):
        raise ValueError(f"evidence file changed during read: {relative}")
    return digest.hexdigest()


def _mapping_from_bytes(data: bytes, record_name: str) -> dict[str, object]:
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError(f"{record_name} exceeds size bound")
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {record_name} JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{record_name} must be a JSON object")
    return payload


def build_evidence_index(root: Path) -> EvidenceIndex:
    root_metadata = root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode) or root.is_symlink():
        raise ValueError("evidence root must be a real directory")
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened_root = os.fstat(descriptor)
        if _stable_metadata(opened_root) != _stable_metadata(root_metadata):
            raise ValueError("evidence root changed during open")
        entries = _index_directory(descriptor, Path())
        current_root = root.lstat()
        if _stable_metadata(current_root) != _stable_metadata(opened_root):
            raise ValueError("evidence root changed during traversal")
    finally:
        os.close(descriptor)
    return EvidenceIndex(FORMAT_VERSION, tuple(sorted(entries, key=lambda entry: entry.path.encode())))


def _index_directory(directory_descriptor: int, relative_directory: Path) -> list[EvidenceIndexEntry]:
    entries: list[EvidenceIndexEntry] = []
    for name in sorted(os.listdir(directory_descriptor), key=os.fsencode):
        relative = relative_directory / name
        relative_name = relative.as_posix()
        item = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if stat.S_ISDIR(item.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            try:
                opened = os.fstat(child)
                if _stable_metadata(opened) != _stable_metadata(item):
                    raise ValueError(f"evidence directory changed during open: {relative_name}")
                entries.extend(_index_directory(child, relative))
            finally:
                os.close(child)
            current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            if _stable_metadata(current) != _stable_metadata(item):
                raise ValueError(f"evidence directory changed during traversal: {relative_name}")
            continue
        if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
            raise ValueError(f"evidence must contain only exclusive regular files: {relative_name}")
        entries.append(
            EvidenceIndexEntry(relative_name, _file_digest_at(directory_descriptor, name, item, relative_name))
        )
    return entries


def verify_evidence_index(root: Path, index: EvidenceIndex) -> EvidenceIndexValidation:
    paths = [entry.path for entry in index.entries]
    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    if duplicates:
        return EvidenceIndexValidation(False, tuple(f"duplicate evidence index path: {path}" for path in duplicates))
    actual = build_evidence_index(root)
    expected_by_path = {entry.path: entry.sha256 for entry in index.entries}
    actual_by_path = {entry.path: entry.sha256 for entry in actual.entries}
    issues = [f"missing evidence path: {path}" for path in sorted(set(expected_by_path) - set(actual_by_path))]
    issues.extend(f"unexpected evidence path: {path}" for path in sorted(set(actual_by_path) - set(expected_by_path)))
    issues.extend(
        f"evidence digest mismatch: {path}"
        for path in sorted(set(actual_by_path) & set(expected_by_path))
        if actual_by_path[path] != expected_by_path[path]
    )
    return EvidenceIndexValidation(not issues, tuple(issues))


def stable_read_evidence_file(root: Path, relative_path: Path, *, max_bytes: int) -> bytes:
    """Read a bounded evidence descendant without following or racing path aliases."""
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError("evidence path must be a normalized relative path")
    root_before = root.lstat()
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    directory_descriptors = [descriptor]
    try:
        root_opened = os.fstat(descriptor)
        if _stable_metadata(root_opened) != _stable_metadata(root_before):
            raise ValueError("evidence root changed during open")
        opened_directories: list[tuple[int, str, os.stat_result]] = []
        for part in relative_path.parts[:-1]:
            before = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            child = os.open(
                part,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            if _stable_metadata(os.fstat(child)) != _stable_metadata(before):
                os.close(child)
                raise ValueError(f"evidence directory changed during open: {part}")
            opened_directories.append((descriptor, part, before))
            descriptor = child
            directory_descriptors.append(child)
        name = relative_path.parts[-1]
        before_file = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(before_file.st_mode) or before_file.st_nlink != 1:
            raise ValueError("evidence file must be an exclusive regular file")
        file_descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor)
        try:
            opened_file = os.fstat(file_descriptor)
            if _stable_metadata(opened_file) != _stable_metadata(before_file):
                raise ValueError("evidence file changed during open")
            data = os.read(file_descriptor, max_bytes + 1)
            finished_file = os.fstat(file_descriptor)
        finally:
            os.close(file_descriptor)
        current_file = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if len(data) > max_bytes:
            raise ValueError("evidence file exceeds size bound")
        if (
            finished_file.st_nlink != 1
            or current_file.st_nlink != 1
            or _stable_metadata(finished_file) != _stable_metadata(opened_file)
            or _stable_metadata(current_file) != _stable_metadata(opened_file)
        ):
            raise ValueError("evidence file changed during read")
        for parent_descriptor, part, before in reversed(opened_directories):
            current = os.stat(part, dir_fd=parent_descriptor, follow_symlinks=False)
            if _stable_metadata(current) != _stable_metadata(before):
                raise ValueError(f"evidence directory changed during read: {part}")
        if _stable_metadata(root.lstat()) != _stable_metadata(root_opened):
            raise ValueError("evidence root changed during read")
        return data
    finally:
        for directory_descriptor in reversed(directory_descriptors):
            os.close(directory_descriptor)


__all__ = [
    "EvidenceIndex",
    "EvidenceIndexEntry",
    "EvidenceIndexValidation",
    "SubmissionExpectation",
    "SubmissionResult",
    "SubmissionToken",
    "build_evidence_index",
    "evidence_index_from_bytes",
    "evidence_index_from_mapping",
    "stable_read_evidence_file",
    "submission_expectation_from_bytes",
    "submission_expectation_from_mapping",
    "submission_result_from_bytes",
    "submission_result_from_mapping",
    "submission_token_from_mapping",
    "validate_submission_result",
    "verify_evidence_index",
]
