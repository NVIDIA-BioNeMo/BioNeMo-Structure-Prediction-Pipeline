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

"""Strict, generic acceptance evidence and publication authorization contracts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from bspp.orchestration.contract.execution_provenance_verify import verify_processing_provenance_boundary
from bspp.orchestration.contract.submission_evidence import (
    MAX_RECORD_BYTES,
    evidence_index_from_bytes,
    stable_read_evidence_file,
)

CANDIDATE_INVENTORY_VERSION: Final = 1
LOCAL_INTEGRITY_VERSION: Final = 1
SEMANTIC_VALIDATION_VERSION: Final = 1
TERMINAL_FAILURE_VERSION: Final = 1
NO_UPLOAD_VERSION: Final = 1
PROVENANCE_IDENTITY_VERSION: Final = 1
ACCEPTANCE_EVIDENCE_VERSION: Final = 1
PUBLICATION_APPROVAL_VERSION: Final = 1
_MAX_APPROVAL_BYTES: Final = 1024 * 1024


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _fields(value: dict[str, object], expected: set[str], name: str) -> None:
    missing, extra = expected - set(value), set(value) - expected
    if missing or extra:
        raise ValueError(f"Invalid {name} fields; missing={sorted(missing)}, extra={sorted(extra)}")


def _version(value: object, expected: int, name: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"Unsupported {name} format_version {value}; supported versions: {expected}")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _hex(value: object, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return text


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be a list")
    result = tuple(_text(item, name) for item in value)
    if result != tuple(sorted(set(result), key=lambda item: item.encode())):
        raise ValueError(f"{name} must be canonical sorted unique")
    return result


def _integers(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{name} must be a list of non-negative integers")
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be canonical sorted unique")
    return result


def _relative_path(value: object, name: str) -> str:
    text = _text(value, name)
    path = PurePosixPath(text)
    if path.is_absolute() or path.as_posix() != text or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    return text


@dataclass(frozen=True)
class CandidateInventoryReport:
    format_version: int
    expected_candidates: tuple[str, ...]
    observed_candidates: tuple[str, ...]

    def __post_init__(self) -> None:
        _version(self.format_version, CANDIDATE_INVENTORY_VERSION, type(self).__name__)
        _strings(self.expected_candidates, "expected_candidates")
        _strings(self.observed_candidates, "observed_candidates")

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "expected_candidates": list(self.expected_candidates),
            "observed_candidates": list(self.observed_candidates),
        }


@dataclass(frozen=True)
class LocalIntegrityReport:
    format_version: int
    expected_members: tuple[str, ...]
    observed_members: tuple[str, ...]
    digest_mismatches: tuple[str, ...]

    def __post_init__(self) -> None:
        _version(self.format_version, LOCAL_INTEGRITY_VERSION, type(self).__name__)
        for name in ("expected_members", "observed_members", "digest_mismatches"):
            _strings(getattr(self, name), name)

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "expected_members": list(self.expected_members),
            "observed_members": list(self.observed_members),
            "digest_mismatches": list(self.digest_mismatches),
        }


@dataclass(frozen=True)
class SemanticValidationReport:
    format_version: int
    expected_count: int
    minimum_pass_count: int
    passed_count: int
    mismatch_count: int

    def __post_init__(self) -> None:
        _version(self.format_version, SEMANTIC_VALIDATION_VERSION, type(self).__name__)
        for name in ("expected_count", "minimum_pass_count", "passed_count", "mismatch_count"):
            _integer(getattr(self, name), name)

    def to_mapping(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in ("format_version", "expected_count", "minimum_pass_count", "passed_count", "mismatch_count")
        }


@dataclass(frozen=True)
class TerminalFailureReport:
    format_version: int
    expected_execution_count: int
    completed_candidates: tuple[str, ...]
    failed_candidates: tuple[str, ...]
    missing_execution_results: tuple[str, ...]

    def __post_init__(self) -> None:
        _version(self.format_version, TERMINAL_FAILURE_VERSION, type(self).__name__)
        _integer(self.expected_execution_count, "expected_execution_count")
        for name in ("completed_candidates", "failed_candidates", "missing_execution_results"):
            _strings(getattr(self, name), name)

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "expected_execution_count": self.expected_execution_count,
            "completed_candidates": list(self.completed_candidates),
            "failed_candidates": list(self.failed_candidates),
            "missing_execution_results": list(self.missing_execution_results),
        }


@dataclass(frozen=True)
class NoUploadReport:
    format_version: int
    external_upload_count: int

    def __post_init__(self) -> None:
        _version(self.format_version, NO_UPLOAD_VERSION, type(self).__name__)
        _integer(self.external_upload_count, "external_upload_count")

    def to_mapping(self) -> dict[str, object]:
        return {"format_version": self.format_version, "external_upload_count": self.external_upload_count}


@dataclass(frozen=True)
class ProvenanceIdentityReport:
    format_version: int
    index_path: str
    index_sha256: str
    processing_step_indices: tuple[int, ...]
    runspec_sha256: str

    def __post_init__(self) -> None:
        _version(self.format_version, PROVENANCE_IDENTITY_VERSION, type(self).__name__)
        _relative_path(self.index_path, "index_path")
        _hex(self.index_sha256, "index_sha256")
        _integers(self.processing_step_indices, "processing_step_indices")
        _hex(self.runspec_sha256, "runspec_sha256")

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "index_path": self.index_path,
            "index_sha256": self.index_sha256,
            "processing_step_indices": list(self.processing_step_indices),
            "runspec_sha256": self.runspec_sha256,
        }


@dataclass(frozen=True)
class AcceptanceEvidence:
    candidate_inventory: CandidateInventoryReport
    local_integrity: LocalIntegrityReport
    semantic_validation: SemanticValidationReport
    terminal_failures: TerminalFailureReport
    no_upload: NoUploadReport
    provenance_identity: ProvenanceIdentityReport
    format_version: int = ACCEPTANCE_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        _version(self.format_version, ACCEPTANCE_EVIDENCE_VERSION, type(self).__name__)

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "candidate_inventory": self.candidate_inventory.to_mapping(),
            "local_integrity": self.local_integrity.to_mapping(),
            "semantic_validation": self.semantic_validation.to_mapping(),
            "terminal_failures": self.terminal_failures.to_mapping(),
            "no_upload": self.no_upload.to_mapping(),
            "provenance_identity": self.provenance_identity.to_mapping(),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.to_mapping())).hexdigest()

    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_mapping())


def _report(payload: object, fields: set[str], name: str, version: int) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be an object")
    _fields(payload, fields | {"format_version"}, name)
    _version(payload["format_version"], version, name)
    return payload


def acceptance_evidence_from_mapping(payload: dict[str, object]) -> AcceptanceEvidence:
    names = {
        "format_version",
        "candidate_inventory",
        "local_integrity",
        "semantic_validation",
        "terminal_failures",
        "no_upload",
        "provenance_identity",
    }
    _fields(payload, names, "AcceptanceEvidence")
    _version(payload["format_version"], ACCEPTANCE_EVIDENCE_VERSION, "AcceptanceEvidence")
    candidate = _report(
        payload["candidate_inventory"], {"expected_candidates", "observed_candidates"}, "CandidateInventoryReport", 1
    )
    integrity = _report(
        payload["local_integrity"],
        {"expected_members", "observed_members", "digest_mismatches"},
        "LocalIntegrityReport",
        1,
    )
    semantic = _report(
        payload["semantic_validation"],
        {"expected_count", "minimum_pass_count", "passed_count", "mismatch_count"},
        "SemanticValidationReport",
        1,
    )
    terminal = _report(
        payload["terminal_failures"],
        {"expected_execution_count", "completed_candidates", "failed_candidates", "missing_execution_results"},
        "TerminalFailureReport",
        1,
    )
    upload = _report(payload["no_upload"], {"external_upload_count"}, "NoUploadReport", 1)
    provenance = _report(
        payload["provenance_identity"],
        {"index_path", "index_sha256", "processing_step_indices", "runspec_sha256"},
        "ProvenanceIdentityReport",
        1,
    )
    return AcceptanceEvidence(
        CandidateInventoryReport(
            1,
            _strings(candidate["expected_candidates"], "expected_candidates"),
            _strings(candidate["observed_candidates"], "observed_candidates"),
        ),
        LocalIntegrityReport(
            1,
            _strings(integrity["expected_members"], "expected_members"),
            _strings(integrity["observed_members"], "observed_members"),
            _strings(integrity["digest_mismatches"], "digest_mismatches"),
        ),
        SemanticValidationReport(
            1,
            *(
                _integer(semantic[name], name)
                for name in ("expected_count", "minimum_pass_count", "passed_count", "mismatch_count")
            ),
        ),
        TerminalFailureReport(
            1,
            _integer(terminal["expected_execution_count"], "expected_execution_count"),
            _strings(terminal["completed_candidates"], "completed_candidates"),
            _strings(terminal["failed_candidates"], "failed_candidates"),
            _strings(terminal["missing_execution_results"], "missing_execution_results"),
        ),
        NoUploadReport(1, _integer(upload["external_upload_count"], "external_upload_count")),
        ProvenanceIdentityReport(
            1,
            _relative_path(provenance["index_path"], "index_path"),
            _hex(provenance["index_sha256"], "index_sha256"),
            _integers(provenance["processing_step_indices"], "processing_step_indices"),
            _hex(provenance["runspec_sha256"], "runspec_sha256"),
        ),
    )


def acceptance_evidence_from_bytes(data: bytes) -> AcceptanceEvidence:
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError("AcceptanceEvidence exceeds size bound")
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid AcceptanceEvidence JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("AcceptanceEvidence must be an object")
    evidence = acceptance_evidence_from_mapping(payload)
    if evidence.canonical_bytes() != data:
        raise ValueError("AcceptanceEvidence is not canonical compact JSON")
    return evidence


@dataclass(frozen=True)
class AcceptanceValidation:
    ok: bool
    issues: tuple[str, ...]
    acceptance_sha256: str


def validate_acceptance(
    evidence: AcceptanceEvidence,
    *,
    evidence_root: Path,
    expected_runspec_sha256: str,
) -> AcceptanceValidation:
    _hex(expected_runspec_sha256, "expected_runspec_sha256")
    issues: list[str] = []
    inventory = evidence.candidate_inventory
    if inventory.expected_candidates != inventory.observed_candidates:
        issues.append("candidate inventory mismatch")
    integrity = evidence.local_integrity
    for member in sorted(set(integrity.expected_members) - set(integrity.observed_members)):
        issues.append(f"missing archive member: {member}")
    for member in sorted(set(integrity.observed_members) - set(integrity.expected_members)):
        issues.append(f"unexpected archive member: {member}")
    if integrity.digest_mismatches:
        issues.append("local package digest mismatches")
    semantic = evidence.semantic_validation
    if semantic.expected_count != len(inventory.expected_candidates):
        issues.append("semantic expected count differs from candidate inventory")
    if semantic.minimum_pass_count > semantic.expected_count:
        issues.append("semantic minimum exceeds expected count")
    if semantic.passed_count + semantic.mismatch_count != semantic.expected_count:
        issues.append("semantic evaluated count mismatch")
    if semantic.mismatch_count:
        issues.append("semantic mismatches")
    if semantic.passed_count < semantic.minimum_pass_count:
        issues.append("semantic pass count below minimum")
    terminal = evidence.terminal_failures
    terminal_sets = (
        set(terminal.completed_candidates),
        set(terminal.failed_candidates),
        set(terminal.missing_execution_results),
    )
    if any(terminal_sets[left].intersection(terminal_sets[right]) for left, right in ((0, 1), (0, 2), (1, 2))):
        issues.append("terminal candidate sets overlap")
    if set(terminal.completed_candidates) != set(inventory.observed_candidates):
        issues.append("completed candidates differ from observed inventory")
    if terminal.failed_candidates:
        issues.append("terminal failures")
    if terminal.missing_execution_results:
        issues.append("missing execution results")
    if len(terminal.completed_candidates) != terminal.expected_execution_count:
        issues.append("terminal execution count mismatch")
    if evidence.no_upload.external_upload_count != 0:
        issues.append("external uploads must be zero before approval")
    if evidence.provenance_identity.runspec_sha256 != expected_runspec_sha256:
        issues.append("provenance RunSpec identity mismatch")
    try:
        index_bytes = stable_read_evidence_file(
            evidence_root,
            Path(evidence.provenance_identity.index_path),
            max_bytes=MAX_RECORD_BYTES,
        )
        if hashlib.sha256(index_bytes).hexdigest() != evidence.provenance_identity.index_sha256:
            raise ValueError("processing provenance index digest mismatch")
        index = evidence_index_from_bytes(index_bytes)
        index_result = verify_processing_provenance_boundary(
            evidence_root,
            index,
            processing_step_indices=evidence.provenance_identity.processing_step_indices,
        )
    except (OSError, ValueError) as error:
        issues.append(f"provenance tree verification failed: {error}")
    else:
        if not index_result.ok:
            issues.append("provenance tree verification failed: " + "; ".join(index_result.issues))
    return AcceptanceValidation(not issues, tuple(issues), evidence.sha256)


@dataclass(frozen=True)
class PublicationApproval:
    acceptance_sha256: str
    destination: str
    approval_sha256: str
    format_version: int = PUBLICATION_APPROVAL_VERSION

    def __post_init__(self) -> None:
        _version(self.format_version, PUBLICATION_APPROVAL_VERSION, type(self).__name__)
        _hex(self.acceptance_sha256, "acceptance_sha256")
        _text(self.destination, "destination")
        _hex(self.approval_sha256, "approval_sha256")
        if hashlib.sha256(self.identity_bytes()).hexdigest() != self.approval_sha256:
            raise ValueError("PublicationApproval altered: approval digest does not match content")

    @classmethod
    def create(cls, *, acceptance_sha256: str, destination: str) -> PublicationApproval:
        identity = {"format_version": 1, "acceptance_sha256": acceptance_sha256, "destination": destination}
        return cls(acceptance_sha256, destination, hashlib.sha256(_canonical(identity)).hexdigest())

    @classmethod
    def create_from_acceptance(
        cls,
        acceptance: AcceptanceValidation,
        *,
        destination: str,
    ) -> PublicationApproval:
        if not acceptance.ok:
            raise ValueError("PublicationApproval requires successful acceptance")
        return cls.create(acceptance_sha256=acceptance.acceptance_sha256, destination=destination)

    def identity_bytes(self) -> bytes:
        return _canonical(
            {
                "format_version": self.format_version,
                "acceptance_sha256": self.acceptance_sha256,
                "destination": self.destination,
            }
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "acceptance_sha256": self.acceptance_sha256,
            "destination": self.destination,
            "approval_sha256": self.approval_sha256,
        }


def publication_approval_from_mapping(payload: dict[str, object]) -> PublicationApproval:
    _fields(payload, {"format_version", "acceptance_sha256", "destination", "approval_sha256"}, "PublicationApproval")
    return PublicationApproval(
        _hex(payload["acceptance_sha256"], "acceptance_sha256"),
        _text(payload["destination"], "destination"),
        _hex(payload["approval_sha256"], "approval_sha256"),
        _version(payload["format_version"], 1, "PublicationApproval"),
    )


class PublicationApprovalStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(os.path.abspath(path))
        self.consumed_path = self.path.with_suffix(self.path.suffix + ".consumed")

    def create(self, approval: PublicationApproval) -> None:
        _write_exclusive_record(self.path, _canonical(approval.to_mapping()))

    def require(self, *, acceptance_sha256: str, destination: str) -> PublicationApproval:
        if os.path.lexists(self.consumed_path):
            raise ValueError("PublicationApproval consumed")
        if not os.path.lexists(self.path):
            raise ValueError("PublicationApproval absent")
        try:
            raw_bytes = _read_stable_record(self.path, maximum_bytes=_MAX_APPROVAL_BYTES)
            raw = json.loads(raw_bytes)
            if not isinstance(raw, dict):
                raise ValueError("record is not an object")
            approval = publication_approval_from_mapping(raw)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"PublicationApproval altered: {error}") from error
        if approval.acceptance_sha256 != acceptance_sha256 or approval.destination != destination:
            raise ValueError("PublicationApproval stale for acceptance or destination")
        return approval

    def consume(self, *, acceptance_sha256: str, destination: str) -> PublicationApproval:
        approval = self.require(acceptance_sha256=acceptance_sha256, destination=destination)
        _write_exclusive_record(
            self.consumed_path,
            _canonical({"approval_sha256": approval.approval_sha256}),
        )
        return approval


def _open_record_parent(path: Path) -> tuple[int, os.stat_result]:
    parent = path.parent
    metadata = parent.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise ValueError("PublicationApproval parent must be a canonical real directory")
    descriptor = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(descriptor)
    if _directory_identity(opened) != _directory_identity(metadata):
        os.close(descriptor)
        raise ValueError("PublicationApproval parent changed during open")
    return descriptor, opened


def _write_exclusive_record(path: Path, data: bytes) -> None:
    parent_descriptor, parent_metadata = _open_record_parent(path)
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("PublicationApproval record write stalled")
                remaining = remaining[written:]
            os.fsync(descriptor)
            written_metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or _stable_metadata(current) != _stable_metadata(written_metadata)
        ):
            raise ValueError("PublicationApproval record changed during publication")
        os.fsync(parent_descriptor)
        if _directory_identity(path.parent.lstat()) != _directory_identity(parent_metadata):
            raise ValueError("PublicationApproval parent changed during publication")
    finally:
        os.close(parent_descriptor)


def _read_stable_record(path: Path, *, maximum_bytes: int) -> bytes:
    parent_descriptor, parent_metadata = _open_record_parent(path)
    try:
        before = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum_bytes:
            raise ValueError("record is not a bounded exclusive regular file")
        descriptor = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_descriptor)
        try:
            opened = os.fstat(descriptor)
            if _stable_metadata(opened) != _stable_metadata(before):
                raise ValueError("record changed during open")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(64 * 1024, maximum_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > maximum_bytes:
                    raise ValueError("record exceeds size bound")
            finished = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            total != before.st_size
            or _stable_metadata(finished) != _stable_metadata(opened)
            or _stable_metadata(current) != _stable_metadata(opened)
            or _directory_identity(path.parent.lstat()) != _directory_identity(parent_metadata)
        ):
            raise ValueError("record changed during read")
        return b"".join(chunks)
    finally:
        os.close(parent_descriptor)


def _stable_metadata(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def validate_processing_upload_policy(*, self_upload: bool, workflow_steps: tuple[str, ...]) -> None:
    upload_steps = {"upload-s3", "upload-gcs", "upload-gcs-direct"}
    if self_upload or upload_steps.intersection(workflow_steps):
        raise ValueError("processing RunSpec cannot upload before publication approval")


__all__ = [
    "AcceptanceEvidence",
    "AcceptanceValidation",
    "CandidateInventoryReport",
    "LocalIntegrityReport",
    "NoUploadReport",
    "ProvenanceIdentityReport",
    "PublicationApproval",
    "PublicationApprovalStore",
    "SemanticValidationReport",
    "TerminalFailureReport",
    "acceptance_evidence_from_bytes",
    "acceptance_evidence_from_mapping",
    "publication_approval_from_mapping",
    "validate_acceptance",
    "validate_processing_upload_policy",
]
