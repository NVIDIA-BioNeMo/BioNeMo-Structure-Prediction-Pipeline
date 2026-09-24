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

"""Target-side preparation and verified copy for Attempt carry-forward."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from bspp.orchestration.contract.phase import PhaseRunSpec
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardAdoptedContent,
    AttemptCarryForwardAdoptionEvidence,
    AttemptCarryForwardRecord,
    attempt_carry_forward_record_from_mapping,
)
from bspp.orchestration.runtime.preprocessing.content_validation import (
    validate_preprocessing_a3m_bytes,
)


def load_attempt_carry_forward_record(path: Path) -> AttemptCarryForwardRecord:
    """Strict-load one canonical JSON carry record from the staged file mount."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("staged carry-forward record must be a regular non-symlink file")
    document = path.read_bytes()
    try:
        payload = json.loads(document)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("staged carry-forward record must be UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("staged carry-forward record must be one JSON mapping")
    canonical = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    if canonical != document:
        raise ValueError("staged carry-forward record must use canonical JSON bytes")
    return attempt_carry_forward_record_from_mapping(payload)


def prepare_carried_execution(
    runspec: PhaseRunSpec,
    record: AttemptCarryForwardRecord,
    *,
    phase_submission_id: str,
) -> tuple[str, ...]:
    """Verify the fresh sentinel-bound workspace and stage complement/full FASTA bytes."""
    _validate_record_identity(runspec, record)
    workspace = record.workspace
    sentinel_relative = PurePosixPath(workspace.identity_sentinel_path).relative_to(
        PurePosixPath(workspace.workspace_root)
    )
    sentinel_path = Path(workspace.private_workspace_mount_path) / Path(*sentinel_relative.parts)
    expected_sentinel = {
        "schema_version": 1,
        "phase_run_id": record.phase_run_id,
        "attempt_id": record.target_attempt_id,
        "action_id": workspace.target_action_id,
        "attempt_carry_forward_id": record.attempt_carry_forward_id,
        "attempt_carry_forward_digest": record.digest,
        "phase_submission_id": phase_submission_id,
        "workspace_mapping_digest": workspace.digest,
    }
    _require_canonical_sentinel(sentinel_path, expected_sentinel)
    for root in workspace.roots:
        directory = Path(root.logical_root)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"carry workspace root must be a real precreated directory: {directory}")
        if any(directory.iterdir()):
            raise ValueError(f"carry workspace root must be empty before Runtime preparation: {directory}")

    action = runspec.payload.actions[0]
    records = {item.source_ordinal: item for item in runspec.payload.work_plan.input.records}
    remaining = tuple(records[ordinal] for ordinal in record.remaining_record_ordinals)
    remaining_bytes = _fasta_bytes(remaining)
    if hashlib.sha256(remaining_bytes).hexdigest() != record.remaining_search_input_sha256:
        raise ValueError("derived remaining-record FASTA does not match carry authority")
    full_ordinals = runspec.payload.work_plan.chunks[0].record_ordinals
    full_bytes = _fasta_bytes(tuple(records[ordinal] for ordinal in full_ordinals))
    search_path = Path(action.payload.search_argv[3])
    split_path = Path(action.payload.package.completed_input_source_path)
    _prepare_bound_parent(record, search_path)
    _prepare_bound_parent(record, split_path)
    _write_exclusive(search_path, remaining_bytes)
    _write_exclusive(split_path, full_bytes)
    if str(search_path) != _logical_from_physical(record, workspace.search_input_physical_path):
        raise ValueError("search-input workspace projection does not match frozen RunSpec path")
    if str(split_path) != _logical_from_physical(record, workspace.split_input_physical_path):
        raise ValueError("split-input workspace projection does not match frozen RunSpec path")
    selected = {item.member_name for item in record.content}
    return tuple(item.member_name for item in action.payload.expected_a3ms if item.member_name not in selected)


def adopt_carried_a3ms(
    runspec: PhaseRunSpec,
    record: AttemptCarryForwardRecord,
    *,
    phase_submission_id: str,
    adopted_at: str,
) -> AttemptCarryForwardAdoptionEvidence:
    """Verify exact private source files and copy them without replacement."""
    _validate_record_identity(runspec, record)
    adopted: list[AttemptCarryForwardAdoptedContent] = []
    for item in record.content:
        source = Path(item.source_private_mount_path)
        target = Path(item.target_declared_path)
        data = _read_exact_regular_no_follow(source, expected_size=item.size_bytes)
        observed_sha256 = hashlib.sha256(data).hexdigest()
        if observed_sha256 != item.sha256:
            raise ValueError(f"carried source SHA-256 mismatch: {item.member_name}")
        validate_preprocessing_a3m_bytes(data, label=item.member_name)
        if str(target) != _logical_from_physical(record, item.target_physical_path):
            raise ValueError("carried target physical/logical mapping does not match workspace authority")
        _write_exclusive(target, data)
        if _read_exact_regular_no_follow(target, expected_size=item.size_bytes) != data:
            raise ValueError(f"published carried target bytes changed: {item.member_name}")
        adopted.append(
            AttemptCarryForwardAdoptedContent(
                member_name=item.member_name,
                source_path=item.source_private_mount_path,
                target_path=item.target_declared_path,
                size_bytes=item.size_bytes,
                sha256=item.sha256,
            )
        )
    return AttemptCarryForwardAdoptionEvidence(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        phase_submission_id=phase_submission_id,
        action_id=runspec.payload.actions[0].action_id,
        attempt_carry_forward_id=record.attempt_carry_forward_id,
        attempt_carry_forward_digest=record.digest,
        remaining_search_input_sha256=record.remaining_search_input_sha256,
        content=tuple(adopted),
        adopted_at=adopted_at,
    )


def reconcile_carry_forward_adoption(
    runspec: PhaseRunSpec,
    evidence: AttemptCarryForwardAdoptionEvidence | None,
    record: AttemptCarryForwardRecord | None,
) -> None:
    """Cross-check passing adoption and current target bytes against carry authority."""
    if record is None:
        if evidence is not None:
            raise ValueError("no-carry RunSpec cannot contain adoption evidence")
        return
    if evidence is None:
        raise ValueError("carried successful action evidence requires passing adoption evidence")
    if (
        evidence.phase_run_id != runspec.phase_run_id
        or evidence.attempt_id != runspec.attempt_id
        or evidence.phase_runspec_digest != runspec.digest
        or evidence.action_id != runspec.payload.actions[0].action_id
        or evidence.attempt_carry_forward_id != record.attempt_carry_forward_id
        or evidence.attempt_carry_forward_digest != record.digest
        or evidence.remaining_search_input_sha256 != record.remaining_search_input_sha256
    ):
        raise ValueError("carry-forward adoption evidence identity does not match RunSpec record")
    expected = tuple(
        (item.member_name, item.source_private_mount_path, item.target_declared_path, item.size_bytes, item.sha256)
        for item in record.content
    )
    observed = tuple(
        (item.member_name, item.source_path, item.target_path, item.size_bytes, item.sha256)
        for item in evidence.content
    )
    if observed != expected:
        raise ValueError("carry-forward adoption content does not match ordered record content")
    for item in record.content:
        data = _read_exact_regular_no_follow(Path(item.target_declared_path), expected_size=item.size_bytes)
        if hashlib.sha256(data).hexdigest() != item.sha256:
            raise ValueError(f"carried target content changed after adoption: {item.member_name}")


def _validate_record_identity(runspec: PhaseRunSpec, record: AttemptCarryForwardRecord) -> None:
    reference = runspec.carry_forward
    action = runspec.payload.actions[0]
    if (
        reference is None
        or reference.attempt_carry_forward_id != record.attempt_carry_forward_id
        or reference.digest != record.digest
        or record.phase_run_id != runspec.phase_run_id
        or record.target_attempt_id != runspec.attempt_id
        or record.workspace.target_action_id != action.action_id
    ):
        raise ValueError("carry-forward record does not match the executing RunSpec")
    expected_ordinals = tuple(item.source_ordinal for item in action.payload.expected_a3ms)
    carried_ordinals = tuple(item.source_ordinal for item in record.content)
    complement = tuple(ordinal for ordinal in expected_ordinals if ordinal not in set(carried_ordinals))
    if record.remaining_record_ordinals != complement:
        raise ValueError("carry-forward remaining ordinals are not the exact action complement")


def _require_canonical_sentinel(path: Path, expected: Mapping[str, object]) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("carry workspace identity sentinel must be a regular non-symlink file")
    document = path.read_bytes()
    canonical = (json.dumps(expected, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    if document != canonical:
        raise ValueError("carry workspace identity sentinel does not match submission authority")


def _logical_from_physical(record: AttemptCarryForwardRecord, physical: str) -> str:
    physical_path = PurePosixPath(physical)
    matches = tuple(
        item
        for item in record.workspace.roots
        if physical_path == PurePosixPath(item.physical_root)
        or PurePosixPath(item.physical_root) in physical_path.parents
    )
    if len(matches) != 1:
        raise ValueError("carry physical path must resolve through exactly one workspace root")
    root = matches[0]
    relative = physical_path.relative_to(PurePosixPath(root.physical_root))
    return str(PurePosixPath(root.logical_root) / relative)


def _prepare_bound_parent(record: AttemptCarryForwardRecord, logical_path: Path) -> None:
    candidate = PurePosixPath(str(logical_path))
    matches = tuple(
        item
        for item in record.workspace.roots
        if candidate == PurePosixPath(item.logical_root) or PurePosixPath(item.logical_root) in candidate.parents
    )
    if len(matches) != 1:
        raise ValueError("carried input path must resolve through exactly one workspace root")
    root = Path(matches[0].logical_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"carry workspace root must remain a real directory: {root}")
    current = root
    relative_parent = logical_path.parent.relative_to(root)
    for part in relative_parent.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"carried input parent cannot follow a symlink: {current}")
        if current.exists():
            if not current.is_dir():
                raise ValueError(f"carried input parent must be a directory: {current}")
        else:
            current.mkdir()


def _fasta_bytes(records: Sequence[object]) -> bytes:
    lines: list[str] = []
    for record in records:
        identity = getattr(record, "identity", None)
        sequence = getattr(record, "sequence", None)
        if not isinstance(identity, str) or not isinstance(sequence, str):
            raise TypeError("carry FASTA derivation requires strict work-plan records")
        lines.extend((f">{identity}", sequence))
    return ("\n".join(lines) + "\n").encode()


def _read_exact_regular_no_follow(path: Path, *, expected_size: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"carried path is not a regular file: {path}")
        if info.st_size != expected_size:
            raise ValueError(f"carried path size mismatch: {path}")
        chunks: list[bytes] = []
        remaining = expected_size + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(data) != expected_size:
        raise ValueError(f"carried path changed during bounded read: {path}")
    return data


def _write_exclusive(path: Path, data: bytes) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(f"carried target parent must be a precreated real directory: {path.parent}")
    temporary = path.with_name(f".{path.name}.carry-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            _write_all(handle, data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_all(handle: BinaryIO, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = handle.write(remaining)
        if written is None or written <= 0:
            raise OSError("carried target temporary write made no progress")
        remaining = remaining[written:]


__all__ = [
    "adopt_carried_a3ms",
    "load_attempt_carry_forward_record",
    "prepare_carried_execution",
    "reconcile_carry_forward_adoption",
]
