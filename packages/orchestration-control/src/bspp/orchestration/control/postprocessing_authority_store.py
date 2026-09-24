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

"""Crash-safe local storage primitives for postprocessing Phase authority."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock
from pathlib import Path
from typing import IO, cast

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_event import (
    PostprocessingEventPayload,
    PostprocessingEventType,
    PostprocessingPhaseEvent,
)
from bspp.orchestration.control.postprocessing_phase_types import PostprocessingAuthority


@contextmanager
def postprocessing_operation_lock(authority_root: Path, phase_run_id: str) -> Iterator[None]:
    authority_root.mkdir(parents=True, exist_ok=True)
    lock_path = authority_root / f".{phase_run_id}.operation.lock"
    with lock_path.open("a+b") as handle:
        _lock(handle)
        try:
            yield
        finally:
            flock(handle.fileno(), LOCK_UN)


@contextmanager
def postprocessing_phase_coordinator_lock(authority_root: Path, phase_run_id: str) -> Iterator[None]:
    """Exclude all long-lived coordinators for one Phase without becoming authority."""
    authority_root.mkdir(parents=True, exist_ok=True)
    lock_path = authority_root / f".{phase_run_id}.coordinator.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("postprocessing coordinator lock must be one regular file")
        flock(descriptor, LOCK_EX | LOCK_NB)
        try:
            yield
        finally:
            flock(descriptor, LOCK_UN)
    finally:
        os.close(descriptor)


@contextmanager
def postprocessing_authority_lock(authority_root: Path) -> Iterator[None]:
    authority_root.mkdir(parents=True, exist_ok=True)
    with (authority_root / ".phase-authority.lock").open("a+b") as handle:
        _lock(handle)
        try:
            yield
        finally:
            flock(handle.fileno(), LOCK_UN)


def read_canonical_json(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"stored authority JSON must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise TypeError(f"stored authority JSON must be a mapping: {path}")
    if canonical_json_bytes(payload) != raw:
        raise ValueError(f"stored authority JSON is not canonical: {path}")
    return payload


def append_event(
    authority: PostprocessingAuthority,
    *,
    event_type: str,
    occurred_at: str,
    payload: PostprocessingEventPayload,
) -> PostprocessingAuthority:
    if authority.sealed:
        raise ValueError(f"Phase Run is already sealed: {authority.phase_run_id}")
    if not authority.current_attempt_projection_complete:
        raise ValueError("postprocessing event append requires complete current Attempt projections")
    sequence = len(authority.events) + 1
    event = PostprocessingPhaseEvent(
        sequence=sequence,
        event_type=cast("PostprocessingEventType", event_type),
        phase_run_id=authority.phase_run_id,
        attempt_id=authority.attempt_id,
        occurred_at=occurred_at,
        payload=payload,
    )
    path = authority.authority_path / "events" / f"{sequence:06d}-{event_type}.json"
    with postprocessing_authority_lock(authority.authority_path.parent):
        write_no_replace(path, canonical_json_bytes(event.to_mapping()))
    from bspp.orchestration.control.postprocessing_authority_v2 import (
        validate_postprocessing_authority,
    )

    return validate_postprocessing_authority(authority.authority_path.parent, authority.phase_run_id)


def write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_no_replace(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    write(temporary, payload)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    fsync_directory(path.parent)


def fsync_tree_directories(root: Path) -> None:
    for path, directories, _files in os.walk(root, topdown=False):
        for directory in directories:
            fsync_directory(Path(path) / directory)
        fsync_directory(Path(path))


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def mapping_digest(payload: Mapping[str, object]) -> str:
    return canonical_mapping_digest(payload)


def required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"postprocessing authority field {key!r} must be a non-empty string")
    return value


def utc_now() -> datetime:
    return datetime.now(UTC)


def format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("postprocessing timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def verify_bytes(payload: bytes, *, expected_sha256: str, expected_size: int, label: str) -> None:
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{label} bytes differ from the declared immutable identity")


def _lock(handle: IO[bytes]) -> None:
    flock(handle.fileno(), LOCK_EX)


__all__ = [
    "append_event",
    "canonical_json_bytes",
    "format_timestamp",
    "fsync_directory",
    "fsync_tree_directories",
    "mapping_digest",
    "postprocessing_authority_lock",
    "postprocessing_operation_lock",
    "postprocessing_phase_coordinator_lock",
    "read_canonical_json",
    "required_string",
    "utc_now",
    "verify_bytes",
    "write",
    "write_no_replace",
]
