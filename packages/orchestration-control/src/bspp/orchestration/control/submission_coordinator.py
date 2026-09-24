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

"""Single-controller create-once submission coordination."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from bspp.orchestration.contract.submission_evidence import SubmissionToken, submission_token_from_mapping

MAX_COORDINATOR_RECORD_BYTES = 64 * 1024
_CLAIM_LOCK = threading.Lock()


class SubmissionFence(Protocol):
    """Optional multi-controller lease/fence adapter; no default implementation."""

    def acquire(self, slot_id: str) -> str: ...
    def validate(self, slot_id: str, fence: str) -> None: ...


@dataclass(frozen=True)
class CoordinatorRecord:
    token: SubmissionToken
    status: str = "prepared"
    job_id: str | None = None
    scheduler_status: str | None = None

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "token": self.token.to_mapping(),
            "status": self.status,
            "job_id": self.job_id,
            "scheduler_status": self.scheduler_status,
        }


@dataclass(frozen=True)
class SubmissionClaim:
    created: bool
    record: CoordinatorRecord


class SubmissionCoordinator:
    """Coordinate one submission per logical slot on a single controller filesystem."""

    def __init__(self, evidence_dir: Path) -> None:
        self.root = evidence_dir / "submission-coordinator"

    def claim(self, token: SubmissionToken) -> SubmissionClaim:
        with _CLAIM_LOCK:
            self.root.mkdir(parents=True, exist_ok=True)
            _validate_root(self.root)
            path = self._path(token)
            record = CoordinatorRecord(token)
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                existing = self._load(path)
                if existing.token != token:
                    raise ValueError("submission slot already claimed with a different token") from None
                return SubmissionClaim(False, existing)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(_record_bytes(record))
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.root)
            return SubmissionClaim(True, record)

    def bind_job(self, token: SubmissionToken, *, job_id: str, scheduler_status: str) -> CoordinatorRecord:
        if not job_id:
            raise ValueError("job_id must be a non-empty string")
        if not scheduler_status:
            raise ValueError("scheduler_status must be a non-empty string")
        path = self._path(token)
        existing = self._load(path)
        if existing.token != token:
            raise ValueError("submission slot already claimed with a different token")
        if existing.job_id is not None:
            if existing.job_id != job_id or existing.scheduler_status != scheduler_status:
                raise ValueError(f"submission token already bound to scheduler job {existing.job_id}")
            return existing
        updated = replace(existing, status="submitted", job_id=job_id, scheduler_status=scheduler_status)
        temporary = path.with_suffix(".tmp")
        if os.path.lexists(temporary):
            raise ValueError("interrupted coordinator publication requires reconciliation")
        with temporary.open("xb") as handle:
            handle.write(_record_bytes(updated))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(self.root)
        return updated

    def load(self, token: SubmissionToken) -> CoordinatorRecord:
        """Load an existing slot without creating submission authority."""
        path = self._path(token)
        try:
            record = self._load(path)
        except FileNotFoundError as exc:
            raise ValueError("submission slot has no coordinator record") from exc
        if record.token != token:
            raise ValueError("submission slot already claimed with a different token")
        return record

    def record_path(self, token: SubmissionToken) -> Path:
        """Return the deterministic record path for an already-bound token."""
        return self._path(token)

    def _path(self, token: SubmissionToken) -> Path:
        slot = {
            "attempt": token.attempt,
            "run_id": token.run_id,
            "slice_id": token.slice_id,
            "step_index": token.step_index,
            "step_name": token.step_name,
        }
        slot_id = hashlib.sha256(json.dumps(slot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return self.root / f"{slot_id}.json"

    @staticmethod
    def _load(path: Path) -> CoordinatorRecord:
        data = _read_record(path)
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid submission coordinator record") from exc
        expected = {"format_version", "token", "status", "job_id", "scheduler_status"}
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload["format_version"] != 1
            or not isinstance(payload["token"], dict)
        ):
            raise ValueError("invalid submission coordinator record")
        status, job_id, scheduler_status = payload["status"], payload["job_id"], payload["scheduler_status"]
        if (
            status not in {"prepared", "submitted"}
            or (job_id is not None and not isinstance(job_id, str))
            or (scheduler_status is not None and not isinstance(scheduler_status, str))
            or (status == "prepared" and (job_id is not None or scheduler_status is not None))
            or (
                status == "submitted"
                and (
                    not isinstance(job_id, str)
                    or not job_id
                    or not isinstance(scheduler_status, str)
                    or not scheduler_status
                )
            )
        ):
            raise ValueError("invalid submission coordinator record")
        record = CoordinatorRecord(submission_token_from_mapping(payload["token"]), status, job_id, scheduler_status)
        if _record_bytes(record) != data:
            raise ValueError("invalid canonical coordinator record")
        return record


def _record_bytes(record: CoordinatorRecord) -> bytes:
    return (json.dumps(record.to_mapping(), sort_keys=True, separators=(",", ":")) + "\n").encode()


def _read_record(path: Path) -> bytes:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("submission coordinator record must be a regular file")
    if metadata.st_nlink != 1:
        raise ValueError("submission coordinator record must not be hard-linked")
    if metadata.st_size > MAX_COORDINATOR_RECORD_BYTES:
        raise ValueError("submission coordinator record exceeds size bound")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError("submission coordinator record changed during open")
        data = handle.read(MAX_COORDINATOR_RECORD_BYTES + 1)
    if len(data) > MAX_COORDINATOR_RECORD_BYTES:
        raise ValueError("submission coordinator record exceeds size bound")
    return data


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_root(path: Path) -> None:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise ValueError("submission coordinator root must be a real directory")


__all__ = ["CoordinatorRecord", "SubmissionClaim", "SubmissionCoordinator", "SubmissionFence"]
