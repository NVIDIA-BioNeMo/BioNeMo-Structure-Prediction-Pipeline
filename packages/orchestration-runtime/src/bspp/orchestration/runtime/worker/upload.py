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

"""Upload planning, retry, accounting, and local fallback helpers."""

from __future__ import annotations

import fcntl
import json
import os
import random
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from bspp.orchestration.runtime.worker.metadata import clean_metadata_json_outputs
from bspp.orchestration.runtime.worker.output_layout import TransferPair, join_destination_prefix

SYNC_KEEP_DIRS: tuple[str, ...] = ("modelcif", "modelpdb", "bcif", "scores", "clash_interface_analysis")
LUSTRE_DIR_RENAME: dict[str, str] = {
    "clash_interface_analysis": "metadata/clashes_and_interfaces_granular",
}
FLAT_DEST_SUBDIR: dict[str, str] = {
    "modelcif": "",
    "modelpdb": "",
    "bcif": "",
    "scores": "",
    "clash_interface_analysis": "metadata/clashes_and_interfaces_granular",
}
S3_UPLOAD_MAX_RETRIES = 2
S3_UPLOAD_RETRY_DELAY_SECONDS = 5.0


class UploadCommandRunner(Protocol):
    """Fakeable command execution boundary for upload helpers."""

    def __call__(self, argv: Sequence[str]) -> int:
        """Run *argv* and return a process exit code."""


@dataclass(frozen=True, slots=True)
class UploadAttemptResult:
    """Result of an upload command with retry accounting."""

    success: bool
    attempts: int
    uploaded_files: tuple[Path, ...]
    destination_prefix: str | None = None


@dataclass(frozen=True, slots=True)
class UploadFallbackResult:
    """Result of an S3 upload attempt with optional Lustre fallback."""

    upload_result: UploadAttemptResult
    fallback_used: bool
    copied_files: tuple[Path, ...]
    status: str


def collect_flat_upload_files(
    source_dir: Path,
    batch_ids: Iterable[str] | None = None,
) -> tuple[tuple[Path, Path], ...]:
    """Return ``(local_path, flat_relative_path)`` pairs for upload/tar packaging."""

    batch_id_tuple = tuple(batch_ids) if batch_ids is not None else None
    pairs: list[tuple[Path, Path]] = []

    for dirname in SYNC_KEEP_DIRS:
        src_sub = source_dir / dirname
        if not src_sub.exists():
            continue
        flat_subdir = FLAT_DEST_SUBDIR[dirname]
        rel_base = Path(flat_subdir) if flat_subdir else Path()
        for src in _collect_batch_files_in_dir(src_sub, batch_id_tuple):
            pairs.append((src, rel_base / src.name))

    for meta_subdir in ("metadata/search", "metadata/collection"):
        meta_dir = source_dir / meta_subdir
        if not meta_dir.exists():
            continue
        for entry in meta_dir.iterdir():
            if entry.is_file():
                pairs.append((entry, Path(meta_subdir) / entry.name))

    return tuple(pairs)


def plan_s3_upload_transfers(
    source_dir: Path,
    s3_prefix: str,
    batch_ids: Iterable[str] | None = None,
) -> tuple[TransferPair, ...]:
    """Plan flat-layout S3 transfers for ``s5cmd run``."""

    return tuple(
        TransferPair(source=local, destination=join_destination_prefix(s3_prefix, relative.as_posix()))
        for local, relative in collect_flat_upload_files(source_dir, batch_ids)
    )


def write_s5cmd_command_file(command_file: Path, transfers: Iterable[TransferPair]) -> int:
    """Write ``s5cmd run`` command-file lines and return the command count."""

    pairs = tuple(transfers)
    command_file.parent.mkdir(parents=True, exist_ok=True)
    command_file.write_text(
        "".join(f"cp {pair.source} {pair.destination}\n" for pair in pairs),
        encoding="utf-8",
    )
    return len(pairs)


def upload_s3_command_file(
    command_file: Path,
    *,
    s5cmd_path: str,
    numworkers: int,
    uploaded_files: Iterable[Path],
    destination_prefix: str,
    runner: UploadCommandRunner | None = None,
    max_retries: int = S3_UPLOAD_MAX_RETRIES,
    retry_delay_seconds: float = S3_UPLOAD_RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> UploadAttemptResult:
    """Run an ``s5cmd run`` command file with legacy retry semantics."""

    uploaded_file_tuple = tuple(uploaded_files)
    attempts = 0
    actual_runner = runner or _run_command
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        rc = actual_runner((s5cmd_path, "--numworkers", str(numworkers), "run", str(command_file)))
        if rc == 0:
            command_file.unlink(missing_ok=True)
            return UploadAttemptResult(
                success=True,
                attempts=attempts,
                uploaded_files=uploaded_file_tuple,
                destination_prefix=destination_prefix,
            )
        if attempt < max_retries:
            sleep(retry_delay_seconds * (2**attempt))

    command_file.unlink(missing_ok=True)
    return UploadAttemptResult(
        success=False,
        attempts=attempts,
        uploaded_files=(),
        destination_prefix=destination_prefix,
    )


def upload_files_to_s3(
    source_dir: Path,
    s3_prefix: str,
    *,
    s5cmd_path: str,
    batch_ids: Iterable[str] | None = None,
    numworkers: int = 256,
    runner: UploadCommandRunner | None = None,
    max_retries: int = S3_UPLOAD_MAX_RETRIES,
    retry_delay_seconds: float = S3_UPLOAD_RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> UploadAttemptResult:
    """Plan, write, and execute a flat-layout file upload."""

    transfers = plan_s3_upload_transfers(source_dir, s3_prefix, batch_ids)
    if not transfers:
        return UploadAttemptResult(success=True, attempts=0, uploaded_files=(), destination_prefix=s3_prefix)
    command_file = source_dir / "_s3_upload_commands.txt"
    write_s5cmd_command_file(command_file, transfers)
    return upload_s3_command_file(
        command_file,
        s5cmd_path=s5cmd_path,
        numworkers=numworkers,
        uploaded_files=(pair.source for pair in transfers),
        destination_prefix=s3_prefix,
        runner=runner,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        sleep=sleep,
    )


def upload_single_file_to_s3(
    local_path: Path,
    s3_uri: str,
    *,
    s5cmd_path: str,
    numworkers: int = 16,
    runner: UploadCommandRunner | None = None,
    max_retries: int = S3_UPLOAD_MAX_RETRIES,
    retry_delay_seconds: float = S3_UPLOAD_RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> UploadAttemptResult:
    """Upload one file with ``s5cmd cp`` and legacy retry semantics."""

    attempts = 0
    actual_runner = runner or _run_command
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        rc = actual_runner((s5cmd_path, "--numworkers", str(numworkers), "cp", str(local_path), s3_uri))
        if rc == 0:
            return UploadAttemptResult(
                success=True,
                attempts=attempts,
                uploaded_files=(local_path,),
                destination_prefix=s3_uri,
            )
        if attempt < max_retries:
            sleep(retry_delay_seconds * (2**attempt))
    return UploadAttemptResult(success=False, attempts=attempts, uploaded_files=(), destination_prefix=s3_uri)


@contextmanager
def upload_slot(slots_dir: Path, num_slots: int) -> Iterator[Path | None]:
    """Acquire one flock-based upload semaphore slot.

    ``num_slots <= 0`` disables throttling and yields ``None``.
    """

    actual_slots = resolve_upload_slot_count(slots_dir, num_slots)
    if actual_slots <= 0:
        yield None
        return

    slots_dir.mkdir(parents=True, exist_ok=True)
    slot_indices = list(range(actual_slots))
    random.shuffle(slot_indices)

    for idx in slot_indices:
        slot_path = slots_dir / f"slot_{idx}"
        fd = os.open(str(slot_path), os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield slot_path
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            return
        except OSError:
            os.close(fd)

    block_idx = random.randint(0, actual_slots - 1)
    slot_path = slots_dir / f"slot_{block_idx}"
    fd = os.open(str(slot_path), os.O_CREAT | os.O_RDWR, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        yield slot_path
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def resolve_upload_slot_count(slots_dir: Path, default_slots: int) -> int:
    """Read the live upload-slot override, falling back to *default_slots*."""

    override_path = slots_dir / "max_slots"
    try:
        raw_value = override_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default_slots
    except OSError:
        return default_slots
    if not raw_value:
        return default_slots
    try:
        override_slots = int(raw_value)
    except ValueError:
        return default_slots
    if override_slots < 0:
        return default_slots
    return override_slots


def copy_batch_flat_to_success(
    source_dir: Path,
    success_dir: Path,
    batch_ids: Iterable[str] | None = None,
) -> tuple[Path, ...]:
    """Copy batch outputs into the flat ``success_outputs`` layout."""

    copied: list[Path] = []
    for source, relative in collect_flat_upload_files(source_dir, batch_ids):
        destination = success_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return tuple(copied)


def try_upload_then_lustre_fallback(
    source_dir: Path,
    success_dir: Path,
    s3_prefix: str,
    *,
    s5cmd_path: str,
    batch_ids: Iterable[str],
    numworkers: int = 256,
    runner: UploadCommandRunner | None = None,
    max_retries: int = S3_UPLOAD_MAX_RETRIES,
    retry_delay_seconds: float = S3_UPLOAD_RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> UploadFallbackResult:
    """Upload a batch to S3, falling back to flat Lustre copy on exhaustion."""

    batch_id_tuple = tuple(batch_ids)
    upload_result = upload_files_to_s3(
        source_dir,
        s3_prefix,
        s5cmd_path=s5cmd_path,
        batch_ids=batch_id_tuple,
        numworkers=numworkers,
        runner=runner,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        sleep=sleep,
    )
    if upload_result.success:
        clean_batch_outputs(source_dir, batch_id_tuple)
        return UploadFallbackResult(
            upload_result=upload_result,
            fallback_used=False,
            copied_files=(),
            status="uploaded",
        )

    copied_files = copy_batch_flat_to_success(source_dir, success_dir, batch_id_tuple)
    clean_batch_outputs(source_dir, batch_id_tuple)
    clean_metadata_json_outputs(source_dir, batch_id_tuple)
    return UploadFallbackResult(
        upload_result=upload_result,
        fallback_used=True,
        copied_files=copied_files,
        status="upload_failed_lustre_fallback",
    )


def clean_batch_outputs(source_dir: Path, batch_ids: Iterable[str] | None = None) -> int:
    """Remove batch payload files from the worker output directories."""

    batch_id_tuple = tuple(batch_ids) if batch_ids is not None else None
    removed = 0
    for dirname in SYNC_KEEP_DIRS:
        src_sub = source_dir / dirname
        if not src_sub.exists():
            continue
        for path in _collect_batch_files_in_dir(src_sub, batch_id_tuple):
            path.unlink()
            removed += 1
    return removed


def cleanup_success_outputs(success_dir: Path) -> bool:
    """Remove ``success_outputs`` after successful self-upload."""

    existed = success_dir.exists()
    shutil.rmtree(success_dir, ignore_errors=True)
    return existed


def write_uploaded_marker(
    shard_dir: Path,
    *,
    s3_prefix: str | None,
    upload_mode: str,
    tar_prefix: str | None,
    total_batches: int,
    metadata_files: Iterable[str | Path],
    shard_id: int,
    model_count: int,
    timestamp: str | None = None,
) -> Path:
    """Write the legacy ``.uploaded`` marker JSON for one shard."""

    metadata_file_strings = [str(path) for path in metadata_files]
    failed_model_count = len(_failed_model_ids(shard_dir / "failed_models.tsv"))
    payload = {
        "s3_prefix": s3_prefix or "",
        "upload_mode": upload_mode,
        "tar_prefix": tar_prefix or "",
        "status": "partial_uploaded" if failed_model_count else "uploaded",
        "total_files": count_batch_marker_files(shard_dir) + len(metadata_file_strings),
        "total_batches": total_batches,
        "failed_models": failed_model_count,
        "metadata_files": metadata_file_strings,
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
        "shard_id": shard_id,
        "model_count": model_count,
    }
    marker_path = shard_dir / ".uploaded"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return marker_path


def uploaded_marker_is_complete(shard_dir: Path) -> bool:
    """Return true only when ``shard_dir/.uploaded`` records a complete upload."""
    marker_path = shard_dir / ".uploaded"
    if not marker_path.exists():
        return False
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "uploaded"


def count_batch_marker_files(shard_dir: Path) -> int:
    """Count uploaded file lines from batch markers.

    This scans marker files on disk and is intended for once-per-shard marker
    creation, not per-model hot loops.
    """

    if not shard_dir.exists():
        return 0
    total = 0
    for entry in shard_dir.iterdir():
        if not entry.is_file():
            continue
        if not (entry.name.startswith(".batch_") or entry.name.startswith(".retry_failed_batch_")):
            continue
        if not (entry.name.endswith("_done") or entry.name.endswith("_partial_uploaded")):
            continue
        total += len([line for line in entry.read_text(encoding="utf-8").splitlines() if line.strip()])
    return total


def _collect_batch_files_in_dir(src_sub: Path, batch_ids: tuple[str, ...] | None) -> tuple[Path, ...]:
    if batch_ids is None:
        return tuple(entry for entry in src_sub.iterdir() if entry.is_file())
    batch_id_set = set(batch_ids)
    prefixes = tuple(prefix for model_id in batch_ids for prefix in (f"{model_id}-", f"{model_id}_"))
    return tuple(
        entry
        for entry in src_sub.iterdir()
        if entry.is_file() and (entry.name in batch_id_set or entry.name.startswith(prefixes))
    )


def _failed_model_ids(failed_path: Path) -> frozenset[str]:
    if not failed_path.exists():
        return frozenset()
    return frozenset(
        line.split("\t", 1)[0]
        for line in failed_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.split("\t", 1)[0]
    )


def _run_command(argv: Sequence[str]) -> int:
    return subprocess.run(argv, capture_output=True, text=True).returncode


__all__ = [
    "FLAT_DEST_SUBDIR",
    "LUSTRE_DIR_RENAME",
    "S3_UPLOAD_MAX_RETRIES",
    "S3_UPLOAD_RETRY_DELAY_SECONDS",
    "SYNC_KEEP_DIRS",
    "UploadAttemptResult",
    "UploadCommandRunner",
    "UploadFallbackResult",
    "clean_batch_outputs",
    "cleanup_success_outputs",
    "collect_flat_upload_files",
    "copy_batch_flat_to_success",
    "count_batch_marker_files",
    "plan_s3_upload_transfers",
    "resolve_upload_slot_count",
    "try_upload_then_lustre_fallback",
    "upload_files_to_s3",
    "upload_s3_command_file",
    "upload_single_file_to_s3",
    "upload_slot",
    "uploaded_marker_is_complete",
    "write_s5cmd_command_file",
    "write_uploaded_marker",
]
