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

"""Verified-copy carry adoption for folding successor Attempts.

A folding carry record (``contract/folding_carry_forward.py``) names the exact
predecessor-produced outputs a successor may adopt. Before any successor
scientific work runs, every carried output is stream-copied from its narrowly
mounted read-only predecessor source into its canonical successor Attempt path,
flushed and ``fsync``'d, atomically renamed, then re-read and re-verified
against the exact byte size and SHA-256 recorded in the carry record.

Adoption is fail-closed: any copy or verification failure aborts the whole fold
action before a scientific backend is invoked and never modifies the read-only
predecessor mount. Each adopted target appends one fsync'd adopted-success
journal event under the successor authority tuple (compatible with the e13s07
``rank_journal.py`` record plus the adopted discriminator, source Attempt id,
and carry-record digest) so the e13s08 closure validation and the e13s10
next-Retry scan both observe a complete journal set.

This module imports only ``contract.*``, the e13s07 ``rank_journal.py`` mirror,
and stdlib: it never imports the control distribution and never invokes a
scientific backend.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.folding_carry_forward import FoldingCarryForwardRecord
from bspp.orchestration.runtime.folding.rank_journal import (
    RankJournalEvent,
    RankJournalOutput,
    rank_journal_event_from_mapping,
)


class CarryAdoptionError(ValueError):
    """Verified-copy carry adoption failed closed before scientific work."""


_ADOPTED_EXTRA_FIELDS = frozenset({"event_kind", "source_attempt_id", "carry_record_digest"})


@dataclass(frozen=True)
class CarryAdoptionAuthorityBinding:
    """The successor authority tuple that binds one adopted journal event."""

    phase_run_id: str
    attempt_id: str
    fold_action_id: str
    fold_action_digest: str
    shard_projection_sha256: str
    shard_projection_worker_count: int
    shard_projection_lpt_version: int
    predecessor_digest: str
    backend: str
    qualification_tuple_id: str
    descriptions: Mapping[str, str]


def copy_and_verify(source: Path, target: Path, *, size_bytes: int, sha256: str) -> None:
    """Stream-copy ``source`` into ``target``, fsync, rename, then re-verify.

    The copy is written to a sibling temporary file in the target directory,
    flushed and ``fsync``'d, then atomically renamed into the canonical target
    path. The target is then re-read and its exact byte size and SHA-256 are
    checked against the sealed carry-record values. Any mismatch or OSError
    raises :class:`CarryAdoptionError` before the read-only source is ever
    modified, and no partial target is left behind.
    """
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
        raise CarryAdoptionError("carry copy requires a positive size_bytes")
    if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
        raise CarryAdoptionError("carry copy requires a lowercase SHA-256")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".bspp-carry-")
    tmp_path = Path(tmp_name)
    replaced = False
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as src:
            while chunk := src.read(1024 * 1024):
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_path, target)
        replaced = True
    except OSError as exc:
        raise CarryAdoptionError(f"carry copy failed for {source} -> {target}: {exc}") from exc
    finally:
        if not replaced:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    digest = hashlib.sha256()
    actual_size = 0
    try:
        with target.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                actual_size += len(chunk)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(target)
        raise CarryAdoptionError(f"carry re-verification failed for {target}: {exc}") from exc
    if actual_size != size_bytes or digest.hexdigest() != sha256:
        with contextlib.suppress(OSError):
            os.unlink(target)
        raise CarryAdoptionError(
            f"carry verification failed for {target}: expected {size_bytes} bytes / {sha256}, "
            f"observed {actual_size} bytes / {digest.hexdigest()}"
        )


def adopt_carried_outputs(
    carry_record: FoldingCarryForwardRecord,
    *,
    successor_action_root: Path,
    rank: int,
    authority_binding: CarryAdoptionAuthorityBinding,
) -> tuple[str, ...]:
    """Copy and verify this rank's carried targets and append adopted events.

    Returns the ordered adopted target ids for this rank. Each carried output is
    copied from its sealed predecessor path (mounted read-only at its exact
    path) into the canonical successor rank output path, then one fsync'd
    adopted journal event is appended beneath ``ranks/<rank>/adopted.jsonl``.
    """
    packed = authority_binding.shard_projection_worker_count > 1
    carried = [item for item in carry_record.content if item.source_rank == rank]
    adopted_path = successor_action_root / "ranks" / str(rank) / "adopted.jsonl"
    adopted_path.parent.mkdir(parents=True, exist_ok=True)
    adopted_target_ids: list[str] = []
    for item in carried:
        target_outputs: list[RankJournalOutput] = []
        for output in item.outputs:
            filename = Path(output.output_path).name
            target_path = _successor_output_path(
                successor_action_root,
                rank,
                item.target_id,
                filename,
                packed=packed,
            )
            copy_and_verify(
                Path(output.output_path),
                target_path,
                size_bytes=output.size_bytes,
                sha256=output.sha256,
            )
            target_outputs.append(
                RankJournalOutput(path=str(target_path), size=output.size_bytes, sha256=output.sha256)
            )
        description = authority_binding.descriptions.get(item.target_id, f"a3ms/{item.target_id}.a3m")
        event = RankJournalEvent(
            phase_run_id=authority_binding.phase_run_id,
            attempt_id=authority_binding.attempt_id,
            rank=rank,
            fold_action_id=authority_binding.fold_action_id,
            fold_action_digest=authority_binding.fold_action_digest,
            shard_projection_sha256=authority_binding.shard_projection_sha256,
            shard_projection_worker_count=authority_binding.shard_projection_worker_count,
            shard_projection_lpt_version=authority_binding.shard_projection_lpt_version,
            predecessor_digest=authority_binding.predecessor_digest,
            target_id=item.target_id,
            sequence_sha256=item.sequence_sha256,
            description=description,
            backend=authority_binding.backend,
            qualification_tuple_id=authority_binding.qualification_tuple_id,
            outputs=tuple(target_outputs),
        )
        _append_adopted_event(
            adopted_path,
            event,
            source_attempt_id=carry_record.source_attempt_id,
            carry_record_digest=carry_record.digest,
        )
        adopted_target_ids.append(item.target_id)
    return tuple(adopted_target_ids)


def adopt_all_carried_outputs(
    carry_record: FoldingCarryForwardRecord,
    *,
    successor_action_root: Path,
    authority_binding: CarryAdoptionAuthorityBinding,
) -> tuple[str, ...]:
    """Full-carry variant: copy/verify every carried target and write adopted
    events per rank so the e13s08 closure validation sees a complete journal
    set."""
    adopted: list[str] = []
    for rank in sorted({item.source_rank for item in carry_record.content}):
        adopted.extend(
            adopt_carried_outputs(
                carry_record,
                successor_action_root=successor_action_root,
                rank=rank,
                authority_binding=authority_binding,
            )
        )
    return tuple(adopted)


def read_adopted_journal(path: Path) -> tuple[RankJournalEvent, ...]:
    """Read one adopted journal, ignoring a torn trailing append.

    Adopted events carry three provenance fields beyond the native
    :class:`RankJournalEvent` authority tuple (``event_kind``,
    ``source_attempt_id``, and ``carry_record_digest``); they are stripped
    before the strict native reader is applied so the e13s08 closure validation
    consumes identical events regardless of their native/adopted origin.
    """
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines = lines[:-1]
    events: list[RankJournalEvent] = []
    for index, line in enumerate(lines):
        is_final = index == len(lines) - 1
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if is_final:
                break
            raise ValueError(f"malformed adopted journal line {index} in {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"malformed adopted journal line {index} in {path}: expected a JSON object")
        native = {key: value for key, value in payload.items() if key not in _ADOPTED_EXTRA_FIELDS}
        events.append(rank_journal_event_from_mapping(native))
    return tuple(events)


def _successor_output_path(
    successor_action_root: Path,
    rank: int,
    target_id: str,
    filename: str,
    *,
    packed: bool,
) -> Path:
    if packed:
        return successor_action_root / "ranks" / str(rank) / "outputs" / target_id / filename
    return successor_action_root / "outputs" / target_id / filename


def _append_adopted_event(
    path: Path,
    event: RankJournalEvent,
    *,
    source_attempt_id: str,
    carry_record_digest: str,
) -> None:
    mapping = event.to_mapping()
    mapping["event_kind"] = "adopted"
    mapping["source_attempt_id"] = source_attempt_id
    mapping["carry_record_digest"] = carry_record_digest
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(mapping, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "CarryAdoptionAuthorityBinding",
    "CarryAdoptionError",
    "adopt_all_carried_outputs",
    "adopt_carried_outputs",
    "copy_and_verify",
    "read_adopted_journal",
]
