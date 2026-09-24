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

"""Per-rank append-only fold completion journal.

Each packed fold rank appends one JSON line per successfully folded target to
``<fold-action-root>/ranks/<rank>/journal.jsonl``. Every event is flushed and
``fsync``'d before the next target is folded, so only complete events are ever
carryable: a torn trailing append (a partial line written by a crashed rank) is
ignored by the bounded reader while every prior fsync'd event remains.

The event binds the full authority tuple: Phase Run/Attempt identity,
the fold Runtime Action digest, the canonical shard projection digest and its
worker count / LPT version, the rank, the predecessor handoff digest, the target
and sequence identity, the backend and qualification tuple id, and the exact
output path / size / SHA-256 of each produced structure and scores file.

This module imports only ``bspp.orchestration.contract.*`` (plus stdlib): the
runtime distribution must never import the control distribution, so the
qualification tuple id is a byte-identical runtime mirror of
``control.folding_phase_types.folding_qualification_tuple_id``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import cast

from bspp.orchestration.contract.phase import FOLDING_BACKENDS, canonical_mapping_digest
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _fsync_directory(path: Path) -> None:
    """fsync a directory so a just-created entry survives a node crash."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "phase_run_id",
        "attempt_id",
        "rank",
        "fold_action_id",
        "fold_action_digest",
        "shard_projection_sha256",
        "shard_projection_worker_count",
        "shard_projection_lpt_version",
        "predecessor_digest",
        "target_id",
        "sequence_sha256",
        "description",
        "backend",
        "qualification_tuple_id",
        "outputs",
    }
)


@dataclass(frozen=True)
class RankJournalOutput:
    """One produced output file bound by its exact path, size, and SHA-256."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("rank journal output path must be non-empty")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ValueError("rank journal output size must be a non-negative integer")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("rank journal output sha256 must be 64 lowercase hexadecimal characters")

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True)
class RankJournalEvent:
    """One fsync'd per-target completion event in a rank journal."""

    phase_run_id: str
    attempt_id: str
    rank: int
    fold_action_id: str
    fold_action_digest: str
    shard_projection_sha256: str
    shard_projection_worker_count: int
    shard_projection_lpt_version: int
    predecessor_digest: str
    target_id: str
    sequence_sha256: str
    description: str
    backend: str
    qualification_tuple_id: str
    outputs: tuple[RankJournalOutput, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="RankJournalEvent")
        if not self.phase_run_id:
            raise ValueError("rank journal event phase_run_id must be non-empty")
        if not self.attempt_id:
            raise ValueError("rank journal event attempt_id must be non-empty")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 0:
            raise ValueError("rank journal event rank must be a non-negative integer")
        if not self.fold_action_id:
            raise ValueError("rank journal event fold_action_id must be non-empty")
        if _SHA256.fullmatch(self.fold_action_digest) is None:
            raise ValueError("rank journal event fold_action_digest must be 64 lowercase hexadecimal characters")
        if _SHA256.fullmatch(self.shard_projection_sha256) is None:
            raise ValueError("rank journal event shard_projection_sha256 must be 64 lowercase hexadecimal characters")
        if not isinstance(self.shard_projection_worker_count, int) or isinstance(
            self.shard_projection_worker_count, bool
        ):
            raise ValueError("rank journal event shard_projection_worker_count must be an integer")
        if self.shard_projection_worker_count <= 0:
            raise ValueError("rank journal event shard_projection_worker_count must be positive")
        if not isinstance(self.shard_projection_lpt_version, int) or isinstance(
            self.shard_projection_lpt_version, bool
        ):
            raise ValueError("rank journal event shard_projection_lpt_version must be an integer")
        if self.shard_projection_lpt_version <= 0:
            raise ValueError("rank journal event shard_projection_lpt_version must be positive")
        if _SHA256.fullmatch(self.predecessor_digest) is None:
            raise ValueError("rank journal event predecessor_digest must be 64 lowercase hexadecimal characters")
        if not self.target_id:
            raise ValueError("rank journal event target_id must be non-empty")
        if _SHA256.fullmatch(self.sequence_sha256) is None:
            raise ValueError("rank journal event sequence_sha256 must be 64 lowercase hexadecimal characters")
        if not self.description:
            raise ValueError("rank journal event description must be non-empty")
        if self.backend not in FOLDING_BACKENDS:
            raise ValueError(f"rank journal event backend must be a supported folding backend; got {self.backend!r}")
        if _SHA256.fullmatch(self.qualification_tuple_id) is None:
            raise ValueError("rank journal event qualification_tuple_id must be 64 lowercase hexadecimal characters")
        if not isinstance(self.outputs, tuple) or any(
            not isinstance(output, RankJournalOutput) for output in self.outputs
        ):
            raise ValueError("rank journal event outputs must be an immutable tuple of RankJournalOutput records")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "rank": self.rank,
            "fold_action_id": self.fold_action_id,
            "fold_action_digest": self.fold_action_digest,
            "shard_projection_sha256": self.shard_projection_sha256,
            "shard_projection_worker_count": self.shard_projection_worker_count,
            "shard_projection_lpt_version": self.shard_projection_lpt_version,
            "predecessor_digest": self.predecessor_digest,
            "target_id": self.target_id,
            "sequence_sha256": self.sequence_sha256,
            "description": self.description,
            "backend": self.backend,
            "qualification_tuple_id": self.qualification_tuple_id,
            "outputs": [output.to_mapping() for output in self.outputs],
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True) + "\n"


def folding_qualification_tuple_id(*, backend: str, kernel_image: str, cluster_snapshot_digest: str) -> str:
    """Return the deterministic folding operational-selection tuple id.

    Runtime mirror of ``control.folding_phase_types.folding_qualification_tuple_id``
    folding has no preprocessing runtime qualification, so the tuple
    binds the backend, its selected kernel image, and the resolved cluster
    snapshot digest. This stays byte-identical to the control formula; any change
    to the control formula must update both sites.
    """
    if backend not in FOLDING_BACKENDS:
        raise ValueError(f"unsupported folding backend: {backend!r}")
    if not kernel_image:
        raise ValueError("folding kernel image must be non-empty")
    if _SHA256.fullmatch(cluster_snapshot_digest) is None:
        raise ValueError("folding cluster snapshot digest must be a lowercase SHA-256")
    return canonical_mapping_digest(
        {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "backend": backend,
            "kernel_image": kernel_image,
            "cluster_snapshot_digest": cluster_snapshot_digest,
        }
    )


class RankJournalWriter:
    """Append-only JSONL rank journal with ``fsync`` per appended event.

    The constructor creates the parent directory and opens the file in append
    mode, so opening a writer for an empty rank still creates the (empty)
    ``journal.jsonl`` — the e13s08 reduce uses that to distinguish "rank ran
    with zero targets" from "rank never ran".
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        # The empty-file existence signal (and the first entry) must survive a
        # crash, so fsync the containing directory after the file is created.
        _fsync_directory(path.parent)

    def append(self, event: RankJournalEvent) -> None:
        self._handle.write(event.to_json_line())
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> RankJournalWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def read_rank_journal(path: Path) -> tuple[RankJournalEvent, ...]:
    """Read a bounded rank journal, ignoring a torn trailing append.

    The file is split on newlines; a trailing empty element (from the final
    newline) is dropped. A JSON parse failure on the **final** line only is a
    torn append and is ignored (prior fsync'd events remain); any malformed
    non-final line raises.
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
            raise ValueError(f"malformed rank journal line {index} in {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"malformed rank journal line {index} in {path}: expected a JSON object")
        events.append(rank_journal_event_from_mapping(payload))
    return tuple(events)


def rank_journal_output_from_mapping(payload: Mapping[str, object]) -> RankJournalOutput:
    _strict(payload, {"path", "size", "sha256"}, "RankJournalOutput")
    return RankJournalOutput(path=_str(payload, "path"), size=_int(payload, "size"), sha256=_str(payload, "sha256"))


def rank_journal_event_from_mapping(payload: Mapping[str, object]) -> RankJournalEvent:
    _strict(payload, _EVENT_FIELDS, "RankJournalEvent")
    return RankJournalEvent(
        schema_version=_schema(payload),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        rank=_int(payload, "rank"),
        fold_action_id=_str(payload, "fold_action_id"),
        fold_action_digest=_str(payload, "fold_action_digest"),
        shard_projection_sha256=_str(payload, "shard_projection_sha256"),
        shard_projection_worker_count=_int(payload, "shard_projection_worker_count"),
        shard_projection_lpt_version=_int(payload, "shard_projection_lpt_version"),
        predecessor_digest=_str(payload, "predecessor_digest"),
        target_id=_str(payload, "target_id"),
        sequence_sha256=_str(payload, "sequence_sha256"),
        description=_str(payload, "description"),
        backend=_str(payload, "backend"),
        qualification_tuple_id=_str(payload, "qualification_tuple_id"),
        outputs=tuple(rank_journal_output_from_mapping(item) for item in _mappings(payload, "outputs")),
    )


def _strict(payload: Mapping[str, object], allowed: frozenset[str] | set[str], name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")


def _schema(payload: Mapping[str, object]) -> int:
    value = payload.get("schema_version")
    if value is None:
        raise ValueError("missing explicit schema_version")
    return validate_schema_version(value, record_name="RankJournalEvent")


def _str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _mappings(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must contain mappings")
    return cast("tuple[Mapping[str, object], ...]", tuple(value))


__all__ = [
    "RankJournalEvent",
    "RankJournalOutput",
    "RankJournalWriter",
    "folding_qualification_tuple_id",
    "rank_journal_event_from_mapping",
    "rank_journal_output_from_mapping",
    "read_rank_journal",
]
