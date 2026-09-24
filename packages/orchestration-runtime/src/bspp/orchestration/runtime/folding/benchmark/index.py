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

"""Canonical-pair index schema, strict loader, and pure builder API.

The canonical-pair index is a read-only mapping produced at folding-run
completion and consumed by the validation-suite discovery surface.
It stores ``structure_path``/``scores_path`` but never opens those
files; validate_run loads pairs through the frozen ``prediction_pair`` contract.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.runspec import VALID_TOOL_USED

_SCHEMA_VERSION = 1
_SEQUENCE_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _validate_nonempty_str(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)


def _validate_sequence_sha256(value: object) -> None:
    if not isinstance(value, str) or _SEQUENCE_SHA256_RE.fullmatch(value) is None:
        msg = "sequence_sha256 must be 64 lowercase hex characters"
        raise ValueError(msg)


def _reject_duplicate_target_ids(entries: tuple[CanonicalPairIndexEntry, ...]) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.target_id in seen:
            msg = f"Duplicate target_id {entry.target_id!r} in CanonicalPairIndex"
            raise ValueError(msg)
        seen.add(entry.target_id)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


@dataclass(frozen=True)
class CanonicalPairIndexEntry:
    """One canonical structure/scores pair keyed by target identity."""

    target_id: str
    sequence_sha256: str
    model_entity_id: str
    tool_used: str
    structure_path: str
    scores_path: str

    def __post_init__(self) -> None:
        _validate_nonempty_str(self.target_id, "target_id")
        _validate_sequence_sha256(self.sequence_sha256)
        _validate_nonempty_str(self.model_entity_id, "model_entity_id")
        if self.tool_used not in VALID_TOOL_USED:
            msg = f"tool_used must be one of {VALID_TOOL_USED!r}; got {self.tool_used!r}"
            raise ValueError(msg)
        _validate_nonempty_str(self.structure_path, "structure_path")
        _validate_nonempty_str(self.scores_path, "scores_path")

    def to_mapping(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "sequence_sha256": self.sequence_sha256,
            "model_entity_id": self.model_entity_id,
            "tool_used": self.tool_used,
            "structure_path": self.structure_path,
            "scores_path": self.scores_path,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class CanonicalPairIndex:
    """An immutable, schema-versioned mapping of canonical prediction pairs."""

    schema_version: int
    run_id: str
    entries: tuple[CanonicalPairIndexEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            msg = (
                f"Unsupported canonical-pair index schema_version {self.schema_version!r}; "
                f"supported version: {_SCHEMA_VERSION}"
            )
            raise ValueError(msg)
        _validate_nonempty_str(self.run_id, "run_id")
        if not isinstance(self.entries, tuple) or not all(
            isinstance(entry, CanonicalPairIndexEntry) for entry in self.entries
        ):
            msg = "entries must be an immutable tuple of CanonicalPairIndexEntry values"
            raise ValueError(msg)
        _reject_duplicate_target_ids(self.entries)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "entries": [entry.to_mapping() for entry in self.entries],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


def _entry_from_mapping(payload: Mapping[str, object]) -> CanonicalPairIndexEntry:
    return CanonicalPairIndexEntry(
        target_id=_required_str(payload, "target_id"),
        sequence_sha256=_required_str(payload, "sequence_sha256"),
        model_entity_id=_required_str(payload, "model_entity_id"),
        tool_used=_required_str(payload, "tool_used"),
        structure_path=_required_str(payload, "structure_path"),
        scores_path=_required_str(payload, "scores_path"),
    )


def load_canonical_pair_index(path: Path) -> CanonicalPairIndex:
    """Load and strictly validate a canonical-pair index JSON file.

    Fails closed (``ValueError``) on an absent/unreadable file, malformed JSON,
    a non-object top level, an unknown/missing schema version, duplicate
    target_ids, or any invalid entry field. Never opens the pair files.
    """
    if not isinstance(path, Path):
        msg = "path must be a pathlib.Path"
        raise ValueError(msg)
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        msg = f"Cannot read canonical-pair index {path}: {exc}"
        raise ValueError(msg) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"Malformed JSON in canonical-pair index {path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = "canonical-pair index must be a JSON object"
        raise ValueError(msg)
    schema_version = _required_int(payload, "schema_version")
    run_id = _required_str(payload, "run_id")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        msg = "entries must be a list"
        raise ValueError(msg)
    entries: list[CanonicalPairIndexEntry] = []
    for position, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            msg = f"entries[{position}] must be an object"
            raise ValueError(msg)
        entries.append(_entry_from_mapping(item))
    return CanonicalPairIndex(schema_version=schema_version, run_id=run_id, entries=tuple(entries))


def build_canonical_pair_index(
    *,
    run_id: str,
    entries: Iterable[tuple[str, str, str, str, str, str]],
) -> CanonicalPairIndex:
    """Build a canonical-pair index, ordering entries by target_id.

    Duplicate target_ids are rejected exactly like the loader.
    """
    records = tuple(CanonicalPairIndexEntry(*entry) for entry in entries)
    ordered = tuple(sorted(records, key=lambda record: record.target_id))
    return CanonicalPairIndex(schema_version=_SCHEMA_VERSION, run_id=run_id, entries=ordered)


def write_canonical_pair_index(index: CanonicalPairIndex, path: Path) -> None:
    """Write a canonical-pair index as deterministic JSON via atomic replace."""
    if not isinstance(index, CanonicalPairIndex):
        msg = "index must be a CanonicalPairIndex"
        raise ValueError(msg)
    if not isinstance(path, Path):
        msg = "path must be a pathlib.Path"
        raise ValueError(msg)
    text = index.to_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


__all__ = [
    "CanonicalPairIndex",
    "CanonicalPairIndexEntry",
    "build_canonical_pair_index",
    "load_canonical_pair_index",
    "write_canonical_pair_index",
]
