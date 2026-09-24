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

"""Strict immutable provenance for automatic folding Attempt carry-forward.

This module is the folding sibling of ``phase_carry_forward.py``: a
folding carry record is a *different* schema from the preprocessing record and
uses its own id namespace and authority filename, so folding and preprocessing
carry authority can never collide. It imports nothing from ``contract.phase``
so ``contract.phase`` can import the folding reference back without a
module-level cycle (the same rule ``folding_shard.py`` documents).

The record names one immediate source Attempt, itemizes every verified carried
target with its exact produced outputs (structure and scores files), and binds
an ordered, duplicate-free closure of ancestor folding carry IDs and digests
. The per-rank completion journal that this record is derived from is
owned by the Runtime (``runtime/folding/rank_journal.py``); this module does not
redefine that schema, and the Control derivation parses the journal strictly on
its own side of the Control/Runtime import boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import cast

from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_FOLD_ACTION_ID = re.compile(r"fold-[0-9]{6}")
_CARRY_ID = re.compile(r"folding-carry-forward-[0-9a-f]{64}")


def _digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FoldingCarryForwardOutput:
    """One exact produced output file bound by its absolute path, size, and SHA-256."""

    output_path: str
    size_bytes: int
    sha256: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes <= 0:
            raise ValueError("folding carry output size_bytes must be positive")
        _absolute(self.output_path, "folding carry output path")
        _sha(self.sha256, "folding carry output")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "output_path": self.output_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class FoldingCarryForwardContent:
    """One carried fold target and its exact verified produced outputs."""

    target_id: str
    sequence_sha256: str
    source_action_id: str
    source_rank: int
    outputs: tuple[FoldingCarryForwardOutput, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not self.target_id:
            raise ValueError("folding carry target_id must be non-empty")
        _sha(self.sequence_sha256, "folding carry sequence identity")
        _match(self.source_action_id, _FOLD_ACTION_ID, "folding carry source action id")
        if not isinstance(self.source_rank, int) or isinstance(self.source_rank, bool) or self.source_rank < 0:
            raise ValueError("folding carry source_rank must be a non-negative integer")
        if not isinstance(self.outputs, tuple) or not self.outputs:
            raise ValueError("folding carry content outputs must be a non-empty immutable tuple")
        if any(not isinstance(item, FoldingCarryForwardOutput) for item in self.outputs):
            raise ValueError("folding carry content outputs must contain FoldingCarryForwardOutput records")
        if len({item.output_path for item in self.outputs}) != len(self.outputs):
            raise ValueError("folding carry content outputs must be unique by output path")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "sequence_sha256": self.sequence_sha256,
            "source_action_id": self.source_action_id,
            "source_rank": self.source_rank,
            "outputs": [item.to_mapping() for item in self.outputs],
        }


@dataclass(frozen=True)
class FoldingCarryAncestorReference:
    """One ancestor folding carry record referenced by id and digest only."""

    folding_carry_forward_id: str
    digest: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.folding_carry_forward_id, _CARRY_ID, "folding carry ancestor id")
        _sha(self.digest, "folding carry ancestor")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "folding_carry_forward_id": self.folding_carry_forward_id,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class FoldingCarryForwardRecord:
    """One immutable automatic folding carry record for a successor Attempt."""

    folding_carry_forward_id: str
    phase_run_id: str
    phase_plan_digest: str
    source_attempt_id: str
    source_attempt_ordinal: int
    source_runspec_digest: str
    target_attempt_id: str
    target_attempt_ordinal: int
    backend: str
    content: tuple[FoldingCarryForwardContent, ...]
    ancestor_closure: tuple[FoldingCarryAncestorReference, ...]
    content_digest: str
    declared_at: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.folding_carry_forward_id, _CARRY_ID, "folding carry-forward id")
        _match(self.phase_run_id, _PHASE_RUN_ID, "folding carry-forward Phase Run id")
        for label, value in (
            ("Phase Plan", self.phase_plan_digest),
            ("source RunSpec", self.source_runspec_digest),
            ("content", self.content_digest),
        ):
            _sha(value, label)
        _match(self.source_attempt_id, _ATTEMPT_ID, "folding carry-forward source Attempt id")
        _match(self.target_attempt_id, _ATTEMPT_ID, "folding carry-forward target Attempt id")
        if not isinstance(self.source_attempt_ordinal, int) or isinstance(self.source_attempt_ordinal, bool):
            raise ValueError("folding carry-forward source_attempt_ordinal must be an integer")
        if self.source_attempt_ordinal <= 0:
            raise ValueError("folding carry-forward source_attempt_ordinal must be positive")
        if self.target_attempt_ordinal != self.source_attempt_ordinal + 1:
            raise ValueError("folding carry-forward target must immediately follow source Attempt")
        if not self.backend:
            raise ValueError("folding carry-forward backend must be non-empty")
        if not isinstance(self.content, tuple) or not self.content:
            raise ValueError("folding carry-forward content must be non-empty")
        if any(not isinstance(item, FoldingCarryForwardContent) for item in self.content):
            raise ValueError("folding carry-forward content must contain strict records")
        keys = tuple((item.source_rank, item.target_id) for item in self.content)
        if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
            raise ValueError("folding carry-forward content must be unique and (source_rank, target_id) ordered")
        if not isinstance(self.ancestor_closure, tuple) or any(
            not isinstance(item, FoldingCarryAncestorReference) for item in self.ancestor_closure
        ):
            raise ValueError("folding carry-forward ancestor_closure must be an immutable tuple of references")
        closure_ids = tuple(item.folding_carry_forward_id for item in self.ancestor_closure)
        if len(set(closure_ids)) != len(closure_ids):
            raise ValueError("folding carry-forward ancestor_closure must be duplicate-free")
        expected_content_digest = _digest(
            {"schema_version": self.schema_version, "content": [item.to_mapping() for item in self.content]}
        )
        if self.content_digest != expected_content_digest:
            raise ValueError("folding carry-forward content digest does not match content")
        _timestamp(self.declared_at, "folding carry-forward declaration")
        if self.folding_carry_forward_id != folding_carry_forward_id(self.identity_mapping()):
            raise ValueError("folding carry-forward id does not match canonical content")

    def identity_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "phase_plan_digest": self.phase_plan_digest,
            "source_attempt_id": self.source_attempt_id,
            "source_attempt_ordinal": self.source_attempt_ordinal,
            "source_runspec_digest": self.source_runspec_digest,
            "target_attempt_id": self.target_attempt_id,
            "target_attempt_ordinal": self.target_attempt_ordinal,
            "backend": self.backend,
            "content": [item.to_mapping() for item in self.content],
            "ancestor_closure": [item.to_mapping() for item in self.ancestor_closure],
            "content_digest": self.content_digest,
            "declared_at": self.declared_at,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {"folding_carry_forward_id": self.folding_carry_forward_id, **self.identity_mapping()}


def folding_carry_forward_id(identity: Mapping[str, object]) -> str:
    return f"folding-carry-forward-{_digest(identity)}"


@dataclass(frozen=True)
class FoldingCarryForwardReference:
    """RunSpec-embedded reference to one sealed folding carry record."""

    folding_carry_forward_id: str
    digest: str
    location: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.folding_carry_forward_id, _CARRY_ID, "folding carry-forward reference id")
        _sha(self.digest, "folding carry-forward reference")
        path = PurePosixPath(self.location)
        if path.is_absolute() or ".." in path.parts or path.name != "folding-carry-forward.json":
            raise ValueError("folding carry-forward reference location must be safe and authority-relative")
        if len(path.parts) != 3 or path.parts[0] != "attempts" or _ATTEMPT_ID.fullmatch(path.parts[1]) is None:
            raise ValueError("folding carry-forward reference location must be Attempt-bound")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "folding_carry_forward_id": self.folding_carry_forward_id,
            "digest": self.digest,
            "location": self.location,
        }


def folding_carry_forward_output_from_mapping(payload: Mapping[str, object]) -> FoldingCarryForwardOutput:
    _strict(payload, set(FoldingCarryForwardOutput.__dataclass_fields__), "FoldingCarryForwardOutput")
    return FoldingCarryForwardOutput(
        schema_version=_version(payload, "FoldingCarryForwardOutput"),
        output_path=_str(payload, "output_path"),
        size_bytes=_int(payload, "size_bytes"),
        sha256=_str(payload, "sha256"),
    )


def folding_carry_forward_content_from_mapping(payload: Mapping[str, object]) -> FoldingCarryForwardContent:
    _strict(payload, set(FoldingCarryForwardContent.__dataclass_fields__), "FoldingCarryForwardContent")
    return FoldingCarryForwardContent(
        schema_version=_version(payload, "FoldingCarryForwardContent"),
        target_id=_str(payload, "target_id"),
        sequence_sha256=_str(payload, "sequence_sha256"),
        source_action_id=_str(payload, "source_action_id"),
        source_rank=_int(payload, "source_rank"),
        outputs=tuple(folding_carry_forward_output_from_mapping(item) for item in _mappings(payload, "outputs")),
    )


def folding_carry_ancestor_reference_from_mapping(payload: Mapping[str, object]) -> FoldingCarryAncestorReference:
    _strict(payload, set(FoldingCarryAncestorReference.__dataclass_fields__), "FoldingCarryAncestorReference")
    return FoldingCarryAncestorReference(
        schema_version=_version(payload, "FoldingCarryAncestorReference"),
        folding_carry_forward_id=_str(payload, "folding_carry_forward_id"),
        digest=_str(payload, "digest"),
    )


def folding_carry_forward_record_from_mapping(payload: Mapping[str, object]) -> FoldingCarryForwardRecord:
    _strict(payload, set(FoldingCarryForwardRecord.__dataclass_fields__), "FoldingCarryForwardRecord")
    return FoldingCarryForwardRecord(
        schema_version=_version(payload, "FoldingCarryForwardRecord"),
        folding_carry_forward_id=_str(payload, "folding_carry_forward_id"),
        phase_run_id=_str(payload, "phase_run_id"),
        phase_plan_digest=_str(payload, "phase_plan_digest"),
        source_attempt_id=_str(payload, "source_attempt_id"),
        source_attempt_ordinal=_int(payload, "source_attempt_ordinal"),
        source_runspec_digest=_str(payload, "source_runspec_digest"),
        target_attempt_id=_str(payload, "target_attempt_id"),
        target_attempt_ordinal=_int(payload, "target_attempt_ordinal"),
        backend=_str(payload, "backend"),
        content=tuple(folding_carry_forward_content_from_mapping(item) for item in _mappings(payload, "content")),
        ancestor_closure=tuple(
            folding_carry_ancestor_reference_from_mapping(item) for item in _mappings(payload, "ancestor_closure")
        ),
        content_digest=_str(payload, "content_digest"),
        declared_at=_str(payload, "declared_at"),
    )


def folding_carry_forward_reference_from_mapping(payload: Mapping[str, object]) -> FoldingCarryForwardReference:
    _strict(payload, set(FoldingCarryForwardReference.__dataclass_fields__), "FoldingCarryForwardReference")
    return FoldingCarryForwardReference(
        schema_version=_version(payload, "FoldingCarryForwardReference"),
        folding_carry_forward_id=_str(payload, "folding_carry_forward_id"),
        digest=_str(payload, "digest"),
        location=_str(payload, "location"),
    )


def _schema(value: int, name: str) -> None:
    if validate_schema_version(value, record_name=name) != value:
        raise ValueError(f"{name} schema_version must be explicit")


def _version(payload: Mapping[str, object], name: str) -> int:
    return validate_schema_version(payload.get("schema_version"), record_name=name)


def _strict(payload: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown or missing:
        raise ValueError(f"invalid {name} fields; missing={missing}, unknown={unknown}")


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
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of mappings")
    return tuple(cast("Mapping[str, object]", item) for item in value)


def _match(value: str, pattern: re.Pattern[str], label: str) -> None:
    if pattern.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")


def _sha(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} digest must be lowercase SHA-256")


def _absolute(value: str, label: str) -> None:
    path = PurePosixPath(value)
    if not value or not path.is_absolute() or ".." in path.parts or "\x00" in value:
        raise ValueError(f"{label} must be a normalized absolute POSIX path")


def _timestamp(value: str, label: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {label} timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} timestamp must include timezone")


__all__ = [
    "FoldingCarryAncestorReference",
    "FoldingCarryForwardContent",
    "FoldingCarryForwardOutput",
    "FoldingCarryForwardRecord",
    "FoldingCarryForwardReference",
    "folding_carry_ancestor_reference_from_mapping",
    "folding_carry_forward_content_from_mapping",
    "folding_carry_forward_id",
    "folding_carry_forward_output_from_mapping",
    "folding_carry_forward_record_from_mapping",
    "folding_carry_forward_reference_from_mapping",
]
