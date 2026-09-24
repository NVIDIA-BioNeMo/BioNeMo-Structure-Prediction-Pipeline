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

"""Immutable fold shard projection contracts.

One canonical ordered projection assigns every fold target to exactly one
global rank using deterministic LPT over ``member_lengths``. The projection is
digest-referenced from the fold Runtime Action via a
:class:`FoldShardProjectionBinding` that freezes its location, SHA-256, byte
size, worker count, and LPT version; each Runtime rank later verifies the whole
projection and selects exactly its global-rank row.

This module deliberately imports no ``contract.phase`` so ``contract.phase``
can import it back without a module-level cycle. It defines its own compact
canonical digest helper identical to ``canonical_mapping_digest``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")

# The single owner of the staged projection basename. The RunSpec binding stores
# the control-side authority-relative location (``attempts/<attempt>/...``);
# Control stages the projection document beside the immutable RunSpec under this
# exact basename and Runtime resolves the same basename against the staged
# RunSpec directory. Keeping one constant prevents the two sides from drifting.
FOLD_SHARD_PROJECTION_FILENAME = "fold-shard-projection.json"


def _compact_digest(payload: Mapping[str, object]) -> str:
    """Return the repository's stable compact-JSON SHA-256 digest."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FoldShardTarget:
    """One fold target and its LPT weight (producer-attested member length)."""

    target_id: str
    member_length: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "FoldShardTarget")
        if not self.target_id:
            raise ValueError("fold shard target_id must be non-empty")
        _validate_positive_int(self.member_length, "fold shard member_length")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "member_length": self.member_length,
        }


@dataclass(frozen=True)
class FoldShardRank:
    """The ordered target assignment for one global rank.

    A rank may carry no targets when the worker count exceeds the target count;
    that is a degenerate but valid LPT outcome and the rank still selects its
    (empty) row.
    """

    global_rank: int
    targets: tuple[FoldShardTarget, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "FoldShardRank")
        if not isinstance(self.global_rank, int) or isinstance(self.global_rank, bool) or self.global_rank < 0:
            raise ValueError("fold shard global_rank must be a non-negative integer")
        if not isinstance(self.targets, tuple) or any(
            not isinstance(target, FoldShardTarget) for target in self.targets
        ):
            raise ValueError("fold shard targets must be an immutable tuple of FoldShardTarget records")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "global_rank": self.global_rank,
            "targets": [target.to_mapping() for target in self.targets],
        }


@dataclass(frozen=True)
class FoldShardProjection:
    """One canonical ordered fold shard projection for every global rank."""

    worker_count: int
    lpt_version: int
    ranks: tuple[FoldShardRank, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "FoldShardProjection")
        _validate_positive_int(self.worker_count, "fold shard worker_count")
        _validate_positive_int(self.lpt_version, "fold shard lpt_version")
        if not isinstance(self.ranks, tuple) or any(not isinstance(rank, FoldShardRank) for rank in self.ranks):
            raise ValueError("fold shard ranks must be an immutable tuple of FoldShardRank records")
        if tuple(rank.global_rank for rank in self.ranks) != tuple(range(self.worker_count)):
            raise ValueError("fold shard ranks must cover 0..worker_count-1 exactly once in ascending order")
        target_ids = [target.target_id for rank in self.ranks for target in rank.targets]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("fold shard targets must be unique across all ranks")

    @property
    def digest(self) -> str:
        return _compact_digest(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "worker_count": self.worker_count,
            "lpt_version": self.lpt_version,
            "ranks": [rank.to_mapping() for rank in self.ranks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


@dataclass(frozen=True)
class FoldShardProjectionBinding:
    """Digest-bound location of one canonical fold shard projection."""

    location: str
    sha256: str
    size_bytes: int
    worker_count: int
    lpt_version: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "FoldShardProjectionBinding")
        if not self.location:
            raise ValueError("fold shard projection location must be non-empty")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("fold shard projection sha256 must be 64 lowercase hexadecimal characters")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("fold shard projection size_bytes must be a non-negative integer")
        _validate_positive_int(self.worker_count, "fold shard projection worker_count")
        _validate_positive_int(self.lpt_version, "fold shard projection lpt_version")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "location": self.location,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "worker_count": self.worker_count,
            "lpt_version": self.lpt_version,
        }


def fold_shard_target_from_mapping(payload: Mapping[str, object]) -> FoldShardTarget:
    _strict(payload, {"schema_version", "target_id", "member_length"}, "FoldShardTarget")
    return FoldShardTarget(
        schema_version=_schema(payload, "FoldShardTarget"),
        target_id=_str(payload, "target_id"),
        member_length=_int(payload, "member_length"),
    )


def fold_shard_rank_from_mapping(payload: Mapping[str, object]) -> FoldShardRank:
    _strict(payload, {"schema_version", "global_rank", "targets"}, "FoldShardRank")
    return FoldShardRank(
        schema_version=_schema(payload, "FoldShardRank"),
        global_rank=_int(payload, "global_rank"),
        targets=tuple(fold_shard_target_from_mapping(item) for item in _mappings(payload, "targets")),
    )


def fold_shard_projection_from_mapping(payload: Mapping[str, object]) -> FoldShardProjection:
    _strict(
        payload,
        {"schema_version", "worker_count", "lpt_version", "ranks"},
        "FoldShardProjection",
    )
    return FoldShardProjection(
        schema_version=_schema(payload, "FoldShardProjection"),
        worker_count=_int(payload, "worker_count"),
        lpt_version=_int(payload, "lpt_version"),
        ranks=tuple(fold_shard_rank_from_mapping(item) for item in _mappings(payload, "ranks")),
    )


def fold_shard_projection_binding_from_mapping(payload: Mapping[str, object]) -> FoldShardProjectionBinding:
    _strict(
        payload,
        {"schema_version", "location", "sha256", "size_bytes", "worker_count", "lpt_version"},
        "FoldShardProjectionBinding",
    )
    return FoldShardProjectionBinding(
        schema_version=_schema(payload, "FoldShardProjectionBinding"),
        location=_str(payload, "location"),
        sha256=_str(payload, "sha256"),
        size_bytes=_int(payload, "size_bytes"),
        worker_count=_int(payload, "worker_count"),
        lpt_version=_int(payload, "lpt_version"),
    )


def fold_shard_projection_document_bytes(projection: FoldShardProjection) -> bytes:
    """Return the exact on-disk canonical projection document bytes."""
    return projection.to_json().encode()


def fold_shard_projection_staged_basename(location: str) -> str:
    """Return the canonical staged basename for one authority-relative projection location.

    The binding ``location`` is the control-side authority-relative path. The
    cluster does not replicate the control authority tree; Control stages the
    projection document beside the immutable RunSpec under
    :data:`FOLD_SHARD_PROJECTION_FILENAME` and Runtime resolves that same
    basename against the staged RunSpec directory. Any location that does not
    end in the canonical basename is rejected so the two sides cannot drift.
    """
    if not location or Path(location).name != FOLD_SHARD_PROJECTION_FILENAME:
        raise ValueError(f"fold shard projection location must end in {FOLD_SHARD_PROJECTION_FILENAME!r}")
    return FOLD_SHARD_PROJECTION_FILENAME


def _strict(payload: Mapping[str, object], allowed: set[str], name: str) -> None:
    if "schema_version" not in payload:
        raise ValueError(f"missing explicit schema_version at {name}")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")


def _schema(payload: Mapping[str, object], name: str) -> int:
    value = payload.get("schema_version")
    if value is None:
        raise ValueError(f"missing explicit schema_version at {name}")
    return validate_schema_version(value, record_name=name)


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
    if not isinstance(value, list | tuple):
        raise ValueError(f"{key} must be a list")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must contain mappings")
    return cast("tuple[Mapping[str, object], ...]", tuple(value))


def _validate_schema(value: int, name: str) -> None:
    if validate_schema_version(value, record_name=name) != value:
        raise ValueError(f"{name} schema_version must be explicit")


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "FOLD_SHARD_PROJECTION_FILENAME",
    "FoldShardProjection",
    "FoldShardProjectionBinding",
    "FoldShardRank",
    "FoldShardTarget",
    "fold_shard_projection_binding_from_mapping",
    "fold_shard_projection_document_bytes",
    "fold_shard_projection_from_mapping",
    "fold_shard_projection_staged_basename",
    "fold_shard_rank_from_mapping",
    "fold_shard_target_from_mapping",
]
