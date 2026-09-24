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

"""Strict contract for one flat ``.tar.lz4`` prediction handoff bundle.

This record represents a single transport archive of complete PDB/JSON
prediction pairs.  It validates the bundle's identity and its internal
membership consistency only:

* the archive name must match the harvested ``bspp_<YYMMDD>_<HHMM>_<letter><batch>.tar.lz4``
  pattern;
* the SHA-256 digest must be a lowercase 64-hex value;
* byte sizes must be positive;
* ``member_count`` must equal ``len(member_ids)``.

It deliberately does **not** validate the format of individual member
identities; that is the prediction-pair schema's responsibility.  It also
deliberately does **not** impose a maximum bundle cardinality: the 5,000-pair
figure is producer policy, not a consumer invariant.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

# Descriptive producer-default bundle cardinality, harvested from:
#   * OpenFold-TRT ``--proteins-per-archive`` default (5000)
#   * Runtime ``MAX_PROTEINS_PER_SHARD`` (5000)
#
# This constant is descriptive producer policy and must never participate in
# ``PredictionArchiveBundle`` validation.
MAX_PAIRS_PER_ARCHIVE = 5000

_BUNDLE_NAME = re.compile(r"bspp_[0-9]{6}_[0-9]{4}_[a-z][0-9]{5}\.tar\.lz4")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PredictionArchiveBundle:
    """One ``.tar.lz4`` prediction handoff bundle with membership consistency."""

    bundle_name: str
    member_ids: tuple[str, ...]
    member_count: int
    sha256: str
    size_bytes: int
    created_at: str | None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PredictionArchiveBundle")
        if _BUNDLE_NAME.fullmatch(self.bundle_name) is None:
            raise ValueError(
                "PredictionArchiveBundle bundle_name must match "
                "bspp_<YYMMDD>_<HHMM>_<lowercase-letter><batch:05d>.tar.lz4"
            )
        if not isinstance(self.member_ids, tuple) or not self.member_ids:
            raise ValueError("PredictionArchiveBundle member_ids must be a non-empty immutable tuple")
        if any(not isinstance(member_id, str) or not member_id for member_id in self.member_ids):
            raise ValueError("PredictionArchiveBundle member_ids must contain only non-empty strings")
        if len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError("PredictionArchiveBundle member_ids must be unique")
        if not isinstance(self.member_count, int) or isinstance(self.member_count, bool):
            raise ValueError("PredictionArchiveBundle member_count must be an integer")
        if self.member_count != len(self.member_ids):
            raise ValueError("PredictionArchiveBundle member_count must match its exact member inventory")
        _validate_sha256(self.sha256, "PredictionArchiveBundle sha256")
        _validate_positive_int(self.size_bytes, "PredictionArchiveBundle size_bytes")
        if self.created_at is not None:
            _validate_timestamp(self.created_at, "PredictionArchiveBundle created_at")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "bundle_name": self.bundle_name,
            "member_ids": list(self.member_ids),
            "member_count": self.member_count,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
        }


def prediction_archive_bundle_from_mapping(payload: Mapping[str, object]) -> PredictionArchiveBundle:
    """Load a strict prediction archive bundle from a serialized mapping."""
    _strict(
        payload,
        {"schema_version", "bundle_name", "member_ids", "member_count", "sha256", "size_bytes", "created_at"},
        {"schema_version", "bundle_name", "member_ids", "member_count", "sha256", "size_bytes"},
        "PredictionArchiveBundle",
    )
    return PredictionArchiveBundle(
        schema_version=_schema(payload, "PredictionArchiveBundle"),
        bundle_name=_str(payload, "bundle_name"),
        member_ids=_strings(payload, "member_ids"),
        member_count=_int(payload, "member_count"),
        sha256=_str(payload, "sha256"),
        size_bytes=_int(payload, "size_bytes"),
        created_at=_optional_str(payload, "created_at"),
    )


def _strict(payload: Mapping[str, object], allowed: set[str], required: set[str], name: str) -> None:
    if "schema_version" not in payload:
        raise ValueError(f"missing explicit schema_version at {name}")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Missing {name} field(s): {', '.join(missing)}")


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


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _strings(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return cast("tuple[str, ...]", tuple(value))


def _validate_schema(value: int, name: str) -> None:
    if validate_schema_version(value, record_name=name) != value:
        raise ValueError(f"{name} schema_version must be explicit")


def _validate_sha256(value: str, name: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")


def _validate_timestamp(value: str, name: str) -> None:
    if not value.endswith("Z"):
        raise ValueError(f"{name} must be an RFC 3339 UTC timestamp with a 'Z' suffix")
    try:
        _parse_timestamp(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an RFC 3339 UTC timestamp with a 'Z' suffix") from exc


__all__ = [
    "MAX_PAIRS_PER_ARCHIVE",
    "PredictionArchiveBundle",
    "prediction_archive_bundle_from_mapping",
]
