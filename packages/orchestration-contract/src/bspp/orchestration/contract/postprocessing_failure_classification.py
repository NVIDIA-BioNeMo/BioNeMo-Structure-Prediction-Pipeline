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

"""Contract-owned pure group-A/group-B postprocessing failure classifier.

This module is the decision core for the postprocessing autorequeue epic: a closed
typed observation record, a closed typed result record, and a total/deterministic
pure classifier that maps only the audited typed transport outcomes to group A and
everything else (including the explicit ``other`` kind) to group B.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.versioning import (
    CURRENT_CONTRACT_SCHEMA_VERSION,
    validate_schema_version,
)

PostprocessingTransportFailureKind = Literal[
    "timeout",
    "connection-reset",
    "temporary-dns",
    "http-500",
    "http-502",
    "http-503",
    "http-504",
    "other",
]

_GROUP_A_KINDS = frozenset(
    {
        "timeout",
        "connection-reset",
        "temporary-dns",
        "http-500",
        "http-502",
        "http-503",
        "http-504",
    }
)
_ALL_KINDS = frozenset(_GROUP_A_KINDS | {"other"})
_GROUP_VALUES = frozenset({"A", "B"})

_OBSERVATION_FIELDS = frozenset({"schema_version", "failure_kind", "failure_detail"})
_CLASSIFICATION_FIELDS = frozenset({"schema_version", "group", "failure_kind", "reason"})


@dataclass(frozen=True)
class PostprocessingTransportFailureObservation:
    """One audited transport failure observation."""

    failure_kind: PostprocessingTransportFailureKind
    failure_detail: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PostprocessingTransportFailureObservation")
        _validate_failure_kind(self.failure_kind)
        _validate_nonempty_text(self.failure_detail, "failure_detail")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-ready data."""
        return {
            "schema_version": self.schema_version,
            "failure_kind": self.failure_kind,
            "failure_detail": self.failure_detail,
        }


@dataclass(frozen=True)
class PostprocessingFailureClassification:
    """One closed group-A/group-B classification result."""

    group: Literal["A", "B"]
    failure_kind: PostprocessingTransportFailureKind
    reason: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PostprocessingFailureClassification")
        _validate_group(self.group)
        _validate_failure_kind(self.failure_kind)
        _validate_nonempty_text(self.reason, "reason")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-ready data."""
        return {
            "schema_version": self.schema_version,
            "group": self.group,
            "failure_kind": self.failure_kind,
            "reason": self.reason,
        }


def classify_transport_failure(
    observation: PostprocessingTransportFailureObservation,
) -> PostprocessingFailureClassification:
    """Classify one transport failure observation into group A or group B.

    Total and deterministic: audited group-A kinds map to group A, and everything
    else (including ``other``) maps to group B.
    """
    if observation.failure_kind in _GROUP_A_KINDS:
        return PostprocessingFailureClassification(
            group="A",
            failure_kind=observation.failure_kind,
            reason=f"audited {observation.failure_kind}",
            schema_version=observation.schema_version,
        )
    return PostprocessingFailureClassification(
        group="B",
        failure_kind=observation.failure_kind,
        reason="not a group-A transport outcome",
        schema_version=observation.schema_version,
    )


def postprocessing_transport_failure_observation_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingTransportFailureObservation:
    """Load one observation and reject unknown or malformed fields."""
    _reject_unknown_fields(payload, _OBSERVATION_FIELDS, record_name="PostprocessingTransportFailureObservation")
    return PostprocessingTransportFailureObservation(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingTransportFailureObservation"
        ),
        failure_kind=_required_failure_kind(payload, "failure_kind"),
        failure_detail=_required_str(payload, "failure_detail"),
    )


def postprocessing_failure_classification_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingFailureClassification:
    """Load one classification result and reject unknown or malformed fields."""
    _reject_unknown_fields(payload, _CLASSIFICATION_FIELDS, record_name="PostprocessingFailureClassification")
    return PostprocessingFailureClassification(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingFailureClassification"
        ),
        group=_required_group(payload, "group"),
        failure_kind=_required_failure_kind(payload, "failure_kind"),
        reason=_required_str(payload, "reason"),
    )


def _validate_direct_schema_version(schema_version: int, record_name: str) -> None:
    validated = validate_schema_version(schema_version, record_name=record_name)
    if validated != schema_version:
        msg = f"{record_name} schema_version must be declared explicitly"
        raise ValueError(msg)


def _validate_failure_kind(value: object) -> None:
    if value not in _ALL_KINDS:
        msg = f"unsupported postprocessing transport failure_kind: {value!r}"
        raise ValueError(msg)


def _validate_group(value: object) -> None:
    if value not in _GROUP_VALUES:
        msg = f"unsupported postprocessing failure group: {value!r}"
        raise ValueError(msg)


def _validate_nonempty_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip() or "\n" in value or "\r" in value:
        msg = f"{field_name} must be a non-empty trimmed single-line string"
        raise ValueError(msg)


def _reject_unknown_fields(payload: Mapping[str, object], allowed: frozenset[str], *, record_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        msg = f"{record_name} has unknown fields: {', '.join(unknown)}"
        raise ValueError(msg)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)
    return value


def _required_failure_kind(payload: Mapping[str, object], key: str) -> PostprocessingTransportFailureKind:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)
    return cast(PostprocessingTransportFailureKind, value)


def _required_group(payload: Mapping[str, object], key: str) -> Literal["A", "B"]:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)
    return cast(Literal["A", "B"], value)


__all__ = [
    "PostprocessingFailureClassification",
    "PostprocessingTransportFailureKind",
    "PostprocessingTransportFailureObservation",
    "classify_transport_failure",
    "postprocessing_failure_classification_from_mapping",
    "postprocessing_transport_failure_observation_from_mapping",
]
