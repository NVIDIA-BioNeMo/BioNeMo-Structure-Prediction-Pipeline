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

"""Focused postprocessing contracts extracted from postprocessing_finalization_bundle.py."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.phase import validate_phase_attempt_id, validate_phase_run_id
from bspp.orchestration.contract.postprocessing_handoff import (
    PostprocessingVerifiedArtifactLocation,
    postprocessing_verified_artifact_location_from_mapping,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PostprocessingArtifactLocationSet:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    artifact_set_id: str
    locations: tuple[PostprocessingVerifiedArtifactLocation, ...]
    location_set_kind: Literal["postprocessing-artifact-locations-v1"] = "postprocessing-artifact-locations-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        _sha(self.phase_runspec_digest, "postprocessing Artifact Location set RunSpec digest")
        location_ids = tuple(item.artifact_location_id for item in self.locations)
        if (
            self.location_set_kind != "postprocessing-artifact-locations-v1"
            or not self.artifact_set_id.startswith("postprocessing-artifact-set-")
            or not self.locations
            or location_ids != tuple(sorted(location_ids))
            or len(set(location_ids)) != len(location_ids)
            or any(item.artifact_set_id != self.artifact_set_id for item in self.locations)
        ):
            raise ValueError("postprocessing Artifact Location set is empty, unordered, or inconsistent")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "location_set_kind": self.location_set_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "artifact_set_id": self.artifact_set_id,
            "locations": [item.to_mapping() for item in self.locations],
        }


def postprocessing_artifact_location_set_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingArtifactLocationSet:
    _fields(
        payload,
        {
            "schema_version",
            "location_set_kind",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "artifact_set_id",
            "locations",
        },
        "PostprocessingArtifactLocationSet",
    )
    if _string(payload, "location_set_kind") != "postprocessing-artifact-locations-v1":
        raise ValueError("unsupported postprocessing Artifact Location set discriminator")
    return PostprocessingArtifactLocationSet(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingArtifactLocationSet"
        ),
        location_set_kind="postprocessing-artifact-locations-v1",
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        artifact_set_id=_string(payload, "artifact_set_id"),
        locations=tuple(
            postprocessing_verified_artifact_location_from_mapping(item) for item in _mapping_list(payload, "locations")
        ),
    )


def _sha(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")


def _schema(value: int, record: str) -> None:
    validate_schema_version(value, record_name=record)


def _fields(payload: Mapping[str, object], allowed: set[str], record: str) -> None:
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown or missing:
        raise ValueError(f"{record} fields differ; missing={missing!r}, unknown={unknown!r}")


def _mapping_list(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of mappings")
    return tuple(cast("Mapping[str, object]", item) for item in value)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


__all__ = ["PostprocessingArtifactLocationSet", "postprocessing_artifact_location_set_from_mapping"]
