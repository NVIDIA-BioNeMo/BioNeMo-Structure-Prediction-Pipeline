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

"""Focused postprocessing contracts extracted from postprocessing_receipt.py."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.phase import (
    canonical_mapping_digest,
    validate_phase_attempt_id,
    validate_phase_run_id,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")


_ARTIFACT_SET_ID = re.compile(r"postprocessing-artifact-set-[0-9a-f]{64}")


@dataclass(frozen=True)
class PostprocessingEvidenceArtifact:
    path: str
    sha256: str
    size_bytes: int
    verification_kind: Literal["content-sha256-v1", "inventory-metadata-v1"] = "content-sha256-v1"
    verification_source_path: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not self.path or self.path.startswith("/") or ".." in self.path.split("/"):
            raise ValueError("postprocessing evidence artifact path must be authority-relative")
        _sha(self.sha256)
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("postprocessing evidence artifact size must be non-negative")
        if self.verification_kind == "content-sha256-v1":
            if self.verification_source_path is not None:
                raise ValueError("content-hashed Artifact member must not name an inventory source")
        elif self.verification_kind == "inventory-metadata-v1":
            if (
                not self.verification_source_path
                or self.verification_source_path.startswith("/")
                or ".." in self.verification_source_path.split("/")
            ):
                raise ValueError("inventory-verified Artifact member requires a root-relative source")
        else:
            raise ValueError("unsupported postprocessing Artifact member verification kind")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "verification_kind": self.verification_kind,
            "verification_source_path": self.verification_source_path,
        }


@dataclass(frozen=True)
class PostprocessingLogicalArtifactSet:
    artifact_set_id: str
    root_name: str
    members: tuple[PostprocessingEvidenceArtifact, ...] = ()
    children: tuple[PostprocessingLogicalArtifactSet, ...] = ()
    artifact_set_kind: str = "postprocessing-logical-artifact-set-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        paths = tuple(item.path for item in self.members)
        child_names = tuple(item.root_name for item in self.children)
        if (
            self.artifact_set_kind != "postprocessing-logical-artifact-set-v1"
            or not self.root_name
            or "/" in self.root_name
            or self.root_name in {".", ".."}
            or bool(paths) == bool(child_names)
            or paths != tuple(sorted(paths))
            or len(set(paths)) != len(paths)
            or child_names != tuple(sorted(child_names))
            or len(set(child_names)) != len(child_names)
        ):
            raise ValueError("postprocessing logical Artifact Set must be one sorted leaf or branch")
        if _ARTIFACT_SET_ID.fullmatch(self.artifact_set_id) is None or self.artifact_set_id != (
            "postprocessing-artifact-set-" + canonical_mapping_digest(self.identity_mapping())
        ):
            raise ValueError("postprocessing logical Artifact Set id differs from its identity")

    def identity_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_set_kind": self.artifact_set_kind,
            "root_name": self.root_name,
            "members": [item.to_mapping() for item in self.members],
            "children": [item.to_mapping() for item in self.children],
        }

    def to_mapping(self) -> dict[str, object]:
        return {"artifact_set_id": self.artifact_set_id, **self.identity_mapping()}

    @property
    def flattened_members(self) -> tuple[PostprocessingEvidenceArtifact, ...]:
        if self.members:
            return tuple(
                PostprocessingEvidenceArtifact(
                    path=f"{self.root_name}/{item.path}",
                    sha256=item.sha256,
                    size_bytes=item.size_bytes,
                    verification_kind=item.verification_kind,
                    verification_source_path=item.verification_source_path,
                )
                for item in self.members
            )
        return tuple(
            PostprocessingEvidenceArtifact(
                path=f"{self.root_name}/{item.path}",
                sha256=item.sha256,
                size_bytes=item.size_bytes,
                verification_kind=item.verification_kind,
                verification_source_path=item.verification_source_path,
            )
            for child in self.children
            for item in child.flattened_members
        )


def postprocessing_artifact_set_id(payload: Mapping[str, object]) -> str:
    return "postprocessing-artifact-set-" + canonical_mapping_digest(payload)


def postprocessing_artifact_inventory_digest(artifact_set: PostprocessingLogicalArtifactSet) -> str:
    return canonical_mapping_digest(
        {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "members": [item.to_mapping() for item in artifact_set.flattened_members],
        }
    )


@dataclass(frozen=True)
class PostprocessingScientificOutputInventory:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    output_root: str
    members: tuple[PostprocessingEvidenceArtifact, ...]
    inventory_kind: str = "postprocessing-scientific-output-inventory-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        _sha(self.phase_runspec_digest)
        paths = tuple(item.path for item in self.members)
        if (
            self.inventory_kind != "postprocessing-scientific-output-inventory-v1"
            or not self.output_root.startswith("/")
            or not paths
            or paths != tuple(sorted(paths))
            or len(set(paths)) != len(paths)
        ):
            raise ValueError("scientific output inventory identity or sorted member set is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "inventory_kind": self.inventory_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "output_root": self.output_root,
            "members": [item.to_mapping() for item in self.members],
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


def _artifact(payload: Mapping[str, object]) -> PostprocessingEvidenceArtifact:
    _fields(
        payload,
        {
            "schema_version",
            "path",
            "sha256",
            "size_bytes",
            "verification_kind",
            "verification_source_path",
        },
        "PostprocessingEvidenceArtifact",
    )
    return PostprocessingEvidenceArtifact(
        schema_version=_schema(payload, "PostprocessingEvidenceArtifact"),
        path=_string(payload, "path"),
        sha256=_string(payload, "sha256"),
        size_bytes=_integer(payload, "size_bytes"),
        verification_kind=cast(
            "Literal['content-sha256-v1', 'inventory-metadata-v1']",
            _string(payload, "verification_kind"),
        ),
        verification_source_path=_optional_string(payload, "verification_source_path"),
    )


def _artifact_set(payload: Mapping[str, object]) -> PostprocessingLogicalArtifactSet:
    _fields(
        payload,
        {"schema_version", "artifact_set_kind", "artifact_set_id", "root_name", "members", "children"},
        "PostprocessingLogicalArtifactSet",
    )
    return PostprocessingLogicalArtifactSet(
        schema_version=_schema(payload, "PostprocessingLogicalArtifactSet"),
        artifact_set_kind=_string(payload, "artifact_set_kind"),
        artifact_set_id=_string(payload, "artifact_set_id"),
        root_name=_string(payload, "root_name"),
        members=tuple(_artifact(item) for item in _mapping_list(payload, "members")),
        children=tuple(_artifact_set(item) for item in _mapping_list(payload, "children")),
    )


def _scientific_inventory(payload: Mapping[str, object]) -> PostprocessingScientificOutputInventory:
    _fields(
        payload,
        {
            "schema_version",
            "inventory_kind",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "output_root",
            "members",
        },
        "PostprocessingScientificOutputInventory",
    )
    return PostprocessingScientificOutputInventory(
        schema_version=_schema(payload, "PostprocessingScientificOutputInventory"),
        inventory_kind=_string(payload, "inventory_kind"),
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        output_root=_string(payload, "output_root"),
        members=tuple(_artifact(item) for item in _mapping_list(payload, "members")),
    )


def postprocessing_scientific_output_inventory_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingScientificOutputInventory:
    return _scientific_inventory(payload)


def _mapping_list(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of mappings")
    return tuple(cast("Mapping[str, object]", item) for item in value)


def _fields(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} has missing or extra fields")


def _schema(payload_or_version: Mapping[str, object] | int, name: str) -> int:
    value = payload_or_version.get("schema_version") if isinstance(payload_or_version, Mapping) else payload_or_version
    return validate_schema_version(value, record_name=name)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{key} must be a non-empty string or null")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _sha(value: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError("postprocessing receipt digest must be lowercase SHA-256")


__all__ = [
    "PostprocessingEvidenceArtifact",
    "PostprocessingLogicalArtifactSet",
    "PostprocessingScientificOutputInventory",
    "postprocessing_artifact_inventory_digest",
    "postprocessing_artifact_set_id",
    "postprocessing_scientific_output_inventory_from_mapping",
]
