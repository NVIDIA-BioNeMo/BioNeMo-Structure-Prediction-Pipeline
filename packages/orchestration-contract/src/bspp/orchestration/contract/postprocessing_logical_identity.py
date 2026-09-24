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

"""Focused postprocessing contracts extracted from phase_postprocessing.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from bspp.orchestration.contract._postprocessing_validation import _nonnegative, _schema, _sha
from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION

InputVerificationKind = Literal["local-content-sha256-v1", "authority-declared-content-v1"]


@dataclass(frozen=True)
class LogicalInputEntry:
    """Authority-level logical input without an unverified remote payload claim."""

    name: str
    verification_kind: InputVerificationKind
    authority: str
    content_sha256: str | None = None
    size_bytes: int | None = None
    member_identity: str | None = None
    expected_content_sha256: str | None = None
    expected_size_bytes: int | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not self.name or not self.authority:
            raise ValueError("logical input name and authority must be non-empty")
        if self.verification_kind == "local-content-sha256-v1":
            if (
                self.content_sha256 is None
                or self.size_bytes is None
                or self.member_identity is not None
                or self.expected_content_sha256 is not None
                or self.expected_size_bytes is not None
            ):
                raise ValueError("local-content input requires only content SHA-256 and size")
            _sha(self.content_sha256, "logical input content SHA-256")
            _nonnegative(self.size_bytes, "logical input size")
        elif self.verification_kind == "authority-declared-content-v1":
            if (
                not self.member_identity
                or self.expected_content_sha256 is None
                or self.expected_size_bytes is None
                or self.content_sha256 is not None
                or self.size_bytes is not None
            ):
                raise ValueError("authority-declared input requires member identity, expected SHA-256, and size")
            _sha(self.expected_content_sha256, "authority-declared expected content SHA-256")
            _nonnegative(self.expected_size_bytes, "authority-declared expected content size")
        else:
            raise ValueError(f"unsupported logical input verification kind: {self.verification_kind!r}")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "verification_kind": self.verification_kind,
            "authority": self.authority,
        }
        if self.content_sha256 is not None:
            result["content_sha256"] = self.content_sha256
            result["size_bytes"] = self.size_bytes
        if self.member_identity is not None:
            result["member_identity"] = self.member_identity
            result["expected_content_sha256"] = self.expected_content_sha256
            result["expected_size_bytes"] = self.expected_size_bytes
        return result


@dataclass(frozen=True)
class LogicalInputManifest:
    entries: tuple[LogicalInputEntry, ...]
    manifest_kind: str = "postprocessing-logical-inputs-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.manifest_kind != "postprocessing-logical-inputs-v1":
            raise ValueError("unsupported postprocessing logical input manifest kind")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("postprocessing logical input manifest must be a non-empty immutable tuple")
        names = tuple(item.name for item in self.entries)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("postprocessing logical inputs must be unique and name-sorted")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_kind": self.manifest_kind,
            "entries": [entry.to_mapping() for entry in self.entries],
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class PostprocessingLogicalInputIdentityV2:
    """Locator-free identity for one scientific input declared by authority."""

    name: str
    member_identity: str
    expected_content_sha256: str
    expected_size_bytes: int
    identity_kind: Literal["postprocessing-logical-input-identity-v2"] = "postprocessing-logical-input-identity-v2"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.identity_kind != "postprocessing-logical-input-identity-v2":
            raise ValueError("unsupported postprocessing logical input identity kind")
        if not self.name or not self.member_identity:
            raise ValueError("postprocessing logical input identity names must be non-empty")
        _sha(self.expected_content_sha256, "postprocessing logical input content SHA-256")
        _nonnegative(self.expected_size_bytes, "postprocessing logical input size")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity_kind": self.identity_kind,
            "name": self.name,
            "member_identity": self.member_identity,
            "expected_content_sha256": self.expected_content_sha256,
            "expected_size_bytes": self.expected_size_bytes,
        }


@dataclass(frozen=True)
class PostprocessingLogicalInputIdentityManifestV2:
    """Sorted scientific-input identities with no physical or document locator."""

    entries: tuple[PostprocessingLogicalInputIdentityV2, ...]
    manifest_kind: Literal["postprocessing-logical-input-identities-v2"] = "postprocessing-logical-input-identities-v2"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        names = tuple(item.name for item in self.entries)
        if (
            self.manifest_kind != "postprocessing-logical-input-identities-v2"
            or not names
            or names != tuple(sorted(names))
            or len(set(names)) != len(names)
        ):
            raise ValueError("postprocessing logical input identities must be non-empty, unique, and sorted")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_kind": self.manifest_kind,
            "entries": [entry.to_mapping() for entry in self.entries],
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class PhysicalInputLocator:
    """Attempt execution locator kept separate from logical content authority."""

    name: str
    locator: str
    purpose: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not self.name or not self.locator or not self.purpose:
            raise ValueError("physical input locator fields must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "locator": self.locator,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class PostprocessingScientificIdentityV1:
    """Auditable, locator-independent scientific identity component digests."""

    dataset_scope_digest: str
    scientific_parameters_digest: str
    logical_input_identity_digest: str
    acceptance_semantic_digest: str
    identity_kind: Literal["postprocessing-scientific-identity-v1"] = "postprocessing-scientific-identity-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.identity_kind != "postprocessing-scientific-identity-v1":
            raise ValueError("unsupported postprocessing scientific identity kind")
        for value in (
            self.dataset_scope_digest,
            self.scientific_parameters_digest,
            self.logical_input_identity_digest,
            self.acceptance_semantic_digest,
        ):
            _sha(value, "postprocessing scientific identity component")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity_kind": self.identity_kind,
            "dataset_scope_digest": self.dataset_scope_digest,
            "scientific_parameters_digest": self.scientific_parameters_digest,
            "logical_input_identity_digest": self.logical_input_identity_digest,
            "acceptance_semantic_digest": self.acceptance_semantic_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class PostprocessingDatasetScopeV2:
    mode: str
    array: str
    archive_source: str
    scope_kind: Literal["postprocessing-dataset-scope-v2"] = "postprocessing-dataset-scope-v2"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.scope_kind != "postprocessing-dataset-scope-v2" or not all(
            (self.mode, self.array, self.archive_source)
        ):
            raise ValueError("postprocessing dataset scope v2 fields must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope_kind": self.scope_kind,
            "mode": self.mode,
            "array": self.array,
            "archive_source": self.archive_source,
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class PostprocessingScientificParametersV2:
    fields: tuple[PostprocessingSemanticField, ...]
    parameters_kind: Literal["postprocessing-scientific-parameters-v2"] = "postprocessing-scientific-parameters-v2"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        paths = tuple(item.path for item in self.fields)
        if (
            self.parameters_kind != "postprocessing-scientific-parameters-v2"
            or not paths
            or paths != tuple(sorted(paths))
            or len(set(paths)) != len(paths)
        ):
            raise ValueError("postprocessing scientific parameter fields must be non-empty, unique, and sorted")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parameters_kind": self.parameters_kind,
            "fields": [item.to_mapping() for item in self.fields],
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class PostprocessingScientificIdentityV2:
    """Locator-free scientific identity; acceptance and runtime remain separate."""

    dataset_scope_digest: str
    scientific_parameters_digest: str
    logical_input_identity_digest: str
    identity_kind: Literal["postprocessing-scientific-identity-v2"] = "postprocessing-scientific-identity-v2"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.identity_kind != "postprocessing-scientific-identity-v2":
            raise ValueError("unsupported postprocessing scientific identity kind")
        for value in (
            self.dataset_scope_digest,
            self.scientific_parameters_digest,
            self.logical_input_identity_digest,
        ):
            _sha(value, "postprocessing scientific identity component")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity_kind": self.identity_kind,
            "dataset_scope_digest": self.dataset_scope_digest,
            "scientific_parameters_digest": self.scientific_parameters_digest,
            "logical_input_identity_digest": self.logical_input_identity_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class PostprocessingSemanticField:
    """One normalized, path-addressed semantic value in an auditable preimage."""

    path: str
    value: object
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not self.path or self.path.startswith(".") or self.path.endswith(".") or ".." in self.path:
            raise ValueError("postprocessing semantic field path is invalid")
        try:
            encoded = json.dumps(self.value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("postprocessing semantic field value must be canonical JSON") from exc
        if json.loads(encoded) != self.value:
            raise ValueError("postprocessing semantic field value must round-trip as canonical JSON")

    def to_mapping(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "path": self.path, "value": self.value}


__all__ = [
    "InputVerificationKind",
    "LogicalInputEntry",
    "LogicalInputManifest",
    "PhysicalInputLocator",
    "PostprocessingDatasetScopeV2",
    "PostprocessingLogicalInputIdentityManifestV2",
    "PostprocessingLogicalInputIdentityV2",
    "PostprocessingScientificIdentityV1",
    "PostprocessingScientificIdentityV2",
    "PostprocessingScientificParametersV2",
    "PostprocessingSemanticField",
]
