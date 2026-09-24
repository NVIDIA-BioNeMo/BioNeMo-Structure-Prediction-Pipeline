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
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, cast

from bspp.orchestration.contract.phase import (
    canonical_mapping_digest,
    validate_phase_attempt_id,
    validate_phase_run_id,
)
from bspp.orchestration.contract.postprocessing_transfer_limits import POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")


_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class PostprocessingSmallOutputIdentity:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_output_path(self.path)
        _sha(self.sha256, "postprocessing small output SHA-256")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("postprocessing small output size must be a non-negative integer")

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class PostprocessingTarManifestReference:
    path: str
    manifest_id: str
    member_count: int

    def __post_init__(self) -> None:
        _sha(self.manifest_id, "postprocessing tar manifest reference id")
        if self.path != f"outputs/tar-manifests/{self.manifest_id}.json":
            raise ValueError("postprocessing tar manifest reference path differs from its id")
        if (
            not isinstance(self.member_count, int)
            or isinstance(self.member_count, bool)
            or self.member_count <= 0
            or self.member_count > POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1.max_members_per_tar_manifest
        ):
            raise ValueError("postprocessing tar manifest reference count is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "manifest_id": self.manifest_id, "member_count": self.member_count}


@dataclass(frozen=True)
class PostprocessingScientificOutputRoot:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    output_root: str
    small_outputs: tuple[PostprocessingSmallOutputIdentity, ...]
    tar_manifests: tuple[PostprocessingTarManifestReference, ...]
    root_kind: Literal["postprocessing-scientific-output-root-v1"] = "postprocessing-scientific-output-root-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        _sha(self.phase_runspec_digest, "postprocessing scientific root RunSpec digest")
        if self.root_kind != "postprocessing-scientific-output-root-v1" or not self.output_root.startswith("/"):
            raise ValueError("postprocessing scientific output root identity is invalid")
        small_paths = tuple(item.path for item in self.small_outputs)
        manifest_paths = tuple(item.path for item in self.tar_manifests)
        if (
            small_paths != tuple(sorted(small_paths))
            or len(set(small_paths)) != len(small_paths)
            or manifest_paths != tuple(sorted(manifest_paths))
            or len(set(manifest_paths)) != len(manifest_paths)
            or not self.tar_manifests
            or len(self.tar_manifests) > POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1.max_tar_manifests
            or sum(item.member_count for item in self.tar_manifests)
            > POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1.max_total_tar_members
        ):
            raise ValueError("postprocessing scientific output root members are unordered, duplicated, or unbounded")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "root_kind": self.root_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "output_root": self.output_root,
            "small_outputs": [item.to_mapping() for item in self.small_outputs],
            "tar_manifests": [item.to_mapping() for item in self.tar_manifests],
        }


@dataclass(frozen=True)
class PostprocessingTarMemberIdentity:
    path: str
    sha256: str
    size_bytes: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _validate_tar_member_path(self.path)
        _sha(self.sha256, "postprocessing tar member SHA-256")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("postprocessing tar member size must be a non-negative integer")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class PostprocessingTarManifest:
    tar_path: str
    tar_size_bytes: int
    stat_device: int
    stat_inode: int
    stat_mtime_ns: int
    members: tuple[PostprocessingTarMemberIdentity, ...]
    manifest_id: str
    manifest_kind: Literal["postprocessing-tar-manifest-v1"] = "postprocessing-tar-manifest-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.manifest_kind != "postprocessing-tar-manifest-v1":
            raise ValueError("unsupported postprocessing tar manifest kind")
        _validate_output_path(self.tar_path)
        for value in (self.tar_size_bytes, self.stat_device, self.stat_inode, self.stat_mtime_ns):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("postprocessing tar stat identities must be non-negative integers")
        paths = tuple(item.path for item in self.members)
        if (
            not paths
            or paths != tuple(sorted(paths))
            or len(set(paths)) != len(paths)
            or len(paths) > POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1.max_members_per_tar_manifest
        ):
            raise ValueError("postprocessing tar members must be nonempty, unique, sorted, and bounded")
        _sha(self.manifest_id, "postprocessing tar manifest id")
        if self.manifest_id != canonical_mapping_digest(self.identity_mapping()):
            raise ValueError("postprocessing tar manifest id differs from its identity preimage")

    def identity_mapping(self) -> dict[str, object]:
        """Return logical member identity without physical stat or tar encoding facts."""
        return {
            "schema_version": self.schema_version,
            "manifest_kind": self.manifest_kind,
            "tar_path": self.tar_path,
            "members": [item.to_mapping() for item in self.members],
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            **self.identity_mapping(),
            "manifest_id": self.manifest_id,
            "tar_size_bytes": self.tar_size_bytes,
            "stat_device": self.stat_device,
            "stat_inode": self.stat_inode,
            "stat_mtime_ns": self.stat_mtime_ns,
        }


def postprocessing_scientific_output_root_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingScientificOutputRoot:
    _fields(
        payload,
        {
            "schema_version",
            "root_kind",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "output_root",
            "small_outputs",
            "tar_manifests",
        },
        "PostprocessingScientificOutputRoot",
    )
    small: list[PostprocessingSmallOutputIdentity] = []
    for item in _mapping_list(payload, "small_outputs"):
        _fields(item, {"path", "sha256", "size_bytes"}, "PostprocessingSmallOutputIdentity")
        small.append(
            PostprocessingSmallOutputIdentity(
                path=_string(item, "path"),
                sha256=_string(item, "sha256"),
                size_bytes=_integer(item, "size_bytes"),
            )
        )
    references: list[PostprocessingTarManifestReference] = []
    for item in _mapping_list(payload, "tar_manifests"):
        _fields(item, {"path", "manifest_id", "member_count"}, "PostprocessingTarManifestReference")
        references.append(
            PostprocessingTarManifestReference(
                path=_string(item, "path"),
                manifest_id=_string(item, "manifest_id"),
                member_count=_integer(item, "member_count"),
            )
        )
    if _string(payload, "root_kind") != "postprocessing-scientific-output-root-v1":
        raise ValueError("unsupported postprocessing scientific root kind")
    return PostprocessingScientificOutputRoot(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingScientificOutputRoot"
        ),
        root_kind="postprocessing-scientific-output-root-v1",
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        output_root=_string(payload, "output_root"),
        small_outputs=tuple(small),
        tar_manifests=tuple(references),
    )


def postprocessing_tar_manifest_from_mapping(payload: Mapping[str, object]) -> PostprocessingTarManifest:
    _fields(
        payload,
        {
            "schema_version",
            "manifest_kind",
            "manifest_id",
            "tar_path",
            "tar_size_bytes",
            "stat_device",
            "stat_inode",
            "stat_mtime_ns",
            "members",
        },
        "PostprocessingTarManifest",
    )
    members = _mapping_list(payload, "members")
    return PostprocessingTarManifest(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PostprocessingTarManifest"),
        manifest_kind=cast("Literal['postprocessing-tar-manifest-v1']", _string(payload, "manifest_kind")),
        manifest_id=_string(payload, "manifest_id"),
        tar_path=_string(payload, "tar_path"),
        tar_size_bytes=_integer(payload, "tar_size_bytes"),
        stat_device=_integer(payload, "stat_device"),
        stat_inode=_integer(payload, "stat_inode"),
        stat_mtime_ns=_integer(payload, "stat_mtime_ns"),
        members=tuple(_tar_member_from_mapping(item) for item in members),
    )


def _tar_member_from_mapping(payload: Mapping[str, object]) -> PostprocessingTarMemberIdentity:
    _fields(payload, {"schema_version", "path", "sha256", "size_bytes"}, "PostprocessingTarMemberIdentity")
    return PostprocessingTarMemberIdentity(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingTarMemberIdentity"
        ),
        path=_string(payload, "path"),
        sha256=_string(payload, "sha256"),
        size_bytes=_integer(payload, "size_bytes"),
    )


def _validate_tar_member_path(value: str) -> None:
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or _CONTROL_CHARACTER.search(value) is not None
        or unicodedata.normalize("NFC", value) != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).as_posix() != value
    ):
        raise ValueError("postprocessing tar member path is unsafe or noncanonical")


def _validate_output_path(value: str) -> None:
    _validate_tar_member_path(value)


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


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


__all__ = [
    "PostprocessingScientificOutputRoot",
    "PostprocessingSmallOutputIdentity",
    "PostprocessingTarManifest",
    "PostprocessingTarManifestReference",
    "PostprocessingTarMemberIdentity",
    "postprocessing_scientific_output_root_from_mapping",
    "postprocessing_tar_manifest_from_mapping",
]
