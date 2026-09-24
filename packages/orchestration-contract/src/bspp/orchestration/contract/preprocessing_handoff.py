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

"""Strict contracts for one finalized preprocessing MSA handoff.

Logical MSA content and its physical bundled location deliberately have
different identities.  The Runtime is the only layer that inspects payload
bytes; Control only reloads and cross-checks these small records.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

ContentValidationOutcome = Literal["passed", "failed"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_ACTION_ID = re.compile(r"preprocessing-chunk-[0-9]{6}")
_CHUNK_NAME = re.compile(r"[^/]+_tranche\d{2}_\d{5}\.fa")
_ARTIFACT_SET_ID = re.compile(r"sha256:[0-9a-f]{64}")
_LOCATION_ID = re.compile(r"artifact-location-[0-9a-f]{64}")
_EVIDENCE_ID = re.compile(r"preprocessing-content-validation-[0-9a-f]{64}")


@dataclass(frozen=True)
class MsaArtifactMember:
    """One logical A3M member, independent of its current storage layout."""

    record_identity: str
    source_ordinal: int
    source_header: str
    logical_path: str
    size_bytes: int
    sha256: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "MsaArtifactMember")
        if not self.record_identity:
            raise ValueError("MSA artifact member record_identity must be non-empty")
        if not isinstance(self.source_ordinal, int) or isinstance(self.source_ordinal, bool) or self.source_ordinal < 0:
            raise ValueError("MSA artifact member source_ordinal must be non-negative")
        header_identity = self.source_header[1:].split(maxsplit=1) if self.source_header.startswith(">") else []
        if not header_identity or header_identity[0] != self.record_identity:
            raise ValueError("MSA artifact member source header must bind its record identity")
        member_name = self.logical_path.removeprefix("a3ms/")
        if (
            not self.logical_path.startswith("a3ms/")
            or not self.logical_path.endswith(".a3m")
            or self.logical_path.count("/") != 1
            or ".." in self.logical_path.split("/")
            or member_name in {".a3m", "..a3m"}
        ):
            raise ValueError("MSA artifact member logical_path must be a stable a3ms/<member>.a3m path")
        _validate_positive_int(self.size_bytes, "MSA artifact member size_bytes")
        _validate_sha256(self.sha256, "MSA artifact member sha256")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "record_identity": self.record_identity,
            "source_ordinal": self.source_ordinal,
            "source_header": self.source_header,
            "logical_path": self.logical_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class MsaChunkManifest:
    """Deterministic logical inventory for one preprocessing chunk."""

    chunk_name: str
    members: tuple[MsaArtifactMember, ...]
    member_count: int
    logical_bytes: int
    artifact_type: str = "bspp.msa-chunk/v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "MsaChunkManifest")
        if self.artifact_type not in ("bspp.msa-chunk/v1", "afcdb.msa-chunk/v1"):
            raise ValueError("unsupported MSA chunk manifest artifact_type")
        if _CHUNK_NAME.fullmatch(self.chunk_name) is None:
            raise ValueError("MSA chunk manifest has an invalid chunk name")
        if not isinstance(self.members, tuple) or not self.members:
            raise ValueError("MSA chunk manifest members must be a non-empty immutable tuple")
        ordinals = tuple(member.source_ordinal for member in self.members)
        if ordinals != tuple(sorted(ordinals)) or len(set(ordinals)) != len(ordinals):
            raise ValueError("MSA chunk manifest members must be uniquely ordered by source_ordinal")
        if len({member.logical_path for member in self.members}) != len(self.members):
            raise ValueError("MSA chunk manifest logical paths must be unique")
        if self.member_count != len(self.members):
            raise ValueError("MSA chunk manifest member_count must match its exact inventory")
        if self.logical_bytes != sum(member.size_bytes for member in self.members):
            raise ValueError("MSA chunk manifest logical_bytes must match its exact inventory")

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "chunk_name": self.chunk_name,
            "members": [member.to_mapping() for member in self.members],
            "member_count": self.member_count,
            "logical_bytes": self.logical_bytes,
        }


@dataclass(frozen=True)
class MsaChunkManifestReference:
    """One root-manifest reference to an immutable chunk manifest."""

    chunk_name: str
    logical_path: str
    sha256: str
    member_count: int
    logical_bytes: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "MsaChunkManifestReference")
        if _CHUNK_NAME.fullmatch(self.chunk_name) is None:
            raise ValueError("MSA chunk reference has an invalid chunk name")
        expected = f"chunks/{self.chunk_name.removesuffix('.fa')}.json"
        if self.logical_path != expected:
            raise ValueError("MSA chunk reference must use its deterministic logical path")
        _validate_sha256(self.sha256, "MSA chunk manifest reference sha256")
        _validate_positive_int(self.member_count, "MSA chunk reference member_count")
        _validate_positive_int(self.logical_bytes, "MSA chunk reference logical_bytes")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "chunk_name": self.chunk_name,
            "logical_path": self.logical_path,
            "sha256": self.sha256,
            "member_count": self.member_count,
            "logical_bytes": self.logical_bytes,
        }


@dataclass(frozen=True)
class MsaArtifactSetManifest:
    """Content-identified root manifest for a logical MSA Artifact Set."""

    artifact_set_id: str
    chunks: tuple[MsaChunkManifestReference, ...]
    member_count: int
    logical_bytes: int
    member_lengths: tuple[int, ...] | None = None
    artifact_type: str = "bspp.msa-set/v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "MsaArtifactSetManifest")
        if self.artifact_type not in ("bspp.msa-set/v1", "afcdb.msa-set/v1"):
            raise ValueError("unsupported MSA Artifact Set type")
        if not isinstance(self.chunks, tuple) or len(self.chunks) != 1:
            raise ValueError("the one-chunk slice requires exactly one chunk manifest reference")
        if self.member_count != sum(chunk.member_count for chunk in self.chunks):
            raise ValueError("MSA Artifact Set member_count must match its chunk references")
        if self.logical_bytes != sum(chunk.logical_bytes for chunk in self.chunks):
            raise ValueError("MSA Artifact Set logical_bytes must match its chunk references")
        if self.member_lengths is not None:
            if not isinstance(self.member_lengths, tuple) or any(
                not isinstance(length, int) or isinstance(length, bool) or length <= 0 for length in self.member_lengths
            ):
                raise ValueError("MSA Artifact Set member_lengths must be a tuple of positive integers")
            if len(self.member_lengths) != self.member_count:
                raise ValueError("MSA Artifact Set member_lengths count must match member_count")
        if self.artifact_set_id not in (
            msa_artifact_set_id(self.chunks, self.member_count, self.logical_bytes, member_lengths=self.member_lengths),
            msa_artifact_set_id(
                self.chunks, self.member_count, self.logical_bytes, member_lengths=self.member_lengths, legacy=True
            ),
        ):
            raise ValueError("MSA Artifact Set id does not match canonical logical content")

    def has_member_lengths(self) -> bool:
        """Return True when this manifest carries producer-attested member lengths."""
        return self.member_lengths is not None

    def identity_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "chunks": [chunk.to_mapping() for chunk in self.chunks],
            "member_count": self.member_count,
            "logical_bytes": self.logical_bytes,
        }
        if self.member_lengths is not None:
            result["member_lengths"] = list(self.member_lengths)
        return result

    def to_mapping(self) -> dict[str, object]:
        return {"artifact_set_id": self.artifact_set_id, **self.identity_mapping()}


def msa_artifact_set_id(
    chunks: tuple[MsaChunkManifestReference, ...],
    member_count: int,
    logical_bytes: int,
    *,
    member_lengths: tuple[int, ...] | None = None,
    legacy: bool = False,
) -> str:
    artifact_type = "afcdb.msa-set/v1" if legacy else "bspp.msa-set/v1"
    payload: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "chunks": [chunk.to_mapping() for chunk in chunks],
        "member_count": member_count,
        "logical_bytes": logical_bytes,
    }
    if member_lengths is not None:
        payload["member_lengths"] = list(member_lengths)
    return f"sha256:{canonical_mapping_digest(payload)}"


@dataclass(frozen=True)
class BundledMemberVerification:
    """Verified mapping from one logical A3M to one exact tar member."""

    logical_path: str
    member_name: str
    raw_member_name: str
    size_bytes: int
    sha256: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "BundledMemberVerification")
        if self.logical_path != f"a3ms/{self.member_name}":
            raise ValueError("bundled member logical path must match its top-level A3M member")
        if not self.member_name.endswith(".a3m") or "/" in self.member_name:
            raise ValueError("bundled member must be a top-level A3M")
        if _normalize_tar_member(self.raw_member_name) != self.member_name:
            raise ValueError("bundled member raw name must normalize to its declared A3M member")
        _validate_positive_int(self.size_bytes, "bundled member size_bytes")
        _validate_sha256(self.sha256, "bundled member sha256")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "logical_path": self.logical_path,
            "member_name": self.member_name,
            "raw_member_name": self.raw_member_name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class VerifiedLocalBundledArtifactLocation:
    """One verified local tar/LZ4 representation of an MSA Artifact Set."""

    artifact_location_id: str
    artifact_set_id: str
    tar_path: str
    bundle_path: str
    bundle_uri: str
    tar_size_bytes: int
    tar_sha256: str
    lz4_size_bytes: int
    lz4_sha256: str
    raw_tar_members: tuple[str, ...]
    members: tuple[BundledMemberVerification, ...]
    verified_at: str
    kind: str = "verified-local-bundled"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "VerifiedLocalBundledArtifactLocation")
        if self.kind != "verified-local-bundled":
            raise ValueError("unsupported MSA Artifact Location kind")
        if _ARTIFACT_SET_ID.fullmatch(self.artifact_set_id) is None:
            raise ValueError("MSA Artifact Location has an invalid Artifact Set id")
        if not Path(self.tar_path).is_absolute() or not Path(self.bundle_path).is_absolute():
            raise ValueError("MSA Artifact Location paths must be absolute")
        if self.bundle_uri != Path(self.bundle_path).as_uri():
            raise ValueError("MSA Artifact Location requires an exact local bundle URI")
        _validate_positive_int(self.tar_size_bytes, "MSA Artifact Location tar_size_bytes")
        _validate_positive_int(self.lz4_size_bytes, "MSA Artifact Location lz4_size_bytes")
        _validate_sha256(self.tar_sha256, "MSA Artifact Location tar_sha256")
        _validate_sha256(self.lz4_sha256, "MSA Artifact Location lz4_sha256")
        if (
            not isinstance(self.raw_tar_members, tuple)
            or not isinstance(self.members, tuple)
            or not self.raw_tar_members
            or not self.members
            or any(not isinstance(name, str) or not name for name in self.raw_tar_members)
            or any(not isinstance(member, BundledMemberVerification) for member in self.members)
        ):
            raise ValueError("MSA Artifact Location requires complete tar and logical member inventories")
        if len({member.member_name for member in self.members}) != len(self.members):
            raise ValueError("MSA Artifact Location member mappings must be unique")
        raw_files = tuple(name for name in self.raw_tar_members if _normalize_tar_member(name) != ".")
        mapped_raw = tuple(member.raw_member_name for member in self.members)
        if len(raw_files) != len(mapped_raw) or set(raw_files) != set(mapped_raw):
            raise ValueError("MSA Artifact Location raw tar names must match its logical member mapping")
        _validate_timestamp(self.verified_at, "MSA Artifact Location verified_at")
        if self.artifact_location_id != verified_local_bundled_artifact_location_id(
            artifact_set_id=self.artifact_set_id,
            tar_path=self.tar_path,
            bundle_path=self.bundle_path,
            bundle_uri=self.bundle_uri,
            tar_size_bytes=self.tar_size_bytes,
            tar_sha256=self.tar_sha256,
            lz4_size_bytes=self.lz4_size_bytes,
            lz4_sha256=self.lz4_sha256,
            raw_tar_members=self.raw_tar_members,
            members=self.members,
        ):
            raise ValueError("MSA Artifact Location id does not match its physical representation")

    def identity_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "artifact_set_id": self.artifact_set_id,
            "tar_path": self.tar_path,
            "bundle_path": self.bundle_path,
            "bundle_uri": self.bundle_uri,
            "tar_size_bytes": self.tar_size_bytes,
            "tar_sha256": self.tar_sha256,
            "lz4_size_bytes": self.lz4_size_bytes,
            "lz4_sha256": self.lz4_sha256,
            "raw_tar_members": list(self.raw_tar_members),
            "members": [member.to_mapping() for member in self.members],
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifact_location_id": self.artifact_location_id,
            **self.identity_mapping(),
            "verified_at": self.verified_at,
        }


def verified_local_bundled_artifact_location_id(
    *,
    artifact_set_id: str,
    tar_path: str,
    bundle_path: str,
    bundle_uri: str,
    tar_size_bytes: int,
    tar_sha256: str,
    lz4_size_bytes: int,
    lz4_sha256: str,
    raw_tar_members: tuple[str, ...],
    members: tuple[BundledMemberVerification, ...],
) -> str:
    payload: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "kind": "verified-local-bundled",
        "artifact_set_id": artifact_set_id,
        "tar_path": tar_path,
        "bundle_path": bundle_path,
        "bundle_uri": bundle_uri,
        "tar_size_bytes": tar_size_bytes,
        "tar_sha256": tar_sha256,
        "lz4_size_bytes": lz4_size_bytes,
        "lz4_sha256": lz4_sha256,
        "raw_tar_members": list(raw_tar_members),
        "members": [member.to_mapping() for member in members],
    }
    return f"artifact-location-{canonical_mapping_digest(payload)}"


@dataclass(frozen=True)
class VerifiedRemoteBundledArtifactLocation:
    """One verified remote LZ4 representation of an MSA Artifact Set.

    The remote record carries only the remote-capable ``bundle_uri`` and the
    content identities; local filesystem paths are materialized later by the
    download-verify-project leg and are never part of this record.
    """

    artifact_location_id: str
    artifact_set_id: str
    bundle_uri: str
    tar_size_bytes: int
    tar_sha256: str
    lz4_size_bytes: int
    lz4_sha256: str
    raw_tar_members: tuple[str, ...]
    members: tuple[BundledMemberVerification, ...]
    verified_at: str
    kind: str = "verified-remote-bundled"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "VerifiedRemoteBundledArtifactLocation")
        if self.kind != "verified-remote-bundled":
            raise ValueError("unsupported MSA Artifact Location kind")
        if _ARTIFACT_SET_ID.fullmatch(self.artifact_set_id) is None:
            raise ValueError("MSA Artifact Location has an invalid Artifact Set id")
        _validate_remote_bundle_uri(self.bundle_uri)
        _validate_positive_int(self.tar_size_bytes, "MSA Artifact Location tar_size_bytes")
        _validate_positive_int(self.lz4_size_bytes, "MSA Artifact Location lz4_size_bytes")
        _validate_sha256(self.tar_sha256, "MSA Artifact Location tar_sha256")
        _validate_sha256(self.lz4_sha256, "MSA Artifact Location lz4_sha256")
        if (
            not isinstance(self.raw_tar_members, tuple)
            or not isinstance(self.members, tuple)
            or not self.raw_tar_members
            or not self.members
            or any(not isinstance(name, str) or not name for name in self.raw_tar_members)
            or any(not isinstance(member, BundledMemberVerification) for member in self.members)
        ):
            raise ValueError("MSA Artifact Location requires complete tar and logical member inventories")
        if len({member.member_name for member in self.members}) != len(self.members):
            raise ValueError("MSA Artifact Location member mappings must be unique")
        raw_files = tuple(name for name in self.raw_tar_members if _normalize_tar_member(name) != ".")
        mapped_raw = tuple(member.raw_member_name for member in self.members)
        if len(raw_files) != len(mapped_raw) or set(raw_files) != set(mapped_raw):
            raise ValueError("MSA Artifact Location raw tar names must match its logical member mapping")
        _validate_timestamp(self.verified_at, "MSA Artifact Location verified_at")
        if self.artifact_location_id != verified_remote_bundled_artifact_location_id(
            artifact_set_id=self.artifact_set_id,
            bundle_uri=self.bundle_uri,
            tar_size_bytes=self.tar_size_bytes,
            tar_sha256=self.tar_sha256,
            lz4_size_bytes=self.lz4_size_bytes,
            lz4_sha256=self.lz4_sha256,
            raw_tar_members=self.raw_tar_members,
            members=self.members,
        ):
            raise ValueError("MSA Artifact Location id does not match its physical representation")

    def identity_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "artifact_set_id": self.artifact_set_id,
            "bundle_uri": self.bundle_uri,
            "tar_size_bytes": self.tar_size_bytes,
            "tar_sha256": self.tar_sha256,
            "lz4_size_bytes": self.lz4_size_bytes,
            "lz4_sha256": self.lz4_sha256,
            "raw_tar_members": list(self.raw_tar_members),
            "members": [member.to_mapping() for member in self.members],
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifact_location_id": self.artifact_location_id,
            **self.identity_mapping(),
            "verified_at": self.verified_at,
        }


def verified_remote_bundled_artifact_location_id(
    *,
    artifact_set_id: str,
    bundle_uri: str,
    tar_size_bytes: int,
    tar_sha256: str,
    lz4_size_bytes: int,
    lz4_sha256: str,
    raw_tar_members: tuple[str, ...],
    members: tuple[BundledMemberVerification, ...],
) -> str:
    payload: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "kind": "verified-remote-bundled",
        "artifact_set_id": artifact_set_id,
        "bundle_uri": bundle_uri,
        "tar_size_bytes": tar_size_bytes,
        "tar_sha256": tar_sha256,
        "lz4_size_bytes": lz4_size_bytes,
        "lz4_sha256": lz4_sha256,
        "raw_tar_members": list(raw_tar_members),
        "members": [member.to_mapping() for member in members],
    }
    return f"artifact-location-{canonical_mapping_digest(payload)}"


@dataclass(frozen=True)
class PreprocessingContentValidationEvidence:
    """Successful or failed Runtime content-validation evidence."""

    evidence_id: str
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_id: str
    action_evidence_digest: str
    chunk_name: str
    started_at: str
    finished_at: str
    outcome: ContentValidationOutcome
    artifact_set_id: str | None
    artifact_location_id: str | None
    chunk_manifest_digest: str | None
    artifact_set_manifest_digest: str | None
    member_count: int | None
    logical_bytes: int | None
    lz4_command_digest: str | None
    lz4_return_code: int | None
    decompressed_tar_size_bytes: int | None
    decompressed_tar_sha256: str | None
    error: str | None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PreprocessingContentValidationEvidence")
        if _PHASE_RUN_ID.fullmatch(self.phase_run_id) is None or _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise ValueError("content-validation evidence has invalid run/attempt identity")
        if _ACTION_ID.fullmatch(self.action_id) is None or _CHUNK_NAME.fullmatch(self.chunk_name) is None:
            raise ValueError("content-validation evidence has invalid action/chunk identity")
        _validate_sha256(self.phase_runspec_digest, "content-validation RunSpec digest")
        _validate_sha256(self.action_evidence_digest, "content-validation action evidence digest")
        _validate_timestamp(self.started_at, "content-validation started_at")
        _validate_timestamp(self.finished_at, "content-validation finished_at")
        if _parse_timestamp(self.finished_at) < _parse_timestamp(self.started_at):
            raise ValueError("content validation cannot finish before it starts")
        success_values = (
            self.artifact_set_id,
            self.artifact_location_id,
            self.chunk_manifest_digest,
            self.artifact_set_manifest_digest,
            self.member_count,
            self.logical_bytes,
            self.lz4_command_digest,
            self.lz4_return_code,
            self.decompressed_tar_size_bytes,
            self.decompressed_tar_sha256,
        )
        if self.outcome == "passed":
            if any(value is None for value in success_values) or self.error is not None:
                raise ValueError("passed content validation requires complete success evidence and no error")
            if self.lz4_return_code != 0:
                raise ValueError("passed content validation requires zero-returning LZ4 verification")
            assert self.artifact_set_id is not None
            assert self.artifact_location_id is not None
            if _ARTIFACT_SET_ID.fullmatch(self.artifact_set_id) is None:
                raise ValueError("content validation has an invalid Artifact Set id")
            if _LOCATION_ID.fullmatch(self.artifact_location_id) is None:
                raise ValueError("content validation has an invalid Artifact Location id")
            for label, digest in (
                ("chunk manifest", self.chunk_manifest_digest),
                ("Artifact Set manifest", self.artifact_set_manifest_digest),
                ("LZ4 command", self.lz4_command_digest),
                ("decompressed tar", self.decompressed_tar_sha256),
            ):
                assert digest is not None
                _validate_sha256(digest, f"content-validation {label} digest")
            assert self.member_count is not None
            assert self.logical_bytes is not None
            assert self.decompressed_tar_size_bytes is not None
            _validate_positive_int(self.member_count, "content-validation member_count")
            _validate_positive_int(self.logical_bytes, "content-validation logical_bytes")
            _validate_positive_int(self.decompressed_tar_size_bytes, "content-validation decompressed tar size")
        elif self.outcome == "failed":
            if any(value is not None for value in success_values) or not self.error:
                raise ValueError("failed content validation may contain only failure evidence")
        else:
            raise ValueError(f"unsupported content-validation outcome: {self.outcome!r}")
        if self.evidence_id != preprocessing_content_validation_evidence_id(self.identity_mapping()):
            raise ValueError("content-validation evidence id does not match canonical evidence")

    def identity_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_id": self.action_id,
            "action_evidence_digest": self.action_evidence_digest,
            "chunk_name": self.chunk_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outcome": self.outcome,
            "artifact_set_id": self.artifact_set_id,
            "artifact_location_id": self.artifact_location_id,
            "chunk_manifest_digest": self.chunk_manifest_digest,
            "artifact_set_manifest_digest": self.artifact_set_manifest_digest,
            "member_count": self.member_count,
            "logical_bytes": self.logical_bytes,
            "lz4_command_digest": self.lz4_command_digest,
            "lz4_return_code": self.lz4_return_code,
            "decompressed_tar_size_bytes": self.decompressed_tar_size_bytes,
            "decompressed_tar_sha256": self.decompressed_tar_sha256,
            "error": self.error,
        }

    def to_mapping(self) -> dict[str, object]:
        return {"evidence_id": self.evidence_id, **self.identity_mapping()}


def preprocessing_content_validation_evidence_id(payload: Mapping[str, object]) -> str:
    return f"preprocessing-content-validation-{canonical_mapping_digest(payload)}"


@dataclass(frozen=True)
class PreprocessingHandoffBundle:
    """Complete successful one-chunk handoff loaded by Control."""

    chunk_manifest: MsaChunkManifest
    artifact_set: MsaArtifactSetManifest
    artifact_location: VerifiedLocalBundledArtifactLocation
    content_validation: PreprocessingContentValidationEvidence

    def __post_init__(self) -> None:
        reference = self.artifact_set.chunks[0]
        if reference.chunk_name != self.chunk_manifest.chunk_name or reference.sha256 != self.chunk_manifest.digest:
            raise ValueError("Artifact Set root does not reference the exact chunk manifest")
        if self.artifact_location.artifact_set_id != self.artifact_set.artifact_set_id:
            raise ValueError("Artifact Location does not reference the exact Artifact Set")
        validation = self.content_validation
        if validation.outcome != "passed":
            raise ValueError("a successful preprocessing handoff requires passed content validation")
        if (
            validation.artifact_set_id != self.artifact_set.artifact_set_id
            or validation.artifact_location_id != self.artifact_location.artifact_location_id
            or validation.chunk_manifest_digest != self.chunk_manifest.digest
            or validation.artifact_set_manifest_digest != canonical_mapping_digest(self.artifact_set.to_mapping())
            or validation.member_count != self.artifact_set.member_count
            or validation.logical_bytes != self.artifact_set.logical_bytes
        ):
            raise ValueError("content validation does not bind the exact successful handoff")

    def to_mapping(self) -> dict[str, object]:
        return {
            "chunk_manifest": self.chunk_manifest.to_mapping(),
            "artifact_set": self.artifact_set.to_mapping(),
            "artifact_location": self.artifact_location.to_mapping(),
            "content_validation": self.content_validation.to_mapping(),
        }


def msa_artifact_member_from_mapping(payload: Mapping[str, object]) -> MsaArtifactMember:
    _strict(
        payload,
        {
            "schema_version",
            "record_identity",
            "source_ordinal",
            "source_header",
            "logical_path",
            "size_bytes",
            "sha256",
        },
        "MsaArtifactMember",
    )
    return MsaArtifactMember(
        schema_version=_schema(payload, "MsaArtifactMember"),
        record_identity=_str(payload, "record_identity"),
        source_ordinal=_int(payload, "source_ordinal"),
        source_header=_str(payload, "source_header"),
        logical_path=_str(payload, "logical_path"),
        size_bytes=_int(payload, "size_bytes"),
        sha256=_str(payload, "sha256"),
    )


def msa_chunk_manifest_from_mapping(payload: Mapping[str, object]) -> MsaChunkManifest:
    _strict(
        payload,
        {"schema_version", "artifact_type", "chunk_name", "members", "member_count", "logical_bytes"},
        "MsaChunkManifest",
    )
    return MsaChunkManifest(
        schema_version=_schema(payload, "MsaChunkManifest"),
        artifact_type=_str(payload, "artifact_type"),
        chunk_name=_str(payload, "chunk_name"),
        members=tuple(msa_artifact_member_from_mapping(item) for item in _mappings(payload, "members")),
        member_count=_int(payload, "member_count"),
        logical_bytes=_int(payload, "logical_bytes"),
    )


def msa_chunk_manifest_reference_from_mapping(payload: Mapping[str, object]) -> MsaChunkManifestReference:
    _strict(
        payload,
        {"schema_version", "chunk_name", "logical_path", "sha256", "member_count", "logical_bytes"},
        "MsaChunkManifestReference",
    )
    return MsaChunkManifestReference(
        schema_version=_schema(payload, "MsaChunkManifestReference"),
        chunk_name=_str(payload, "chunk_name"),
        logical_path=_str(payload, "logical_path"),
        sha256=_str(payload, "sha256"),
        member_count=_int(payload, "member_count"),
        logical_bytes=_int(payload, "logical_bytes"),
    )


def msa_artifact_set_manifest_from_mapping(payload: Mapping[str, object]) -> MsaArtifactSetManifest:
    _strict(
        payload,
        {
            "schema_version",
            "artifact_type",
            "artifact_set_id",
            "chunks",
            "member_count",
            "logical_bytes",
            "member_lengths",
        },
        "MsaArtifactSetManifest",
    )
    return MsaArtifactSetManifest(
        schema_version=_schema(payload, "MsaArtifactSetManifest"),
        artifact_type=_str(payload, "artifact_type"),
        artifact_set_id=_str(payload, "artifact_set_id"),
        chunks=tuple(msa_chunk_manifest_reference_from_mapping(item) for item in _mappings(payload, "chunks")),
        member_count=_int(payload, "member_count"),
        logical_bytes=_int(payload, "logical_bytes"),
        member_lengths=_optional_int_tuple(payload, "member_lengths"),
    )


def bundled_member_verification_from_mapping(payload: Mapping[str, object]) -> BundledMemberVerification:
    _strict(
        payload,
        {"schema_version", "logical_path", "member_name", "raw_member_name", "size_bytes", "sha256"},
        "BundledMemberVerification",
    )
    return BundledMemberVerification(
        schema_version=_schema(payload, "BundledMemberVerification"),
        logical_path=_str(payload, "logical_path"),
        member_name=_str(payload, "member_name"),
        raw_member_name=_str(payload, "raw_member_name"),
        size_bytes=_int(payload, "size_bytes"),
        sha256=_str(payload, "sha256"),
    )


def verified_local_bundled_artifact_location_from_mapping(
    payload: Mapping[str, object],
) -> VerifiedLocalBundledArtifactLocation:
    _strict(
        payload,
        {
            "schema_version",
            "kind",
            "artifact_location_id",
            "artifact_set_id",
            "tar_path",
            "bundle_path",
            "bundle_uri",
            "tar_size_bytes",
            "tar_sha256",
            "lz4_size_bytes",
            "lz4_sha256",
            "raw_tar_members",
            "members",
            "verified_at",
        },
        "VerifiedLocalBundledArtifactLocation",
    )
    return VerifiedLocalBundledArtifactLocation(
        schema_version=_schema(payload, "VerifiedLocalBundledArtifactLocation"),
        kind=_str(payload, "kind"),
        artifact_location_id=_str(payload, "artifact_location_id"),
        artifact_set_id=_str(payload, "artifact_set_id"),
        tar_path=_str(payload, "tar_path"),
        bundle_path=_str(payload, "bundle_path"),
        bundle_uri=_str(payload, "bundle_uri"),
        tar_size_bytes=_int(payload, "tar_size_bytes"),
        tar_sha256=_str(payload, "tar_sha256"),
        lz4_size_bytes=_int(payload, "lz4_size_bytes"),
        lz4_sha256=_str(payload, "lz4_sha256"),
        raw_tar_members=_strings(payload, "raw_tar_members"),
        members=tuple(bundled_member_verification_from_mapping(item) for item in _mappings(payload, "members")),
        verified_at=_str(payload, "verified_at"),
    )


def verified_remote_bundled_artifact_location_from_mapping(
    payload: Mapping[str, object],
) -> VerifiedRemoteBundledArtifactLocation:
    _strict(
        payload,
        {
            "schema_version",
            "kind",
            "artifact_location_id",
            "artifact_set_id",
            "bundle_uri",
            "tar_size_bytes",
            "tar_sha256",
            "lz4_size_bytes",
            "lz4_sha256",
            "raw_tar_members",
            "members",
            "verified_at",
        },
        "VerifiedRemoteBundledArtifactLocation",
    )
    return VerifiedRemoteBundledArtifactLocation(
        schema_version=_schema(payload, "VerifiedRemoteBundledArtifactLocation"),
        kind=_str(payload, "kind"),
        artifact_location_id=_str(payload, "artifact_location_id"),
        artifact_set_id=_str(payload, "artifact_set_id"),
        bundle_uri=_str(payload, "bundle_uri"),
        tar_size_bytes=_int(payload, "tar_size_bytes"),
        tar_sha256=_str(payload, "tar_sha256"),
        lz4_size_bytes=_int(payload, "lz4_size_bytes"),
        lz4_sha256=_str(payload, "lz4_sha256"),
        raw_tar_members=_strings(payload, "raw_tar_members"),
        members=tuple(bundled_member_verification_from_mapping(item) for item in _mappings(payload, "members")),
        verified_at=_str(payload, "verified_at"),
    )


def preprocessing_content_validation_evidence_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingContentValidationEvidence:
    allowed = {
        "schema_version",
        "evidence_id",
        "phase_run_id",
        "attempt_id",
        "phase_runspec_digest",
        "action_id",
        "action_evidence_digest",
        "chunk_name",
        "started_at",
        "finished_at",
        "outcome",
        "artifact_set_id",
        "artifact_location_id",
        "chunk_manifest_digest",
        "artifact_set_manifest_digest",
        "member_count",
        "logical_bytes",
        "lz4_command_digest",
        "lz4_return_code",
        "decompressed_tar_size_bytes",
        "decompressed_tar_sha256",
        "error",
    }
    _strict(payload, allowed, "PreprocessingContentValidationEvidence")
    return PreprocessingContentValidationEvidence(
        schema_version=_schema(payload, "PreprocessingContentValidationEvidence"),
        evidence_id=_str(payload, "evidence_id"),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        phase_runspec_digest=_str(payload, "phase_runspec_digest"),
        action_id=_str(payload, "action_id"),
        action_evidence_digest=_str(payload, "action_evidence_digest"),
        chunk_name=_str(payload, "chunk_name"),
        started_at=_str(payload, "started_at"),
        finished_at=_str(payload, "finished_at"),
        outcome=cast("ContentValidationOutcome", _str(payload, "outcome")),
        artifact_set_id=_optional_str(payload, "artifact_set_id"),
        artifact_location_id=_optional_str(payload, "artifact_location_id"),
        chunk_manifest_digest=_optional_str(payload, "chunk_manifest_digest"),
        artifact_set_manifest_digest=_optional_str(payload, "artifact_set_manifest_digest"),
        member_count=_optional_int(payload, "member_count"),
        logical_bytes=_optional_int(payload, "logical_bytes"),
        lz4_command_digest=_optional_str(payload, "lz4_command_digest"),
        lz4_return_code=_optional_int(payload, "lz4_return_code"),
        decompressed_tar_size_bytes=_optional_int(payload, "decompressed_tar_size_bytes"),
        decompressed_tar_sha256=_optional_str(payload, "decompressed_tar_sha256"),
        error=_optional_str(payload, "error"),
    )


def _strict(payload: Mapping[str, object], allowed: set[str], name: str) -> None:
    if "schema_version" not in payload:
        raise ValueError(f"missing explicit schema_version at {name}")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")


def _schema(payload: Mapping[str, object], name: str) -> int:
    return validate_schema_version(payload.get("schema_version"), record_name=name)


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


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be null or an integer")
    return value


def _optional_int_tuple(payload: Mapping[str, object], key: str) -> tuple[int, ...] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, list | tuple) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{key} must be null or a list of integers")
    return cast("tuple[int, ...]", tuple(value))


def _mappings(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        raise ValueError(f"{key} must be a list")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must contain mappings")
    return cast("tuple[Mapping[str, object], ...]", tuple(value))


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


def _validate_remote_bundle_uri(value: str) -> None:
    if not value.startswith("s3://") or value == "s3://":
        raise ValueError("MSA Artifact Location bundle_uri must be a non-empty s3:// object key")


def _normalize_tar_member(value: str) -> str:
    while value.startswith("./"):
        value = value[2:]
    return value or "."


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")


def _validate_timestamp(value: str, name: str) -> None:
    if not value.endswith("Z"):
        raise ValueError(f"{name} must be an RFC 3339 UTC timestamp")
    try:
        parsed = _parse_timestamp(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an RFC 3339 UTC timestamp") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{name} must be an RFC 3339 UTC timestamp")


__all__ = [
    "BundledMemberVerification",
    "ContentValidationOutcome",
    "MsaArtifactMember",
    "MsaArtifactSetManifest",
    "MsaChunkManifest",
    "MsaChunkManifestReference",
    "PreprocessingContentValidationEvidence",
    "PreprocessingHandoffBundle",
    "VerifiedLocalBundledArtifactLocation",
    "VerifiedRemoteBundledArtifactLocation",
    "bundled_member_verification_from_mapping",
    "msa_artifact_member_from_mapping",
    "msa_artifact_set_id",
    "msa_artifact_set_manifest_from_mapping",
    "msa_chunk_manifest_from_mapping",
    "msa_chunk_manifest_reference_from_mapping",
    "preprocessing_content_validation_evidence_from_mapping",
    "preprocessing_content_validation_evidence_id",
    "verified_local_bundled_artifact_location_from_mapping",
    "verified_local_bundled_artifact_location_id",
    "verified_remote_bundled_artifact_location_from_mapping",
    "verified_remote_bundled_artifact_location_id",
]
