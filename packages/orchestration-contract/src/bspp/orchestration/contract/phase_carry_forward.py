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

"""Strict immutable provenance for verified Attempt carry-forward."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal, cast

from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

CarryContentKind = Literal["preprocessing-a3m"]
CarrySourcePathMode = Literal["predecessor-workspace", "frozen-mount-inverse"]
CarryAdoptionResult = Literal["passed"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_ACTION_ID = re.compile(r"preprocessing-chunk-[0-9]{6}")
_CARRY_ID = re.compile(r"attempt-carry-forward-[0-9a-f]{64}")
_ATTESTATION_ID = re.compile(r"phase-action-evidence-attestation-[0-9a-f]{64}")
_SUBMISSION_ID = re.compile(r"phase-submission-[0-9a-f]{64}")


def _digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttemptCarryForwardContent:
    """One exact verified A3M copied from a predecessor Attempt."""

    source_action_id: str
    target_action_id: str
    member_name: str
    source_ordinal: int
    record_identity: str
    source_header: str
    source_declared_path: str
    target_declared_path: str
    source_physical_path: str
    target_physical_path: str
    source_private_mount_path: str
    size_bytes: int
    sha256: str
    content_kind: CarryContentKind = "preprocessing-a3m"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.content_kind != "preprocessing-a3m":
            raise ValueError("carry-forward content kind must be preprocessing-a3m")
        _match(self.source_action_id, _ACTION_ID, "source action id")
        _match(self.target_action_id, _ACTION_ID, "target action id")
        if self.source_action_id != self.target_action_id:
            raise ValueError("carry-forward source and target action ids must match")
        if not self.member_name.endswith(".a3m") or "/" in self.member_name:
            raise ValueError("carry-forward member_name must be a top-level A3M member")
        if not isinstance(self.source_ordinal, int) or isinstance(self.source_ordinal, bool) or self.source_ordinal < 0:
            raise ValueError("carry-forward source_ordinal must be non-negative")
        if not self.record_identity or not self.source_header:
            raise ValueError("carry-forward logical record identity/header must be non-empty")
        for label, value in (
            ("source declared path", self.source_declared_path),
            ("target declared path", self.target_declared_path),
            ("source physical path", self.source_physical_path),
            ("target physical path", self.target_physical_path),
            ("private source mount path", self.source_private_mount_path),
        ):
            _absolute(value, label)
        if PurePosixPath(self.source_declared_path).name != self.member_name:
            raise ValueError("carry-forward source declared path must end with member_name")
        if PurePosixPath(self.target_declared_path).name != self.member_name:
            raise ValueError("carry-forward target declared path must end with member_name")
        if PurePosixPath(self.source_private_mount_path).name != self.member_name:
            raise ValueError("private source mount path must end with member_name")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes <= 0:
            raise ValueError("carry-forward size_bytes must be positive")
        _sha(self.sha256, "carry-forward content")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "content_kind": self.content_kind,
            "source_action_id": self.source_action_id,
            "target_action_id": self.target_action_id,
            "member_name": self.member_name,
            "source_ordinal": self.source_ordinal,
            "record_identity": self.record_identity,
            "source_header": self.source_header,
            "source_declared_path": self.source_declared_path,
            "target_declared_path": self.target_declared_path,
            "source_physical_path": self.source_physical_path,
            "target_physical_path": self.target_physical_path,
            "source_private_mount_path": self.source_private_mount_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class AttemptWorkspaceRootBinding:
    """One mutable logical root backed by one target-Attempt host directory."""

    role: str
    logical_root: str
    physical_root: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not self.role or "/" in self.role:
            raise ValueError("workspace root role must be a non-empty token")
        _absolute(self.logical_root, "workspace logical root")
        _absolute(self.physical_root, "workspace physical root")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "logical_root": self.logical_root,
            "physical_root": self.physical_root,
        }


@dataclass(frozen=True)
class AttemptWorkspaceBinding:
    """Exact host/container workspace projection for a carried target Attempt."""

    phase_run_id: str
    target_attempt_id: str
    target_action_id: str
    workspace_root: str
    roots: tuple[AttemptWorkspaceRootBinding, ...]
    search_input_physical_path: str
    split_input_physical_path: str
    identity_sentinel_path: str
    private_workspace_mount_path: str
    source_path_mode: CarrySourcePathMode
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.phase_run_id, _PHASE_RUN_ID, "workspace Phase Run id")
        _match(self.target_attempt_id, _ATTEMPT_ID, "workspace target Attempt id")
        _match(self.target_action_id, _ACTION_ID, "workspace target action id")
        _absolute(self.workspace_root, "workspace root")
        if not isinstance(self.roots, tuple) or not self.roots:
            raise ValueError("workspace roots must be a non-empty immutable tuple")
        if any(not isinstance(item, AttemptWorkspaceRootBinding) for item in self.roots):
            raise ValueError("workspace roots must contain AttemptWorkspaceRootBinding records")
        roles = tuple(item.role for item in self.roots)
        logical = tuple(item.logical_root for item in self.roots)
        physical = tuple(item.physical_root for item in self.roots)
        if len(set(roles)) != len(roles) or len(set(logical)) != len(logical) or len(set(physical)) != len(physical):
            raise ValueError("workspace root roles and paths must be unique")
        root = PurePosixPath(self.workspace_root)
        for item in self.roots:
            _within(PurePosixPath(item.physical_root), root, "workspace physical root")
        for label, value in (
            ("search input physical path", self.search_input_physical_path),
            ("split input physical path", self.split_input_physical_path),
            ("identity sentinel path", self.identity_sentinel_path),
            ("private workspace mount path", self.private_workspace_mount_path),
        ):
            _absolute(value, label)
        _within(PurePosixPath(self.search_input_physical_path), root, "search input")
        _within(PurePosixPath(self.split_input_physical_path), root, "split input")
        _within(PurePosixPath(self.identity_sentinel_path), root, "identity sentinel")
        if self.source_path_mode not in {"predecessor-workspace", "frozen-mount-inverse"}:
            raise ValueError("unsupported carry source path mode")

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "target_attempt_id": self.target_attempt_id,
            "target_action_id": self.target_action_id,
            "workspace_root": self.workspace_root,
            "roots": [item.to_mapping() for item in self.roots],
            "search_input_physical_path": self.search_input_physical_path,
            "split_input_physical_path": self.split_input_physical_path,
            "identity_sentinel_path": self.identity_sentinel_path,
            "private_workspace_mount_path": self.private_workspace_mount_path,
            "source_path_mode": self.source_path_mode,
        }


@dataclass(frozen=True)
class AttemptCarryForwardVerificationReference:
    attestation_id: str
    attestation_digest: str
    attestation_event_sequence: int
    evidence_path: str
    evidence_document_sha256: str
    evidence_mapping_digest: str
    terminal_event_digest: str
    evidence_finished_at: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.attestation_id, _ATTESTATION_ID, "evidence attestation id")
        for label, value in (
            ("attestation", self.attestation_digest),
            ("evidence document", self.evidence_document_sha256),
            ("evidence mapping", self.evidence_mapping_digest),
            ("terminal event", self.terminal_event_digest),
        ):
            _sha(value, label)
        if not isinstance(self.attestation_event_sequence, int) or self.attestation_event_sequence <= 0:
            raise ValueError("attestation event sequence must be positive")
        _absolute(self.evidence_path, "attested evidence path")
        _timestamp(self.evidence_finished_at, "attested evidence finished_at")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attestation_id": self.attestation_id,
            "attestation_digest": self.attestation_digest,
            "attestation_event_sequence": self.attestation_event_sequence,
            "evidence_path": self.evidence_path,
            "evidence_document_sha256": self.evidence_document_sha256,
            "evidence_mapping_digest": self.evidence_mapping_digest,
            "terminal_event_digest": self.terminal_event_digest,
            "evidence_finished_at": self.evidence_finished_at,
        }


@dataclass(frozen=True)
class AttemptCarryForwardRecord:
    attempt_carry_forward_id: str
    phase_run_id: str
    phase_kind: str
    phase_plan_digest: str
    input_set_identity_digest: str
    scientific_identity_digest: str
    source_attempt_id: str
    source_attempt_ordinal: int
    source_runspec_digest: str
    target_attempt_id: str
    target_attempt_ordinal: int
    verification: AttemptCarryForwardVerificationReference
    workspace: AttemptWorkspaceBinding
    content: tuple[AttemptCarryForwardContent, ...]
    content_digest: str
    remaining_record_ordinals: tuple[int, ...]
    remaining_search_input_sha256: str
    declared_at: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.attempt_carry_forward_id, _CARRY_ID, "carry-forward id")
        _match(self.phase_run_id, _PHASE_RUN_ID, "carry-forward Phase Run id")
        if self.phase_kind != "preprocessing":
            raise ValueError("carry-forward phase_kind must be preprocessing")
        for label, value in (
            ("Phase Plan", self.phase_plan_digest),
            ("input-set identity", self.input_set_identity_digest),
            ("scientific identity", self.scientific_identity_digest),
            ("source RunSpec", self.source_runspec_digest),
            ("content", self.content_digest),
            ("remaining search input", self.remaining_search_input_sha256),
        ):
            _sha(value, label)
        _match(self.source_attempt_id, _ATTEMPT_ID, "source Attempt id")
        _match(self.target_attempt_id, _ATTEMPT_ID, "target Attempt id")
        if self.source_attempt_ordinal <= 0 or self.target_attempt_ordinal != self.source_attempt_ordinal + 1:
            raise ValueError("carry-forward target must immediately follow source Attempt")
        if not isinstance(self.content, tuple) or not self.content:
            raise ValueError("carry-forward content must be non-empty")
        if any(not isinstance(item, AttemptCarryForwardContent) for item in self.content):
            raise ValueError("carry-forward content must contain strict records")
        keys = tuple((item.member_name, item.source_ordinal) for item in self.content)
        if len(set(keys)) != len(keys) or tuple(item.source_ordinal for item in self.content) != tuple(
            sorted(item.source_ordinal for item in self.content)
        ):
            raise ValueError("carry-forward content must be unique and source-ordinal ordered")
        expected_content_digest = _digest(
            {"schema_version": self.schema_version, "content": [x.to_mapping() for x in self.content]}
        )
        if self.content_digest != expected_content_digest:
            raise ValueError("carry-forward content digest does not match content")
        if not isinstance(self.remaining_record_ordinals, tuple) or not self.remaining_record_ordinals:
            raise ValueError("carry-forward requires a non-empty remaining-record tuple")
        if len(set(self.remaining_record_ordinals)) != len(
            self.remaining_record_ordinals
        ) or self.remaining_record_ordinals != tuple(sorted(self.remaining_record_ordinals)):
            raise ValueError("remaining record ordinals must be unique and ordered")
        if set(self.remaining_record_ordinals) & {item.source_ordinal for item in self.content}:
            raise ValueError("carried and remaining record ordinals must be disjoint")
        if (
            self.workspace.phase_run_id != self.phase_run_id
            or self.workspace.target_attempt_id != self.target_attempt_id
        ):
            raise ValueError("workspace identity must match carry-forward target")
        _timestamp(self.declared_at, "carry-forward declaration")
        if self.attempt_carry_forward_id != attempt_carry_forward_id(self.identity_mapping()):
            raise ValueError("carry-forward id does not match canonical content")

    def identity_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "phase_kind": self.phase_kind,
            "phase_plan_digest": self.phase_plan_digest,
            "input_set_identity_digest": self.input_set_identity_digest,
            "scientific_identity_digest": self.scientific_identity_digest,
            "source_attempt_id": self.source_attempt_id,
            "source_attempt_ordinal": self.source_attempt_ordinal,
            "source_runspec_digest": self.source_runspec_digest,
            "target_attempt_id": self.target_attempt_id,
            "target_attempt_ordinal": self.target_attempt_ordinal,
            "verification": self.verification.to_mapping(),
            "workspace": self.workspace.to_mapping(),
            "content": [item.to_mapping() for item in self.content],
            "content_digest": self.content_digest,
            "remaining_record_ordinals": list(self.remaining_record_ordinals),
            "remaining_search_input_sha256": self.remaining_search_input_sha256,
            "declared_at": self.declared_at,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {"attempt_carry_forward_id": self.attempt_carry_forward_id, **self.identity_mapping()}


def attempt_carry_forward_id(identity: Mapping[str, object]) -> str:
    return f"attempt-carry-forward-{_digest(identity)}"


@dataclass(frozen=True)
class AttemptCarryForwardReference:
    attempt_carry_forward_id: str
    digest: str
    location: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.attempt_carry_forward_id, _CARRY_ID, "carry-forward reference id")
        _sha(self.digest, "carry-forward reference")
        path = PurePosixPath(self.location)
        if path.is_absolute() or ".." in path.parts or path.name != "attempt-carry-forward.json":
            raise ValueError("carry-forward reference location must be safe and authority-relative")
        if len(path.parts) != 3 or path.parts[0] != "attempts" or _ATTEMPT_ID.fullmatch(path.parts[1]) is None:
            raise ValueError("carry-forward reference location must be Attempt-bound")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempt_carry_forward_id": self.attempt_carry_forward_id,
            "digest": self.digest,
            "location": self.location,
        }


@dataclass(frozen=True)
class AttemptCarryForwardAdoptedContent:
    member_name: str
    source_path: str
    target_path: str
    size_bytes: int
    sha256: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not self.member_name.endswith(".a3m") or "/" in self.member_name:
            raise ValueError("adopted content member must be a top-level A3M")
        _absolute(self.source_path, "adoption source path")
        _absolute(self.target_path, "adoption target path")
        if self.size_bytes <= 0:
            raise ValueError("adopted content must be non-empty")
        _sha(self.sha256, "adopted content")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "member_name": self.member_name,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class AttemptCarryForwardAdoptionEvidence:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    phase_submission_id: str
    action_id: str
    attempt_carry_forward_id: str
    attempt_carry_forward_digest: str
    remaining_search_input_sha256: str
    content: tuple[AttemptCarryForwardAdoptedContent, ...]
    adopted_at: str
    result: CarryAdoptionResult = "passed"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.phase_run_id, _PHASE_RUN_ID, "adoption Phase Run id")
        _match(self.attempt_id, _ATTEMPT_ID, "adoption Attempt id")
        _match(self.phase_submission_id, _SUBMISSION_ID, "adoption submission id")
        _match(self.action_id, _ACTION_ID, "adoption action id")
        _match(self.attempt_carry_forward_id, _CARRY_ID, "adoption carry-forward id")
        for label, value in (
            ("adoption RunSpec", self.phase_runspec_digest),
            ("adoption carry-forward", self.attempt_carry_forward_digest),
            ("adoption remaining search input", self.remaining_search_input_sha256),
        ):
            _sha(value, label)
        if not isinstance(self.content, tuple) or not self.content:
            raise ValueError("adoption evidence requires non-empty content")
        if len({item.member_name for item in self.content}) != len(self.content):
            raise ValueError("adoption evidence members must be unique")
        _timestamp(self.adopted_at, "adoption timestamp")
        if self.result != "passed":
            raise ValueError("adoption evidence can only attest passed adoption")

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "phase_submission_id": self.phase_submission_id,
            "action_id": self.action_id,
            "attempt_carry_forward_id": self.attempt_carry_forward_id,
            "attempt_carry_forward_digest": self.attempt_carry_forward_digest,
            "remaining_search_input_sha256": self.remaining_search_input_sha256,
            "content": [item.to_mapping() for item in self.content],
            "adopted_at": self.adopted_at,
            "result": self.result,
        }


@dataclass(frozen=True)
class AttemptCarryForwardReceiptReference:
    attempt_carry_forward_id: str
    attempt_carry_forward_digest: str
    source_attempt_id: str
    target_attempt_id: str
    adopted_content_digest: str
    source_verification_evidence_digest: str
    target_adoption_evidence_digest: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.attempt_carry_forward_id, _CARRY_ID, "receipt carry-forward id")
        _match(self.source_attempt_id, _ATTEMPT_ID, "receipt source Attempt")
        _match(self.target_attempt_id, _ATTEMPT_ID, "receipt target Attempt")
        for label, value in (
            ("receipt carry-forward", self.attempt_carry_forward_digest),
            ("receipt adopted content", self.adopted_content_digest),
            ("receipt source verification", self.source_verification_evidence_digest),
            ("receipt target adoption", self.target_adoption_evidence_digest),
        ):
            _sha(value, label)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempt_carry_forward_id": self.attempt_carry_forward_id,
            "attempt_carry_forward_digest": self.attempt_carry_forward_digest,
            "source_attempt_id": self.source_attempt_id,
            "target_attempt_id": self.target_attempt_id,
            "adopted_content_digest": self.adopted_content_digest,
            "source_verification_evidence_digest": self.source_verification_evidence_digest,
            "target_adoption_evidence_digest": self.target_adoption_evidence_digest,
        }


@dataclass(frozen=True)
class AttemptCarryForwardSelection:
    member_name: str
    size_bytes: int
    sha256: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not self.member_name.endswith(".a3m") or "/" in self.member_name:
            raise ValueError("carry request member must be a top-level A3M")
        if self.size_bytes <= 0:
            raise ValueError("carry request size_bytes must be positive")
        _sha(self.sha256, "carry request member")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "member_name": self.member_name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class AttemptCarryForwardRequest:
    source_attempt_id: str
    content: tuple[AttemptCarryForwardSelection, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.source_attempt_id, _ATTEMPT_ID, "carry request source Attempt")
        if not isinstance(self.content, tuple) or not self.content:
            raise ValueError("carry request content must be non-empty")
        if len({item.member_name for item in self.content}) != len(self.content):
            raise ValueError("carry request members must be unique")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_attempt_id": self.source_attempt_id,
            "content": [item.to_mapping() for item in self.content],
        }


def attempt_carry_forward_content_from_mapping(payload: Mapping[str, object]) -> AttemptCarryForwardContent:
    _strict(payload, set(AttemptCarryForwardContent.__dataclass_fields__), "AttemptCarryForwardContent")
    return AttemptCarryForwardContent(
        schema_version=_version(payload, "AttemptCarryForwardContent"),
        content_kind=cast("CarryContentKind", _str(payload, "content_kind")),
        source_action_id=_str(payload, "source_action_id"),
        target_action_id=_str(payload, "target_action_id"),
        member_name=_str(payload, "member_name"),
        source_ordinal=_int(payload, "source_ordinal"),
        record_identity=_str(payload, "record_identity"),
        source_header=_str(payload, "source_header"),
        source_declared_path=_str(payload, "source_declared_path"),
        target_declared_path=_str(payload, "target_declared_path"),
        source_physical_path=_str(payload, "source_physical_path"),
        target_physical_path=_str(payload, "target_physical_path"),
        source_private_mount_path=_str(payload, "source_private_mount_path"),
        size_bytes=_int(payload, "size_bytes"),
        sha256=_str(payload, "sha256"),
    )


def attempt_workspace_root_binding_from_mapping(payload: Mapping[str, object]) -> AttemptWorkspaceRootBinding:
    _strict(payload, set(AttemptWorkspaceRootBinding.__dataclass_fields__), "AttemptWorkspaceRootBinding")
    return AttemptWorkspaceRootBinding(
        schema_version=_version(payload, "AttemptWorkspaceRootBinding"),
        role=_str(payload, "role"),
        logical_root=_str(payload, "logical_root"),
        physical_root=_str(payload, "physical_root"),
    )


def attempt_workspace_binding_from_mapping(payload: Mapping[str, object]) -> AttemptWorkspaceBinding:
    _strict(payload, set(AttemptWorkspaceBinding.__dataclass_fields__), "AttemptWorkspaceBinding")
    return AttemptWorkspaceBinding(
        schema_version=_version(payload, "AttemptWorkspaceBinding"),
        phase_run_id=_str(payload, "phase_run_id"),
        target_attempt_id=_str(payload, "target_attempt_id"),
        target_action_id=_str(payload, "target_action_id"),
        workspace_root=_str(payload, "workspace_root"),
        roots=tuple(attempt_workspace_root_binding_from_mapping(x) for x in _mappings(payload, "roots")),
        search_input_physical_path=_str(payload, "search_input_physical_path"),
        split_input_physical_path=_str(payload, "split_input_physical_path"),
        identity_sentinel_path=_str(payload, "identity_sentinel_path"),
        private_workspace_mount_path=_str(payload, "private_workspace_mount_path"),
        source_path_mode=cast("CarrySourcePathMode", _str(payload, "source_path_mode")),
    )


def attempt_carry_forward_verification_reference_from_mapping(
    payload: Mapping[str, object],
) -> AttemptCarryForwardVerificationReference:
    _strict(
        payload,
        set(AttemptCarryForwardVerificationReference.__dataclass_fields__),
        "AttemptCarryForwardVerificationReference",
    )
    return AttemptCarryForwardVerificationReference(
        schema_version=_version(payload, "AttemptCarryForwardVerificationReference"),
        attestation_id=_str(payload, "attestation_id"),
        attestation_digest=_str(payload, "attestation_digest"),
        attestation_event_sequence=_int(payload, "attestation_event_sequence"),
        evidence_path=_str(payload, "evidence_path"),
        evidence_document_sha256=_str(payload, "evidence_document_sha256"),
        evidence_mapping_digest=_str(payload, "evidence_mapping_digest"),
        terminal_event_digest=_str(payload, "terminal_event_digest"),
        evidence_finished_at=_str(payload, "evidence_finished_at"),
    )


def attempt_carry_forward_record_from_mapping(payload: Mapping[str, object]) -> AttemptCarryForwardRecord:
    _strict(payload, set(AttemptCarryForwardRecord.__dataclass_fields__), "AttemptCarryForwardRecord")
    return AttemptCarryForwardRecord(
        schema_version=_version(payload, "AttemptCarryForwardRecord"),
        attempt_carry_forward_id=_str(payload, "attempt_carry_forward_id"),
        phase_run_id=_str(payload, "phase_run_id"),
        phase_kind=_str(payload, "phase_kind"),
        phase_plan_digest=_str(payload, "phase_plan_digest"),
        input_set_identity_digest=_str(payload, "input_set_identity_digest"),
        scientific_identity_digest=_str(payload, "scientific_identity_digest"),
        source_attempt_id=_str(payload, "source_attempt_id"),
        source_attempt_ordinal=_int(payload, "source_attempt_ordinal"),
        source_runspec_digest=_str(payload, "source_runspec_digest"),
        target_attempt_id=_str(payload, "target_attempt_id"),
        target_attempt_ordinal=_int(payload, "target_attempt_ordinal"),
        verification=attempt_carry_forward_verification_reference_from_mapping(_mapping(payload, "verification")),
        workspace=attempt_workspace_binding_from_mapping(_mapping(payload, "workspace")),
        content=tuple(attempt_carry_forward_content_from_mapping(x) for x in _mappings(payload, "content")),
        content_digest=_str(payload, "content_digest"),
        remaining_record_ordinals=_ints(payload, "remaining_record_ordinals"),
        remaining_search_input_sha256=_str(payload, "remaining_search_input_sha256"),
        declared_at=_str(payload, "declared_at"),
    )


def attempt_carry_forward_reference_from_mapping(payload: Mapping[str, object]) -> AttemptCarryForwardReference:
    _strict(payload, set(AttemptCarryForwardReference.__dataclass_fields__), "AttemptCarryForwardReference")
    return AttemptCarryForwardReference(
        schema_version=_version(payload, "AttemptCarryForwardReference"),
        attempt_carry_forward_id=_str(payload, "attempt_carry_forward_id"),
        digest=_str(payload, "digest"),
        location=_str(payload, "location"),
    )


def attempt_carry_forward_adopted_content_from_mapping(
    payload: Mapping[str, object],
) -> AttemptCarryForwardAdoptedContent:
    _strict(payload, set(AttemptCarryForwardAdoptedContent.__dataclass_fields__), "AttemptCarryForwardAdoptedContent")
    return AttemptCarryForwardAdoptedContent(
        schema_version=_version(payload, "AttemptCarryForwardAdoptedContent"),
        member_name=_str(payload, "member_name"),
        source_path=_str(payload, "source_path"),
        target_path=_str(payload, "target_path"),
        size_bytes=_int(payload, "size_bytes"),
        sha256=_str(payload, "sha256"),
    )


def attempt_carry_forward_adoption_evidence_from_mapping(
    payload: Mapping[str, object],
) -> AttemptCarryForwardAdoptionEvidence:
    _strict(
        payload,
        set(AttemptCarryForwardAdoptionEvidence.__dataclass_fields__),
        "AttemptCarryForwardAdoptionEvidence",
    )
    return AttemptCarryForwardAdoptionEvidence(
        schema_version=_version(payload, "AttemptCarryForwardAdoptionEvidence"),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        phase_runspec_digest=_str(payload, "phase_runspec_digest"),
        phase_submission_id=_str(payload, "phase_submission_id"),
        action_id=_str(payload, "action_id"),
        attempt_carry_forward_id=_str(payload, "attempt_carry_forward_id"),
        attempt_carry_forward_digest=_str(payload, "attempt_carry_forward_digest"),
        remaining_search_input_sha256=_str(payload, "remaining_search_input_sha256"),
        content=tuple(attempt_carry_forward_adopted_content_from_mapping(x) for x in _mappings(payload, "content")),
        adopted_at=_str(payload, "adopted_at"),
        result=cast("CarryAdoptionResult", _str(payload, "result")),
    )


def attempt_carry_forward_receipt_reference_from_mapping(
    payload: Mapping[str, object],
) -> AttemptCarryForwardReceiptReference:
    _strict(
        payload,
        set(AttemptCarryForwardReceiptReference.__dataclass_fields__),
        "AttemptCarryForwardReceiptReference",
    )
    return AttemptCarryForwardReceiptReference(
        schema_version=_version(payload, "AttemptCarryForwardReceiptReference"),
        attempt_carry_forward_id=_str(payload, "attempt_carry_forward_id"),
        attempt_carry_forward_digest=_str(payload, "attempt_carry_forward_digest"),
        source_attempt_id=_str(payload, "source_attempt_id"),
        target_attempt_id=_str(payload, "target_attempt_id"),
        adopted_content_digest=_str(payload, "adopted_content_digest"),
        source_verification_evidence_digest=_str(payload, "source_verification_evidence_digest"),
        target_adoption_evidence_digest=_str(payload, "target_adoption_evidence_digest"),
    )


def attempt_carry_forward_request_from_mapping(payload: Mapping[str, object]) -> AttemptCarryForwardRequest:
    _strict(payload, set(AttemptCarryForwardRequest.__dataclass_fields__), "AttemptCarryForwardRequest")
    return AttemptCarryForwardRequest(
        schema_version=_version(payload, "AttemptCarryForwardRequest"),
        source_attempt_id=_str(payload, "source_attempt_id"),
        content=tuple(_selection_from_mapping(item) for item in _mappings(payload, "content")),
    )


def _selection_from_mapping(payload: Mapping[str, object]) -> AttemptCarryForwardSelection:
    _strict(payload, set(AttemptCarryForwardSelection.__dataclass_fields__), "AttemptCarryForwardSelection")
    return AttemptCarryForwardSelection(
        schema_version=_version(payload, "AttemptCarryForwardSelection"),
        member_name=_str(payload, "member_name"),
        size_bytes=_int(payload, "size_bytes"),
        sha256=_str(payload, "sha256"),
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


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _mappings(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of mappings")
    return tuple(cast("Mapping[str, object]", item) for item in value)


def _ints(payload: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{key} must be a list of integers")
    return tuple(cast("int", item) for item in value)


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


def _within(path: PurePosixPath, root: PurePosixPath, label: str) -> None:
    if path == root or root not in path.parents:
        raise ValueError(f"{label} must be strictly below workspace root")


def _timestamp(value: str, label: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {label} timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} timestamp must include timezone")


__all__ = [
    "AttemptCarryForwardAdoptedContent",
    "AttemptCarryForwardAdoptionEvidence",
    "AttemptCarryForwardContent",
    "AttemptCarryForwardReceiptReference",
    "AttemptCarryForwardRecord",
    "AttemptCarryForwardReference",
    "AttemptCarryForwardRequest",
    "AttemptCarryForwardSelection",
    "AttemptCarryForwardVerificationReference",
    "AttemptWorkspaceBinding",
    "AttemptWorkspaceRootBinding",
    "attempt_carry_forward_adoption_evidence_from_mapping",
    "attempt_carry_forward_content_from_mapping",
    "attempt_carry_forward_id",
    "attempt_carry_forward_receipt_reference_from_mapping",
    "attempt_carry_forward_record_from_mapping",
    "attempt_carry_forward_reference_from_mapping",
    "attempt_carry_forward_request_from_mapping",
    "attempt_carry_forward_verification_reference_from_mapping",
    "attempt_workspace_binding_from_mapping",
]
