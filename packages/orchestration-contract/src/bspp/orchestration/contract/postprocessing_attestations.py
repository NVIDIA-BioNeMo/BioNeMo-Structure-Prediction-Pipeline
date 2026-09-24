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


@dataclass(frozen=True)
class PostprocessingRuntimeInputAttestation:
    logical_input_name: str
    verification_kind: Literal["local-content-sha256-v1", "authority-declared-content-v1"]
    authority: str
    content_sha256: str
    size_bytes: int
    verification_source: Literal["control-materialization", "runtime-preflight"]
    observed_at: str
    accessible: Literal[True] = True
    physical_input_name: str | None = None
    physical_locator: str | None = None
    member_identity: str | None = None
    accessibility_evidence_path: str | None = None
    accessibility_evidence_sha256: str | None = None
    accessibility_evidence_size_bytes: int | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not self.logical_input_name or not self.authority or not self.observed_at:
            raise ValueError("postprocessing runtime input attestation identity is incomplete")
        _sha(self.content_sha256)
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("postprocessing runtime input attestation size must be non-negative")
        if self.verification_kind == "local-content-sha256-v1":
            if (
                self.verification_source != "control-materialization"
                or self.physical_input_name is not None
                or self.physical_locator is not None
                or self.member_identity is not None
                or self.accessibility_evidence_path is not None
                or self.accessibility_evidence_sha256 is not None
                or self.accessibility_evidence_size_bytes is not None
            ):
                raise ValueError("local input attestation must preserve the materialization content proof")
        elif self.verification_kind == "authority-declared-content-v1":
            if (
                self.verification_source != "runtime-preflight"
                or not self.physical_input_name
                or not self.physical_locator
                or not self.member_identity
                or not self.accessibility_evidence_path
                or self.accessibility_evidence_sha256 is None
                or self.accessibility_evidence_size_bytes is None
            ):
                raise ValueError("authority-declared input attestation requires runtime locator verification")
            _sha(self.accessibility_evidence_sha256)
            if (
                self.accessibility_evidence_path.startswith("/")
                or ".." in self.accessibility_evidence_path.split("/")
                or not isinstance(self.accessibility_evidence_size_bytes, int)
                or isinstance(self.accessibility_evidence_size_bytes, bool)
                or self.accessibility_evidence_size_bytes <= 0
            ):
                raise ValueError("runtime input accessibility proof identity is invalid")
        else:
            raise ValueError("unsupported postprocessing runtime input attestation kind")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "logical_input_name": self.logical_input_name,
            "verification_kind": self.verification_kind,
            "authority": self.authority,
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
            "verification_source": self.verification_source,
            "observed_at": self.observed_at,
            "accessible": self.accessible,
            "physical_input_name": self.physical_input_name,
            "physical_locator": self.physical_locator,
            "member_identity": self.member_identity,
            "accessibility_evidence_path": self.accessibility_evidence_path,
            "accessibility_evidence_sha256": self.accessibility_evidence_sha256,
            "accessibility_evidence_size_bytes": self.accessibility_evidence_size_bytes,
        }


@dataclass(frozen=True)
class PostprocessingRuntimeQualificationAttestation:
    """Attempt-exact proof of the Runtime Qualification consumed by Action 01."""

    qualified_runtime_digest: str
    record_location: str
    record_sha256: str
    record_size_bytes: int
    tuple_id: str
    source_identity_digest: str
    source_package_identity_digest: str
    toolkit_identity_digest: str
    runtime_component_identity_digest: str
    observed_at: str
    attestation_kind: str = "postprocessing-runtime-qualification-attestation-v2"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if (
            self.attestation_kind != "postprocessing-runtime-qualification-attestation-v2"
            or not self.record_location.startswith("attempts/")
            or not self.record_location.endswith("/runtime-qualification.json")
            or not self.observed_at
        ):
            raise ValueError("postprocessing Runtime Qualification attestation identity is invalid")
        for value in (
            self.qualified_runtime_digest,
            self.record_sha256,
            self.tuple_id,
            self.source_identity_digest,
            self.source_package_identity_digest,
            self.toolkit_identity_digest,
            self.runtime_component_identity_digest,
        ):
            _sha(value)
        if (
            not isinstance(self.record_size_bytes, int)
            or isinstance(self.record_size_bytes, bool)
            or self.record_size_bytes <= 0
        ):
            raise ValueError("postprocessing Runtime Qualification record size must be positive")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attestation_kind": self.attestation_kind,
            "qualified_runtime_digest": self.qualified_runtime_digest,
            "record_location": self.record_location,
            "record_sha256": self.record_sha256,
            "record_size_bytes": self.record_size_bytes,
            "tuple_id": self.tuple_id,
            "source_identity_digest": self.source_identity_digest,
            "source_package_identity_digest": self.source_package_identity_digest,
            "toolkit_identity_digest": self.toolkit_identity_digest,
            "runtime_component_identity_digest": self.runtime_component_identity_digest,
            "observed_at": self.observed_at,
        }


@dataclass(frozen=True)
class PostprocessingRuntimeInputAttestationSet:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    runtime_qualification: PostprocessingRuntimeQualificationAttestation
    attestations: tuple[PostprocessingRuntimeInputAttestation, ...]
    attestation_kind: str = "postprocessing-runtime-input-attestations-v2"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        _sha(self.phase_runspec_digest)
        names = tuple(item.logical_input_name for item in self.attestations)
        if (
            self.attestation_kind != "postprocessing-runtime-input-attestations-v2"
            or not names
            or names != tuple(sorted(names))
            or len(set(names)) != len(names)
        ):
            raise ValueError("runtime input attestations must be complete-looking, unique, and name-sorted")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attestation_kind": self.attestation_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "runtime_qualification": self.runtime_qualification.to_mapping(),
            "attestations": [item.to_mapping() for item in self.attestations],
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


def _attestation(payload: Mapping[str, object]) -> PostprocessingRuntimeInputAttestation:
    _fields(
        payload,
        {
            "schema_version",
            "logical_input_name",
            "verification_kind",
            "authority",
            "content_sha256",
            "size_bytes",
            "verification_source",
            "observed_at",
            "accessible",
            "physical_input_name",
            "physical_locator",
            "member_identity",
            "accessibility_evidence_path",
            "accessibility_evidence_sha256",
            "accessibility_evidence_size_bytes",
        },
        "PostprocessingRuntimeInputAttestation",
    )
    return PostprocessingRuntimeInputAttestation(
        schema_version=_schema(payload, "PostprocessingRuntimeInputAttestation"),
        logical_input_name=_string(payload, "logical_input_name"),
        verification_kind=cast(
            "Literal['local-content-sha256-v1', 'authority-declared-content-v1']",
            _string(payload, "verification_kind"),
        ),
        authority=_string(payload, "authority"),
        content_sha256=_string(payload, "content_sha256"),
        size_bytes=_integer(payload, "size_bytes"),
        verification_source=cast(
            "Literal['control-materialization', 'runtime-preflight']",
            _string(payload, "verification_source"),
        ),
        observed_at=_string(payload, "observed_at"),
        accessible=_true(payload, "accessible"),
        physical_input_name=_optional_string(payload, "physical_input_name"),
        physical_locator=_optional_string(payload, "physical_locator"),
        member_identity=_optional_string(payload, "member_identity"),
        accessibility_evidence_path=_optional_string(payload, "accessibility_evidence_path"),
        accessibility_evidence_sha256=_optional_string(payload, "accessibility_evidence_sha256"),
        accessibility_evidence_size_bytes=_optional_integer(payload, "accessibility_evidence_size_bytes"),
    )


def _attestation_set(payload: Mapping[str, object]) -> PostprocessingRuntimeInputAttestationSet:
    _fields(
        payload,
        {
            "schema_version",
            "attestation_kind",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "runtime_qualification",
            "attestations",
        },
        "PostprocessingRuntimeInputAttestationSet",
    )
    return PostprocessingRuntimeInputAttestationSet(
        schema_version=_schema(payload, "PostprocessingRuntimeInputAttestationSet"),
        attestation_kind=_string(payload, "attestation_kind"),
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        runtime_qualification=_runtime_qualification_attestation(_mapping(payload, "runtime_qualification")),
        attestations=tuple(_attestation(item) for item in _mapping_list(payload, "attestations")),
    )


def _runtime_qualification_attestation(
    payload: Mapping[str, object],
) -> PostprocessingRuntimeQualificationAttestation:
    _fields(
        payload,
        {
            "schema_version",
            "attestation_kind",
            "qualified_runtime_digest",
            "record_location",
            "record_sha256",
            "record_size_bytes",
            "tuple_id",
            "source_identity_digest",
            "source_package_identity_digest",
            "toolkit_identity_digest",
            "runtime_component_identity_digest",
            "observed_at",
        },
        "PostprocessingRuntimeQualificationAttestation",
    )
    return PostprocessingRuntimeQualificationAttestation(
        schema_version=_schema(payload, "PostprocessingRuntimeQualificationAttestation"),
        attestation_kind=_string(payload, "attestation_kind"),
        qualified_runtime_digest=_string(payload, "qualified_runtime_digest"),
        record_location=_string(payload, "record_location"),
        record_sha256=_string(payload, "record_sha256"),
        record_size_bytes=_integer(payload, "record_size_bytes"),
        tuple_id=_string(payload, "tuple_id"),
        source_identity_digest=_string(payload, "source_identity_digest"),
        source_package_identity_digest=_string(payload, "source_package_identity_digest"),
        toolkit_identity_digest=_string(payload, "toolkit_identity_digest"),
        runtime_component_identity_digest=_string(payload, "runtime_component_identity_digest"),
        observed_at=_string(payload, "observed_at"),
    )


def postprocessing_runtime_input_attestation_set_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingRuntimeInputAttestationSet:
    return _attestation_set(payload)


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


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


def _optional_integer(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{key} must be an integer or null")
    return value


def _true(payload: Mapping[str, object], key: str) -> Literal[True]:
    if payload.get(key) is not True:
        raise ValueError(f"{key} must be true")
    return True


def _sha(value: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError("postprocessing receipt digest must be lowercase SHA-256")


__all__ = [
    "PostprocessingRuntimeInputAttestation",
    "PostprocessingRuntimeInputAttestationSet",
    "PostprocessingRuntimeQualificationAttestation",
    "postprocessing_runtime_input_attestation_set_from_mapping",
]
