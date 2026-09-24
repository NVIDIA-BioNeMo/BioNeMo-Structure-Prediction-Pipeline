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

"""Strict evidence for a staged Database Replica shared lease."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, cast

from bspp.orchestration.contract.database_placement import (
    DATABASE_REPLICA_LEASE_TARGET,
    LEGACY_DATABASE_REPLICA_LEASE_TARGET,
    LEGACY_SELECTED_DATABASE_ROOT,
    SELECTED_DATABASE_ROOT,
)
from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaColdFailureEvidence,
    DatabaseReplicaColdResult,
    DatabaseReplicaResult,
    DatabaseReplicaWarmResult,
    database_replica_cold_failure_evidence_from_mapping,
    database_replica_result_digest,
    database_replica_result_from_mapping,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

DatabaseReplicaLeaseFailureClassification = Literal[
    "lease-authority-invalid",
    "lease-open-failed",
    "lease-contended",
    "replica-revalidation-failed",
]
DatabaseReplicaLeaseAcquisition = Literal["shared-nonblocking"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FAILURES = frozenset(
    {
        "lease-authority-invalid",
        "lease-open-failed",
        "lease-contended",
        "replica-revalidation-failed",
    }
)


@dataclass(frozen=True)
class DatabaseReplicaLeaseFailureEvidence:
    """Bounded proof that shared lease/revalidation failed before science."""

    database_replica_cold_result_digest: str
    source_manifest_sha256: str
    replica_manifest_sha256: str
    selected_container_root: str
    lease_target: str
    verification: Literal["metadata-verified"]
    classification: DatabaseReplicaLeaseFailureClassification
    error: str
    lease_outcome_kind: Literal["failure"] = "failure"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_common(self)
        if self.lease_outcome_kind != "failure":
            raise ValueError("lease failure evidence must use the failure outcome kind")
        if self.classification not in _FAILURES:
            raise ValueError(f"unsupported Database Replica Lease failure classification: {self.classification!r}")
        if not self.error or len(self.error) > 2048 or "\x00" in self.error:
            raise ValueError("Database Replica Lease failure error must be non-empty and bounded to 2048 characters")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lease_outcome_kind": self.lease_outcome_kind,
            "database_replica_cold_result_digest": self.database_replica_cold_result_digest,
            "source_manifest_sha256": self.source_manifest_sha256,
            "replica_manifest_sha256": self.replica_manifest_sha256,
            "selected_container_root": self.selected_container_root,
            "lease_target": self.lease_target,
            "verification": self.verification,
            "classification": self.classification,
            "error": self.error,
        }


@dataclass(frozen=True)
class DatabaseReplicaLeaseTerminalEvidence:
    """Proof that one shared lease covered the complete started kernel interval."""

    database_replica_cold_result_digest: str
    source_manifest_sha256: str
    replica_manifest_sha256: str
    selected_container_root: str
    lease_target: str
    verification: Literal["metadata-verified"]
    acquisition: DatabaseReplicaLeaseAcquisition
    kernel_started: bool
    held_through_kernel_exit: bool
    lease_outcome_kind: Literal["terminal"] = "terminal"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_common(self)
        if self.lease_outcome_kind != "terminal":
            raise ValueError("terminal lease evidence must use the terminal outcome kind")
        if self.acquisition != "shared-nonblocking":
            raise ValueError("Database Replica Lease acquisition must be shared-nonblocking")
        if not isinstance(self.kernel_started, bool) or not isinstance(self.held_through_kernel_exit, bool):
            raise ValueError("kernel_started and held_through_kernel_exit must be booleans")
        if self.kernel_started != self.held_through_kernel_exit:
            raise ValueError("terminal lease evidence kernel lifetime booleans must be both false or both true")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lease_outcome_kind": self.lease_outcome_kind,
            "database_replica_cold_result_digest": self.database_replica_cold_result_digest,
            "source_manifest_sha256": self.source_manifest_sha256,
            "replica_manifest_sha256": self.replica_manifest_sha256,
            "selected_container_root": self.selected_container_root,
            "lease_target": self.lease_target,
            "verification": self.verification,
            "acquisition": self.acquisition,
            "kernel_started": self.kernel_started,
            "held_through_kernel_exit": self.held_through_kernel_exit,
        }


@dataclass(frozen=True)
class DatabaseReplicaWarmLeaseFailureEvidence:
    """Bounded proof that a warm Result lease/revalidation failed before science."""

    database_replica_warm_result_digest: str
    source_manifest_sha256: str
    replica_manifest_sha256: str
    selected_container_root: str
    lease_target: str
    verification: Literal["metadata-verified"]
    classification: DatabaseReplicaLeaseFailureClassification
    error: str
    lease_outcome_kind: Literal["failure"] = "failure"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_common(self)
        if self.lease_outcome_kind != "failure":
            raise ValueError("warm lease failure evidence must use the failure outcome kind")
        if self.classification not in _FAILURES:
            raise ValueError(f"unsupported Database Replica Lease failure classification: {self.classification!r}")
        if not self.error or len(self.error) > 2048 or "\x00" in self.error:
            raise ValueError("Database Replica Lease failure error must be non-empty and bounded to 2048 characters")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lease_outcome_kind": self.lease_outcome_kind,
            "database_replica_warm_result_digest": self.database_replica_warm_result_digest,
            "source_manifest_sha256": self.source_manifest_sha256,
            "replica_manifest_sha256": self.replica_manifest_sha256,
            "selected_container_root": self.selected_container_root,
            "lease_target": self.lease_target,
            "verification": self.verification,
            "classification": self.classification,
            "error": self.error,
        }


@dataclass(frozen=True)
class DatabaseReplicaWarmLeaseTerminalEvidence:
    """Proof that a warm Result shared lease covered the complete kernel interval."""

    database_replica_warm_result_digest: str
    source_manifest_sha256: str
    replica_manifest_sha256: str
    selected_container_root: str
    lease_target: str
    verification: Literal["metadata-verified"]
    acquisition: DatabaseReplicaLeaseAcquisition
    kernel_started: bool
    held_through_kernel_exit: bool
    lease_outcome_kind: Literal["terminal"] = "terminal"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_common(self)
        if self.lease_outcome_kind != "terminal":
            raise ValueError("warm terminal lease evidence must use the terminal outcome kind")
        if self.acquisition != "shared-nonblocking":
            raise ValueError("Database Replica Lease acquisition must be shared-nonblocking")
        if not isinstance(self.kernel_started, bool) or not isinstance(self.held_through_kernel_exit, bool):
            raise ValueError("kernel_started and held_through_kernel_exit must be booleans")
        if self.kernel_started != self.held_through_kernel_exit:
            raise ValueError("terminal lease evidence kernel lifetime booleans must be both false or both true")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lease_outcome_kind": self.lease_outcome_kind,
            "database_replica_warm_result_digest": self.database_replica_warm_result_digest,
            "source_manifest_sha256": self.source_manifest_sha256,
            "replica_manifest_sha256": self.replica_manifest_sha256,
            "selected_container_root": self.selected_container_root,
            "lease_target": self.lease_target,
            "verification": self.verification,
            "acquisition": self.acquisition,
            "kernel_started": self.kernel_started,
            "held_through_kernel_exit": self.held_through_kernel_exit,
        }


DatabaseReplicaLeaseEvidence = (
    DatabaseReplicaLeaseFailureEvidence
    | DatabaseReplicaLeaseTerminalEvidence
    | DatabaseReplicaWarmLeaseFailureEvidence
    | DatabaseReplicaWarmLeaseTerminalEvidence
)


@dataclass(frozen=True)
class PreprocessingStagedDatabasePlacementEvidence:
    """Staged-placement authority and distinct lease outcome state."""

    result: DatabaseReplicaResult | None
    result_digest: str | None
    failure: DatabaseReplicaColdFailureEvidence | None
    lease_outcome: DatabaseReplicaLeaseEvidence | None
    science_started: bool
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="PreprocessingStagedDatabasePlacementEvidence")
        if not isinstance(self.science_started, bool):
            raise ValueError("staged database placement science_started must be a boolean")
        if (self.result is None) == (self.failure is None):
            raise ValueError("staged database placement Result and cold failure must be mutually exclusive")
        if self.failure is not None:
            if self.result_digest is not None or self.lease_outcome is not None or self.science_started:
                raise ValueError("cold placement failure excludes Result, lease outcome, and science")
            return
        assert self.result is not None
        if self.result_digest != database_replica_result_digest(self.result):
            raise ValueError("staged placement Result digest must bind exact canonical Result bytes")
        lease = self.lease_outcome
        if lease is None:
            if self.science_started:
                raise ValueError("successful cold placement with science started requires a subsequent lease outcome")
            return
        if (
            _lease_result_digest(lease) != self.result_digest
            or lease.source_manifest_sha256 != self.result.source_manifest_sha256
            or lease.replica_manifest_sha256 != self.result.replica_manifest_sha256
            or lease.selected_container_root != self.result.selected_container_root
        ):
            raise ValueError("staged lease outcome must bind the exact cold Result authority")
        if isinstance(self.result, DatabaseReplicaColdResult) and not isinstance(
            lease, DatabaseReplicaLeaseFailureEvidence | DatabaseReplicaLeaseTerminalEvidence
        ):
            raise ValueError("cold Result requires the cold lease evidence family")
        if isinstance(self.result, DatabaseReplicaWarmResult) and not isinstance(
            lease, DatabaseReplicaWarmLeaseFailureEvidence | DatabaseReplicaWarmLeaseTerminalEvidence
        ):
            raise ValueError("warm Result requires the warm lease evidence family")
        if isinstance(lease, DatabaseReplicaLeaseFailureEvidence | DatabaseReplicaWarmLeaseFailureEvidence):
            if self.science_started:
                raise ValueError("lease/revalidation failure must occur before science starts")
        elif isinstance(lease, DatabaseReplicaLeaseTerminalEvidence | DatabaseReplicaWarmLeaseTerminalEvidence):
            if self.science_started != lease.kernel_started:
                raise ValueError("science_started must exactly match terminal lease kernel_started")
        else:
            raise ValueError("staged placement has an unsupported lease outcome type")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "result": None if self.result is None else self.result.to_mapping(),
            "result_digest": self.result_digest,
            "failure": None if self.failure is None else self.failure.to_mapping(),
            "lease_outcome": None if self.lease_outcome is None else self.lease_outcome.to_mapping(),
            "science_started": self.science_started,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PreprocessingStagedDatabasePlacementEvidence:
        fields = {"schema_version", "result", "result_digest", "failure", "lease_outcome", "science_started"}
        _require_exact_fields(payload, fields, "PreprocessingStagedDatabasePlacementEvidence")
        result_payload = payload.get("result")
        failure_payload = payload.get("failure")
        lease_payload = payload.get("lease_outcome")
        science_started = payload.get("science_started")
        for value, label in (
            (result_payload, "result"),
            (failure_payload, "failure"),
            (lease_payload, "lease_outcome"),
        ):
            if value is not None and not isinstance(value, Mapping):
                raise ValueError(f"staged database placement {label} must be a mapping or null")
        if not isinstance(science_started, bool):
            raise ValueError("staged database placement science_started must be a boolean")
        result_digest = payload.get("result_digest")
        if result_digest is not None and (not isinstance(result_digest, str) or not result_digest):
            raise ValueError("result_digest must be a non-empty string or null")
        return cls(
            result=(
                database_replica_result_from_mapping(cast("Mapping[str, object]", result_payload))
                if result_payload is not None
                else None
            ),
            result_digest=result_digest,
            failure=(
                database_replica_cold_failure_evidence_from_mapping(cast("Mapping[str, object]", failure_payload))
                if failure_payload is not None
                else None
            ),
            lease_outcome=(
                database_replica_lease_evidence_from_mapping(cast("Mapping[str, object]", lease_payload))
                if lease_payload is not None
                else None
            ),
            science_started=science_started,
            schema_version=validate_schema_version(
                payload.get("schema_version"), record_name="PreprocessingStagedDatabasePlacementEvidence"
            ),
        )


def database_replica_lease_evidence_from_mapping(payload: Mapping[str, object]) -> DatabaseReplicaLeaseEvidence:
    kind = payload.get("lease_outcome_kind")
    cold_family = "database_replica_cold_result_digest" in payload
    warm_family = "database_replica_warm_result_digest" in payload
    if cold_family == warm_family:
        raise ValueError("Database Replica Lease evidence requires exactly one Result digest family")
    digest_field = "database_replica_cold_result_digest" if cold_family else "database_replica_warm_result_digest"
    common = {
        "schema_version",
        "lease_outcome_kind",
        digest_field,
        "source_manifest_sha256",
        "replica_manifest_sha256",
        "selected_container_root",
        "lease_target",
        "verification",
    }
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="DatabaseReplicaLeaseEvidence")
    source_manifest_sha256 = _required_str(payload, "source_manifest_sha256")
    replica_manifest_sha256 = _required_str(payload, "replica_manifest_sha256")
    selected_container_root = _required_str(payload, "selected_container_root")
    lease_target = _required_str(payload, "lease_target")
    verification = cast("Literal['metadata-verified']", _required_str(payload, "verification"))
    result_digest = _required_str(payload, digest_field)
    if kind == "failure":
        _require_exact_fields(payload, common | {"classification", "error"}, "lease failure evidence")
        classification = cast("DatabaseReplicaLeaseFailureClassification", _required_str(payload, "classification"))
        error = _required_str(payload, "error")
        if warm_family:
            return DatabaseReplicaWarmLeaseFailureEvidence(
                database_replica_warm_result_digest=result_digest,
                source_manifest_sha256=source_manifest_sha256,
                replica_manifest_sha256=replica_manifest_sha256,
                selected_container_root=selected_container_root,
                lease_target=lease_target,
                verification=verification,
                classification=classification,
                error=error,
                schema_version=schema_version,
            )
        return DatabaseReplicaLeaseFailureEvidence(
            database_replica_cold_result_digest=result_digest,
            source_manifest_sha256=source_manifest_sha256,
            replica_manifest_sha256=replica_manifest_sha256,
            selected_container_root=selected_container_root,
            lease_target=lease_target,
            verification=verification,
            classification=classification,
            error=error,
            schema_version=schema_version,
        )
    if kind == "terminal":
        _require_exact_fields(
            payload,
            common | {"acquisition", "kernel_started", "held_through_kernel_exit"},
            "terminal lease evidence",
        )
        kernel_started = payload.get("kernel_started")
        held = payload.get("held_through_kernel_exit")
        if not isinstance(kernel_started, bool) or not isinstance(held, bool):
            raise ValueError("kernel_started and held_through_kernel_exit must be booleans")
        acquisition = cast("DatabaseReplicaLeaseAcquisition", _required_str(payload, "acquisition"))
        if warm_family:
            return DatabaseReplicaWarmLeaseTerminalEvidence(
                database_replica_warm_result_digest=result_digest,
                source_manifest_sha256=source_manifest_sha256,
                replica_manifest_sha256=replica_manifest_sha256,
                selected_container_root=selected_container_root,
                lease_target=lease_target,
                verification=verification,
                acquisition=acquisition,
                kernel_started=kernel_started,
                held_through_kernel_exit=held,
                schema_version=schema_version,
            )
        return DatabaseReplicaLeaseTerminalEvidence(
            database_replica_cold_result_digest=result_digest,
            source_manifest_sha256=source_manifest_sha256,
            replica_manifest_sha256=replica_manifest_sha256,
            selected_container_root=selected_container_root,
            lease_target=lease_target,
            verification=verification,
            acquisition=acquisition,
            kernel_started=kernel_started,
            held_through_kernel_exit=held,
            schema_version=schema_version,
        )
    raise ValueError(f"unsupported Database Replica Lease outcome kind: {kind!r}")


def _validate_common(
    evidence: DatabaseReplicaLeaseEvidence,
) -> None:
    validate_schema_version(evidence.schema_version, record_name=type(evidence).__name__)
    for value, label in (
        (_lease_result_digest(evidence), "Result digest"),
        (evidence.source_manifest_sha256, "source-manifest digest"),
        (evidence.replica_manifest_sha256, "replica-manifest digest"),
    ):
        if _SHA256.fullmatch(value) is None:
            raise ValueError(f"Database Replica Lease {label} must be a lowercase SHA-256")
    if evidence.selected_container_root not in (SELECTED_DATABASE_ROOT, LEGACY_SELECTED_DATABASE_ROOT):
        raise ValueError("Database Replica Lease selected root must match protected authority")
    if evidence.lease_target not in (DATABASE_REPLICA_LEASE_TARGET, LEGACY_DATABASE_REPLICA_LEASE_TARGET):
        raise ValueError("Database Replica Lease target must match protected authority")
    if evidence.verification != "metadata-verified":
        raise ValueError("Database Replica Lease verification must be metadata-verified")
    for value, label in (
        (evidence.selected_container_root, "selected root"),
        (evidence.lease_target, "target"),
    ):
        path = PurePosixPath(value)
        if not path.is_absolute() or path.as_posix() != value or ".." in path.parts:
            raise ValueError(f"Database Replica Lease {label} must be an absolute normalized path")


def _lease_result_digest(evidence: DatabaseReplicaLeaseEvidence) -> str:
    if isinstance(evidence, DatabaseReplicaLeaseFailureEvidence | DatabaseReplicaLeaseTerminalEvidence):
        return evidence.database_replica_cold_result_digest
    return evidence.database_replica_warm_result_digest


def _require_exact_fields(payload: Mapping[str, object], fields: set[str], name: str) -> None:
    if set(payload) != fields:
        raise ValueError(f"{name} requires exact canonical fields")


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


__all__ = [
    "DatabaseReplicaLeaseAcquisition",
    "DatabaseReplicaLeaseEvidence",
    "DatabaseReplicaLeaseFailureClassification",
    "DatabaseReplicaLeaseFailureEvidence",
    "DatabaseReplicaLeaseTerminalEvidence",
    "DatabaseReplicaWarmLeaseFailureEvidence",
    "DatabaseReplicaWarmLeaseTerminalEvidence",
    "PreprocessingStagedDatabasePlacementEvidence",
    "database_replica_lease_evidence_from_mapping",
]
