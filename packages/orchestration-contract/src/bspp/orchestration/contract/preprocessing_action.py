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

"""Strict execution attestations for one preprocessing Runtime Action.

These records add process and content attestations to the existing
``PreprocessingPairedEvidence`` and ``PreprocessingArchiveEvidence`` records.
They do not replace preprocessing state interpretation or Phase authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

from bspp.orchestration.contract.database_direct_result import (
    DatabaseDirectResult,
    database_direct_result_digest,
    database_direct_result_from_mapping,
)
from bspp.orchestration.contract.database_placement import DatabaseAccessPolicy
from bspp.orchestration.contract.database_placement_result import (
    DatabasePlacementFailureEvidence,
    DatabasePostScienceEvidence,
    DatabasePostScienceObservation,
    database_placement_failure_evidence_from_mapping,
    database_post_science_evidence_from_mapping,
)
from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaResult,
    database_replica_result_digest,
    database_replica_result_from_mapping,
)
from bspp.orchestration.contract.database_replica_lease import (
    DatabaseReplicaLeaseTerminalEvidence,
    DatabaseReplicaWarmLeaseTerminalEvidence,
    PreprocessingStagedDatabasePlacementEvidence,
)
from bspp.orchestration.contract.database_replica_result import DatabaseReplicaColdResult
from bspp.orchestration.contract.database_replica_warm_result import DatabaseReplicaWarmResult
from bspp.orchestration.contract.database_set_provisioning import DatabaseSetIdentity
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardAdoptionEvidence,
    attempt_carry_forward_adoption_evidence_from_mapping,
)
from bspp.orchestration.contract.preprocessing_identity import PREPROCESSING_ADAPTER_VERSION
from bspp.orchestration.contract.preprocessing_state import (
    PreprocessingArchiveEvidence,
    PreprocessingPairedEvidence,
    preprocessing_archive_evidence_from_mapping,
    preprocessing_paired_evidence_from_mapping,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

PreprocessingCommandKind = Literal["gpuserver", "search", "record-ls", "tar", "lz4"]
PreprocessingCommandDisposition = Literal[
    "completed",
    "terminated-by-adapter",
    "killed-by-adapter",
    "exited-unexpectedly",
    "failed-to-start",
    "cleanup-failed",
]
PreprocessingActionOutcome = Literal["succeeded", "failed"]
PreprocessingOutputRole = Literal["a3m", "log", "record", "tar", "tar-lz4", "completed-input"]
PreprocessingRawSearchArtifactRole = Literal["named-a3m", "numeric-placeholder"]
PreprocessingDatabasePlacementCommandFailureClassification = Literal[
    "nonzero-after-result",
    "evidence-reconciliation-failed",
]

PREPROCESSING_COMMAND_ORDER: tuple[PreprocessingCommandKind, ...] = (
    "gpuserver",
    "search",
    "record-ls",
    "tar",
    "lz4",
)

_COMMAND_KINDS = frozenset(PREPROCESSING_COMMAND_ORDER)
_COMMAND_DISPOSITIONS = frozenset(
    {
        "completed",
        "terminated-by-adapter",
        "killed-by-adapter",
        "exited-unexpectedly",
        "failed-to-start",
        "cleanup-failed",
    }
)
_OUTPUT_ROLES = frozenset({"a3m", "log", "record", "tar", "tar-lz4", "completed-input"})
_RAW_SEARCH_ARTIFACT_ROLES = frozenset({"named-a3m", "numeric-placeholder"})
_PLACEMENT_COMMAND_FAILURE_CLASSIFICATIONS = frozenset({"nonzero-after-result", "evidence-reconciliation-failed"})
_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_ACTION_ID = re.compile(r"preprocessing-chunk-[0-9]{6}")
_CHUNK_NAME = re.compile(r"[^/]+_tranche\d{2}_\d{5}\.fa")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def preprocessing_command_digest(
    argv: tuple[str, ...],
    environment: tuple[tuple[str, str], ...],
) -> str:
    """Bind one command attestation to its exact argv and environment overlay."""
    payload = {"argv": list(argv), "environment": [list(entry) for entry in environment]}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PreprocessingCommandOutcome:
    """One attempted adapter command, in the action's observed command order."""

    command_kind: PreprocessingCommandKind
    command_digest: str
    disposition: PreprocessingCommandDisposition
    return_code: int | None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingCommandOutcome")
        if self.command_kind not in _COMMAND_KINDS:
            msg = f"unsupported preprocessing command kind: {self.command_kind!r}"
            raise ValueError(msg)
        if _SHA256.fullmatch(self.command_digest) is None:
            msg = "preprocessing command digest must be a lowercase SHA-256"
            raise ValueError(msg)
        if self.disposition not in _COMMAND_DISPOSITIONS:
            msg = f"unsupported preprocessing command disposition: {self.disposition!r}"
            raise ValueError(msg)
        if self.disposition in {"failed-to-start", "cleanup-failed"}:
            if self.return_code is not None:
                msg = "failed-to-start and cleanup-failed commands cannot have a return code"
                raise ValueError(msg)
        elif not isinstance(self.return_code, int) or isinstance(self.return_code, bool):
            msg = "an attempted preprocessing command must have an integer return code"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "command_kind": self.command_kind,
            "command_digest": self.command_digest,
            "disposition": self.disposition,
            "return_code": self.return_code,
        }


@dataclass(frozen=True)
class PreprocessingOutputHash:
    """Content attestation for one exact action output."""

    role: PreprocessingOutputRole
    path: str
    size_bytes: int
    sha256: str
    member_name: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingOutputHash")
        if self.role not in _OUTPUT_ROLES:
            msg = f"unsupported preprocessing output role: {self.role!r}"
            raise ValueError(msg)
        if not self.path:
            msg = "preprocessing output hash path must be non-empty"
            raise ValueError(msg)
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            msg = "preprocessing output hash size_bytes must be a non-negative integer"
            raise ValueError(msg)
        if _SHA256.fullmatch(self.sha256) is None:
            msg = "preprocessing output hash sha256 must be 64 lowercase hexadecimal characters"
            raise ValueError(msg)
        if self.role == "a3m":
            if (
                self.member_name is None
                or not self.member_name.endswith(".a3m")
                or "/" in self.member_name
                or self.path.rsplit("/", maxsplit=1)[-1] != self.member_name
            ):
                msg = "A3M output hashes require a matching top-level member_name"
                raise ValueError(msg)
        elif self.member_name is not None:
            msg = "only A3M output hashes may declare member_name"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "member_name": self.member_name,
        }


@dataclass(frozen=True)
class PreprocessingRawSearchArtifact:
    """One direct child in the exact paired-query raw search closure."""

    role: PreprocessingRawSearchArtifactRole
    member_name: str
    path: str
    size_bytes: int
    sha256: str
    source_ordinal: int | None
    declared_member: str | None
    raw_query_id: int | None
    modeled_chain_length: int | None
    modeled_cardinality: int | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingRawSearchArtifact")
        if self.role not in _RAW_SEARCH_ARTIFACT_ROLES:
            raise ValueError(f"unsupported raw-search artifact role: {self.role!r}")
        if (
            not self.member_name
            or "/" in self.member_name
            or self.member_name in {".", ".."}
            or not self.member_name.endswith(".a3m")
        ):
            raise ValueError("raw-search artifact member_name must be a safe top-level .a3m basename")
        if not self.path or self.path.rsplit("/", maxsplit=1)[-1] != self.member_name:
            raise ValueError("raw-search artifact path must end in its exact member_name")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes <= 0:
            raise ValueError("raw-search artifact size_bytes must be a positive integer")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("raw-search artifact sha256 must be 64 lowercase hexadecimal characters")
        if self.role == "named-a3m":
            if (
                not isinstance(self.source_ordinal, int)
                or isinstance(self.source_ordinal, bool)
                or self.source_ordinal < 0
                or self.declared_member != self.member_name
                or self.raw_query_id is not None
                or self.modeled_chain_length is not None
                or self.modeled_cardinality is not None
            ):
                raise ValueError("named raw A3Ms require source_ordinal and matching declared_member only")
        elif (
            self.source_ordinal is not None
            or self.declared_member is not None
            or not isinstance(self.raw_query_id, int)
            or isinstance(self.raw_query_id, bool)
            or self.raw_query_id < 0
            or not isinstance(self.modeled_chain_length, int)
            or isinstance(self.modeled_chain_length, bool)
            or self.modeled_chain_length <= 0
            or self.member_name != f"{self.raw_query_id}.a3m"
        ):
            raise ValueError("numeric raw placeholders require matching raw_query_id and modeled_chain_length only")
        if self.role == "numeric-placeholder":
            if self.modeled_cardinality is None:
                object.__setattr__(self, "modeled_cardinality", 1)
            if (
                not isinstance(self.modeled_cardinality, int)
                or isinstance(self.modeled_cardinality, bool)
                or self.modeled_cardinality <= 0
            ):
                raise ValueError("numeric raw placeholder modeled_cardinality must be a positive integer")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "role": self.role,
            "member_name": self.member_name,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "source_ordinal": self.source_ordinal,
            "declared_member": self.declared_member,
            "raw_query_id": self.raw_query_id,
            "modeled_chain_length": self.modeled_chain_length,
        }
        # Only serialize modeled_cardinality when it carries a non-default
        # value (not None for named-a3m, not 1 for numeric-placeholder) so old
        # sealed evidence digests remain stable.
        if self.modeled_cardinality is not None and self.modeled_cardinality != 1:
            result["modeled_cardinality"] = self.modeled_cardinality
        return result


@dataclass(frozen=True)
class PreprocessingRawSearchEvidence:
    """Exact deterministic M-named plus M-placeholder raw inventory."""

    raw_search_output_directory: str
    searched_source_ordinals: tuple[int, ...]
    artifacts: tuple[PreprocessingRawSearchArtifact, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingRawSearchEvidence")
        if not self.raw_search_output_directory:
            raise ValueError("raw-search evidence directory must be non-empty")
        if (
            not isinstance(self.searched_source_ordinals, tuple)
            or not self.searched_source_ordinals
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in self.searched_source_ordinals
            )
            or len(set(self.searched_source_ordinals)) != len(self.searched_source_ordinals)
        ):
            raise ValueError("raw-search evidence requires unique non-negative searched source ordinals")
        if not isinstance(self.artifacts, tuple):
            raise ValueError("raw-search artifacts must be an immutable tuple")
        count = len(self.searched_source_ordinals)
        total = len(self.artifacts)
        if total < count:
            raise ValueError("raw-search evidence requires at least M named A3Ms")
        named = self.artifacts[:count]
        numeric = self.artifacts[count:]
        if tuple(item.role for item in named) != ("named-a3m",) * count:
            raise ValueError("raw-search named artifacts must precede numeric placeholders")
        if tuple(item.source_ordinal for item in named) != self.searched_source_ordinals:
            raise ValueError("raw-search named artifacts must follow searched source order")
        expected_raw_ids = tuple(range(count, total))
        if (
            tuple(item.role for item in numeric) != ("numeric-placeholder",) * (total - count)
            or tuple(item.raw_query_id for item in numeric) != expected_raw_ids
        ):
            raise ValueError("raw-search placeholders must follow exact M through (U-1) raw-id order")
        paths = tuple(item.path for item in self.artifacts)
        members = tuple(item.member_name for item in self.artifacts)
        if len(set(paths)) != len(paths) or len(set(members)) != len(members):
            raise ValueError("raw-search artifact paths and members must be unique")
        prefix = self.raw_search_output_directory.rstrip("/") + "/"
        if any(item.path != prefix + item.member_name for item in self.artifacts):
            raise ValueError("raw-search artifact paths must be direct children of the declared directory")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "raw_search_output_directory": self.raw_search_output_directory,
            "searched_source_ordinals": list(self.searched_source_ordinals),
            "artifacts": [item.to_mapping() for item in self.artifacts],
        }


@dataclass(frozen=True)
class PreprocessingDatabasePlacementEvidence:
    """Direct-placement authority nested separately from scientific outcomes."""

    result: DatabaseDirectResult | None
    result_digest: str | None
    failure: DatabasePlacementFailureEvidence | None
    science_started: bool
    post_science_observation: DatabasePostScienceEvidence | None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingDatabasePlacementEvidence")
        if not isinstance(self.science_started, bool):
            raise ValueError("database placement science_started must be a boolean")
        if (self.result is None) == (self.failure is None):
            raise ValueError("database placement result and failure must be mutually exclusive")
        if self.failure is not None:
            if self.result_digest is not None or self.post_science_observation is not None or self.science_started:
                raise ValueError("pre-science placement failure cannot retain a result, post observation, or science")
            return
        assert self.result is not None
        if self.result_digest != database_direct_result_digest(self.result):
            raise ValueError("database placement result digest must bind exact canonical Result bytes")
        if self.science_started != (self.post_science_observation is not None):
            raise ValueError("science_started must exactly govern the post-science observation")
        if (
            self.post_science_observation is not None
            and self.post_science_observation.source_manifest_sha256 != self.result.source_manifest_sha256
        ):
            raise ValueError("post-science observation must bind the Result source manifest digest")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "result": None if self.result is None else self.result.to_mapping(),
            "result_digest": self.result_digest,
            "failure": None if self.failure is None else self.failure.to_mapping(),
            "science_started": self.science_started,
            "post_science_observation": (
                None if self.post_science_observation is None else self.post_science_observation.to_mapping()
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PreprocessingDatabasePlacementEvidence:
        _reject_unknown_fields(
            payload,
            {
                "schema_version",
                "result",
                "result_digest",
                "failure",
                "science_started",
                "post_science_observation",
            },
            "PreprocessingDatabasePlacementEvidence",
        )
        if set(payload) != {
            "schema_version",
            "result",
            "result_digest",
            "failure",
            "science_started",
            "post_science_observation",
        }:
            raise ValueError("PreprocessingDatabasePlacementEvidence requires its exact canonical fields")
        result_payload = payload.get("result")
        failure_payload = payload.get("failure")
        post_payload = payload.get("post_science_observation")
        science_started = payload.get("science_started")
        if result_payload is not None and not isinstance(result_payload, Mapping):
            raise ValueError("database placement result must be a mapping or null")
        if failure_payload is not None and not isinstance(failure_payload, Mapping):
            raise ValueError("database placement failure must be a mapping or null")
        if post_payload is not None and not isinstance(post_payload, Mapping):
            raise ValueError("post_science_observation must be a mapping or null")
        if not isinstance(science_started, bool):
            raise ValueError("database placement science_started must be a boolean")
        return cls(
            result=(
                database_direct_result_from_mapping(cast("Mapping[str, object]", result_payload))
                if result_payload is not None
                else None
            ),
            result_digest=_required_optional_str(payload, "result_digest"),
            failure=(
                database_placement_failure_evidence_from_mapping(cast("Mapping[str, object]", failure_payload))
                if failure_payload is not None
                else None
            ),
            science_started=science_started,
            post_science_observation=(
                database_post_science_evidence_from_mapping(cast("Mapping[str, object]", post_payload))
                if post_payload is not None
                else None
            ),
            schema_version=validate_schema_version(
                payload.get("schema_version"), record_name="PreprocessingDatabasePlacementEvidence"
            ),
        )


@dataclass(frozen=True)
class PreprocessingDatabasePlacementCommandFailureEvidence:
    """Action-level failure when placement command authority cannot select science."""

    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_id: str
    database_set: DatabaseSetIdentity
    requested_policy: DatabaseAccessPolicy
    source_manifest_sha256: str
    placement_process_status: int
    result: DatabaseDirectResult | DatabaseReplicaResult | None
    result_digest: str | None
    science_started: Literal[False]
    classification: PreprocessingDatabasePlacementCommandFailureClassification
    error: str
    placement_evidence_kind: Literal["command-failure"] = "command-failure"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(
            self.schema_version,
            "PreprocessingDatabasePlacementCommandFailureEvidence",
        )
        if self.placement_evidence_kind != "command-failure":
            raise ValueError("placement command failure requires its exact discriminator")
        if _PHASE_RUN_ID.fullmatch(self.phase_run_id) is None:
            raise ValueError("placement command failure has an invalid Phase Run id")
        if _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise ValueError("placement command failure has an invalid Phase Attempt id")
        if _SHA256.fullmatch(self.phase_runspec_digest) is None:
            raise ValueError("placement command failure RunSpec digest must be a lowercase SHA-256")
        if _ACTION_ID.fullmatch(self.action_id) is None:
            raise ValueError("placement command failure has an invalid action id")
        if not isinstance(self.database_set, DatabaseSetIdentity):
            raise ValueError("placement command failure requires an exact Database Set identity")
        if not isinstance(self.requested_policy, DatabaseAccessPolicy):
            raise ValueError("placement command failure requires a canonical Database Access Policy")
        if _SHA256.fullmatch(self.source_manifest_sha256) is None:
            raise ValueError("placement command failure source manifest must be a lowercase SHA-256")
        if (
            not isinstance(self.placement_process_status, int)
            or isinstance(self.placement_process_status, bool)
            or not 0 <= self.placement_process_status <= 255
        ):
            raise ValueError("placement process status must be a non-boolean integer in 0..255")
        if self.science_started is not False:
            raise ValueError("placement command failure science_started must be false")
        if self.classification not in _PLACEMENT_COMMAND_FAILURE_CLASSIFICATIONS:
            raise ValueError(f"unsupported placement command failure classification: {self.classification!r}")
        if not self.error or len(self.error) > 2048 or "\x00" in self.error:
            raise ValueError("placement command failure error must be non-empty and bounded to 2048 characters")
        if self.classification == "evidence-reconciliation-failed":
            if self.result is not None or self.result_digest is not None:
                raise ValueError("evidence reconciliation failure cannot retain unauthorized Result authority")
            return
        if self.placement_process_status == 0:
            raise ValueError("nonzero-after-result requires a nonzero placement process status")
        if self.result is None:
            raise ValueError("nonzero-after-result requires one exact placement Result")
        expected_digest = _placement_command_result_digest(self.result)
        if self.result_digest != expected_digest:
            raise ValueError("placement command failure Result digest must bind exact canonical Result bytes")
        if (
            self.result.phase_run_id != self.phase_run_id
            or self.result.attempt_id != self.attempt_id
            or self.result.phase_runspec_digest != self.phase_runspec_digest
            or self.result.action_id != self.action_id
            or self.result.database_set != self.database_set
            or self.result.requested_policy != self.requested_policy
            or self.result.source_manifest_sha256 != self.source_manifest_sha256
        ):
            raise ValueError("placement command failure Result must match outer trusted identity")
        if self.requested_policy == DatabaseAccessPolicy.DIRECT and isinstance(
            self.result, DatabaseReplicaColdResult | DatabaseReplicaWarmResult
        ):
            raise ValueError("direct placement command failure requires the direct Result family")
        if self.requested_policy == DatabaseAccessPolicy.STAGE_REQUIRED and not isinstance(
            self.result, DatabaseReplicaColdResult | DatabaseReplicaWarmResult
        ):
            raise ValueError("stage-required placement command failure requires the staged Result family")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "placement_evidence_kind": self.placement_evidence_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_id": self.action_id,
            "database_set": self.database_set.to_mapping(),
            "requested_policy": self.requested_policy.value,
            "source_manifest_sha256": self.source_manifest_sha256,
            "placement_process_status": self.placement_process_status,
            "result": None if self.result is None else self.result.to_mapping(),
            "result_digest": self.result_digest,
            "science_started": self.science_started,
            "classification": self.classification,
            "error": self.error,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PreprocessingDatabasePlacementCommandFailureEvidence:
        fields = {
            "schema_version",
            "placement_evidence_kind",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "action_id",
            "database_set",
            "requested_policy",
            "source_manifest_sha256",
            "placement_process_status",
            "result",
            "result_digest",
            "science_started",
            "classification",
            "error",
        }
        if set(payload) != fields:
            raise ValueError("PreprocessingDatabasePlacementCommandFailureEvidence requires its exact canonical fields")
        database_set_payload = _required_mapping(payload, "database_set")
        result_payload = payload.get("result")
        if result_payload is not None and not isinstance(result_payload, Mapping):
            raise ValueError("placement command failure Result must be a mapping or null")
        policy = _required_str(payload, "requested_policy")
        if policy not in {item.value for item in DatabaseAccessPolicy}:
            raise ValueError("placement command failure requested_policy must be canonical")
        classification = _required_str(payload, "classification")
        science_started = payload.get("science_started")
        if science_started is not False:
            raise ValueError("placement command failure science_started must be false")
        discriminator = _required_str(payload, "placement_evidence_kind")
        return cls(
            phase_run_id=_required_str(payload, "phase_run_id"),
            attempt_id=_required_str(payload, "attempt_id"),
            phase_runspec_digest=_required_str(payload, "phase_runspec_digest"),
            action_id=_required_str(payload, "action_id"),
            database_set=_database_set_identity_from_mapping(database_set_payload),
            requested_policy=DatabaseAccessPolicy(policy),
            source_manifest_sha256=_required_str(payload, "source_manifest_sha256"),
            placement_process_status=_required_int(payload, "placement_process_status"),
            result=(
                _placement_command_result_from_mapping(cast("Mapping[str, object]", result_payload))
                if result_payload is not None
                else None
            ),
            result_digest=_required_optional_str(payload, "result_digest"),
            science_started=science_started,
            classification=cast("PreprocessingDatabasePlacementCommandFailureClassification", classification),
            error=_required_str(payload, "error"),
            placement_evidence_kind=cast("Literal['command-failure']", discriminator),
            schema_version=validate_schema_version(
                payload.get("schema_version"),
                record_name="PreprocessingDatabasePlacementCommandFailureEvidence",
            ),
        )


@dataclass(frozen=True)
class PreprocessingChunkActionEvidence:
    """Additional execution-time attestation for one declared chunk action."""

    adapter_version: str
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_id: str
    placement_process_status: int
    chunk_name: str
    started_at: str
    finished_at: str
    outcome: PreprocessingActionOutcome
    command_outcomes: tuple[PreprocessingCommandOutcome, ...]
    paired_evidence: PreprocessingPairedEvidence
    archive_evidence: PreprocessingArchiveEvidence
    output_hashes: tuple[PreprocessingOutputHash, ...]
    error: str | None
    database_placement: (
        PreprocessingDatabasePlacementEvidence
        | PreprocessingStagedDatabasePlacementEvidence
        | PreprocessingDatabasePlacementCommandFailureEvidence
    )
    raw_search_evidence: PreprocessingRawSearchEvidence | None = None
    carry_forward_adoption: AttemptCarryForwardAdoptionEvidence | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingChunkActionEvidence")
        if self.adapter_version != PREPROCESSING_ADAPTER_VERSION:
            raise ValueError(f"unsupported preprocessing adapter version: {self.adapter_version!r}")
        if _PHASE_RUN_ID.fullmatch(self.phase_run_id) is None:
            msg = "preprocessing action evidence has an invalid Phase Run id"
            raise ValueError(msg)
        if _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            msg = "preprocessing action evidence has an invalid Phase Attempt id"
            raise ValueError(msg)
        if _SHA256.fullmatch(self.phase_runspec_digest) is None:
            msg = "preprocessing action evidence RunSpec digest must be a lowercase SHA-256"
            raise ValueError(msg)
        if _ACTION_ID.fullmatch(self.action_id) is None:
            msg = "preprocessing action evidence has an invalid action id"
            raise ValueError(msg)
        if (
            not isinstance(self.placement_process_status, int)
            or isinstance(self.placement_process_status, bool)
            or not 0 <= self.placement_process_status <= 255
        ):
            raise ValueError("placement process status must be a non-boolean integer in 0..255")
        if _CHUNK_NAME.fullmatch(self.chunk_name) is None:
            msg = "preprocessing action evidence has an invalid chunk name"
            raise ValueError(msg)
        started = _parse_timestamp(self.started_at, "started_at")
        finished = _parse_timestamp(self.finished_at, "finished_at")
        if finished < started:
            msg = "preprocessing action evidence cannot finish before it starts"
            raise ValueError(msg)
        if self.outcome not in {"succeeded", "failed"}:
            msg = f"unsupported preprocessing action outcome: {self.outcome!r}"
            raise ValueError(msg)
        if not isinstance(self.command_outcomes, tuple) or not isinstance(self.output_hashes, tuple):
            msg = "preprocessing action evidence collections must be immutable tuples"
            raise ValueError(msg)
        observed_command_order = tuple(item.command_kind for item in self.command_outcomes)
        if observed_command_order != PREPROCESSING_COMMAND_ORDER[: len(observed_command_order)]:
            msg = "preprocessing command outcomes must be an ordered prefix of the five-command action"
            raise ValueError(msg)
        if self.paired_evidence.chunk_name != self.chunk_name or self.archive_evidence.chunk_name != self.chunk_name:
            msg = "nested preprocessing evidence must reference the action chunk"
            raise ValueError(msg)
        _validate_database_placement_identity(self)
        _validate_output_hashes(self.output_hashes)
        _validate_nested_output_consistency(self)
        if self.outcome == "succeeded" and self.raw_search_evidence is None:
            raise ValueError("successful adapter-v3 evidence requires raw-search evidence")
        if self.outcome == "succeeded":
            _validate_success(self)
        elif not self.error:
            msg = "failed preprocessing action evidence requires an error"
            raise ValueError(msg)
        placement = self.database_placement
        pre_science_placement_failure = (
            isinstance(placement, PreprocessingDatabasePlacementCommandFailureEvidence) or placement.failure is not None
        )
        if pre_science_placement_failure and (
            self.outcome != "failed"
            or self.command_outcomes
            or self.output_hashes
            or self.raw_search_evidence is not None
            or self.paired_evidence.record_lines is not None
            or self.paired_evidence.log_lines is not None
            or self.archive_evidence.tar_size_bytes is not None
            or self.archive_evidence.lz4_size_bytes is not None
            or self.archive_evidence.tar_members is not None
            or self.carry_forward_adoption is not None
        ):
            raise ValueError("pre-science placement failure action evidence must be payload-free")
        if isinstance(placement, PreprocessingDatabasePlacementCommandFailureEvidence):
            if self.placement_process_status != placement.placement_process_status:
                raise ValueError("action and placement command failure statuses must match exactly")
        elif placement.failure is None and self.placement_process_status != 0:
            raise ValueError("Result-selected action evidence requires zero placement process status")
        if (
            isinstance(placement, PreprocessingStagedDatabasePlacementEvidence)
            and isinstance(
                placement.lease_outcome,
                DatabaseReplicaLeaseTerminalEvidence | DatabaseReplicaWarmLeaseTerminalEvidence,
            )
            and not placement.lease_outcome.kernel_started
            and (
                self.outcome != "failed"
                or len(self.command_outcomes) != 1
                or self.command_outcomes[0].command_kind != "gpuserver"
                or self.command_outcomes[0].disposition != "failed-to-start"
                or self.command_outcomes[0].return_code is not None
                or self.output_hashes
                or self.raw_search_evidence is not None
            )
        ):
            raise ValueError("no-kernel staged evidence requires the exact failed-to-start action shape")
        if (
            isinstance(placement, PreprocessingStagedDatabasePlacementEvidence)
            and self.outcome == "failed"
            and (self.output_hashes or self.raw_search_evidence is not None)
        ):
            raise ValueError("failed staged action evidence cannot retain accepted science evidence")
        if self.carry_forward_adoption is not None and (
            self.carry_forward_adoption.phase_run_id != self.phase_run_id
            or self.carry_forward_adoption.attempt_id != self.attempt_id
            or self.carry_forward_adoption.phase_runspec_digest != self.phase_runspec_digest
            or self.carry_forward_adoption.action_id != self.action_id
        ):
            raise ValueError("carry-forward adoption evidence must match action evidence identity")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "adapter_version": self.adapter_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_id": self.action_id,
            "placement_process_status": self.placement_process_status,
            "chunk_name": self.chunk_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outcome": self.outcome,
            "command_outcomes": [item.to_mapping() for item in self.command_outcomes],
            "paired_evidence": self.paired_evidence.to_mapping(),
            "archive_evidence": self.archive_evidence.to_mapping(),
            "output_hashes": [item.to_mapping() for item in self.output_hashes],
            "error": self.error,
            "database_placement": self.database_placement.to_mapping(),
        }
        if self.raw_search_evidence is not None:
            result["raw_search_evidence"] = self.raw_search_evidence.to_mapping()
        if self.carry_forward_adoption is not None:
            result["carry_forward_adoption"] = self.carry_forward_adoption.to_mapping()
        return result


def preprocessing_command_outcome_from_mapping(payload: Mapping[str, object]) -> PreprocessingCommandOutcome:
    _reject_unknown_fields(
        payload,
        {"schema_version", "command_kind", "command_digest", "disposition", "return_code"},
        "PreprocessingCommandOutcome",
    )
    command_kind = _required_str(payload, "command_kind")
    disposition = _required_str(payload, "disposition")
    return PreprocessingCommandOutcome(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingCommandOutcome"
        ),
        command_kind=cast("PreprocessingCommandKind", command_kind),
        command_digest=_required_str(payload, "command_digest"),
        disposition=cast("PreprocessingCommandDisposition", disposition),
        return_code=_required_optional_int(payload, "return_code"),
    )


def preprocessing_output_hash_from_mapping(payload: Mapping[str, object]) -> PreprocessingOutputHash:
    _reject_unknown_fields(
        payload,
        {"schema_version", "role", "path", "size_bytes", "sha256", "member_name"},
        "PreprocessingOutputHash",
    )
    return PreprocessingOutputHash(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PreprocessingOutputHash"),
        role=cast("PreprocessingOutputRole", _required_str(payload, "role")),
        path=_required_str(payload, "path"),
        size_bytes=_required_int(payload, "size_bytes"),
        sha256=_required_str(payload, "sha256"),
        member_name=_required_optional_str(payload, "member_name"),
    )


def preprocessing_raw_search_artifact_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingRawSearchArtifact:
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "role",
            "member_name",
            "path",
            "size_bytes",
            "sha256",
            "source_ordinal",
            "declared_member",
            "raw_query_id",
            "modeled_chain_length",
            "modeled_cardinality",
        },
        "PreprocessingRawSearchArtifact",
    )
    return PreprocessingRawSearchArtifact(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingRawSearchArtifact"
        ),
        role=cast("PreprocessingRawSearchArtifactRole", _required_str(payload, "role")),
        member_name=_required_str(payload, "member_name"),
        path=_required_str(payload, "path"),
        size_bytes=_required_int(payload, "size_bytes"),
        sha256=_required_str(payload, "sha256"),
        source_ordinal=_required_optional_int(payload, "source_ordinal"),
        declared_member=_required_optional_str(payload, "declared_member"),
        raw_query_id=_required_optional_int(payload, "raw_query_id"),
        modeled_chain_length=_required_optional_int(payload, "modeled_chain_length"),
        modeled_cardinality=(
            _required_optional_int(payload, "modeled_cardinality") if "modeled_cardinality" in payload else None
        ),
    )


def preprocessing_raw_search_evidence_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingRawSearchEvidence:
    _reject_unknown_fields(
        payload,
        {"schema_version", "raw_search_output_directory", "searched_source_ordinals", "artifacts"},
        "PreprocessingRawSearchEvidence",
    )
    return PreprocessingRawSearchEvidence(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingRawSearchEvidence"
        ),
        raw_search_output_directory=_required_str(payload, "raw_search_output_directory"),
        searched_source_ordinals=_required_int_tuple(payload, "searched_source_ordinals"),
        artifacts=tuple(
            preprocessing_raw_search_artifact_from_mapping(item)
            for item in _required_mapping_sequence(payload, "artifacts")
        ),
    )


def preprocessing_chunk_action_evidence_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingChunkActionEvidence:
    """Strict-load an additive preprocessing action attestation."""
    placement_mapping = payload.get("database_placement")
    schema_checked_payload = dict(payload)
    schema_checked_payload.pop("database_placement", None)
    _require_explicit_schema_versions(schema_checked_payload, path=("preprocessing_chunk_action_evidence",))
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "adapter_version",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "action_id",
            "placement_process_status",
            "chunk_name",
            "started_at",
            "finished_at",
            "outcome",
            "command_outcomes",
            "paired_evidence",
            "archive_evidence",
            "output_hashes",
            "error",
            "database_placement",
            "raw_search_evidence",
            "carry_forward_adoption",
        },
        "PreprocessingChunkActionEvidence",
    )
    adapter_version = _required_str(payload, "adapter_version")
    if adapter_version != PREPROCESSING_ADAPTER_VERSION:
        raise ValueError(f"unsupported preprocessing adapter version: {adapter_version!r}")
    adoption_mapping = payload.get("carry_forward_adoption")
    if "carry_forward_adoption" in payload and not isinstance(adoption_mapping, Mapping):
        raise ValueError("carry_forward_adoption must be a mapping when present")
    raw_mapping = payload.get("raw_search_evidence")
    if "raw_search_evidence" in payload and not isinstance(raw_mapping, Mapping):
        raise ValueError("raw_search_evidence must be a mapping when present")
    return PreprocessingChunkActionEvidence(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingChunkActionEvidence"
        ),
        adapter_version=adapter_version,
        phase_run_id=_required_str(payload, "phase_run_id"),
        attempt_id=_required_str(payload, "attempt_id"),
        phase_runspec_digest=_required_str(payload, "phase_runspec_digest"),
        action_id=_required_str(payload, "action_id"),
        placement_process_status=_required_int(payload, "placement_process_status"),
        chunk_name=_required_str(payload, "chunk_name"),
        started_at=_required_str(payload, "started_at"),
        finished_at=_required_str(payload, "finished_at"),
        outcome=cast("PreprocessingActionOutcome", _required_str(payload, "outcome")),
        command_outcomes=tuple(
            preprocessing_command_outcome_from_mapping(item)
            for item in _required_mapping_sequence(payload, "command_outcomes")
        ),
        paired_evidence=preprocessing_paired_evidence_from_mapping(_required_mapping(payload, "paired_evidence")),
        archive_evidence=preprocessing_archive_evidence_from_mapping(_required_mapping(payload, "archive_evidence")),
        output_hashes=tuple(
            preprocessing_output_hash_from_mapping(item)
            for item in _required_mapping_sequence(payload, "output_hashes")
        ),
        error=_required_optional_str(payload, "error"),
        database_placement=_database_placement_evidence_from_mapping(
            cast("Mapping[str, object]", placement_mapping)
            if isinstance(placement_mapping, Mapping)
            else _required_mapping(payload, "database_placement")
        ),
        raw_search_evidence=(
            preprocessing_raw_search_evidence_from_mapping(cast("Mapping[str, object]", raw_mapping))
            if raw_mapping is not None
            else None
        ),
        carry_forward_adoption=(
            attempt_carry_forward_adoption_evidence_from_mapping(cast("Mapping[str, object]", adoption_mapping))
            if adoption_mapping is not None
            else None
        ),
    )


def _validate_success(evidence: PreprocessingChunkActionEvidence) -> None:
    if evidence.error is not None:
        msg = "successful preprocessing action evidence cannot contain an error"
        raise ValueError(msg)
    placement = evidence.database_placement
    if isinstance(placement, PreprocessingDatabasePlacementEvidence):
        if (
            placement.result is None
            or not isinstance(placement.post_science_observation, DatabasePostScienceObservation)
            or not placement.post_science_observation.matches(placement.result.pre_science_observation)
        ):
            raise ValueError(
                "successful preprocessing evidence requires an unchanged direct Database Source observation"
            )
    elif isinstance(placement, PreprocessingStagedDatabasePlacementEvidence):
        lease = placement.lease_outcome
        valid_family = (
            isinstance(placement.result, DatabaseReplicaColdResult)
            and placement.result.outcome.value == "replica-cold"
            and isinstance(lease, DatabaseReplicaLeaseTerminalEvidence)
        ) or (
            isinstance(placement.result, DatabaseReplicaWarmResult)
            and placement.result.outcome.value == "replica-warm"
            and isinstance(lease, DatabaseReplicaWarmLeaseTerminalEvidence)
        )
        if (
            placement.result is None
            or placement.failure is not None
            or not placement.science_started
            or placement.result.requested_policy.value not in {"stage-required", "stage-preferred"}
            or placement.result.branch_kind != "staged"
            or placement.result.verification != "metadata-verified"
            or not valid_family
            or not isinstance(
                lease,
                DatabaseReplicaLeaseTerminalEvidence | DatabaseReplicaWarmLeaseTerminalEvidence,
            )
            or not lease.kernel_started
            or not lease.held_through_kernel_exit
        ):
            raise ValueError("successful staged preprocessing evidence requires held terminal lease authority")
    else:
        raise ValueError("successful preprocessing evidence has an unsupported placement wrapper")
    if tuple(item.command_kind for item in evidence.command_outcomes) != PREPROCESSING_COMMAND_ORDER:
        msg = "successful preprocessing action evidence requires all five commands in order"
        raise ValueError(msg)
    gpuserver = evidence.command_outcomes[0]
    if gpuserver.disposition not in {"terminated-by-adapter", "killed-by-adapter"}:
        msg = "successful preprocessing action evidence requires adapter-owned gpuserver shutdown"
        raise ValueError(msg)
    if any(item.disposition != "completed" or item.return_code != 0 for item in evidence.command_outcomes[1:]):
        msg = "successful preprocessing action evidence requires zero-returning foreground commands"
        raise ValueError(msg)
    roles = tuple(item.role for item in evidence.output_hashes)
    for required in ("log", "record", "tar", "tar-lz4", "completed-input"):
        if roles.count(required) != 1:
            msg = f"successful preprocessing action evidence requires exactly one {required} output hash"
            raise ValueError(msg)
    if "a3m" not in roles:
        msg = "successful preprocessing action evidence requires at least one A3M output hash"
        raise ValueError(msg)
    if any(item.size_bytes <= 0 for item in evidence.output_hashes):
        msg = "successful preprocessing action evidence requires nonempty outputs"
        raise ValueError(msg)
    if (
        not evidence.paired_evidence.record_lines
        or not evidence.paired_evidence.log_lines
        or evidence.archive_evidence.tar_size_bytes is None
        or evidence.archive_evidence.lz4_size_bytes is None
        or evidence.archive_evidence.tar_members is None
    ):
        msg = "successful preprocessing action evidence requires complete paired and archive observations"
        raise ValueError(msg)


def _validate_database_placement_identity(evidence: PreprocessingChunkActionEvidence) -> None:
    placement = evidence.database_placement
    if isinstance(placement, PreprocessingDatabasePlacementCommandFailureEvidence):
        if (
            placement.phase_run_id != evidence.phase_run_id
            or placement.attempt_id != evidence.attempt_id
            or placement.phase_runspec_digest != evidence.phase_runspec_digest
            or placement.action_id != evidence.action_id
        ):
            raise ValueError("nested Database Placement command failure must match action evidence identity")
        return
    authority = placement.result if placement.result is not None else placement.failure
    assert authority is not None
    if (
        authority.phase_run_id != evidence.phase_run_id
        or authority.attempt_id != evidence.attempt_id
        or authority.phase_runspec_digest != evidence.phase_runspec_digest
        or authority.action_id != evidence.action_id
    ):
        raise ValueError("nested Database Placement evidence must match action evidence identity")


def _database_placement_evidence_from_mapping(
    payload: Mapping[str, object],
) -> (
    PreprocessingDatabasePlacementEvidence
    | PreprocessingStagedDatabasePlacementEvidence
    | PreprocessingDatabasePlacementCommandFailureEvidence
):
    if payload.get("placement_evidence_kind") == "command-failure":
        return PreprocessingDatabasePlacementCommandFailureEvidence.from_mapping(payload)
    if "lease_outcome" in payload:
        return PreprocessingStagedDatabasePlacementEvidence.from_mapping(payload)
    return PreprocessingDatabasePlacementEvidence.from_mapping(payload)


def _placement_command_result_from_mapping(
    payload: Mapping[str, object],
) -> DatabaseDirectResult | DatabaseReplicaResult:
    if set(payload) in ({"database_placement_result"}, {"database_capacity_fallback_result"}):
        return database_direct_result_from_mapping(payload)
    return database_replica_result_from_mapping(payload)


def _placement_command_result_digest(result: DatabaseDirectResult | DatabaseReplicaResult) -> str:
    if isinstance(result, DatabaseReplicaColdResult | DatabaseReplicaWarmResult):
        return database_replica_result_digest(result)
    return database_direct_result_digest(result)


def _database_set_identity_from_mapping(payload: Mapping[str, object]) -> DatabaseSetIdentity:
    if set(payload) != {"identifier", "version"}:
        raise ValueError("database_set requires exact identifier and version fields")
    return DatabaseSetIdentity(
        identifier=_required_str(payload, "identifier"),
        version=_required_str(payload, "version"),
    )


def _validate_output_hashes(outputs: tuple[PreprocessingOutputHash, ...]) -> None:
    paths = tuple(item.path for item in outputs)
    if len(set(paths)) != len(paths):
        msg = "preprocessing output hash paths must be unique"
        raise ValueError(msg)
    a3m_members = tuple(item.member_name for item in outputs if item.role == "a3m")
    if len(set(a3m_members)) != len(a3m_members):
        msg = "preprocessing A3M output hash members must be unique"
        raise ValueError(msg)
    fixed_roles = tuple(item.role for item in outputs if item.role != "a3m")
    if len(set(fixed_roles)) != len(fixed_roles):
        msg = "non-A3M preprocessing output hash roles must be unique"
        raise ValueError(msg)


def _validate_nested_output_consistency(evidence: PreprocessingChunkActionEvidence) -> None:
    by_role: dict[str, PreprocessingOutputHash] = {
        item.role: item for item in evidence.output_hashes if item.role != "a3m"
    }
    paired = evidence.paired_evidence
    archive = evidence.archive_evidence
    for role, path in (("record", paired.durable_record_path), ("log", paired.durable_log_path)):
        item = by_role.get(role)
        if item is not None and item.path != path:
            msg = f"{role} output hash path must match nested paired evidence"
            raise ValueError(msg)
    for role, path, size in (
        ("tar", archive.durable_tar_path, archive.tar_size_bytes),
        ("tar-lz4", archive.durable_lz4_path, archive.lz4_size_bytes),
    ):
        item = by_role.get(role)
        if item is not None and (item.path != path or (size is not None and item.size_bytes != size)):
            msg = f"{role} output hash must match nested archive evidence"
            raise ValueError(msg)


def _parse_timestamp(value: str, field_name: str) -> datetime:
    if not value.endswith("Z"):
        msg = f"{field_name} must be an RFC 3339 UTC timestamp ending in Z"
        raise ValueError(msg)
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        msg = f"{field_name} must be an RFC 3339 UTC timestamp"
        raise ValueError(msg) from exc
    if parsed.utcoffset() is None:
        msg = f"{field_name} must include timezone authority"
        raise ValueError(msg)
    return parsed


def _require_explicit_schema_versions(value: object, *, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        location = ".".join(path)
        if "schema_version" not in value:
            msg = f"{location}.schema_version must be declared explicitly"
            raise ValueError(msg)
        for key, item in value.items():
            _require_explicit_schema_versions(item, path=(*path, str(key)))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _require_explicit_schema_versions(item, path=(*path, str(index)))


def _validate_direct_schema_version(schema_version: int, record_name: str) -> None:
    validated = validate_schema_version(schema_version, record_name=record_name)
    if validated != schema_version:
        msg = f"{record_name} schema_version must be declared explicitly"
        raise ValueError(msg)


def _reject_unknown_fields(payload: Mapping[str, object], allowed: set[str], record_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        msg = f"Unknown {record_name} field(s): {', '.join(unknown)}"
        raise ValueError(msg)


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be a mapping"
        raise ValueError(msg)
    return value


def _required_mapping_sequence(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"{key}[{index}] must be a mapping"
            raise ValueError(msg)
        result.append(item)
    return tuple(result)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{key} must be a non-empty string"
        raise ValueError(msg)
    return value


def _required_optional_str(payload: Mapping[str, object], key: str) -> str | None:
    if key not in payload:
        msg = f"{key} is required"
        raise ValueError(msg)
    value = payload[key]
    if value is None:
        return None
    return _required_str(payload, key)


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


def _required_optional_int(payload: Mapping[str, object], key: str) -> int | None:
    if key not in payload:
        msg = f"{key} is required"
        raise ValueError(msg)
    value = payload[key]
    if value is None:
        return None
    return _required_int(payload, key)


def _required_int_tuple(payload: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        raise ValueError(f"{key} must be a list")
    result: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{key}[{index}] must be an integer")
        result.append(item)
    return tuple(result)


__all__ = [
    "PREPROCESSING_COMMAND_ORDER",
    "PreprocessingActionOutcome",
    "PreprocessingChunkActionEvidence",
    "PreprocessingCommandDisposition",
    "PreprocessingCommandKind",
    "PreprocessingCommandOutcome",
    "PreprocessingDatabasePlacementCommandFailureClassification",
    "PreprocessingDatabasePlacementCommandFailureEvidence",
    "PreprocessingDatabasePlacementEvidence",
    "PreprocessingOutputHash",
    "PreprocessingOutputRole",
    "PreprocessingRawSearchArtifact",
    "PreprocessingRawSearchArtifactRole",
    "PreprocessingRawSearchEvidence",
    "PreprocessingStagedDatabasePlacementEvidence",
    "preprocessing_chunk_action_evidence_from_mapping",
    "preprocessing_command_digest",
    "preprocessing_command_outcome_from_mapping",
    "preprocessing_output_hash_from_mapping",
    "preprocessing_raw_search_artifact_from_mapping",
    "preprocessing_raw_search_evidence_from_mapping",
]
