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

"""Durable direct-Slurm submission intent and assignment contracts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Literal, Protocol, cast

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

PhaseSubmissionIntendedEventType = Literal["phase-submission-intended"]
PhaseActionDispatchIntendedEventType = Literal["phase-action-dispatch-intended"]
PhaseActionSubmittedEventType = Literal["phase-action-submitted"]
PhaseActionDispatchRejectedEventType = Literal["phase-action-dispatch-rejected"]
PhaseActionSatisfiedWithoutDispatchEventType = Literal["phase-action-satisfied-without-dispatch"]
PhaseActionSubmissionStatus = Literal["planned", "dispatching", "submitted", "satisfied", "rejected"]
PhaseSubmissionStatus = Literal["submitting", "submitted", "failed"]
PhaseMountSourceKind = Literal["file", "directory"]
PhaseMountOrigin = Literal[
    "authored",
    "cluster-extra",
    "runspec",
    "action-root",
    "source-bundle",
    "database-protected",
    "carry-record",
    "carry-workspace",
    "carry-source",
]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_ACTION_ID = re.compile(r"(?:preprocessing-chunk|msa-flatten|split|preprocess|fold|canonical-pair)-[0-9]{6}")
_SUBMISSION_ID = re.compile(r"phase-submission-[0-9a-f]{64}")
_CORRELATION_TOKEN = re.compile(r"bspp-phase-[0-9a-f]{64}")
_LEGACY_CORRELATION_TOKEN = re.compile(r"afcdb-phase-[0-9a-f]{64}")
_JOB_ID = re.compile(r"[0-9]+")
_SLURM_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_MAX_SLURM_JOB_NAME_LENGTH = 128
_MAX_SLURM_COMMENT_LENGTH = 256


@dataclass(frozen=True)
class PhaseContainerMountDescriptor:
    """One Phase-local container mount with explicit kind and access mode."""

    source: str
    target: str
    source_kind: PhaseMountSourceKind
    read_only: bool
    origin: PhaseMountOrigin
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, type(self).__name__)
        _absolute_posix_path(self.source, "mount source")
        _absolute_posix_path(self.target, "mount target")
        if self.source_kind not in {"file", "directory"}:
            raise ValueError("Phase mount source_kind must be file or directory")
        if not isinstance(self.read_only, bool):
            raise ValueError("Phase mount read_only must be boolean")
        if self.origin not in {
            "authored",
            "cluster-extra",
            "runspec",
            "action-root",
            "source-bundle",
            "database-protected",
            "carry-record",
            "carry-workspace",
            "carry-source",
        }:
            raise ValueError("unsupported Phase mount origin")
        if self.origin == "carry-source" and (self.source_kind != "file" or not self.read_only):
            raise ValueError("carry source mounts must be read-only files")
        if self.origin == "carry-workspace" and self.read_only:
            raise ValueError("carry workspace mounts must be writable")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "target": self.target,
            "source_kind": self.source_kind,
            "read_only": self.read_only,
            "origin": self.origin,
        }


def phase_action_submission_identity_mapping(
    *,
    action_id: str,
    runtime_action_digest: str,
    dependency_action_ids: Sequence[str],
    cluster_script_path: str,
    action_evidence_path: str,
    handoff_path: str,
    carry_forward_record_path: str | None = None,
    carry_forward_record_sha256: str | None = None,
    carry_forward_mounts: Sequence[PhaseContainerMountDescriptor] = (),
) -> dict[str, object]:
    """Build one validated, pre-token action identity for a submission ID."""
    identity: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "action_id": action_id,
        "runtime_action_digest": runtime_action_digest,
        "dependency_action_ids": list(dependency_action_ids),
        "cluster_script_path": cluster_script_path,
        "action_evidence_path": action_evidence_path,
        "handoff_path": handoff_path,
    }
    if carry_forward_record_path is not None:
        identity["carry_forward_record_path"] = carry_forward_record_path
        identity["carry_forward_record_sha256"] = carry_forward_record_sha256
        identity["carry_forward_mounts"] = [mount.to_mapping() for mount in carry_forward_mounts]
    return _normalize_action_submission_identity(identity)


def phase_submission_id(
    *,
    phase_run_id: str,
    attempt_id: str,
    phase_runspec_digest: str,
    phase_runspec_document_sha256: str,
    qualification_tuple_id: str,
    actions: Sequence[Mapping[str, object]],
) -> str:
    """Return the canonical ID for a submission before renderer-derived fields.

    The preimage binds the run, attempt, canonical RunSpec digest, exact stored
    RunSpec document hash, qualification tuple, and ordered action execution
    identities/target paths. It deliberately excludes ``job_name``,
    ``scheduler_correlation_token``, ``script_body``, and ``script_sha256``:
    those fields derive from this ID and including them would create a cyclic
    identity. The complete rendered plans are still retained in durable intent.
    """
    _validate_run_attempt(phase_run_id, attempt_id)
    _sha(phase_runspec_digest, "Phase Submission RunSpec digest")
    _sha(phase_runspec_document_sha256, "Phase Submission RunSpec document hash")
    _sha(qualification_tuple_id, "Phase Submission qualification tuple id")
    normalized_actions = tuple(_normalize_action_submission_identity(action) for action in actions)
    _validate_action_identities(normalized_actions)
    payload: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "phase_run_id": phase_run_id,
        "attempt_id": attempt_id,
        "phase_runspec_digest": phase_runspec_digest,
        "phase_runspec_document_sha256": phase_runspec_document_sha256,
        "qualification_tuple_id": qualification_tuple_id,
        "actions": list(normalized_actions),
    }
    return f"phase-submission-{canonical_mapping_digest(payload)}"


def phase_action_scheduler_correlation_token(submission_id: str, action_id: str, *, legacy: bool = False) -> str:
    """Return the exact Slurm-safe job name/comment for one frozen action."""
    _submission_id(submission_id)
    _action_id(action_id)
    digest = canonical_mapping_digest(
        {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "submission_id": submission_id,
            "action_id": action_id,
        }
    )
    prefix = "afcdb-phase-" if legacy else "bspp-phase-"
    return f"{prefix}{digest}"


@dataclass(frozen=True)
class PhaseActionSubmissionPlan:
    """One complete rendered Runtime Action retained before scheduler effects."""

    action_id: str
    runtime_action_digest: str
    dependency_action_ids: tuple[str, ...]
    cluster_script_path: str
    script_sha256: str
    script_body: str
    job_name: str
    scheduler_correlation_token: str
    action_evidence_path: str
    handoff_path: str
    carry_forward_record_path: str | None = None
    carry_forward_record_sha256: str | None = None
    carry_forward_mounts: tuple[PhaseContainerMountDescriptor, ...] = ()
    carry_forward_submission_id: str | None = None
    expected_task_indexes: tuple[int, ...] = ()
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PhaseActionSubmissionPlan")
        _action_id(self.action_id)
        _sha(self.runtime_action_digest, "Runtime Action digest")
        _dependency_action_ids(self.dependency_action_ids, action_id=self.action_id)
        _task_indexes(self.expected_task_indexes)
        _absolute_posix_path(self.cluster_script_path, "cluster script path", suffix=".sbatch")
        _absolute_posix_path(self.action_evidence_path, "action evidence path", suffix=".json")
        _absolute_posix_path(self.handoff_path, "handoff path")
        carry_present = self.carry_forward_record_path is not None
        if carry_present != (self.carry_forward_record_sha256 is not None):
            raise ValueError("carry record path/hash must be present together")
        if carry_present != bool(self.carry_forward_mounts) or carry_present != (
            self.carry_forward_submission_id is not None
        ):
            raise ValueError("carried submission fields must be present together")
        if carry_present:
            assert self.carry_forward_record_path is not None
            assert self.carry_forward_record_sha256 is not None
            assert self.carry_forward_submission_id is not None
            _absolute_posix_path(self.carry_forward_record_path, "carry record path", suffix=".json")
            _sha(self.carry_forward_record_sha256, "carry record document hash")
            _submission_id(self.carry_forward_submission_id)
            if any(not isinstance(item, PhaseContainerMountDescriptor) for item in self.carry_forward_mounts):
                raise ValueError("carry mounts must be strict PhaseContainerMountDescriptor records")
        _sha(self.script_sha256, "rendered script hash")
        if not self.script_body or "\x00" in self.script_body:
            raise ValueError("rendered script body must be non-empty UTF-8 text without NUL")
        actual_script_sha256 = hashlib.sha256(self.script_body.encode("utf-8")).hexdigest()
        if self.script_sha256 != actual_script_sha256:
            raise ValueError("rendered script hash does not match its exact UTF-8 body")
        _slurm_name(self.job_name, "Slurm job name", maximum=_MAX_SLURM_JOB_NAME_LENGTH)
        _correlation_token(self.scheduler_correlation_token)
        _slurm_name(
            self.scheduler_correlation_token,
            "Slurm correlation comment",
            maximum=_MAX_SLURM_COMMENT_LENGTH,
        )
        if self.job_name != self.scheduler_correlation_token:
            raise ValueError("Slurm job name must equal the exact scheduler correlation token")
        _require_exact_directive(self.script_body, "job-name", self.job_name)
        _require_exact_directive(self.script_body, "comment", self.scheduler_correlation_token)

    def submission_identity_mapping(self) -> dict[str, object]:
        """Return the non-cyclic action preimage used by ``phase_submission_id``."""
        return phase_action_submission_identity_mapping(
            action_id=self.action_id,
            runtime_action_digest=self.runtime_action_digest,
            dependency_action_ids=self.dependency_action_ids,
            cluster_script_path=self.cluster_script_path,
            action_evidence_path=self.action_evidence_path,
            handoff_path=self.handoff_path,
            carry_forward_record_path=self.carry_forward_record_path,
            carry_forward_record_sha256=self.carry_forward_record_sha256,
            carry_forward_mounts=self.carry_forward_mounts,
        )

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "runtime_action_digest": self.runtime_action_digest,
            "dependency_action_ids": list(self.dependency_action_ids),
            "cluster_script_path": self.cluster_script_path,
            "script_sha256": self.script_sha256,
            "script_body": self.script_body,
            "job_name": self.job_name,
            "scheduler_correlation_token": self.scheduler_correlation_token,
            "action_evidence_path": self.action_evidence_path,
            "handoff_path": self.handoff_path,
        }
        if self.carry_forward_record_path is not None:
            result.update(
                {
                    "carry_forward_record_path": self.carry_forward_record_path,
                    "carry_forward_record_sha256": self.carry_forward_record_sha256,
                    "carry_forward_mounts": [item.to_mapping() for item in self.carry_forward_mounts],
                    "carry_forward_submission_id": self.carry_forward_submission_id,
                }
            )
        if self.expected_task_indexes:
            result["expected_task_indexes"] = list(self.expected_task_indexes)
        return result


@dataclass(frozen=True)
class PhaseSubmissionIntendedPayload:
    """Complete immutable graph intent recorded before any remote effect."""

    submission_id: str
    phase_runspec_location: str
    phase_runspec_digest: str
    phase_runspec_document_sha256: str
    qualification_tuple_id: str
    actions: tuple[PhaseActionSubmissionPlan, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PhaseSubmissionIntendedPayload")
        _submission_id(self.submission_id)
        if not self.phase_runspec_location or PurePosixPath(self.phase_runspec_location).is_absolute():
            raise ValueError("Phase Submission RunSpec location must be authority-relative")
        if "\x00" in self.phase_runspec_location or ".." in PurePosixPath(self.phase_runspec_location).parts:
            raise ValueError("Phase Submission RunSpec location must stay within authority")
        _sha(self.phase_runspec_digest, "Phase Submission RunSpec digest")
        _sha(self.phase_runspec_document_sha256, "Phase Submission RunSpec document hash")
        _sha(self.qualification_tuple_id, "Phase Submission qualification tuple id")
        if not isinstance(self.actions, tuple) or not self.actions:
            raise ValueError("Phase Submission intent requires an immutable non-empty action tuple")
        if any(not isinstance(action, PhaseActionSubmissionPlan) for action in self.actions):
            raise ValueError("Phase Submission intent actions must be PhaseActionSubmissionPlan records")
        _validate_planned_action_graph(self.actions)
        for action in self.actions:
            expected_token = phase_action_scheduler_correlation_token(self.submission_id, action.action_id)
            legacy_expected_token = phase_action_scheduler_correlation_token(
                self.submission_id, action.action_id, legacy=True
            )
            if action.scheduler_correlation_token not in (
                expected_token,
                legacy_expected_token,
            ) or action.job_name not in (expected_token, legacy_expected_token):
                raise ValueError("planned action scheduler identity does not match its Phase Submission")
            if (
                action.carry_forward_submission_id is not None
                and action.carry_forward_submission_id != self.submission_id
            ):
                raise ValueError("carried action submission id must match its outer Phase Submission")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "submission_id": self.submission_id,
            "phase_runspec_location": self.phase_runspec_location,
            "phase_runspec_digest": self.phase_runspec_digest,
            "phase_runspec_document_sha256": self.phase_runspec_document_sha256,
            "qualification_tuple_id": self.qualification_tuple_id,
            "actions": [action.to_mapping() for action in self.actions],
        }


@dataclass(frozen=True)
class PhaseSubmissionIntendedEvent:
    """Append-only authority event freezing a complete submission graph."""

    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseSubmissionIntendedPayload
    event_type: PhaseSubmissionIntendedEventType = "phase-submission-intended"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _event_envelope(
            self.sequence,
            self.phase_run_id,
            self.attempt_id,
            self.occurred_at,
            actual_event_type=self.event_type,
            expected_event_type="phase-submission-intended",
            record_name="PhaseSubmissionIntendedEvent",
            schema_version=self.schema_version,
        )
        expected_location = f"attempts/{self.attempt_id}/phase-runspec.json"
        if self.payload.phase_runspec_location != expected_location:
            raise ValueError("Phase Submission intent RunSpec location must match its event Attempt")
        expected_submission_id = phase_submission_id(
            phase_run_id=self.phase_run_id,
            attempt_id=self.attempt_id,
            phase_runspec_digest=self.payload.phase_runspec_digest,
            phase_runspec_document_sha256=self.payload.phase_runspec_document_sha256,
            qualification_tuple_id=self.payload.qualification_tuple_id,
            actions=tuple(action.submission_identity_mapping() for action in self.payload.actions),
        )
        if self.payload.submission_id != expected_submission_id:
            raise ValueError("Phase Submission id does not match its canonical event-bound content")

    def to_mapping(self) -> dict[str, object]:
        return _event_mapping(self, self.payload.to_mapping())


@dataclass(frozen=True)
class PhaseActionDispatchIntendedPayload:
    """Durable last-safe point immediately before one ``sbatch`` call."""

    submission_id: str
    action_id: str
    script_sha256: str
    scheduler_correlation_token: str
    dependency_job_ids: tuple[str, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_dispatch_identity(
            self.schema_version,
            type(self).__name__,
            self.submission_id,
            self.action_id,
            self.script_sha256,
            self.scheduler_correlation_token,
            self.dependency_job_ids,
        )

    def to_mapping(self) -> dict[str, object]:
        return _dispatch_mapping(self)


@dataclass(frozen=True)
class PhaseActionDispatchIntendedEvent:
    """Append-only pre-scheduler boundary for one Runtime Action."""

    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseActionDispatchIntendedPayload
    event_type: PhaseActionDispatchIntendedEventType = "phase-action-dispatch-intended"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _event_envelope(
            self.sequence,
            self.phase_run_id,
            self.attempt_id,
            self.occurred_at,
            actual_event_type=self.event_type,
            expected_event_type="phase-action-dispatch-intended",
            record_name=type(self).__name__,
            schema_version=self.schema_version,
        )

    def to_mapping(self) -> dict[str, object]:
        return _event_mapping(self, self.payload.to_mapping())


@dataclass(frozen=True)
class PhaseActionSubmittedPayload:
    """One durable exact Slurm job assignment."""

    submission_id: str
    action_id: str
    script_sha256: str
    scheduler_correlation_token: str
    dependency_job_ids: tuple[str, ...]
    job_id: str
    sbatch_argv: tuple[str, ...]
    expected_task_indexes: tuple[int, ...] = ()
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_dispatch_identity(
            self.schema_version,
            type(self).__name__,
            self.submission_id,
            self.action_id,
            self.script_sha256,
            self.scheduler_correlation_token,
            self.dependency_job_ids,
        )
        _job_id(self.job_id, "submitted Slurm job id")
        _argv(self.sbatch_argv)
        _task_indexes(self.expected_task_indexes)

    def to_mapping(self) -> dict[str, object]:
        result = {**_dispatch_mapping(self), "job_id": self.job_id, "sbatch_argv": list(self.sbatch_argv)}
        if self.expected_task_indexes:
            result["expected_task_indexes"] = list(self.expected_task_indexes)
        return result


@dataclass(frozen=True)
class PhaseActionSubmittedEvent:
    """Append-only assignment event for one accepted Slurm job."""

    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseActionSubmittedPayload
    event_type: PhaseActionSubmittedEventType = "phase-action-submitted"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _event_envelope(
            self.sequence,
            self.phase_run_id,
            self.attempt_id,
            self.occurred_at,
            actual_event_type=self.event_type,
            expected_event_type="phase-action-submitted",
            record_name=type(self).__name__,
            schema_version=self.schema_version,
        )

    def to_mapping(self) -> dict[str, object]:
        return _event_mapping(self, self.payload.to_mapping())


@dataclass(frozen=True)
class PhaseActionDispatchRejectedPayload:
    """One proven local ``sbatch`` rejection with its exact command result."""

    submission_id: str
    action_id: str
    script_sha256: str
    scheduler_correlation_token: str
    dependency_job_ids: tuple[str, ...]
    sbatch_argv: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_dispatch_identity(
            self.schema_version,
            type(self).__name__,
            self.submission_id,
            self.action_id,
            self.script_sha256,
            self.scheduler_correlation_token,
            self.dependency_job_ids,
        )
        _argv(self.sbatch_argv)
        if not isinstance(self.return_code, int) or isinstance(self.return_code, bool) or self.return_code == 0:
            raise ValueError("rejected sbatch return_code must be a nonzero integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise ValueError("rejected sbatch stdout and stderr must be strings")

    def to_mapping(self) -> dict[str, object]:
        return {
            **_dispatch_mapping(self),
            "sbatch_argv": list(self.sbatch_argv),
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class PhaseActionDispatchRejectedEvent:
    """Append-only definitive rejection event for one Runtime Action."""

    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseActionDispatchRejectedPayload
    event_type: PhaseActionDispatchRejectedEventType = "phase-action-dispatch-rejected"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _event_envelope(
            self.sequence,
            self.phase_run_id,
            self.attempt_id,
            self.occurred_at,
            actual_event_type=self.event_type,
            expected_event_type="phase-action-dispatch-rejected",
            record_name=type(self).__name__,
            schema_version=self.schema_version,
        )

    def to_mapping(self) -> dict[str, object]:
        return _event_mapping(self, self.payload.to_mapping())


@dataclass(frozen=True)
class PhaseActionSatisfiedWithoutDispatchPayload:
    """One proven no-dispatch satisfaction for a fully carried fold action.

    Binds the exact submission, fold action, RunSpec, sealed carry record, and
    canonical shard projection digests plus the ordered, duplicate-free carried
    target closure. There is no ``sbatch`` effect: the action is satisfied
    without a Slurm job and the dependent canonical-pair action becomes
    directly dispatchable.
    """

    submission_id: str
    action_id: str
    runtime_action_digest: str
    scheduler_correlation_token: str
    phase_runspec_digest: str
    carry_record_digest: str
    shard_manifest_digest: str
    carried_target_ids: tuple[str, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, type(self).__name__)
        _submission_id(self.submission_id)
        _action_id(self.action_id)
        _sha(self.runtime_action_digest, "satisfied action runtime digest")
        _correlation_token(self.scheduler_correlation_token)
        expected_token = phase_action_scheduler_correlation_token(self.submission_id, self.action_id)
        if self.scheduler_correlation_token != expected_token:
            raise ValueError("satisfied action correlation token does not match its submission and action")
        _sha(self.phase_runspec_digest, "satisfied action RunSpec digest")
        _sha(self.carry_record_digest, "satisfied action carry record digest")
        _sha(self.shard_manifest_digest, "satisfied action shard manifest digest")
        if not isinstance(self.carried_target_ids, tuple) or not self.carried_target_ids:
            raise ValueError("satisfied action requires a non-empty carried target closure")
        if any(not isinstance(item, str) or not item for item in self.carried_target_ids):
            raise ValueError("satisfied action carried target ids must be non-empty strings")
        if len(set(self.carried_target_ids)) != len(self.carried_target_ids):
            raise ValueError("satisfied action carried target ids must be duplicate-free")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "submission_id": self.submission_id,
            "action_id": self.action_id,
            "runtime_action_digest": self.runtime_action_digest,
            "scheduler_correlation_token": self.scheduler_correlation_token,
            "phase_runspec_digest": self.phase_runspec_digest,
            "carry_record_digest": self.carry_record_digest,
            "shard_manifest_digest": self.shard_manifest_digest,
            "carried_target_ids": list(self.carried_target_ids),
        }


@dataclass(frozen=True)
class PhaseActionSatisfiedWithoutDispatchEvent:
    """Append-only no-dispatch satisfaction event for one carried fold action."""

    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseActionSatisfiedWithoutDispatchPayload
    event_type: PhaseActionSatisfiedWithoutDispatchEventType = "phase-action-satisfied-without-dispatch"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _event_envelope(
            self.sequence,
            self.phase_run_id,
            self.attempt_id,
            self.occurred_at,
            actual_event_type=self.event_type,
            expected_event_type="phase-action-satisfied-without-dispatch",
            record_name=type(self).__name__,
            schema_version=self.schema_version,
        )

    def to_mapping(self) -> dict[str, object]:
        return _event_mapping(self, self.payload.to_mapping())


@dataclass(frozen=True)
class PhaseActionSubmissionView:
    """Replay-only current state for one frozen action plan."""

    plan: PhaseActionSubmissionPlan
    status: PhaseActionSubmissionStatus
    dependency_job_ids: tuple[str, ...] = ()
    job_id: str | None = None
    sbatch_argv: tuple[str, ...] = ()
    return_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PhaseActionSubmissionView")
        if not isinstance(self.plan, PhaseActionSubmissionPlan):
            raise ValueError("action submission view requires its frozen action plan")
        if self.status not in {"planned", "dispatching", "submitted", "satisfied", "rejected"}:
            raise ValueError(f"unsupported Phase Action submission status: {self.status!r}")
        _dependency_job_ids(self.dependency_job_ids)
        if self.status == "planned":
            if (
                self.dependency_job_ids
                or self.job_id is not None
                or self.sbatch_argv
                or self.return_code is not None
                or self.stdout is not None
                or self.stderr is not None
            ):
                raise ValueError("planned action view cannot contain dispatch or outcome data")
        elif self.status == "dispatching":
            if any(value is not None for value in (self.job_id, self.return_code, self.stdout, self.stderr)):
                raise ValueError("dispatching action view cannot contain an outcome")
            if self.sbatch_argv:
                raise ValueError("dispatching action view cannot contain sbatch result argv")
        elif self.status == "submitted":
            if self.job_id is None:
                raise ValueError("submitted action view requires a Slurm job id")
            _job_id(self.job_id, "submitted action view job id")
            _argv(self.sbatch_argv)
            if any(value is not None for value in (self.return_code, self.stdout, self.stderr)):
                raise ValueError("submitted action view cannot contain rejection data")
        elif self.status == "satisfied":
            if self.dependency_job_ids or self.job_id is not None or self.sbatch_argv:
                raise ValueError("satisfied action view cannot contain dispatch or job identity")
            if any(value is not None for value in (self.return_code, self.stdout, self.stderr)):
                raise ValueError("satisfied action view cannot contain rejection data")
        else:
            if self.job_id is not None:
                raise ValueError("rejected action view cannot contain a Slurm job id")
            _argv(self.sbatch_argv)
            if (
                self.return_code is None
                or not isinstance(self.return_code, int)
                or isinstance(self.return_code, bool)
                or self.return_code == 0
                or self.stdout is None
                or self.stderr is None
            ):
                raise ValueError("rejected action view requires a complete nonzero sbatch result")

    @property
    def action_id(self) -> str:
        return self.plan.action_id

    @property
    def scheduler_correlation_token(self) -> str:
        return self.plan.scheduler_correlation_token

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan": self.plan.to_mapping(),
            "status": self.status,
            "dependency_job_ids": list(self.dependency_job_ids),
            "job_id": self.job_id,
            "sbatch_argv": list(self.sbatch_argv),
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class PhaseSubmissionLifecycleView:
    """Replay-only current state for one exact Phase Submission intent."""

    phase_run_id: str
    attempt_id: str
    submission_id: str
    phase_runspec_location: str
    phase_runspec_digest: str
    phase_runspec_document_sha256: str
    qualification_tuple_id: str
    actions: tuple[PhaseActionSubmissionView, ...]
    status: PhaseSubmissionStatus
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PhaseSubmissionLifecycleView")
        _validate_run_attempt(self.phase_run_id, self.attempt_id)
        _submission_id(self.submission_id)
        if self.phase_runspec_location != f"attempts/{self.attempt_id}/phase-runspec.json":
            raise ValueError("submission lifecycle RunSpec location must be attempt-bound")
        _sha(self.phase_runspec_digest, "submission lifecycle RunSpec digest")
        _sha(self.phase_runspec_document_sha256, "submission lifecycle RunSpec document hash")
        _sha(self.qualification_tuple_id, "submission lifecycle qualification tuple id")
        if not isinstance(self.actions, tuple) or not self.actions:
            raise ValueError("submission lifecycle requires an immutable non-empty action tuple")
        if any(not isinstance(action, PhaseActionSubmissionView) for action in self.actions):
            raise ValueError("submission lifecycle actions must be PhaseActionSubmissionView records")
        action_ids = tuple(action.action_id for action in self.actions)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("submission lifecycle action ids must be unique")
        for action in self.actions:
            expected_token = phase_action_scheduler_correlation_token(self.submission_id, action.action_id)
            legacy_expected_token = phase_action_scheduler_correlation_token(
                self.submission_id, action.action_id, legacy=True
            )
            if action.plan.scheduler_correlation_token not in (
                expected_token,
                legacy_expected_token,
            ) or action.plan.job_name not in (expected_token, legacy_expected_token):
                raise ValueError("submission lifecycle action scheduler identity does not match its submission")
        expected_submission_id = phase_submission_id(
            phase_run_id=self.phase_run_id,
            attempt_id=self.attempt_id,
            phase_runspec_digest=self.phase_runspec_digest,
            phase_runspec_document_sha256=self.phase_runspec_document_sha256,
            qualification_tuple_id=self.qualification_tuple_id,
            actions=tuple(action.plan.submission_identity_mapping() for action in self.actions),
        )
        if self.submission_id != expected_submission_id:
            raise ValueError("submission lifecycle id does not match its exact action identities")
        expected_status: PhaseSubmissionStatus
        if any(action.status == "rejected" for action in self.actions):
            expected_status = "failed"
        elif all(action.status in {"submitted", "satisfied"} for action in self.actions):
            expected_status = "submitted"
        else:
            expected_status = "submitting"
        if self.status != expected_status:
            raise ValueError("submission lifecycle status does not match its action states")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "submission_id": self.submission_id,
            "phase_runspec_location": self.phase_runspec_location,
            "phase_runspec_digest": self.phase_runspec_digest,
            "phase_runspec_document_sha256": self.phase_runspec_document_sha256,
            "qualification_tuple_id": self.qualification_tuple_id,
            "actions": [action.to_mapping() for action in self.actions],
            "status": self.status,
        }


def phase_container_mount_descriptor_from_mapping(payload: Mapping[str, object]) -> PhaseContainerMountDescriptor:
    _strict(
        payload,
        {"schema_version", "source", "target", "source_kind", "read_only", "origin"},
        "PhaseContainerMountDescriptor",
    )
    source_kind = _str(payload, "source_kind")
    origin = _str(payload, "origin")
    read_only = payload.get("read_only")
    if not isinstance(read_only, bool):
        raise ValueError("read_only must be boolean")
    return PhaseContainerMountDescriptor(
        schema_version=_schema(payload, "PhaseContainerMountDescriptor"),
        source=_str(payload, "source"),
        target=_str(payload, "target"),
        source_kind=cast("PhaseMountSourceKind", source_kind),
        read_only=read_only,
        origin=cast("PhaseMountOrigin", origin),
    )


def phase_action_submission_plan_from_mapping(payload: Mapping[str, object]) -> PhaseActionSubmissionPlan:
    _strict(
        payload,
        {
            "schema_version",
            "action_id",
            "runtime_action_digest",
            "dependency_action_ids",
            "cluster_script_path",
            "script_sha256",
            "script_body",
            "job_name",
            "scheduler_correlation_token",
            "action_evidence_path",
            "handoff_path",
            "carry_forward_record_path",
            "carry_forward_record_sha256",
            "carry_forward_mounts",
            "carry_forward_submission_id",
            "expected_task_indexes",
        },
        "PhaseActionSubmissionPlan",
    )
    carry_mounts = payload.get("carry_forward_mounts")
    if "carry_forward_mounts" in payload and not isinstance(carry_mounts, list):
        raise ValueError("carry_forward_mounts must be a list when present")
    return PhaseActionSubmissionPlan(
        schema_version=_schema(payload, "PhaseActionSubmissionPlan"),
        action_id=_str(payload, "action_id"),
        runtime_action_digest=_str(payload, "runtime_action_digest"),
        dependency_action_ids=_str_sequence(payload, "dependency_action_ids"),
        cluster_script_path=_str(payload, "cluster_script_path"),
        script_sha256=_str(payload, "script_sha256"),
        script_body=_str(payload, "script_body"),
        job_name=_str(payload, "job_name"),
        scheduler_correlation_token=_str(payload, "scheduler_correlation_token"),
        action_evidence_path=_str(payload, "action_evidence_path"),
        handoff_path=_str(payload, "handoff_path"),
        carry_forward_record_path=_optional_str(payload, "carry_forward_record_path"),
        carry_forward_record_sha256=_optional_str(payload, "carry_forward_record_sha256"),
        carry_forward_mounts=tuple(
            phase_container_mount_descriptor_from_mapping(item)
            for item in cast("list[Mapping[str, object]]", carry_mounts or [])
        ),
        carry_forward_submission_id=_optional_str(payload, "carry_forward_submission_id"),
        expected_task_indexes=_optional_integers(payload, "expected_task_indexes"),
    )


def phase_submission_intended_payload_from_mapping(
    payload: Mapping[str, object],
) -> PhaseSubmissionIntendedPayload:
    _strict(
        payload,
        {
            "schema_version",
            "submission_id",
            "phase_runspec_location",
            "phase_runspec_digest",
            "phase_runspec_document_sha256",
            "qualification_tuple_id",
            "actions",
        },
        "PhaseSubmissionIntendedPayload",
    )
    return PhaseSubmissionIntendedPayload(
        schema_version=_schema(payload, "PhaseSubmissionIntendedPayload"),
        submission_id=_str(payload, "submission_id"),
        phase_runspec_location=_str(payload, "phase_runspec_location"),
        phase_runspec_digest=_str(payload, "phase_runspec_digest"),
        phase_runspec_document_sha256=_str(payload, "phase_runspec_document_sha256"),
        qualification_tuple_id=_str(payload, "qualification_tuple_id"),
        actions=tuple(
            phase_action_submission_plan_from_mapping(item) for item in _mapping_sequence(payload, "actions")
        ),
    )


def phase_submission_intended_event_from_mapping(payload: Mapping[str, object]) -> PhaseSubmissionIntendedEvent:
    event_type = _event_type(payload, "PhaseSubmissionIntendedEvent", "phase-submission-intended")
    return PhaseSubmissionIntendedEvent(
        schema_version=_schema(payload, "PhaseSubmissionIntendedEvent"),
        sequence=_int(payload, "sequence"),
        event_type=cast("PhaseSubmissionIntendedEventType", event_type),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        occurred_at=_str(payload, "occurred_at"),
        payload=phase_submission_intended_payload_from_mapping(_mapping(payload, "payload")),
    )


def phase_action_dispatch_intended_payload_from_mapping(
    payload: Mapping[str, object],
) -> PhaseActionDispatchIntendedPayload:
    _strict(payload, _DISPATCH_FIELDS, "PhaseActionDispatchIntendedPayload")
    return PhaseActionDispatchIntendedPayload(
        schema_version=_schema(payload, "PhaseActionDispatchIntendedPayload"),
        submission_id=_str(payload, "submission_id"),
        action_id=_str(payload, "action_id"),
        script_sha256=_str(payload, "script_sha256"),
        scheduler_correlation_token=_str(payload, "scheduler_correlation_token"),
        dependency_job_ids=_str_sequence(payload, "dependency_job_ids"),
    )


def phase_action_dispatch_intended_event_from_mapping(
    payload: Mapping[str, object],
) -> PhaseActionDispatchIntendedEvent:
    event_type = _event_type(payload, "PhaseActionDispatchIntendedEvent", "phase-action-dispatch-intended")
    return PhaseActionDispatchIntendedEvent(
        schema_version=_schema(payload, "PhaseActionDispatchIntendedEvent"),
        sequence=_int(payload, "sequence"),
        event_type=cast("PhaseActionDispatchIntendedEventType", event_type),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        occurred_at=_str(payload, "occurred_at"),
        payload=phase_action_dispatch_intended_payload_from_mapping(_mapping(payload, "payload")),
    )


def phase_action_submitted_payload_from_mapping(payload: Mapping[str, object]) -> PhaseActionSubmittedPayload:
    _strict(
        payload,
        {*_DISPATCH_FIELDS, "job_id", "sbatch_argv", "expected_task_indexes"},
        "PhaseActionSubmittedPayload",
    )
    return PhaseActionSubmittedPayload(
        schema_version=_schema(payload, "PhaseActionSubmittedPayload"),
        submission_id=_str(payload, "submission_id"),
        action_id=_str(payload, "action_id"),
        script_sha256=_str(payload, "script_sha256"),
        scheduler_correlation_token=_str(payload, "scheduler_correlation_token"),
        dependency_job_ids=_str_sequence(payload, "dependency_job_ids"),
        job_id=_str(payload, "job_id"),
        sbatch_argv=_str_sequence(payload, "sbatch_argv"),
        expected_task_indexes=_optional_integers(payload, "expected_task_indexes"),
    )


def phase_action_submitted_event_from_mapping(payload: Mapping[str, object]) -> PhaseActionSubmittedEvent:
    event_type = _event_type(payload, "PhaseActionSubmittedEvent", "phase-action-submitted")
    return PhaseActionSubmittedEvent(
        schema_version=_schema(payload, "PhaseActionSubmittedEvent"),
        sequence=_int(payload, "sequence"),
        event_type=cast("PhaseActionSubmittedEventType", event_type),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        occurred_at=_str(payload, "occurred_at"),
        payload=phase_action_submitted_payload_from_mapping(_mapping(payload, "payload")),
    )


def phase_action_dispatch_rejected_payload_from_mapping(
    payload: Mapping[str, object],
) -> PhaseActionDispatchRejectedPayload:
    _strict(
        payload,
        {*_DISPATCH_FIELDS, "sbatch_argv", "return_code", "stdout", "stderr"},
        "PhaseActionDispatchRejectedPayload",
    )
    return PhaseActionDispatchRejectedPayload(
        schema_version=_schema(payload, "PhaseActionDispatchRejectedPayload"),
        submission_id=_str(payload, "submission_id"),
        action_id=_str(payload, "action_id"),
        script_sha256=_str(payload, "script_sha256"),
        scheduler_correlation_token=_str(payload, "scheduler_correlation_token"),
        dependency_job_ids=_str_sequence(payload, "dependency_job_ids"),
        sbatch_argv=_str_sequence(payload, "sbatch_argv"),
        return_code=_int(payload, "return_code"),
        stdout=_present_str(payload, "stdout"),
        stderr=_present_str(payload, "stderr"),
    )


def phase_action_dispatch_rejected_event_from_mapping(
    payload: Mapping[str, object],
) -> PhaseActionDispatchRejectedEvent:
    event_type = _event_type(payload, "PhaseActionDispatchRejectedEvent", "phase-action-dispatch-rejected")
    return PhaseActionDispatchRejectedEvent(
        schema_version=_schema(payload, "PhaseActionDispatchRejectedEvent"),
        sequence=_int(payload, "sequence"),
        event_type=cast("PhaseActionDispatchRejectedEventType", event_type),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        occurred_at=_str(payload, "occurred_at"),
        payload=phase_action_dispatch_rejected_payload_from_mapping(_mapping(payload, "payload")),
    )


def phase_action_satisfied_without_dispatch_payload_from_mapping(
    payload: Mapping[str, object],
) -> PhaseActionSatisfiedWithoutDispatchPayload:
    _strict(
        payload,
        {
            "schema_version",
            "submission_id",
            "action_id",
            "runtime_action_digest",
            "scheduler_correlation_token",
            "phase_runspec_digest",
            "carry_record_digest",
            "shard_manifest_digest",
            "carried_target_ids",
        },
        "PhaseActionSatisfiedWithoutDispatchPayload",
    )
    return PhaseActionSatisfiedWithoutDispatchPayload(
        schema_version=_schema(payload, "PhaseActionSatisfiedWithoutDispatchPayload"),
        submission_id=_str(payload, "submission_id"),
        action_id=_str(payload, "action_id"),
        runtime_action_digest=_str(payload, "runtime_action_digest"),
        scheduler_correlation_token=_str(payload, "scheduler_correlation_token"),
        phase_runspec_digest=_str(payload, "phase_runspec_digest"),
        carry_record_digest=_str(payload, "carry_record_digest"),
        shard_manifest_digest=_str(payload, "shard_manifest_digest"),
        carried_target_ids=_str_sequence(payload, "carried_target_ids"),
    )


def phase_action_satisfied_without_dispatch_event_from_mapping(
    payload: Mapping[str, object],
) -> PhaseActionSatisfiedWithoutDispatchEvent:
    event_type = _event_type(
        payload,
        "PhaseActionSatisfiedWithoutDispatchEvent",
        "phase-action-satisfied-without-dispatch",
    )
    return PhaseActionSatisfiedWithoutDispatchEvent(
        schema_version=_schema(payload, "PhaseActionSatisfiedWithoutDispatchEvent"),
        sequence=_int(payload, "sequence"),
        event_type=cast("PhaseActionSatisfiedWithoutDispatchEventType", event_type),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        occurred_at=_str(payload, "occurred_at"),
        payload=phase_action_satisfied_without_dispatch_payload_from_mapping(_mapping(payload, "payload")),
    )


def phase_action_submission_view_from_mapping(payload: Mapping[str, object]) -> PhaseActionSubmissionView:
    _strict(
        payload,
        {
            "schema_version",
            "plan",
            "status",
            "dependency_job_ids",
            "job_id",
            "sbatch_argv",
            "return_code",
            "stdout",
            "stderr",
        },
        "PhaseActionSubmissionView",
    )
    status = _str(payload, "status")
    return PhaseActionSubmissionView(
        schema_version=_schema(payload, "PhaseActionSubmissionView"),
        plan=phase_action_submission_plan_from_mapping(_mapping(payload, "plan")),
        status=cast("PhaseActionSubmissionStatus", status),
        dependency_job_ids=_str_sequence(payload, "dependency_job_ids"),
        job_id=_optional_str(payload, "job_id"),
        sbatch_argv=_str_sequence(payload, "sbatch_argv"),
        return_code=_optional_int(payload, "return_code"),
        stdout=_optional_str(payload, "stdout", allow_empty=True),
        stderr=_optional_str(payload, "stderr", allow_empty=True),
    )


def phase_submission_lifecycle_view_from_mapping(payload: Mapping[str, object]) -> PhaseSubmissionLifecycleView:
    _strict(
        payload,
        {
            "schema_version",
            "phase_run_id",
            "attempt_id",
            "submission_id",
            "phase_runspec_location",
            "phase_runspec_digest",
            "phase_runspec_document_sha256",
            "qualification_tuple_id",
            "actions",
            "status",
        },
        "PhaseSubmissionLifecycleView",
    )
    status = _str(payload, "status")
    return PhaseSubmissionLifecycleView(
        schema_version=_schema(payload, "PhaseSubmissionLifecycleView"),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        submission_id=_str(payload, "submission_id"),
        phase_runspec_location=_str(payload, "phase_runspec_location"),
        phase_runspec_digest=_str(payload, "phase_runspec_digest"),
        phase_runspec_document_sha256=_str(payload, "phase_runspec_document_sha256"),
        qualification_tuple_id=_str(payload, "qualification_tuple_id"),
        actions=tuple(
            phase_action_submission_view_from_mapping(item) for item in _mapping_sequence(payload, "actions")
        ),
        status=cast("PhaseSubmissionStatus", status),
    )


_EVENT_FIELDS = {
    "schema_version",
    "sequence",
    "event_type",
    "phase_run_id",
    "attempt_id",
    "occurred_at",
    "payload",
}
_DISPATCH_FIELDS = {
    "schema_version",
    "submission_id",
    "action_id",
    "script_sha256",
    "scheduler_correlation_token",
    "dependency_job_ids",
}


def _event_type(payload: Mapping[str, object], name: str, expected: str) -> str:
    _strict(payload, _EVENT_FIELDS, name)
    event_type = _str(payload, "event_type")
    if event_type != expected:
        raise ValueError(f"unsupported {name} event type: {event_type!r}")
    return event_type


def _normalize_action_submission_identity(payload: Mapping[str, object]) -> dict[str, object]:
    name = "Phase Action submission identity"
    _strict(
        payload,
        {
            "schema_version",
            "action_id",
            "runtime_action_digest",
            "dependency_action_ids",
            "cluster_script_path",
            "action_evidence_path",
            "handoff_path",
            "carry_forward_record_path",
            "carry_forward_record_sha256",
            "carry_forward_mounts",
        },
        name,
    )
    _schema(payload, name)
    action_id = _str(payload, "action_id")
    _action_id(action_id)
    runtime_action_digest = _str(payload, "runtime_action_digest")
    _sha(runtime_action_digest, "Runtime Action digest")
    dependency_action_ids = _str_sequence(payload, "dependency_action_ids")
    _dependency_action_ids(dependency_action_ids, action_id=action_id)
    cluster_script_path = _str(payload, "cluster_script_path")
    action_evidence_path = _str(payload, "action_evidence_path")
    handoff_path = _str(payload, "handoff_path")
    _absolute_posix_path(cluster_script_path, "cluster script path", suffix=".sbatch")
    _absolute_posix_path(action_evidence_path, "action evidence path", suffix=".json")
    _absolute_posix_path(handoff_path, "handoff path")
    result: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "action_id": action_id,
        "runtime_action_digest": runtime_action_digest,
        "dependency_action_ids": list(dependency_action_ids),
        "cluster_script_path": cluster_script_path,
        "action_evidence_path": action_evidence_path,
        "handoff_path": handoff_path,
    }
    carry_path = payload.get("carry_forward_record_path")
    carry_sha = payload.get("carry_forward_record_sha256")
    carry_mounts = payload.get("carry_forward_mounts")
    present = carry_path is not None or carry_sha is not None or carry_mounts is not None
    if present:
        if not isinstance(carry_path, str) or not carry_path:
            raise ValueError("carry_forward_record_path must be a non-empty string")
        if not isinstance(carry_sha, str):
            raise ValueError("carry_forward_record_sha256 must be a string")
        if not isinstance(carry_mounts, list) or not carry_mounts:
            raise ValueError("carry_forward_mounts must be a non-empty list")
        mounts = tuple(phase_container_mount_descriptor_from_mapping(item) for item in carry_mounts)
        _absolute_posix_path(carry_path, "carry record path", suffix=".json")
        _sha(carry_sha, "carry record document hash")
        result.update(
            {
                "carry_forward_record_path": carry_path,
                "carry_forward_record_sha256": carry_sha,
                "carry_forward_mounts": [item.to_mapping() for item in mounts],
            }
        )
    return result


def _validate_action_identities(actions: tuple[Mapping[str, object], ...]) -> None:
    if not actions:
        raise ValueError("Phase Submission identity requires at least one action")
    action_ids = tuple(cast("str", action["action_id"]) for action in actions)
    if len(set(action_ids)) != len(action_ids):
        raise ValueError("Phase Submission action ids must be unique")
    known = set(action_ids)
    dependencies = {
        cast("str", action["action_id"]): tuple(cast("list[str]", action["dependency_action_ids"]))
        for action in actions
    }
    if any(dependency not in known for values in dependencies.values() for dependency in values):
        raise ValueError("Phase Submission dependencies must reference declared actions")
    _reject_dependency_cycles(dependencies)


def _validate_planned_action_graph(actions: tuple[PhaseActionSubmissionPlan, ...]) -> None:
    _validate_action_identities(tuple(action.submission_identity_mapping() for action in actions))


def _reject_dependency_cycles(dependencies: Mapping[str, tuple[str, ...]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(action_id: str) -> None:
        if action_id in visiting:
            raise ValueError("Phase Submission action dependencies must be acyclic")
        if action_id in visited:
            return
        visiting.add(action_id)
        for dependency in dependencies[action_id]:
            visit(dependency)
        visiting.remove(action_id)
        visited.add(action_id)

    for action_id in dependencies:
        visit(action_id)


def _validate_dispatch_identity(
    schema_version: int,
    name: str,
    submission_id: str,
    action_id: str,
    script_sha256: str,
    scheduler_correlation_token: str,
    dependency_job_ids: tuple[str, ...],
) -> None:
    _validate_schema(schema_version, name)
    _submission_id(submission_id)
    _action_id(action_id)
    _sha(script_sha256, "dispatch script hash")
    _correlation_token(scheduler_correlation_token)
    expected_token = phase_action_scheduler_correlation_token(submission_id, action_id)
    legacy_expected_token = phase_action_scheduler_correlation_token(submission_id, action_id, legacy=True)
    if scheduler_correlation_token not in (expected_token, legacy_expected_token):
        raise ValueError("dispatch scheduler correlation token does not match its submission and action")
    _dependency_job_ids(dependency_job_ids)


class _DispatchRecord(Protocol):
    @property
    def schema_version(self) -> int: ...

    @property
    def submission_id(self) -> str: ...

    @property
    def action_id(self) -> str: ...

    @property
    def script_sha256(self) -> str: ...

    @property
    def scheduler_correlation_token(self) -> str: ...

    @property
    def dependency_job_ids(self) -> tuple[str, ...]: ...


class _EventRecord(Protocol):
    @property
    def schema_version(self) -> int: ...

    @property
    def sequence(self) -> int: ...

    @property
    def event_type(self) -> str: ...

    @property
    def phase_run_id(self) -> str: ...

    @property
    def attempt_id(self) -> str: ...

    @property
    def occurred_at(self) -> str: ...


def _dispatch_mapping(payload: _DispatchRecord) -> dict[str, object]:
    return {
        "schema_version": payload.schema_version,
        "submission_id": payload.submission_id,
        "action_id": payload.action_id,
        "script_sha256": payload.script_sha256,
        "scheduler_correlation_token": payload.scheduler_correlation_token,
        "dependency_job_ids": list(payload.dependency_job_ids),
    }


def _event_mapping(event: _EventRecord, payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": event.schema_version,
        "sequence": event.sequence,
        "event_type": event.event_type,
        "phase_run_id": event.phase_run_id,
        "attempt_id": event.attempt_id,
        "occurred_at": event.occurred_at,
        "payload": dict(payload),
    }


def _event_envelope(
    sequence: int,
    phase_run_id: str,
    attempt_id: str,
    occurred_at: str,
    *,
    actual_event_type: str,
    expected_event_type: str,
    record_name: str,
    schema_version: int,
) -> None:
    _validate_schema(schema_version, record_name)
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 1:
        raise ValueError(f"{record_name} sequence must be greater than one")
    if actual_event_type != expected_event_type:
        raise ValueError(f"{record_name} must be named {expected_event_type!r}")
    _validate_run_attempt(phase_run_id, attempt_id)
    _timestamp(occurred_at, f"{record_name} occurred_at")


def _strict(payload: Mapping[str, object], allowed: set[str], name: str) -> None:
    if "schema_version" not in payload:
        raise ValueError(f"missing explicit schema_version at {name}")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")


def _schema(payload: Mapping[str, object], name: str) -> int:
    value = payload.get("schema_version")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} schema_version must be declared explicitly")
    return validate_schema_version(value, record_name=name)


def _validate_schema(value: int, name: str) -> None:
    if validate_schema_version(value, record_name=name) != value:
        raise ValueError(f"{name} schema_version must be explicit")


def _str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _present_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_str(payload: Mapping[str, object], key: str, *, allow_empty: bool = False) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ValueError(f"{key} must be null or a{' possibly empty' if allow_empty else ' non-empty'} string")
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


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _mapping_sequence(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        raise ValueError(f"{key} must be a list")
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{key}[{index}] must be a mapping")
        result.append(item)
    return tuple(result)


def _str_sequence(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        raise ValueError(f"{key} must be a list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{key}[{index}] must be a non-empty string")
        result.append(item)
    return tuple(result)


def _validate_run_attempt(phase_run_id: str, attempt_id: str) -> None:
    if _PHASE_RUN_ID.fullmatch(phase_run_id) is None or _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise ValueError("submission record has invalid Phase Run or Attempt identity")


def _action_id(value: str) -> None:
    if _ACTION_ID.fullmatch(value) is None:
        raise ValueError("submission record has invalid Runtime Action identity")


def _submission_id(value: str) -> None:
    if _SUBMISSION_ID.fullmatch(value) is None:
        raise ValueError("submission_id must be a canonical phase-submission id")


def _correlation_token(value: str) -> None:
    if _CORRELATION_TOKEN.fullmatch(value) is None and _LEGACY_CORRELATION_TOKEN.fullmatch(value) is None:
        raise ValueError("scheduler correlation token must be a canonical bspp-phase token")


def _sha(value: str, name: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _dependency_action_ids(values: tuple[str, ...], *, action_id: str | None = None) -> None:
    if not isinstance(values, tuple):
        raise ValueError("dependency action ids must be an immutable tuple")
    for value in values:
        _action_id(value)
    if len(set(values)) != len(values):
        raise ValueError("dependency action ids must be unique")
    if action_id is not None and action_id in values:
        raise ValueError("an action cannot depend on itself")


def _dependency_job_ids(values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple):
        raise ValueError("dependency job ids must be an immutable tuple")
    for value in values:
        _job_id(value, "dependency job id")
    if len(set(values)) != len(values):
        raise ValueError("dependency job ids must be unique")


def _job_id(value: str, name: str) -> None:
    if _JOB_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be numeric")


def _argv(value: tuple[str, ...]) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError("sbatch_argv must be an immutable non-empty tuple")
    if any(not isinstance(part, str) or not part or "\x00" in part for part in value):
        raise ValueError("sbatch_argv entries must be non-empty strings without NUL")


def _task_indexes(values: tuple[int, ...]) -> None:
    if not isinstance(values, tuple):
        raise ValueError("expected task indexes must be an immutable tuple")
    if tuple(sorted(set(values))) != values or any(item < 0 for item in values):
        raise ValueError("expected task indexes must be sorted, unique, and non-negative")


def _optional_integers(payload: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{key} must be a list of integers")
    return tuple(value)


def _absolute_posix_path(value: str, name: str, *, suffix: str | None = None) -> None:
    path = PurePosixPath(value)
    if not value or "\x00" in value or "\n" in value or not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a safe absolute POSIX path")
    if suffix is not None and not value.endswith(suffix):
        raise ValueError(f"{name} must end with {suffix}")


def _slurm_name(value: str, name: str, *, maximum: int) -> None:
    if len(value) > maximum or _SLURM_SAFE_NAME.fullmatch(value) is None:
        raise ValueError(f"{name} must be at most {maximum} Slurm-safe ASCII characters")


def _require_exact_directive(script_body: str, directive: str, value: str) -> None:
    expected = f"#SBATCH --{directive}={value}"
    if script_body.splitlines().count(expected) != 1:
        raise ValueError(f"rendered script must contain exactly one {expected!r} directive")


def _timestamp(value: str, name: str) -> None:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", value) is None:
        raise ValueError(f"{name} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid UTC timestamp") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{name} must be an explicit UTC timestamp")


__all__ = [
    "PhaseActionDispatchIntendedEvent",
    "PhaseActionDispatchIntendedEventType",
    "PhaseActionDispatchIntendedPayload",
    "PhaseActionDispatchRejectedEvent",
    "PhaseActionDispatchRejectedEventType",
    "PhaseActionDispatchRejectedPayload",
    "PhaseActionSatisfiedWithoutDispatchEvent",
    "PhaseActionSatisfiedWithoutDispatchEventType",
    "PhaseActionSatisfiedWithoutDispatchPayload",
    "PhaseActionSubmissionPlan",
    "PhaseActionSubmissionStatus",
    "PhaseActionSubmissionView",
    "PhaseActionSubmittedEvent",
    "PhaseActionSubmittedEventType",
    "PhaseActionSubmittedPayload",
    "PhaseContainerMountDescriptor",
    "PhaseSubmissionIntendedEvent",
    "PhaseSubmissionIntendedEventType",
    "PhaseSubmissionIntendedPayload",
    "PhaseSubmissionLifecycleView",
    "PhaseSubmissionStatus",
    "phase_action_dispatch_intended_event_from_mapping",
    "phase_action_dispatch_intended_payload_from_mapping",
    "phase_action_dispatch_rejected_event_from_mapping",
    "phase_action_dispatch_rejected_payload_from_mapping",
    "phase_action_satisfied_without_dispatch_event_from_mapping",
    "phase_action_satisfied_without_dispatch_payload_from_mapping",
    "phase_action_scheduler_correlation_token",
    "phase_action_submission_identity_mapping",
    "phase_action_submission_plan_from_mapping",
    "phase_action_submission_view_from_mapping",
    "phase_action_submitted_event_from_mapping",
    "phase_action_submitted_payload_from_mapping",
    "phase_container_mount_descriptor_from_mapping",
    "phase_submission_id",
    "phase_submission_intended_event_from_mapping",
    "phase_submission_intended_payload_from_mapping",
    "phase_submission_lifecycle_view_from_mapping",
]
