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

"""Restart-safe coordinator for one explicit postprocessing Phase Retry."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from bspp.orchestration.contract.source_package import source_package_identity_from_mapping
from bspp.orchestration.control.postprocessing_authority_store import postprocessing_phase_coordinator_lock
from bspp.orchestration.control.postprocessing_phase_operator import (
    DEFAULT_POSTPROCESSING_PHASE_OPERATOR_DEPENDENCIES,
    PostprocessingPhaseOperatorDependencies,
    PostprocessingPhaseOperatorResult,
    PostprocessingPhaseOperatorStatus,
    _bounded_detail,
    _canonical_json,
    _drive_postprocessing_attempt,
    _intent_path,
    _operation_lock,
    _OperatorMismatchError,
    _OperatorStopError,
    _reject_overlapping_roots,
    _required_mapping,
    _required_positive_number,
    _required_string,
    _stable_regular_bytes,
    _stop_if_requested,
    _TransitionWriter,
    _with_diagnostics,
    _write_create_once_json,
)
from bspp.orchestration.control.postprocessing_phase_retry import (
    PostprocessingRetryResult,
    PreparedPostprocessingRetry,
    apply_prepared_postprocessing_retry,
    prepare_postprocessing_retry,
    prepared_postprocessing_retry_from_mapping,
)
from bspp.orchestration.control.runtime_qualification_validation import validate_runtime_qualification_source

_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_MAX_RETRY_INTENT_BYTES = 4 * 1024 * 1024
_MAX_INPUT_DOCUMENT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class PostprocessingPhaseRetryOperatorDependencies:
    lifecycle: PostprocessingPhaseOperatorDependencies
    prepare_retry: Callable[..., PreparedPostprocessingRetry]
    apply_retry: Callable[..., PostprocessingRetryResult]


DEFAULT_POSTPROCESSING_PHASE_RETRY_OPERATOR_DEPENDENCIES = PostprocessingPhaseRetryOperatorDependencies(
    lifecycle=DEFAULT_POSTPROCESSING_PHASE_OPERATOR_DEPENDENCIES,
    prepare_retry=prepare_postprocessing_retry,
    apply_retry=apply_prepared_postprocessing_retry,
)


@dataclass(frozen=True)
class PostprocessingPhaseRetryOperatorResult:
    status: PostprocessingPhaseOperatorStatus
    execution_root: Path
    phase_run_id: str
    predecessor_attempt_id: str | None
    successor_attempt_id: str | None
    retry_id: str | None
    recovery_command: str | None
    detail: str | None = None
    diagnostics: str | None = None
    explicit_cancel_command: str | None = None

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase_kind": "postprocessing",
            "operation": "retry-postprocessing",
            "status": self.status,
            "execution_root": str(self.execution_root),
            "phase_run_id": self.phase_run_id,
            "predecessor_attempt_id": self.predecessor_attempt_id,
            "successor_attempt_id": self.successor_attempt_id,
            "retry_id": self.retry_id,
            "recovery_command": self.recovery_command,
            "explicit_cancel_command": self.explicit_cancel_command,
            "diagnostics": self.diagnostics,
            "detail": self.detail,
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


def retry_postprocessing_phase_to_completion(
    phase_run_id: str,
    *,
    authority_root: Path,
    execution_root: Path,
    source_repo: Path,
    config_path: Path,
    poll_interval_seconds: float = 30.0,
    timeout_seconds: float = 24 * 60 * 60,
    dependencies: PostprocessingPhaseRetryOperatorDependencies = (
        DEFAULT_POSTPROCESSING_PHASE_RETRY_OPERATOR_DEPENDENCIES
    ),
) -> PostprocessingPhaseRetryOperatorResult:
    """Explicitly Retry one failed/cancelled Attempt and drive only its exact successor."""
    if _PHASE_RUN_ID.fullmatch(phase_run_id) is None:
        raise ValueError("postprocessing Retry coordinator Phase Run id is invalid")
    if poll_interval_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("postprocessing Retry coordinator poll interval and timeout must be positive")
    source_root = source_repo.resolve(strict=True)
    config = config_path.resolve(strict=True)
    authority = authority_root.resolve()
    execution = execution_root.resolve()
    _reject_overlapping_roots(authority, execution, source_root)
    recovery = _retry_recovery_command(
        phase_run_id=phase_run_id,
        authority_root=authority,
        execution_root=execution,
        source_repo=source_root,
        config_path=config,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )
    started = dependencies.lifecycle.monotonic_clock()
    prepared: PreparedPostprocessingRetry | None = None
    successor_driver_entered = False
    try:
        with ExitStack() as locks:
            locks.enter_context(_operation_lock(execution))
            locks.enter_context(postprocessing_phase_coordinator_lock(authority, phase_run_id))
            config_document = _stable_regular_bytes(config, maximum=_MAX_INPUT_DOCUMENT_BYTES)
            intent_path = execution / "operation-intent.json"
            if os.path.lexists(intent_path):
                persisted = _load_retry_intent(intent_path)
                recovery = _retry_recovery_command_from_intent(persisted)
                try:
                    retry_event = _required_mapping(persisted, "retry_event")
                    prepared = prepared_postprocessing_retry_from_mapping(retry_event)
                    _require_prepared_phase(prepared, expected_phase_run_id=phase_run_id)
                    expected = _retry_intent_identity(
                        prepared=prepared,
                        authority_root=authority,
                        execution_root=execution,
                        source_repo=source_root,
                        config_path=config,
                        config_document=config_document,
                        poll_interval_seconds=poll_interval_seconds,
                        timeout_seconds=timeout_seconds,
                    )
                except _OperatorMismatchError:
                    raise
                except (OSError, TypeError, ValueError) as exc:
                    raise _OperatorMismatchError(
                        "postprocessing Retry invocation no longer matches its persisted qualification"
                    ) from exc
                if persisted != expected:
                    raise _OperatorMismatchError("postprocessing Retry invocation differs from its create-once intent")
            else:
                prepared = dependencies.prepare_retry(
                    phase_run_id,
                    authority_root=authority,
                    config_path=config,
                    source_repo=source_root,
                    clock=dependencies.lifecycle.wall_clock,
                )
                _require_prepared_phase(prepared, expected_phase_run_id=phase_run_id)
                persisted = _retry_intent_identity(
                    prepared=prepared,
                    authority_root=authority,
                    execution_root=execution,
                    source_repo=source_root,
                    config_path=config,
                    config_document=config_document,
                    poll_interval_seconds=poll_interval_seconds,
                    timeout_seconds=timeout_seconds,
                )
                _write_create_once_json(intent_path, persisted)
            _stop_if_requested(dependencies.lifecycle, started, timeout_seconds)
            applied = dependencies.apply_retry(prepared, authority_root=authority)
            _require_exact_retry_apply_result(
                applied,
                prepared=prepared,
                expected_phase_run_id=phase_run_id,
            )
            writer = _TransitionWriter(
                execution,
                phase_run_id,
                dependencies.lifecycle,
                operation="retry-postprocessing",
            )
            successor_driver_entered = True
            base = _drive_postprocessing_attempt(
                phase_run_id=phase_run_id,
                expected_attempt_id=prepared.successor_attempt_id,
                authority_root=authority,
                execution_root=execution,
                recovery_command=recovery,
                poll_interval_seconds=poll_interval_seconds,
                timeout_seconds=timeout_seconds,
                started=started,
                writer=writer,
                dependencies=dependencies.lifecycle,
            )
            return _retry_operator_result(base, prepared=prepared)
    except BlockingIOError:
        return _retry_failure(
            status="lock-conflict",
            execution_root=execution,
            phase_run_id=phase_run_id,
            prepared=prepared,
            recovery=recovery,
            detail="another postprocessing coordinator owns an operation lock; no cancellation was requested",
        )
    except _OperatorMismatchError as exc:
        return _retry_failure(
            status="configuration-mismatch",
            execution_root=execution,
            phase_run_id=phase_run_id,
            prepared=prepared,
            recovery=recovery,
            detail=_bounded_detail(str(exc)),
        )
    except KeyboardInterrupt:
        return _retry_interruption_result(
            execution=execution,
            authority=authority,
            phase_run_id=phase_run_id,
            prepared=prepared,
            recovery=recovery,
            status="interrupted",
            detail="postprocessing Retry coordinator interrupted; no cancellation was requested",
            dependencies=dependencies,
            diagnostics_allowed=successor_driver_entered,
        )
    except _OperatorStopError as exc:
        return _retry_interruption_result(
            execution=execution,
            authority=authority,
            phase_run_id=phase_run_id,
            prepared=prepared,
            recovery=recovery,
            status=exc.status,
            detail=exc.detail,
            dependencies=dependencies,
            diagnostics_allowed=successor_driver_entered,
        )
    except (OSError, TypeError, ValueError) as exc:
        return _retry_interruption_result(
            execution=execution,
            authority=authority,
            phase_run_id=phase_run_id,
            prepared=prepared,
            recovery=recovery,
            status="lifecycle-failed",
            detail=_bounded_detail(str(exc)),
            dependencies=dependencies,
            diagnostics_allowed=successor_driver_entered,
        )


def _require_prepared_phase(
    prepared: PreparedPostprocessingRetry,
    *,
    expected_phase_run_id: str,
) -> None:
    if prepared.phase_run_id != expected_phase_run_id:
        raise _OperatorMismatchError(
            "postprocessing Retry coordinator expected Phase Run "
            f"{expected_phase_run_id}, but the prepared transition selects {prepared.phase_run_id}"
        )


def _require_exact_retry_apply_result(
    result: object,
    *,
    prepared: PreparedPostprocessingRetry,
    expected_phase_run_id: str,
) -> None:
    successor = prepared.payload.successor_phase_runspec
    expected = {
        "phase_run_id": expected_phase_run_id,
        "predecessor_attempt_id": prepared.predecessor_attempt_id,
        "successor_attempt_id": prepared.successor_attempt_id,
        "retry_id": prepared.retry_id,
        "phase_runspec_digest": successor.digest,
        "phase_runspec_location": f"attempts/{prepared.successor_attempt_id}/phase-runspec.json",
        "phase_plan_digest": prepared.payload.phase_plan_digest,
        "logical_input_manifest_digest": prepared.payload.logical_input_manifest_digest,
        "selected_cluster_profile": successor.cluster.profile_name,
        "status": "materialized",
    }
    mismatches = tuple(
        field for field, expected_value in expected.items() if getattr(result, field, None) != expected_value
    )
    if mismatches:
        raise _OperatorMismatchError(
            "postprocessing Retry materialization differs from its intent in: " + ", ".join(mismatches)
        )


def _retry_intent_identity(
    *,
    prepared: PreparedPostprocessingRetry,
    authority_root: Path,
    execution_root: Path,
    source_repo: Path,
    config_path: Path,
    config_document: bytes,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, object]:
    payload = prepared.payload
    qualification_bytes = payload.successor_runtime_qualification_json.encode()
    qualification_document = json.loads(qualification_bytes)
    if not isinstance(qualification_document, Mapping):
        raise ValueError("postprocessing Retry qualification must be a mapping")
    tuple_mapping = qualification_document.get("tuple")
    if not isinstance(tuple_mapping, Mapping):
        raise ValueError("postprocessing Retry qualification tuple must be a mapping")
    source_mapping = tuple_mapping.get("source_package_identity")
    if not isinstance(source_mapping, Mapping):
        raise ValueError("postprocessing Retry qualification source identity must be a mapping")
    source = source_package_identity_from_mapping(cast("Mapping[str, object]", source_mapping))
    validate_runtime_qualification_source(source_repo, commit=source.commit, tree=source.tree)
    qualified = payload.successor_phase_runspec.payload.qualified_runtime
    return {
        "format_version": 1,
        "operation_kind": "postprocessing-phase-retry-operator-v1",
        "phase_run_id": prepared.phase_run_id,
        "authority_root": str(authority_root),
        "execution_root": str(execution_root),
        "phase_plan_digest": payload.phase_plan_digest,
        "transition": {
            "predecessor_attempt_id": prepared.predecessor_attempt_id,
            "predecessor_phase_runspec_digest": payload.predecessor_phase_runspec_digest,
            "predecessor_outcome": payload.predecessor_outcome,
            "successor_attempt_id": prepared.successor_attempt_id,
            "successor_phase_runspec_digest": payload.successor_phase_runspec.digest,
            "retry_id": prepared.retry_id,
            "materialized_at": prepared.event.occurred_at,
        },
        "source": {
            "repository": str(source_repo),
            "commit": source.commit,
            "tree": source.tree,
            "source_package_sha256": source.package_sha256,
            "source_package_manifest_sha256": source.manifest_sha256,
        },
        "profile": {
            "name": payload.successor_phase_runspec.cluster.profile_name,
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_document).hexdigest(),
            "config_size_bytes": len(config_document),
        },
        "qualification": {
            "tuple_id": qualified.tuple_id,
            "record_sha256": hashlib.sha256(qualification_bytes).hexdigest(),
            "record_size_bytes": len(qualification_bytes),
            "qualified_runtime_digest": qualified.digest,
            "document": dict(qualification_document),
        },
        "retry_event": prepared.to_mapping(),
        "options": {
            "poll_interval_seconds": poll_interval_seconds,
            "timeout_seconds": timeout_seconds,
        },
    }


def _load_retry_intent(path: Path) -> dict[str, object]:
    document = _stable_regular_bytes(path, maximum=_MAX_RETRY_INTENT_BYTES)
    try:
        payload = json.loads(document)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _OperatorMismatchError("postprocessing Retry intent is not valid JSON") from exc
    expected_fields = {
        "format_version",
        "operation_kind",
        "phase_run_id",
        "authority_root",
        "execution_root",
        "phase_plan_digest",
        "transition",
        "source",
        "profile",
        "qualification",
        "retry_event",
        "options",
    }
    if not isinstance(payload, dict) or _canonical_json(payload) != document:
        raise _OperatorMismatchError("postprocessing Retry intent is not canonical JSON")
    if set(payload) != expected_fields:
        raise _OperatorMismatchError("postprocessing Retry intent has missing or extra fields")
    if payload.get("format_version") != 1 or payload.get("operation_kind") != "postprocessing-phase-retry-operator-v1":
        raise _OperatorMismatchError("postprocessing Retry intent has an unsupported format")
    return cast("dict[str, object]", payload)


def _retry_recovery_command(
    *,
    phase_run_id: str,
    authority_root: Path,
    execution_root: Path,
    source_repo: Path,
    config_path: Path,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> str:
    return shlex.join(
        (
            "bsppctl",
            "--config",
            str(config_path),
            "phase",
            "retry-postprocessing",
            phase_run_id,
            "--authority-root",
            str(authority_root),
            "--execution-root",
            str(execution_root),
            "--source-repo",
            str(source_repo),
            "--poll-interval",
            f"{poll_interval_seconds:g}",
            "--timeout",
            f"{timeout_seconds:g}",
        )
    )


def _retry_recovery_command_from_intent(intent: Mapping[str, object]) -> str:
    return _retry_recovery_command(
        phase_run_id=_required_string(intent, "phase_run_id"),
        authority_root=_intent_path(intent, None, "authority_root"),
        execution_root=_intent_path(intent, None, "execution_root"),
        source_repo=_intent_path(intent, "source", "repository"),
        config_path=_intent_path(intent, "profile", "config_path"),
        poll_interval_seconds=_required_positive_number(_required_mapping(intent, "options"), "poll_interval_seconds"),
        timeout_seconds=_required_positive_number(_required_mapping(intent, "options"), "timeout_seconds"),
    )


def _retry_operator_result(
    result: PostprocessingPhaseOperatorResult,
    *,
    prepared: PreparedPostprocessingRetry,
) -> PostprocessingPhaseRetryOperatorResult:
    return PostprocessingPhaseRetryOperatorResult(
        status=result.status,
        execution_root=result.execution_root,
        phase_run_id=prepared.phase_run_id,
        predecessor_attempt_id=prepared.predecessor_attempt_id,
        successor_attempt_id=prepared.successor_attempt_id,
        retry_id=prepared.retry_id,
        recovery_command=result.recovery_command,
        detail=result.detail,
        diagnostics=result.diagnostics,
        explicit_cancel_command=result.explicit_cancel_command,
    )


def _retry_failure(
    *,
    status: PostprocessingPhaseOperatorStatus,
    execution_root: Path,
    phase_run_id: str,
    prepared: PreparedPostprocessingRetry | None,
    recovery: str,
    detail: str,
) -> PostprocessingPhaseRetryOperatorResult:
    return PostprocessingPhaseRetryOperatorResult(
        status=status,
        execution_root=execution_root,
        phase_run_id=phase_run_id,
        predecessor_attempt_id=prepared.predecessor_attempt_id if prepared else None,
        successor_attempt_id=prepared.successor_attempt_id if prepared else None,
        retry_id=prepared.retry_id if prepared else None,
        recovery_command=recovery,
        detail=detail,
    )


def _retry_interruption_result(
    *,
    execution: Path,
    authority: Path,
    phase_run_id: str,
    prepared: PreparedPostprocessingRetry | None,
    recovery: str,
    status: Literal["lifecycle-failed", "timed-out", "interrupted"],
    detail: str,
    dependencies: PostprocessingPhaseRetryOperatorDependencies,
    diagnostics_allowed: bool,
) -> PostprocessingPhaseRetryOperatorResult:
    base = PostprocessingPhaseOperatorResult(
        status=status,
        execution_root=execution,
        phase_run_id=phase_run_id,
        attempt_id=prepared.successor_attempt_id if prepared else None,
        recovery_command=recovery,
        detail=detail,
    )
    if prepared is not None and diagnostics_allowed:
        base = _with_diagnostics(
            base,
            phase_run_id=phase_run_id,
            authority_root=authority,
            execution_root=execution,
            dependencies=dependencies.lifecycle,
        )
        return _retry_operator_result(base, prepared=prepared)
    return _retry_failure(
        status=status,
        execution_root=execution,
        phase_run_id=phase_run_id,
        prepared=prepared,
        recovery=recovery,
        detail=detail,
    )


__all__ = [
    "DEFAULT_POSTPROCESSING_PHASE_RETRY_OPERATOR_DEPENDENCIES",
    "PostprocessingPhaseRetryOperatorDependencies",
    "PostprocessingPhaseRetryOperatorResult",
    "retry_postprocessing_phase_to_completion",
]
