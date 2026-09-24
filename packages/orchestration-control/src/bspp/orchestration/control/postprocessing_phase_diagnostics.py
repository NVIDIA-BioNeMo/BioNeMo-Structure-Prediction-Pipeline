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

"""Bounded, non-authoritative diagnostics for postprocessing Phase operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from bspp.orchestration.contract.postprocessing_acceptance_adjudication import (
    postprocessing_acceptance_adjudication_from_mapping,
)
from bspp.orchestration.contract.postprocessing_acceptance_capture import (
    PostprocessingAcceptanceCapture,
    postprocessing_acceptance_capture_from_mapping,
)
from bspp.orchestration.contract.postprocessing_acceptance_diagnostics import (
    canonical_allowance_id,
    diagnostic_allowance_reference,
    evaluate_postprocessing_acceptance_reports,
    project_unallowlisted_occurrences,
)
from bspp.orchestration.contract.postprocessing_acceptance_policy import PostprocessingAcceptancePolicySnapshot
from bspp.orchestration.contract.postprocessing_action_contract import PostprocessingRuntimeAction
from bspp.orchestration.contract.postprocessing_submission_events import (
    POSTPROCESSING_RENDERER_CONTRACT_SUPPORTED,
)
from bspp.orchestration.contract.postprocessing_terminal_events import (
    PostprocessingActionTerminalObservedPayload,
)
from bspp.orchestration.control.monitoring import SlurmJobRecord, SlurmObservation
from bspp.orchestration.control.postprocessing_authority_reader import require_postprocessing_v2_authority
from bspp.orchestration.control.postprocessing_phase_lifecycle import (
    _expected_terminal_job_ids,
    postprocessing_transport,
)
from bspp.orchestration.control.postprocessing_phase_types import PostprocessingAuthority
from bspp.orchestration.control.transport import (
    TERMINAL_SLURM_STATES,
    CommandRunner,
    default_command_runner,
)

MAX_DIAGNOSTIC_LOG_FILES = 16
MAX_REMOTE_LOG_BYTES = 64 * 1024
MAX_RENDERED_TAIL_BYTES = 8 * 1024
MAX_DIAGNOSTIC_WARNINGS = 16
MAX_DIAGNOSTIC_WARNING_CHARS = 1024
MAX_ACCEPTANCE_SUPPORT_FILES = 16
MAX_ACCEPTANCE_SUPPORT_BYTES = 1024 * 1024
MAX_ACCEPTANCE_OCCURRENCE_ROWS = 64
MAX_ACCEPTANCE_OCCURRENCE_VALUE_CHARS = 1024

_AcceptanceSupportRequest = tuple[str, str, str | None, PurePosixPath]

_URI_USERINFO = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^/@\s]+@")
_URI_QUERY_VALUE = re.compile(r"([?&][^=&#\s]+=)[^&#\s]*")


@dataclass(frozen=True)
class PostprocessingDiagnosticsResult:
    phase_run_id: str
    attempt_id: str
    output: Path
    captured_log_files: int
    missing_log_files: int
    warnings: tuple[str, ...]
    acceptance_adjudication_status: str = "not-selected"
    acceptance_support_status: str = "not-applicable"
    acceptance_support_files: int = 0

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "phase_kind": "postprocessing",
            "operation": "diagnostics",
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "output": str(self.output),
            "captured_log_files": self.captured_log_files,
            "missing_log_files": self.missing_log_files,
            "warnings": list(self.warnings),
            "acceptance_adjudication_status": self.acceptance_adjudication_status,
            "acceptance_support_status": self.acceptance_support_status,
            "acceptance_support_files": self.acceptance_support_files,
            "status": "captured",
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class _DiagnosticFailure:
    action_id: str
    source: str
    scheduler_job_id: str
    task_index: int | None
    log_job_token: str
    state: str
    exit_code: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "source": self.source,
            "scheduler_job_id": self.scheduler_job_id,
            "task_index": self.task_index,
            "log_job_token": self.log_job_token,
            "state": self.state,
            "exit_code": self.exit_code,
        }


def capture_postprocessing_phase_diagnostics(
    phase_run_id: str,
    *,
    authority_root: Path,
    diagnostics_root: Path,
    runner: CommandRunner = default_command_runner,
    clock: Callable[[], datetime] | None = None,
) -> PostprocessingDiagnosticsResult:
    """Capture one compact summary from durable assignments and exact V3 log paths."""
    authority = require_postprocessing_v2_authority(authority_root, phase_run_id)
    _require_diagnostics_root_outside_authority(authority_root, diagnostics_root)
    submission = authority.submission_state
    assigned = submission.job_ids_by_action() if submission is not None else {}
    parent_job_ids = tuple(
        assigned[action.action_id] for action in authority.runspec.payload.actions if action.action_id in assigned
    )
    transport = postprocessing_transport(authority, runner=runner)
    scheduler: list[dict[str, object]] = []
    warnings: list[str] = []
    observation = None
    if parent_job_ids:
        assert submission is not None
        observation = transport.query_observation_best_effort(
            parent_job_ids,
            require_exact_terminal_exit=True,
            expected_terminal_job_ids=_expected_terminal_job_ids(authority, submission),
        )
        selected = observation.selected_state_by_job_id()
        scheduler = [
            selected[parent_job_id].to_mapping() for parent_job_id in parent_job_ids if parent_job_id in selected
        ]
        warnings.extend(_bounded_warning(item) for item in observation.warnings[:MAX_DIAGNOSTIC_WARNINGS])

    renderer_versions = tuple(
        sorted({item.renderer_contract_version for item in submission.actions}) if submission is not None else ()
    )
    if any(version not in POSTPROCESSING_RENDERER_CONTRACT_SUPPORTED for version in renderer_versions):
        raise ValueError("postprocessing diagnostics found an unsupported renderer contract version")
    if len(renderer_versions) > 1:
        raise ValueError("postprocessing diagnostics found mixed renderer contract versions")
    failures = _diagnostic_failures(
        authority,
        assigned=assigned,
        observation=observation,
    )
    failed_action_ids = frozenset(item.action_id for item in failures)
    durable_terminal_action_ids = frozenset(payload.action_id for payload in authority.terminal_payloads)
    missing_durable_terminal_action_ids = tuple(
        action.action_id
        for action in authority.runspec.payload.actions
        if action.action_id in assigned and action.action_id not in durable_terminal_action_ids
    )
    candidates = (
        _v3_log_paths(authority, failures) if renderer_versions in {(3,), (4,), (5,)} and failed_action_ids else ()
    )
    logs: list[dict[str, object]] = []
    for action_id, stream, remote_path in candidates[:MAX_DIAGNOSTIC_LOG_FILES]:
        row: dict[str, object] = {
            "action_id": action_id,
            "stream": stream,
            "remote_path": remote_path,
        }
        try:
            document = transport.read_immutable_bytes_artifact_no_follow(
                remote_path,
                max_bytes=MAX_REMOTE_LOG_BYTES,
            )
        except FileNotFoundError:
            row["status"] = "missing"
        except (OSError, TypeError, ValueError) as exc:
            row["status"] = "unreadable"
            row["warning"] = _bounded_warning(str(exc))
        else:
            tail = document[-MAX_RENDERED_TAIL_BYTES:].decode("utf-8", errors="replace")
            row.update(
                status="captured",
                size_bytes=len(document),
                sha256=hashlib.sha256(document).hexdigest(),
                tail=_redact_credentials(tail),
            )
        logs.append(row)

    acceptance_adjudication, acceptance_support = _capture_acceptance_diagnostics(
        authority,
        transport=transport,
        renderer_versions=renderer_versions,
        failures=failures,
    )

    captured_at = _timestamp((clock or _now)())
    summary: dict[str, object] = {
        "schema_version": 3,
        "diagnostic_kind": "postprocessing-phase-diagnostics-v3",
        "phase_run_id": authority.phase_run_id,
        "attempt_id": authority.attempt_id,
        "captured_at": captured_at,
        "authority_status": authority.status,
        "parent_job_ids": list(parent_job_ids),
        "renderer_contract_versions": list(renderer_versions),
        "scheduler": scheduler,
        "scheduler_warnings": warnings,
        "selected_failure_actions": [item.to_mapping() for item in failures],
        "missing_durable_terminal_action_ids": list(missing_durable_terminal_action_ids),
        "logs": logs,
        "omitted_log_file_count": max(0, len(candidates) - MAX_DIAGNOSTIC_LOG_FILES),
        "log_policy": (
            "exact-renderer-v3-v5-paths"
            if renderer_versions in {(3,), (4,), (5,)}
            else "scheduler-only-for-renderer-contracts-1-and-2"
        ),
        "acceptance_adjudication": acceptance_adjudication,
        "acceptance_support": acceptance_support,
    }
    output = _publish_summary(
        diagnostics_root,
        summary,
        authority_root=authority_root,
    )
    return PostprocessingDiagnosticsResult(
        phase_run_id=authority.phase_run_id,
        attempt_id=authority.attempt_id,
        output=output,
        captured_log_files=sum(item.get("status") == "captured" for item in logs),
        missing_log_files=sum(item.get("status") == "missing" for item in logs),
        warnings=tuple(warnings),
        acceptance_adjudication_status=str(acceptance_adjudication["status"]),
        acceptance_support_status=str(acceptance_support["status"]),
        acceptance_support_files=_captured_support_file_count(acceptance_support),
    )


def _capture_acceptance_diagnostics(
    authority: PostprocessingAuthority,
    *,
    transport: object,
    renderer_versions: tuple[int, ...],
    failures: tuple[_DiagnosticFailure, ...],
) -> tuple[dict[str, object], dict[str, object]]:
    """Read only action09's fixed attempt-bound adjudication and its exact support."""
    base_support: dict[str, object] = {"status": "not-applicable", "files": [], "omitted_file_count": 0}
    if renderer_versions in {(1,), (2,)}:
        return {"status": "not-applicable"}, base_support
    action = next(
        (item for item in authority.runspec.payload.actions if item.step_name == "acceptance-adjudication"),
        None,
    )
    if action is None or action.action_id not in {item.action_id for item in failures}:
        return {"status": "not-selected"}, base_support
    evidence = PurePosixPath(authority.runspec.payload.attempt_paths.evidence_dir)
    if not evidence.is_absolute() or ".." in evidence.parts:
        return {"status": "invalid", "warning": "attempt evidence directory is not canonical"}, base_support
    path = str(evidence / "phase-acceptance" / "adjudication.json")
    try:
        document = _read_diagnostic_artifact(transport, path)
    except FileNotFoundError:
        return {"status": "missing", "remote_path": path}, base_support
    except (OSError, TypeError, ValueError) as exc:
        return {"status": "unreadable", "remote_path": path, "warning": _bounded_warning(str(exc))}, base_support
    adjudication_row: dict[str, object] = {
        "status": "invalid",
        "remote_path": path,
        "size_bytes": len(document),
        "sha256": hashlib.sha256(document).hexdigest(),
    }
    try:
        payload = _json_object(document, label="acceptance adjudication")
        adjudication = postprocessing_acceptance_adjudication_from_mapping(payload)
        policy_sha = hashlib.sha256(authority.acceptance_policy_bytes).hexdigest()
        if (
            adjudication.phase_run_id != authority.phase_run_id
            or adjudication.attempt_id != authority.attempt_id
            or adjudication.policy_id != authority.acceptance_policy.policy_id
            or adjudication.policy_sha256 != policy_sha
        ):
            raise ValueError("acceptance adjudication identity or policy binding differs")
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        adjudication_row["warning"] = _bounded_warning(str(exc))
        return adjudication_row, base_support
    adjudication_row.update(
        status="captured",
        result=adjudication.result,
        non_allowlisted_errors=adjudication.non_allowlisted_errors,
        residual_cardinalities=_safe_residual_cardinalities(
            adjudication.residual_cardinalities,
            authority.acceptance_policy,
        ),
        failed_reconciliation_ids=[
            _bounded_warning(item.reconciliation_id) for item in adjudication.reconciliation_results if not item.matched
        ],
    )
    if adjudication.result == "passed":
        adjudication_row["post_adjudication_failure"] = True
    if adjudication.result != "failed" or adjudication.non_allowlisted_errors == 0:
        return adjudication_row, base_support
    support = _capture_acceptance_support(
        authority,
        transport=transport,
        evidence_root=evidence,
        expected_non_allowlisted_errors=adjudication.non_allowlisted_errors,
        expected_capture_digests=adjudication.capture_digests,
    )
    return adjudication_row, support


def _read_diagnostic_artifact(transport: object, path: str) -> bytes:
    if not hasattr(transport, "read_immutable_bytes_artifact_no_follow"):
        raise TypeError("diagnostic transport cannot read immutable artifacts")
    document = transport.read_immutable_bytes_artifact_no_follow(path, max_bytes=MAX_REMOTE_LOG_BYTES)
    if not isinstance(document, bytes):
        raise TypeError("diagnostic transport returned non-bytes")
    return document


def _json_object(document: bytes, *, label: str) -> Mapping[str, object]:
    value = json.loads(document.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _capture_acceptance_support(
    authority: PostprocessingAuthority,
    *,
    transport: object,
    evidence_root: PurePosixPath,
    expected_non_allowlisted_errors: int,
    expected_capture_digests: tuple[str, ...],
) -> dict[str, object]:
    policy = authority.acceptance_policy
    policy_sha = hashlib.sha256(authority.acceptance_policy_bytes).hexdigest()
    steps: tuple[str, str, str] = (
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    )
    requests: list[_AcceptanceSupportRequest] = [
        (
            "capture",
            f"phase-acceptance/{step}-capture.json",
            step,
            evidence_root / "phase-acceptance" / f"{step}-capture.json",
        )
        for step in steps
    ]
    for step in steps:
        contract = next(item for item in policy.completion_exit_contracts if item.step_name == step)
        for path in sorted(contract.report_paths):
            relative_path = PurePosixPath(path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("acceptance diagnostic policy path is not a safe relative path")
            requests.append(("report", path, step, evidence_root / relative_path))
    deduplicated = _deduplicate_acceptance_support_requests(requests)
    return _capture_acceptance_support_reads(
        authority,
        transport=transport,
        policy=policy,
        policy_sha=policy_sha,
        evidence_root=evidence_root,
        steps=steps,
        deduplicated=deduplicated,
        requests=requests,
        expected_non_allowlisted_errors=expected_non_allowlisted_errors,
        expected_capture_digests=expected_capture_digests,
    )


def _deduplicate_acceptance_support_requests(
    requests: list[_AcceptanceSupportRequest],
) -> list[_AcceptanceSupportRequest]:
    """Keep the first request for each normalized exact remote path."""
    deduplicated: list[_AcceptanceSupportRequest] = []
    seen: set[PurePosixPath] = set()
    for item in requests:
        if item[3] not in seen:
            seen.add(item[3])
            deduplicated.append(item)
    return deduplicated


def _capture_acceptance_support_reads(
    authority: PostprocessingAuthority,
    *,
    transport: object,
    policy: PostprocessingAcceptancePolicySnapshot,
    policy_sha: str,
    evidence_root: PurePosixPath,
    steps: tuple[str, str, str],
    deduplicated: list[_AcceptanceSupportRequest],
    requests: list[_AcceptanceSupportRequest],
    expected_non_allowlisted_errors: int,
    expected_capture_digests: tuple[str, ...],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    captures: dict[str, PostprocessingAcceptanceCapture] = {}
    reports: dict[str, Mapping[str, object]] = {}
    report_documents: dict[PurePosixPath, Mapping[str, object]] = {}
    bytes_read = 0
    for kind, request_relative, request_step, absolute_path in deduplicated[:MAX_ACCEPTANCE_SUPPORT_FILES]:
        remote = str(absolute_path)
        row: dict[str, object] = {"kind": kind, "path": request_relative, "remote_path": remote}
        if bytes_read + MAX_REMOTE_LOG_BYTES > MAX_ACCEPTANCE_SUPPORT_BYTES:
            row["status"] = "omitted-limit"
            rows.append(row)
            continue
        try:
            document = _read_diagnostic_artifact(transport, remote)
        except FileNotFoundError:
            row["status"] = "missing"
        except (OSError, TypeError, ValueError) as exc:
            row.update(status="unreadable", warning=_bounded_warning(str(exc)))
        else:
            bytes_read += len(document)
            row.update(size_bytes=len(document), sha256=hashlib.sha256(document).hexdigest())
            try:
                mapping = _json_object(document, label=f"acceptance {kind}")
                if kind == "capture":
                    capture = postprocessing_acceptance_capture_from_mapping(mapping)
                    contract = next(item for item in policy.completion_exit_contracts if item.step_name == request_step)
                    if (
                        capture.phase_run_id != authority.phase_run_id
                        or capture.attempt_id != authority.attempt_id
                        or capture.step_name != request_step
                        or capture.policy_sha256 != policy_sha
                        or capture.raw_exit_code not in contract.allowed_raw_exit_codes
                        or tuple(item.path for item in capture.reports) != tuple(sorted(contract.report_paths))
                    ):
                        raise ValueError("acceptance capture identity, policy, or report set differs")
                    captures[str(request_step)] = capture
                else:
                    reports[request_relative] = mapping
                    report_documents[absolute_path] = mapping
                row["status"] = "captured"
            except (TypeError, ValueError, UnicodeDecodeError) as exc:
                row.update(status="invalid", warning=_bounded_warning(str(exc)))
        rows.append(row)
    expected_paths = {str(absolute) for _kind, _relative, _step, absolute in deduplicated}
    captured_paths = {str(row["remote_path"]) for row in rows if row.get("status") == "captured"}
    complete = len(rows) == len(deduplicated) and expected_paths == captured_paths
    result: dict[str, object] = {
        "status": "captured" if complete else "partial",
        "files": rows,
        "omitted_file_count": max(0, len(deduplicated) - MAX_ACCEPTANCE_SUPPORT_FILES),
        "returned_bytes": bytes_read,
        "complete": complete,
    }
    if not complete:
        return result
    for kind, raw_path, _step, absolute_path in requests:
        if kind == "report" and absolute_path in report_documents:
            reports[raw_path] = report_documents[absolute_path]
    ordered_digests = tuple(captures[step].digest for step in steps)
    if ordered_digests != expected_capture_digests:
        result.update(
            status="partial",
            complete=False,
            consistency_failure="acceptance capture digests differ from adjudication",
        )
        return result
    for step in steps:
        loaded_capture = captures.get(step)
        if loaded_capture is None:
            result["status"] = "partial"
            result["complete"] = False
            return result
        for binding in loaded_capture.reports:
            report_path = str(evidence_root / PurePosixPath(binding.path))
            document_row = next((row for row in rows if row["remote_path"] == report_path), None)
            if document_row is None or document_row.get("status") != "captured":
                result["status"] = "partial"
                result["complete"] = False
                return result
            # Report bytes are represented by their validated diagnostic hash/size.
            if document_row["size_bytes"] != binding.size_bytes or document_row["sha256"] != binding.sha256:
                result.update(status="partial", complete=False, consistency_failure="captured report binding differs")
                return result
    evaluation = evaluate_postprocessing_acceptance_reports(policy, reports)
    canonical_count = len(evaluation.unallowlisted_occurrences)
    if canonical_count > expected_non_allowlisted_errors:
        result.update(
            status="partial",
            complete=False,
            consistency_failure="canonical occurrence count exceeds adjudicated non-allowlisted count",
        )
        return result
    result["unallowlisted_projection"] = project_unallowlisted_occurrences(
        evaluation.unallowlisted_occurrences,
        max_rows=MAX_ACCEPTANCE_OCCURRENCE_ROWS,
        max_display_value_chars=MAX_ACCEPTANCE_OCCURRENCE_VALUE_CHARS,
    )
    result["unattributed_non_allowlisted_errors"] = expected_non_allowlisted_errors - canonical_count
    return result


def _safe_residual_cardinalities(
    items: object,
    policy: PostprocessingAcceptancePolicySnapshot,
) -> list[dict[str, object]]:
    if not isinstance(items, tuple):
        return []
    references: dict[str, list[dict[str, str]]] = {}
    for allowance in policy.residual_allowances:
        references.setdefault(canonical_allowance_id(allowance), []).append(diagnostic_allowance_reference(allowance))
    rows: list[dict[str, object]] = []
    for item in items:
        matching = references.get(item.allowance_id, [])
        reference: Mapping[str, object]
        if len(matching) == 1:
            reference = matching[0]
        else:
            reference = {
                "status": "unmatched-allowance",
                "allowance_id_sha256": hashlib.sha256(item.allowance_id.encode("utf-8")).hexdigest(),
            }
        rows.append(
            {
                "allowance_id": "<redacted>",
                "reference": reference,
                "observed": item.observed,
                "permitted_min": item.permitted_min,
                "permitted_max": item.permitted_max,
            }
        )
    return rows


def _captured_support_file_count(support: Mapping[str, object]) -> int:
    files = support.get("files")
    return (
        sum(isinstance(item, Mapping) and item.get("status") == "captured" for item in files)
        if isinstance(files, list)
        else 0
    )


def _diagnostic_failures(
    authority: PostprocessingAuthority,
    *,
    assigned: dict[str, str],
    observation: SlurmObservation | None,
) -> tuple[_DiagnosticFailure, ...]:
    durable = {payload.action_id: payload for payload in authority.terminal_payloads}
    result: list[_DiagnosticFailure] = []
    for action in authority.runspec.payload.actions:
        terminal = durable.get(action.action_id)
        if terminal is not None:
            if isinstance(terminal, PostprocessingActionTerminalObservedPayload):
                parent_job_id = assigned.get(action.action_id)
                if parent_job_id is None:
                    continue
                for task in terminal.tasks:
                    if not _is_execution_failure(task.state, task.exit_code):
                        continue
                    log_job_token = _exact_log_job_token(
                        action,
                        parent_job_id=parent_job_id,
                        scheduler_job_id=task.scheduler_job_id,
                        task_index=task.task_index,
                    )
                    if log_job_token is None:
                        continue
                    result.append(
                        _DiagnosticFailure(
                            action_id=action.action_id,
                            source="durable-terminal",
                            scheduler_job_id=task.scheduler_job_id,
                            task_index=task.task_index,
                            log_job_token=log_job_token,
                            state=task.state,
                            exit_code=task.exit_code,
                        )
                    )
            continue
        parent_job_id = assigned.get(action.action_id)
        if parent_job_id is not None and observation is not None:
            result.extend(_fresh_action_failures(action, parent_job_id=parent_job_id, observation=observation))
    return tuple(result)


def _fresh_action_failures(
    action: PostprocessingRuntimeAction,
    *,
    parent_job_id: str,
    observation: SlurmObservation,
) -> tuple[_DiagnosticFailure, ...]:
    records = _associated_accounting_records(observation, parent_job_id=parent_job_id)
    expected_indexes = action.expected_task_indexes
    selected: tuple[tuple[int | None, SlurmJobRecord, str], ...]
    if not expected_indexes:
        exact = tuple(item for item in records if item.job_id == parent_job_id)
        if len(exact) != 1 or any(item.job_id != parent_job_id for item in records):
            return ()
        selected = ((None, exact[0], parent_job_id),)
    else:
        expected_job_ids = {index: f"{parent_job_id}_{index}" for index in expected_indexes}
        valid_job_ids = frozenset(expected_job_ids.values())
        if any(item.job_id != parent_job_id and item.job_id not in valid_job_ids for item in records):
            return ()
        exact_children = {
            index: tuple(item for item in records if item.job_id == expected_job_id)
            for index, expected_job_id in expected_job_ids.items()
        }
        if any(len(exact_children[index]) != 1 for index in expected_indexes):
            return ()
        selected = tuple((index, exact_children[index][0], expected_job_ids[index]) for index in expected_indexes)
    if any(not _is_exact_terminal_record(record) for _task_index, record, _log_job_token in selected):
        return ()
    return tuple(
        _DiagnosticFailure(
            action_id=action.action_id,
            source="fresh-scheduler",
            scheduler_job_id=record.job_id,
            task_index=task_index,
            log_job_token=log_job_token,
            state=record.state,
            exit_code=record.exit_code,
        )
        for task_index, record, log_job_token in selected
        if record.state is not None
        and record.exit_code is not None
        and _is_execution_failure(record.state, record.exit_code)
    )


def _associated_accounting_records(
    observation: SlurmObservation,
    *,
    parent_job_id: str,
) -> tuple[SlurmJobRecord, ...]:
    return tuple(
        item
        for item in observation.sacct_jobs
        if item.source == "sacct"
        and "." not in item.job_id
        and (
            item.requested_job_id == parent_job_id
            or item.job_id == parent_job_id
            or item.job_id.startswith((f"{parent_job_id}_", f"{parent_job_id}["))
        )
    )


def _is_exact_terminal_record(record: SlurmJobRecord) -> bool:
    return (
        record.state in TERMINAL_SLURM_STATES
        and record.exit_code is not None
        and re.fullmatch(r"\d+:\d+", record.exit_code) is not None
    )


def _exact_log_job_token(
    action: PostprocessingRuntimeAction,
    *,
    parent_job_id: str,
    scheduler_job_id: str,
    task_index: int | None,
) -> str | None:
    if not action.expected_task_indexes:
        return parent_job_id if task_index is None and scheduler_job_id == parent_job_id else None
    if task_index is None:
        return None
    expected = f"{parent_job_id}_{task_index}"
    return expected if task_index in action.expected_task_indexes and scheduler_job_id == expected else None


def _is_execution_failure(state: str, exit_code: str) -> bool:
    return (
        state in TERMINAL_SLURM_STATES
        and state not in {"COMPLETED", "CANCELLED"}
        and bool(re.fullmatch(r"\d+:\d+", exit_code))
    )


def _v3_log_paths(
    authority: PostprocessingAuthority,
    failures: tuple[_DiagnosticFailure, ...],
) -> tuple[tuple[str, str, str], ...]:
    runspec = authority.runspec
    evidence = PurePosixPath(runspec.payload.attempt_paths.evidence_dir)
    if not evidence.is_absolute() or ".." in evidence.parts:
        raise ValueError("postprocessing diagnostic evidence root must be an absolute canonical path")
    log_root = evidence / "slurm-logs"
    result: list[tuple[str, str, str]] = []
    for action in runspec.payload.actions:
        action_failures = tuple(item for item in failures if item.action_id == action.action_id)
        for failure in action_failures:
            for stream in ("out", "err"):
                path = log_root / f"{action.step_name}.{failure.log_job_token}.{stream}"
                if path.parent != log_root or not path.is_absolute() or ".." in path.parts:
                    raise ValueError("postprocessing diagnostic log path escaped its frozen evidence directory")
                result.append((action.action_id, stream, str(path)))
    return tuple(result)


def _redact_credentials(value: str) -> str:
    redacted = _URI_USERINFO.sub(r"\1<redacted>@", value)
    return _URI_QUERY_VALUE.sub(r"\1<redacted>", redacted)


def _bounded_warning(value: str) -> str:
    return _redact_credentials(value.replace("\x00", ""))[:MAX_DIAGNOSTIC_WARNING_CHARS]


def _publish_summary(root: Path, summary: dict[str, object], *, authority_root: Path) -> Path:
    _require_diagnostics_root_outside_authority(authority_root, root)
    if os.path.lexists(root):
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
            raise ValueError("postprocessing diagnostics root must be a real directory")
    else:
        root.mkdir(parents=True, mode=0o700)
    output = root / "summary.json"
    if os.path.lexists(output) and (output.is_symlink() or not output.is_file()):
        raise ValueError("postprocessing diagnostics summary must be a regular file")
    document = (json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=".summary.", suffix=".tmp", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
        directory = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _require_diagnostics_root_outside_authority(authority_root: Path, diagnostics_root: Path) -> None:
    resolved_authority = authority_root.resolve(strict=False)
    resolved_diagnostics = diagnostics_root.resolve(strict=False)
    if resolved_diagnostics == resolved_authority or resolved_authority in resolved_diagnostics.parents:
        raise ValueError("postprocessing diagnostics root must be outside Phase authority")


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("postprocessing diagnostics clock must be timezone-aware")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "MAX_DIAGNOSTIC_LOG_FILES",
    "MAX_REMOTE_LOG_BYTES",
    "MAX_RENDERED_TAIL_BYTES",
    "PostprocessingDiagnosticsResult",
    "capture_postprocessing_phase_diagnostics",
]
