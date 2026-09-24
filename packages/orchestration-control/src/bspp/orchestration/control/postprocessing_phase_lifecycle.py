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

"""Shared typed replay projections for postprocessing lifecycle services."""

from __future__ import annotations

import hashlib
import re
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_action_contract import (
    PostprocessingRuntimeAction,
)
from bspp.orchestration.contract.postprocessing_cancellation_events import (
    PostprocessingCancellationIntendedPayload,
    PostprocessingJobCancellationRequestIntendedPayload,
    PostprocessingJobCancellationRequestResultPayload,
)
from bspp.orchestration.contract.postprocessing_runspec_v3 import PostprocessingPhaseRunSpecV3
from bspp.orchestration.contract.postprocessing_submission_events import (
    POSTPROCESSING_RENDERER_CONTRACT_CURRENT,
    PostprocessingActionSubmittedPayload,
)
from bspp.orchestration.contract.postprocessing_terminal_events import (
    PostprocessingActionTerminalObservedPayload,
    PostprocessingArrayParentCancelledObservedPayload,
    PostprocessingTaskTerminalEvidence,
    PostprocessingTerminalObservation,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    append_event as _append_event,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    format_timestamp as _format_timestamp,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    mapping_digest as _mapping_digest,
)
from bspp.orchestration.control.postprocessing_phase_rendering import (
    PostprocessingRenderInput,
    _cluster_attempt_root,
    render_postprocessing_action_script,
)
from bspp.orchestration.control.postprocessing_phase_types import (
    PostprocessingAuthority,
    PostprocessingLifecycleResult,
    PostprocessingSubmissionState,
    ReadablePostprocessingAuthority,
)
from bspp.orchestration.control.postprocessing_scheduler_identity import (
    postprocessing_cluster_action_script,
    postprocessing_scheduler_correlation_token,
)
from bspp.orchestration.control.transport import (
    TERMINAL_SLURM_STATES,
    CommandRunner,
    RemoteSlurmTransport,
)

Clock = Callable[[], datetime]
_JOB_ID = re.compile(r"[0-9]+")


def _validate_postprocessing_authority(authority_root: Path, phase_run_id: str) -> PostprocessingAuthority:
    from bspp.orchestration.control.postprocessing_authority_reader import require_postprocessing_v2_authority

    return require_postprocessing_v2_authority(authority_root, phase_run_id)


def _rendered_action_scripts(
    authority: PostprocessingAuthority, *, renderer_contract_version: int | None = None
) -> dict[str, str]:
    renderer_contract_version = renderer_contract_version or _renderer_contract_for_authority(authority)
    render_input = PostprocessingRenderInput(runspec=authority.runspec, legacy_runspec=authority.legacy_runspec)
    return {
        action.action_id: render_postprocessing_action_script(
            render_input, action, renderer_contract_version=renderer_contract_version
        )
        for action in authority.runspec.payload.actions
    }


def _renderer_contract_for_authority(authority: PostprocessingAuthority) -> int:
    if isinstance(authority.runspec, PostprocessingPhaseRunSpecV3):
        if authority.runspec.credential_mounts is None:
            raise ValueError(
                "pre-renderer-4 postprocessing V3 authority cannot create a new submission; rematerialize the Phase Run"
            )
        return POSTPROCESSING_RENDERER_CONTRACT_CURRENT
    return 2


def _postprocessing_submission_id(
    authority: PostprocessingAuthority,
    scripts: Mapping[str, str],
    *,
    renderer_contract_version: int,
) -> str:
    return "postprocessing-submission-" + _mapping_digest(
        {
            "schema_version": 1,
            "phase_run_id": authority.phase_run_id,
            "attempt_id": authority.attempt_id,
            "phase_runspec_digest": authority.runspec.digest,
            "actions": [
                {
                    "action_id": action.action_id,
                    "script_sha256": hashlib.sha256(scripts[action.action_id].encode()).hexdigest(),
                    "renderer_contract_version": renderer_contract_version,
                }
                for action in authority.runspec.payload.actions
            ],
        }
    )


def postprocessing_transport(
    authority: ReadablePostprocessingAuthority,
    *,
    runner: CommandRunner,
) -> RemoteSlurmTransport:
    return RemoteSlurmTransport(
        kind=authority.runspec.cluster.transport,
        ssh_target=authority.runspec.cluster.ssh_target,
        runner=runner,
    )


def _stage_postprocessing_attempt(
    authority: PostprocessingAuthority,
    *,
    scripts: Mapping[str, str],
    transport: RemoteSlurmTransport,
    submission: PostprocessingSubmissionState,
) -> None:
    attempt_root = authority.authority_path / "attempts" / authority.attempt_id
    cluster_root = _cluster_attempt_root(authority.runspec)
    token = submission.submission_id[-24:]
    artifacts = (
        (
            attempt_root / "phase-runspec.json",
            cluster_root / "phase-runspec.json",
            hashlib.sha256((attempt_root / "phase-runspec.json").read_bytes()).hexdigest(),
        ),
        (
            attempt_root / "legacy-runspec.yaml",
            cluster_root / "legacy-runspec.yaml",
            authority.runspec.payload.execution_projection.document_sha256,
        ),
        (
            attempt_root / "acceptance-policy.json",
            cluster_root / "acceptance-policy.json",
            authority.runspec.payload.acceptance_policy.sha256,
        ),
        (
            attempt_root / "runtime-qualification.json",
            cluster_root / "runtime-qualification.json",
            authority.runspec.payload.qualified_runtime.qualification_sha256,
        ),
    )
    for local_path, remote_path, digest in artifacts:
        transport.stage_immutable_artifact(
            local_path,
            str(remote_path),
            expected_sha256=digest,
            staging_token=f"{token}-{local_path.stem}",
        )
    with tempfile.TemporaryDirectory(prefix="bspp-postprocessing-actions-") as temporary:
        temporary_root = Path(temporary)
        for action in authority.runspec.payload.actions:
            script = scripts[action.action_id]
            local_script = temporary_root / f"{action.action_id}.sbatch"
            local_script.write_text(script)
            transport.stage_immutable_artifact(
                local_script,
                str(postprocessing_cluster_action_script(authority.runspec, action)),
                expected_sha256=hashlib.sha256(script.encode()).hexdigest(),
                staging_token=f"{token}-{action.action_id}",
            )
    result = transport.command(("mkdir", "-p", authority.runspec.payload.attempt_paths.evidence_dir + "/slurm-logs"))
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "postprocessing evidence directory creation failed")


def _submission_view(authority: PostprocessingAuthority) -> PostprocessingSubmissionState | None:
    return authority.submission_state


def _required_submission_view(authority: PostprocessingAuthority) -> PostprocessingSubmissionState:
    submission = _submission_view(authority)
    if submission is None:
        raise ValueError("postprocessing operation requires a durable submission intent")
    return submission


def _all_actions_submitted(authority: PostprocessingAuthority, submission: PostprocessingSubmissionState) -> bool:
    return len(submission.job_ids) == len(authority.runspec.payload.actions)


def _expected_terminal_job_ids(
    authority: ReadablePostprocessingAuthority,
    submission: PostprocessingSubmissionState,
) -> tuple[str, ...]:
    """Derive canonical exact scheduler endpoints from frozen actions and assignments."""
    assigned = submission.job_ids_by_action()
    action_ids = {action.action_id for action in authority.runspec.payload.actions}
    if any(not action_id or action_id not in action_ids for action_id in assigned):
        raise ValueError("postprocessing assignment references an unknown or missing action id")
    parents = tuple(assigned.values())
    if any(_JOB_ID.fullmatch(parent) is None for parent in parents):
        raise ValueError("postprocessing assignment must contain a numeric parent job id")
    if len(set(parents)) != len(parents):
        raise ValueError("postprocessing assignments contain duplicate parent job ids")
    endpoints: list[str] = []
    for action in authority.runspec.payload.actions:
        parent = assigned.get(action.action_id)
        if parent is None:
            continue
        indexes = action.expected_task_indexes
        if len(set(indexes)) != len(indexes) or any(index < 0 for index in indexes):
            raise ValueError("postprocessing action contains invalid or duplicate task indexes")
        if indexes:
            endpoints.extend(f"{parent}_{index}" for index in indexes)
        else:
            endpoints.append(parent)
    if any(not endpoint for endpoint in endpoints) or len(set(endpoints)) != len(endpoints):
        raise ValueError("postprocessing assignments produce missing or duplicate scheduler endpoints")
    return tuple(endpoints)


def _append_action_assignment(
    authority: PostprocessingAuthority,
    action: PostprocessingRuntimeAction,
    job_id: str,
    *,
    now: Clock,
) -> PostprocessingAuthority:
    if _JOB_ID.fullmatch(job_id) is None:
        raise ValueError("postprocessing Slurm assignment must be a numeric parent job id")
    submission = _required_submission_view(authority)
    return _append_event(
        authority,
        event_type="phase-action-submitted",
        occurred_at=_format_timestamp(now()),
        payload=PostprocessingActionSubmittedPayload(
            submission_id=submission.submission_id,
            action_id=action.action_id,
            parent_job_id=job_id,
            scheduler_correlation_token=postprocessing_scheduler_correlation_token(authority.runspec, action),
            expected_task_indexes=action.expected_task_indexes,
        ),
    )


def _recover_dispatching_actions(
    authority: PostprocessingAuthority,
    *,
    transport: RemoteSlurmTransport,
    now: Clock,
) -> PostprocessingAuthority:
    submission = _required_submission_view(authority)
    dispatching = tuple(submission.dispatching)
    by_id = {action.action_id: action for action in authority.runspec.payload.actions}
    for action_id in dispatching:
        action = by_id[action_id]
        token = postprocessing_scheduler_correlation_token(authority.runspec, action)
        matches = transport.query_submissions_by_correlation(
            job_name=token,
            comment=token,
            submitted_after=submission.intended_at,
        )
        if len(matches) > 1:
            raise ValueError(f"ambiguous postprocessing Slurm correlation for {action_id}")
        if len(matches) == 1:
            authority = _append_action_assignment(authority, action, matches[0].job_id, now=now)
    return authority


def _submission_result(
    authority: PostprocessingAuthority,
    submission: PostprocessingSubmissionState,
) -> PostprocessingLifecycleResult:
    jobs = submission.job_ids_by_action()
    return PostprocessingLifecycleResult(
        operation="submit",
        phase_run_id=authority.phase_run_id,
        attempt_id=authority.attempt_id,
        status="submitted",
        details={
            "phase_runspec_digest": authority.runspec.digest,
            "submission_id": submission.submission_id,
            "actions": [
                {"action_id": action.action_id, "job_id": jobs[action.action_id]}
                for action in authority.runspec.payload.actions
            ],
        },
    )


def _complete_action_task_set(
    action: PostprocessingRuntimeAction,
    parent_job_id: str,
    records: tuple[object, ...],
    *,
    autorequeue_enabled: bool,
) -> tuple[PostprocessingTaskTerminalEvidence, ...] | None:
    from bspp.orchestration.control.monitoring import SlurmJobRecord

    typed = tuple(item for item in records if isinstance(item, SlurmJobRecord) and item.source == "sacct")
    tasks: list[tuple[int | None, SlurmJobRecord]] = []
    if action.expected_task_indexes:
        for task_index in action.expected_task_indexes:
            job_id = f"{parent_job_id}_{task_index}"
            matches = tuple(item for item in typed if item.job_id == job_id)
            if len(matches) != 1 or matches[0].state not in TERMINAL_SLURM_STATES or matches[0].exit_code is None:
                return None
            record = matches[0]
            if (
                autorequeue_enabled
                and record.state == "COMPLETED"
                and record.exit_code in {"0", "0:0"}
                and not _exact_success_restarts(record)
            ):
                return None
            tasks.append((task_index, record))
    else:
        matches = tuple(item for item in typed if item.job_id == parent_job_id)
        if len(matches) != 1 or matches[0].state not in TERMINAL_SLURM_STATES or matches[0].exit_code is None:
            return None
        record = matches[0]
        if (
            autorequeue_enabled
            and record.state == "COMPLETED"
            and record.exit_code in {"0", "0:0"}
            and not _exact_success_restarts(record)
        ):
            return None
        tasks.append((None, record))
    return tuple(
        PostprocessingTaskTerminalEvidence(
            task_index=task_index,
            scheduler_job_id=item.job_id,
            state=cast("str", item.state),
            exit_code=cast("str", item.exit_code),
            source="sacct",
            restarts=item.restarts if item.state == "COMPLETED" and item.exit_code in {"0", "0:0"} else None,
        )
        for task_index, item in tasks
    )


def _append_complete_terminal_actions(
    authority: PostprocessingAuthority,
    *,
    submission: PostprocessingSubmissionState,
    observation: object,
    now: Clock,
) -> tuple[PostprocessingAuthority, tuple[str, ...]]:
    from bspp.orchestration.control.monitoring import SlurmObservation

    if not isinstance(observation, SlurmObservation):
        raise TypeError("postprocessing reconciliation requires a SlurmObservation")
    job_ids = submission.job_ids_by_action()
    terminal_by_action = _terminal_views(authority)
    autorequeue_policy = getattr(authority.runspec.payload, "autorequeue_policy", None)
    autorequeue_enabled = autorequeue_policy is not None and autorequeue_policy.mode == "enabled"
    newly_terminal: list[str] = []
    for action in authority.runspec.payload.actions:
        job_id = job_ids.get(action.action_id)
        if job_id is None or action.action_id in terminal_by_action:
            continue
        tasks = _complete_action_task_set(
            action,
            job_id,
            observation.sacct_jobs,
            autorequeue_enabled=autorequeue_enabled,
        )
        if tasks is None:
            continue
        outcome: Literal["succeeded", "failed"] = (
            "succeeded" if all(_task_succeeded(item) for item in tasks) else "failed"
        )
        authority = _append_event(
            authority,
            event_type="phase-action-terminal-observed",
            occurred_at=_format_timestamp(now()),
            payload=PostprocessingActionTerminalObservedPayload(
                submission_id=submission.submission_id,
                phase_runspec_digest=authority.runspec.digest,
                action_id=action.action_id,
                runtime_action_digest=_mapping_digest(action.to_mapping()),
                parent_job_id=job_id,
                expected_task_indexes=action.expected_task_indexes,
                tasks=tasks,
                outcome=outcome,
            ),
        )
        newly_terminal.append(action.action_id)
    return authority, tuple(newly_terminal)


def _array_parent_cancelled_before_task_instantiation(
    action: PostprocessingRuntimeAction,
    parent_job_id: str,
    records: tuple[object, ...],
) -> PostprocessingTaskTerminalEvidence | None:
    """Return exact parent evidence only when the complete parent record set proves no children exist."""
    from bspp.orchestration.control.monitoring import SlurmJobRecord

    if not action.expected_task_indexes:
        return None
    typed = tuple(item for item in records if isinstance(item, SlurmJobRecord) and item.source == "sacct")
    associated = tuple(
        item
        for item in typed
        if item.requested_job_id == parent_job_id
        or item.job_id == parent_job_id
        or item.job_id.startswith((f"{parent_job_id}_", f"{parent_job_id}["))
    )
    if len(associated) != 1:
        return None
    parent = associated[0]
    if parent.job_id != parent_job_id or parent.state != "CANCELLED" or parent.exit_code != "0:0":
        return None
    return PostprocessingTaskTerminalEvidence(
        task_index=None,
        scheduler_job_id=parent_job_id,
        state="CANCELLED",
        exit_code="0:0",
        source="sacct",
    )


def _append_array_parent_cancelled_actions(
    authority: PostprocessingAuthority,
    *,
    submission: PostprocessingSubmissionState,
    observation: object,
    now: Clock,
) -> tuple[PostprocessingAuthority, tuple[str, ...]]:
    """Append cancellation-only terminal events proven by one parent and no child records."""
    from bspp.orchestration.control.monitoring import SlurmObservation

    if not isinstance(observation, SlurmObservation):
        raise TypeError("postprocessing cancellation reconciliation requires a SlurmObservation")
    if not _complete_parent_hierarchy_observation(observation):
        return authority, ()
    cancellation_intended = any(
        event.attempt_id == authority.attempt_id
        and isinstance(event.payload, PostprocessingCancellationIntendedPayload)
        for event in authority.events
    )
    if not cancellation_intended:
        return authority, ()
    job_ids = submission.job_ids_by_action()
    terminal_by_action = _terminal_views(authority)
    newly_terminal: list[str] = []
    for action in authority.runspec.payload.actions:
        parent_job_id = job_ids.get(action.action_id)
        if parent_job_id is None or action.action_id in terminal_by_action:
            continue
        parent = _array_parent_cancelled_before_task_instantiation(
            action,
            parent_job_id,
            observation.sacct_jobs,
        )
        if parent is None:
            continue
        authority = _append_event(
            authority,
            event_type="phase-array-parent-cancelled-observed",
            occurred_at=_format_timestamp(now()),
            payload=PostprocessingArrayParentCancelledObservedPayload(
                submission_id=submission.submission_id,
                phase_runspec_digest=authority.runspec.digest,
                action_id=action.action_id,
                runtime_action_digest=_mapping_digest(action.to_mapping()),
                parent_job_id=parent_job_id,
                expected_task_indexes=action.expected_task_indexes,
                parent=parent,
            ),
        )
        newly_terminal.append(action.action_id)
    return authority, tuple(newly_terminal)


def _complete_parent_hierarchy_observation(observation: object) -> bool:
    """Accept absence proof only from one clean, complete accounting parser result."""
    from bspp.orchestration.control.monitoring import SlurmObservation

    if not isinstance(observation, SlurmObservation):
        return False
    snapshot = observation.sacct
    if snapshot.returncode != 0:
        return False
    if snapshot.parser in {"json", "fixture"}:
        return snapshot.warning is None
    if (
        snapshot.parser != "parsable-fallback"
        or snapshot.raw_text is None
        or not any(
            fmt in snapshot.argv
            for fmt in (
                "--format=JobIDRaw,JobID,State,ExitCode,Restarts",
                # Slurm builds without the Restarts field fall back to this form.
                "--format=JobIDRaw,JobID,State,ExitCode",
            )
        )
    ):
        return False
    from bspp.orchestration.control.monitoring import parse_sacct_identity_parsable_rows

    try:
        reparsed = parse_sacct_identity_parsable_rows(
            snapshot.raw_text,
            requested=observation.requested_job_ids,
        )
    except ValueError:
        return False
    scoped = tuple(item for item in reparsed if item.requested_job_id is not None)
    return tuple(
        (item.job_id, item.requested_job_id, item.state, item.exit_code, item.restarts) for item in scoped
    ) == tuple(
        (item.job_id, item.requested_job_id, item.state, item.exit_code, item.restarts)
        for item in observation.sacct_jobs
    )


def _current_task_statuses(
    action: PostprocessingRuntimeAction,
    parent_job_id: str | None,
    records: tuple[object, ...],
    terminal: PostprocessingTerminalObservation | None = None,
) -> list[dict[str, object]]:
    from bspp.orchestration.control.monitoring import SlurmJobRecord

    expected: tuple[int | None, ...] = action.expected_task_indexes or (None,)
    result: list[dict[str, object]] = []
    if isinstance(terminal, PostprocessingArrayParentCancelledObservedPayload):
        return [
            {
                "task_index": task_index,
                "scheduler_job_id": f"{terminal.parent_job_id}_{task_index}",
                "observation_status": "not-instantiated",
                "state": None,
                "exit_code": None,
                "source": None,
                "restarts": None,
            }
            for task_index in terminal.expected_task_indexes
        ]
    typed = tuple(item for item in records if isinstance(item, SlurmJobRecord) and item.source == "sacct")
    for task_index in expected:
        expected_job_id = (
            None if parent_job_id is None else parent_job_id if task_index is None else f"{parent_job_id}_{task_index}"
        )
        matches = tuple(item for item in typed if item.job_id == expected_job_id)
        observed = matches[0] if len(matches) == 1 else None
        result.append(
            {
                "task_index": task_index,
                "scheduler_job_id": expected_job_id,
                "observation_status": (
                    "not-applicable" if parent_job_id is None else "observed" if observed is not None else "missing"
                ),
                "state": observed.state if observed is not None else None,
                "exit_code": observed.exit_code if observed is not None else None,
                "source": observed.source if observed is not None else None,
                "restarts": observed.restarts if observed is not None else None,
            }
        )
    return result


def _task_succeeded(task: PostprocessingTaskTerminalEvidence) -> bool:
    return task.state == "COMPLETED" and task.exit_code in {"0:0", "0"}


def _exact_success_restarts(record: object) -> bool:
    """Require exact non-negative integer restarts accounting on a success record."""
    from bspp.orchestration.control.monitoring import SlurmJobRecord

    if not isinstance(record, SlurmJobRecord):
        return False
    restarts = record.restarts
    return isinstance(restarts, int) and not isinstance(restarts, bool) and restarts >= 0


def _terminal_views(
    authority: ReadablePostprocessingAuthority,
) -> dict[str, PostprocessingTerminalObservation]:
    return {payload.action_id: payload for payload in authority.terminal_payloads}


def _cancel_requested_jobs(authority: PostprocessingAuthority) -> set[str]:
    return {
        event.payload.parent_job_id
        for event in authority.events
        if event.attempt_id == authority.attempt_id
        and isinstance(event.payload, PostprocessingJobCancellationRequestResultPayload)
        and event.payload.return_code == 0
    }


def _cancellation_request_history(
    authority: PostprocessingAuthority,
    *,
    action_id: str,
) -> list[
    tuple[
        PostprocessingJobCancellationRequestIntendedPayload,
        PostprocessingJobCancellationRequestResultPayload | None,
    ]
]:
    history: list[
        tuple[
            PostprocessingJobCancellationRequestIntendedPayload,
            PostprocessingJobCancellationRequestResultPayload | None,
        ]
    ] = []
    for event in authority.events:
        if event.attempt_id != authority.attempt_id:
            continue
        payload = event.payload
        if isinstance(payload, PostprocessingJobCancellationRequestIntendedPayload) and payload.action_id == action_id:
            history.append((payload, None))
        elif isinstance(payload, PostprocessingJobCancellationRequestResultPayload) and payload.action_id == action_id:
            if not history or history[-1][1] is not None:
                raise ValueError("postprocessing cancellation result lacks one incomplete intent")
            history[-1] = (history[-1][0], payload)
    return history


def _terminal_task_evidence_digest(
    authority: PostprocessingAuthority,
    terminal_actions: Mapping[str, PostprocessingTerminalObservation],
) -> str:
    return canonical_mapping_digest(
        {
            "schema_version": 1,
            "phase_runspec_digest": authority.runspec.digest,
            "terminal_actions": [
                terminal_actions[action.action_id].to_mapping()
                for action in authority.runspec.payload.actions
                if action.action_id in terminal_actions
            ],
        }
    )


def _cancellation_result(
    authority: PostprocessingAuthority,
    *,
    requested: tuple[str, ...],
    pending: tuple[str, ...],
) -> PostprocessingLifecycleResult:
    return PostprocessingLifecycleResult(
        operation="cancel",
        phase_run_id=authority.phase_run_id,
        attempt_id=authority.attempt_id,
        status=authority.status,
        details={
            "cancellation_requested_job_ids": list(requested),
            "pending_parent_job_ids": list(pending),
        },
    )


def _require_fresh_qualification(authority: PostprocessingAuthority, observed_at: datetime) -> None:
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("postprocessing submission clock must be timezone-aware")
    qualified_at = parse_postprocessing_timestamp(authority.runspec.payload.qualified_runtime.qualified_at)
    expires_at = parse_postprocessing_timestamp(authority.runspec.payload.qualified_runtime.expires_at)
    moment = observed_at.astimezone(UTC)
    if moment < qualified_at or moment >= expires_at:
        raise ValueError("postprocessing Runtime Qualification is not current at submission")


def parse_postprocessing_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid postprocessing timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("postprocessing timestamps must include an offset")
    return parsed.astimezone(UTC)
