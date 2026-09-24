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

"""Runtime-only scientific roots and atomic postprocessing finalization bundles.

Large scientific tar payloads never enter the bounded handoff.  Runtime streams
their regular-file members into content manifests, hashes ordinary small output
files, verifies all prior Runtime and acceptance evidence, and atomically
publishes the JSON-only Action 09 handoff.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

from bspp.orchestration.contract.postprocessing_action_contract import (
    PostprocessingRuntimeAction,
)
from bspp.orchestration.contract.postprocessing_phase_ids import POSTPROCESSING_ACTION_IDS
from bspp.orchestration.contract.postprocessing_runspec import (
    ExecutablePostprocessingPhaseRunSpec,
)
from bspp.orchestration.contract.postprocessing_runtime_evidence import (
    PostprocessingCompletedRuntimeActionEvidence,
    PostprocessingRuntimeActionEvidenceAggregate,
    PostprocessingRuntimeTaskEvidence,
    postprocessing_runtime_task_evidence_from_mapping,
)
from bspp.orchestration.runtime.postprocessing.finalization_io import (
    _absolute_directory,
    _canonical_bytes,
    _canonical_mapping,
    _load_runspec,
    _stable_file_bytes,
    _strict_tree,
    _timestamp,
    _write_create_once,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TAR_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".tar.zst",
    ".tzst",
)
_ACCEPTANCE_STEPS = (
    "acceptance-tar-payload-parity",
    "acceptance-semantic",
    "acceptance-verify-evidence",
)
_CAPTURE_DESTINATIONS = {step: f"acceptance/captures/{step}.json" for step in _ACCEPTANCE_STEPS}
_REPORT_DESTINATIONS = {
    "acceptance-tar-payload-parity": "acceptance/reports/tar-payload-parity-report.json",
    "acceptance-semantic": "acceptance/reports/semantic-acceptance-summary.json",
    "acceptance-verify-evidence": "acceptance/reports/acceptance-evidence-report.json",
}
_ACTION09_ID = POSTPROCESSING_ACTION_IDS["acceptance-adjudication"]


def record_successful_action_task(
    *,
    phase_runspec_path: Path,
    action_id: str,
    command_digest: str,
    scheduler_job_id: str,
    task_index: int | None = None,
    completed_at: str | None = None,
) -> PostprocessingRuntimeTaskEvidence:
    """Create once one exact Runtime task record for an included prior action."""
    runspec = _load_runspec(phase_runspec_path)
    action = _action(runspec, action_id)
    if action.action_id == _ACTION09_ID:
        raise ValueError("Action 09 must only use the non-success prepublication assembly witness")
    _validate_task_index(action, task_index)
    evidence_root = _absolute_directory(Path(runspec.payload.attempt_paths.evidence_dir), "evidence root", create=True)
    path = _runtime_task_evidence_path(evidence_root, action, task_index)
    if path.exists() or os.path.lexists(path):
        existing = postprocessing_runtime_task_evidence_from_mapping(
            _canonical_mapping(_stable_file_bytes(path), label=f"Runtime task evidence {path}")
        )
        _reconcile_runtime_task_evidence(
            existing,
            runspec=runspec,
            action=action,
            command_digest=command_digest,
            scheduler_job_id=scheduler_job_id,
            task_index=task_index,
        )
        return existing
    evidence = PostprocessingRuntimeTaskEvidence(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        action_graph_digest=runspec.payload.action_graph_digest,
        action_id=action.action_id,
        runtime_action_digest=action.digest,
        command_digest=command_digest,
        task_index=task_index,
        scheduler_job_id=scheduler_job_id,
        completed_at=completed_at or _timestamp(),
    )
    _write_create_once(path, _canonical_bytes(evidence.to_mapping()))
    return evidence


def validate_postprocessing_runtime_action_evidence_aggregate(
    aggregate: PostprocessingRuntimeActionEvidenceAggregate,
    runspec: ExecutablePostprocessingPhaseRunSpec,
    *,
    expected_command_digests: Mapping[str, str],
) -> None:
    """Cross-check aggregate action, task, and command closure against a RunSpec."""
    expected_actions = runspec.payload.actions
    observed_by_id = {item.action_id: item for item in aggregate.completed_actions}
    if (
        aggregate.phase_run_id,
        aggregate.attempt_id,
        aggregate.phase_runspec_digest,
        aggregate.action_graph_digest,
    ) != (runspec.phase_run_id, runspec.attempt_id, runspec.digest, runspec.payload.action_graph_digest):
        raise ValueError("postprocessing Runtime aggregate differs from the staged RunSpec identity")
    if set(expected_command_digests) != {item.action_id for item in expected_actions}:
        raise ValueError("expected Runtime command digests do not cover the exact frozen action graph")
    for action in expected_actions[:-1]:
        observed = observed_by_id.get(action.action_id)
        if (
            observed is None
            or observed.runtime_action_digest != action.digest
            or observed.expected_task_indexes != action.expected_task_indexes
            or tuple(item.task_index for item in observed.tasks) != (action.expected_task_indexes or (None,))
            or any(item.command_digest != expected_command_digests[action.action_id] for item in observed.tasks)
        ):
            raise ValueError(f"Runtime aggregate differs from frozen action/task/command: {action.action_id}")
    witness = aggregate.action09_prepublication_witness
    action09 = expected_actions[-1]
    if (
        action09.action_id != _ACTION09_ID
        or witness.runtime_action_digest != action09.digest
        or witness.command_digest != expected_command_digests[action09.action_id]
    ):
        raise ValueError("Runtime aggregate Action 09 witness differs from its frozen action/command")


def _load_prior_runtime_actions(
    runspec: ExecutablePostprocessingPhaseRunSpec,
    evidence_root: Path,
) -> tuple[PostprocessingCompletedRuntimeActionEvidence, ...]:
    root = evidence_root / "phase-actions/runtime"
    expected_paths: dict[str, tuple[PostprocessingRuntimeAction, int | None]] = {}
    for action in runspec.payload.actions[:-1]:
        for task_index in action.expected_task_indexes or (None,):
            path = _runtime_task_evidence_path(evidence_root, action, task_index)
            expected_paths[path.relative_to(root).as_posix()] = (action, task_index)
    observed_files, observed_directories = _strict_tree(root)
    expected_directories = {path.split("/", 1)[0] for path in expected_paths}
    if observed_files != set(expected_paths) or observed_directories != expected_directories:
        raise ValueError("prior Runtime action/task evidence set is missing, extra, or unsafe")
    completed: list[PostprocessingCompletedRuntimeActionEvidence] = []
    for action in runspec.payload.actions[:-1]:
        tasks: list[PostprocessingRuntimeTaskEvidence] = []
        for task_index in action.expected_task_indexes or (None,):
            path = _runtime_task_evidence_path(evidence_root, action, task_index)
            evidence = postprocessing_runtime_task_evidence_from_mapping(
                _canonical_mapping(_stable_file_bytes(path), label=f"Runtime task evidence {path}")
            )
            _reconcile_runtime_task_evidence(
                evidence,
                runspec=runspec,
                action=action,
                command_digest=evidence.command_digest,
                scheduler_job_id=evidence.scheduler_job_id,
                task_index=task_index,
            )
            tasks.append(evidence)
        completed.append(
            PostprocessingCompletedRuntimeActionEvidence(
                action_id=action.action_id,
                runtime_action_digest=action.digest,
                expected_task_indexes=action.expected_task_indexes,
                tasks=tuple(tasks),
            )
        )
    return tuple(completed)


def _runtime_task_evidence_path(
    evidence_root: Path,
    action: PostprocessingRuntimeAction,
    task_index: int | None,
) -> Path:
    task_name = "task.json" if task_index is None else f"task-{task_index:010d}.json"
    return evidence_root / "phase-actions/runtime" / action.action_id / task_name


def _validate_task_index(action: PostprocessingRuntimeAction, task_index: int | None) -> None:
    expected = action.expected_task_indexes
    if (not expected and task_index is not None) or (expected and task_index not in expected):
        raise ValueError(f"Runtime task index differs from frozen action {action.action_id!r}")


def _reconcile_runtime_task_evidence(
    evidence: PostprocessingRuntimeTaskEvidence,
    *,
    runspec: ExecutablePostprocessingPhaseRunSpec,
    action: PostprocessingRuntimeAction,
    command_digest: str,
    scheduler_job_id: str,
    task_index: int | None,
) -> None:
    if (
        evidence.phase_run_id != runspec.phase_run_id
        or evidence.attempt_id != runspec.attempt_id
        or evidence.phase_runspec_digest != runspec.digest
        or evidence.action_graph_digest != runspec.payload.action_graph_digest
        or evidence.action_id != action.action_id
        or evidence.runtime_action_digest != action.digest
        or evidence.command_digest != command_digest
        or evidence.scheduler_job_id != scheduler_job_id
        or evidence.task_index != task_index
        or evidence.outcome != "succeeded"
    ):
        raise ValueError("Runtime task evidence differs from the frozen action/task identity")


def _action(runspec: ExecutablePostprocessingPhaseRunSpec, action_id: str) -> PostprocessingRuntimeAction:
    try:
        return next(item for item in runspec.payload.actions if item.action_id == action_id)
    except StopIteration as exc:
        raise ValueError(f"action is absent from the postprocessing RunSpec: {action_id!r}") from exc
