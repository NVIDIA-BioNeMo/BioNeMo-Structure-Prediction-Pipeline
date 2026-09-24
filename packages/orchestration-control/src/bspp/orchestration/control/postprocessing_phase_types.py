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

"""Shared immutable views for the postprocessing Phase lifecycle."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    PostprocessingAcceptancePolicySnapshot,
)
from bspp.orchestration.contract.postprocessing_event import (
    PostprocessingPhaseEvent,
)
from bspp.orchestration.contract.postprocessing_plan import (
    PostprocessingPhasePlan,
)
from bspp.orchestration.contract.postprocessing_runspec import (
    ExecutablePostprocessingPhaseRunSpec,
    ReadablePostprocessingPhaseRunSpec,
)
from bspp.orchestration.contract.postprocessing_runspec_v1 import (
    HistoricalPostprocessingPhaseRunSpecV1,
)
from bspp.orchestration.contract.postprocessing_submission_events import (
    PostprocessingSubmissionActionPlan,
)
from bspp.orchestration.contract.postprocessing_terminal_events import (
    PostprocessingTerminalObservation,
)
from bspp.orchestration.contract.runspec import RunSpec

PostprocessingLifecycleStatus = Literal[
    "materialized",
    "submitted",
    "failed",
    "cancelling",
    "cancelled",
    "accepted",
]


@dataclass(frozen=True)
class PostprocessingInitialAttemptSnapshot:
    attempt_id: str
    ordinal: int
    phase_runspec_location: str
    phase_runspec_digest: str
    created_at: str
    status: Literal["materialized"] = "materialized"
    schema_version: int = 1


@dataclass(frozen=True)
class PostprocessingPhaseRunSnapshot:
    phase_run_id: str
    phase_plan_location: str
    phase_plan_digest: str
    created_at: str
    current_attempt_id: str
    initial_attempt: PostprocessingInitialAttemptSnapshot
    status: Literal["materialized"] = "materialized"
    sealed: Literal[False] = False
    phase_kind: Literal["postprocessing"] = "postprocessing"
    schema_version: int = 1


@dataclass(frozen=True)
class PostprocessingSubmissionState:
    submission_id: str
    intended_at: str
    actions: tuple[PostprocessingSubmissionActionPlan, ...]
    job_ids: tuple[tuple[str, str], ...]
    dispatching: frozenset[str]
    rejected: frozenset[str]

    def job_id_for(self, action_id: str) -> str | None:
        return dict(self.job_ids).get(action_id)

    def job_ids_by_action(self) -> dict[str, str]:
        return dict(self.job_ids)


@dataclass(frozen=True)
class PostprocessingReplayState:
    status: PostprocessingLifecycleStatus
    sealed: bool
    runspec: ReadablePostprocessingPhaseRunSpec
    embedded_legacy_runspec_bytes: bytes | None
    embedded_runtime_qualification_bytes: bytes | None
    submission_state: PostprocessingSubmissionState | None
    terminal_payloads: tuple[PostprocessingTerminalObservation, ...]


@dataclass(frozen=True)
class PostprocessingPhaseMaterializationResult:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    authority_root: Path

    def to_mapping(self) -> dict[str, object]:
        return {
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "authority_root": str(self.authority_root),
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class PostprocessingAuthority:
    authority_path: Path
    phase_plan: PostprocessingPhasePlan
    phase_run: PostprocessingPhaseRunSnapshot
    runspec: ExecutablePostprocessingPhaseRunSpec
    legacy_runspec: RunSpec
    acceptance_policy: PostprocessingAcceptancePolicySnapshot
    legacy_runspec_bytes: bytes
    acceptance_policy_bytes: bytes
    runtime_qualification_bytes: bytes
    events: tuple[PostprocessingPhaseEvent, ...]
    status: PostprocessingLifecycleStatus
    sealed: bool
    current_attempt_projection_complete: bool
    submission_state: PostprocessingSubmissionState | None
    terminal_payloads: tuple[PostprocessingTerminalObservation, ...]

    @property
    def attempt_id(self) -> str:
        return self.runspec.attempt_id

    @property
    def phase_run_id(self) -> str:
        return self.runspec.phase_run_id


@dataclass(frozen=True)
class HistoricalPostprocessingAuthorityV1:
    """Validated read-only authority for a persisted V1 postprocessing run."""

    authority_path: Path
    phase_plan: PostprocessingPhasePlan
    phase_run: PostprocessingPhaseRunSnapshot
    runspec: HistoricalPostprocessingPhaseRunSpecV1
    legacy_runspec: RunSpec
    acceptance_policy: PostprocessingAcceptancePolicySnapshot
    legacy_runspec_bytes: bytes
    acceptance_policy_bytes: bytes
    runtime_qualification_bytes: bytes
    events: tuple[PostprocessingPhaseEvent, ...]
    status: PostprocessingLifecycleStatus
    sealed: bool
    current_attempt_projection_complete: bool
    submission_state: PostprocessingSubmissionState | None
    terminal_payloads: tuple[PostprocessingTerminalObservation, ...]

    @property
    def attempt_id(self) -> str:
        return self.runspec.attempt_id

    @property
    def phase_run_id(self) -> str:
        return self.runspec.phase_run_id


ReadablePostprocessingAuthority = PostprocessingAuthority | HistoricalPostprocessingAuthorityV1


@dataclass(frozen=True)
class PostprocessingLifecycleResult:
    """Stable JSON projection shared by additive postprocessing operations."""

    operation: str
    phase_run_id: str
    attempt_id: str
    status: str
    details: Mapping[str, object]

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase_kind": "postprocessing",
            "operation": self.operation,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "status": self.status,
            **self.details,
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"

    def render_table(self) -> str:
        actions = self.details.get("actions")
        lines = [
            f"phase_run_id: {self.phase_run_id}",
            f"attempt_id: {self.attempt_id}",
            f"status: {self.status}",
        ]
        if isinstance(actions, list):
            lines.append("actions:")
            for action in actions:
                if isinstance(action, Mapping):
                    if self.operation != "status":
                        lines.append(
                            "  "
                            + " ".join(
                                f"{key}={action[key]}"
                                for key in ("action_id", "durable_status", "job_id")
                                if key in action
                            )
                        )
                        continue
                    scheduler = action.get("scheduler")
                    scheduler_mapping = scheduler if isinstance(scheduler, Mapping) else None
                    lines.append(
                        "  "
                        + " ".join(
                            (
                                f"action_id={_table_value(action, 'action_id')}",
                                f"durable_status={_table_value(action, 'durable_status')}",
                                f"job_id={_table_value(action, 'job_id')}",
                                f"scheduler_status={_table_value(scheduler_mapping, 'status')}",
                                f"scheduler_state={_table_value(scheduler_mapping, 'state')}",
                                f"scheduler_source={_table_value(scheduler_mapping, 'source')}",
                                f"scheduler_exit_code={_table_value(scheduler_mapping, 'exit_code')}",
                            )
                        )
                    )
                    tasks = action.get("tasks")
                    if isinstance(tasks, list):
                        for task in tasks:
                            if isinstance(task, Mapping):
                                lines.append(
                                    "    "
                                    + " ".join(
                                        (
                                            f"task_index={_table_value(task, 'task_index')}",
                                            f"scheduler_job_id={_table_value(task, 'scheduler_job_id')}",
                                            f"observation_status={_table_value(task, 'observation_status')}",
                                            f"state={_table_value(task, 'state')}",
                                            f"source={_table_value(task, 'source')}",
                                            f"exit_code={_table_value(task, 'exit_code')}",
                                        )
                                    )
                                )
        warnings = self.details.get("warnings")
        if isinstance(warnings, list) and warnings:
            lines.append("warnings:")
            lines.extend(f"  - {warning}" for warning in warnings)
        return "\n".join(lines) + "\n"


def _table_value(mapping: Mapping[str, object] | None, key: str) -> str:
    """Render absent and null status fields distinctly in the human table."""
    if mapping is None or key not in mapping:
        return "missing"
    value = mapping[key]
    return "null" if value is None else str(value)


__all__ = [
    "HistoricalPostprocessingAuthorityV1",
    "PostprocessingAuthority",
    "PostprocessingInitialAttemptSnapshot",
    "PostprocessingLifecycleResult",
    "PostprocessingLifecycleStatus",
    "PostprocessingPhaseMaterializationResult",
    "PostprocessingPhaseRunSnapshot",
    "PostprocessingReplayState",
    "PostprocessingSubmissionState",
    "ReadablePostprocessingAuthority",
]
