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

"""Strict terminal accounting Contract tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from bspp.orchestration.contract.phase_reconciliation import (
    FoldingActionTerminalObservationView,
    FoldingActionTerminalObservedEvent,
    FoldingActionTerminalObservedPayload,
    FoldingTaskTerminalEvidence,
    PhaseActionTerminalObservationView,
    PhaseActionTerminalObservedEvent,
    PhaseActionTerminalObservedPayload,
    folding_action_terminal_observed_event_from_mapping,
    folding_task_terminal_evidence_from_mapping,
    phase_action_terminal_observed_event_from_mapping,
    phase_action_terminal_outcome,
)
from bspp.orchestration.contract.phase_state import PhaseRunLifecycleView


def _payload(**changes: object) -> PhaseActionTerminalObservedPayload:
    values: dict[str, object] = {
        "submission_id": "phase-submission-" + "1" * 64,
        "phase_runspec_digest": "2" * 64,
        "action_id": "preprocessing-chunk-000000",
        "runtime_action_digest": "3" * 64,
        "scheduler_correlation_token": "bspp-phase-" + "4" * 64,
        "job_id": "123456",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "outcome": "succeeded",
    }
    values.update(changes)
    return PhaseActionTerminalObservedPayload(**cast(Any, values))


def _event(**payload_changes: object) -> PhaseActionTerminalObservedEvent:
    return PhaseActionTerminalObservedEvent(
        sequence=5,
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        occurred_at="2026-08-20T12:00:00.000000Z",
        payload=_payload(**payload_changes),
    )


def test_terminal_event_strict_round_trip_and_view() -> None:
    event = _event()

    assert phase_action_terminal_observed_event_from_mapping(event.to_mapping()) == event
    assert PhaseActionTerminalObservationView.from_event(event).to_mapping() == {
        "action_id": "preprocessing-chunk-000000",
        "job_id": "123456",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "outcome": "succeeded",
        "source": "sacct",
        "observed_at": "2026-08-20T12:00:00.000000Z",
    }


@pytest.mark.parametrize(
    ("state", "exit_code", "expected"),
    [
        ("COMPLETED", "0:0", "succeeded"),
        ("COMPLETED", "1:0", "failed"),
        ("FAILED", "1:0", "failed"),
        ("TIMEOUT", None, "failed"),
        ("RUNNING", None, None),
        ("COMPLETED", None, None),
        ("COMPLETED", "invalid", None),
    ],
)
def test_terminal_outcome_is_closed_and_conclusive(
    state: str,
    exit_code: str | None,
    expected: str | None,
) -> None:
    assert phase_action_terminal_outcome(state, exit_code) == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "RUNNING", "exit_code": None, "outcome": "failed"},
        {"state": "COMPLETED", "exit_code": None, "outcome": "succeeded"},
        {"state": "COMPLETED", "exit_code": "0:0", "outcome": "failed"},
        {"state": "FAILED", "exit_code": "bad", "outcome": "failed"},
        {"source": "squeue"},
        {"job_id": "123.batch"},
        {"phase_runspec_digest": "A" * 64},
    ],
)
def test_terminal_payload_rejects_inconclusive_or_drifting_records(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _payload(**changes)


def test_terminal_loaders_reject_unknown_missing_and_discriminator_drift() -> None:
    mapping = _event().to_mapping()
    mapping["unknown"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        phase_action_terminal_observed_event_from_mapping(mapping)

    mapping = _event().to_mapping()
    del mapping["payload"]
    with pytest.raises(ValueError, match="fields mismatch"):
        phase_action_terminal_observed_event_from_mapping(mapping)

    mapping = _event().to_mapping()
    mapping["event_type"] = "phase-action-submitted"
    with pytest.raises(ValueError, match="event type"):
        phase_action_terminal_observed_event_from_mapping(mapping)


def test_phase_lifecycle_requires_exact_five_way_tuple() -> None:
    active = PhaseRunLifecycleView(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        current_attempt_id="attempt-0001",
        attempt_status="materialized",
        run_status="materialized",
        sealed=False,
    )
    failed = replace(active, attempt_status="failed", run_status="failed")
    assert failed.attempt_status == "failed"
    cancelling = replace(active, attempt_status="cancelling", run_status="cancelling")
    cancelled = replace(active, attempt_status="cancelled", run_status="cancelled")
    assert cancelling.run_status == "cancelling"
    assert cancelled.run_status == "cancelled"
    with pytest.raises(ValueError, match="exactly active, failed, cancelling, cancelled, or accepted"):
        replace(active, attempt_status="failed")
    with pytest.raises(ValueError, match="exactly active, failed, cancelling, cancelled, or accepted"):
        replace(active, run_status="failed")


def _fold_task(
    task_index: int,
    *,
    state: str = "COMPLETED",
    exit_code: str = "0:0",
    scheduler_job_id: str | None = None,
) -> FoldingTaskTerminalEvidence:
    return FoldingTaskTerminalEvidence(
        scheduler_job_id=scheduler_job_id or f"1004_{task_index}",
        state=state,
        exit_code=exit_code,
        source="sacct",
        task_index=task_index,
    )


def _fold_payload(**changes: object) -> FoldingActionTerminalObservedPayload:
    values: dict[str, object] = {
        "submission_id": "phase-submission-" + "1" * 64,
        "phase_runspec_digest": "2" * 64,
        "action_id": "fold-000001",
        "runtime_action_digest": "3" * 64,
        "parent_job_id": "1004",
        "expected_task_indexes": (0, 1, 2),
        "tasks": (_fold_task(0), _fold_task(1), _fold_task(2)),
        "outcome": "succeeded",
    }
    values.update(changes)
    return FoldingActionTerminalObservedPayload(**cast(Any, values))


def _fold_event(**payload_changes: object) -> FoldingActionTerminalObservedEvent:
    return FoldingActionTerminalObservedEvent(
        sequence=5,
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        occurred_at="2026-08-20T12:00:00.000000Z",
        payload=_fold_payload(**payload_changes),
    )


def test_folding_array_terminal_event_strict_round_trip_and_view() -> None:
    event = _fold_event()

    assert folding_action_terminal_observed_event_from_mapping(event.to_mapping()) == event
    view = FoldingActionTerminalObservationView.from_event(event)
    assert view.action_id == "fold-000001"
    assert view.parent_job_id == "1004"
    assert view.expected_task_indexes == (0, 1, 2)
    assert view.outcome == "succeeded"
    assert view.to_mapping()["expected_task_indexes"] == [0, 1, 2]


@pytest.mark.parametrize(
    "changes",
    [
        # missing task index on one array task
        {
            "tasks": (
                _fold_task(0),
                _fold_task(1),
                FoldingTaskTerminalEvidence(
                    scheduler_job_id="1004_2",
                    state="COMPLETED",
                    exit_code="0:0",
                    source="sacct",
                ),
            ),
        },
        # mismatched scheduler id (parent/index binding broken)
        {
            "tasks": (_fold_task(0, scheduler_job_id="1005_0"), _fold_task(1), _fold_task(2)),
        },
        # out-of-order expected indexes
        {"expected_task_indexes": (2, 0, 1)},
        # duplicate expected indexes
        {"expected_task_indexes": (0, 0, 1)},
        # negative expected index
        {"expected_task_indexes": (-1, 0, 1)},
        # non-COMPLETED task with a succeeded outcome
        {
            "tasks": (_fold_task(0, state="FAILED", exit_code="1:0"), _fold_task(1), _fold_task(2)),
        },
        # all-COMPLETED/0:0 set with a failed outcome
        {"outcome": "failed"},
    ],
)
def test_folding_array_terminal_payload_rejects_incoherent_records(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _fold_payload(**changes)


def test_folding_array_terminal_loaders_reject_unknown_missing_and_discriminator_drift() -> None:
    mapping = _fold_event().to_mapping()
    mapping["unknown"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        folding_action_terminal_observed_event_from_mapping(mapping)

    mapping = _fold_event().to_mapping()
    del mapping["payload"]
    with pytest.raises(ValueError, match="fields mismatch"):
        folding_action_terminal_observed_event_from_mapping(mapping)

    mapping = _fold_event().to_mapping()
    mapping["event_type"] = "phase-action-terminal-observed"
    with pytest.raises(ValueError, match="event type"):
        folding_action_terminal_observed_event_from_mapping(mapping)


def test_folding_task_evidence_loader_rejects_missing_or_extra_fields() -> None:
    task = _fold_task(0)
    mapping = task.to_mapping()
    mapping["extra"] = True
    with pytest.raises(ValueError, match="missing or extra fields"):
        folding_task_terminal_evidence_from_mapping(mapping)

    mapping = task.to_mapping()
    del mapping["exit_code"]
    with pytest.raises(ValueError, match="missing or extra fields"):
        folding_task_terminal_evidence_from_mapping(mapping)
