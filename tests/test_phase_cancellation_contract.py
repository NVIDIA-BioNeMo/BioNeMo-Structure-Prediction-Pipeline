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

"""Strict cancellation contract tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bspp.orchestration.contract.phase_cancellation import (
    PhaseCancellationActionTarget,
    PhaseCancellationActionView,
    PhaseCancellationCompletedEvent,
    PhaseCancellationCompletedPayload,
    PhaseCancellationIntendedEvent,
    PhaseCancellationIntendedPayload,
    PhaseCancellationLifecycleView,
    PhaseCancellationTerminalReference,
    PhaseJobCancellationRequestIntendedEvent,
    PhaseJobCancellationRequestIntendedPayload,
    PhaseJobCancellationRequestResultEvent,
    PhaseJobCancellationRequestResultPayload,
    phase_cancellation_completed_event_from_mapping,
    phase_cancellation_id,
    phase_cancellation_intended_event_from_mapping,
    phase_job_cancellation_request_intended_event_from_mapping,
    phase_job_cancellation_request_result_event_from_mapping,
)
from bspp.orchestration.contract.phase_submission import phase_action_scheduler_correlation_token

RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
ATTEMPT_ID = "attempt-0001"
SUBMISSION_ID = "phase-submission-" + "1" * 64
RUNSPEC_DIGEST = "2" * 64
CANCELLATION_TIME = "2026-08-20T13:00:00.000000Z"
ACTION_ID = "preprocessing-chunk-000000"
TOKEN = phase_action_scheduler_correlation_token(SUBMISSION_ID, ACTION_ID)


def _intent() -> PhaseCancellationIntendedEvent:
    targets = (
        PhaseCancellationActionTarget(
            action_id=ACTION_ID,
            initial_submission_status="submitted",
            scheduler_correlation_token=TOKEN,
            job_id="123456",
        ),
    )
    cancellation_id = phase_cancellation_id(
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        submission_id=SUBMISSION_ID,
        phase_runspec_digest=RUNSPEC_DIGEST,
        targets=targets,
    )
    return PhaseCancellationIntendedEvent(
        sequence=2,
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        occurred_at=CANCELLATION_TIME,
        payload=PhaseCancellationIntendedPayload(
            cancellation_id=cancellation_id,
            submission_id=SUBMISSION_ID,
            phase_runspec_digest=RUNSPEC_DIGEST,
            targets=targets,
        ),
    )


def test_cancellation_events_round_trip_exactly() -> None:
    intended = _intent()
    request = PhaseJobCancellationRequestIntendedEvent(
        sequence=3,
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        occurred_at=CANCELLATION_TIME,
        payload=PhaseJobCancellationRequestIntendedPayload(
            cancellation_id=intended.payload.cancellation_id,
            action_id=ACTION_ID,
            job_id="123456",
            request_ordinal=1,
            scancel_argv=("scancel", "123456"),
        ),
    )
    result = PhaseJobCancellationRequestResultEvent(
        sequence=4,
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        occurred_at=CANCELLATION_TIME,
        payload=PhaseJobCancellationRequestResultPayload(
            cancellation_id=intended.payload.cancellation_id,
            action_id=ACTION_ID,
            job_id="123456",
            request_ordinal=1,
            scancel_argv=("scancel", "123456"),
            return_code=0,
            stdout="",
            stderr="",
        ),
    )
    completed = PhaseCancellationCompletedEvent(
        sequence=6,
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        occurred_at=CANCELLATION_TIME,
        payload=PhaseCancellationCompletedPayload(
            cancellation_id=intended.payload.cancellation_id,
            terminal_references=(
                PhaseCancellationTerminalReference(
                    action_id=ACTION_ID,
                    job_id="123456",
                    terminal_observation_digest="4" * 64,
                ),
            ),
        ),
    )

    assert phase_cancellation_intended_event_from_mapping(intended.to_mapping()) == intended
    assert phase_job_cancellation_request_intended_event_from_mapping(request.to_mapping()) == request
    assert phase_job_cancellation_request_result_event_from_mapping(result.to_mapping()) == result
    assert phase_cancellation_completed_event_from_mapping(completed.to_mapping()) == completed


def test_cancellation_contract_rejects_unknown_fields_and_noncanonical_target_shapes() -> None:
    intended = _intent()
    malformed = intended.to_mapping()
    malformed["unknown"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        phase_cancellation_intended_event_from_mapping(malformed)

    with pytest.raises(ValueError, match="only a submitted cancellation target may have a job id"):
        replace(intended.payload.targets[0], initial_submission_status="planned")
    with pytest.raises(ValueError, match="request ordinal"):
        PhaseJobCancellationRequestIntendedPayload(
            cancellation_id=intended.payload.cancellation_id,
            action_id=ACTION_ID,
            job_id="123456",
            request_ordinal=0,
            scancel_argv=("scancel", "123456"),
        )


def test_cancellation_id_changes_with_frozen_target_state() -> None:
    intended = _intent()
    dispatching = replace(
        intended.payload.targets[0],
        initial_submission_status="dispatching",
        job_id=None,
    )
    changed = phase_cancellation_id(
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        submission_id=SUBMISSION_ID,
        phase_runspec_digest=RUNSPEC_DIGEST,
        targets=(dispatching,),
    )
    assert changed != intended.payload.cancellation_id


def test_synthetic_multi_action_lifecycle_preserves_target_order_and_independent_dispositions() -> None:
    intended = _intent()
    planned = PhaseCancellationActionTarget(
        action_id="preprocessing-chunk-000001",
        initial_submission_status="planned",
        scheduler_correlation_token=phase_action_scheduler_correlation_token(
            SUBMISSION_ID, "preprocessing-chunk-000001"
        ),
    )
    lifecycle_id = phase_cancellation_id(
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        submission_id=SUBMISSION_ID,
        phase_runspec_digest=RUNSPEC_DIGEST,
        targets=(intended.payload.targets[0], planned),
    )
    lifecycle = PhaseCancellationLifecycleView(
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        cancellation_id=lifecycle_id,
        submission_id=SUBMISSION_ID,
        phase_runspec_digest=RUNSPEC_DIGEST,
        actions=(
            PhaseCancellationActionView(
                target=intended.payload.targets[0],
                disposition="cancel-pending",
                bound_job_id="123456",
            ),
            PhaseCancellationActionView(target=planned, disposition="no-job"),
        ),
        status="cancelling",
    )

    assert [item.action_id for item in lifecycle.actions] == [ACTION_ID, "preprocessing-chunk-000001"]
    assert [item.disposition for item in lifecycle.actions] == ["cancel-pending", "no-job"]

    duplicate_job = replace(
        intended.payload.targets[0],
        action_id="preprocessing-chunk-000001",
    )
    with pytest.raises(ValueError, match="target job ids must be unique"):
        PhaseCancellationIntendedPayload(
            cancellation_id=intended.payload.cancellation_id,
            submission_id=SUBMISSION_ID,
            phase_runspec_digest=RUNSPEC_DIGEST,
            targets=(intended.payload.targets[0], duplicate_job),
        )
