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

"""Exact sacct Restarts accounting on successful postprocessing task records."""

from __future__ import annotations

import json

import pytest

from bspp.orchestration.contract.postprocessing_handoff import (
    PostprocessingTaskReceiptEvidence,
)
from bspp.orchestration.contract.postprocessing_handoff import (
    _task as _handoff_task,
)
from bspp.orchestration.contract.postprocessing_scheduler_evidence import (
    _task as _scheduler_evidence_task,
)
from bspp.orchestration.contract.postprocessing_terminal_events import (
    PostprocessingTaskTerminalEvidence,
)
from bspp.orchestration.control.monitoring import (
    SlurmJobRecord,
    parse_sacct_identity_parsable_rows,
    parse_sacct_json,
    parse_sacct_parsable_rows,
)
from bspp.orchestration.control.postprocessing_phase_lifecycle import _complete_action_task_set


def _receipt_mapping(*, restarts: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_index": None,
        "scheduler_job_id": "1001",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "source": "sacct",
        "restarts": restarts,
    }


def test_receipt_rejects_negative_bool_and_allows_missing_restarts() -> None:
    for bad in (-1, True, False):
        with pytest.raises(ValueError, match="restarts"):
            PostprocessingTaskReceiptEvidence(
                scheduler_job_id="1001",
                state="COMPLETED",
                exit_code="0:0",
                source="sacct",
                restarts=bad,
            )
    assert (
        PostprocessingTaskReceiptEvidence(
            scheduler_job_id="1001",
            state="COMPLETED",
            exit_code="0:0",
            source="sacct",
        ).restarts
        is None
    )


def test_receipt_round_trips_non_negative_restarts_through_loaders() -> None:
    receipt = PostprocessingTaskReceiptEvidence(
        scheduler_job_id="1001",
        state="COMPLETED",
        exit_code="0:0",
        source="sacct",
        restarts=3,
    )
    mapping = receipt.to_mapping()

    assert mapping["restarts"] == 3
    assert _handoff_task(mapping).restarts == 3
    assert _scheduler_evidence_task(mapping).restarts == 3


def test_receipt_loaders_accept_missing_restarts_as_none_but_reject_malformed() -> None:
    missing = _receipt_mapping(restarts=3)
    del missing["restarts"]
    assert _handoff_task(missing).restarts is None
    assert _scheduler_evidence_task(missing).restarts is None
    for bad in (-1, True, "3"):
        payload = _receipt_mapping(restarts=bad)
        with pytest.raises(ValueError, match="restarts"):
            _handoff_task(payload)
        with pytest.raises(ValueError, match="restarts"):
            _scheduler_evidence_task(payload)


def test_terminal_accepts_absent_none_and_exact_restarts() -> None:
    base = {
        "scheduler_job_id": "1001",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "source": "sacct",
    }
    assert PostprocessingTaskTerminalEvidence(**base).restarts is None
    assert PostprocessingTaskTerminalEvidence(**base, restarts=None).restarts is None
    assert PostprocessingTaskTerminalEvidence(**base, restarts=2).restarts == 2


def test_terminal_rejects_present_negative_or_bool_restarts() -> None:
    for bad in (-1, True):
        with pytest.raises(ValueError, match="restarts"):
            PostprocessingTaskTerminalEvidence(
                scheduler_job_id="1001",
                state="COMPLETED",
                exit_code="0:0",
                source="sacct",
                restarts=bad,
            )


def test_parse_sacct_json_extracts_non_negative_restarts() -> None:
    records = parse_sacct_json(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id_raw": "1001",
                        "state": "COMPLETED",
                        "exit_code": "0:0",
                        "restarts": 3,
                    },
                    {
                        "job_id_raw": "1002",
                        "state": "COMPLETED",
                        "exit_code": "0:0",
                        "restarts": {"number": 2, "set": True},
                    },
                ]
            }
        ),
        requested=("1001", "1002"),
    )

    assert [record.restarts for record in records] == [3, 2]


def test_parse_sacct_json_leaves_malformed_restarts_unresolved() -> None:
    records = parse_sacct_json(
        json.dumps(
            {
                "jobs": [
                    {"job_id_raw": "1001", "state": "COMPLETED", "exit_code": "0:0", "restarts": -1},
                    {"job_id_raw": "1002", "state": "COMPLETED", "exit_code": "0:0", "restarts": "n/a"},
                    {"job_id_raw": "1003", "state": "COMPLETED", "exit_code": "0:0", "restarts": True},
                ]
            }
        ),
        requested=("1001", "1002", "1003"),
    )

    assert [record.restarts for record in records] == [None, None, None]


def test_parse_sacct_parsable_rows_carries_restarts_and_still_rejects_wrong_columns() -> None:
    records = parse_sacct_parsable_rows("1001|COMPLETED|0:0|3|\n", requested=("1001",))

    assert [(record.job_id, record.restarts) for record in records] == [("1001", 3)]
    with pytest.raises(ValueError, match="malformed sacct parsable row"):
        parse_sacct_parsable_rows("1001|COMPLETED|\n", requested=("1001",))


def test_parse_sacct_identity_parsable_rows_carries_restarts() -> None:
    records = parse_sacct_identity_parsable_rows("1001|1001_853|COMPLETED|0:0|3|\n", requested=("1001",))

    assert [(record.job_id, record.restarts) for record in records] == [("1001_853", 3)]
    with pytest.raises(ValueError, match="malformed sacct identity parsable row"):
        parse_sacct_identity_parsable_rows("1001|1001_853|COMPLETED|\n", requested=("1001",))


def _action(*, indexes: tuple[int, ...]) -> object:
    return type("ArrayAction", (), {"expected_task_indexes": indexes})()


def _record(job_id: str, *, restarts: object) -> SlurmJobRecord:
    return SlurmJobRecord(
        job_id=job_id,
        source="sacct",
        requested_job_id=job_id.split("_", 1)[0],
        state="COMPLETED",
        exit_code="0:0",
        restarts=restarts,
    )


def test_complete_action_task_set_requires_exact_restarts_on_success() -> None:
    action = _action(indexes=(853,))

    missing = _complete_action_task_set(action, "1001", (_record("1001_853", restarts=None),), autorequeue_enabled=True)
    assert missing is None

    malformed = _complete_action_task_set(action, "1001", (_record("1001_853", restarts=-1),), autorequeue_enabled=True)
    assert malformed is None

    exact = _complete_action_task_set(action, "1001", (_record("1001_853", restarts=4),), autorequeue_enabled=True)
    assert exact is not None
    assert [item.restarts for item in exact] == [4]
    assert [item.state for item in exact] == ["COMPLETED"]
    assert [item.exit_code for item in exact] == ["0:0"]


def test_complete_action_task_set_preserves_exact_zero_success_invariant() -> None:
    action = _action(indexes=())

    completed_zero = SlurmJobRecord(
        job_id="1001",
        source="sacct",
        requested_job_id="1001",
        state="COMPLETED",
        exit_code="0",
        restarts=0,
    )
    failed = SlurmJobRecord(
        job_id="1001",
        source="sacct",
        requested_job_id="1001",
        state="FAILED",
        exit_code="1:0",
        restarts=None,
    )

    assert _complete_action_task_set(action, "1001", (completed_zero,), autorequeue_enabled=True) is not None
    assert _complete_action_task_set(action, "1001", (failed,), autorequeue_enabled=True) is not None
