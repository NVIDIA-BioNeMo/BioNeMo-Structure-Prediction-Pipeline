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

"""Independent scheduler/result reconciliation for release acceptance."""

from __future__ import annotations

from collections.abc import Mapping

from bspp.orchestration.contract.release_acceptance import TerminalFailureReport

SUCCESS_STATE = "COMPLETED"


def validate_terminal_results(
    *,
    expected_candidates: tuple[str, ...],
    scheduler_states: Mapping[str, str],
    result_candidates: tuple[str, ...],
) -> TerminalFailureReport:
    """Require scheduler completion and a retained immutable result for every candidate."""
    expected = _canonical_ids(expected_candidates, "expected_candidates")
    results = set(_canonical_ids(result_candidates, "result_candidates"))
    expected_set = set(expected)
    unexpected_scheduler = set(scheduler_states) - expected_set
    unexpected_results = results - expected_set
    if unexpected_scheduler:
        raise ValueError(f"unexpected scheduler candidates: {sorted(unexpected_scheduler)}")
    if unexpected_results:
        raise ValueError(f"unexpected result candidates: {sorted(unexpected_results)}")
    completed: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
    for candidate_id in expected:
        state = scheduler_states.get(candidate_id)
        if state == SUCCESS_STATE and candidate_id in results:
            completed.append(candidate_id)
        elif state == SUCCESS_STATE or state is None:
            missing.append(candidate_id)
        else:
            failed.append(candidate_id)
    return TerminalFailureReport(1, len(expected), tuple(completed), tuple(failed), tuple(missing))


def _canonical_ids(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if values != tuple(sorted(set(values), key=str.encode)):
        raise ValueError(f"{name} must be canonical sorted unique")
    return values


__all__ = ["validate_terminal_results"]
