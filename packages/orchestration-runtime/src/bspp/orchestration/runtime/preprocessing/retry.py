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

"""Pure preprocessing retry/top-up planning.

Port Baseline:
419813dbb5a3949e5e16f289f974d9f95e94bf01:utils/get_remaining.sh:4-10 and
419813dbb5a3949e5e16f289f974d9f95e94bf01:scripts/examine_results_and_replace_input.py:46-184.
"""

from __future__ import annotations

from bspp.orchestration.contract.preprocessing import PreprocessingWorkPlan
from bspp.orchestration.contract.preprocessing_state import (
    PreprocessingChunkState,
    PreprocessingRetryPlan,
    summarize_preprocessing_tranche_progress,
)


def plan_preprocessing_retries(
    *,
    work_plan: PreprocessingWorkPlan,
    states: tuple[PreprocessingChunkState, ...],
) -> PreprocessingRetryPlan:
    """Select exactly eligible remaining chunks without executing retry actions."""
    expected_chunk_refs = tuple((chunk.name, chunk.tranche_name, chunk.ordinal) for chunk in work_plan.chunks)
    state_chunk_refs = tuple((state.chunk_name, state.tranche_name, state.chunk_ordinal) for state in states)
    if state_chunk_refs != expected_chunk_refs:
        msg = "states must cover the work-plan chunks exactly once in deterministic order"
        raise ValueError(msg)

    eligible_names = tuple(state.chunk_name for state in states if state.eligible_for_retry)
    eligible_name_set = set(eligible_names)
    eligible_chunks = tuple(chunk for chunk in work_plan.chunks if chunk.name in eligible_name_set)
    eligible_assignments = tuple(
        assignment for assignment in work_plan.assignments if assignment.chunk_name in eligible_name_set
    )
    invalid_names = tuple(state.chunk_name for state in states if state.state == "invalid")
    actions = tuple(action for state in states for action in state.retry_actions)

    return PreprocessingRetryPlan(
        work_plan=work_plan,
        states=states,
        eligible_chunk_names=eligible_names,
        eligible_chunks=eligible_chunks,
        eligible_assignments=eligible_assignments,
        invalid_chunk_names=invalid_names,
        actions=actions,
        tranche_progress=summarize_preprocessing_tranche_progress(states),
    )


__all__ = ["plan_preprocessing_retries"]
