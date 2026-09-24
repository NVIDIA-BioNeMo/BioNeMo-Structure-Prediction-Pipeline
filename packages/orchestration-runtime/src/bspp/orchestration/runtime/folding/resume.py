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

"""Pure folding checkpoint failure classification and remaining-work planning.

Port Baseline:
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/python/checkpoint_manager.py:488-592``
and
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/scripts/preprocess_trt_bionemo.py:269-354``.
No checkpoint discovery, result scan, scheduler operation, or scientific
execution occurs in this module; callers provide all planned identities and
canonical checkpoint evidence explicitly.
"""

from __future__ import annotations

from bspp.orchestration.contract.folding_checkpoint import (
    FailureDisposition,
    FoldingCheckpointState,
    FoldingRemainingWorkPlan,
)
from bspp.orchestration.runtime.folding.checkpoint import normalize_checkpoint_protein_id


def classify_failure(error_message: str) -> FailureDisposition:
    """Return the baseline failure class, including its transient precedence."""
    if not isinstance(error_message, str):
        msg = "error_message must be a string"
        raise ValueError(msg)
    normalized = error_message.upper()
    if "TIMEOUT" in normalized or "INCOMPLETE" in normalized:
        return "transient"
    if (
        "OOM" in normalized
        or "OUT_OF_MEMORY" in normalized
        or "CUDA_OUT_OF_MEMORY" in normalized
        or "EXIT_CODE" in normalized
    ):
        return "permanent"
    return "permanent"


def plan_checkpoint_resume(
    planned_model_ids: tuple[str, ...],
    *,
    checkpoint_state: FoldingCheckpointState,
    retry_failed: bool = False,
) -> FoldingRemainingWorkPlan:
    """Derive remaining identities without discovering files or results.

    Completion is always excluded. Transient failure is always retried. A
    permanent failure is excluded by default and re-included only when
    ``retry_failed`` is true, matching the pinned preprocessing filter.
    Checkpoint identities absent from the declared plan are reported rather
    than silently affecting remaining work.
    """
    if not isinstance(planned_model_ids, tuple):
        msg = "planned_model_ids must be an explicit immutable tuple"
        raise ValueError(msg)
    canonical_planned_ids = tuple(normalize_checkpoint_protein_id(protein_id) for protein_id in planned_model_ids)
    if len(canonical_planned_ids) != len(set(canonical_planned_ids)):
        msg = "duplicate planned model identity after baseline normalization"
        raise ValueError(msg)
    if not isinstance(checkpoint_state, FoldingCheckpointState):
        msg = "checkpoint_state must be a FoldingCheckpointState"
        raise ValueError(msg)
    if not isinstance(retry_failed, bool):
        msg = "retry_failed must be a boolean"
        raise ValueError(msg)

    planned_set = set(canonical_planned_ids)
    completed_set = {normalize_checkpoint_protein_id(record.protein_id) for record in checkpoint_state.completions}
    transient_set = {
        normalize_checkpoint_protein_id(record.protein_id)
        for record in checkpoint_state.failures
        if classify_failure(record.error_message) == "transient"
    }
    permanent_set = {
        normalize_checkpoint_protein_id(record.protein_id) for record in checkpoint_state.failures
    } - transient_set

    completed_model_ids = tuple(
        protein_id
        for protein_id, canonical_id in zip(planned_model_ids, canonical_planned_ids, strict=True)
        if canonical_id in completed_set
    )
    transient_failure_ids = tuple(
        protein_id
        for protein_id, canonical_id in zip(planned_model_ids, canonical_planned_ids, strict=True)
        if canonical_id in transient_set
    )
    permanent_failure_ids = tuple(
        protein_id
        for protein_id, canonical_id in zip(planned_model_ids, canonical_planned_ids, strict=True)
        if canonical_id in permanent_set
    )
    remaining_model_ids = tuple(
        protein_id
        for protein_id, canonical_id in zip(planned_model_ids, canonical_planned_ids, strict=True)
        if canonical_id not in completed_set and (retry_failed or canonical_id not in permanent_set)
    )
    checkpoint_ids = completed_set | transient_set | permanent_set
    unplanned_checkpoint_ids = tuple(sorted(checkpoint_ids - planned_set))
    return FoldingRemainingWorkPlan(
        planned_model_ids=planned_model_ids,
        completed_model_ids=completed_model_ids,
        transient_failure_ids=transient_failure_ids,
        permanent_failure_ids=permanent_failure_ids,
        remaining_model_ids=remaining_model_ids,
        unplanned_checkpoint_ids=unplanned_checkpoint_ids,
        retry_failed=retry_failed,
    )


__all__ = ["classify_failure", "plan_checkpoint_resume"]
