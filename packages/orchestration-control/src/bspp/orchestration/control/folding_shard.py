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

"""Deterministic LPT fold shard planning.

Materializes one canonical ordered fold shard projection from the Plan's
enriched ``msa_set_manifest.member_lengths`` and the fold action's typed
topology (``worker_count = nodes * tasks_per_node``). The LPT semantics are
identical to the reference ``balanced_shards``/``assign_stage`` port: sort
targets by ``(-member_length, target_id)``, greedily assign each to the
currently least-loaded rank, and tie-break by ascending rank index.

This module is control-pure: it imports only ``contract.*`` plus stdlib.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from bspp.orchestration.contract.folding_shard import (
    FOLD_SHARD_PROJECTION_FILENAME,
    FoldShardProjection,
    FoldShardProjectionBinding,
    FoldShardRank,
    FoldShardTarget,
    fold_shard_projection_document_bytes,
)
from bspp.orchestration.contract.model_identity import normalize_model_entity_id
from bspp.orchestration.contract.phase import FoldingPhasePlan, FoldingPhaseRunSpec, PhaseSlurmResources

FOLD_SHARD_LPT_VERSION = 1

_LEGACY_MSA_SET_MANIFEST_ERROR = (
    "legacy MSA manifest requires Runtime length enrichment before folding; "
    "run the legacy-MSA import operation to publish an enriched manifest"
)


def fold_shard_projection_location(attempt_id: str) -> str:
    """Return the canonical attempt-scoped projection document location."""
    return f"attempts/{attempt_id}/{FOLD_SHARD_PROJECTION_FILENAME}"


def fold_shard_targets_from_plan(phase_plan: FoldingPhasePlan) -> tuple[tuple[str, int], ...]:
    """Derive ordered ``(target_id, member_length)`` targets from the Plan.

    Target identity is the normalized model entity ID of each consumed A3M
    member stem; the LPT weight is the producer-attested member length. The two
    sequences align positionally by member order.
    """
    manifest = phase_plan.payload.msa_set_manifest
    if manifest is None or not manifest.has_member_lengths():
        raise ValueError(_LEGACY_MSA_SET_MANIFEST_ERROR)
    paths = phase_plan.payload.msa_set.member_a3m_paths
    lengths = manifest.member_lengths
    if lengths is None:
        raise ValueError(_LEGACY_MSA_SET_MANIFEST_ERROR)
    if len(paths) != len(lengths):
        raise ValueError("folding MSA member paths and member lengths must align positionally")
    return tuple(
        (normalize_model_entity_id(Path(path).stem), length) for path, length in zip(paths, lengths, strict=True)
    )


def plan_fold_shards(
    targets: tuple[tuple[str, int], ...],
    worker_count: int,
    *,
    lpt_version: int = FOLD_SHARD_LPT_VERSION,
) -> FoldShardProjection:
    """Assign every target to exactly one rank using deterministic LPT."""
    loads = [0] * worker_count
    rank_targets: list[list[FoldShardTarget]] = [[] for _ in range(worker_count)]
    for target_id, member_length in sorted(targets, key=lambda target: (-target[1], target[0])):
        rank = min(range(worker_count), key=lambda index: (loads[index], index))
        rank_targets[rank].append(FoldShardTarget(target_id=target_id, member_length=member_length))
        loads[rank] += member_length
    ranks = tuple(FoldShardRank(global_rank=rank, targets=tuple(rank_targets[rank])) for rank in range(worker_count))
    return FoldShardProjection(worker_count=worker_count, lpt_version=lpt_version, ranks=ranks)


def derive_fold_shard_projection(
    phase_plan: FoldingPhasePlan,
    fold_resources: PhaseSlurmResources,
    attempt_id: str,
) -> tuple[FoldShardProjection, FoldShardProjectionBinding]:
    """Derive the canonical projection and its digest-bound location binding."""
    worker_count = fold_resources.workers
    projection = plan_fold_shards(fold_shard_targets_from_plan(phase_plan), worker_count)
    document = fold_shard_projection_document_bytes(projection)
    binding = FoldShardProjectionBinding(
        location=fold_shard_projection_location(attempt_id),
        sha256=hashlib.sha256(document).hexdigest(),
        size_bytes=len(document),
        worker_count=worker_count,
        lpt_version=FOLD_SHARD_LPT_VERSION,
    )
    return projection, binding


def fold_shard_projection_from_runspec(
    phase_plan: FoldingPhasePlan,
    phase_runspec: FoldingPhaseRunSpec,
) -> tuple[FoldShardProjection, FoldShardProjectionBinding]:
    """Re-derive the projection from a stored RunSpec's exact fold action."""
    fold_action = next(action for action in phase_runspec.payload.actions if action.action_kind == "fold")
    return derive_fold_shard_projection(phase_plan, fold_action.resources, phase_runspec.attempt_id)


__all__ = [
    "FOLD_SHARD_LPT_VERSION",
    "derive_fold_shard_projection",
    "fold_shard_projection_from_runspec",
    "fold_shard_projection_location",
    "fold_shard_targets_from_plan",
    "plan_fold_shards",
]
