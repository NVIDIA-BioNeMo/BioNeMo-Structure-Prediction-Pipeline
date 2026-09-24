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

"""Pure CSV and preprocessing-manifest rendering for folding queues.

Port Baseline sources:
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/scripts/preprocess_trt_bionemo.py:491-629``
and
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/python/batch_csv_manager.py:27-179``.
The port returns immutable values and never writes a CSV or JSON file.
"""

from __future__ import annotations

import csv
import io
import json
import math

from bspp.orchestration.contract.folding_queue import (
    FoldingQueueArtifact,
    FoldingQueueAssignment,
    FoldingQueuePlan,
    FoldingQueueRenderResult,
)

_CSV_COLUMNS = ("batch_id", "seq_length", "msa_path", "protein_id")


def render_folding_queues(plan: FoldingQueuePlan) -> FoldingQueueRenderResult:
    """Render one byte-stable CSV string for each nonempty worker."""
    artifacts: list[FoldingQueueArtifact] = []
    prefix = "node" if plan.config.layout == "per_node" else "gpu"
    assignments_by_worker = _assignments_by_worker(plan)
    for worker_id, assignments in assignments_by_worker.items():
        artifacts.append(
            FoldingQueueArtifact(
                worker_id=worker_id,
                file_name=f"{prefix}{worker_id}_batches.csv",
                content=_render_csv(assignments),
                total_proteins=len(assignments),
                num_batches=len({assignment.batch_id for assignment in assignments}),
            )
        )
    return FoldingQueueRenderResult(layout=plan.config.layout, artifacts=tuple(artifacts))


def render_folding_queue_manifest(
    plan: FoldingQueuePlan,
    *,
    created: str,
    batch_info_path: str,
    output_dir: str,
    num_gpus: int,
    num_nodes: int | None,
    target_runtime_hours: float,
    max_total_residues: int,
    force_rerun: bool,
    rebalance: bool,
    skip_completed: bool,
) -> str:
    """Return the pinned manifest shape with all volatile/resource facts explicit."""
    _validate_manifest_inputs(
        plan,
        created=created,
        batch_info_path=batch_info_path,
        output_dir=output_dir,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        target_runtime_hours=target_runtime_hours,
        max_total_residues=max_total_residues,
    )
    lengths = sorted(assignment.sequence_length for assignment in plan.assignments)
    length_stats: dict[str, int | float]
    if lengths:
        length_range = (lengths[0], lengths[-1])
        length_stats = {
            "p10": int(_linear_quantile(lengths, 0.10)),
            "p25": int(_linear_quantile(lengths, 0.25)),
            "p50": int(_linear_quantile(lengths, 0.50)),
            "p75": int(_linear_quantile(lengths, 0.75)),
            "p90": int(_linear_quantile(lengths, 0.90)),
            "mean": sum(lengths) / len(lengths),
        }
    else:
        length_range = (0, 0)
        length_stats = {}

    assignment_key = "node_assignments" if plan.config.layout == "per_node" else "gpu_assignments"
    assignment_summaries: dict[str, dict[str, int]] = {}
    for worker_id, assignments in _assignments_by_worker(plan).items():
        assignment_summaries[str(worker_id)] = {
            "total_proteins": len(assignments),
            "num_batches": len({assignment.batch_id for assignment in assignments}),
        }

    manifest: dict[str, object] = {
        "version": "2.0.0",
        "created": created,
        "pipeline": "trt-bionemo",
        "adapted_from": "colabfold-slurm",
        "batch_info_path": batch_info_path,
        "output_dir": output_dir,
        "total_proteins": len(plan.assignments),
        "num_gpus": num_gpus,
        "execution_mode": "per_node" if plan.config.layout == "per_node" else "per_gpu",
        "num_nodes": num_nodes if plan.config.layout == "per_node" else None,
        "sharding_mode": "runtime" if plan.config.strategy == "runtime_balanced" else plan.config.strategy,
        "max_proteins_per_batch": plan.config.max_proteins_per_batch,
        "target_runtime_hours": float(target_runtime_hours),
        "max_total_residues": max_total_residues,
        "unique_lengths": len(set(lengths)),
        "length_range": list(length_range),
        "length_stats": length_stats,
        "force_rerun": force_rerun,
        "rebalance": rebalance,
        "skip_completed": skip_completed,
        assignment_key: assignment_summaries,
    }
    return json.dumps(manifest, indent=2)


def _render_csv(assignments: tuple[FoldingQueueAssignment, ...]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_COLUMNS)
    writer.writeheader()
    for assignment in assignments:
        writer.writerow(
            {
                "batch_id": assignment.batch_id,
                "seq_length": assignment.sequence_length,
                "msa_path": assignment.msa_path,
                "protein_id": assignment.protein_id,
            }
        )
    return output.getvalue()


def _assignments_by_worker(plan: FoldingQueuePlan) -> dict[int, tuple[FoldingQueueAssignment, ...]]:
    buckets: dict[int, list[FoldingQueueAssignment]] = {}
    for assignment in plan.assignments:
        buckets.setdefault(assignment.worker_id, []).append(assignment)
    return {
        worker_id: tuple(sorted(assignments, key=lambda assignment: (assignment.batch_id, assignment.source_ordinal)))
        for worker_id, assignments in sorted(buckets.items())
    }


def _linear_quantile(sorted_values: list[int], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    if fraction >= 0.5:
        return upper - (upper - lower) * (1 - fraction)
    return lower + (upper - lower) * fraction


def _validate_manifest_inputs(
    plan: FoldingQueuePlan,
    *,
    created: str,
    batch_info_path: str,
    output_dir: str,
    num_gpus: int,
    num_nodes: int | None,
    target_runtime_hours: float,
    max_total_residues: int,
) -> None:
    for field_name, value in {
        "created": created,
        "batch_info_path": batch_info_path,
        "output_dir": output_dir,
    }.items():
        if not value.strip():
            msg = f"{field_name} must be a non-empty string"
            raise ValueError(msg)
    if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus <= 0:
        msg = "num_gpus must be a positive integer"
        raise ValueError(msg)
    if (
        not isinstance(target_runtime_hours, int | float)
        or isinstance(target_runtime_hours, bool)
        or not math.isfinite(target_runtime_hours)
        or target_runtime_hours <= 0
    ):
        msg = "target_runtime_hours must be a finite positive number"
        raise ValueError(msg)
    if not isinstance(max_total_residues, int) or isinstance(max_total_residues, bool) or max_total_residues <= 0:
        msg = "max_total_residues must be a positive integer"
        raise ValueError(msg)

    if plan.config.layout == "per_node":
        if not isinstance(num_nodes, int) or isinstance(num_nodes, bool) or num_nodes != plan.config.worker_count:
            msg = "num_nodes must equal worker_count for per_node layout"
            raise ValueError(msg)
    elif num_nodes is not None:
        msg = "num_nodes must be null for legacy_per_gpu layout"
        raise ValueError(msg)
    if plan.config.layout == "legacy_per_gpu" and num_gpus != plan.config.worker_count:
        msg = "num_gpus must equal worker_count for legacy_per_gpu layout"
        raise ValueError(msg)


__all__ = ["render_folding_queue_manifest", "render_folding_queues"]
