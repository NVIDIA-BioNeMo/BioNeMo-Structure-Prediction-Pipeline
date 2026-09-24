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

"""Focused tests for deterministic folding queue planning and rendering."""

from __future__ import annotations

import ast
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_index import FoldingIndexRecord
from bspp.orchestration.contract.folding_queue import (
    FoldingQueueAssignment,
    FoldingQueueConfig,
    FoldingQueueLayout,
    folding_queue_config_from_mapping,
    folding_queue_plan_from_mapping,
    folding_queue_render_result_from_mapping,
)
from bspp.orchestration.contract.versioning import UnsupportedSchemaVersionError
from bspp.orchestration.runtime.folding.queue import plan_folding_queue
from bspp.orchestration.runtime.folding.queue_render import (
    _linear_quantile,
    render_folding_queue_manifest,
    render_folding_queues,
)


def _record(
    protein_id: str,
    sequence_length: int,
    source_ordinal: int,
    *,
    msa_depth: int = 1,
    msa_path: str | None = None,
) -> FoldingIndexRecord:
    return FoldingIndexRecord(
        source_ordinal=source_ordinal,
        protein_id=protein_id,
        msa_path=msa_path or f"/a3m/{protein_id}.a3m",
        query_sequence="A" * sequence_length,
        sequence_length=sequence_length,
        chain_lengths=(sequence_length,),
        chain_count=1,
        chain_cardinalities=(1,),
        msa_depth=msa_depth,
        total_length=sequence_length,
    )


def _historical_heterodimer(
    protein_id: str,
    scheduling_length: int,
    source_ordinal: int,
    *,
    msa_depth: int,
) -> FoldingIndexRecord:
    return FoldingIndexRecord(
        source_ordinal=source_ordinal,
        protein_id=protein_id,
        msa_path=f"/a3m/{protein_id}.a3m",
        query_sequence="ABCDE",
        sequence_length=scheduling_length,
        chain_lengths=(3, 2),
        chain_count=2,
        chain_cardinalities=(1, 1),
        msa_depth=msa_depth,
        total_length=5,
    )


def test_round_robin_uses_scheduling_length_then_source_order() -> None:
    records = (
        _record("long", 30, 0),
        _record("short-first", 10, 1),
        _record("middle", 20, 2),
        _record("short-second", 10, 3),
    )
    config = FoldingQueueConfig(strategy="round_robin", layout="per_node", worker_count=3)

    first = plan_folding_queue(records, config)
    second = plan_folding_queue(reversed(records), config)

    assert first == second
    assert tuple(assignment.protein_id for assignment in first.assignments) == (
        "short-first",
        "short-second",
        "middle",
        "long",
    )
    assert tuple(assignment.worker_id for assignment in first.assignments) == (0, 1, 2, 0)
    assert tuple(assignment.batch_id for assignment in first.assignments) == (0, 1, 2, 3)
    assert all(assignment.length_batch is None for assignment in first.assignments)


def test_exact_length_keeps_groups_and_count_limited_batches_together() -> None:
    records = (
        _record("ten-a", 10, 0, msa_depth=1),
        _record("ten-b", 10, 1, msa_depth=100),
        _record("ten-c", 10, 2, msa_depth=1),
        _record("ten-d", 10, 3, msa_depth=1),
        _record("ten-e", 10, 4, msa_depth=1),
        _record("twenty", 20, 5, msa_depth=1),
    )
    config = FoldingQueueConfig(
        strategy="exact_length",
        layout="per_node",
        worker_count=2,
        max_proteins_per_batch=2,
    )

    plan = plan_folding_queue(records, config)
    by_batch: dict[int, list[FoldingQueueAssignment]] = {}
    for assignment in plan.assignments:
        by_batch.setdefault(assignment.batch_id, []).append(assignment)

    assert tuple(sorted((assignment.length_batch, assignment.batch_id) for assignment in plan.assignments)) == (
        ("10_0", 0),
        ("10_0", 0),
        ("10_1", 1),
        ("10_1", 1),
        ("10_2", 2),
        ("20_0", 3),
    )
    assert all(len(batch) <= 2 for batch in by_batch.values())
    assert all(len({assignment.sequence_length for assignment in batch}) == 1 for batch in by_batch.values())
    assert all(len({assignment.worker_id for assignment in batch}) == 1 for batch in by_batch.values())
    assert {assignment.batch_id: assignment.worker_id for assignment in plan.assignments} == {
        0: 1,
        1: 1,
        2: 0,
        3: 0,
    }


def test_exact_length_equal_load_ties_choose_lowest_worker() -> None:
    records = tuple(_record(f"same-{index}", 10, index) for index in range(4))
    config = FoldingQueueConfig(
        strategy="exact_length",
        layout="legacy_per_gpu",
        worker_count=2,
        max_proteins_per_batch=1,
    )

    plan = plan_folding_queue(records, config)

    assert {assignment.batch_id: assignment.worker_id for assignment in plan.assignments} == {
        0: 0,
        1: 1,
        2: 0,
        3: 1,
    }


def test_exact_length_csv_keeps_multi_protein_batch_contiguous() -> None:
    plan = plan_folding_queue(
        tuple(_record(f"same-{index}", 10, index) for index in range(3)),
        FoldingQueueConfig(
            strategy="exact_length",
            layout="per_node",
            worker_count=1,
            max_proteins_per_batch=2,
        ),
    )

    artifact = render_folding_queues(plan).artifacts[0]

    assert artifact.total_proteins == 3
    assert artifact.num_batches == 2
    assert artifact.content.startswith(
        "batch_id,seq_length,msa_path,protein_id\r\n"
        "0,10,/a3m/same-0.a3m,same-0\r\n"
        "0,10,/a3m/same-1.a3m,same-1\r\n"
        "1,10,/a3m/same-2.a3m,same-2\r\n"
    )


def test_runtime_balanced_uses_pinned_formula_clamp_and_heap_ties() -> None:
    defaults = FoldingQueueConfig(
        strategy="runtime_balanced",
        layout="legacy_per_gpu",
        worker_count=2,
    )
    assert defaults.runtime_coefficient == 0.0001
    assert defaults.base_overhead_seconds == 30.0

    records = (
        _record("three-hundred", 100, 0, msa_depth=3),
        _record("two-hundred", 200, 1, msa_depth=1),
        _record("one-hundred", 100, 2, msa_depth=1),
        _record("clamped", 50, 3, msa_depth=1),
    )
    config = FoldingQueueConfig(
        strategy="runtime_balanced",
        layout="legacy_per_gpu",
        worker_count=2,
        runtime_coefficient=1.0,
        base_overhead_seconds=0.0,
    )

    plan = plan_folding_queue(records, config)

    assert tuple(assignment.protein_id for assignment in plan.assignments) == (
        "three-hundred",
        "two-hundred",
        "one-hundred",
        "clamped",
    )
    assert tuple(assignment.worker_id for assignment in plan.assignments) == (0, 1, 1, 0)
    assert tuple(assignment.batch_id for assignment in plan.assignments) == (0, 1, 2, 3)


def test_runtime_balanced_uses_index_scheduling_length_not_physical_chain_total() -> None:
    historical = _historical_heterodimer("historical", 100, 0, msa_depth=2)
    ordinary = _record("ordinary", 50, 1, msa_depth=3)
    config = FoldingQueueConfig(
        strategy="runtime_balanced",
        layout="per_node",
        worker_count=1,
        runtime_coefficient=1.0,
        base_overhead_seconds=0.0,
    )

    plan = plan_folding_queue((ordinary, historical), config)

    # 100 scheduling units * depth 2 = 200, rather than the parsed
    # heterodimer total_length 5 * depth 2 = 10.
    assert tuple(assignment.protein_id for assignment in plan.assignments) == ("historical", "ordinary")


@pytest.mark.parametrize("duplicate_field", ["source_ordinal", "protein_id", "msa_path"])
def test_queue_planning_rejects_duplicate_input_identity_fields(duplicate_field: str) -> None:
    first = _record("first", 10, 0)
    values: dict[str, object] = {"protein_id": "second", "sequence_length": 20, "source_ordinal": 1}
    if duplicate_field == "source_ordinal":
        values["source_ordinal"] = first.source_ordinal
    elif duplicate_field == "protein_id":
        values["protein_id"] = first.protein_id
    else:
        values["msa_path"] = first.msa_path
    second = _record(**values)  # type: ignore[arg-type]
    config = FoldingQueueConfig(strategy="round_robin", layout="per_node", worker_count=2)

    with pytest.raises(ValueError, match=f"duplicate {duplicate_field}"):
        plan_folding_queue((first, second), config)


@pytest.mark.parametrize("layout", ["per_node", "legacy_per_gpu"])
def test_queue_render_is_byte_stable_with_exact_columns_and_newlines(
    layout: FoldingQueueLayout,
) -> None:
    records = (
        _record("first", 10, 0),
        _record("second", 20, 1),
        _record("third", 30, 2, msa_path="/a3m/with,comma.a3m"),
    )
    config = FoldingQueueConfig(strategy="round_robin", layout=layout, worker_count=2)
    plan = plan_folding_queue(records, config)

    first = render_folding_queues(plan)
    second = render_folding_queues(plan)

    assert first == second
    prefix = "node" if layout == "per_node" else "gpu"
    assert tuple(artifact.file_name for artifact in first.artifacts) == (
        f"{prefix}0_batches.csv",
        f"{prefix}1_batches.csv",
    )
    assert first.artifacts[0].content == (
        'batch_id,seq_length,msa_path,protein_id\r\n0,10,/a3m/first.a3m,first\r\n2,30,"/a3m/with,comma.a3m",third\r\n'
    )
    assert first.artifacts[0].total_proteins == 2
    assert first.artifacts[0].num_batches == 2


def test_empty_and_fewer_than_workers_emit_only_nonempty_artifacts() -> None:
    config = FoldingQueueConfig(strategy="round_robin", layout="per_node", worker_count=4)

    empty_plan = plan_folding_queue((), config)
    empty_render = render_folding_queues(empty_plan)
    short_plan = plan_folding_queue((_record("a", 10, 0), _record("b", 20, 1)), config)
    short_render = render_folding_queues(short_plan)

    assert empty_plan.assignments == ()
    assert empty_render.artifacts == ()
    assert tuple(artifact.worker_id for artifact in short_render.artifacts) == (0, 1)


def test_manifest_render_is_deterministic_explicit_and_non_mutating(tmp_path: Path) -> None:
    config = FoldingQueueConfig(
        strategy="exact_length",
        layout="per_node",
        worker_count=2,
        max_proteins_per_batch=2,
    )
    plan = plan_folding_queue(
        (_record("a", 10, 0), _record("b", 20, 1), _record("c", 30, 2)),
        config,
    )
    output_dir = tmp_path / "not-created"
    first = render_folding_queue_manifest(
        plan,
        created="2026-08-14 12:00:00",
        batch_info_path="/input/batch_info.parquet",
        output_dir=str(output_dir),
        num_gpus=16,
        num_nodes=2,
        target_runtime_hours=168.0,
        max_total_residues=100_000,
        force_rerun=False,
        rebalance=True,
        skip_completed=True,
    )
    second = render_folding_queue_manifest(
        plan,
        created="2026-08-14 12:00:00",
        batch_info_path="/input/batch_info.parquet",
        output_dir=str(output_dir),
        num_gpus=16,
        num_nodes=2,
        target_runtime_hours=168.0,
        max_total_residues=100_000,
        force_rerun=False,
        rebalance=True,
        skip_completed=True,
    )
    payload = json.loads(first)

    assert first == second
    assert payload == {
        "version": "2.0.0",
        "created": "2026-08-14 12:00:00",
        "pipeline": "trt-bionemo",
        "adapted_from": "colabfold-slurm",
        "batch_info_path": "/input/batch_info.parquet",
        "output_dir": str(output_dir),
        "total_proteins": 3,
        "num_gpus": 16,
        "execution_mode": "per_node",
        "num_nodes": 2,
        "sharding_mode": "exact_length",
        "max_proteins_per_batch": 2,
        "target_runtime_hours": 168.0,
        "max_total_residues": 100_000,
        "unique_lengths": 3,
        "length_range": [10, 30],
        "length_stats": {"p10": 12, "p25": 15, "p50": 20, "p75": 25, "p90": 28, "mean": 20.0},
        "force_rerun": False,
        "rebalance": True,
        "skip_completed": True,
        "node_assignments": {
            "0": {"total_proteins": 1, "num_batches": 1},
            "1": {"total_proteins": 2, "num_batches": 2},
        },
    }
    assert not output_dir.exists()


def test_legacy_per_gpu_manifest_uses_historical_layout_and_validates_resources() -> None:
    plan = plan_folding_queue(
        (_record("a", 10, 0), _record("b", 20, 1)),
        FoldingQueueConfig(strategy="runtime_balanced", layout="legacy_per_gpu", worker_count=2),
    )

    manifest = render_folding_queue_manifest(
        plan,
        created="2026-08-14 12:00:00",
        batch_info_path="/input/batch_info.parquet",
        output_dir="/output",
        num_gpus=2,
        num_nodes=None,
        target_runtime_hours=168.0,
        max_total_residues=100_000,
        force_rerun=False,
        rebalance=False,
        skip_completed=True,
    )
    payload = json.loads(manifest)

    assert payload["execution_mode"] == "per_gpu"
    assert payload["num_nodes"] is None
    assert payload["sharding_mode"] == "runtime"
    assert "gpu_assignments" in payload
    assert "node_assignments" not in payload
    with pytest.raises(ValueError, match="num_nodes must be null"):
        render_folding_queue_manifest(
            plan,
            created="2026-08-14 12:00:00",
            batch_info_path="/input/batch_info.parquet",
            output_dir="/output",
            num_gpus=2,
            num_nodes=1,
            target_runtime_hours=168.0,
            max_total_residues=100_000,
            force_rerun=False,
            rebalance=False,
            skip_completed=True,
        )
    with pytest.raises(ValueError, match="num_gpus must equal worker_count"):
        render_folding_queue_manifest(
            plan,
            created="2026-08-14 12:00:00",
            batch_info_path="/input/batch_info.parquet",
            output_dir="/output",
            num_gpus=3,
            num_nodes=None,
            target_runtime_hours=168,
            max_total_residues=100_000,
            force_rerun=False,
            rebalance=False,
            skip_completed=True,
        )


def test_linear_quantile_preserves_numpy_upper_half_rounding() -> None:
    values = [
        21_843_168,
        24_892_249,
        36_874_040,
        50_916_420,
        82_504_833,
        96_874_973,
        141_315_464,
        182_229_963,
        216_639_479,
        265_147_242,
        288_849_869,
        357_748_446,
        383_436_199,
        423_585_965,
        426_933_064,
        441_030_624,
        463_527_029,
        529_361_027,
        546_224_025,
        547_925_481,
        584_432_260,
        672_372_950,
        692_291_371,
        888_653_167,
    ]

    assert int(_linear_quantile(values, 0.9)) == 645_990_742


def test_queue_contract_loaders_are_immutable_versioned_and_fail_closed() -> None:
    config = FoldingQueueConfig(strategy="round_robin", layout="per_node", worker_count=2)
    plan = plan_folding_queue((_record("a", 10, 0),), config)
    rendered = render_folding_queues(plan)

    assert folding_queue_config_from_mapping(config.to_mapping()) == config
    assert folding_queue_plan_from_mapping(plan.to_mapping()) == plan
    assert folding_queue_render_result_from_mapping(rendered.to_mapping()) == rendered
    with pytest.raises(FrozenInstanceError):
        config.worker_count = 3  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.assignments = ()  # type: ignore[misc]
    with pytest.raises(UnsupportedSchemaVersionError):
        folding_queue_config_from_mapping({**config.to_mapping(), "schema_version": 2})
    with pytest.raises(ValueError, match="unknown fields"):
        folding_queue_config_from_mapping({**config.to_mapping(), "extra": "no"})
    with pytest.raises(ValueError, match="worker_id"):
        folding_queue_plan_from_mapping(
            {
                **plan.to_mapping(),
                "assignments": [{**plan.assignments[0].to_mapping(), "worker_id": 2}],
            }
        )
    with pytest.raises(ValueError, match="total_proteins"):
        folding_queue_render_result_from_mapping(
            {
                **rendered.to_mapping(),
                "artifacts": [{**rendered.artifacts[0].to_mapping(), "total_proteins": 2}],
            }
        )

    two_record_plan = plan_folding_queue(
        (_record("a", 10, 0), _record("b", 20, 1)),
        FoldingQueueConfig(strategy="round_robin", layout="per_node", worker_count=2),
    )
    incoherent_batches = two_record_plan.to_mapping()
    assert isinstance(incoherent_batches["assignments"], list)
    incoherent_batches["assignments"][1]["batch_id"] = 0
    incoherent_batches["assignments"][1]["worker_id"] = incoherent_batches["assignments"][0]["worker_id"]
    with pytest.raises(ValueError, match="must identify exactly one protein"):
        folding_queue_plan_from_mapping(incoherent_batches)

    exact_plan = plan_folding_queue(
        (_record("c", 10, 0), _record("d", 10, 1)),
        FoldingQueueConfig(
            strategy="exact_length",
            layout="per_node",
            worker_count=2,
            max_proteins_per_batch=2,
        ),
    )
    split_batch = exact_plan.to_mapping()
    assert isinstance(split_batch["assignments"], list)
    original_worker = split_batch["assignments"][0]["worker_id"]
    split_batch["assignments"][1]["worker_id"] = 1 - original_worker
    with pytest.raises(ValueError, match="exactly one worker"):
        folding_queue_plan_from_mapping(split_batch)

    for record in (config, plan.assignments[0], plan, rendered.artifacts[0], rendered):
        with pytest.raises(ValueError, match="schema_version must be declared explicitly"):
            replace(record, schema_version=None)


def test_queue_modules_do_not_import_execution_surfaces() -> None:
    import bspp.orchestration.runtime.folding.queue as queue_module
    import bspp.orchestration.runtime.folding.queue_planning as planning_module
    import bspp.orchestration.runtime.folding.queue_render as render_module

    imports: set[str] = set()
    for module in (queue_module, planning_module, render_module):
        module_path = module.__file__
        assert module_path is not None
        source = Path(module_path).read_text()
        tree = ast.parse(source)
        imports.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        imports.update(
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
        )

    assert not imports & {"subprocess", "paramiko", "docker"}
    assert not any("slurm" in name for name in imports)
