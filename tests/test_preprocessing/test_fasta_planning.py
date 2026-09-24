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

"""Public-seam tests for deterministic preprocessing FASTA planning."""

from __future__ import annotations

import copy
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from bspp.orchestration.contract.preprocessing import (
    FastaNormalizationMode,
    PreprocessingChunk,
    PreprocessingChunkAssignment,
    PreprocessingFastaRecord,
    PreprocessingInput,
    PreprocessingPlanOptions,
    PreprocessingTranche,
    PreprocessingWorkPlan,
    preprocessing_chunk_assignment_from_mapping,
    preprocessing_chunk_from_mapping,
    preprocessing_fasta_record_from_mapping,
    preprocessing_input_from_mapping,
    preprocessing_plan_options_from_mapping,
    preprocessing_tranche_from_mapping,
    preprocessing_work_plan_from_mapping,
)
from bspp.orchestration.runtime.preprocessing.planning import plan_preprocessing_fasta, plan_preprocessing_records


def test_strict_two_line_fasta_preserves_records_and_source_order(tmp_path: Path) -> None:
    source = tmp_path / "proteins.fa"
    source.write_text(">alpha first description\nMKT\n>beta second description\nAA*\n")

    plan = plan_preprocessing_fasta(source, PreprocessingPlanOptions())

    assert tuple(
        (record.header, record.sequence, record.identity, record.source_ordinal) for record in plan.input.records
    ) == (
        (">alpha first description", "MKT", "alpha", 0),
        (">beta second description", "AA*", "beta", 1),
    )


def test_named_multiline_normalization_matches_canonical_two_line_input(tmp_path: Path) -> None:
    strict_source = tmp_path / "strict.fa"
    strict_source.write_text(">alpha description\nMKTW\n>beta\nAACC\n")
    multiline_source = tmp_path / "multiline.fa"
    multiline_source.write_text(">alpha description\nMK\nTW\n>beta\nAA\nCC\n")

    strict = plan_preprocessing_fasta(strict_source, PreprocessingPlanOptions())
    normalized = plan_preprocessing_fasta(
        multiline_source,
        PreprocessingPlanOptions(normalization_mode="normalize-multiline"),
    )

    assert tuple((record.header, record.sequence, record.identity) for record in normalized.input.records) == tuple(
        (record.header, record.sequence, record.identity) for record in strict.input.records
    )


def test_strict_default_rejects_well_formed_multiline_fasta(tmp_path: Path) -> None:
    source = tmp_path / "multiline.fa"
    source.write_text(">alpha\nAA\nTT\n>beta\nGG\nCC\n")

    with pytest.raises(ValueError, match="malformed header"):
        plan_preprocessing_fasta(source, PreprocessingPlanOptions())


@pytest.mark.parametrize(
    ("contents", "mode", "message"),
    [
        ("", "strict-two-line", "empty"),
        (">alpha\nAAA\n>beta\n", "strict-two-line", "truncated"),
        ("alpha\nAAA\n", "strict-two-line", "malformed header"),
        ("> alpha\nAAA\n", "strict-two-line", "malformed header"),
        (">alpha\n>beta\n", "strict-two-line", "missing sequence"),
        (">alpha\nAA AA\n", "strict-two-line", "whitespace"),
        ("AAA\n>alpha\nTTT\n", "normalize-multiline", "before the first header"),
        (">alpha\n\n", "normalize-multiline", "missing sequence"),
        (">alpha\nAA\n\nTT\n", "normalize-multiline", "blank sequence line"),
    ],
)
def test_fasta_parser_rejects_invalid_record_structures(
    tmp_path: Path,
    contents: str,
    mode: FastaNormalizationMode,
    message: str,
) -> None:
    source = tmp_path / "invalid.fa"
    source.write_text(contents)

    with pytest.raises(ValueError, match=message):
        plan_preprocessing_fasta(source, PreprocessingPlanOptions(normalization_mode=mode))


@pytest.mark.parametrize("mode", ["strict-two-line", "normalize-multiline"])
def test_fasta_parser_rejects_duplicate_canonical_identities(
    tmp_path: Path,
    mode: FastaNormalizationMode,
) -> None:
    source = tmp_path / "duplicates.fa"
    source.write_text(">alpha first\nAAA\n>alpha second\nTTT\n")

    with pytest.raises(ValueError, match="duplicate FASTA identity 'alpha'"):
        plan_preprocessing_fasta(source, PreprocessingPlanOptions(normalization_mode=mode))


def test_fasta_planner_rejects_non_fa_source_suffix(tmp_path: Path) -> None:
    source = tmp_path / "proteins.fasta"
    source.write_text(">alpha\nAAA\n")

    with pytest.raises(ValueError, match=r"must use the .fa suffix"):
        plan_preprocessing_fasta(source, PreprocessingPlanOptions())


@pytest.mark.parametrize("source_kind", ["missing", "invalid-utf8"])
def test_fasta_read_failures_are_normalized_to_validation_errors(tmp_path: Path, source_kind: str) -> None:
    source = tmp_path / "unreadable.fa"
    if source_kind == "invalid-utf8":
        source.write_bytes(b">alpha\n\xff\n")

    with pytest.raises(ValueError, match="cannot read preprocessing FASTA"):
        plan_preprocessing_fasta(source, PreprocessingPlanOptions())


def test_uneven_tranches_and_chunks_use_exact_baseline_suffixes_without_record_loss(tmp_path: Path) -> None:
    source = tmp_path / "proteins.fa"
    source.write_text("".join(f">protein-{ordinal}\nSEQ{ordinal}\n" for ordinal in range(7)))

    plan = plan_preprocessing_fasta(
        source,
        PreprocessingPlanOptions(requested_tranches=3, records_per_chunk=2),
    )

    assert tuple((tranche.name, tranche.fasta_name, tranche.record_ordinals) for tranche in plan.tranches) == (
        ("tranche00", "proteins_tranche00.fa", (0, 1, 2)),
        ("tranche01", "proteins_tranche01.fa", (3, 4, 5)),
        ("tranche02", "proteins_tranche02.fa", (6,)),
    )
    assert tuple((chunk.name, chunk.tranche_name, chunk.record_ordinals) for chunk in plan.chunks) == (
        ("proteins_tranche00_00000.fa", "tranche00", (0, 1)),
        ("proteins_tranche00_00001.fa", "tranche00", (2,)),
        ("proteins_tranche01_00000.fa", "tranche01", (3, 4)),
        ("proteins_tranche01_00001.fa", "tranche01", (5,)),
        ("proteins_tranche02_00000.fa", "tranche02", (6,)),
    )
    assert tuple(ordinal for chunk in plan.chunks for ordinal in chunk.record_ordinals) == tuple(range(7))


def test_requesting_more_tranches_than_records_returns_every_real_record_once(tmp_path: Path) -> None:
    source = tmp_path / "small.fa"
    source.write_text(">alpha\nAAA\n>beta\nTTT\n")

    plan = plan_preprocessing_fasta(source, PreprocessingPlanOptions(requested_tranches=5))

    assert tuple((tranche.name, tranche.record_ordinals) for tranche in plan.tranches) == (
        ("tranche00", (0,)),
        ("tranche01", (1,)),
    )
    assert tuple(chunk.record_ordinals for chunk in plan.chunks) == ((0,), (1,))


def test_sorted_chunks_are_assigned_round_robin_with_copy_staging_and_worker_order(tmp_path: Path) -> None:
    source = tmp_path / "proteins.fa"
    source.write_text("".join(f">p{ordinal}\nSEQ{ordinal}\n" for ordinal in range(5)))

    plan = plan_preprocessing_fasta(
        source,
        PreprocessingPlanOptions(records_per_chunk=1, nodes=2, gpus_per_node=2),
    )

    assert tuple(
        (
            assignment.chunk_name,
            assignment.global_worker_index,
            assignment.node_index,
            assignment.gpu_index,
            assignment.worker_label,
            assignment.worker_ordinal,
            assignment.staging_operation,
            assignment.source_path,
            assignment.staged_path,
        )
        for assignment in plan.assignments
    ) == (
        (
            "proteins_tranche00_00000.fa",
            0,
            0,
            0,
            "n0g0",
            0,
            "copy",
            "splitted/proteins_tranche00_00000.fa",
            "n0g0/proteins_tranche00_00000.fa",
        ),
        (
            "proteins_tranche00_00001.fa",
            1,
            0,
            1,
            "n0g1",
            0,
            "copy",
            "splitted/proteins_tranche00_00001.fa",
            "n0g1/proteins_tranche00_00001.fa",
        ),
        (
            "proteins_tranche00_00002.fa",
            2,
            1,
            0,
            "n1g0",
            0,
            "copy",
            "splitted/proteins_tranche00_00002.fa",
            "n1g0/proteins_tranche00_00002.fa",
        ),
        (
            "proteins_tranche00_00003.fa",
            3,
            1,
            1,
            "n1g1",
            0,
            "copy",
            "splitted/proteins_tranche00_00003.fa",
            "n1g1/proteins_tranche00_00003.fa",
        ),
        (
            "proteins_tranche00_00004.fa",
            0,
            0,
            0,
            "n0g0",
            1,
            "copy",
            "splitted/proteins_tranche00_00004.fa",
            "n0g0/proteins_tranche00_00004.fa",
        ),
    )


def test_worker_distribution_and_counters_restart_for_each_tranche(tmp_path: Path) -> None:
    source = tmp_path / "proteins.fa"
    source.write_text("".join(f">p{ordinal}\nSEQ{ordinal}\n" for ordinal in range(10)))

    plan = plan_preprocessing_fasta(
        source,
        PreprocessingPlanOptions(requested_tranches=2, records_per_chunk=2, nodes=1, gpus_per_node=2),
    )

    assert tuple(
        (assignment.chunk_name, assignment.worker_label, assignment.worker_ordinal) for assignment in plan.assignments
    ) == (
        ("proteins_tranche00_00000.fa", "n0g0", 0),
        ("proteins_tranche00_00001.fa", "n0g1", 0),
        ("proteins_tranche00_00002.fa", "n0g0", 1),
        ("proteins_tranche01_00000.fa", "n0g0", 0),
        ("proteins_tranche01_00001.fa", "n0g1", 0),
        ("proteins_tranche01_00002.fa", "n0g0", 1),
    )


def test_explicitly_empty_selected_records_return_an_empty_plan() -> None:
    plan = plan_preprocessing_records(
        source_path="selected.fa",
        records=(),
        options=PreprocessingPlanOptions(nodes=2, gpus_per_node=4),
    )

    assert plan.input.records == ()
    assert plan.tranches == ()
    assert plan.chunks == ()
    assert plan.assignments == ()


def test_explicit_selection_rejects_duplicate_canonical_identities() -> None:
    records = (
        PreprocessingFastaRecord(header=">alpha first", sequence="AAA", identity="alpha", source_ordinal=0),
        PreprocessingFastaRecord(header=">alpha second", sequence="TTT", identity="alpha", source_ordinal=1),
    )

    with pytest.raises(ValueError, match="duplicate FASTA identity 'alpha'"):
        plan_preprocessing_records(
            source_path="selected.fa",
            records=records,
            options=PreprocessingPlanOptions(),
        )


def test_explicit_selection_requires_increasing_source_ordinals_and_fa_identity() -> None:
    first = PreprocessingFastaRecord(header=">first", sequence="AAA", identity="first", source_ordinal=1)
    second = PreprocessingFastaRecord(header=">second", sequence="TTT", identity="second", source_ordinal=0)

    with pytest.raises(ValueError, match="strictly increasing source ordinals"):
        PreprocessingInput(
            source_path="selected.fa",
            normalization_mode="strict-two-line",
            records=(first, second),
        )
    with pytest.raises(ValueError, match=r"must use the .fa suffix"):
        plan_preprocessing_records(
            source_path="selected.fasta",
            records=(first,),
            options=PreprocessingPlanOptions(),
        )


def test_fewer_chunks_than_workers_and_replay_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "small.fa"
    source.write_text(">alpha\nAAA\n>beta\nTTT\n")
    options = PreprocessingPlanOptions(records_per_chunk=1, nodes=2, gpus_per_node=4)

    first = plan_preprocessing_fasta(source, options)
    replay = plan_preprocessing_fasta(source, options)

    assert tuple(assignment.worker_label for assignment in first.assignments) == ("n0g0", "n0g1")
    assert replay == first


def test_work_plan_mapping_round_trip_is_versioned_and_recursively_immutable(tmp_path: Path) -> None:
    source = tmp_path / "proteins.fa"
    source.write_text(">alpha\nAAA\n>beta\nTTT\n")
    plan = plan_preprocessing_fasta(
        source,
        PreprocessingPlanOptions(requested_tranches=2, records_per_chunk=1, nodes=1, gpus_per_node=2),
    )

    mapping = plan.to_mapping()
    loaded = preprocessing_work_plan_from_mapping(mapping)

    assert mapping["schema_version"] == 1
    assert loaded == plan
    with pytest.raises(FrozenInstanceError):
        loaded.assignments[0].worker_label = "n9g9"  # type: ignore[misc]


def test_chunk_suffix_capacity_is_exactly_five_digits() -> None:
    boundary = PreprocessingChunk(
        name="proteins_tranche00_99999.fa",
        tranche_name="tranche00",
        ordinal=99_999,
        tranche_chunk_ordinal=99_999,
        record_ordinals=(0,),
    )

    assert boundary.tranche_chunk_ordinal == 99_999
    with pytest.raises(ValueError, match="five-digit suffix capacity"):
        PreprocessingChunk(
            name="proteins_tranche00_100000.fa",
            tranche_name="tranche00",
            ordinal=100_000,
            tranche_chunk_ordinal=100_000,
            record_ordinals=(0,),
        )


def test_direct_contract_construction_rejects_invalid_version_staging_and_plan_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported PreprocessingTranche schema_version 2"):
        PreprocessingTranche(
            name="tranche00",
            fasta_name="proteins_tranche00.fa",
            ordinal=0,
            record_ordinals=(0,),
            schema_version=2,
        )
    with pytest.raises(ValueError, match="staging_operation must be 'copy'"):
        PreprocessingChunkAssignment(
            chunk_name="proteins_tranche00_00000.fa",
            global_worker_index=0,
            node_index=0,
            gpu_index=0,
            worker_label="n0g0",
            worker_ordinal=0,
            source_path="splitted/proteins_tranche00_00000.fa",
            staged_path="n0g0/proteins_tranche00_00000.fa",
            staging_operation="move",  # type: ignore[arg-type]
        )

    plan = _small_plan(tmp_path)
    with pytest.raises(ValueError, match="exactly one assignment per chunk"):
        replace(plan, assignments=())
    wrong_worker = replace(plan.assignments[0], global_worker_index=1)
    with pytest.raises(ValueError, match="worker arithmetic"):
        replace(plan, assignments=(wrong_worker,))

    for record in (
        plan.input.records[0],
        plan.input,
        plan.options,
        plan.tranches[0],
        plan.chunks[0],
        plan.assignments[0],
        plan,
    ):
        with pytest.raises(ValueError, match=r"Unsupported .* schema_version 2"):
            replace(record, schema_version=2)
    with pytest.raises(ValueError, match="tranche name must match"):
        replace(plan.tranches[0], name="tranche01")
    with pytest.raises(ValueError, match="chunk name must match"):
        replace(plan.chunks[0], name="small_tranche00_00001.fa")
    with pytest.raises(ValueError, match="worker_label must match"):
        replace(plan.assignments[0], worker_label="n9g9")


@pytest.mark.parametrize(
    "tampering",
    [
        "duplicate-tranche",
        "missing-tranche",
        "duplicate-chunk",
        "missing-chunk",
        "duplicate-assignment",
        "missing-assignment",
        "unknown-assignment",
        "unknown-tranche-reference",
        "record-ordinal",
    ],
)
def test_recursive_work_plan_loader_rejects_referential_integrity_tampering(
    tmp_path: Path,
    tampering: str,
) -> None:
    mapping = copy.deepcopy(_small_plan(tmp_path).to_mapping())
    tranches = mapping["tranches"]
    chunks = mapping["chunks"]
    assignments = mapping["assignments"]
    assert isinstance(tranches, list)
    assert isinstance(chunks, list)
    assert isinstance(assignments, list)
    if tampering == "duplicate-tranche":
        tranches.append(copy.deepcopy(tranches[0]))
    elif tampering == "missing-tranche":
        tranches.clear()
    elif tampering == "duplicate-chunk":
        chunks.append(copy.deepcopy(chunks[0]))
    elif tampering == "missing-chunk":
        chunks.clear()
    elif tampering == "duplicate-assignment":
        assignments.append(copy.deepcopy(assignments[0]))
    elif tampering == "missing-assignment":
        assignments.clear()
    elif tampering == "unknown-assignment":
        assignment = assignments[0]
        assert isinstance(assignment, dict)
        assignment["chunk_name"] = "ghost_tranche00_00000.fa"
        assignment["source_path"] = "splitted/ghost_tranche00_00000.fa"
        assignment["staged_path"] = "n0g0/ghost_tranche00_00000.fa"
    elif tampering == "unknown-tranche-reference":
        chunk = chunks[0]
        assignment = assignments[0]
        assert isinstance(chunk, dict)
        assert isinstance(assignment, dict)
        chunk["name"] = "small_tranche99_00000.fa"
        chunk["tranche_name"] = "tranche99"
        assignment["chunk_name"] = "small_tranche99_00000.fa"
        assignment["source_path"] = "splitted/small_tranche99_00000.fa"
        assignment["staged_path"] = "n0g0/small_tranche99_00000.fa"
    else:
        chunk = chunks[0]
        assert isinstance(chunk, dict)
        chunk["record_ordinals"] = [99]

    with pytest.raises(ValueError):
        preprocessing_work_plan_from_mapping(mapping)


def test_every_preprocessing_record_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    plan = _small_plan(tmp_path)
    cases = (
        ("PreprocessingFastaRecord", preprocessing_fasta_record_from_mapping, plan.input.records[0].to_mapping()),
        ("PreprocessingInput", preprocessing_input_from_mapping, plan.input.to_mapping()),
        ("PreprocessingPlanOptions", preprocessing_plan_options_from_mapping, plan.options.to_mapping()),
        ("PreprocessingTranche", preprocessing_tranche_from_mapping, plan.tranches[0].to_mapping()),
        ("PreprocessingChunk", preprocessing_chunk_from_mapping, plan.chunks[0].to_mapping()),
        (
            "PreprocessingChunkAssignment",
            preprocessing_chunk_assignment_from_mapping,
            plan.assignments[0].to_mapping(),
        ),
        ("PreprocessingWorkPlan", preprocessing_work_plan_from_mapping, plan.to_mapping()),
    )

    for record_name, loader, mapping in cases:
        mapping["unexpected"] = True
        with pytest.raises(ValueError, match=rf"Unknown {record_name} field\(s\): unexpected"):
            loader(mapping)


def test_every_preprocessing_record_loader_rejects_unsupported_versions(tmp_path: Path) -> None:
    plan = _small_plan(tmp_path)
    cases = (
        ("PreprocessingFastaRecord", preprocessing_fasta_record_from_mapping, plan.input.records[0].to_mapping()),
        ("PreprocessingInput", preprocessing_input_from_mapping, plan.input.to_mapping()),
        ("PreprocessingPlanOptions", preprocessing_plan_options_from_mapping, plan.options.to_mapping()),
        ("PreprocessingTranche", preprocessing_tranche_from_mapping, plan.tranches[0].to_mapping()),
        ("PreprocessingChunk", preprocessing_chunk_from_mapping, plan.chunks[0].to_mapping()),
        (
            "PreprocessingChunkAssignment",
            preprocessing_chunk_assignment_from_mapping,
            plan.assignments[0].to_mapping(),
        ),
        ("PreprocessingWorkPlan", preprocessing_work_plan_from_mapping, plan.to_mapping()),
    )

    for record_name, loader, mapping in cases:
        mapping["schema_version"] = 2
        with pytest.raises(
            ValueError,
            match=rf"Unsupported {record_name} schema_version 2; supported versions: 1",
        ):
            loader(mapping)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"requested_tranches": 0}, "requested_tranches"),
        ({"requested_tranches": 101}, "requested_tranches"),
        ({"records_per_chunk": 0}, "records_per_chunk"),
        ({"nodes": 0}, "nodes"),
        ({"gpus_per_node": 0}, "gpus_per_node"),
    ],
)
def test_planning_options_reject_impossible_worker_and_partition_shapes(
    options: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PreprocessingPlanOptions(**options)  # type: ignore[arg-type]


def test_contract_import_does_not_load_runtime_or_control_modules() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import bspp.orchestration.contract.preprocessing; "
                "assert not any(name.startswith(('bspp.orchestration.runtime', "
                "'bspp.orchestration.control')) for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_planning_is_non_executing_and_does_not_mutate_the_declared_input(tmp_path: Path) -> None:
    source = tmp_path / "proteins.fa"
    contents = ">alpha\nAAA\n"
    source.write_text(contents)

    plan_preprocessing_fasta(source, PreprocessingPlanOptions())

    assert source.read_text() == contents
    assert tuple(path.name for path in tmp_path.iterdir()) == ("proteins.fa",)


def _small_plan(tmp_path: Path) -> PreprocessingWorkPlan:
    source = tmp_path / "small.fa"
    source.write_text(">alpha\nAAA\n")
    return plan_preprocessing_fasta(source, PreprocessingPlanOptions())
