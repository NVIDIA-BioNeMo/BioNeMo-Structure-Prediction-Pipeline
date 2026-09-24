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

from __future__ import annotations

import csv

import pytest

from bspp.orchestration.runtime.worker.manifest import (
    FilteredManifest,
    apply_heterodimer_id_rewrites,
    filter_heterodimer_manifest_rows,
    filter_homodimer_manifest_rows,
    filter_manifest_csv,
    persist_shard_manifest_parquet,
    plan_heterodimer_id_rewrites,
    plan_shard_manifest_persistence,
    read_heterodimer_id_rewrite_plan,
    write_shard_manifest_csv,
    write_shard_manifest_parquet,
)

FIELDNAMES = ("model_entity_id", "entity_id", "chain_id", "uniprot_ac", "source")


def test_filter_homodimer_manifest_rows_preserves_manifest_rows_and_exact_ids() -> None:
    rows = [
        {
            "model_entity_id": "AF-0000000000000002",
            "entity_id": "1",
            "chain_id": "A",
            "uniprot_ac": "P2",
            "source": "b",
        },
        {
            "model_entity_id": "AF-0000000000000001",
            "entity_id": "1",
            "chain_id": "A",
            "uniprot_ac": "P1",
            "source": "a",
        },
        {
            "model_entity_id": "AF_0000000000000001",
            "entity_id": "1",
            "chain_id": "A",
            "uniprot_ac": "PX",
            "source": "x",
        },
    ]

    manifest = filter_homodimer_manifest_rows(
        rows,
        ["AF-0000000000000001", "AF-0000000000000002"],
        fieldnames=FIELDNAMES,
    )

    assert manifest.row_count == 2
    assert manifest.matched_model_ids == ("AF-0000000000000002", "AF-0000000000000001")
    assert [row["source"] for row in manifest.rows] == ["b", "a"]


def test_filter_heterodimer_manifest_rows_expands_compounds_from_component_manifest() -> None:
    rows = [
        {"model_entity_id": "AF-1001", "entity_id": "1", "chain_id": "A", "uniprot_ac": "P01001", "source": "a"},
        {"model_entity_id": "AF-1002", "entity_id": "1", "chain_id": "A", "uniprot_ac": "P01002", "source": "b"},
        {"model_entity_id": "AF-1002", "entity_id": "2", "chain_id": "B", "uniprot_ac": "DUP", "source": "dup"},
        {"model_entity_id": "AF-9001", "entity_id": "1", "chain_id": "A", "uniprot_ac": "P09001", "source": "skip"},
    ]

    manifest = filter_heterodimer_manifest_rows(
        rows,
        ["AF_1001_AF_1002"],
        fieldnames=FIELDNAMES,
    )

    assert manifest.row_count == 2
    assert manifest.matched_model_ids == ("AF_1001_AF_1002",)
    assert manifest.rows == (
        {
            "model_entity_id": "AF_1001_AF_1002",
            "entity_id": "1",
            "chain_id": "A",
            "uniprot_ac": "P01001",
            "source": "a",
        },
        {
            "model_entity_id": "AF_1001_AF_1002",
            "entity_id": "2",
            "chain_id": "B",
            "uniprot_ac": "P01002",
            "source": "b",
        },
    )


def test_filter_heterodimer_manifest_rows_uses_legacy_digits_only_compounds() -> None:
    rows = [
        {"model_entity_id": "AF-1001", "entity_id": "1", "chain_id": "A", "uniprot_ac": "P01001", "source": "a"},
        {"model_entity_id": "AF-1002", "entity_id": "1", "chain_id": "A", "uniprot_ac": "P01002", "source": "b"},
    ]

    manifest = filter_heterodimer_manifest_rows(
        rows,
        ["AF_1001A_AF_1002"],
        fieldnames=FIELDNAMES,
    )

    assert manifest.row_count == 0
    assert manifest.matched_model_ids == ()


def test_heterodimer_id_rewrite_plan_maps_reversed_ac_pairs_and_swaps_chains() -> None:
    shard_manifest = FilteredManifest(
        fieldnames=FIELDNAMES,
        matched_model_ids=("AF_1001_AF_1002",),
        rows=(
            {
                "model_entity_id": "AF_1001_AF_1002",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "P01001",
                "source": "a",
            },
            {
                "model_entity_id": "AF_1001_AF_1002",
                "entity_id": "2",
                "chain_id": "B",
                "uniprot_ac": "P01002",
                "source": "b",
            },
        ),
    )
    heterodimer_id_rows = [
        {"model_entity_id": "AF-7777", "entity_id": "1", "chain_id": "A", "uniprot_ac": "P01002"},
        {"model_entity_id": "AF-7777", "entity_id": "2", "chain_id": "B", "uniprot_ac": "P01001"},
    ]

    rewrite_plan = plan_heterodimer_id_rewrites(shard_manifest, heterodimer_id_rows)
    rewritten = apply_heterodimer_id_rewrites(shard_manifest, rewrite_plan)

    assert rewrite_plan.rename_pairs == (("AF_1001_AF_1002", "AF-7777"),)
    assert rewrite_plan.swapped_model_ids == ("AF_1001_AF_1002",)
    assert rewritten.matched_model_ids == ("AF-7777",)
    assert [(row["model_entity_id"], row["entity_id"], row["chain_id"]) for row in rewritten.rows] == [
        ("AF-7777", "2", "B"),
        ("AF-7777", "1", "A"),
    ]


def test_heterodimer_id_rewrite_dedups_forward_and_swapped_compound_rows() -> None:
    """Forward + swapped compound IDs rename to one unified ID; duplicate chain rows dedup."""
    shard_manifest = FilteredManifest(
        fieldnames=FIELDNAMES,
        matched_model_ids=("AF_1001_AF_1002", "AF_1002_AF_1001"),
        rows=(
            {
                "model_entity_id": "AF_1001_AF_1002",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "P01001",
                "source": "a",
            },
            {
                "model_entity_id": "AF_1001_AF_1002",
                "entity_id": "2",
                "chain_id": "B",
                "uniprot_ac": "P01002",
                "source": "b",
            },
            {
                "model_entity_id": "AF_1002_AF_1001",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "P01002",
                "source": "c",
            },
            {
                "model_entity_id": "AF_1002_AF_1001",
                "entity_id": "2",
                "chain_id": "B",
                "uniprot_ac": "P01001",
                "source": "d",
            },
        ),
    )
    # The unified manifest lists the (P01002, P01001) pair, so the forward compound
    # AF_1001_AF_1002 (pair P01001,P01002) is matched in reversed order and swapped.
    heterodimer_id_rows = [
        {"model_entity_id": "AF-7777", "entity_id": "1", "chain_id": "A", "uniprot_ac": "P01002"},
        {"model_entity_id": "AF-7777", "entity_id": "2", "chain_id": "B", "uniprot_ac": "P01001"},
    ]

    rewrite_plan = plan_heterodimer_id_rewrites(shard_manifest, heterodimer_id_rows)
    rewritten = apply_heterodimer_id_rewrites(shard_manifest, rewrite_plan)

    # Both compounds rename to AF-7777; after the swap-flip their rows coincide,
    # so the result must carry exactly one row per (unified_id, entity, chain).
    assert rewritten.matched_model_ids == ("AF-7777",)
    assert [(row["model_entity_id"], row["entity_id"], row["chain_id"]) for row in rewritten.rows] == [
        ("AF-7777", "2", "B"),
        ("AF-7777", "1", "A"),
    ]
    assert rewritten.row_count == 2


def test_heterodimer_id_rewrite_plan_uses_stable_duplicate_pair_tiebreak() -> None:
    shard_manifest = FilteredManifest(
        fieldnames=FIELDNAMES,
        matched_model_ids=("AF_1001_AF_1002",),
        rows=(
            {
                "model_entity_id": "AF_1001_AF_1002",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "P01001",
                "source": "a",
            },
            {
                "model_entity_id": "AF_1001_AF_1002",
                "entity_id": "2",
                "chain_id": "B",
                "uniprot_ac": "P01002",
                "source": "b",
            },
        ),
    )
    heterodimer_id_rows = [
        {"model_entity_id": "AF-0000000204896696", "entity_id": "1", "chain_id": "A", "uniprot_ac": "P01001"},
        {"model_entity_id": "AF-0000000204896696", "entity_id": "2", "chain_id": "B", "uniprot_ac": "P01002"},
        {"model_entity_id": "AF-0000000203841184", "entity_id": "1", "chain_id": "A", "uniprot_ac": "P01001"},
        {"model_entity_id": "AF-0000000203841184", "entity_id": "2", "chain_id": "B", "uniprot_ac": "P01002"},
    ]

    rewrite_plan = plan_heterodimer_id_rewrites(shard_manifest, heterodimer_id_rows)

    assert rewrite_plan.rename_pairs == (("AF_1001_AF_1002", "AF-0000000204896696"),)
    assert rewrite_plan.swapped_model_ids == ()


def test_heterodimer_id_rewrite_plan_uses_stable_duplicate_reversed_pair_tiebreak() -> None:
    shard_manifest = FilteredManifest(
        fieldnames=FIELDNAMES,
        matched_model_ids=("AF_1001_AF_1002",),
        rows=(
            {
                "model_entity_id": "AF_1001_AF_1002",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "Q5BJA5",
                "source": "a",
            },
            {
                "model_entity_id": "AF_1001_AF_1002",
                "entity_id": "2",
                "chain_id": "B",
                "uniprot_ac": "F1QEB4",
                "source": "b",
            },
        ),
    )
    heterodimer_id_rows = [
        {"model_entity_id": "AF-0000000204896696", "entity_id": "1", "chain_id": "A", "uniprot_ac": "F1QEB4"},
        {"model_entity_id": "AF-0000000204896696", "entity_id": "2", "chain_id": "B", "uniprot_ac": "Q5BJA5"},
        {"model_entity_id": "AF-0000000203841184", "entity_id": "1", "chain_id": "A", "uniprot_ac": "F1QEB4"},
        {"model_entity_id": "AF-0000000203841184", "entity_id": "2", "chain_id": "B", "uniprot_ac": "Q5BJA5"},
    ]

    rewrite_plan = plan_heterodimer_id_rewrites(shard_manifest, heterodimer_id_rows)

    assert rewrite_plan.rename_pairs == (("AF_1001_AF_1002", "AF-0000000204896696"),)
    assert rewrite_plan.swapped_model_ids == ("AF_1001_AF_1002",)


def test_read_heterodimer_id_rewrite_plan_reads_csv_manifest(tmp_path) -> None:
    shard_manifest = FilteredManifest(
        fieldnames=FIELDNAMES,
        matched_model_ids=("AF_1001_AF_1002",),
        rows=(
            {
                "model_entity_id": "AF_1001_AF_1002",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "P01001",
                "source": "a",
            },
            {
                "model_entity_id": "AF_1001_AF_1002",
                "entity_id": "2",
                "chain_id": "B",
                "uniprot_ac": "P01002",
                "source": "b",
            },
        ),
    )
    heterodimer_id_manifest = tmp_path / "heterodimer_ids.csv"
    _write_manifest_csv(
        heterodimer_id_manifest,
        [
            {"model_entity_id": "AF-7777", "entity_id": "1", "chain_id": "A", "uniprot_ac": "P01001", "source": ""},
            {"model_entity_id": "AF-7777", "entity_id": "2", "chain_id": "B", "uniprot_ac": "P01002", "source": ""},
        ],
    )

    rewrite_plan = read_heterodimer_id_rewrite_plan(shard_manifest, heterodimer_id_manifest)

    assert rewrite_plan.rename_pairs == (("AF_1001_AF_1002", "AF-7777"),)
    assert rewrite_plan.swapped_model_ids == ()


def test_plan_shard_manifest_persistence_does_not_write_until_explicit_write(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    manifest = FilteredManifest(
        fieldnames=FIELDNAMES,
        matched_model_ids=("AF-0000000000000001",),
        rows=(
            {
                "model_entity_id": "AF-0000000000000001",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "P00001",
                "source": "a",
            },
        ),
    )
    output_path = tmp_path / "shard_0" / "shard_manifest.parquet"

    plan = plan_shard_manifest_persistence(
        manifest,
        shard_id=7,
        dataset_tag="wp8c",
        output_path=output_path,
    )

    assert not output_path.exists()

    result = write_shard_manifest_parquet(plan)
    table = pq.read_table(result.output_path)

    assert result.row_count == 1
    assert table.column("model_entity_id").to_pylist() == ["AF-0000000000000001"]
    assert table.column("entity_id").to_pylist() == [1]
    assert table.column("shard_id").to_pylist() == [7]
    assert table.column("dataset_tag").to_pylist() == ["wp8c"]
    assert table.schema.field("shard_id").type == pa.int32()
    assert table.schema.field("dataset_tag").type == pa.string()


def test_write_shard_manifest_parquet_deduplicates_full_rows_before_metadata(tmp_path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")

    manifest = FilteredManifest(
        fieldnames=FIELDNAMES,
        matched_model_ids=("AF-0000000000000001",),
        rows=(
            {
                "model_entity_id": "AF-0000000000000001",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "P00001",
                "source": "dup",
            },
            {
                "model_entity_id": "AF-0000000000000001",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "P00001",
                "source": "dup",
            },
        ),
    )

    result = write_shard_manifest_parquet(
        plan_shard_manifest_persistence(
            manifest,
            shard_id=5,
            dataset_tag="dedup",
            output_path=tmp_path / "shard_manifest.parquet",
        ),
    )

    table = pq.read_table(result.output_path)
    assert result.row_count == 1
    assert table.num_rows == 1
    assert table.column("source").to_pylist() == ["dup"]
    assert table.column("shard_id").to_pylist() == [5]
    assert table.column("dataset_tag").to_pylist() == ["dedup"]


def test_write_shard_manifest_parquet_keeps_distinct_rows_after_dedup(tmp_path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")

    manifest = FilteredManifest(
        fieldnames=FIELDNAMES,
        matched_model_ids=("AF-0000000000000001", "AF-0000000000000002"),
        rows=(
            {
                "model_entity_id": "AF-0000000000000001",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "P00001",
                "source": "a",
            },
            {
                "model_entity_id": "AF-0000000000000002",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "P00002",
                "source": "b",
            },
        ),
    )

    result = write_shard_manifest_parquet(
        plan_shard_manifest_persistence(
            manifest,
            shard_id=6,
            dataset_tag="distinct",
            output_path=tmp_path / "shard_manifest.parquet",
        ),
    )

    table = pq.read_table(result.output_path)
    rows = sorted(zip(table.column("model_entity_id").to_pylist(), table.column("source").to_pylist(), strict=True))
    assert result.row_count == 2
    assert rows == [("AF-0000000000000001", "a"), ("AF-0000000000000002", "b")]
    assert table.column("shard_id").to_pylist() == [6, 6]
    assert table.column("dataset_tag").to_pylist() == ["distinct", "distinct"]


def test_filter_csv_to_shard_manifest_parquet_smoke(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    manifest_csv = tmp_path / "manifest.csv"
    _write_manifest_csv(
        manifest_csv,
        [
            {
                "model_entity_id": "AF-0000000000000001",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "P00001",
                "source": "keep",
            },
            {
                "model_entity_id": "AF-0000000000000002",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "P00002",
                "source": "drop",
            },
        ],
    )

    manifest = filter_manifest_csv(
        manifest_csv,
        ["AF-0000000000000001"],
    )
    result = write_shard_manifest_parquet(
        plan_shard_manifest_persistence(
            manifest,
            shard_id=11,
            dataset_tag="wp8c-e2e",
            output_path=tmp_path / "shard_11" / "shard_manifest.parquet",
        ),
    )

    table = pq.read_table(result.output_path)
    assert result.row_count == 1
    assert table.column("model_entity_id").to_pylist() == ["AF-0000000000000001"]
    assert table.column("source").to_pylist() == ["keep"]
    assert table.column("shard_id").to_pylist() == [11]
    assert table.column("dataset_tag").to_pylist() == ["wp8c-e2e"]
    assert table.schema.field("shard_id").type == pa.int32()
    assert table.schema.field("dataset_tag").type == pa.string()


def test_persist_shard_manifest_parquet_writes_empty_manifest_with_metadata_schema(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    output_path = tmp_path / "shard_manifest.parquet"
    manifest = FilteredManifest(fieldnames=FIELDNAMES, rows=(), matched_model_ids=())

    result = persist_shard_manifest_parquet(
        manifest,
        shard_id=3,
        dataset_tag="empty",
        output_path=output_path,
    )

    table = pq.read_table(output_path)
    assert result.row_count == 0
    assert table.num_rows == 0
    assert table.schema.names == [*FIELDNAMES, "shard_id", "dataset_tag"]
    assert table.schema.field("shard_id").type == pa.int32()
    assert table.schema.field("dataset_tag").type == pa.string()


def test_write_shard_manifest_csv_preserves_intermediate_compatibility_file(tmp_path) -> None:
    manifest = FilteredManifest(
        fieldnames=FIELDNAMES,
        rows=(
            {
                "model_entity_id": "AF-0000000000000001",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_ac": "P00001",
                "source": "keep",
            },
        ),
        matched_model_ids=("AF-0000000000000001",),
    )

    result = write_shard_manifest_csv(manifest, tmp_path / "shard_0" / "shard_manifest.csv")

    assert result.row_count == 1
    assert result.output_path.read_text() == (
        "model_entity_id,entity_id,chain_id,uniprot_ac,source\nAF-0000000000000001,1,A,P00001,keep\n"
    )


def _write_manifest_csv(path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
