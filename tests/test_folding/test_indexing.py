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

"""Focused contract tests for the folding A3M indexing seam."""

from __future__ import annotations

import ast
import os
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from bspp.orchestration.contract.folding_index import (
    FoldingIndex,
    FoldingIndexRecord,
    folding_index_from_mapping,
    folding_index_record_from_mapping,
)
from bspp.orchestration.contract.versioning import UnsupportedSchemaVersionError
from bspp.orchestration.runtime.folding import a3m as a3m_module
from bspp.orchestration.runtime.folding.batch_info import normalize_batch_info
from bspp.orchestration.runtime.folding.indexing import index_folding_a3ms


def _write_a3m(path: Path, header: str, query: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{header}\n>query\n{query}\n>hit\n{query}\n")
    return path


def _record_mapping(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "source_ordinal": 0,
        "protein_id": "model-a",
        "msa_path": "/input/model-a.a3m",
        "query_sequence": "ABCDE",
        "sequence_length": 5,
        "chain_lengths": [5],
        "chain_count": 1,
        "chain_cardinalities": [1],
        "msa_depth": 1,
        "total_length": 5,
    }
    payload.update(overrides)
    return payload


def test_index_folding_a3ms_parses_supported_headers(tmp_path: Path) -> None:
    monomer = _write_a3m(tmp_path / "a-monomer.a3m", "#5\t1", "ABCDE")
    homodimer = _write_a3m(tmp_path / "b-homodimer.a3m", "#4\t2", "WXYZ")
    heterodimer = _write_a3m(tmp_path / "c-heterodimer.a3m", "#3,2\t1,1", "ABCDE")

    result = index_folding_a3ms((tmp_path,))

    assert result.records == (
        FoldingIndexRecord(
            source_ordinal=0,
            protein_id="a-monomer",
            msa_path=str(monomer),
            query_sequence="ABCDE",
            sequence_length=5,
            chain_lengths=(5,),
            chain_count=1,
            chain_cardinalities=(1,),
            msa_depth=1,
            total_length=5,
        ),
        FoldingIndexRecord(
            source_ordinal=1,
            protein_id="b-homodimer",
            msa_path=str(homodimer),
            query_sequence="WXYZ",
            sequence_length=4,
            chain_lengths=(4,),
            chain_count=2,
            chain_cardinalities=(2,),
            msa_depth=1,
            total_length=8,
        ),
        FoldingIndexRecord(
            source_ordinal=2,
            protein_id="c-heterodimer",
            msa_path=str(heterodimer),
            query_sequence="ABCDE",
            sequence_length=5,
            chain_lengths=(3, 2),
            chain_count=2,
            chain_cardinalities=(1, 1),
            msa_depth=1,
            total_length=5,
        ),
    )


def test_index_folding_a3ms_preserves_baseline_heterodimer_cardinality_semantics(tmp_path: Path) -> None:
    heterodimer = _write_a3m(tmp_path / "heterodimer.a3m", "#3,2\t2,1", "ABCDE")

    record = index_folding_a3ms((heterodimer,)).records[0]

    assert record.chain_lengths == (3, 2)
    assert record.chain_cardinalities == (2, 1)
    assert record.chain_count == 2
    assert record.total_length == 5


def test_index_folding_a3ms_enumerates_flat_nested_and_mixed_inputs_deterministically(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    nested = _write_a3m(first_root / "nested" / "z.a3m", "#2 1", "AA")
    flat = _write_a3m(first_root / "a.a3m", "#4 1", "AAAA")
    declared_file = _write_a3m(second_root / "m.a3m", "#3 1", "AAA")
    (first_root / "ignore.txt").write_text("not an A3M")

    first = index_folding_a3ms((first_root, declared_file))
    second = index_folding_a3ms((first_root, declared_file))

    assert first == second
    assert tuple(record.msa_path for record in first.records) == (str(flat), str(nested), str(declared_file))
    assert tuple(record.source_ordinal for record in first.records) == (0, 1, 2)
    assert tuple(record.protein_id for record in first.length_sorted_records) == ("z", "m", "a")

    sorted_primary = index_folding_a3ms((first_root, declared_file), sort_by_length=True)
    assert sorted_primary.records == sorted_primary.length_sorted_records
    assert tuple(record.protein_id for record in sorted_primary.records) == ("z", "m", "a")


def test_index_folding_a3ms_follows_symlinked_directories_without_following_cycles(tmp_path: Path) -> None:
    declared_root = tmp_path / "declared"
    declared_root.mkdir()
    external_shard = tmp_path / "external-shard"
    linked = _write_a3m(external_shard / "linked.a3m", "#3 1", "AAA")
    (declared_root / "shard-link").symlink_to(external_shard, target_is_directory=True)
    (external_shard / "cycle").symlink_to(declared_root, target_is_directory=True)

    result = index_folding_a3ms((declared_root,))

    assert len(result.records) == 1
    assert result.records[0].msa_path == str(declared_root / "shard-link" / linked.name)


def test_index_folding_a3ms_rejects_dangling_a3m_symlink_in_declared_directory(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "broken.a3m").symlink_to(root / "missing.a3m")

    with pytest.raises(ValueError, match=r"broken\.a3m.*not a readable file"):
        index_folding_a3ms((root,))


def test_index_folding_a3ms_fails_closed_when_scandir_fails_after_partial_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    _write_a3m(root / "visible.a3m", "#2 1", "AA")
    blocked = root / "blocked"
    blocked.mkdir()
    _write_a3m(blocked / "hidden.a3m", "#2 1", "AA")
    real_scandir = os.scandir

    def injected_scandir(path: Path) -> object:
        if path == blocked:
            raise PermissionError("injected scan denial")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", injected_scandir)

    with pytest.raises(ValueError, match="Cannot scan declared A3M directory"):
        index_folding_a3ms((root,))


def test_normalize_batch_info_supports_current_and_historical_aliases(tmp_path: Path) -> None:
    current = _write_a3m(tmp_path / "current.a3m", "#3 1", "AAA")
    generated = _write_a3m(tmp_path / "generated.a3m", "#4 2", "AAAA")
    historical = _write_a3m(tmp_path / "historical.a3m", "#2 2", "AA")

    result = normalize_batch_info(
        (
            {"protein_id": "explicit-current", "msa_path": str(current), "seq_length": 3, "msa_depth": 7},
            {
                "protein_id": "explicit-generated",
                "a3m_path": str(generated),
                "sequence": "AAAA",
                "seq_length": 4,
                "chain_count": 2,
                "msa_depth": 2,
                "total_length": 8,
            },
            {"protein_id": "explicit-historical", "path": str(historical), "total_length": 4},
        )
    )

    assert tuple(record.protein_id for record in result.records) == (
        "explicit-current",
        "explicit-generated",
        "explicit-historical",
    )
    assert tuple(record.sequence_length for record in result.records) == (3, 4, 4)
    assert tuple(record.total_length for record in result.records) == (3, 8, 4)
    assert tuple(record.msa_depth for record in result.records) == (7, 2, 1)
    assert result.records[2].query_sequence == "AA"
    assert result.records[2].chain_cardinalities == (2,)


def test_normalize_batch_info_flattens_retained_list_paths_once_in_declared_order(tmp_path: Path) -> None:
    second = _write_a3m(tmp_path / "second.a3m", "#3,2 1,1", "ABCDE")
    first = _write_a3m(tmp_path / "first.a3m", "#2 2", "AA")

    result = normalize_batch_info(
        (
            {
                "path": [str(second), str(first)],
                "seq_length": 97,
                "sequence": "ignored-for-retained-list",
                "chain_count": 99,
                "msa_depth": 42,
            },
        )
    )

    assert tuple(record.msa_path for record in result.records) == (str(second), str(first))
    assert tuple(record.protein_id for record in result.records) == ("second", "first")
    assert tuple(record.source_ordinal for record in result.records) == (0, 1)
    assert tuple(record.sequence_length for record in result.records) == (97, 97)
    assert tuple(record.query_sequence for record in result.records) == ("ABCDE", "AA")
    assert tuple(record.chain_lengths for record in result.records) == ((3, 2), (2,))
    assert tuple(record.chain_count for record in result.records) == (2, 2)
    assert tuple(record.msa_depth for record in result.records) == (1, 1)


def test_normalize_batch_info_accepts_array_shaped_historical_values(tmp_path: Path) -> None:
    first = _write_a3m(tmp_path / "first.a3m", "#2 1", "AA")
    second = _write_a3m(tmp_path / "second.a3m", "#3 1", "AAA")

    result = normalize_batch_info(
        (
            {
                "path": np.asarray([str(first), str(second)], dtype=object),
                "seq_length": np.int64(97),
            },
        )
    )

    assert tuple(record.protein_id for record in result.records) == ("first", "second")
    assert tuple(record.sequence_length for record in result.records) == (97, 97)


def test_normalize_batch_info_does_not_mask_a_partially_invalid_preferred_alias(tmp_path: Path) -> None:
    fallback = _write_a3m(tmp_path / "fallback.a3m", "#3 1", "AAA")

    with pytest.raises(ValueError, match="msa_path"):
        normalize_batch_info(
            (
                {
                    "protein_id": "must-not-fallback",
                    "msa_path": [str(fallback), ""],
                    "path": str(fallback),
                    "seq_length": 3,
                },
            )
        )


def test_normalize_batch_info_uses_baseline_path_alias_precedence(tmp_path: Path) -> None:
    msa = _write_a3m(tmp_path / "msa.a3m", "#3 1", "AAA")
    generated = _write_a3m(tmp_path / "generated.a3m", "#3 1", "AAA")
    historical = _write_a3m(tmp_path / "historical.a3m", "#3 1", "AAA")

    result = normalize_batch_info(
        (
            {
                "protein_id": "msa-wins",
                "msa_path": str(msa),
                "a3m_path": str(generated),
                "path": str(historical),
                "seq_length": 3,
            },
            {
                "protein_id": "generated-fallback",
                "msa_path": None,
                "a3m_path": str(generated),
                "path": str(historical),
                "seq_length": 3,
            },
            {
                "protein_id": "historical-fallback",
                "msa_path": [],
                "a3m_path": None,
                "path": str(historical),
                "seq_length": 3,
            },
        )
    )

    assert tuple(record.msa_path for record in result.records) == (str(msa), str(generated), str(historical))


def test_normalize_batch_info_sorts_primary_view_and_parses_heterodimer(tmp_path: Path) -> None:
    long = _write_a3m(tmp_path / "long.a3m", "#3,2 1,1", "ABCDE")
    short = _write_a3m(tmp_path / "short.a3m", "#2 1", "AA")

    result = normalize_batch_info(
        (
            {"protein_id": "long", "msa_path": str(long), "seq_length": 5, "total_length": 5},
            {"protein_id": "short", "msa_path": str(short), "seq_length": 2},
        ),
        sort_by_length=True,
    )

    assert result.records == result.length_sorted_records
    assert tuple(record.protein_id for record in result.records) == ("short", "long")
    assert result.records[1].chain_lengths == (3, 2)
    assert result.records[1].chain_cardinalities == (1, 1)


def test_parse_a3m_wraps_injected_read_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_a3m(tmp_path / "unreadable.a3m", "#3 1", "AAA")

    def injected_read_error(_: Path) -> tuple[str, str, str]:
        raise PermissionError("injected read denial")

    monkeypatch.setattr(a3m_module, "_read_a3m_prefix", injected_read_error)

    with pytest.raises(ValueError, match="Cannot read A3M input"):
        index_folding_a3ms((path,))


@pytest.mark.parametrize(
    ("writer", "match"),
    [
        (lambda path: path.write_text("not-a-header\n>query\nAAAA\n"), "header"),
        (lambda path: path.write_text("#4 1\n>query\nAAA\n"), "length"),
        (lambda path: path.write_text("#4 0\n>query\nAAAA\n"), "cardinality"),
        (lambda path: path.write_text("#4 1\nquery\nAAAA\n"), "query header"),
        (lambda path: path.write_text("#3,2,4 1,1,1\n>query\nABCDEFGHI\n"), "unsupported or malformed header"),
        (lambda path: path.write_bytes(b"#4 1\n>query\n\xff\n"), "Cannot read A3M input"),
    ],
)
def test_index_folding_a3ms_rejects_malformed_or_impossible_a3ms(
    tmp_path: Path,
    writer: Callable[[Path], object],
    match: str,
) -> None:
    path = tmp_path / "bad.a3m"
    writer(path)

    with pytest.raises(ValueError, match=match):
        index_folding_a3ms((path,))


def test_index_folding_a3ms_rejects_missing_empty_duplicate_and_overlapping_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        index_folding_a3ms((tmp_path / "missing.a3m",))

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match=r"contains no \.a3m"):
        index_folding_a3ms((empty,))

    wrong_kind = tmp_path / "not-a3m.txt"
    wrong_kind.write_text("not an A3M")
    with pytest.raises(ValueError, match=r"not an \.a3m file"):
        index_folding_a3ms((wrong_kind,))

    left = _write_a3m(tmp_path / "left" / "same.a3m", "#2 1", "AA")
    right = _write_a3m(tmp_path / "right" / "same.a3m", "#2 1", "AA")
    with pytest.raises(ValueError, match="Duplicate protein_id"):
        index_folding_a3ms((left, right))

    unique = _write_a3m(tmp_path / "unique.a3m", "#2 1", "AA")
    with pytest.raises(ValueError, match="declared more than once"):
        index_folding_a3ms((tmp_path, unique))


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ({"msa_path": "/missing.a3m", "seq_length": 3}, "protein_id"),
        ({"protein_id": "x", "msa_path": "/missing.a3m"}, "seq_length or total_length"),
        ({"protein_id": "x", "protein_path": "/missing.a3m", "seq_length": 3}, "unknown fields"),
        ({"protein_id": "x", "msa_path": "/missing.a3m", "seq_len": 3}, "unknown fields"),
        ({"protein_id": "x", "msa_path": [], "seq_length": 3}, "usable path column"),
        ({"protein_id": "x", "msa_path": "/missing.a3m", "seq_length": True}, "positive integer"),
        ({"protein_id": "x", "msa_path": "/missing.a3m", "seq_length": 3, "msa_depth": 0}, "msa_depth"),
    ],
)
def test_normalize_batch_info_rejects_missing_unsupported_or_invalid_columns(
    row: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_batch_info((row,))


def test_normalize_batch_info_rejects_unreadable_length_mismatch_and_duplicate_identity(tmp_path: Path) -> None:
    path = _write_a3m(tmp_path / "model.a3m", "#3 1", "AAA")
    with pytest.raises(ValueError, match="protein_id"):
        normalize_batch_info(({"msa_path": str(path), "seq_length": 3},))

    with pytest.raises(ValueError, match="does not exist"):
        normalize_batch_info(({"protein_id": "x", "msa_path": str(tmp_path / "missing.a3m"), "seq_length": 3},))

    with pytest.raises(ValueError, match="seq_length"):
        normalize_batch_info(({"protein_id": "x", "msa_path": str(path), "seq_length": 4},))

    duplicate_identity_path = _write_a3m(tmp_path / "other.a3m", "#3 1", "AAA")
    with pytest.raises(ValueError, match="Duplicate protein_id"):
        normalize_batch_info(
            (
                {"protein_id": "duplicate", "msa_path": str(path), "seq_length": 3},
                {"protein_id": "duplicate", "msa_path": str(duplicate_identity_path), "seq_length": 3},
            )
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"total_length": 4}, "total_length"),
        ({"sequence": "AAT"}, "sequence"),
        ({"chain_count": 2}, "chain_count"),
    ],
)
def test_normalize_batch_info_rejects_every_scalar_declared_metadata_mismatch(
    tmp_path: Path,
    overrides: dict[str, object],
    match: str,
) -> None:
    path = _write_a3m(tmp_path / "model.a3m", "#3 1", "AAA")
    row: dict[str, object] = {
        "protein_id": "model",
        "msa_path": str(path),
        "seq_length": 3,
    }
    row.update(overrides)

    with pytest.raises(ValueError, match=match):
        normalize_batch_info((row,))


def test_normalize_batch_info_rejects_duplicate_physical_path_with_distinct_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_a3m(tmp_path / "same.a3m", "#3 1", "AAA")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="same physical A3M"):
        normalize_batch_info(
            (
                {"protein_id": "first", "msa_path": path.name, "seq_length": 3},
                {"protein_id": "second", "msa_path": str(path), "seq_length": 3},
            )
        )


def test_folding_index_contract_loaders_are_versioned_immutable_and_fail_closed() -> None:
    record = folding_index_record_from_mapping(_record_mapping())
    index = FoldingIndex(records=(record,), length_sorted_records=(record,))
    restored = folding_index_from_mapping(index.to_mapping())

    assert restored == index
    assert record.to_mapping() == _record_mapping()
    with pytest.raises(FrozenInstanceError):
        record.protein_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        index.records = ()  # type: ignore[misc]

    with pytest.raises(UnsupportedSchemaVersionError):
        folding_index_record_from_mapping(_record_mapping(schema_version=2))
    with pytest.raises(UnsupportedSchemaVersionError):
        folding_index_from_mapping({**index.to_mapping(), "schema_version": 2})
    with pytest.raises(ValueError, match="unknown fields"):
        folding_index_record_from_mapping(_record_mapping(extra="no"))
    with pytest.raises(ValueError, match="unknown fields"):
        folding_index_from_mapping({**index.to_mapping(), "extra": "no"})
    with pytest.raises(ValueError, match="contiguous"):
        folding_index_record_from_mapping(_record_mapping(source_ordinal=-1))
    with pytest.raises(ValueError, match="total_length"):
        folding_index_record_from_mapping(_record_mapping(total_length=4))

    short = _record_mapping(
        source_ordinal=1,
        protein_id="model-b",
        msa_path="/input/model-b.a3m",
        query_sequence="AB",
        sequence_length=2,
        chain_lengths=[2],
        total_length=2,
    )
    with pytest.raises(ValueError, match="length_sorted_records"):
        folding_index_from_mapping(
            {
                "schema_version": 1,
                "records": [_record_mapping(), short],
                "length_sorted_records": [_record_mapping(), short],
            }
        )


def test_folding_index_contract_module_remains_dependency_light() -> None:
    import bspp.orchestration.contract.folding_index as module

    source = Path(module.__file__).read_text()
    imports = {
        alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(name.startswith("bspp.orchestration.runtime") for name in imports)
    assert not imports & {"pandas", "pyarrow", "numpy", "cudf"}
