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

"""Focused tests for the strict chain-manifest reader and leaked-homodimer classifier."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from bspp.orchestration.runtime.folding.execution.chain_manifest import (
    REQUIRED_COLUMNS,
    ChainManifest,
    ChainManifestRow,
    ambiguous_chain_manifest_metadata,
    classify_target,
    parse_chain_manifest,
    resolve_tool_used,
)

HEADER = "model_entity_id,entity_id,chain_id,uniprot_ac\n"
_T = VALID_TOOL_USED[0]


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text(HEADER + body)
    return path


def _row(model_entity_id: str, entity_id: str, chain_id: str, uniprot_ac: str) -> ChainManifestRow:
    return ChainManifestRow(model_entity_id, entity_id, chain_id, uniprot_ac)


@pytest.mark.parametrize(
    "content",
    [
        # Missing the trailing uniprot_ac column.
        "model_entity_id,entity_id,chain_id\nAF_1_AF_2,e1,A\n",
        # Unknown/extra column beyond the four named ones.
        "model_entity_id,entity_id,chain_id,uniprot_ac,extra\nAF_1_AF_2,e1,A,P1,x\n",
        # Reordered columns.
        "entity_id,chain_id,uniprot_ac,model_entity_id\ne1,A,P1,AF_1_AF_2\n",
        # Empty file: no header row at all.
        "",
    ],
)
def test_strict_header_rejects(content: str, tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        parse_chain_manifest(path)


def test_exact_header_parses(tmp_path: Path) -> None:
    manifest = parse_chain_manifest(_write(tmp_path, "AF_1_AF_2,e1,A,P1\n"))
    assert len(manifest.rows) == 1
    assert manifest.rows[0].uniprot_ac == "P1"


@pytest.mark.parametrize(
    "row",
    [
        ",e1,A,P1",  # blank model_entity_id
        "AF_1_AF_2,,A,P1",  # blank entity_id
        "AF_1_AF_2,e1,,P1",  # blank chain_id
        "AF_1_AF_2,e1,A,",  # blank uniprot_ac
        "   ,e1,A,P1",  # whitespace model_entity_id
        "AF_1_AF_2,   ,A,P1",  # whitespace entity_id
        "AF_1_AF_2,e1,   ,P1",  # whitespace chain_id
        "AF_1_AF_2,e1,A,   ",  # whitespace uniprot_ac
        "AF_1_AF_2,e1,A",  # short row missing the trailing uniprot_ac
    ],
)
def test_blank_field_rejected(row: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        parse_chain_manifest(_write(tmp_path, row + "\n"))


@pytest.mark.parametrize(
    ("body", "line_number"),
    [
        # Five values under the valid four-column header: DictReader collects
        # the surplus under the None key.
        ("AF_1_AF_2,e1,A,P1,EXTRA\n", 2),
        # A valid row followed by a doubly-surplus row fails at its own line.
        ("AF_1_AF_2,e1,A,P1\nAF_3_AF_4,e2,A,P2,x,y\n", 3),
    ],
)
def test_surplus_fields_rejected(body: str, line_number: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=rf"line {line_number} has surplus fields"):
        parse_chain_manifest(_write(tmp_path, body))


def test_duplicate_chain_row_parsed_as_ambiguous(tmp_path: Path) -> None:
    body = "AF_1_AF_2,e1,A,P1\nAF_1_AF_2,e1,A,P1\n"
    manifest = parse_chain_manifest(_write(tmp_path, body))
    assert len(manifest.rows) == 2
    assert classify_target(manifest, "AF_1_AF_2") == "ambiguous"


@pytest.mark.parametrize(
    "second_row",
    [
        "AF_1_AF_2,e2,A,P1",  # same chain, different entity_id
        "AF_1_AF_2,e1,A,P2",  # same chain, different uniprot_ac
    ],
)
def test_conflicting_chain_row_parsed_as_ambiguous(second_row: str, tmp_path: Path) -> None:
    body = f"AF_1_AF_2,e1,A,P1\n{second_row}\n"
    manifest = parse_chain_manifest(_write(tmp_path, body))
    assert len(manifest.rows) == 2
    assert classify_target(manifest, "AF_1_AF_2") == "ambiguous"


@pytest.mark.parametrize(
    ("rows", "target", "expected"),
    [
        ((_row("X", "e1", "A", "P1"), _row("X", "e2", "B", "P1")), "X", "leaked"),
        ((_row("X", "e1", "A", "P1"),), "X", "not_leaked"),
        ((_row("X", "e1", "A", "P1"), _row("X", "e2", "B", "P2")), "X", "not_leaked"),
        ((_row("X", "e1", "A", "P1"),), "Y", "ambiguous"),
        ((_row("X", "e1", "A", "P1"), _row("X", "e1", "A", "P1")), "X", "ambiguous"),
        ((_row("X", "e1", "A", "P1"), _row("X", "e2", "A", "P2")), "X", "ambiguous"),
    ],
)
def test_classify_target_matrix(rows: tuple[ChainManifestRow, ...], target: str, expected: str) -> None:
    assert classify_target(ChainManifest(rows), target) == expected


def test_resolve_tool_used_leaked_returns_tool() -> None:
    manifest = ChainManifest((_row("AF_1_AF_2", "e1", "A", "P1"), _row("AF_1_AF_2", "e2", "B", "P1")))
    assert resolve_tool_used(manifest, "AF_1_AF_2", tool_used=_T) == _T


def test_resolve_tool_used_not_leaked_returns_tool() -> None:
    manifest = ChainManifest((_row("AF_1_AF_2", "e1", "A", "P1"), _row("AF_1_AF_2", "e2", "B", "P2")))
    assert resolve_tool_used(manifest, "AF_1_AF_2", tool_used=_T) == _T


def test_resolve_tool_used_unresolved_returns_none() -> None:
    manifest = ChainManifest((_row("AF_1_AF_2", "e1", "A", "P1"),))
    assert resolve_tool_used(manifest, "AF_9_AF_9", tool_used=_T) is None


def test_resolve_tool_used_malformed_tool_rejected() -> None:
    manifest = ChainManifest((_row("AF_1_AF_2", "e1", "A", "P1"), _row("AF_1_AF_2", "e2", "B", "P1")))
    with pytest.raises(ValueError):
        resolve_tool_used(manifest, "AF_1_AF_2", tool_used="bogus")


def test_target_scoped_failure(tmp_path: Path) -> None:
    body = "AF_1_AF_2,e1,A,P1\nAF_1_AF_2,e2,B,P1\nAF_3_AF_4,e3,A,P1\nAF_3_AF_4,e4,A,P2\n"
    manifest = parse_chain_manifest(_write(tmp_path, body))
    assert classify_target(manifest, "AF_1_AF_2") == "leaked"
    assert classify_target(manifest, "AF_3_AF_4") == "ambiguous"
    assert resolve_tool_used(manifest, "AF_1_AF_2", tool_used=_T) == _T
    assert resolve_tool_used(manifest, "AF_3_AF_4", tool_used=_T) is None


def test_ambiguous_chain_manifest_metadata_is_fresh_and_exact() -> None:
    first = ambiguous_chain_manifest_metadata()
    second = ambiguous_chain_manifest_metadata()
    assert first == {"failed_closed": True, "failure": "ambiguous_chain_manifest"}
    assert first == second
    assert first is not second


def test_required_columns_is_frozen_tuple() -> None:
    assert REQUIRED_COLUMNS == ("model_entity_id", "entity_id", "chain_id", "uniprot_ac")
    assert isinstance(REQUIRED_COLUMNS, tuple)


@pytest.mark.parametrize(
    ("obj", "attr", "value"),
    [
        (ChainManifestRow("X", "e1", "A", "P1"), "uniprot_ac", "P2"),
        (ChainManifest((ChainManifestRow("X", "e1", "A", "P1"),)), "rows", ()),
    ],
)
def test_chain_manifest_shapes_are_frozen(obj: object, attr: str, value: object) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(obj, attr, value)


def test_parse_preserves_row_order(tmp_path: Path) -> None:
    body = "AF_1_AF_2,e1,A,P1\nAF_1_AF_2,e2,B,P2\nAF_1_AF_2,e3,C,P3\n"
    manifest = parse_chain_manifest(_write(tmp_path, body))
    assert [row.chain_id for row in manifest.rows] == ["A", "B", "C"]
