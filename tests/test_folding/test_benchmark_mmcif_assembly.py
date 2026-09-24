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

"""Tests for the mmCIF ``_atom_site`` C-alpha parser."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from bspp.orchestration.control.folding_benchmark.mmcif_assembly import (
    MmcifCaData,
    parse_mmcif_assembly,
)

# Canonical column order used in most fixtures.
_ATOM_SITE_HEADER = """\
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
"""


def _row(
    group: str = "ATOM",
    atom_id: str = "CA",
    alt_id: str = ".",
    comp_id: str = "ALA",
    asym_id: str = "A",
    seq_id: str = "1",
    ins_code: str = "?",
    x: str = "10.000",
    y: str = "20.000",
    z: str = "30.000",
) -> str:
    return f"{group} 1 C {atom_id} {alt_id} {comp_id} {asym_id} 1 {seq_id} {ins_code} {x} {y} {z} 1.00 0.00"


def _mmcif(header: str, *rows: str) -> str:
    return header + "\n".join(rows) + "\n"


def test_mmcif_single_chain_ca_parsing() -> None:
    text = _mmcif(
        _ATOM_SITE_HEADER,
        _row(comp_id="ALA", seq_id="1"),
        _row(comp_id="GLY", seq_id="2", x="11.000"),
    )
    data = parse_mmcif_assembly(text)
    assert set(data.ca_residues) == {"A"}
    assert data.ca_residues["A"] == (
        (1, "ALA", (10.0, 20.0, 30.0)),
        (2, "GLY", (11.0, 20.0, 30.0)),
    )


def test_mmcif_two_chain() -> None:
    text = _mmcif(
        _ATOM_SITE_HEADER,
        _row(comp_id="ALA", asym_id="A", seq_id="1"),
        _row(comp_id="GLY", asym_id="B", seq_id="1"),
    )
    data = parse_mmcif_assembly(text)
    assert set(data.ca_residues) == {"A", "B"}
    assert data.ca_residues["A"] == ((1, "ALA", (10.0, 20.0, 30.0)),)
    assert data.ca_residues["B"] == ((1, "GLY", (10.0, 20.0, 30.0)),)


def test_mmcif_multi_char_label_asym_id() -> None:
    """B1: Multi-character label_asym_id values are grouped correctly."""
    text = _mmcif(
        _ATOM_SITE_HEADER,
        _row(comp_id="ALA", asym_id="AA", seq_id="1"),
        _row(comp_id="GLY", asym_id="BB", seq_id="1"),
    )
    data = parse_mmcif_assembly(text)
    assert set(data.ca_residues) == {"AA", "BB"}
    assert data.ca_residues["AA"] == ((1, "ALA", (10.0, 20.0, 30.0)),)
    assert data.ca_residues["BB"] == ((1, "GLY", (10.0, 20.0, 30.0)),)


def test_mmcif_hetatm_excluded() -> None:
    """Decision C: HETATM rows (including MSE) are excluded."""
    text = _mmcif(
        _ATOM_SITE_HEADER,
        _row(group="ATOM", comp_id="ALA", seq_id="1"),
        _row(group="HETATM", comp_id="MSE", seq_id="2"),
    )
    data = parse_mmcif_assembly(text)
    assert data.ca_residues["A"] == ((1, "ALA", (10.0, 20.0, 30.0)),)


def test_mmcif_altloc_filter() -> None:
    """altloc B is excluded; "", ".", "A" are included."""
    text = _mmcif(
        _ATOM_SITE_HEADER,
        _row(comp_id="ALA", seq_id="1", alt_id="."),
        _row(comp_id="ALA", seq_id="2", alt_id="A"),
        _row(comp_id="GLY", seq_id="3", alt_id="B"),
    )
    data = parse_mmcif_assembly(text)
    assert data.ca_residues["A"] == (
        (1, "ALA", (10.0, 20.0, 30.0)),
        (2, "ALA", (10.0, 20.0, 30.0)),
    )


def test_mmcif_unknown_residue_skipped() -> None:
    """A CA row with label_comp_id=UNK is skipped."""
    text = _mmcif(
        _ATOM_SITE_HEADER,
        _row(comp_id="ALA", seq_id="1"),
        _row(comp_id="UNK", seq_id="2"),
    )
    data = parse_mmcif_assembly(text)
    assert data.ca_residues["A"] == ((1, "ALA", (10.0, 20.0, 30.0)),)


def test_mmcif_dedup_on_label_seq_id_ins_code() -> None:
    """Duplicate (label_seq_id, pdbx_PDB_ins_code) pairs are deduped."""
    text = _mmcif(
        _ATOM_SITE_HEADER,
        _row(comp_id="ALA", seq_id="1", ins_code="?"),
        _row(comp_id="ALA", seq_id="1", ins_code="?"),
    )
    data = parse_mmcif_assembly(text)
    assert data.ca_residues["A"] == ((1, "ALA", (10.0, 20.0, 30.0)),)


def test_mmcif_order_by_label_seq_id() -> None:
    """Residues ordered by label_seq_id numerically; "." and "?" sort last (N2)."""
    text = _mmcif(
        _ATOM_SITE_HEADER,
        _row(comp_id="GLY", seq_id="3"),
        _row(comp_id="ALA", seq_id="1"),
        _row(comp_id="SER", seq_id="."),
    )
    data = parse_mmcif_assembly(text)
    residues = data.ca_residues["A"]
    assert len(residues) == 3
    # seq_id 1 → ALA, seq_id 3 → GLY, seq_id "." → SER (sorts last)
    assert residues[0][1] == "ALA"
    assert residues[1][1] == "GLY"
    assert residues[2][1] == "SER"


def test_mmcif_question_mark_label_seq_id() -> None:
    """N2: label_seq_id="?" handled without error; sorts last."""
    text = _mmcif(
        _ATOM_SITE_HEADER,
        _row(comp_id="ALA", seq_id="1"),
        _row(comp_id="GLY", seq_id="?"),
    )
    data = parse_mmcif_assembly(text)
    residues = data.ca_residues["A"]
    assert len(residues) == 2
    assert residues[0][1] == "ALA"
    assert residues[1][1] == "GLY"


def test_mmcif_no_atom_site_rejected() -> None:
    """Empty/no _atom_site loop raises ValueError."""
    text = "data_test\n# some comment\n"
    with pytest.raises(ValueError, match="no _atom_site loop"):
        parse_mmcif_assembly(text)


def test_mmcif_dataclass_is_frozen() -> None:
    data = MmcifCaData(ca_residues={"A": ()})
    with pytest.raises(FrozenInstanceError):
        data.ca_residues = {}  # type: ignore[misc]


def test_mmcif_reordered_columns() -> None:
    """N1: Columns in non-canonical order; verify correct parsing."""
    header = """\
loop_
_atom_site.label_asym_id
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
"""
    # Row: A ATOM CA . ALA 1 ? 10.000 20.000 30.000
    row = "A ATOM CA . ALA 1 ? 10.000 20.000 30.000"
    data = parse_mmcif_assembly(header + row + "\n")
    assert data.ca_residues["A"] == ((1, "ALA", (10.0, 20.0, 30.0)),)


def test_mmcif_extra_intervening_columns() -> None:
    """N1: Extra unused columns interspersed; verify correct parsing."""
    header = """\
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.extra_column
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
"""
    # Row: ATOM 1 EXTRA C CA . ALA A 1 1 ? 10.000 20.000 30.000 1.00 0.00
    row = "ATOM 1 EXTRA C CA . ALA A 1 1 ? 10.000 20.000 30.000 1.00 0.00"
    data = parse_mmcif_assembly(header + row + "\n")
    assert data.ca_residues["A"] == ((1, "ALA", (10.0, 20.0, 30.0)),)


def test_mmcif_missing_required_column() -> None:
    """N1: Missing a needed column raises ValueError."""
    header = """\
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
"""
    row = "ATOM CA ALA A"
    with pytest.raises(ValueError, match="missing required column"):
        parse_mmcif_assembly(header + row + "\n")


def test_mmcif_trailing_loop_block() -> None:
    """N1: A _atom_site_anisotrop loop_ block after _atom_site data rows."""
    text = _mmcif(
        _ATOM_SITE_HEADER,
        _row(comp_id="ALA", seq_id="1"),
    )
    # Append a trailing loop block.
    text += "loop_\n_atom_site_anisotrop.id\n1\n"
    data = parse_mmcif_assembly(text)
    assert data.ca_residues["A"] == ((1, "ALA", (10.0, 20.0, 30.0)),)


def test_mmcif_first_model_filter() -> None:
    """Only the first model's rows are kept when pdbx_PDB_model_num is present."""
    header = """\
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.pdbx_PDB_model_num
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
"""
    rows = (
        "ATOM 1 CA ALA A 1 ? 1 10.000 20.000 30.000",
        "ATOM 2 CA GLY A 1 ? 2 99.000 99.000 99.000",
    )
    data = parse_mmcif_assembly(header + "\n".join(rows) + "\n")
    assert data.ca_residues["A"] == ((1, "ALA", (10.0, 20.0, 30.0)),)


def test_mmcif_label_alt_id_absent_keeps_all() -> None:
    """When label_alt_id is absent, all rows are kept."""
    header = """\
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
"""
    rows = (
        "ATOM 1 CA ALA A 1 ? 10.000 20.000 30.000",
        "ATOM 2 CA GLY A 2 ? 11.000 20.000 30.000",
    )
    data = parse_mmcif_assembly(header + "\n".join(rows) + "\n")
    assert data.ca_residues["A"] == (
        (1, "ALA", (10.0, 20.0, 30.0)),
        (2, "GLY", (11.0, 20.0, 30.0)),
    )
