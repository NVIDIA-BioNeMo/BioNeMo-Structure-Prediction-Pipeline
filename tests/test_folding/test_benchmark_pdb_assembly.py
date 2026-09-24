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

"""Tests for the folding benchmark PDB assembly parser."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from bspp.orchestration.control.folding_benchmark.pdb_assembly import (
    ParsedAssembly,
    normalize_ca_residue_numbers,
    parse_pdb_assembly,
)


def test_seqres_and_ca_parsing() -> None:
    text = (
        "SEQRES   1 A   1  ALA\n"
        "SEQRES   1 A   2  GLY\n"
        "ATOM      1  CA  ALA A   1      11.104  13.207  10.000  1.00 20.00           C\n"
        "ATOM      2  CA  GLY A   2      12.000  14.000  11.000  1.00 20.00           C\n"
    )
    parsed = parse_pdb_assembly(text)
    assert parsed.chains == {"A": ("ALA", "GLY")}
    assert parsed.ca_residues["A"] == (
        (1, "ALA", (11.104, 13.207, 10.0)),
        (2, "GLY", (12.0, 14.0, 11.0)),
    )


def test_two_chain_mapping() -> None:
    text = (
        "SEQRES   1 A   1  ALA\n"
        "SEQRES   1 B   1  GLY\n"
        "ATOM      1  CA  ALA A   1      11.104  13.207  10.000  1.00 20.00           C\n"
        "ATOM      2  CA  GLY B   1      12.000  14.000  11.000  1.00 20.00           C\n"
    )
    parsed = parse_pdb_assembly(text)
    assert set(parsed.chains) == {"A", "B"}
    assert parsed.chains["A"] == ("ALA",)
    assert parsed.chains["B"] == ("GLY",)
    assert parsed.ca_residues["A"] == ((1, "ALA", (11.104, 13.207, 10.0)),)
    assert parsed.ca_residues["B"] == ((1, "GLY", (12.0, 14.0, 11.0)),)


def test_normalize_renumbering() -> None:
    text = (
        "SEQRES   1 A   3  ALA GLY SER\n"
        "ATOM      1  CA  ALA A  10      11.104  13.207  10.000  1.00 20.00           C\n"
        "ATOM      2  CA  GLY A  11      12.000  14.000  11.000  1.00 20.00           C\n"
        "ATOM      3  CA  SER A  12      13.000  15.000  12.000  1.00 20.00           C\n"
    )
    parsed = parse_pdb_assembly(text)
    lines = normalize_ca_residue_numbers(parsed)
    assert [line[22:26].strip() for line in lines] == ["1", "2", "3"]


def test_normalize_gap_maps_to_seqres_positions() -> None:
    text = (
        "SEQRES   1 A   3  ALA GLY SER\n"
        "ATOM      1  CA  ALA A   1      11.104  13.207  10.000  1.00 20.00           C\n"
        "ATOM      2  CA  SER A   3      13.000  15.000  12.000  1.00 20.00           C\n"
    )
    parsed = parse_pdb_assembly(text)
    lines = normalize_ca_residue_numbers(parsed)
    assert [line[22:26].strip() for line in lines] == ["1", "3"]


def test_blank_seqres_chain_rejected() -> None:
    text = "SEQRES   1     1  ALA\n"
    with pytest.raises(ValueError):
        parse_pdb_assembly(text)


def test_no_seqres_rejected() -> None:
    text = "ATOM      1  CA  ALA A   1      11.104  13.207  10.000  1.00 20.00           C\n"
    with pytest.raises(ValueError):
        parse_pdb_assembly(text)


def test_unsupported_seqres_residue_rejected() -> None:
    text = "SEQRES   1 A   1  XXX\n"
    with pytest.raises(ValueError):
        parse_pdb_assembly(text)


def test_non_subsequence_ca_rejected() -> None:
    text = "SEQRES   1 A   2  ALA GLY\nATOM      1  CA  SER A   1      11.104  13.207  10.000  1.00 20.00           C\n"
    parsed = parse_pdb_assembly(text)
    with pytest.raises(ValueError):
        normalize_ca_residue_numbers(parsed)


def test_parsed_assembly_is_frozen() -> None:
    parsed = ParsedAssembly(chains={"A": ("ALA",)}, ca_residues={})
    with pytest.raises(FrozenInstanceError):
        parsed.chains = {}  # type: ignore[misc]
