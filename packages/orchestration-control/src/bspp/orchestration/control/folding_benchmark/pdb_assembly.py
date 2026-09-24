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

"""Pure-stdlib PDB assembly parser.

Faithful port of ``parse_pdb_assembly`` and ``normalize_ca_residue_numbers`` from
the frozen reference pipeline harvest source
(``src/afdb_pipeline/benchmark_dataset.py``), adapted to expose parsed
``(resnum, resname, (x, y, z))`` tuples rather
than raw re-sliced coordinate lines.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

PDB_RESIDUES: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}

CaResidue = tuple[int, str, tuple[float, float, float]]


@dataclass(frozen=True)
class ParsedAssembly:
    """Parsed SEQRES sequences and first-model C-alpha coordinates."""

    chains: Mapping[str, tuple[str, ...]]  # chain -> SEQRES 3-letter names
    ca_residues: Mapping[str, tuple[CaResidue, ...]]  # chain -> first-model C-alpha records


def parse_pdb_assembly(text: str) -> ParsedAssembly:
    """Parse SEQRES protein chains and first-model C-alpha ATOM records."""
    chains: dict[str, list[str]] = {}
    chain_order: list[str] = []
    ca_residues: dict[str, list[CaResidue]] = {}
    ca_seen: dict[str, set[tuple[str, str]]] = {}
    saw_model = False
    in_first_model = True

    for line in text.splitlines():
        if line.startswith("SEQRES"):
            chain_id = line[11:12].strip()
            if not chain_id:
                raise ValueError("reference has a blank SEQRES chain ID")
            if chain_id not in chains:
                chains[chain_id] = []
                chain_order.append(chain_id)
                ca_residues[chain_id] = []
                ca_seen[chain_id] = set()
            tokens = line.split()[4:]
            for token in tokens:
                if token.upper() not in PDB_RESIDUES:
                    raise ValueError(f"reference contains unsupported residue {token}")
                chains[chain_id].append(token.upper())
            continue

        if line.startswith("MODEL"):
            if saw_model:
                in_first_model = False
            saw_model = True
            continue

        if line.startswith("ENDMDL") and saw_model:
            break

        if not in_first_model or not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA" or line[16:17] not in {" ", "A"}:
            continue
        chain_id = line[21:22].strip()
        if chain_id not in chains:
            continue
        residue = (line[22:26], line[26:27])
        if residue in ca_seen[chain_id]:
            continue
        residue_name = line[17:20].strip().upper()
        if residue_name not in PDB_RESIDUES:
            continue
        try:
            resnum = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        ca_seen[chain_id].add(residue)
        ca_residues[chain_id].append((resnum, residue_name, (x, y, z)))

    if not chain_order:
        raise ValueError("reference has no SEQRES protein chains")

    return ParsedAssembly(
        chains={chain_id: tuple(chains[chain_id]) for chain_id in chain_order},
        ca_residues={chain_id: tuple(ca_residues[chain_id]) for chain_id in chain_order},
    )


def _find_residue(sequence: tuple[str, ...], residue: str, offset: int) -> int:
    try:
        return sequence.index(residue, offset)
    except ValueError:
        return -1


def normalize_ca_residue_numbers(parsed: ParsedAssembly) -> tuple[str, ...]:
    """Renumber C-alpha records to one-based SEQRES positions.

    Returns one canonical, re-parseable PDB ATOM line per C-alpha record, in chain
    insertion order then C-alpha order.
    """
    normalized: list[str] = []
    serial = 0
    for chain_id, seqres in parsed.chains.items():
        offset = 0
        for _resnum, residue_name, (x, y, z) in parsed.ca_residues[chain_id]:
            position = _find_residue(seqres, residue_name, offset)
            if position < 0:
                raise ValueError("coordinate residue sequence is not a subsequence of SEQRES")
            offset = position + 1
            serial += 1
            normalized.append(
                f"ATOM  {serial:5d}  CA  {residue_name:>3} {chain_id}{position + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
    return tuple(normalized)
