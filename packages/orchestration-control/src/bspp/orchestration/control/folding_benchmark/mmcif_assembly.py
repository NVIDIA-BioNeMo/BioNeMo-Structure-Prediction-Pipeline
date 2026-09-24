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

"""Pure-stdlib mmCIF ``_atom_site`` parser returning C-alpha coordinates only.

A minimal, purpose-built parser that extracts ATOM-grouped C-alpha records from
the ``_atom_site`` loop block of an mmCIF file.  It is **not** a general-purpose
mmCIF parser — it only resolves the ``_atom_site`` loop, maps columns by their
declared header names (N1), and terminates the loop on any structural boundary
(N1).  HETATM rows are skipped entirely (Decision C), matching the original
``parse_pdb_assembly`` behaviour which only reads ``ATOM`` records.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from bspp.orchestration.control.folding_benchmark.pdb_assembly import (
    PDB_RESIDUES,
    CaResidue,
)

# Column names we need from the _atom_site loop.
_REQUIRED_COLUMNS = (
    "group_PDB",
    "label_atom_id",
    "label_comp_id",
    "label_asym_id",
    "label_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
)

# Tokens that terminate the _atom_site data block (N1).
_LOOP_TERMINATORS = frozenset(
    {
        "loop_",
        "data_",
        "save_",
        "stop_",
        "global_",
    }
)


def _parse_label_seq_id(value: str) -> float:
    """Convert ``label_seq_id`` to a sortable float; non-integer → +inf (N2)."""
    try:
        return int(value)
    except ValueError:
        return float("inf")


def _ins_code_sort_key(value: str) -> str:
    """Secondary sort key for ``pdbx_PDB_ins_code``; "."/"?"/"" sort first (N2)."""
    if value in {"", ".", "?"}:
        return ""
    return value


@dataclass(frozen=True)
class MmcifCaData:
    """C-alpha records grouped by ``label_asym_id``."""

    ca_residues: Mapping[str, tuple[CaResidue, ...]]


def _find_atom_site_loop_start(lines: list[str]) -> int:
    """Return the index of the ``loop_`` line that precedes ``_atom_site`` columns."""
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped != "loop_":
            continue
        # Look ahead for _atom_site column declarations.
        for peek in range(index + 1, min(index + 1 + 50, len(lines))):
            peek_stripped = lines[peek].strip()
            if peek_stripped.startswith("_atom_site."):
                return index
            # A non-loop_, non-_atom_site line means this loop_ is for something else.
            if peek_stripped and not peek_stripped.startswith("_"):
                break
    raise ValueError("mmCIF text has no _atom_site loop")


def _parse_column_headers(lines: list[str], loop_start: int) -> dict[str, int]:
    """Build a column-name → index mapping from the ``loop_`` header (N1)."""
    columns: dict[str, int] = {}
    index = 0
    for line in lines[loop_start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("_atom_site."):
            # First non-_atom_site line: end of header.
            break
        col_name = stripped.split(".", 1)[1].split()[0]
        columns[col_name] = index
        index += 1
    if not columns:
        raise ValueError("mmCIF _atom_site loop has no column declarations")
    for required in _REQUIRED_COLUMNS:
        if required not in columns:
            raise ValueError(f"mmCIF _atom_site loop is missing required column: {required}")
    return columns


def _is_terminator(first_token: str) -> bool:
    """Whether the first token of a line terminates the data block (N1)."""
    return (
        first_token in _LOOP_TERMINATORS
        or first_token.startswith("_")
        or first_token == "#"
        or first_token.startswith(";")
    )


def parse_mmcif_assembly(text: str) -> MmcifCaData:
    """Parse the ``_atom_site`` loop, returning ATOM-grouped CA records keyed by ``label_asym_id``.

    * Column indices are resolved from the ``loop_`` header declarations (N1).
    * The loop terminates on any structural boundary or a short row (N1).
    * Only rows with ``group_PDB == "ATOM"`` and ``label_atom_id == "CA"`` are kept (Decision C).
    * ``label_alt_id`` altloc filter: keep "", ".", "A" (or if column absent, keep all).
    * ``pdbx_PDB_model_num`` first-model filter: if present, only the first model's rows are kept.
    * ``label_seq_id`` of "."/"?"/non-integer sorts last (N2).
    * ``pdbx_PDB_ins_code`` of "."/"?"/"" sorts first as secondary key (N2).
    * Dedup on ``(label_seq_id, pdbx_PDB_ins_code)`` within a chain.
    * ``label_comp_id`` must be a known PDB residue; unknown residues are skipped.
    """
    lines = text.splitlines()
    loop_start = _find_atom_site_loop_start(lines)
    columns = _parse_column_headers(lines, loop_start)
    num_columns = len(columns)
    alt_id_present = "label_alt_id" in columns
    model_num_present = "pdbx_PDB_model_num" in columns
    first_model: str | None = None

    # Collect qualifying rows with their sort keys.
    # Each entry: (seq_sort, ins_sort, CaResidue)
    grouped: dict[str, list[tuple[float, str, CaResidue]]] = {}
    seen: dict[str, set[tuple[str, str]]] = {}

    data_start = loop_start + 1
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("_atom_site."):
            continue
        tokens = stripped.split()
        if not tokens:
            continue
        first = tokens[0]
        if _is_terminator(first):
            break
        # Short row: fewer tokens than columns → end of data block (N1, defensive).
        if len(tokens) < num_columns:
            break

        if model_num_present:
            model_num = tokens[columns["pdbx_PDB_model_num"]]
            if first_model is None:
                first_model = model_num
            elif model_num != first_model:
                continue

        group_pdb = tokens[columns["group_PDB"]]
        if group_pdb != "ATOM":
            continue
        atom_id = tokens[columns["label_atom_id"]]
        if atom_id != "CA":
            continue
        if alt_id_present:
            alt_id = tokens[columns["label_alt_id"]]
            if alt_id not in {"", ".", "A"}:
                continue
        comp_id = tokens[columns["label_comp_id"]]
        if comp_id not in PDB_RESIDUES:
            continue
        asym_id = tokens[columns["label_asym_id"]]
        seq_id_raw = tokens[columns["label_seq_id"]]
        ins_code_raw = tokens[columns["pdbx_PDB_ins_code"]]

        dedup_key = (seq_id_raw, ins_code_raw)
        if asym_id not in seen:
            seen[asym_id] = set()
        if dedup_key in seen[asym_id]:
            continue
        seen[asym_id].add(dedup_key)

        seq_sort = _parse_label_seq_id(seq_id_raw)
        ins_sort = _ins_code_sort_key(ins_code_raw)
        try:
            resnum = int(seq_id_raw)
        except ValueError:
            resnum = 0
        try:
            x = float(tokens[columns["Cartn_x"]])
            y = float(tokens[columns["Cartn_y"]])
            z = float(tokens[columns["Cartn_z"]])
        except ValueError:
            continue

        grouped.setdefault(asym_id, []).append((seq_sort, ins_sort, (resnum, comp_id, (x, y, z))))

    if not grouped:
        raise ValueError("mmCIF _atom_site loop has no ATOM CA records")

    result: dict[str, tuple[CaResidue, ...]] = {}
    for asym_id, entries in grouped.items():
        entries.sort(key=lambda item: (item[0], item[1]))
        result[asym_id] = tuple(entry[2] for entry in entries)

    return MmcifCaData(ca_residues=result)
