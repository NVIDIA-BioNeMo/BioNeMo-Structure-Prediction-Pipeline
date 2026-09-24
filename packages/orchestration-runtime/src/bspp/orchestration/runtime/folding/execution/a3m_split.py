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

"""N-ary A3M split and Stockholm derivation for Track-A folding inputs.

Port Baseline: frozen reference pipeline src/afdb_pipeline/backends/local.py.
The harvested helpers and the two public functions are behavior-identical; only
``BackendError`` becomes ``FoldingBackendError``.  The split rule follows
the N-ary ``#<L0>,<L1>,...\t<C0>,<C1>,...`` header with equal arity
and positive values, lowercase insertions never advance the aligned-position
counter, the first record is the query, and homomeric copies are mapped by
multiset match of ungapped query pieces against target chains.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .errors import FoldingBackendError
from .models import ProteinTarget

__all__ = [
    "_a3m_layout",
    "_a3m_query",
    "_a3m_query_chains",
    "_first_a3m_sequence",
    "_read_a3m",
    "_split_a3m_sequence",
    "_ungapped_a3m",
    "a3m_target_length",
    "a3m_to_stockholm",
    "split_merged_a3m",
]


def _a3m_layout(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Read ColabFold's unique-chain lengths and homomer cardinalities."""
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if not line.startswith("#"):
                return None
            fields = line[1:].split()
            if len(fields) != 2:
                return None
            try:
                lengths = tuple(int(value) for value in fields[0].split(","))
                cardinalities = tuple(int(value) for value in fields[1].split(","))
            except ValueError:
                return None
            if (
                not lengths
                or len(lengths) != len(cardinalities)
                or any(value <= 0 for value in (*lengths, *cardinalities))
            ):
                return None
            return lengths, cardinalities
    return None


def _ungapped_a3m(sequence: str) -> str:
    return "".join(character for character in sequence if not character.islower() and character not in "-.")


def _first_a3m_sequence(path: Path) -> str | None:
    sequence: list[str] = []
    seen_header = False
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if seen_header:
                    break
                seen_header = True
                continue
            if seen_header:
                sequence.append(line)
    return "".join(sequence) if seen_header else None


def _a3m_query_chains(path: Path) -> tuple[str, ...]:
    query = _first_a3m_sequence(path)
    if query is None:
        return ()
    layout = _a3m_layout(path)
    if layout is None:
        return (_ungapped_a3m(query),)
    lengths, cardinalities = layout
    pieces = _split_a3m_sequence(query, lengths)
    chains: list[str] = []
    for piece, cardinality in zip(pieces, cardinalities, strict=True):
        chains.extend([_ungapped_a3m(piece)] * cardinality)
    return tuple(chains)


def _a3m_query(path: Path) -> str:
    return "".join(_a3m_query_chains(path))


def a3m_target_length(path: Path) -> int:
    """Return expanded target residues using the execution N-ary query rules.

    Validate the metadata and aligned query widths, then weight each ungapped
    unique-chain query length by its cardinality. Arithmetic avoids allocating
    a repeated chain list controlled by cardinalities in the input artifact.
    The historical compatibility-port parser/index semantics remain separate.
    """
    layout = _a3m_layout(path)
    if layout is None:
        raise FoldingBackendError(f"A3M has missing or malformed chain metadata: {path}")
    query = _first_a3m_sequence(path)
    if not query or any(character.isspace() for character in query):
        raise FoldingBackendError(f"A3M has an empty or invalid query sequence: {path}")
    lengths, cardinalities = layout
    pieces = _split_a3m_sequence(query, lengths)
    query_lengths = tuple(len(_ungapped_a3m(piece)) for piece in pieces)
    if any(length == 0 for length in query_lengths):
        raise FoldingBackendError(f"A3M has an empty normalized query chain: {path}")
    return sum(length * cardinality for length, cardinality in zip(query_lengths, cardinalities, strict=True))


def _read_a3m(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence)))
            header = line[1:]
            sequence = []
        elif header is not None:
            sequence.append(line)
    if header is not None:
        records.append((header, "".join(sequence)))
    return records


def _split_a3m_sequence(sequence: str, chain_lengths: tuple[int, ...]) -> list[str]:
    boundaries: list[int] = []
    total = 0
    for length in chain_lengths:
        total += length
        boundaries.append(total)
    segments: list[list[str]] = [[] for _ in chain_lengths]
    aligned_position = 0
    chain_index = 0
    for character in sequence:
        while chain_index < len(boundaries) - 1 and aligned_position >= boundaries[chain_index]:
            chain_index += 1
        segments[chain_index].append(character)
        if not character.islower():
            aligned_position += 1
    if aligned_position != total:
        raise FoldingBackendError(f"merged A3M row has {aligned_position} aligned columns; expected {total}")
    return ["".join(segment) for segment in segments]


def split_merged_a3m(source: Path, target: ProteinTarget, output_dir: Path) -> list[Path]:
    records = _read_a3m(source)
    if not records:
        raise FoldingBackendError(f"A3M contains no records: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = _a3m_layout(source)
    lengths = layout[0] if layout is not None else tuple(map(len, target.chains))
    unique_records: list[list[tuple[str, str]]] = [[] for _ in lengths]
    for record_index, (header, sequence) in enumerate(records):
        pieces = _split_a3m_sequence(sequence, lengths)
        for index, piece in enumerate(pieces):
            if record_index == 0 or any(character not in "-." and not character.islower() for character in piece):
                unique_records[index].append((header, piece))

    if layout is None:
        chain_records = unique_records
    else:
        _, cardinalities = layout
        query_pieces = tuple(_ungapped_a3m(piece) for piece in _split_a3m_sequence(records[0][1], lengths))
        if Counter(target.chains) != Counter(
            sequence
            for sequence, cardinality in zip(query_pieces, cardinalities, strict=True)
            for _ in range(cardinality)
        ):
            raise FoldingBackendError(f"ColabFold A3M chain layout does not match target {target.target_id}: {source}")
        chain_records = []
        for sequence in target.chains:
            try:
                unique_index = query_pieces.index(sequence)
            except ValueError as exc:
                raise FoldingBackendError(
                    f"ColabFold A3M has no chain matching target {target.target_id}: {source}"
                ) from exc
            chain_records.append(unique_records[unique_index])

    paths: list[Path] = []
    for index, records_for_chain in enumerate(chain_records, start=1):
        path = output_dir / f"chain_{index}.a3m"
        with path.open("w", encoding="utf-8") as handle:
            for header, sequence in records_for_chain:
                handle.write(f">{header}\n{sequence}\n")
        paths.append(path)
    return paths


def a3m_to_stockholm(path: Path) -> str:
    rows: list[tuple[str, str]] = []
    seen_ids: dict[str, int] = {}
    for header, sequence in _read_a3m(path):
        base_id = header.split(maxsplit=1)[0].replace("/", "_") or "sequence"
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        identifier = base_id if seen_ids[base_id] == 1 else f"{base_id}_{seen_ids[base_id]}"
        aligned = "".join(character for character in sequence if not character.islower()).replace(".", "-")
        rows.append((identifier, aligned))
    if not rows:
        raise FoldingBackendError(f"cannot create Stockholm alignment from empty A3M: {path}")
    width = max(len(identifier) for identifier, _ in rows) + 1
    body = "".join(f"{identifier:<{width}}{sequence}\n" for identifier, sequence in rows)
    return f"# STOCKHOLM 1.0\n{body}//\n"
