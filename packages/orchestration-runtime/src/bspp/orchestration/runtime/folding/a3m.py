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

"""Fail-closed A3M header parsing for the folding Compatibility Port.

Port Baseline:
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/scripts/preprocessing/generate_batch_info.py:27-67``.
The port deliberately strengthens the prefix parser with full-header matching,
a query-header check, positive cardinalities, and query/header length agreement.
Any invalid file fails the whole declared index instead of being silently
dropped from an unordered result stream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HETERODIMER_HEADER = re.compile(r"#(\d+),(\d+)\s+(\d+),(\d+)")
_MONOMER_OR_HOMOMER_HEADER = re.compile(r"#(\d+)\s+(\d+)")


@dataclass(frozen=True)
class ParsedA3m:
    """Header and query facts read from one A3M without scientific execution."""

    path: Path
    query_sequence: str
    sequence_length: int
    chain_lengths: tuple[int, ...]
    chain_count: int
    chain_cardinalities: tuple[int, ...]
    total_length: int


def parse_a3m(path: Path) -> ParsedA3m:
    """Read and validate the baseline's three-line A3M metadata prefix."""
    try:
        header, query_header, query_sequence = _read_a3m_prefix(path)
    except (OSError, UnicodeError) as exc:
        msg = f"Cannot read A3M input {path}: {exc}"
        raise ValueError(msg) from exc

    if not query_header.startswith(">"):
        msg = f"A3M {path} has an invalid query header on line 2"
        raise ValueError(msg)
    if not query_sequence or any(character.isspace() for character in query_sequence):
        msg = f"A3M {path} has an empty or invalid query sequence"
        raise ValueError(msg)

    chain_lengths: tuple[int, ...]
    cardinalities: tuple[int, ...]
    heterodimer = _HETERODIMER_HEADER.fullmatch(header)
    if heterodimer is not None:
        chain_lengths = (int(heterodimer.group(1)), int(heterodimer.group(2)))
        cardinalities = (int(heterodimer.group(3)), int(heterodimer.group(4)))
        _validate_positive_values(chain_lengths, path=path, field_name="chain length")
        _validate_positive_values(cardinalities, path=path, field_name="chain cardinality")
        sequence_length = sum(chain_lengths)
        chain_count = 2
        total_length = sequence_length
    else:
        monomer_or_homomer = _MONOMER_OR_HOMOMER_HEADER.fullmatch(header)
        if monomer_or_homomer is None:
            msg = f"A3M {path} has an unsupported or malformed header {header!r}"
            raise ValueError(msg)
        sequence_length = int(monomer_or_homomer.group(1))
        chain_count = int(monomer_or_homomer.group(2))
        _validate_positive_values((sequence_length,), path=path, field_name="chain length")
        _validate_positive_values((chain_count,), path=path, field_name="chain cardinality")
        chain_lengths = (sequence_length,)
        cardinalities = (chain_count,)
        total_length = sequence_length * chain_count

    if len(query_sequence) != sequence_length:
        msg = f"A3M {path} query length {len(query_sequence)} does not match header length {sequence_length}"
        raise ValueError(msg)

    return ParsedA3m(
        path=path,
        query_sequence=query_sequence,
        sequence_length=sequence_length,
        chain_lengths=chain_lengths,
        chain_count=chain_count,
        chain_cardinalities=cardinalities,
        total_length=total_length,
    )


def _read_a3m_prefix(path: Path) -> tuple[str, str, str]:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip()
        query_header = handle.readline().strip()
        query_sequence = handle.readline().strip()
    return header, query_header, query_sequence


def _validate_positive_values(values: tuple[int, ...], *, path: Path, field_name: str) -> None:
    if any(value <= 0 for value in values):
        msg = f"A3M {path} has an impossible {field_name}: values must be positive"
        raise ValueError(msg)


__all__ = ["ParsedA3m", "parse_a3m"]
