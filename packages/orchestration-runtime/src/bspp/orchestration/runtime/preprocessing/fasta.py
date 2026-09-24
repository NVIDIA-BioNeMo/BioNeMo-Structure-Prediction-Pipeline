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

"""FASTA parsing for the pinned MMSA Compatibility Port.

Port Baseline: 419813dbb5a3949e5e16f289f974d9f95e94bf01:README.md:17-41.
"""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.contract.preprocessing import FastaNormalizationMode, PreprocessingFastaRecord


def parse_preprocessing_fasta(
    path: Path,
    *,
    normalization_mode: FastaNormalizationMode = "strict-two-line",
) -> tuple[PreprocessingFastaRecord, ...]:
    """Parse strict two-line FASTA, or explicitly normalize multiline FASTA."""
    if path.suffix != ".fa":
        msg = "preprocessing FASTA input must use the .fa suffix"
        raise ValueError(msg)
    if normalization_mode not in {"strict-two-line", "normalize-multiline"}:
        msg = f"unsupported FASTA normalization mode: {normalization_mode!r}"
        raise ValueError(msg)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        msg = f"cannot read preprocessing FASTA {path}: {error}"
        raise ValueError(msg) from error
    if not lines:
        msg = "FASTA input is empty"
        raise ValueError(msg)
    if normalization_mode == "normalize-multiline":
        multiline_records = _parse_multiline(lines)
        _reject_duplicate_identities(multiline_records)
        return multiline_records
    if len(lines) % 2:
        msg = "strict two-line FASTA input has a truncated record"
        raise ValueError(msg)

    strict_records: list[PreprocessingFastaRecord] = []
    for ordinal in range(len(lines) // 2):
        header = lines[ordinal * 2]
        sequence = lines[ordinal * 2 + 1]
        if not header.startswith(">") or len(header) == 1 or header[1].isspace():
            msg = f"FASTA record {ordinal} has a malformed header"
            raise ValueError(msg)
        if not sequence or sequence.startswith(">"):
            msg = f"FASTA record {ordinal} has a missing sequence"
            raise ValueError(msg)
        if any(character.isspace() for character in sequence):
            msg = f"FASTA record {ordinal} sequence contains whitespace"
            raise ValueError(msg)
        identity = header[1:].split(maxsplit=1)[0]
        strict_records.append(
            PreprocessingFastaRecord(
                header=header,
                sequence=sequence,
                identity=identity,
                source_ordinal=ordinal,
            )
        )
    result = tuple(strict_records)
    _reject_duplicate_identities(result)
    return result


def _parse_multiline(lines: list[str]) -> tuple[PreprocessingFastaRecord, ...]:
    records: list[PreprocessingFastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []
    for line in lines:
        if line.startswith(">"):
            if header is not None:
                records.append(_record(header, "".join(sequence_parts), len(records)))
            header = line
            sequence_parts = []
        else:
            if header is None:
                msg = "multiline FASTA sequence appears before the first header"
                raise ValueError(msg)
            if not line:
                msg = (
                    f"FASTA record {len(records)} has a missing sequence"
                    if not sequence_parts
                    else "multiline FASTA contains a blank sequence line"
                )
                raise ValueError(msg)
            sequence_parts.append(line)
    if header is not None:
        records.append(_record(header, "".join(sequence_parts), len(records)))
    return tuple(records)


def _record(header: str, sequence: str, ordinal: int) -> PreprocessingFastaRecord:
    if len(header) == 1 or header[1].isspace():
        msg = f"FASTA record {ordinal} has a malformed header"
        raise ValueError(msg)
    if not sequence:
        msg = f"FASTA record {ordinal} has a missing sequence"
        raise ValueError(msg)
    if any(character.isspace() for character in sequence):
        msg = f"FASTA record {ordinal} sequence contains whitespace"
        raise ValueError(msg)
    return PreprocessingFastaRecord(
        header=header,
        sequence=sequence,
        identity=header[1:].split(maxsplit=1)[0],
        source_ordinal=ordinal,
    )


def _reject_duplicate_identities(records: tuple[PreprocessingFastaRecord, ...]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.identity in seen:
            msg = f"duplicate FASTA identity {record.identity!r}"
            raise ValueError(msg)
        seen.add(record.identity)


__all__ = ["parse_preprocessing_fasta"]
