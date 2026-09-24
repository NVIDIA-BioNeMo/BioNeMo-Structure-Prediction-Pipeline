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

"""Shared byte-level validation for preprocessing execution and finalization."""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

from bspp.orchestration.contract.phase import PreprocessingRuntimeAction


def validate_preprocessing_a3m_bytes(payload: bytes, *, label: str) -> None:
    """Validate the exact minimum A3M text rules owned by orchestration."""
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"declared A3M is not UTF-8 text: {label}") from exc
    if lines and lines[0].startswith("#"):
        match = re.fullmatch(r"#([1-9]\d*(?:,[1-9]\d*)*)\t([1-9]\d*(?:,[1-9]\d*)*)", lines[0])
        if match is None or len(match.group(1).split(",")) != len(match.group(2).split(",")):
            raise ValueError(f"declared A3M has malformed ColabFold multimer metadata: {label}")
        lines = lines[1:]
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"declared A3M must start with a non-empty header: {label}")
    has_header = False
    current_has_sequence = False
    for line in lines:
        if line.startswith(">"):
            if not line[1:].strip() or (has_header and not current_has_sequence):
                raise ValueError(f"declared A3M contains an empty record: {label}")
            has_header = True
            current_has_sequence = False
        elif line.strip():
            if not has_header:
                raise ValueError(f"declared A3M sequence appears before its header: {label}")
            current_has_sequence = True
    if not current_has_sequence:
        raise ValueError(f"declared A3M final record has no sequence: {label}")


def validate_preprocessing_paired_a3m_bytes(
    payload: bytes,
    *,
    chain_lengths: tuple[int, int],
    label: str,
) -> None:
    """Validate a named paired-query A3M and its exact multimer declaration."""
    validate_preprocessing_a3m_bytes(payload, label=label)
    expected = f"#{chain_lengths[0]},{chain_lengths[1]}\t1,1".encode()
    first_line = payload.splitlines()[0] if payload.splitlines() else b""
    if first_line != expected:
        raise ValueError(f"declared A3M does not contain exact paired multimer metadata: {label}")


def validate_preprocessing_a3m_header_bytes(
    payload: bytes,
    *,
    chain_lengths: tuple[int, ...],
    label: str,
) -> None:
    """Validate a named A3M header for any arity with cardinality expansion.

    Extends the owned A3M validation from the paired-only
    ``#<L0>,<L1>\t1,1`` shape to any arity.  The ``#`` metadata header carries
    comma-separated chain lengths and cardinalities; cardinalities are expanded
    into a flat multiset and compared order-insensitively against
    ``chain_lengths`` (deliberate multiset relaxation, consistent with
    the downstream multiset matching).
    Cardinalities are artifact-controlled, so each one is rejected against the
    remaining expected chain count BEFORE expansion: the allocation is bounded
    by the declared chain count, never by the artifact's cardinality value.

    Grounded in mmsa: mmsa (``msa_batch.sh:38-41``) checks only ``a3m_count <= 2``
    as a safety floor — it never validates A3M header content.  The Phase's
    paired validator was an owned validator; this generalizes it to 1-N chains.
    """
    validate_preprocessing_a3m_bytes(payload, label=label)
    lines = payload.decode("utf-8").splitlines()
    if not lines or not lines[0].startswith("#"):
        raise ValueError(f"declared A3M must start with a ColabFold metadata header: {label}")
    match = re.fullmatch(r"#([1-9]\d*(?:,[1-9]\d*)*)\t([1-9]\d*(?:,[1-9]\d*)*)", lines[0])
    if match is None:
        raise ValueError(f"declared A3M has malformed ColabFold multimer metadata: {label}")
    lengths = tuple(int(part) for part in match.group(1).split(","))
    cardinalities = tuple(int(part) for part in match.group(2).split(","))
    if len(lengths) != len(cardinalities):
        raise ValueError(f"declared A3M metadata length/cardinality count mismatch: {label}")
    expanded: list[int] = []
    remaining = len(chain_lengths)
    for length, cardinality in zip(lengths, cardinalities, strict=True):
        if length <= 0 or cardinality <= 0:
            raise ValueError(f"declared A3M metadata contains non-positive values: {label}")
        if cardinality > remaining:
            raise ValueError(f"declared A3M metadata does not match expected chain lengths: {label}")
        expanded.extend([length] * cardinality)
        remaining -= cardinality
    if sorted(expanded) != sorted(chain_lengths):
        raise ValueError(f"declared A3M metadata does not match expected chain lengths: {label}")


def normalize_preprocessing_tar_member(name: str) -> str:
    """Apply the baseline's repeated leading-``./`` normalization."""
    while name.startswith("./"):
        name = name[2:]
    return name or "."


def validate_preprocessing_tar_headers(
    archive: tarfile.TarFile,
    action: PreprocessingRuntimeAction,
) -> tuple[tarfile.TarInfo, ...]:
    """Require the exact safe top-level A3M inventory for one action."""
    members = tuple(archive.getmembers())
    file_names: list[str] = []
    for member in members:
        normalized = normalize_preprocessing_tar_member(member.name)
        if member.name.startswith("/") or ".." in Path(normalized).parts:
            raise ValueError(f"unsafe preprocessing tar member: {member.name!r}")
        if member.isdir() and normalized == ".":
            continue
        if not member.isfile() or "/" in normalized:
            raise ValueError(f"undeclared preprocessing tar member type or path: {member.name!r}")
        file_names.append(normalized)
    expected = tuple(item.member_name for item in action.payload.expected_a3ms)
    if len(file_names) != len(expected) or set(file_names) != set(expected):
        raise ValueError(f"preprocessing tar inventory does not match declared A3Ms: {file_names}")
    return members


__all__ = [
    "normalize_preprocessing_tar_member",
    "validate_preprocessing_a3m_bytes",
    "validate_preprocessing_a3m_header_bytes",
    "validate_preprocessing_paired_a3m_bytes",
    "validate_preprocessing_tar_headers",
]
