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

"""Frozen model shapes exchanged by folding backends.

Port Baseline: frozen reference pipeline src/afdb_pipeline/models.py.
Only the field shapes are ported; the harvest's __post_init__ validation,
total_length/sequence_hash properties, as_dict, and safe_id helper are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bspp.orchestration.contract.model_identity import normalize_model_entity_id, parse_human_string_model_entity_id

from .errors import FoldingBackendError


@dataclass(frozen=True)
class ProteinTarget:
    """One prediction target, composed of one or more protein chains."""

    target_id: str
    description: str
    chains: tuple[str, ...]


def require_valid_target_identity(target: ProteinTarget) -> str:
    """Return the normalized model entity identity for ``target``, failing closed.

    ``target_id`` flows into chain IDs, directory names, and FASTA/canonical
    file names, so it must satisfy the contract model-ID grammar
    (``contract.model_identity``: homodimer ``AF-\\d{16}``, compound
    ``AF_<A>_AF_<B>`` / ``AF-`` variants, ``AFDB_`` strip) before any
    filesystem path is derived from it. PDB assembly and exact HumanSTRING
    roots are also accepted; HumanSTRING targets require two expanded chains.
    Identities outside the grammar raise
    :class:`FoldingBackendError` wrapping the grammar's ``ValueError``.
    """

    try:
        model_id = normalize_model_entity_id(target.target_id)
        accessions = parse_human_string_model_entity_id(model_id)
        if accessions is not None:
            if len(target.chains) != 2 or any(not chain for chain in target.chains):
                raise ValueError("HumanSTRING targets require exactly two nonempty expanded chains")
            if len(accessions) == 1 and target.chains[0] != target.chains[1]:
                raise ValueError("HumanSTRING homo targets require two identical chains")
        return model_id
    except ValueError as exc:
        raise FoldingBackendError(f"target has an invalid model entity identity: {target.target_id!r}") from exc


@dataclass(frozen=True)
class PreparedInput:
    """Filesystem layout consumed by a folding backend."""

    backend: str
    fasta_dir: Path
    alignment_dir: Path
    template_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructurePrediction:
    """One ranked structure produced by a folding backend."""

    rank: int
    structure_path: Path
    scores_path: Path
    confidence: float | None = None


@dataclass(frozen=True)
class FoldingResult:
    """Normalized output of a folding backend."""

    backend: str
    predictions: tuple[StructurePrediction, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
