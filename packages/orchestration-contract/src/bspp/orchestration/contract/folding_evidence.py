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

"""Shared folding action-evidence contracts.

The five folding action-evidence records plus ``CanonicalPairEvidenceEntry``
are frozen here so Runtime can emit and Control can validate the exact same
shapes without crossing the Control/Runtime import boundary. Each loader is
strict: known-field validation is identical to the historical Control-side
loaders, and unknown top-level keys are rejected.

The module deliberately imports only ``contract.prediction_pair`` and never
``contract.phase`` so there is no module-level import cycle.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from bspp.orchestration.contract.prediction_pair import PredictionPair, prediction_pair_from_mapping

_SEQUENCE_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _reject_unknown_fields(payload: Mapping[str, object], allowed: set[str], record_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        msg = f"Unknown {record_name} field(s): {', '.join(unknown)}"
        raise ValueError(msg)


@dataclass(frozen=True)
class MsaFlattenActionEvidence:
    """Projected A3M workspace evidence: the present ``a3ms/`` member inventory."""

    a3m_paths: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> MsaFlattenActionEvidence:
        _reject_unknown_fields(payload, {"a3m_paths"}, "MsaFlattenActionEvidence")
        raw = payload.get("a3m_paths")
        if not isinstance(raw, list) or not raw or any(not isinstance(item, str) or not item for item in raw):
            raise ValueError("msa-flatten evidence a3m_paths must be a non-empty list of non-empty strings")
        return cls(a3m_paths=tuple(raw))


@dataclass(frozen=True)
class SplitActionEvidence:
    """Split evidence: the produced ``chain_*.a3m`` files."""

    chain_files: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> SplitActionEvidence:
        _reject_unknown_fields(payload, {"chain_files"}, "SplitActionEvidence")
        raw = payload.get("chain_files")
        if (
            not isinstance(raw, list)
            or not raw
            or any(
                not isinstance(item, str) or not item.startswith("chain_") or not item.endswith(".a3m") for item in raw
            )
        ):
            raise ValueError("split evidence chain_files must be a non-empty list of chain_*.a3m names")
        return cls(chain_files=tuple(raw))


@dataclass(frozen=True)
class PreprocessActionEvidence:
    """Prepared-input evidence: layout discriminator plus the prepared dirs."""

    fasta_dir: str
    alignment_dir: str
    layout: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PreprocessActionEvidence:
        _reject_unknown_fields(payload, {"fasta_dir", "alignment_dir", "layout"}, "PreprocessActionEvidence")
        fasta_dir = payload.get("fasta_dir")
        alignment_dir = payload.get("alignment_dir")
        layout = payload.get("layout")
        if not isinstance(fasta_dir, str) or not fasta_dir:
            raise ValueError("preprocess evidence fasta_dir must be a non-empty string")
        if not isinstance(alignment_dir, str) or not alignment_dir:
            raise ValueError("preprocess evidence alignment_dir must be a non-empty string")
        if layout not in {"openfold", "bioir", "colabfold"}:
            raise ValueError("preprocess evidence layout must be 'openfold', 'bioir', or 'colabfold'")
        return cls(fasta_dir=fasta_dir, alignment_dir=alignment_dir, layout=layout)


@dataclass(frozen=True)
class FoldActionEvidence:
    """Fold evidence: the canonical prediction pairs for the fold action."""

    pairs: tuple[PredictionPair, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> FoldActionEvidence:
        _reject_unknown_fields(payload, {"pairs"}, "FoldActionEvidence")
        raw = payload.get("pairs")
        if not isinstance(raw, list) or not raw:
            raise ValueError("fold evidence pairs must be a non-empty list")
        pairs: list[PredictionPair] = []
        for position, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ValueError(f"fold evidence pairs[{position}] must be a prediction-pair mapping")
            pairs.append(prediction_pair_from_mapping(cast("Mapping[str, object]", item)))
        return cls(pairs=tuple(pairs))


@dataclass(frozen=True)
class CanonicalPairEvidenceEntry:
    """One completed-run pair entry binding a target to its prediction pair."""

    target_id: str
    sequence_sha256: str
    pair: PredictionPair

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> CanonicalPairEvidenceEntry:
        _reject_unknown_fields(payload, {"target_id", "sequence_sha256", "pair"}, "CanonicalPairEvidenceEntry")
        target_id = payload.get("target_id")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("canonical-pair evidence target_id must be a non-empty string")
        sequence_sha256 = payload.get("sequence_sha256")
        if not isinstance(sequence_sha256, str) or _SEQUENCE_SHA256_RE.fullmatch(sequence_sha256) is None:
            raise ValueError("canonical-pair evidence sequence_sha256 must be 64 lowercase hex characters")
        raw_pair = payload.get("pair")
        if not isinstance(raw_pair, Mapping):
            raise ValueError("canonical-pair evidence pair must be a prediction-pair mapping")
        pair = prediction_pair_from_mapping(cast("Mapping[str, object]", raw_pair))
        return cls(target_id=target_id, sequence_sha256=sequence_sha256, pair=pair)


@dataclass(frozen=True)
class CanonicalPairActionEvidence:
    """Canonical-pair action evidence: the completed-run pair list."""

    entries: tuple[CanonicalPairEvidenceEntry, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> CanonicalPairActionEvidence:
        _reject_unknown_fields(payload, {"entries"}, "CanonicalPairActionEvidence")
        raw = payload.get("entries")
        if not isinstance(raw, list) or not raw:
            raise ValueError("canonical-pair evidence entries must be a non-empty list")
        entries: list[CanonicalPairEvidenceEntry] = []
        for position, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ValueError(f"canonical-pair evidence entries[{position}] must be a mapping")
            entries.append(CanonicalPairEvidenceEntry.from_mapping(item))
        return cls(entries=tuple(entries))


__all__ = [
    "CanonicalPairActionEvidence",
    "CanonicalPairEvidenceEntry",
    "FoldActionEvidence",
    "MsaFlattenActionEvidence",
    "PreprocessActionEvidence",
    "SplitActionEvidence",
]
