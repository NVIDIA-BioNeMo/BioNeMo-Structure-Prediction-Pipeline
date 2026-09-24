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

"""Versioned fold-to-postprocessing prediction-pair handoff schema.

A prediction pair binds one normalized model entity ID to its structure and
scores files and the harvested prediction scores payload. The outer pair
schema is closed and strict; only the score payload's ``extras`` mapping is
intentionally open so downstream consumers can preserve unknown
score keys without interpreting them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeGuard, cast

from bspp.orchestration.contract.model_identity import (
    CANONICAL_META_SUFFIX,
    CANONICAL_MODEL_SUFFIX,
    normalize_model_entity_id,
)
from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SCORE_FIELDS = frozenset({"schema_version", "plddt", "pae", "max_pae", "ptm", "iptm"})
_PAIR_FIELDS = frozenset({"schema_version", "model_entity_id", "tool_used", "structure_path", "scores_path", "scores"})

# Each tuple is a (structure suffix, scores suffix) naming profile. The first
# profile is canonical; the remaining two are the harvested raw forms.
_SUFFIX_PAIRS: tuple[tuple[str, str], ...] = (
    (CANONICAL_MODEL_SUFFIX, CANONICAL_META_SUFFIX),
    (
        ".merged_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb",
        ".merged_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
    ),
    (
        "_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb",
        "_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
    ),
)


@dataclass(frozen=True)
class PredictionScoresPayload:
    """Harvested prediction scores with an intentionally open extras mapping."""

    plddt: tuple[float | int, ...]
    pae: tuple[tuple[float | int, ...], ...]
    max_pae: float | int
    ptm: float | int | None
    iptm: float | int | None
    extras: Mapping[str, object]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PredictionScoresPayload")
        if not isinstance(self.plddt, tuple) or not self.plddt:
            raise ValueError("plddt must be a non-empty tuple")
        if not all(_is_finite_number(value) for value in self.plddt):
            raise ValueError("plddt must contain only finite numbers")
        if not isinstance(self.pae, tuple) or len(self.pae) != len(self.plddt):
            raise ValueError("pae must be a tuple with exactly len(plddt) rows")
        for row in self.pae:
            if not isinstance(row, tuple) or not row:
                raise ValueError("pae rows must be non-empty tuples")
            if not all(_is_finite_number(value) for value in row):
                raise ValueError("pae rows must contain only finite numbers")
        if not _is_finite_number(self.max_pae):
            raise ValueError("max_pae must be a finite number")
        if self.ptm is not None and not _is_finite_number(self.ptm):
            raise ValueError("ptm must be null or a finite number")
        if self.iptm is not None and not _is_finite_number(self.iptm):
            raise ValueError("iptm must be null or a finite number")
        if not isinstance(self.extras, Mapping):
            raise ValueError("extras must be a mapping")
        collision = sorted(set(self.extras) & _SCORE_FIELDS)
        if collision:
            msg = f"extras must not collide with reserved score fields: {', '.join(collision)}"
            raise ValueError(msg)
        object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-ready mapping with required fields and preserved extras."""
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "plddt": list(self.plddt),
            "pae": [list(row) for row in self.pae],
            "max_pae": self.max_pae,
            "ptm": self.ptm,
            "iptm": self.iptm,
        }
        payload.update(self.extras)
        return payload


@dataclass(frozen=True)
class PredictionPair:
    """One fold-to-postprocessing prediction-pair handoff record."""

    model_entity_id: str
    tool_used: str
    structure_path: str
    scores_path: str
    scores: PredictionScoresPayload
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PredictionPair")
        normalized_id = normalize_model_entity_id(self.model_entity_id)
        object.__setattr__(self, "model_entity_id", normalized_id)
        if self.tool_used not in VALID_TOOL_USED:
            msg = f"tool_used must be one of {VALID_TOOL_USED!r}; got {self.tool_used!r}"
            raise ValueError(msg)
        if not isinstance(self.structure_path, str) or not self.structure_path:
            raise ValueError("structure_path must be a non-empty string")
        if not isinstance(self.scores_path, str) or not self.scores_path:
            raise ValueError("scores_path must be a non-empty string")
        structure_root, scores_root = _match_suffix_profile(self.structure_path, self.scores_path)
        if normalize_model_entity_id(structure_root) != normalized_id:
            raise ValueError("structure_path root does not match model_entity_id")
        if normalize_model_entity_id(scores_root) != normalized_id:
            raise ValueError("scores_path root does not match model_entity_id")
        if not isinstance(self.scores, PredictionScoresPayload):
            raise ValueError("scores must be a PredictionScoresPayload")
        if self.scores.schema_version != self.schema_version:
            raise ValueError("scores schema_version must match the prediction pair")

    def to_mapping(self) -> dict[str, object]:
        """Return the closed outer mapping with the nested score payload."""
        return {
            "schema_version": self.schema_version,
            "model_entity_id": self.model_entity_id,
            "tool_used": self.tool_used,
            "structure_path": self.structure_path,
            "scores_path": self.scores_path,
            "scores": self.scores.to_mapping(),
        }


def prediction_scores_payload_from_mapping(payload: Mapping[str, object]) -> PredictionScoresPayload:
    """Build a score payload from a parsed mapping with an open extras schema."""
    schema_version = _require_explicit_schema(payload, "PredictionScoresPayload")
    plddt = _require_number_list(payload, "plddt")
    pae = _require_number_matrix(payload, "pae")
    max_pae = _require_number(payload, "max_pae")
    ptm = _require_optional_number(payload, "ptm")
    iptm = _require_optional_number(payload, "iptm")
    extras = {key: value for key, value in payload.items() if key not in _SCORE_FIELDS}
    return PredictionScoresPayload(
        schema_version=schema_version,
        plddt=plddt,
        pae=pae,
        max_pae=max_pae,
        ptm=ptm,
        iptm=iptm,
        extras=extras,
    )


def prediction_pair_from_mapping(payload: Mapping[str, object]) -> PredictionPair:
    """Build a prediction pair from a parsed, closed outer mapping."""
    schema_version = _require_explicit_schema(payload, "PredictionPair")
    unknown = sorted(set(payload) - _PAIR_FIELDS)
    if unknown:
        msg = f"Unknown PredictionPair field(s): {', '.join(unknown)}"
        raise ValueError(msg)
    model_entity_id = _require_str(payload, "model_entity_id")
    tool_used = _require_str(payload, "tool_used")
    structure_path = _require_str(payload, "structure_path")
    scores_path = _require_str(payload, "scores_path")
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, Mapping):
        raise ValueError("scores must be a mapping")
    scores = prediction_scores_payload_from_mapping(cast("Mapping[str, object]", raw_scores))
    return PredictionPair(
        schema_version=schema_version,
        model_entity_id=model_entity_id,
        tool_used=tool_used,
        structure_path=structure_path,
        scores_path=scores_path,
        scores=scores,
    )


def _is_finite_number(value: object) -> TypeGuard[float | int]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _require_number(payload: Mapping[str, object], key: str) -> float | int:
    value = payload.get(key)
    if not _is_finite_number(value):
        raise ValueError(f"{key} must be a finite number")
    return value


def _require_optional_number(payload: Mapping[str, object], key: str) -> float | int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not _is_finite_number(value):
        raise ValueError(f"{key} must be null or a finite number")
    return value


def _require_number_list(payload: Mapping[str, object], key: str) -> tuple[float | int, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON array")
    if not value:
        raise ValueError(f"{key} must be a non-empty JSON array")
    numbers: list[float | int] = []
    for item in value:
        if not _is_finite_number(item):
            raise ValueError(f"{key} must contain only finite numbers")
        numbers.append(item)
    return tuple(numbers)


def _require_number_matrix(payload: Mapping[str, object], key: str) -> tuple[tuple[float | int, ...], ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON array")
    rows: list[tuple[float | int, ...]] = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError(f"{key} rows must be JSON arrays")
        if not row:
            raise ValueError(f"{key} rows must be non-empty")
        row_numbers: list[float | int] = []
        for item in row:
            if not _is_finite_number(item):
                raise ValueError(f"{key} rows must contain only finite numbers")
            row_numbers.append(item)
        rows.append(tuple(row_numbers))
    return tuple(rows)


def _require_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_explicit_schema(payload: Mapping[str, object], name: str) -> int:
    if "schema_version" not in payload:
        raise ValueError(f"missing explicit schema_version at {name}")
    value = payload.get("schema_version")
    if value is None:
        raise ValueError(f"missing explicit schema_version at {name}")
    return validate_schema_version(value, record_name=name)


def _validate_schema(value: int, name: str) -> None:
    if validate_schema_version(value, record_name=name) != value:
        raise ValueError(f"{name} schema_version must be explicit")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _match_suffix_profile(structure_path: str, scores_path: str) -> tuple[str, str]:
    structure_name = _basename(structure_path)
    scores_name = _basename(scores_path)
    structure_suffix = max(
        (suffix for suffix, _scores_suffix in _SUFFIX_PAIRS if structure_name.endswith(suffix)),
        key=len,
        default=None,
    )
    scores_suffix = max(
        (suffix for _structure_suffix, suffix in _SUFFIX_PAIRS if scores_name.endswith(suffix)),
        key=len,
        default=None,
    )
    if structure_suffix is None:
        raise ValueError("structure_path must end with a known PDB naming suffix")
    if scores_suffix is None:
        raise ValueError("scores_path must end with a known JSON naming suffix")
    structure_index = next(
        index for index, (suffix, _scores_suffix) in enumerate(_SUFFIX_PAIRS) if suffix == structure_suffix
    )
    scores_index = next(
        index for index, (_structure_suffix, suffix) in enumerate(_SUFFIX_PAIRS) if suffix == scores_suffix
    )
    if structure_index != scores_index:
        raise ValueError("structure_path and scores_path must use the same naming profile")
    structure_root = structure_name[: -len(structure_suffix)]
    scores_root = scores_name[: -len(scores_suffix)]
    if not structure_root or not scores_root:
        raise ValueError("structure_path and scores_path must contain a model entity ID root")
    return structure_root, scores_root


__all__ = [
    "PredictionPair",
    "PredictionScoresPayload",
    "prediction_pair_from_mapping",
    "prediction_scores_payload_from_mapping",
]
