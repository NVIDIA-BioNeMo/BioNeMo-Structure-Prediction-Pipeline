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

"""Checksum-pinned validation-suite model for folding benchmark runs.

Track-C port of the validation-suite model from ``frozen reference
pipeline`` (``src/afdb_pipeline/validation.py``). This module owns only the
suite-side half of the port: the frozen ``ValidationCase`` and
``ValidationSuite`` dataclasses and the fail-closed ``load_validation_suite``
loader. The run-side checks (``validate_run`` and per-case verification) land in
later stories and are intentionally out of scope here.

Track-C deviations from the reference (authoritative):
- ``thresholds`` is a flat ``Mapping[str, float]`` rather than the reference's
  nested min/max bounds.
- ``ValidationSuite`` carries ``schema_version``, ``dataset_id``,
  ``fingerprint``, ``cases``, and a trailing ``checksum``.
- ``expected_pair_mode``, ``reference_structure``, and ``reference_sha256`` are
  required non-empty strings.
- The checksum is computed over the raw JSON bytes, not a canonical
  re-serialization.
- Fail-closed errors are plain ``ValueError``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeGuard, cast

VALIDATION_SUITE_SCHEMA_VERSION = 1

_HEX_DIGITS = "0123456789abcdef"


def _digest(value: object, name: str) -> str:
    """Validate and normalize a 64-character SHA-256 hex digest."""
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    lowered = value.lower()
    if any(character not in _HEX_DIGITS for character in lowered):
        raise ValueError(f"{name} must be a hexadecimal SHA-256 digest")
    return lowered


def _is_finite_number(value: object) -> TypeGuard[float | int]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _require_nonempty_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_str(payload: Mapping[str, object], key: str) -> str:
    return _require_nonempty_str(payload.get(key), key)


def _require_bool(payload: Mapping[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _require_mapping(
    payload: Mapping[str, object],
    key: str,
    default: Mapping[str, object],
) -> Mapping[str, object]:
    value = payload.get(key, default)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return cast("Mapping[str, object]", value)


def _require_schema_version(payload: Mapping[str, object]) -> int:
    value = payload.get("schema_version")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("schema_version must be an integer")
    if value != VALIDATION_SUITE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported validation suite schema_version {value!r}; "
            f"supported version: {VALIDATION_SUITE_SCHEMA_VERSION}"
        )
    return value


@dataclass(frozen=True)
class ValidationCase:
    """One validation target with its expected identity and quality thresholds."""

    target_id: str
    sequence_sha256: str
    thresholds: Mapping[str, float]
    require_no_nan: bool
    expected_pair_mode: str
    reference_structure: str
    reference_sha256: str
    chain_map: Mapping[str, str]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_nonempty_str(self.target_id, "target_id")
        object.__setattr__(self, "sequence_sha256", _digest(self.sequence_sha256, "sequence_sha256"))
        _require_nonempty_str(self.expected_pair_mode, "expected_pair_mode")
        _require_nonempty_str(self.reference_structure, "reference_structure")
        _require_nonempty_str(self.reference_sha256, "reference_sha256")
        if not isinstance(self.require_no_nan, bool):
            raise ValueError("require_no_nan must be a boolean")
        if not isinstance(self.thresholds, Mapping):
            raise ValueError("thresholds must be a mapping")
        coerced_thresholds: dict[str, float] = {}
        for key, value in self.thresholds.items():
            if not isinstance(key, str) or not key:
                raise ValueError("thresholds keys must be non-empty strings")
            if not _is_finite_number(value):
                raise ValueError(f"thresholds[{key!r}] must be a finite number")
            coerced_thresholds[key] = float(value)
        object.__setattr__(self, "thresholds", MappingProxyType(coerced_thresholds))
        if not isinstance(self.chain_map, Mapping):
            raise ValueError("chain_map must be a mapping")
        object.__setattr__(self, "chain_map", MappingProxyType(dict(self.chain_map)))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ValidationSuite:
    """An immutable, checksum-pinned collection of validation cases."""

    schema_version: int
    dataset_id: str
    fingerprint: str
    cases: tuple[ValidationCase, ...]
    checksum: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != VALIDATION_SUITE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported validation suite schema_version {self.schema_version!r}; "
                f"supported version: {VALIDATION_SUITE_SCHEMA_VERSION}"
            )
        _require_nonempty_str(self.dataset_id, "dataset_id")
        _require_nonempty_str(self.fingerprint, "fingerprint")
        if (
            not isinstance(self.cases, tuple)
            or not self.cases
            or not all(isinstance(case, ValidationCase) for case in self.cases)
        ):
            raise ValueError("cases must be a non-empty tuple of ValidationCase values")


def _case_from_mapping(case: Mapping[str, object]) -> ValidationCase:
    """Build a validated case from a parsed mapping (defaults via loader rules)."""
    return ValidationCase(
        target_id=_require_str(case, "target_id"),
        sequence_sha256=_require_str(case, "sequence_sha256"),
        thresholds=cast("Mapping[str, float]", _require_mapping(case, "thresholds", {})),
        require_no_nan=_require_bool(case, "require_no_nan", True),
        expected_pair_mode=_require_str(case, "expected_pair_mode"),
        reference_structure=_require_str(case, "reference_structure"),
        reference_sha256=_require_str(case, "reference_sha256"),
        chain_map=cast("Mapping[str, str]", _require_mapping(case, "chain_map", {})),
        metadata=_require_mapping(case, "metadata", {}),
    )


def load_validation_suite(path: Path) -> ValidationSuite:
    """Load and strictly validate a checksum-pinned validation-suite JSON file.

    Fails closed (``ValueError``) on an absent/unreadable file, malformed JSON,
    a non-object top level, an unknown/missing schema version, or any invalid
    case. The returned suite's ``checksum`` is the SHA-256 of the raw file bytes.
    """
    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read validation suite {source}: {exc}") from exc
    try:
        raw = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed JSON in validation suite {source}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("validation suite must be a JSON object")
    payload = cast("Mapping[str, object]", raw)
    schema_version = _require_schema_version(payload)
    dataset_id = _require_str(payload, "dataset_id")
    fingerprint = _require_str(payload, "fingerprint")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("validation suite must contain at least one case")
    cases: list[ValidationCase] = []
    for index, item in enumerate(raw_cases):
        if not isinstance(item, Mapping):
            raise ValueError(f"validation case {index} must be an object")
        cases.append(_case_from_mapping(cast("Mapping[str, object]", item)))
    checksum = hashlib.sha256(raw_bytes).hexdigest()
    return ValidationSuite(
        schema_version=schema_version,
        dataset_id=dataset_id,
        fingerprint=fingerprint,
        cases=tuple(cases),
        checksum=checksum,
    )


__all__ = [
    "VALIDATION_SUITE_SCHEMA_VERSION",
    "ValidationCase",
    "ValidationSuite",
    "load_validation_suite",
]
