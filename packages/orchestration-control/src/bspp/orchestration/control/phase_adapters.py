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

"""Exact Phase-family dispatch helpers for additive lifecycle adapters."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import yaml

from bspp.orchestration.control.phase_authority import validate_phase_run_id

if TYPE_CHECKING:
    from bspp.orchestration.contract.phase import FoldingPhaseRunSpec, PhaseRunSpec

PhaseFamily = Literal["preprocessing", "postprocessing", "folding"]


def phase_family_from_mapping(payload: Mapping[str, object], *, record_name: str) -> PhaseFamily:
    """Return only a recognized explicit Phase family discriminator."""
    value = payload.get("phase_kind")
    if value not in {"preprocessing", "postprocessing", "folding"}:
        raise ValueError(f"{record_name} requires phase_kind 'preprocessing', 'postprocessing', or 'folding'")
    return cast("PhaseFamily", value)


def phase_runspec_family(runspec: PhaseRunSpec | FoldingPhaseRunSpec) -> PhaseFamily:
    """Return the exact Phase family for one already-loaded RunSpec record."""
    from bspp.orchestration.contract.phase import FoldingPhaseRunSpec

    return "folding" if isinstance(runspec, FoldingPhaseRunSpec) else "preprocessing"


def phase_plan_family(path: Path) -> PhaseFamily:
    payload = _mapping(path, "Phase Plan")
    return phase_family_from_mapping(payload, record_name="Phase Plan")


def phase_authority_family(authority_root: Path, phase_run_id: str) -> PhaseFamily:
    validate_phase_run_id(phase_run_id)
    authority = authority_root / phase_run_id
    if authority.is_symlink() or not authority.is_dir():
        raise ValueError(f"missing Phase Run authority directory: {authority}")
    payload = _mapping(authority / "phase-plan.json", "stored Phase Plan")
    return phase_family_from_mapping(payload, record_name="stored Phase Plan")


def _mapping(path: Path, name: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")
    payload = yaml.safe_load(path.read_bytes())
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected {name} mapping in {path}")
    return payload


__all__ = [
    "PhaseFamily",
    "phase_authority_family",
    "phase_family_from_mapping",
    "phase_plan_family",
    "phase_runspec_family",
]
