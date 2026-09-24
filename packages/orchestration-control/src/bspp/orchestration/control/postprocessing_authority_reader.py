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

"""Version routing for readable postprocessing authority."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from bspp.orchestration.contract.postprocessing_runspec import read_postprocessing_phase_runspec_from_mapping
from bspp.orchestration.contract.postprocessing_runspec_v1 import HistoricalPostprocessingPhaseRunSpecV1
from bspp.orchestration.contract.postprocessing_runspec_v3 import PostprocessingPhaseRunSpecV3
from bspp.orchestration.control.postprocessing_authority_store import read_canonical_json, required_string
from bspp.orchestration.control.postprocessing_authority_v1 import (
    validate_historical_postprocessing_authority_v1,
)
from bspp.orchestration.control.postprocessing_authority_v2 import validate_postprocessing_authority
from bspp.orchestration.control.postprocessing_phase_types import (
    PostprocessingAuthority,
    ReadablePostprocessingAuthority,
)

PostprocessingAuthorityContractVersion = Literal[1, 2, 3]
HISTORICAL_V1_MUTATION_ERROR = (
    "historical postprocessing V1 authority is read-only; rematerialize from the Phase Plan to execute"
)


def postprocessing_authority_contract_version(
    authority_root: Path,
    phase_run_id: str,
) -> PostprocessingAuthorityContractVersion:
    """Strictly identify the initial persisted RunSpec family without mutation."""
    authority_path = authority_root / phase_run_id
    run_mapping = read_canonical_json(authority_path / "phase-run.json")
    initial_attempt_id = required_string(run_mapping, "current_attempt_id")
    runspec_mapping = read_canonical_json(authority_path / "attempts" / initial_attempt_id / "phase-runspec.json")
    runspec = read_postprocessing_phase_runspec_from_mapping(runspec_mapping)
    if isinstance(runspec, HistoricalPostprocessingPhaseRunSpecV1):
        return 1
    return 3 if isinstance(runspec, PostprocessingPhaseRunSpecV3) else 2


def read_postprocessing_authority(
    authority_root: Path,
    phase_run_id: str,
) -> ReadablePostprocessingAuthority:
    """Validate either supported postprocessing authority family for inspection."""
    if postprocessing_authority_contract_version(authority_root, phase_run_id) == 1:
        return validate_historical_postprocessing_authority_v1(authority_root, phase_run_id)
    return validate_postprocessing_authority(authority_root, phase_run_id)


def require_postprocessing_v2_authority(
    authority_root: Path,
    phase_run_id: str,
) -> PostprocessingAuthority:
    """Reject V1 before any lifecycle operation can write or reach transport."""
    if postprocessing_authority_contract_version(authority_root, phase_run_id) == 1:
        raise ValueError(HISTORICAL_V1_MUTATION_ERROR)
    return validate_postprocessing_authority(authority_root, phase_run_id)


def reject_historical_postprocessing_mutation(authority_root: Path, phase_run_id: str) -> None:
    """Apply the stable V1 mutation barrier before operation-specific work."""
    if postprocessing_authority_contract_version(authority_root, phase_run_id) == 1:
        raise ValueError(HISTORICAL_V1_MUTATION_ERROR)


__all__ = [
    "HISTORICAL_V1_MUTATION_ERROR",
    "PostprocessingAuthorityContractVersion",
    "postprocessing_authority_contract_version",
    "read_postprocessing_authority",
    "reject_historical_postprocessing_mutation",
    "require_postprocessing_v2_authority",
]
