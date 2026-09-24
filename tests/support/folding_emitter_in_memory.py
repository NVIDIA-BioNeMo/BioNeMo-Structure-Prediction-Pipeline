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

"""Test-only in-memory folding backend for the canonical-pair emitter seam.

This backend never runs a real kernel. It writes a deterministic canonical
``-model_v1.pdb`` / ``-meta_v1.json`` pair into ``output_dir`` using the
backend-neutral emitter helpers and returns a single ``StructurePrediction``,
so the emitted pair can be validated against the frozen contract
``prediction_pair.py`` surface end to end.
"""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.contract.model_identity import normalize_model_entity_id
from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from bspp.orchestration.runtime.folding.execution.emitter_support import (
    canonical_pair_names,
    select_tool_used,
    serialize_scores_json,
)
from bspp.orchestration.runtime.folding.execution.models import (
    FoldingResult,
    PreparedInput,
    ProteinTarget,
    StructurePrediction,
)

# Fixed deterministic scores; 81.585 exercises the half-even boundary (81.585 -> 81.58).
_PLDDT = (81.585, 90.123, 75.0)
_PAE = (
    (0.0, 1.234, 2.345),
    (1.234, 0.0, 3.456),
    (2.345, 3.456, 0.0),
)
_MAX_PAE = 3.456
_PTM = 0.8765
_IPTM = 0.8125


class InMemoryFoldingBackend:
    """Deterministic, non-scientific folding backend for the emitter seam."""

    name = "in-memory"

    def run(
        self,
        target: ProteinTarget,
        prepared: PreparedInput,
        output_dir: Path,
    ) -> FoldingResult:
        del prepared  # in-memory backend consumes no prepared inputs
        output_dir.mkdir(parents=True, exist_ok=True)
        structure_name, scores_name = canonical_pair_names(target.target_id)
        structure_path = output_dir / structure_name
        scores_path = output_dir / scores_name

        normalized_id = normalize_model_entity_id(target.target_id)
        structure_path.write_text(f"HEADER    {normalized_id}\nEND\n", encoding="utf-8")
        scores_path.write_text(
            serialize_scores_json(
                plddt=_PLDDT,
                pae=_PAE,
                max_pae=_MAX_PAE,
                ptm=_PTM,
                iptm=_IPTM,
                decimals=2,
            ),
            encoding="utf-8",
        )

        tool_used = select_tool_used(
            target.target_id,
            # The fixture target is a homodimer form, so the flag is unused;
            # real callers resolve it from the external chain manifest.
            leaked_homodimer=False,
            homodimer_tool_used=VALID_TOOL_USED[0],
            heterodimer_tool_used=VALID_TOOL_USED[1],
        )
        return FoldingResult(
            backend=self.name,
            predictions=(
                StructurePrediction(
                    rank=1,
                    structure_path=structure_path,
                    scores_path=scores_path,
                ),
            ),
            metadata={"tool_used": tool_used},
        )
