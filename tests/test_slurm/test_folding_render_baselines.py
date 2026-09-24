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

"""Golden baseline tests for the scalar folding Slurm render."""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.contract.phase_submission import PhaseSubmissionIntendedPayload
from bspp.orchestration.control.folding_phase_adapter import render_folding_submission_intent
from tests.test_phase_folding_materialization import _materialized
from tests.test_slurm.test_render_baselines import (
    RenderOutputs,
    _assert_outputs_match_baselines,
    _normalize_script,
)

_RUNSPEC_LOCATION = "attempts/attempt-0001/phase-runspec.json"
_RUNSPEC_DOCUMENT_SHA256 = "e" * 64


def test_folding_scalar_render_matches_normalized_baselines(tmp_path: Path) -> None:
    outputs = _render_folding_outputs(tmp_path)

    _assert_outputs_match_baselines("folding_scalar", outputs)


def test_folding_scalar_render_is_deterministic() -> None:
    first = _render_folding_intent()
    second = _render_folding_intent()

    assert [plan.script_body for plan in first.actions] == [plan.script_body for plan in second.actions]
    assert first.submission_id == second.submission_id
    assert first.qualification_tuple_id == second.qualification_tuple_id


def _render_folding_intent() -> PhaseSubmissionIntendedPayload:
    runspec = _materialized()
    fold_action = next(action for action in runspec.payload.actions if action.action_kind == "fold")
    assert fold_action.resources.array is None  # scalar, non-array (parity oracle)

    return render_folding_submission_intent(
        phase_runspec=runspec,
        phase_runspec_location=_RUNSPEC_LOCATION,
        phase_runspec_document_sha256=_RUNSPEC_DOCUMENT_SHA256,
    )


def _render_folding_outputs(tmp_path: Path) -> RenderOutputs:
    intent = _render_folding_intent()
    return {f"{plan.action_id}.sbatch": _normalize_script(plan.script_body, tmp_path) for plan in intent.actions}
