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

"""Version opt-in, scientific stability and frozen legacy evidence envelopes."""

from dataclasses import replace

import pytest

from bspp.orchestration.contract.phase import (
    folding_phase_plan_from_mapping,
    folding_phase_runspec_from_mapping,
)
from bspp.orchestration.contract.phase_retry import (
    compare_retry_invariants,
    phase_input_set_identity_digest,
    phase_scientific_identity_digest,
)
from tests.support.folding_artifact_fixture import fixture
from tests.test_folding_bioir_policy_contract import bioir_plan, bioir_runspec


@pytest.mark.parametrize(
    "explicit,plan_digest,runspec_digest",
    [
        (
            False,
            "8991764488d3f5001351a7c88552e9c561f2c4bb5c88adeb1d53f01b8bfa64a5",
            "3b2b687c1c7def1273f92cd364fc40a7d86b3ec9b57db6f5d483bf40e9899a05",
        ),
        (
            True,
            "e8ad9a0009995ef432913640f510e430f6d18ca21efff0369e9d34927d7de5b2",
            "17b838ee67e54e099d5afdf145c133289a931d7b03773a30c40bb3b50e74745f",
        ),
    ],
)
def test_f246_legacy_mappings_and_digests_are_unchanged(explicit, plan_digest, runspec_digest):
    # Frozen from the unmodified f246 worktree, both existing policy modes.
    plan = bioir_plan(explicit=explicit)
    runspec = bioir_runspec(plan)
    assert "evidence_profile" not in plan.payload.to_mapping()
    assert "evidence_profile" not in runspec.payload.to_mapping()
    assert plan.digest == plan_digest
    assert runspec.digest == runspec_digest
    assert folding_phase_plan_from_mapping(plan.to_mapping()) == plan
    assert folding_phase_runspec_from_mapping(runspec.to_mapping()) == runspec


def test_profile_changes_execution_identity_but_not_inputs_or_science(tmp_path):
    old = bioir_plan()
    plan = replace(old, payload=replace(old.payload, evidence_profile="artifact-backed-v2"))
    assert plan.digest != old.digest
    assert phase_input_set_identity_digest(plan) == phase_input_set_identity_digest(old)
    assert phase_scientific_identity_digest(plan) == phase_scientific_identity_digest(old)
    assert folding_phase_plan_from_mapping(plan.to_mapping()) == plan
    runspec, _ = fixture(tmp_path)
    assert folding_phase_runspec_from_mapping(runspec.to_mapping()) == runspec
    old_actions = tuple(
        replace(
            a, payload=replace(a.payload, params=tuple((k, v) for k, v in a.payload.params if k != "evidence_profile"))
        )
        for a in runspec.payload.actions
    )
    legacy = replace(runspec, payload=replace(runspec.payload, evidence_profile=None, actions=old_actions))
    assert legacy.digest != runspec.digest
    for a, b in zip(legacy.payload.actions, runspec.payload.actions, strict=True):
        assert (a != b) == (a.action_kind in {"fold", "canonical-pair"})


@pytest.mark.parametrize("value", [None, "", "artifact-backed-v3", 2, {}, True])
def test_present_unknown_or_null_profile_fails_closed(value):
    mapping = bioir_plan().to_mapping()
    mapping["payload"]["evidence_profile"] = value
    with pytest.raises(ValueError):
        folding_phase_plan_from_mapping(mapping)


def test_profile_requires_packed_bioir_and_explicit_policy(tmp_path):
    with pytest.raises(ValueError):
        replace(bioir_plan(explicit=False).payload, evidence_profile="artifact-backed-v2")
    with pytest.raises(ValueError):
        replace(bioir_plan().payload, backend="colabfold", evidence_profile="artifact-backed-v2")
    scalar = bioir_runspec(bioir_plan())
    with pytest.raises(ValueError, match="packed"):
        replace(scalar.payload, evidence_profile="artifact-backed-v2")
    runspec, _ = fixture(tmp_path)
    actions = tuple(
        replace(
            a, payload=replace(a.payload, params=tuple((k, v) for k, v in a.payload.params if k != "evidence_profile"))
        )
        if a.action_kind == "fold"
        else a
        for a in runspec.payload.actions
    )
    with pytest.raises(ValueError, match="profile"):
        replace(runspec.payload, actions=actions)


def test_retry_freezes_profile_and_accepts_operational_successor(tmp_path):
    runspec, _ = fixture(tmp_path)
    base = bioir_plan()
    plan = replace(
        base,
        input_location=runspec.input_location,
        payload=replace(
            base.payload,
            msa_set=runspec.payload.msa_set,
            msa_set_manifest=runspec.payload.msa_set_manifest,
            bioir_model_policy=runspec.payload.bioir_model_policy,
            evidence_profile="artifact-backed-v2",
        ),
    )
    runspec = replace(runspec, phase_plan_digest=plan.digest)
    successor = replace(runspec, attempt_id="attempt-0002")
    compare_retry_invariants(plan, runspec, successor)
    actions = tuple(
        replace(
            a, payload=replace(a.payload, params=tuple((k, v) for k, v in a.payload.params if k != "evidence_profile"))
        )
        for a in successor.payload.actions
    )
    legacy = replace(successor, payload=replace(successor.payload, evidence_profile=None, actions=actions))
    with pytest.raises(ValueError, match="non-allowlisted"):
        compare_retry_invariants(plan, runspec, legacy)
