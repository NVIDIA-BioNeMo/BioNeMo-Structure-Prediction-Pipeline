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

"""Explicit mixed BioIR science, with frozen policy-free serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from bspp.orchestration.contract.folding_bioir import (
    BIOIR_MONOMER_TOOL_USED,
    BIOIR_MULTIMER_TOOL_USED,
    BioIRModelPolicy,
    bioir_model_policy_from_mapping,
)
from bspp.orchestration.contract.folding_execution import (
    FoldingBackendAssetsSnapshot,
    folding_backend_assets_snapshot_from_mapping,
)
from bspp.orchestration.contract.phase import (
    FoldingPhasePlan,
    FoldingPhaseRunSpec,
    canonical_mapping_digest,
    folding_phase_plan_from_mapping,
    folding_phase_plan_payload_from_mapping,
    folding_phase_runspec_from_mapping,
    folding_phase_runspec_payload_from_mapping,
)
from bspp.orchestration.contract.phase_retry import (
    compare_retry_invariants,
    phase_input_set_identity_digest,
    phase_scientific_identity_digest,
)
from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from tests.test_phase_folding_contract import make_plan, make_runspec


def policy() -> BioIRModelPolicy:
    return BioIRModelPolicy(
        monomer_checkpoint_sha256="1" * 64,
        monomer_checkpoint_size_bytes=123,
        multimer_checkpoint_sha256="2" * 64,
        multimer_checkpoint_size_bytes=456,
    )


def bioir_plan(*, explicit: bool = True) -> FoldingPhasePlan:
    original = make_plan()
    return replace(
        original,
        payload=replace(original.payload, backend="bioir", bioir_model_policy=policy() if explicit else None),
    )


def bioir_runspec(plan: FoldingPhasePlan) -> FoldingPhaseRunSpec:
    old = make_runspec(plan)
    assets = FoldingBackendAssetsSnapshot(
        backend="bioir",
        bioir_checkpoint="/weights/multimer.pt",
        bioir_monomer_checkpoint="/weights/monomer.pt" if plan.payload.bioir_model_policy else None,
    )
    return replace(
        old,
        cluster=replace(old.cluster, backend_assets=assets),
        payload=replace(old.payload, bioir_model_policy=plan.payload.bioir_model_policy),
    )


def test_policy_strict_round_trip_and_expanded_chain_routing() -> None:
    value = policy()
    assert bioir_model_policy_from_mapping(value.to_mapping()) == value
    assert value.digest == canonical_mapping_digest(value.to_mapping())
    assert value.model_source_for_chain_count(1) == "openfold2_ptm_1"
    # A homodimer has one unique polymer but two expanded chains.
    assert [value.model_source_for_chain_count(n) for n in (2, 3, 4)] == ["alphafold2_multimer_1"] * 3


@pytest.mark.parametrize("count", [0, -1, True, 1.0, "1", None])
def test_policy_rejects_nonpositive_or_noninteger_chain_counts(count: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        policy().model_source_for_chain_count(count)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", "1"),
        ("policy", "infer-from-sequence"),
        ("monomer_model_source", "alphafold2_1"),
        ("multimer_model_source", "alphafold2_multimer_2"),
        ("monomer_model_source", None),
        ("monomer_checkpoint_sha256", "A" * 64),
        ("monomer_checkpoint_sha256", "a" * 63),
        ("multimer_checkpoint_sha256", "g" * 64),
        ("multimer_checkpoint_sha256", None),
        ("monomer_checkpoint_size_bytes", 0),
        ("monomer_checkpoint_size_bytes", True),
        ("monomer_checkpoint_size_bytes", "123"),
        ("multimer_checkpoint_size_bytes", -1),
        ("multimer_checkpoint_size_bytes", 1.5),
    ],
)
def test_policy_rejects_invalid_version_models_and_content_identity(field: str, value: object) -> None:
    mapping = policy().to_mapping()
    mapping[field] = value
    with pytest.raises(ValueError):
        bioir_model_policy_from_mapping(mapping)


@pytest.mark.parametrize("field", tuple(policy().to_mapping()))
def test_every_policy_field_is_explicit(field: str) -> None:
    mapping = policy().to_mapping()
    del mapping[field]
    with pytest.raises(ValueError, match="fields differ"):
        bioir_model_policy_from_mapping(mapping)


def test_policy_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown"):
        bioir_model_policy_from_mapping({**policy().to_mapping(), "config_override": "unsafe"})


def test_new_plan_runspec_round_trip_and_content_scientific_identity() -> None:
    old = bioir_plan(explicit=False)
    plan = bioir_plan()
    runspec = bioir_runspec(plan)
    assert folding_phase_plan_from_mapping(plan.to_mapping()) == plan
    assert folding_phase_runspec_from_mapping(runspec.to_mapping()) == runspec
    assert phase_input_set_identity_digest(old) == phase_input_set_identity_digest(plan)
    assert phase_scientific_identity_digest(old) != phase_scientific_identity_digest(plan)
    changed_policy = replace(policy(), monomer_checkpoint_sha256="3" * 64)
    changed = replace(plan, payload=replace(plan.payload, bioir_model_policy=changed_policy))
    assert phase_input_set_identity_digest(changed) == phase_input_set_identity_digest(plan)
    assert phase_scientific_identity_digest(changed) != phase_scientific_identity_digest(plan)


@pytest.mark.parametrize("backend", ["openfold-cli", "colabfold", "openfold-trt"])
def test_other_backends_cannot_adopt_bioir_policy(backend: str) -> None:
    plan = make_plan()
    with pytest.raises(ValueError, match="requires the bioir backend"):
        replace(plan.payload, backend=backend, bioir_model_policy=policy())
    with pytest.raises(ValueError, match="requires the bioir backend"):
        replace(make_runspec().payload, backend=backend, bioir_model_policy=policy())


@pytest.mark.parametrize("value", [None, [], "policy"])
def test_present_policy_must_be_a_mapping(value: object) -> None:
    plan_mapping = bioir_plan().payload.to_mapping()
    plan_mapping["bioir_model_policy"] = value
    with pytest.raises(ValueError):
        folding_phase_plan_payload_from_mapping(plan_mapping)
    run_mapping = bioir_runspec(bioir_plan()).payload.to_mapping()
    run_mapping["bioir_model_policy"] = value
    with pytest.raises(ValueError):
        folding_phase_runspec_payload_from_mapping(run_mapping)


def test_policy_and_monomer_asset_must_be_selected_together() -> None:
    current = bioir_runspec(bioir_plan())
    assets = current.cluster.backend_assets
    assert assets is not None
    assert folding_backend_assets_snapshot_from_mapping(assets.to_mapping()) == assets
    with pytest.raises(ValueError, match="requires bioir_monomer_checkpoint"):
        replace(current, cluster=replace(current.cluster, backend_assets=None))
    with pytest.raises(ValueError, match="requires bioir_monomer_checkpoint"):
        replace(
            current, cluster=replace(current.cluster, backend_assets=replace(assets, bioir_monomer_checkpoint=None))
        )
    with pytest.raises(ValueError, match="requires an explicit BioIR model policy"):
        replace(current, payload=replace(current.payload, bioir_model_policy=None))


@pytest.mark.parametrize("path", ["relative.pt", "/weights/monomer.npz", ""])
def test_monomer_checkpoint_asset_has_explicit_absolute_pt_path(path: str) -> None:
    with pytest.raises(ValueError):
        FoldingBackendAssetsSnapshot(backend="bioir", bioir_checkpoint="/multi.pt", bioir_monomer_checkpoint=path)


def test_monomer_asset_is_not_allowed_for_another_backend() -> None:
    with pytest.raises(ValueError, match="does not accept"):
        FoldingBackendAssetsSnapshot(
            backend="colabfold",
            chain_manifest_csv="/chains.csv",
            colabfold_weights_dir="/weights",
            bioir_monomer_checkpoint="/mono.pt",
        )


def test_retry_preserves_policy_content_while_allowing_operational_path_change() -> None:
    plan = bioir_plan()
    predecessor = bioir_runspec(plan)
    assets = predecessor.cluster.backend_assets
    assert assets is not None
    successor = replace(
        predecessor,
        attempt_id="attempt-0002",
        cluster=replace(
            predecessor.cluster, backend_assets=replace(assets, bioir_monomer_checkpoint="/replica/monomer.pt")
        ),
    )
    compare_retry_invariants(plan, predecessor, successor)
    changed = replace(
        successor,
        payload=replace(successor.payload, bioir_model_policy=replace(policy(), multimer_checkpoint_size_bytes=999)),
    )
    with pytest.raises(ValueError, match="non-allowlisted"):
        compare_retry_invariants(plan, predecessor, changed)


def test_tool_vocabulary_is_additive_and_honest() -> None:
    assert VALID_TOOL_USED[:4] == (
        "ColabFold v1.6.0 / AlphaFold-Multimer",
        "OpenFold-TRT / AlphaFold-Multimer",
        "OpenFold / AlphaFold-Multimer",
        BIOIR_MULTIMER_TOOL_USED,
    )
    assert VALID_TOOL_USED[4:] == (BIOIR_MONOMER_TOOL_USED,)


# Independently captured from clean 64f71776, before Contract edits. The formatted
# mapping hashes freeze complete bytes, not just an implementation round trip.
@pytest.mark.parametrize(
    ("backend", "plan_digest", "runspec_digest", "plan_bytes", "runspec_bytes", "input_digest", "science_digest"),
    [
        (
            "openfold-cli",
            "0f3f7d1d49590542d2ba8ebeddc551a70572badcb014bcc622609c36f7014c4d",
            "3ea72cd988dbc80d9aa2dd03b77429e75cf23fc3a2ba3314d3ffa91b1f60fe6f",
            "3ab5c0da12fba54656f66fdf92143c85f07fb7737f643fb9ca0f3b2285582cc4",
            "4cc67dde1666831de4ad34b5f991f6ea9312119c0c23c5978c65db7af76e9e0d",
            "862ed44add48bfcbb77cab5516604ebca9ee2323c0bfc5c7c542fad0935080cb",
            "019b10048a34df2673d799cbedb93953b3fe250c13fc16788b3def37f2483192",
        ),
        (
            "bioir",
            "8991764488d3f5001351a7c88552e9c561f2c4bb5c88adeb1d53f01b8bfa64a5",
            "e56268e482d66a60d8d59a1d461d4eea4464573b877a399a9cb87de8331cdca4",
            "d722f1c291d7f1ed8e89e1d75d28f3171b7cd21ac8c5cc840935d818d5875250",
            "185d19e9529215c62cec6fcef1a6ee9c53917165644f58046804deec3ece7c93",
            "f649c7a77422bd95b6a60edb8f354d9654d2c74811a9e9044fbdd498de4fb1d0",
            "2b8aa1a8f690c7f1464a47eb5138270532eb12f9a0749e10d24034bd8dffffbe",
        ),
    ],
)
def test_policy_absence_preserves_complete_legacy_mapping_and_identities(
    backend: str,
    plan_digest: str,
    runspec_digest: str,
    plan_bytes: str,
    runspec_bytes: str,
    input_digest: str,
    science_digest: str,
) -> None:
    original = make_plan()
    plan = replace(original, payload=replace(original.payload, backend=backend))
    runspec = make_runspec(plan)
    for record, byte_digest in ((plan, plan_bytes), (runspec, runspec_bytes)):
        raw = json.dumps(record.to_mapping(), sort_keys=True, indent=2).encode()
        assert hashlib.sha256(raw).hexdigest() == byte_digest
        assert b"bioir_model_policy" not in raw
        assert b"bioir_monomer_checkpoint" not in raw
    assert plan.digest == plan_digest
    assert runspec.digest == runspec_digest
    assert folding_phase_plan_from_mapping(plan.to_mapping()) == plan
    assert folding_phase_runspec_from_mapping(runspec.to_mapping()) == runspec
    assert phase_input_set_identity_digest(plan) == input_digest
    assert phase_scientific_identity_digest(plan) == science_digest
