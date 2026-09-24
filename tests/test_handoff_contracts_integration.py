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

"""Cross-story integration guards for the phase-0 handoff contracts.

These guards pin the three coherence properties that the Stage-10
integration pass exists to close:

* every public name in the five handoff modules is re-exported from the
  package root with the same object identity (B1);
* all eight versioned loaders reject an explicit JSON ``schema_version: null``
  uniformly (B3);
* the runtime constants remain direct name bindings to the contract
  ``model_identity`` surface.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import ModuleType

import pytest

import bspp.orchestration.contract as contract
from bspp.orchestration.contract import (
    folding_input,
    master_parquet_projection,
    model_identity,
    prediction_bundle,
    prediction_pair,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    MsaArtifactSetManifest,
    MsaChunkManifestReference,
    msa_artifact_set_id,
)
from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION
from bspp.orchestration.runtime import constants as runtime_constants

_HANDOFF_MODULES = (
    folding_input,
    master_parquet_projection,
    model_identity,
    prediction_bundle,
    prediction_pair,
)


@pytest.mark.parametrize(
    "module",
    _HANDOFF_MODULES,
    ids=[module.__name__.rsplit(".", 1)[-1] for module in _HANDOFF_MODULES],
)
def test_package_root_reexports_every_public_name(module: ModuleType) -> None:
    for name in module.__all__:
        assert name in contract.__all__, f"{module.__name__}.{name} missing from contract.__all__"
        assert getattr(contract, name) is getattr(module, name), (
            f"{module.__name__}.{name} is not the same object as contract.{name}"
        )


def _scores_mapping(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "plddt": [90.0, 85.0, 80.0],
        "pae": [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
        "max_pae": 3.0,
        "ptm": 0.9,
        "iptm": 0.8,
    }
    payload.update(overrides)
    return payload


def _pair_mapping(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "model_entity_id": "AF-0000000000000001",
        "tool_used": VALID_TOOL_USED[0],
        "structure_path": "/data/AF-0000000000000001-model_v1.pdb",
        "scores_path": "/data/AF-0000000000000001-meta_v1.json",
        "scores": _scores_mapping(),
    }
    payload.update(overrides)
    return payload


def _bundle_mapping(**overrides: object) -> dict[str, object]:
    bundle = prediction_bundle.PredictionArchiveBundle(
        bundle_name="bspp_260903_1234_a00001.tar.lz4",
        member_ids=("AF-0000000000000001",),
        member_count=1,
        sha256="a" * 64,
        size_bytes=100,
        created_at=None,
    )
    payload = bundle.to_mapping()
    payload.update(overrides)
    return payload


def _folding_layout_mapping(**overrides: object) -> dict[str, object]:
    layout = folding_input.FoldingInputLayout(
        layout="openfold",
        chain_ids=("A", "B"),
        template_mode="none",
        pairing="species-from-colabfold-headers",
    )
    payload = layout.to_mapping()
    payload.update(overrides)
    return payload


def _bioir_polymer_mapping(**overrides: object) -> dict[str, object]:
    polymer = folding_input.BioIRPolymer(("A",), "SEQ", "alignments/polymer_00/unpaired.a3m")
    payload = polymer.to_mapping()
    payload.update(overrides)
    return payload


def _bioir_request_mapping(**overrides: object) -> dict[str, object]:
    request = folding_input.BioIRRequestManifest(
        input_id="input-1",
        polymers=(folding_input.BioIRPolymer(("A",), "SEQ", "alignments/polymer_00/unpaired.a3m"),),
    )
    payload = request.to_mapping()
    payload.update(overrides)
    return payload


def _msa_consumption_mapping(**overrides: object) -> dict[str, object]:
    reference = MsaChunkManifestReference(
        chunk_name="sample_tranche00_00001.fa",
        logical_path="chunks/sample_tranche00_00001.json",
        sha256="a" * 64,
        member_count=1,
        logical_bytes=100,
    )
    manifest = MsaArtifactSetManifest(
        artifact_set_id=msa_artifact_set_id((reference,), 1, 100),
        chunks=(reference,),
        member_count=1,
        logical_bytes=100,
    )
    consumption = folding_input.MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=("a3ms/AFDB_AF-0000000000000001.a3m",),
        requires_paired_query_header=True,
    )
    payload = consumption.to_mapping()
    payload.update(overrides)
    return payload


def _a3m_rules_mapping(**overrides: object) -> dict[str, object]:
    rules = folding_input.A3mSplitRules()
    payload = rules.to_mapping()
    payload.update(overrides)
    return payload


VERSIONED_LOADERS: tuple[
    tuple[Callable[[Mapping[str, object]], object], Callable[..., dict[str, object]]],
    ...,
] = (
    (prediction_pair.prediction_pair_from_mapping, _pair_mapping),
    (prediction_pair.prediction_scores_payload_from_mapping, _scores_mapping),
    (prediction_bundle.prediction_archive_bundle_from_mapping, _bundle_mapping),
    (folding_input.folding_input_layout_from_mapping, _folding_layout_mapping),
    (folding_input.bioir_polymer_from_mapping, _bioir_polymer_mapping),
    (folding_input.bioir_request_manifest_from_mapping, _bioir_request_mapping),
    (folding_input.msa_set_consumption_from_mapping, _msa_consumption_mapping),
    (folding_input.a3m_split_rules_from_mapping, _a3m_rules_mapping),
)


@pytest.mark.parametrize(
    ("loader", "mapping_factory"),
    VERSIONED_LOADERS,
    ids=[loader.__name__ for loader, _ in VERSIONED_LOADERS],
)
def test_versioned_loaders_reject_explicit_null_schema_version(
    loader: Callable[[Mapping[str, object]], object],
    mapping_factory: Callable[..., dict[str, object]],
) -> None:
    payload = mapping_factory(schema_version=None)
    with pytest.raises(ValueError, match="schema_version"):
        loader(payload)


def test_adr_0025_runtime_identity_bindings() -> None:
    assert runtime_constants.KNOWN_SUFFIXES is model_identity.KNOWN_SUFFIXES
    assert runtime_constants.MAX_PROTEINS_PER_SHARD is model_identity.MAX_PROTEINS_PER_SHARD
