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

"""HumanSTRING compatibility over tiny synthetic A3M fixtures, without inference."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
from tests.test_folding.test_legacy_msa_import import _build_legacy_handoff

from bspp.orchestration.contract.folding_bioir import BIOIR_MULTIMER_TOOL_USED, BioIRModelPolicy
from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.prediction_pair import (
    PredictionPair,
    PredictionScoresPayload,
    prediction_pair_from_mapping,
)
from bspp.orchestration.runtime.folding.execution.emitter_support import canonical_pair_names, select_tool_used
from bspp.orchestration.runtime.folding.execution.errors import FoldingBackendError
from bspp.orchestration.runtime.folding.execution.models import ProteinTarget, require_valid_target_identity
from bspp.orchestration.runtime.folding.execution.msa_preparation import prepare_projected_msa
from bspp.orchestration.runtime.folding.legacy_msa_import import run_legacy_msa_import


@pytest.mark.parametrize(
    "target_id,chains,a3m",
    [
        ("homo_P12345", ("ACD", "ACD"), "#3\t2\n>query\nACD\n>hit\nAC-\n"),
        ("hetero_Q9Y6K9_P12345", ("ACD", "GG"), "#3,2\t1,1\n>query\nACDGG\n>hit\nAC-G-\n"),
        ("hetero_Q9Y6K9_P12345", ("ACD", "ACD"), "#3\t2\n>query\nACD\n>hit\nAC-\n"),
    ],
)
def test_preparation_and_canonical_pair_preserve_original_identity(
    tmp_path: Path, target_id: str, chains: tuple[str, ...], a3m: str
) -> None:
    target = ProteinTarget(target_id, "synthetic identity compatibility fixture", chains)
    assert require_valid_target_identity(target) == target_id
    policy = BioIRModelPolicy(
        monomer_checkpoint_sha256="1" * 64,
        monomer_checkpoint_size_bytes=1,
        multimer_checkpoint_sha256="2" * 64,
        multimer_checkpoint_size_bytes=1,
    )
    assert policy.model_source_for_chain_count(len(target.chains)) == "alphafold2_multimer_1"
    logical = f"a3ms/{target_id}.a3m"
    source = tmp_path / f"{target_id}.a3m"
    source.write_text(a3m)
    consumption = MsaSetConsumption("sha256:" + "a" * 64, 1, (logical,), True)
    prepared = prepare_projected_msa(consumption, {logical: source}, target, tmp_path / "split")
    assert tuple(chain.query_sequence for chain in prepared.chains) == chains
    assert prepared.metadata["selected_logical_path"] == logical
    structure, scores = canonical_pair_names(target_id)
    assert structure == f"{target_id}-model_v1.pdb"
    assert scores == f"{target_id}-meta_v1.json"
    pair = PredictionPair(
        target_id,
        BIOIR_MULTIMER_TOOL_USED,
        structure,
        scores,
        PredictionScoresPayload((90.0,), ((0.0,),), 0.0, None, None, {}),
    )
    assert prediction_pair_from_mapping(pair.to_mapping()).to_mapping() == pair.to_mapping()


@pytest.mark.parametrize(
    "target_id,chains",
    [
        ("homo_P12345", ("AA",)),
        ("homo_P12345", ("AA", "BB")),
        ("hetero_P12345_Q9Y6K9", ("AA", "BB", "CC")),
        ("hetero_P12345_Q9Y6K9", ("AA", "")),
    ],
)
def test_new_identity_requires_consistent_expanded_dimer(target_id: str, chains: tuple[str, ...]) -> None:
    with pytest.raises(FoldingBackendError, match="invalid model entity identity"):
        require_valid_target_identity(ProteinTarget(target_id, "", chains))


@pytest.mark.parametrize("target_id", ["AF-0000000000000001", "pdb_7amq_assembly_1"])
def test_legacy_target_shape_unchanged(target_id: str) -> None:
    assert require_valid_target_identity(ProteinTarget(target_id, "", ("AA",))) == target_id


@pytest.mark.parametrize(
    "target_id,leaked,expected",
    [
        ("homo_P12345", False, "OpenFold / AlphaFold-Multimer"),
        ("homo_P12345", True, "OpenFold / AlphaFold-Multimer"),
        ("hetero_P12345_Q9Y6K9", False, "ColabFold v1.6.0 / AlphaFold-Multimer"),
        ("hetero_P12345_Q9Y6K9", True, "OpenFold / AlphaFold-Multimer"),
    ],
)
def test_explicit_human_tool_classification(target_id: str, leaked: bool, expected: str) -> None:
    assert (
        select_tool_used(
            target_id,
            leaked_homodimer=leaked,
            homodimer_tool_used="OpenFold / AlphaFold-Multimer",
            heterodimer_tool_used="ColabFold v1.6.0 / AlphaFold-Multimer",
        )
        == expected
    )


@pytest.mark.skipif(shutil.which("lz4") is None, reason="real tar/LZ4 import fixture")
def test_public_import_keeps_original_human_names_and_payloads(tmp_path: Path) -> None:
    members = {
        "homo_P12345.a3m": b"#3\t2\n>query\nACD\n",
        "hetero_Q9Y6K9_P12345.a3m": b"#3,2\t1,1\n>query\nACDGG\n",
    }
    handoff, legacy = _build_legacy_handoff(tmp_path, members_data=members)
    paths = [path for path in handoff.rglob("*") if path.is_file()]
    paths += [Path(legacy.artifact_location.tar_path), Path(legacy.artifact_location.bundle_path)]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    imported = run_legacy_msa_import(handoff, tmp_path / "enriched")
    assert imported.member_lengths == (6, 5)
    assert imported.enriched_manifest.chunks == legacy.artifact_set.chunks
    assert imported.rebound_location.members == legacy.artifact_location.members
    assert tuple(member.logical_path for member in imported.rebound_location.members) == tuple(
        f"a3ms/{name}" for name in members
    )
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
