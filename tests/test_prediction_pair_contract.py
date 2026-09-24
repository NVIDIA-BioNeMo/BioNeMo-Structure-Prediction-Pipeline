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

"""Contract tests for the fold-to-postprocessing prediction-pair schema."""

from __future__ import annotations

import math

import pytest

from bspp.orchestration.contract.prediction_pair import (
    PredictionPair,
    PredictionScoresPayload,
    prediction_pair_from_mapping,
    prediction_scores_payload_from_mapping,
)
from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION

MODEL_ID = "AF-0000000000000001"
STRUCTURE_SUFFIXES = (
    "-model_v1.pdb",
    ".merged_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb",
    "_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb",
)
SCORES_SUFFIXES = (
    "-meta_v1.json",
    ".merged_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
    "_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
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
        "model_entity_id": MODEL_ID,
        "tool_used": VALID_TOOL_USED[0],
        "structure_path": f"/data/{MODEL_ID}-model_v1.pdb",
        "scores_path": f"/data/{MODEL_ID}-meta_v1.json",
        "scores": _scores_mapping(),
    }
    payload.update(overrides)
    return payload


def _make_pair(**overrides: object) -> PredictionPair:
    return prediction_pair_from_mapping(_pair_mapping(**overrides))


def test_construction_and_exact_to_mapping() -> None:
    pair = _make_pair()
    assert pair.to_mapping() == {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "model_entity_id": MODEL_ID,
        "tool_used": VALID_TOOL_USED[0],
        "structure_path": f"/data/{MODEL_ID}-model_v1.pdb",
        "scores_path": f"/data/{MODEL_ID}-meta_v1.json",
        "scores": {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "plddt": [90.0, 85.0, 80.0],
            "pae": [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
            "max_pae": 3.0,
            "ptm": 0.9,
            "iptm": 0.8,
        },
    }


def test_from_mapping_round_trip() -> None:
    pair = _make_pair()
    assert prediction_pair_from_mapping(pair.to_mapping()) == pair


def test_explicit_current_schema_versions() -> None:
    pair = _make_pair()
    assert pair.schema_version == CURRENT_CONTRACT_SCHEMA_VERSION
    assert pair.scores.schema_version == CURRENT_CONTRACT_SCHEMA_VERSION


def test_missing_schema_versions_are_rejected() -> None:
    with pytest.raises(ValueError):
        prediction_pair_from_mapping(_without(_pair_mapping(), "schema_version"))
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_without(_scores_mapping(), "schema_version"))


def test_explicit_null_schema_version_rejected() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        prediction_pair_from_mapping(_pair_mapping(schema_version=None))
    with pytest.raises(ValueError, match="schema_version"):
        prediction_scores_payload_from_mapping(_scores_mapping(schema_version=None))


def test_unsupported_schema_versions_are_rejected() -> None:
    with pytest.raises(ValueError):
        prediction_pair_from_mapping(_pair_mapping(schema_version=99))
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_scores_mapping(schema_version=99))


def test_outer_unknown_field_rejection() -> None:
    with pytest.raises(ValueError):
        prediction_pair_from_mapping(_pair_mapping(unexpected_field=1))


@pytest.mark.parametrize("missing_key", ["plddt", "pae", "max_pae"])
def test_missing_required_score_keys(missing_key: str) -> None:
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_without(_scores_mapping(), missing_key))


def test_empty_plddt_is_rejected() -> None:
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_scores_mapping(plddt=[]))


def test_non_list_plddt_is_rejected() -> None:
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_scores_mapping(plddt="not-a-list"))
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_scores_mapping(plddt=(90.0, 85.0, 80.0)))


def test_non_list_pae_is_rejected() -> None:
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_scores_mapping(pae="not-a-list"))
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_scores_mapping(pae=((0.0,), (1.0,), (2.0,))))


def test_non_list_pae_rows_are_rejected() -> None:
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_scores_mapping(pae=[[0.0, 1.0], "row", [2.0, 3.0]]))


def test_empty_pae_rows_are_rejected() -> None:
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_scores_mapping(pae=[[], [0.0], [1.0]]))


def test_pae_plddt_row_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_scores_mapping(pae=[[0.0], [1.0]]))


@pytest.mark.parametrize("field", ["plddt", "pae", "max_pae", "ptm", "iptm"])
def test_non_numeric_values_are_rejected(field: str) -> None:
    bad_value: object = "not-a-number"
    overrides = {field: bad_value}
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_scores_mapping(**overrides))


@pytest.mark.parametrize("field", ["plddt", "max_pae", "ptm", "iptm"])
def test_non_finite_values_are_rejected(field: str) -> None:
    overrides = {field: math.inf}
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_scores_mapping(**overrides))
    overrides = {field: math.nan}
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_scores_mapping(**overrides))


@pytest.mark.parametrize("field", ["plddt", "max_pae", "ptm", "iptm"])
def test_boolean_values_are_rejected_as_numbers(field: str) -> None:
    overrides = {field: True}
    with pytest.raises(ValueError):
        prediction_scores_payload_from_mapping(_scores_mapping(**overrides))


def test_optional_ptm_and_iptm() -> None:
    both = prediction_scores_payload_from_mapping(_scores_mapping())
    assert both.ptm == 0.9
    assert both.iptm == 0.8

    neither = prediction_scores_payload_from_mapping(_scores_mapping(ptm=None, iptm=None))
    assert neither.ptm is None
    assert neither.iptm is None
    assert neither.to_mapping()["ptm"] is None
    assert neither.to_mapping()["iptm"] is None

    only_ptm = prediction_scores_payload_from_mapping(_scores_mapping(iptm=None))
    assert only_ptm.ptm == 0.9
    assert only_ptm.iptm is None


def test_unknown_score_keys_are_preserved() -> None:
    payload = _scores_mapping(ranking_confidence=0.95, ranking_confidence_source="plddt")
    record = prediction_scores_payload_from_mapping(payload)
    assert record.extras["ranking_confidence"] == 0.95
    assert record.extras["ranking_confidence_source"] == "plddt"
    assert record.to_mapping()["ranking_confidence"] == 0.95
    assert record.to_mapping()["ranking_confidence_source"] == "plddt"


def test_ragged_pae_rows_are_allowed_when_row_count_matches() -> None:
    payload = _scores_mapping(plddt=[1.0, 2.0, 3.0], pae=[[0.0], [1.0, 2.0], [3.0, 4.0, 5.0]], max_pae=5.0)
    record = prediction_scores_payload_from_mapping(payload)
    assert len(record.pae) == 3
    assert record.pae[0] == (0.0,)
    assert record.pae[2] == (3.0, 4.0, 5.0)


def test_extras_are_defensively_copied() -> None:
    extras: dict[str, object] = {"ranking_confidence": 0.9}
    record = PredictionScoresPayload(
        plddt=(90.0, 80.0),
        pae=((0.0, 1.0), (1.0, 0.0)),
        max_pae=1.0,
        ptm=None,
        iptm=None,
        extras=extras,
    )
    extras["ranking_confidence"] = 0.1
    assert record.extras["ranking_confidence"] == 0.9
    with pytest.raises(TypeError):
        record.extras["ranking_confidence"] = 0.2  # type: ignore[index]


@pytest.mark.parametrize("tool", VALID_TOOL_USED)
def test_every_valid_tool_used_value(tool: str) -> None:
    pair = _make_pair(tool_used=tool)
    assert pair.tool_used == tool


def test_unknown_tool_used_is_rejected() -> None:
    with pytest.raises(ValueError):
        _make_pair(tool_used="Unknown Folding Tool / AlphaFold-Multimer")


def test_canonical_pair_acceptance() -> None:
    pair = _make_pair(
        structure_path=f"/data/{MODEL_ID}-model_v1.pdb",
        scores_path=f"/data/{MODEL_ID}-meta_v1.json",
    )
    assert pair.structure_path.endswith("-model_v1.pdb")
    assert pair.scores_path.endswith("-meta_v1.json")


@pytest.mark.parametrize("index", range(3))
def test_all_suffix_pairings_are_accepted(index: int) -> None:
    pair = _make_pair(
        structure_path=f"/data/{MODEL_ID}{STRUCTURE_SUFFIXES[index]}",
        scores_path=f"/data/{MODEL_ID}{SCORES_SUFFIXES[index]}",
    )
    assert pair.structure_path.endswith(STRUCTURE_SUFFIXES[index])
    assert pair.scores_path.endswith(SCORES_SUFFIXES[index])


def test_crossed_suffix_kinds_are_rejected() -> None:
    with pytest.raises(ValueError):
        _make_pair(
            structure_path=f"/data/{MODEL_ID}-meta_v1.json",
            scores_path=f"/data/{MODEL_ID}-model_v1.pdb",
        )


def test_mixed_naming_profiles_are_rejected() -> None:
    with pytest.raises(ValueError):
        _make_pair(
            structure_path=f"/data/{MODEL_ID}-model_v1.pdb",
            scores_path=f"/data/{MODEL_ID}.merged_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
        )


def test_mismatched_filename_roots_are_rejected() -> None:
    with pytest.raises(ValueError):
        _make_pair(
            structure_path="/data/AF-0000000000000001-model_v1.pdb",
            scores_path="/data/AF-0000000000000002-meta_v1.json",
        )


def test_normalized_root_disagreement_is_rejected() -> None:
    with pytest.raises(ValueError):
        _make_pair(
            model_entity_id=MODEL_ID,
            structure_path="/data/AFDB_AF_0000000000000002-model_v1.pdb",
            scores_path="/data/AFDB_AF_0000000000000002-meta_v1.json",
        )


def test_full_path_strings_are_preserved() -> None:
    structure_path = "/srv/example/run/AF-0000000000000001-model_v1.pdb"
    scores_path = "/srv/example/run/AF-0000000000000001-meta_v1.json"
    pair = _make_pair(structure_path=structure_path, scores_path=scores_path)
    assert pair.structure_path == structure_path
    assert pair.scores_path == scores_path


def test_scores_must_be_a_mapping() -> None:
    with pytest.raises(ValueError):
        prediction_pair_from_mapping(_pair_mapping(scores="not-a-mapping"))


def _without(payload: dict[str, object], key: str) -> dict[str, object]:
    return {name: value for name, value in payload.items() if name != key}
