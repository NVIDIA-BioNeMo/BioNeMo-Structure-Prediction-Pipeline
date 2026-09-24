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

"""Focused tests for the backend-neutral canonical-pair emitter helpers."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from tests.support.folding_emitter_in_memory import InMemoryFoldingBackend

from bspp.orchestration.contract.prediction_pair import prediction_pair_from_mapping
from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from bspp.orchestration.runtime.folding.execution.backend import FoldingBackend
from bspp.orchestration.runtime.folding.execution.emitter_support import (
    atomic_write_bytes,
    atomic_write_text,
    canonical_pair_names,
    select_tool_used,
    serialize_scores_json,
)
from bspp.orchestration.runtime.folding.execution.models import PreparedInput, ProteinTarget

_HOMO = VALID_TOOL_USED[0]
_HETERO = VALID_TOOL_USED[2]


def test_canonical_pair_names_harvested_form() -> None:
    assert canonical_pair_names("AFDB_AF_0000000000000001_model") == (
        "AF-0000000000000001-model_v1.pdb",
        "AF-0000000000000001-meta_v1.json",
    )


@pytest.mark.parametrize(
    "model_entity_id",
    [
        "AF_0000000000000001",
        "AF-0000000000000001",
    ],
)
def test_canonical_pair_names_single_forms(model_entity_id: str) -> None:
    assert canonical_pair_names(model_entity_id) == (
        "AF-0000000000000001-model_v1.pdb",
        "AF-0000000000000001-meta_v1.json",
    )


def test_canonical_pair_names_compound_form() -> None:
    assert canonical_pair_names("AF_0000000000000001_AF_0000000000000002") == (
        "AF-0000000000000001_AF_0000000000000002-model_v1.pdb",
        "AF-0000000000000001_AF_0000000000000002-meta_v1.json",
    )


@pytest.mark.parametrize("model_entity_id", ["not-a-model-id", ""])
def test_canonical_pair_names_rejects_malformed(model_entity_id: str) -> None:
    with pytest.raises(ValueError):
        canonical_pair_names(model_entity_id)


@pytest.mark.parametrize(
    "model_entity_id",
    [
        "AF-0000000000000001",
        "AF_0000000000000001",
        "AFDB_AF_0000000000000001_model",
    ],
)
@pytest.mark.parametrize("leaked_homodimer", [False, True])
def test_select_tool_used_homodimer(model_entity_id: str, leaked_homodimer: bool) -> None:
    # Homodimer forms always take the homodimer string; the flag is unused.
    assert (
        select_tool_used(
            model_entity_id,
            leaked_homodimer=leaked_homodimer,
            homodimer_tool_used=_HOMO,
            heterodimer_tool_used=_HETERO,
        )
        == _HOMO
    )


@pytest.mark.parametrize(
    ("model_entity_id", "leaked_homodimer", "expected"),
    [
        # Caller-resolved classification is authoritative for compounds; the
        # components are never consulted.
        ("AF_0000000000000001_AF_0000000000000001", True, _HOMO),
        ("AF_0000000000000001_AF_0000000000000001", False, _HETERO),
        # Distinct components resolving to one accession via the manifest:
        # the case the ID-equality proxy would have misclassified.
        ("AF_0000000000000001_AF_0000000000000002", True, _HOMO),
        ("AF_0000000000000001_AF_0000000000000002", False, _HETERO),
    ],
)
def test_select_tool_used_compound_follows_caller_classification(
    model_entity_id: str, leaked_homodimer: bool, expected: str
) -> None:
    assert (
        select_tool_used(
            model_entity_id,
            leaked_homodimer=leaked_homodimer,
            homodimer_tool_used=_HOMO,
            heterodimer_tool_used=_HETERO,
        )
        == expected
    )


def test_select_tool_used_rejects_non_bool_leaked_homodimer() -> None:
    with pytest.raises(ValueError):
        select_tool_used(
            "AF-0000000000000001",
            leaked_homodimer="yes",  # type: ignore[arg-type]
            homodimer_tool_used=_HOMO,
            heterodimer_tool_used=_HETERO,
        )


def test_select_tool_used_rejects_invalid_homodimer_tool() -> None:
    with pytest.raises(ValueError):
        select_tool_used(
            "AF-0000000000000001", leaked_homodimer=False, homodimer_tool_used="bogus", heterodimer_tool_used=_HETERO
        )


def test_select_tool_used_rejects_invalid_heterodimer_tool() -> None:
    with pytest.raises(ValueError):
        select_tool_used(
            "AF-0000000000000001", leaked_homodimer=False, homodimer_tool_used=_HOMO, heterodimer_tool_used="bogus"
        )


def test_select_tool_used_rejects_malformed_id() -> None:
    with pytest.raises(ValueError):
        select_tool_used(
            "not-a-model-id", leaked_homodimer=False, homodimer_tool_used=_HOMO, heterodimer_tool_used=_HETERO
        )


def test_select_tool_used_pdb_assembly_returns_heterodimer_when_not_leaked() -> None:
    """PDB assembly identity returns heterodimer_tool_used when leaked_homodimer is False."""
    result = select_tool_used(
        "pdb_5snm_assembly_1", leaked_homodimer=False, homodimer_tool_used=_HOMO, heterodimer_tool_used=_HETERO
    )
    assert result == _HETERO


def test_select_tool_used_pdb_assembly_returns_homodimer_when_leaked() -> None:
    """PDB assembly identity returns homodimer_tool_used when leaked_homodimer is True."""
    result = select_tool_used(
        "pdb_5snm_assembly_1", leaked_homodimer=True, homodimer_tool_used=_HOMO, heterodimer_tool_used=_HETERO
    )
    assert result == _HOMO


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (81.585, 81.58),
        (81.575, 81.58),
        (1.005, 1.0),
        (1.015, 1.02),
        (12.345, 12.34),
    ],
)
def test_serialize_scores_json_half_even_boundaries(value: float, expected: float) -> None:
    payload = json.loads(serialize_scores_json(plddt=[value], pae=[[value]], max_pae=value))
    assert payload["plddt"] == [expected]
    assert payload["pae"] == [[expected]]
    assert payload["max_pae"] == expected


def test_serialize_scores_json_is_deterministic() -> None:
    first = serialize_scores_json(plddt=[1.0, 2.0], pae=[[1.0, 2.0], [2.0, 1.0]], max_pae=2.0)
    second = serialize_scores_json(plddt=[1.0, 2.0], pae=[[1.0, 2.0], [2.0, 1.0]], max_pae=2.0)
    assert first == second


def test_serialize_scores_json_rejects_pae_length_mismatch() -> None:
    with pytest.raises(ValueError):
        serialize_scores_json(plddt=[1.0, 2.0], pae=[[1.0]], max_pae=2.0)


def test_serialize_scores_json_optional_scores_omitted_as_null() -> None:
    payload = json.loads(serialize_scores_json(plddt=[1.0], pae=[[1.0]], max_pae=1.0))
    assert payload["ptm"] is None
    assert payload["iptm"] is None


def test_serialize_scores_json_optional_scores_present() -> None:
    payload = json.loads(serialize_scores_json(plddt=[1.0], pae=[[1.0]], max_pae=1.0, ptm=0.8765, iptm=0.8125))
    assert payload["ptm"] == 0.88
    assert payload["iptm"] == 0.81


def test_serialize_scores_json_preserves_extras_and_schema_version() -> None:
    extras = {"ranking_confidence": 0.99, "chain_order": ["A", "B"]}
    payload = json.loads(serialize_scores_json(plddt=[1.0], pae=[[1.0]], max_pae=1.0, extras=extras))
    assert payload["schema_version"] == 1
    assert payload["ranking_confidence"] == 0.99
    assert payload["chain_order"] == ["A", "B"]


def test_serialize_scores_json_rejects_reserved_field_collision() -> None:
    with pytest.raises(ValueError):
        serialize_scores_json(plddt=[1.0], pae=[[1.0]], max_pae=1.0, extras={"plddt": [9.0]})


def test_serialize_scores_json_defaults_two_decimals_everywhere() -> None:
    payload = json.loads(
        serialize_scores_json(
            plddt=[1.005, 1.015],
            pae=[[1.005, 1.015], [1.015, 1.005]],
            max_pae=1.005,
            ptm=1.015,
            iptm=1.005,
        )
    )
    assert payload["plddt"] == [1.0, 1.02]
    assert payload["pae"] == [[1.0, 1.02], [1.02, 1.0]]
    assert payload["max_pae"] == 1.0
    assert payload["ptm"] == 1.02
    assert payload["iptm"] == 1.0


def test_serialize_scores_json_scalar_decimals_override_only_scalars() -> None:
    payload = json.loads(
        serialize_scores_json(
            plddt=[1.005, 1.015],
            pae=[[1.005, 1.015], [1.015, 1.005]],
            max_pae=0.12345,
            ptm=0.12345,
            iptm=0.12355,
            scalar_decimals=4,
        )
    )
    assert payload["plddt"] == [1.0, 1.02]
    assert payload["pae"][0] == [1.0, 1.02]
    assert payload["max_pae"] == 0.1234
    assert payload["ptm"] == 0.1234
    assert payload["iptm"] == 0.1236


def test_serialize_scores_json_scalar_decimals_preserves_extras() -> None:
    extras = {"ranking_confidence": 0.99, "chain_order": ["A", "B"]}
    payload = json.loads(serialize_scores_json(plddt=[1.0], pae=[[1.0]], max_pae=1.0, scalar_decimals=4, extras=extras))
    assert payload["ranking_confidence"] == 0.99
    assert payload["chain_order"] == ["A", "B"]


def test_atomic_write_bytes_replaces_and_sets_mode(tmp_path: Path) -> None:
    path = tmp_path / "out.bin"
    path.write_bytes(b"old")
    atomic_write_bytes(path, b"new-content")
    assert path.read_bytes() == b"new-content"
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_atomic_write_text_replaces_and_sets_mode(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    atomic_write_text(path, "hello")
    assert path.read_text(encoding="utf-8") == "hello"
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_atomic_write_leaves_no_temp_files_on_success(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    atomic_write_text(path, "content")
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["out.txt"]


def test_atomic_write_leaves_no_temp_files_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "out.txt"
    with pytest.raises(OSError):
        atomic_write_text(path, "content")
    assert not (tmp_path / "missing").exists()
    assert list(tmp_path.iterdir()) == []


def test_in_memory_backend_emits_canonical_pair_matching_golden(tmp_path: Path, fixtures_dir: Path) -> None:
    backend = InMemoryFoldingBackend()
    result = backend.run(
        ProteinTarget("AF-0000000000000001", "desc", ("A",)),
        PreparedInput("in-memory", tmp_path, tmp_path, tmp_path, {}),
        tmp_path,
    )

    assert result.backend == backend.name
    assert len(result.predictions) == 1
    prediction = result.predictions[0]
    assert prediction.structure_path.name.endswith("-model_v1.pdb")
    assert prediction.scores_path.name.endswith("-meta_v1.json")

    golden_dir = fixtures_dir / "folding" / "canonical_pair"
    assert prediction.structure_path.read_bytes() == (golden_dir / "AF-0000000000000001-model_v1.pdb").read_bytes()
    assert prediction.scores_path.read_bytes() == (golden_dir / "AF-0000000000000001-meta_v1.json").read_bytes()


def test_in_memory_backend_returns_declared_path_types(tmp_path: Path) -> None:
    backend = InMemoryFoldingBackend()
    result = backend.run(
        ProteinTarget("AF-0000000000000001", "desc", ("A",)),
        PreparedInput("in-memory", tmp_path, tmp_path, tmp_path, {}),
        tmp_path,
    )

    prediction = result.predictions[0]
    assert isinstance(prediction.structure_path, Path)
    assert isinstance(prediction.scores_path, Path)
    assert prediction.structure_path.parent == tmp_path
    assert prediction.scores_path.parent == tmp_path
    assert prediction.structure_path.name.endswith("-model_v1.pdb")
    assert prediction.scores_path.name.endswith("-meta_v1.json")


def test_in_memory_backend_pair_round_trips_through_contract(tmp_path: Path, fixtures_dir: Path) -> None:
    backend = InMemoryFoldingBackend()
    result = backend.run(
        ProteinTarget("AF-0000000000000001", "desc", ("A",)),
        PreparedInput("in-memory", tmp_path, tmp_path, tmp_path, {}),
        tmp_path,
    )
    prediction = result.predictions[0]
    scores = json.loads(prediction.scores_path.read_bytes())

    pair = prediction_pair_from_mapping(
        {
            "schema_version": 1,
            "model_entity_id": "AF-0000000000000001",
            "tool_used": result.metadata["tool_used"],
            "structure_path": str(prediction.structure_path),
            "scores_path": str(prediction.scores_path),
            "scores": scores,
        }
    )

    assert pair.model_entity_id == "AF-0000000000000001"
    assert pair.tool_used in VALID_TOOL_USED
    assert pair.scores.plddt == (81.58, 90.12, 75.0)
    assert pair.to_mapping()["structure_path"] == str(prediction.structure_path)
    assert pair.to_mapping()["scores_path"] == str(prediction.scores_path)
    assert pair.to_mapping()["scores"] == scores


def test_in_memory_backend_satisfies_protocol() -> None:
    backend: FoldingBackend = InMemoryFoldingBackend()
    assert backend.name == "in-memory"
    assert callable(backend.run)
