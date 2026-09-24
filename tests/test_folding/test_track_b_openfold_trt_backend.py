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

"""Focused tests for the OpenFold-TRT backend emitter."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import numpy as np
import pytest

from bspp.orchestration.contract.prediction_pair import prediction_pair_from_mapping
from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from bspp.orchestration.runtime.folding.execution.chain_manifest import (
    ChainManifest,
    ChainManifestRow,
    ambiguous_chain_manifest_metadata,
)
from bspp.orchestration.runtime.folding.execution.models import PreparedInput, ProteinTarget
from bspp.orchestration.runtime.folding.execution.openfold_trt_backend import (
    OPENFOLD_TRT_TOOL_USED,
    OpenFoldTrtBackend,
    build_scores_payload,
    normalize_pdb,
)

_COMPOUND_TARGET = "AF_0000000000000001_AF_0000000000000002"
_SINGLE_TARGET = "AF-0000000000000001"


def _atom_line(chain: str, resid: int, serial: int = 1, atom: str = "CA", resname: str = "GLY") -> str:
    """Build an ATOM line with chain at index 21 and resid at indices 22:26."""
    return f"ATOM  {serial:>5d} {atom:>4s} {resname:>3s} {chain}{resid:>4d}    "


def _pdb_fixture() -> str:
    """A two-chain PDB exercising duplicate REMARK, PARENT, and standalone TERs."""
    return (
        "\n".join(
            [
                "REMARK   1 FIRST",
                "REMARK   1 SECOND",
                "PARENT N/A",
                _atom_line("A", 42, serial=1),
                _atom_line("A", 43, serial=2),
                "TER",
                _atom_line("B", 7, serial=3),
                _atom_line("B", 8, serial=4),
                "TER",
                "ENDMDL",
                "END",
            ]
        )
        + "\n"
    )


def _model_fn_output(pdb_string: str) -> dict[str, object]:
    return {
        "plddt": np.array([81.585, 90.123]),
        "predicted_aligned_error": np.array([[0.0, 1.234], [1.234, 0.0]]),
        "ptm": np.float64(0.8765),
        "iptm": np.float64(0.8125),
        "pdb_string": pdb_string,
    }


class _RecordingModelFn:
    def __init__(self, output: dict[str, object]) -> None:
        self._output = output
        self.calls: list[object] = []

    def __call__(self, batch: object) -> dict[str, object]:
        self.calls.append(batch)
        return self._output


def _prepared(tmp_path: Path) -> PreparedInput:
    return PreparedInput(
        "openfold-trt",
        fasta_dir=tmp_path / "fasta",
        alignment_dir=tmp_path / "alignments",
        template_dir=tmp_path / "templates",
    )


def _single_manifest() -> ChainManifest:
    return ChainManifest((ChainManifestRow(_SINGLE_TARGET, "e1", "A", "P1"),))


def _compound_manifest() -> ChainManifest:
    return ChainManifest(
        (
            ChainManifestRow(_COMPOUND_TARGET, "e1", "A", "P1"),
            ChainManifestRow(_COMPOUND_TARGET, "e2", "B", "P2"),
        )
    )


def test_normalize_pdb_removes_duplicate_remark_parent_and_extra_ter() -> None:
    out = normalize_pdb(_pdb_fixture())
    lines = out.splitlines()

    assert out.count("REMARK") == 1
    assert "PARENT" not in out
    # The two standalone TER records are removed; only the boundary TER and the
    # pre-ENDMDL TER are re-inserted.
    assert sum(line.startswith("TER") for line in lines) == 2
    assert "ENDMDL" in out
    assert "END" in out


def test_normalize_pdb_renumbers_residues_and_atoms() -> None:
    out = normalize_pdb(_pdb_fixture())

    # Chain A renumbered from resid 42/43 to 1/2; chain B from 7/8 to 1/2.
    assert "ATOM      1   CA GLY A   1" in out
    assert "ATOM      2   CA GLY A   2" in out
    assert "ATOM      4   CA GLY B   1" in out
    assert "ATOM      5   CA GLY B   2" in out
    # Boundary TER consumes serial 3; pre-ENDMDL TER consumes serial 6.
    assert "TER       3      GLY A   2" in out
    assert "TER       6      GLY B   2" in out


def test_build_scores_payload_rounds_half_even_at_per_field_precision() -> None:
    payload = json.loads(
        build_scores_payload(
            plddt=[81.585, 1.005, 1.015],
            pae=[[1.005, 1.015], [1.015, 1.005], [1.005, 1.015]],
            max_pae=0.12345,
            ptm=0.12345,
            iptm=0.12355,
        )
    )

    assert payload["plddt"] == [81.58, 1.0, 1.02]
    assert payload["pae"][0] == [1.0, 1.02]
    assert payload["max_pae"] == 0.1234
    assert payload["ptm"] == 0.1234
    assert payload["iptm"] == 0.1236


def test_build_scores_payload_required_fields_and_optional_nulls() -> None:
    payload = json.loads(build_scores_payload(plddt=[1.0, 2.0], pae=[[1.0, 2.0], [2.0, 1.0]], max_pae=2.0))

    assert payload["schema_version"] == 1
    assert payload["plddt"] == [1.0, 2.0]
    assert payload["pae"] == [[1.0, 2.0], [2.0, 1.0]]
    assert payload["max_pae"] == 2.0
    assert payload["ptm"] is None
    assert payload["iptm"] is None


def test_canonical_pair_atomic_writes(tmp_path: Path) -> None:
    model_fn = _RecordingModelFn(_model_fn_output(_pdb_fixture()))
    backend = OpenFoldTrtBackend(_compound_manifest(), model_fn=model_fn)
    target = ProteinTarget(_COMPOUND_TARGET, "desc", ("A", "B"))
    output_dir = tmp_path / "out"

    result = backend.run(target, _prepared(tmp_path), output_dir)

    structure_path = output_dir / "AF-0000000000000001_AF_0000000000000002-model_v1.pdb"
    scores_path = output_dir / "AF-0000000000000001_AF_0000000000000002-meta_v1.json"
    assert structure_path.exists()
    assert scores_path.exists()
    assert structure_path.read_text(encoding="utf-8") == normalize_pdb(_pdb_fixture())

    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    assert payload["plddt"] == [81.58, 90.12]
    assert payload["pae"] == [[0.0, 1.23], [1.23, 0.0]]
    assert payload["max_pae"] == 1.234
    assert payload["ptm"] == 0.8765
    assert payload["iptm"] == 0.8125

    # Only the two canonical files remain; no mkstemp temp files are left behind.
    assert sorted(p.name for p in output_dir.iterdir()) == sorted([structure_path.name, scores_path.name])
    assert stat.S_IMODE(structure_path.stat().st_mode) == 0o644
    assert stat.S_IMODE(scores_path.stat().st_mode) == 0o644

    assert len(result.predictions) == 1
    assert result.predictions[0].rank == 1
    assert result.metadata["tool_used"] == OPENFOLD_TRT_TOOL_USED


def test_ambiguous_manifest_fails_closed_without_invoking_model_fn(tmp_path: Path) -> None:
    manifest = ChainManifest((ChainManifestRow("AF_0000000000000009_AF_0000000000000008", "e1", "A", "P1"),))
    model_fn = _RecordingModelFn(_model_fn_output(_pdb_fixture()))
    backend = OpenFoldTrtBackend(manifest, model_fn=model_fn)
    target = ProteinTarget(_COMPOUND_TARGET, "desc", ("A", "B"))

    result = backend.run(target, _prepared(tmp_path), tmp_path / "out")

    assert result.backend == "openfold-trt"
    assert result.predictions == ()
    assert result.metadata == ambiguous_chain_manifest_metadata()
    assert model_fn.calls == []


def test_emitted_pair_round_trips_through_contract(tmp_path: Path) -> None:
    model_fn = _RecordingModelFn(_model_fn_output(_pdb_fixture()))
    backend = OpenFoldTrtBackend(_compound_manifest(), model_fn=model_fn)
    target = ProteinTarget(_COMPOUND_TARGET, "desc", ("A", "B"))

    result = backend.run(target, _prepared(tmp_path), tmp_path / "out")

    prediction = result.predictions[0]
    scores = json.loads(prediction.scores_path.read_bytes())
    pair = prediction_pair_from_mapping(
        {
            "schema_version": 1,
            "model_entity_id": target.target_id,
            "tool_used": result.metadata["tool_used"],
            "structure_path": str(prediction.structure_path),
            "scores_path": str(prediction.scores_path),
            "scores": scores,
        }
    )

    assert pair.model_entity_id == "AF-0000000000000001_AF_0000000000000002"
    assert pair.tool_used in VALID_TOOL_USED
    assert pair.scores.plddt == (81.58, 90.12)
    assert pair.scores.pae == ((0.0, 1.23), (1.23, 0.0))
    assert pair.scores.max_pae == 1.234
    assert pair.scores.ptm == 0.8765
    assert pair.scores.iptm == 0.8125


def test_optional_score_fallback_keys(tmp_path: Path) -> None:
    output = {
        "plddt": np.array([1.0]),
        "predicted_aligned_error": np.array([[1.0]]),
        "ptm_score": np.float64(0.5),
        "iptm_score": np.float64(0.5),
        "pdb_string": _pdb_fixture(),
    }
    backend = OpenFoldTrtBackend(_single_manifest(), model_fn=_RecordingModelFn(output))

    result = backend.run(ProteinTarget(_SINGLE_TARGET, "desc", ("A",)), _prepared(tmp_path), tmp_path / "out")

    payload = json.loads(result.predictions[0].scores_path.read_text(encoding="utf-8"))
    assert payload["ptm"] == 0.5
    assert payload["iptm"] == 0.5


def test_optional_score_nan_becomes_null(tmp_path: Path) -> None:
    output = {
        "plddt": np.array([1.0]),
        "predicted_aligned_error": np.array([[1.0]]),
        "ptm": np.nan,
        "iptm": np.nan,
        "pdb_string": _pdb_fixture(),
    }
    backend = OpenFoldTrtBackend(_single_manifest(), model_fn=_RecordingModelFn(output))

    result = backend.run(ProteinTarget(_SINGLE_TARGET, "desc", ("A",)), _prepared(tmp_path), tmp_path / "out")

    payload = json.loads(result.predictions[0].scores_path.read_text(encoding="utf-8"))
    assert payload["ptm"] is None
    assert payload["iptm"] is None


def test_missing_pdb_string_raises(tmp_path: Path) -> None:
    output = {"plddt": np.array([1.0]), "predicted_aligned_error": np.array([[1.0]])}
    backend = OpenFoldTrtBackend(_single_manifest(), model_fn=_RecordingModelFn(output))

    with pytest.raises(RuntimeError, match="pdb_string"):
        backend.run(ProteinTarget(_SINGLE_TARGET, "desc", ("A",)), _prepared(tmp_path), tmp_path / "out")


def test_run_requires_manifest_and_model_fn(tmp_path: Path) -> None:
    target = ProteinTarget(_SINGLE_TARGET, "desc", ("A",))
    with pytest.raises(ValueError, match="ChainManifest and a model_fn"):
        OpenFoldTrtBackend().run(target, _prepared(tmp_path), tmp_path / "out")


def test_openfold_trt_tool_used_constant_is_valid() -> None:
    assert VALID_TOOL_USED[1] == OPENFOLD_TRT_TOOL_USED
    assert OPENFOLD_TRT_TOOL_USED in VALID_TOOL_USED
