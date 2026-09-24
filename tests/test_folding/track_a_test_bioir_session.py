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

"""Track-A tests for BioIR full-score capture, registry quality, and checkpoint wiring.

Adapted from the reference pipeline's ``test_bioir_backend.py``, plus the new
score-capture and checkpoint-overlay cases.
No BioIR/torch install is required: the capture/quality helpers are pure, the
overlay uses a fake ``model_source``, and the session test only exercises the
missing-checkpoint fail-closed path before any BioIR import.
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest

from bspp.orchestration.runtime.folding.execution.bioir_config import OpenFoldModelSettings, OpenFoldSettings
from bspp.orchestration.runtime.folding.execution.bioir_scores import (
    capture_bioir_full_scores,
    quality_from_bioir_scores,
)
from bspp.orchestration.runtime.folding.execution.bioir_session import (
    BioIRFoldSession,
    bioir_checkpoint_env_overlay,
)
from bspp.orchestration.runtime.folding.execution.errors import FoldingBackendError
from bspp.orchestration.runtime.folding.execution.models import FoldingResult, PreparedInput, ProteinTarget


def test_capture_bioir_full_scores_returns_full_arrays() -> None:
    plddt, pae, max_pae, ptm, iptm = capture_bioir_full_scores(
        {
            "plddt": [50.0, 80.0, 90.0],
            "pae": [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
            "ptm": 0.5,
            "iptm": 0.75,
        }
    )

    assert plddt == (50.0, 80.0, 90.0)
    assert pae == ((0.0, 1.0, 2.0), (1.0, 0.0, 3.0), (2.0, 3.0, 0.0))
    assert max_pae == 3.0
    assert ptm == 0.5
    assert iptm == 0.75


def test_capture_bioir_full_scores_raises_on_missing_pae() -> None:
    with pytest.raises(FoldingBackendError, match="missing 'pae'"):
        capture_bioir_full_scores({"plddt": [1.0, 2.0]})


@pytest.mark.parametrize(
    ("scores", "match"),
    [
        ({"pae": [[0.0]]}, "missing 'plddt'"),
        ({"plddt": [], "pae": [[0.0]]}, "'plddt' is empty"),
        ({"plddt": [1.0], "pae": []}, "'pae' is empty"),
        ({"plddt": [1.0, float("nan")], "pae": [[0.0, 1.0], [1.0, 0.0]]}, "'plddt' must contain only finite"),
        ({"plddt": [1.0, 2.0], "pae": [[0.0, float("inf")], [1.0, 0.0]]}, "'pae' must contain only finite"),
        ({"plddt": [1.0, 2.0, 3.0], "pae": [[0.0, 1.0], [1.0, 0.0]]}, "2 rows but 'plddt' has 3"),
    ],
)
def test_capture_bioir_full_scores_raises_on_invalid_arrays(scores: dict[str, object], match: str) -> None:
    with pytest.raises(FoldingBackendError, match=match):
        capture_bioir_full_scores(scores)


@pytest.mark.parametrize(
    ("scores", "match"),
    [
        # Ragged: one row is narrower than the residue count.
        (
            {"plddt": [1.0, 2.0], "pae": [[0.0, 1.0], [1.0]]},
            "row 1 has 1 columns but 'plddt' has 2 residues",
        ),
        # Rectangular but non-square: every row is wider than the residue count.
        (
            {"plddt": [1.0, 2.0], "pae": [[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]]},
            "row 0 has 3 columns but 'plddt' has 2 residues",
        ),
    ],
)
def test_capture_bioir_full_scores_rejects_non_square_pae(scores: dict[str, object], match: str) -> None:
    """A ragged/non-square PAE must fail capture; only square matrices pass."""
    with pytest.raises(FoldingBackendError, match=match):
        capture_bioir_full_scores(scores)


def test_quality_from_bioir_scores_matches_neel_registry_metrics() -> None:
    quality = quality_from_bioir_scores(
        {
            "plddt": [50.0, 80.0, 90.0],
            "ptm": 0.5,
            "iptm": 0.75,
            "max_pae": 31.75,
        }
    )

    assert quality["mean_plddt"] == pytest.approx(220.0 / 3.0)
    assert quality["plddt_above_70"] == pytest.approx(2.0 / 3.0)
    assert quality["ranking_confidence"] == pytest.approx(0.7)
    assert quality["ranking_confidence_source"] == "0.8_iptm_plus_0.2_ptm"
    assert quality["max_pae"] == pytest.approx(31.75)
    assert quality["output_has_nan"] is False
    assert quality["residue_count"] == 3
    assert quality["source"] == "bioir_scores_json"
    assert quality["min_plddt"] == 50.0
    assert quality["max_plddt"] == 90.0
    assert quality["ptm"] == 0.5
    assert quality["iptm"] == 0.75


def test_bioir_checkpoint_env_overlay_sets_and_restores_prior_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_var = "ALPHAFOLD2_MULTIMER_3_CKPT"
    checkpoint = tmp_path / "fake.pt"
    monkeypatch.setenv(env_var, "before")

    with bioir_checkpoint_env_overlay(checkpoint, "alphafold2_multimer_3"):
        assert os.environ[env_var] == str(checkpoint.resolve())

    assert os.environ[env_var] == "before"


def test_bioir_checkpoint_env_overlay_restores_absence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_var = "ALPHAFOLD2_MULTIMER_3_CKPT"
    checkpoint = tmp_path / "fake.pt"
    monkeypatch.delenv(env_var, raising=False)

    with bioir_checkpoint_env_overlay(checkpoint, "alphafold2_multimer_3"):
        assert os.environ[env_var] == str(checkpoint.resolve())

    assert env_var not in os.environ


def test_bioir_session_missing_checkpoint_raises_before_bioir_import(tmp_path: Path) -> None:
    openfold_model = OpenFoldModelSettings(
        model_id="x",
        model_preset="model_3_multimer_v3",
        parameter_file="p.pt",
        seed=42,
    )
    settings = OpenFoldSettings()
    checkpoint = tmp_path / "missing.pt"

    with pytest.raises(FoldingBackendError, match="checkpoint is missing"):
        BioIRFoldSession(openfold_model, settings, checkpoint, tmp_path)

    # The missing-checkpoint fail-closed path must run before any BioIR import.
    assert "bionemo_ir" not in sys.modules


def test_bioir_session_rejects_non_pt_checkpoint_before_bioir_import(tmp_path: Path) -> None:
    openfold_model = OpenFoldModelSettings(
        model_id="x",
        model_preset="model_3_multimer_v3",
        parameter_file="p.pt",
        seed=42,
    )
    settings = OpenFoldSettings()
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_text("data", encoding="utf-8")

    with pytest.raises(FoldingBackendError, match=r"must be a \.pt file"):
        BioIRFoldSession(openfold_model, settings, checkpoint, tmp_path)

    assert "bionemo_ir" not in sys.modules


@pytest.mark.parametrize(
    "scores",
    [
        {"plddt": [1.0], "pae": [[0.0]], "ptm": True},
        {"plddt": [1.0], "pae": [[0.0]], "iptm": True},
        {"plddt": [1.0], "pae": [[0.0]], "ptm": float("nan")},
        {"plddt": [1.0], "pae": [[0.0]], "iptm": float("inf")},
        {"plddt": [1.0], "pae": [[0.0]], "ptm": "abc"},
    ],
)
def test_capture_bioir_full_scores_rejects_malformed_optional_scalars(scores: dict[str, object]) -> None:
    with pytest.raises(FoldingBackendError, match="must be a finite number"):
        capture_bioir_full_scores(scores)


def test_bioir_session_run_is_protocol_shaped() -> None:
    assert callable(BioIRFoldSession.run)
    signature = inspect.signature(BioIRFoldSession.run)
    assert list(signature.parameters) == ["self", "target", "prepared", "output_dir"]


def test_bioir_session_run_delegates_without_closing(tmp_path: Path) -> None:
    session = BioIRFoldSession.__new__(BioIRFoldSession)
    session.output_dir = tmp_path / "out"
    sentinel = object()
    session.processor = sentinel
    calls: list[tuple[ProteinTarget, Path]] = []

    def fake_predict(target: ProteinTarget, prepared_root: Path) -> FoldingResult:
        calls.append((target, prepared_root))
        return FoldingResult("bioir", (), {})

    session.predict = fake_predict  # type: ignore[method-assign]

    target = ProteinTarget("t", "t", ("A",))
    prepared = PreparedInput(
        "bioir-inputs",
        tmp_path / "prepared" / "fasta",
        tmp_path / "prepared" / "alignments",
        tmp_path / "prepared" / "templates",
        {},
    )

    result = session.run(target, prepared, session.output_dir)

    assert result.backend == "bioir"
    assert calls == [(target, tmp_path / "prepared")]
    assert session.processor is sentinel


def test_bioir_session_run_rejects_wrong_output_root(tmp_path: Path) -> None:
    session = BioIRFoldSession.__new__(BioIRFoldSession)
    session.output_dir = tmp_path / "out"
    calls: list[tuple[ProteinTarget, Path]] = []

    def fake_predict(target: ProteinTarget, prepared_root: Path) -> FoldingResult:
        calls.append((target, prepared_root))
        return FoldingResult("bioir", (), {})

    session.predict = fake_predict  # type: ignore[method-assign]

    target = ProteinTarget("t", "t", ("A",))
    prepared = PreparedInput(
        "bioir-inputs",
        tmp_path / "prepared" / "fasta",
        tmp_path / "prepared" / "alignments",
        tmp_path / "prepared" / "templates",
        {},
    )

    with pytest.raises(FoldingBackendError, match="does not match the session output root"):
        session.run(target, prepared, tmp_path / "other")

    assert calls == []
