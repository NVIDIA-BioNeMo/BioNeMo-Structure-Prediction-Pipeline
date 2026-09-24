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

"""Focused tests for the folding backend protocol and model shapes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from bspp.orchestration.runtime.folding.execution.backend import FoldingBackend
from bspp.orchestration.runtime.folding.execution.models import (
    FoldingResult,
    PreparedInput,
    ProteinTarget,
    StructurePrediction,
)


class _LocalBackend:
    """Test-only backend structurally satisfying FoldingBackend."""

    name: str = "local-test"

    def run(self, target: ProteinTarget, prepared: PreparedInput, output_dir: Path) -> FoldingResult:
        prediction = StructurePrediction(
            rank=1,
            structure_path=output_dir / "model.pdb",
            scores_path=output_dir / "scores.json",
            confidence=0.9,
        )
        return FoldingResult(backend=self.name, predictions=(prediction,))


def test_local_backend_satisfies_protocol() -> None:
    backend: FoldingBackend = _LocalBackend()
    assert backend.name == "local-test"
    assert callable(backend.run)


def test_protein_target_shape_and_fields() -> None:
    target = ProteinTarget("AF-0000000000000001", "desc", ("A", "C"))
    assert target.target_id == "AF-0000000000000001"
    assert target.description == "desc"
    assert target.chains == ("A", "C")


def test_prepared_input_shape_and_metadata_default() -> None:
    prepared = PreparedInput("x", Path("f"), Path("a"), Path("t"))
    assert prepared.backend == "x"
    assert prepared.fasta_dir == Path("f")
    assert prepared.metadata == {}


def test_structure_prediction_default_confidence() -> None:
    pred = StructurePrediction(rank=2, structure_path=Path("s.pdb"), scores_path=Path("s.json"))
    assert pred.confidence is None


def test_folding_result_composition() -> None:
    pred = StructurePrediction(rank=1, structure_path=Path("s.pdb"), scores_path=Path("s.json"))
    result = FoldingResult(backend="x", predictions=(pred,))
    assert result.backend == "x"
    assert result.predictions == (pred,)
    assert result.metadata == {}
    assert FoldingResult(backend="x", predictions=()).backend == "x"


@pytest.mark.parametrize(
    ("obj", "attr", "value"),
    [
        (ProteinTarget("id", "d", ("A",)), "target_id", "other"),
        (PreparedInput("x", Path("f"), Path("a"), Path("t")), "backend", "y"),
        (StructurePrediction(1, Path("s.pdb"), Path("s.json")), "rank", 9),
        (FoldingResult("x", ()), "backend", "y"),
    ],
)
def test_model_shapes_are_frozen(obj: object, attr: str, value: object) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(obj, attr, value)
