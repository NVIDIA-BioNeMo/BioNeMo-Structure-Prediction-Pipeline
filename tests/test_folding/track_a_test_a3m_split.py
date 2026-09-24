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

"""Track-A tests for the N-ary A3M split and Stockholm derivation.

Adapted from the reference pipeline's ``test_local_backends.py``.  These tests
exercise the split semantics directly with small merged-A3M fixtures;
no MSA backend is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bspp.orchestration.runtime.folding.execution.a3m_split import (
    _a3m_query,
    _a3m_query_chains,
    a3m_target_length,
    a3m_to_stockholm,
    split_merged_a3m,
)
from bspp.orchestration.runtime.folding.execution.errors import FoldingBackendError
from bspp.orchestration.runtime.folding.execution.models import ProteinTarget


def test_homomer_cardinality_expands_to_target_chains(tmp_path: Path) -> None:
    source = tmp_path / "homomer.a3m"
    source.write_text(
        "#4\t2\n>query\nACDE\n>hit\nAC-E\n",
        encoding="utf-8",
    )
    target = ProteinTarget("homomer", "homomer", ("ACDE", "ACDE"))

    paths = split_merged_a3m(source, target, tmp_path / "output")

    assert _a3m_query(source) == "ACDEACDE"
    assert len(paths) == 2
    assert paths[0].read_text() == paths[1].read_text()
    assert paths[0].read_text().startswith(">query\nACDE\n")


def test_cardinalities_restore_original_chain_order(tmp_path: Path) -> None:
    source = tmp_path / "mixed.a3m"
    source.write_text(
        "#4,3\t2,1\n>query\nACDEGGX\n>hit\nAC-EG-X\n",
        encoding="utf-8",
    )
    target = ProteinTarget("mixed", "mixed", ("ACDE", "GGX", "ACDE"))

    paths = split_merged_a3m(source, target, tmp_path / "output")

    assert len(paths) == 3
    assert paths[0].read_text() == paths[2].read_text()
    assert paths[1].read_text().startswith(">query\nGGX\n")


def test_chain_outputs_are_one_based(tmp_path: Path) -> None:
    source = tmp_path / "mixed.a3m"
    source.write_text(
        "#4,3\t2,1\n>query\nACDEGGX\n>hit\nAC-EG-X\n",
        encoding="utf-8",
    )
    target = ProteinTarget("mixed", "mixed", ("ACDE", "GGX", "ACDE"))
    output_dir = tmp_path / "output"

    paths = split_merged_a3m(source, target, output_dir)

    assert [path.name for path in paths] == ["chain_1.a3m", "chain_2.a3m", "chain_3.a3m"]
    assert paths[0] == output_dir / "chain_1.a3m"
    assert paths[1] == output_dir / "chain_2.a3m"
    assert paths[2] == output_dir / "chain_3.a3m"


def test_a3m_to_stockholm_derives_uniprot_hits(tmp_path: Path) -> None:
    source = tmp_path / "stockholm.a3m"
    source.write_text(
        ">query\nACDEacGG\n>hit\nAC-E..X\n",
        encoding="utf-8",
    )

    stockholm = a3m_to_stockholm(source)

    assert stockholm.startswith("# STOCKHOLM 1.0\n")
    assert stockholm.endswith("//\n")
    assert "query" in stockholm
    assert "hit" in stockholm
    # Lowercase insertions are dropped from the aligned row.
    assert "ac" not in stockholm
    assert "ACDEGG" in stockholm
    # '.' becomes '-' in the aligned row.
    assert "AC-E--X" in stockholm


@pytest.mark.parametrize(
    ("header", "query", "expected"),
    [
        ("#2\t1", "AA", 2),
        ("#2\t4", "AA", 8),
        ("#2,3\t1,1", "AAGGG", 5),
        ("#2,3\t2,1", "AAGGG", 7),
        ("#2,3,4\t1,1,1", "AAGGGTTTT", 9),
        ("#2,3,4\t2,1,1", "AAGGGTTTT", 11),
        ("#2,3,4,1\t1,1,1,1", "AAGGGTTTTC", 10),
        ("#2,3\t2,1", "AaAG-G", 6),
    ],
)
def test_a3m_target_length_matches_expanded_execution_chains(
    tmp_path: Path, header: str, query: str, expected: int
) -> None:
    source = tmp_path / "target.a3m"
    source.write_text(f"{header}\n>query\n{query}\n", encoding="utf-8")

    assert a3m_target_length(source) == expected
    assert a3m_target_length(source) == sum(map(len, _a3m_query_chains(source)))


@pytest.mark.parametrize(
    "payload",
    [
        ">query\nAA\n",
        "#2,3,4\t1,1\n>query\nAAGGGTTTT\n",
        "#2,0\t1,1\n>query\nAA\n",
        "#2,3\t1,-1\n>query\nAAGGG\n",
        "#2,3\t1,0\n>query\nAAGGG\n",
        "#2,3\t1,1\n>query\nAA\n",
        "#2\t1\n",
        "#2\t1\n>query\n\n",
        "#4\t1\n>query\nAA A\n",
        "#2,3\t1,1\n>query\n--GGG\n",
    ],
)
def test_a3m_target_length_fails_closed_for_invalid_query_metadata(tmp_path: Path, payload: str) -> None:
    source = tmp_path / "invalid.a3m"
    source.write_text(payload, encoding="utf-8")

    with pytest.raises(FoldingBackendError):
        a3m_target_length(source)


def test_a3m_target_length_does_not_expand_artifact_cardinalities(tmp_path: Path) -> None:
    source = tmp_path / "large-cardinality.a3m"
    source.write_text("#2,3\t1000000000,1\n>query\nAAGGG\n", encoding="utf-8")

    assert a3m_target_length(source) == 2000000003
