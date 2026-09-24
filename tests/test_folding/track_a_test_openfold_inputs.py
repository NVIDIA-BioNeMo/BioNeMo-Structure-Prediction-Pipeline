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

"""Track-A tests for the OpenFold input preprocessor.

Adapted from the reference pipeline's ``test_local_backends.py``, with the
ColabFold MSA backend removed.  Each test builds an
``MSAResult`` directly from ``split_merged_a3m`` output on a small merged-A3M
fixture, then runs ``OpenFoldInputPreprocessor``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_input import folding_input_layout_from_mapping
from bspp.orchestration.runtime.folding.execution.a3m_split import split_merged_a3m
from bspp.orchestration.runtime.folding.execution.errors import FoldingBackendError
from bspp.orchestration.runtime.folding.execution.models import ProteinTarget
from bspp.orchestration.runtime.folding.execution.msa_models import (
    ChainAlignment,
    MSAResult,
)
from bspp.orchestration.runtime.folding.execution.openfold_inputs import (
    NO_TEMPLATE_CIF,
    OpenFoldInputPreprocessor,
)

_MERGED_A3M = "#4,3\t1,1\n>query\nACDEGGX\n>hit\nAC-EG-X\n"

# Grammar-valid compound model entity ID (contract.model_identity): the
# preprocessor rejects identifiers outside the model-ID grammar before any
# filesystem operation.
_TARGET_ID = "AF-0000000000000001_AF-0000000000000002"


def _build_msa_result(source: Path, target: ProteinTarget, split_dir: Path) -> MSAResult:
    paths = split_merged_a3m(source, target, split_dir)
    chains: list[ChainAlignment] = []
    for index, (sequence, path) in enumerate(zip(target.chains, paths, strict=True), start=1):
        count = sum(line.startswith(">") for line in path.read_text(encoding="utf-8").splitlines())
        chains.append(
            ChainAlignment(
                chain_index=index,
                query_sequence=sequence,
                alignments={"colabfold": path},
                sequence_counts={"colabfold": count},
            )
        )
    return MSAResult(backend="track-a-test", chains=tuple(chains))


def _write_merged(tmp_path: Path) -> Path:
    source = tmp_path / "merged.a3m"
    source.write_text(_MERGED_A3M, encoding="utf-8")
    return source


def test_openfold_inputs_writes_full_layout(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = ProteinTarget(_TARGET_ID, "sample", ("ACDE", "GGX"))
    msa = _build_msa_result(source, target, tmp_path / "split")
    output_dir = tmp_path / "preprocess"

    prepared = OpenFoldInputPreprocessor().run(target, msa, output_dir)

    assert prepared.backend == "openfold-inputs"

    fasta = (prepared.fasta_dir / f"{_TARGET_ID}.fasta").read_text(encoding="utf-8")
    assert f">{_TARGET_ID}_A\nACDE" in fasta
    assert f">{_TARGET_ID}_B\nGGX" in fasta

    assert (prepared.alignment_dir / f"{_TARGET_ID}_A" / "alignment.a3m").is_file()
    assert (prepared.alignment_dir / f"{_TARGET_ID}_B" / "uniprot_hits.sto").is_file()

    sto = (prepared.alignment_dir / f"{_TARGET_ID}_B" / "uniprot_hits.sto").read_text(encoding="utf-8")
    assert sto.startswith("# STOCKHOLM 1.0")

    layout_payload = json.loads((output_dir / "layout.json").read_text(encoding="utf-8"))
    layout = folding_input_layout_from_mapping(layout_payload)
    assert layout.layout == "openfold"
    assert layout.chain_ids == (f"{_TARGET_ID}_A", f"{_TARGET_ID}_B")
    assert layout.template_mode == "none"
    assert layout.pairing == "species-from-colabfold-headers"
    assert layout.fasta_dir == "fasta"
    assert layout.alignment_dir == "alignments"
    assert layout.template_dir == "templates"
    assert prepared.metadata["chain_ids"] == [f"{_TARGET_ID}_A", f"{_TARGET_ID}_B"]


def test_openfold_no_template_cif_written_unconditionally(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = ProteinTarget(_TARGET_ID, "sample", ("ACDE", "GGX"))
    msa = _build_msa_result(source, target, tmp_path / "split")

    prepared = OpenFoldInputPreprocessor().run(target, msa, tmp_path / "preprocess")

    cif = prepared.template_dir / "openfold_no_template.cif"
    assert cif.is_file()
    assert cif.read_text(encoding="utf-8") == NO_TEMPLATE_CIF
    assert [path.name for path in prepared.template_dir.iterdir()] == ["openfold_no_template.cif"]


def test_chain_index_is_one_based(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = ProteinTarget(_TARGET_ID, "sample", ("ACDE", "GGX"))
    msa = _build_msa_result(source, target, tmp_path / "split")

    assert [chain.chain_index for chain in msa.chains] == [1, 2]


def test_mismatched_chain_count_raises(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = ProteinTarget(_TARGET_ID, "sample", ("ACDE", "GGX"))
    msa = _build_msa_result(source, target, tmp_path / "split")
    wrong_target = ProteinTarget(_TARGET_ID, "sample", ("ACDE",))

    with pytest.raises(FoldingBackendError):
        OpenFoldInputPreprocessor().run(wrong_target, msa, tmp_path / "preprocess")


@pytest.mark.parametrize("bad_id", ["../../shared/job", "a/b", "/abs/target", "..", "plain"])
def test_openfold_inputs_rejects_target_ids_outside_model_grammar_before_writes(tmp_path: Path, bad_id: str) -> None:
    """Traversal/non-grammar target IDs fail closed before any filesystem write."""
    source = _write_merged(tmp_path)
    valid_target = ProteinTarget(_TARGET_ID, "sample", ("ACDE", "GGX"))
    msa = _build_msa_result(source, valid_target, tmp_path / "split")
    bad_target = ProteinTarget(bad_id, "bad", ("ACDE", "GGX"))
    output_dir = tmp_path / "preprocess"

    with pytest.raises(FoldingBackendError, match="invalid model entity identity"):
        OpenFoldInputPreprocessor().run(bad_target, msa, output_dir)

    assert not output_dir.exists()
    assert not (tmp_path / "shared").exists()
