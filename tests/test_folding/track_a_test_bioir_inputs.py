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

"""Track-A tests for the BioIR input preprocessor and source helpers.

Adapted from the reference pipeline's ``test_bioir_backend.py``, with the
``quality_from_bioir_scores`` test dropped (owned by the BioIR session
story).  No BioIR/torch import is required: each test builds an
``MSAResult`` from ``split_merged_a3m`` output on a small merged-A3M fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_input import (
    BioIRRequestManifest,
    FoldingInputLayout,
    bioir_request_manifest_from_mapping,
    folding_input_layout_from_mapping,
)
from bspp.orchestration.runtime.folding.execution.a3m_split import (
    _read_a3m,
    split_merged_a3m,
)
from bspp.orchestration.runtime.folding.execution.bioir_config import (
    OpenFoldModelSettings,
    bioir_checkpoint_env,
    bioir_model_source,
)
from bspp.orchestration.runtime.folding.execution.bioir_inputs import (
    BioIRInputPreprocessor,
)
from bspp.orchestration.runtime.folding.execution.errors import FoldingBackendError
from bspp.orchestration.runtime.folding.execution.models import ProteinTarget
from bspp.orchestration.runtime.folding.execution.msa_models import (
    ChainAlignment,
    MSAResult,
)

_HETERO_A3M = "#4,3\t1,1\n>query\nACDEGGX\n>only_a\nAC-E---\n>only_b\n----G-X\n>paired\nACDEGG-\n"

# Grammar-valid model entity IDs (contract.model_identity): the preprocessor
# rejects identifiers outside the model-ID grammar before any filesystem op.
_HETERO_TARGET_ID = "AF-0000000000000001_AF-0000000000000002"
_HOMO_TARGET_ID = "AF-0000000000000001"


def _msa_result(source: Path, target: ProteinTarget) -> MSAResult:
    paths = split_merged_a3m(source, target, source.parent)
    chains = tuple(
        ChainAlignment(
            chain_index=index,
            query_sequence=sequence,
            alignments={"colabfold": path},
            sequence_counts={"colabfold": len(_read_a3m(path))},
        )
        for index, (sequence, path) in enumerate(zip(target.chains, paths, strict=True), start=1)
    )
    return MSAResult("colabfold", chains, {"merged_source_path": str(source)})


def _load_request(output: Path) -> BioIRRequestManifest:
    return bioir_request_manifest_from_mapping(json.loads((output / "bioir-request.json").read_text(encoding="utf-8")))


def _load_layout(output: Path) -> FoldingInputLayout:
    return folding_input_layout_from_mapping(json.loads((output / "layout.json").read_text(encoding="utf-8")))


def test_bioir_preprocessor_preserves_heteromer_pair_rows(tmp_path: Path) -> None:
    target = ProteinTarget(_HETERO_TARGET_ID, "hetero", ("ACDE", "GGX"))
    source = tmp_path / "source.a3m"
    source.write_text(_HETERO_A3M, encoding="utf-8")

    output = tmp_path / "prepared"
    prepared = BioIRInputPreprocessor(use_paired_msa=True).run(target, _msa_result(source, target), output)
    manifest = _load_request(output)
    layout = _load_layout(output)
    layout.validate_against_request(manifest)

    assert prepared.backend == "bioir-inputs"
    assert [polymer.chain_ids for polymer in manifest.polymers] == [("A",), ("B",)]
    paired_paths = []
    for polymer in manifest.polymers:
        assert polymer.paired_msa is not None
        paired_paths.append(output / polymer.paired_msa)
    paired = [_read_a3m(path) for path in paired_paths]
    assert [len(records) for records in paired] == [2, 2]
    assert [header for header, _ in paired[0]] == [header for header, _ in paired[1]]
    assert [sequence for _, sequence in paired[0]] == ["ACDE", "ACDE"]
    assert [sequence for _, sequence in paired[1]] == ["GGX", "GG-"]
    assert layout.pairing == "colabfold-merged-row-index"


def test_bioir_preprocessor_defaults_to_capped_method_c_unpaired_msa(tmp_path: Path) -> None:
    target = ProteinTarget(_HETERO_TARGET_ID, "hetero", ("ACDE", "GGX"))
    source = tmp_path / "source.a3m"
    source.write_text(_HETERO_A3M, encoding="utf-8")

    output = tmp_path / "prepared"
    prepared = BioIRInputPreprocessor(max_non_query_msa_rows=1).run(target, _msa_result(source, target), output)
    manifest = _load_request(output)
    layout = _load_layout(output)
    layout.validate_against_request(manifest)

    assert [polymer.paired_msa for polymer in manifest.polymers] == [None, None]
    assert [len(_read_a3m(output / polymer.unpaired_msa)) for polymer in manifest.polymers] == [2, 2]
    assert layout.pairing == "method-c-unpaired"
    assert layout.use_paired_msa is False
    assert layout.max_non_query_msa_rows == 1

    layout_payload = json.loads((output / "layout.json").read_text(encoding="utf-8"))
    assert "original_unpaired_rows" not in layout_payload
    assert "retained_unpaired_rows" not in layout_payload
    assert prepared.metadata["original_unpaired_rows"] == 6
    assert prepared.metadata["retained_unpaired_rows"] == 4


def test_bioir_preprocessor_groups_homomer_chains(tmp_path: Path) -> None:
    target = ProteinTarget(_HOMO_TARGET_ID, "homo", ("ACDE", "ACDE"))
    source = tmp_path / "source.a3m"
    source.write_text("#4\t2\n>query\nACDE\n>hit\nAC-E\n", encoding="utf-8")

    output = tmp_path / "prepared"
    BioIRInputPreprocessor().run(target, _msa_result(source, target), output)
    manifest = _load_request(output)
    layout = _load_layout(output)

    assert len(manifest.polymers) == 1
    assert manifest.polymers[0].chain_ids == ("A", "B")
    assert manifest.polymers[0].paired_msa is None
    assert layout.pairing == "bioir-homomer-dummy"


def test_bioir_multimer_model_mapping() -> None:
    model = OpenFoldModelSettings(
        model_id="model_3",
        model_preset="model_3_multimer_v3",
        parameter_file="params_model_3_multimer_v3.pt",
        seed=42,
    )

    assert bioir_model_source(model) == "alphafold2_multimer_3"
    assert bioir_checkpoint_env("alphafold2_multimer_3") == "ALPHAFOLD2_MULTIMER_3_CKPT"

    unsupported = OpenFoldModelSettings("x", "unknown", "x.pt")
    with pytest.raises(FoldingBackendError, match="model_source is required"):
        bioir_model_source(unsupported)

    with pytest.raises(FoldingBackendError, match="unsupported"):
        bioir_checkpoint_env("alphafold2_multimer_9")


def test_bioir_request_round_trips_through_contract(tmp_path: Path) -> None:
    target = ProteinTarget(_HETERO_TARGET_ID, "hetero", ("ACDE", "GGX"))
    source = tmp_path / "source.a3m"
    source.write_text(_HETERO_A3M, encoding="utf-8")

    output = tmp_path / "prepared"
    BioIRInputPreprocessor(use_paired_msa=True).run(target, _msa_result(source, target), output)

    manifest = _load_request(output)
    assert manifest.input_id == _HETERO_TARGET_ID
    assert manifest.chain_ids == ("A", "B")

    layout = _load_layout(output)
    layout.validate_against_request(manifest)
    assert layout.layout == "bioir"
    assert layout.chain_ids == ("A", "B")


@pytest.mark.parametrize(
    ("target", "msa", "match"),
    [
        (ProteinTarget(_HOMO_TARGET_ID, "empty", ()), MSAResult("colabfold", ()), "non-empty target"),
        (
            ProteinTarget(_HOMO_TARGET_ID, "one", ("ACDE",)),
            MSAResult("colabfold", ()),
            "one alignment bundle per chain",
        ),
    ],
)
def test_bioir_preprocessor_rejects_empty_input_without_mutation(
    tmp_path: Path, target: ProteinTarget, msa: MSAResult, match: str
) -> None:
    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FoldingBackendError, match=match):
        BioIRInputPreprocessor().run(target, msa, output)

    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_bioir_preprocessor_rejects_more_than_100_polymers_before_mutation(tmp_path: Path) -> None:
    sequences = tuple(f"ACDE{i}" for i in range(101))
    target = ProteinTarget(_HETERO_TARGET_ID, "many", sequences)
    chains = tuple(
        ChainAlignment(chain_index=index, query_sequence=sequence, alignments={}, sequence_counts={})
        for index, sequence in enumerate(sequences, start=1)
    )
    msa = MSAResult("colabfold", chains)

    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FoldingBackendError, match="100 distinct polymers"):
        BioIRInputPreprocessor().run(target, msa, output)

    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("bad_id", ["../../shared/job", "a/b", "/abs/target", "..", "plain"])
def test_bioir_preprocessor_rejects_target_ids_outside_model_grammar_before_writes(tmp_path: Path, bad_id: str) -> None:
    """Traversal/non-grammar target IDs fail closed before any filesystem op.

    The marker survives because rejection precedes even the destructive
    re-creation of ``output_dir``.
    """
    target = ProteinTarget(bad_id, "bad", ("ACDE",))
    chain = ChainAlignment(chain_index=1, query_sequence="ACDE", alignments={}, sequence_counts={})
    msa = MSAResult("colabfold", (chain,))

    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FoldingBackendError, match="invalid model entity identity"):
        BioIRInputPreprocessor().run(target, msa, output)

    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (output / "fasta").exists()
    assert not (tmp_path / "shared").exists()
