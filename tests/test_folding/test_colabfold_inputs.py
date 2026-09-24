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

"""Unit tests for the ColabFold input preparation adapter.

The adapter stages one verified merged A3M at ``alignments/<target_id>.a3m`` and
never routes ColabFold through ``OpenFoldInputPreprocessor``.  No GPU stack,
subprocess, or object storage is involved.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from bspp.orchestration.runtime.folding.execution import openfold_inputs
from bspp.orchestration.runtime.folding.execution.a3m_split import split_merged_a3m
from bspp.orchestration.runtime.folding.execution.colabfold_inputs import prepare
from bspp.orchestration.runtime.folding.execution.errors import FoldingBackendError
from bspp.orchestration.runtime.folding.execution.models import ProteinTarget
from bspp.orchestration.runtime.folding.execution.msa_models import MSAResult
from bspp.orchestration.runtime.folding.execution.msa_preparation import msa_result_from_split_paths

_MERGED_A3M = "#4,3\t1,1\n>query\nACDEGGX\n>hit\nAC-EG-X\n"
_TARGET_ID = "AF-0000000000000001_AF-0000000000000002"
_MEMBER_NAME = "AFDB_AF-0000000000000001_AF-0000000000000002.a3m"
_LOGICAL_PATH = f"a3ms/{_MEMBER_NAME}"
_ARTIFACT_SET_ID = "sha256:" + "a" * 64


def _target() -> ProteinTarget:
    return ProteinTarget(_TARGET_ID, "compound", ("ACDE", "GGX"))


def _write_merged(tmp_path: Path) -> Path:
    source = tmp_path / "merged.a3m"
    source.write_text(_MERGED_A3M, encoding="utf-8")
    return source


def _build_msa(source: Path, target: ProteinTarget, split_dir: Path) -> MSAResult:
    split_paths = split_merged_a3m(source, target, split_dir)
    return msa_result_from_split_paths(
        split_paths,
        target,
        artifact_set_id=_ARTIFACT_SET_ID,
        selected_logical_path=_LOGICAL_PATH,
        merged_source_path=source,
    )


def _colabfold_inputs_source() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "packages"
        / "orchestration-runtime"
        / "src"
        / "bspp"
        / "orchestration"
        / "runtime"
        / "folding"
        / "execution"
        / "colabfold_inputs.py"
    )


def test_prepare_stages_merged_a3m_byte_identical(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    msa = _build_msa(source, target, tmp_path / "split")

    prepared = prepare(target, msa, tmp_path / "prepared")

    staged = prepared.alignment_dir / f"{_TARGET_ID}.a3m"
    assert staged.is_file()
    assert staged.read_bytes() == source.read_bytes()
    assert [path.name for path in prepared.alignment_dir.iterdir()] == [f"{_TARGET_ID}.a3m"]


def test_prepare_fails_closed_on_non_fresh_output_dir(tmp_path: Path) -> None:
    """Regression (council e09s04 dissent): a reused output dir must not let stale
    FASTA/template/alignment files survive into the PreparedInput, and preparation
    must not silently wipe a partially completed rerun — it fails closed."""
    source = _write_merged(tmp_path)
    target = _target()
    msa = _build_msa(source, target, tmp_path / "split")
    output_dir = tmp_path / "prepared"
    stale = output_dir / "fasta" / "stale.fa"
    stale.parent.mkdir(parents=True)
    stale.write_text(">stale\nAAA\n")

    with pytest.raises(FoldingBackendError, match="fresh output directory"):
        prepare(target, msa, output_dir)

    # An existing but completely empty directory remains acceptable (fresh).
    empty_dir = tmp_path / "fresh"
    (empty_dir / "fasta").mkdir(parents=True)
    prepared = prepare(target, msa, empty_dir)
    assert (prepared.alignment_dir / f"{_TARGET_ID}.a3m").is_file()


def test_prepare_creates_empty_fasta_and_templates_dirs(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    msa = _build_msa(source, target, tmp_path / "split")

    prepared = prepare(target, msa, tmp_path / "prepared")

    assert prepared.fasta_dir.is_dir()
    assert prepared.template_dir.is_dir()
    assert list(prepared.fasta_dir.iterdir()) == []
    assert list(prepared.template_dir.iterdir()) == []


def test_prepare_metadata(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    msa = _build_msa(source, target, tmp_path / "split")

    prepared = prepare(target, msa, tmp_path / "prepared")

    assert prepared.backend == "colabfold-inputs"
    assert prepared.metadata["layout"] == "colabfold"
    assert prepared.metadata["target_id"] == _TARGET_ID
    assert prepared.metadata["chain_count"] == 2
    assert prepared.metadata["merged_source_path"] == str(source)


def test_colabfold_inputs_never_imports_openfold_inputs() -> None:
    source = _colabfold_inputs_source()
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        else:
            continue
        assert not any("openfold_inputs" in name for name in imported)


def test_prepare_never_invokes_openfold_preprocessor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    msa = _build_msa(source, target, tmp_path / "split")

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("OpenFoldInputPreprocessor.run must not be called")

    monkeypatch.setattr(openfold_inputs.OpenFoldInputPreprocessor, "run", boom)

    prepared = prepare(target, msa, tmp_path / "prepared")
    assert prepared.backend == "colabfold-inputs"


@pytest.mark.parametrize("bad_id", ["../../shared/job", "a/b", "/abs/target", "..", "plain"])
def test_prepare_rejects_target_ids_outside_model_grammar_before_writes(tmp_path: Path, bad_id: str) -> None:
    source = _write_merged(tmp_path)
    valid_target = _target()
    msa = _build_msa(source, valid_target, tmp_path / "split")
    bad_target = ProteinTarget(bad_id, "bad", ("ACDE", "GGX"))
    output_dir = tmp_path / "prepared"

    with pytest.raises(FoldingBackendError, match="invalid model entity identity"):
        prepare(bad_target, msa, output_dir)

    assert not output_dir.exists()
    assert not (tmp_path / "shared").exists()


def test_prepare_rejects_missing_merged_source_metadata(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    msa = _build_msa(source, target, tmp_path / "split")
    msa_bad = MSAResult(backend=msa.backend, chains=msa.chains, metadata={})

    with pytest.raises(FoldingBackendError, match="missing a non-empty merged_source_path"):
        prepare(target, msa_bad, tmp_path / "prepared")


def test_prepare_rejects_non_file_merged_source(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    msa = _build_msa(source, target, tmp_path / "split")
    msa_bad = MSAResult(
        backend=msa.backend,
        chains=msa.chains,
        metadata={"merged_source_path": str(tmp_path / "nope.a3m")},
    )

    with pytest.raises(FoldingBackendError, match="merged MSA source is not a regular file"):
        prepare(target, msa_bad, tmp_path / "prepared")


def test_prepare_rejects_chain_count_mismatch(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    msa = _build_msa(source, target, tmp_path / "split")
    wrong_target = ProteinTarget(_TARGET_ID, "compound", ("ACDE",))

    with pytest.raises(FoldingBackendError, match="one alignment bundle per chain"):
        prepare(wrong_target, msa, tmp_path / "prepared")
