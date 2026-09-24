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

"""Unit tests for the Track-A split boundary in ``msa_preparation``.

These tests cover the public ``msa_result_from_split_paths`` constructor
(the split-action boundary)
and the behavior-compatible ``prepare_projected_msa`` wrapper.  The
fixtures are small in-memory merged-A3M files; no lz4, subprocess, or object
storage is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.runtime.folding.execution import msa_preparation as mp
from bspp.orchestration.runtime.folding.execution.a3m_split import split_merged_a3m
from bspp.orchestration.runtime.folding.execution.errors import FoldingBackendError
from bspp.orchestration.runtime.folding.execution.models import ProteinTarget
from bspp.orchestration.runtime.folding.execution.msa_preparation import (
    msa_result_from_split_paths,
    prepare_projected_msa,
)

_MERGED_A3M = "#4,3\t1,1\n>query\nACDEGGX\n>hit\nAC-EG-X\n"
_TARGET_ID = "AF-0000000000000001_AF-0000000000000002"
_MEMBER_NAME = "AFDB_AF-0000000000000001_AF-0000000000000002.a3m"
_LOGICAL_PATH = f"a3ms/{_MEMBER_NAME}"
_ARTIFACT_SET_ID = "sha256:" + "a" * 64


def _target() -> ProteinTarget:
    return ProteinTarget(_TARGET_ID, "compound", ("ACDE", "GGX"))


def _consumption() -> MsaSetConsumption:
    return MsaSetConsumption(
        artifact_set_id=_ARTIFACT_SET_ID,
        expected_chunk_count=1,
        member_a3m_paths=(_LOGICAL_PATH,),
        requires_paired_query_header=True,
    )


def _write_merged(tmp_path: Path) -> Path:
    source = tmp_path / "merged.a3m"
    source.write_text(_MERGED_A3M, encoding="utf-8")
    return source


def test_constructor_happy_path(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    output_dir = tmp_path / "split"
    split_paths = split_merged_a3m(source, target, output_dir)

    msa = msa_result_from_split_paths(
        split_paths,
        target,
        artifact_set_id=_ARTIFACT_SET_ID,
        selected_logical_path=_LOGICAL_PATH,
        merged_source_path=source,
    )

    assert msa.backend == "track-a-projected-msa"
    assert [chain.chain_index for chain in msa.chains] == [1, 2]
    assert [chain.query_sequence for chain in msa.chains] == ["ACDE", "GGX"]
    assert [chain.alignments["colabfold"] for chain in msa.chains] == split_paths
    assert [chain.sequence_counts for chain in msa.chains] == [{"colabfold": 2}, {"colabfold": 2}]
    assert set(msa.metadata) == {"artifact_set_id", "selected_logical_path", "merged_source_path"}
    assert msa.metadata["artifact_set_id"] == _ARTIFACT_SET_ID
    assert msa.metadata["selected_logical_path"] == _LOGICAL_PATH
    assert msa.metadata["merged_source_path"] == str(source)


def test_constructor_rejects_missing_inventory(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    split_paths = split_merged_a3m(source, target, tmp_path / "split")
    missing = tmp_path / "missing.a3m"

    with pytest.raises(FoldingBackendError, match="split chain A3M is not a regular file"):
        msa_result_from_split_paths(
            [split_paths[0], missing],
            target,
            artifact_set_id=_ARTIFACT_SET_ID,
            selected_logical_path=_LOGICAL_PATH,
            merged_source_path=source,
        )


def test_constructor_rejects_directory_inventory(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    split_paths = split_merged_a3m(source, target, tmp_path / "split")
    directory = tmp_path / "adir"
    directory.mkdir()

    with pytest.raises(FoldingBackendError, match="split chain A3M is not a regular file"):
        msa_result_from_split_paths(
            [split_paths[0], directory],
            target,
            artifact_set_id=_ARTIFACT_SET_ID,
            selected_logical_path=_LOGICAL_PATH,
            merged_source_path=source,
        )


def test_constructor_rejects_target_mismatch(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    split_paths = split_merged_a3m(source, target, tmp_path / "split")
    other_logical = "a3ms/AFDB_AF-0000000000000001_AF-0000000000000003.a3m"

    with pytest.raises(FoldingBackendError, match="selected MSA member does not match target"):
        msa_result_from_split_paths(
            split_paths,
            target,
            artifact_set_id=_ARTIFACT_SET_ID,
            selected_logical_path=other_logical,
            merged_source_path=source,
        )


def test_constructor_rejects_invalid_member_identity(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    split_paths = split_merged_a3m(source, target, tmp_path / "split")

    with pytest.raises(FoldingBackendError, match="invalid model entity identity"):
        msa_result_from_split_paths(
            split_paths,
            target,
            artifact_set_id=_ARTIFACT_SET_ID,
            selected_logical_path="a3ms/plain.a3m",
            merged_source_path=source,
        )


def test_constructor_rejects_chain_count_mismatch(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()

    with pytest.raises(FoldingBackendError, match="merged A3M split produced 1 chains for 2 target chains"):
        msa_result_from_split_paths(
            [source],
            target,
            artifact_set_id=_ARTIFACT_SET_ID,
            selected_logical_path=_LOGICAL_PATH,
            merged_source_path=source,
        )


def test_constructor_rejects_empty_split_a3m(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    split_paths = split_merged_a3m(source, target, tmp_path / "split")
    empty = tmp_path / "empty.a3m"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(FoldingBackendError, match="split chain A3M contains no records"):
        msa_result_from_split_paths(
            [split_paths[0], empty],
            target,
            artifact_set_id=_ARTIFACT_SET_ID,
            selected_logical_path=_LOGICAL_PATH,
            merged_source_path=source,
        )


def test_constructor_rejects_non_file_merged_source(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    split_paths = split_merged_a3m(source, target, tmp_path / "split")

    with pytest.raises(FoldingBackendError, match="merged MSA source is not a regular file"):
        msa_result_from_split_paths(
            split_paths,
            target,
            artifact_set_id=_ARTIFACT_SET_ID,
            selected_logical_path=_LOGICAL_PATH,
            merged_source_path=tmp_path / "nope.a3m",
        )


@pytest.mark.parametrize(
    ("artifact_set_id", "selected_logical_path"),
    [
        ("", _LOGICAL_PATH),
        (_ARTIFACT_SET_ID, ""),
    ],
)
def test_constructor_rejects_empty_metadata_strings(
    tmp_path: Path,
    artifact_set_id: str,
    selected_logical_path: str,
) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    split_paths = split_merged_a3m(source, target, tmp_path / "split")

    with pytest.raises(FoldingBackendError):
        msa_result_from_split_paths(
            split_paths,
            target,
            artifact_set_id=artifact_set_id,
            selected_logical_path=selected_logical_path,
            merged_source_path=source,
        )


def test_wrapper_splits_once_and_delegates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    projected = {_LOGICAL_PATH: source}
    output_dir = tmp_path / "split"

    real_split = mp.split_merged_a3m
    split_calls: list[tuple[Path, ProteinTarget, Path]] = []

    def spy_split(source_path: Path, tgt: ProteinTarget, out_dir: Path) -> list[Path]:
        split_calls.append((source_path, tgt, out_dir))
        return real_split(source_path, tgt, out_dir)

    monkeypatch.setattr(mp, "split_merged_a3m", spy_split)

    sentinel = mp.MSAResult(backend="sentinel", chains=())
    captured: dict[str, object] = {}

    def spy_constructor(
        split_paths: list[Path],
        tgt: ProteinTarget,
        *,
        artifact_set_id: str,
        selected_logical_path: str,
        merged_source_path: Path,
    ) -> mp.MSAResult:
        captured["split_paths"] = split_paths
        captured["target"] = tgt
        captured["artifact_set_id"] = artifact_set_id
        captured["selected_logical_path"] = selected_logical_path
        captured["merged_source_path"] = merged_source_path
        return sentinel

    monkeypatch.setattr(mp, "msa_result_from_split_paths", spy_constructor)

    result = prepare_projected_msa(_consumption(), projected, target, output_dir)

    assert result is sentinel
    assert len(split_calls) == 1
    assert split_calls[0][0] == source
    assert split_calls[0][1] == target
    assert split_calls[0][2] == output_dir
    assert captured["artifact_set_id"] == _ARTIFACT_SET_ID
    assert captured["selected_logical_path"] == _LOGICAL_PATH
    assert captured["merged_source_path"] == source
    assert captured["target"] == target
    split_paths = captured["split_paths"]
    assert isinstance(split_paths, list)
    assert len(split_paths) == 2
    assert all(path.is_file() for path in split_paths)


def test_wrapper_result_equals_direct_constructor(tmp_path: Path) -> None:
    source = _write_merged(tmp_path)
    target = _target()
    projected = {_LOGICAL_PATH: source}
    output_dir = tmp_path / "split"

    wrapper_result = prepare_projected_msa(_consumption(), projected, target, output_dir)

    split_paths = split_merged_a3m(source, target, output_dir)
    direct = msa_result_from_split_paths(
        split_paths,
        target,
        artifact_set_id=_ARTIFACT_SET_ID,
        selected_logical_path=_LOGICAL_PATH,
        merged_source_path=source,
    )

    assert wrapper_result == direct
