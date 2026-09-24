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

"""Tests for bspp.orchestration.runtime.discovery."""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.runtime.discovery import build_file_index, discover, extract_model_id

# --- extract_model_id ---


def test_extract_model_id_canonical_pdb() -> None:
    assert extract_model_id("AF-A0A000-model_v1.pdb") == "AF-A0A000"


def test_extract_model_id_canonical_json() -> None:
    assert extract_model_id("AF-A0A000-meta_v1.json") == "AF-A0A000"


def test_extract_model_id_colabfold_pdb() -> None:
    name = "AF_A0A000_AF_B0B000.merged_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
    assert extract_model_id(name) == "AF_A0A000_AF_B0B000"


def test_extract_model_id_colabfold_scores_json() -> None:
    name = "AF_A0A000_AF_B0B000.merged_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json"
    assert extract_model_id(name) == "AF_A0A000_AF_B0B000"


def test_extract_model_id_strips_afdb_prefix() -> None:
    name = "AFDB_AF_X0X000-model_v1.pdb"
    assert extract_model_id(name) == "AF_X0X000"


def test_extract_model_id_unknown_suffix_returns_none() -> None:
    assert extract_model_id("readme.txt") is None
    assert extract_model_id("archive.tar.gz") is None


def test_extract_model_id_empty_string() -> None:
    assert extract_model_id("") is None


# --- discover ---


def test_discover_scans_directory(tmp_path: Path) -> None:
    (tmp_path / "AF-P12345-model_v1.pdb").write_bytes(b"")
    (tmp_path / "AF-P12345-meta_v1.json").write_bytes(b"")
    (tmp_path / "AF-Q99999-model_v1.pdb").write_bytes(b"")
    (tmp_path / "readme.txt").write_bytes(b"")

    result = discover(tmp_path)
    assert result == ["AF-P12345", "AF-Q99999"]


def test_discover_ignores_directories(tmp_path: Path) -> None:
    (tmp_path / "AF-P12345-model_v1.pdb").mkdir()
    result = discover(tmp_path)
    assert result == []


def test_discover_empty_directory(tmp_path: Path) -> None:
    result = discover(tmp_path)
    assert result == []


def test_discover_deduplicates(tmp_path: Path) -> None:
    # Same model_id from pdb and json files
    (tmp_path / "AF-DUP001-model_v1.pdb").write_bytes(b"")
    (tmp_path / "AF-DUP001-meta_v1.json").write_bytes(b"")
    result = discover(tmp_path)
    assert result == ["AF-DUP001"]


# --- build_file_index ---


def test_build_file_index(tmp_path: Path) -> None:
    (tmp_path / "AF-P12345-model_v1.pdb").write_bytes(b"")
    (tmp_path / "AF-P12345-meta_v1.json").write_bytes(b"")
    (tmp_path / "AF-Q99999-model_v1.pdb").write_bytes(b"")

    index = build_file_index(tmp_path)
    assert "AF-P12345" in index
    assert "AF-Q99999" in index
    assert sorted(index["AF-P12345"]) == ["AF-P12345-meta_v1.json", "AF-P12345-model_v1.pdb"]
    assert index["AF-Q99999"] == ["AF-Q99999-model_v1.pdb"]


def test_build_file_index_empty(tmp_path: Path) -> None:
    index = build_file_index(tmp_path)
    assert index == {}
