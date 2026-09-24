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

"""Contract tests for the canonical fold-to-postprocessing model identity surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from bspp.orchestration.contract import model_identity
from bspp.orchestration.contract.model_identity import (
    CANONICAL_META_SUFFIX,
    CANONICAL_MODEL_SUFFIX,
    KNOWN_SUFFIXES,
    MAX_PROTEINS_PER_SHARD,
    is_compound_model_entity_id,
    is_homodimer_model_entity_id,
    is_pdb_assembly_model_entity_id,
    normalize_model_entity_id,
    parse_compound_model_entity_id,
)

HOMODIMER = "AF-0000000000000001"


def test_known_suffixes_exact_order() -> None:
    assert KNOWN_SUFFIXES == [
        "-model_v1.pdb",
        "-meta_v1.json",
        ".merged_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb",
        ".merged_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
        "_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb",
        "_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
    ]
    assert isinstance(KNOWN_SUFFIXES, list)


def test_shard_and_canonical_suffix_constants() -> None:
    assert MAX_PROTEINS_PER_SHARD == 5000
    assert CANONICAL_MODEL_SUFFIX == "-model_v1.pdb"
    assert CANONICAL_META_SUFFIX == "-meta_v1.json"


def test_homodimer_acceptance() -> None:
    assert normalize_model_entity_id(HOMODIMER) == HOMODIMER
    assert is_homodimer_model_entity_id(HOMODIMER) is True
    assert is_compound_model_entity_id(HOMODIMER) is False
    assert parse_compound_model_entity_id(HOMODIMER) is None


def test_af_underscore_normalization() -> None:
    assert normalize_model_entity_id("AF_0000000000000001") == HOMODIMER


def test_afdb_prefix_stripping() -> None:
    assert normalize_model_entity_id("AFDB_AF_0000000000000001") == HOMODIMER
    assert normalize_model_entity_id("AFDB_AF_0000000000000001_model") == HOMODIMER


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("AF-0000000000000001-model_v1.pdb", HOMODIMER),
        ("AF-0000000000000001-meta_v1.json", HOMODIMER),
        (
            "AF-0000000000000001.merged_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb",
            HOMODIMER,
        ),
        (
            "AF-0000000000000001.merged_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
            HOMODIMER,
        ),
        (
            "AF-0000000000000001_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb",
            HOMODIMER,
        ),
        (
            "AF-0000000000000001_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
            HOMODIMER,
        ),
        ("AFDB_AF_0000000000000001-model_v1.pdb", HOMODIMER),
    ],
)
def test_normalization_from_harvested_filename_forms(filename: str, expected: str) -> None:
    assert normalize_model_entity_id(filename) == expected


@pytest.mark.parametrize(
    "compound_id",
    [
        "AF_ABC_AF_DEF",
        "AF-ABC-AF-DEF",
        "AF-ABC_AF-DEF",
        "AF_ABC-AF_DEF",
    ],
)
def test_compound_acceptance_and_parsing(compound_id: str) -> None:
    normalized = normalize_model_entity_id(compound_id)
    assert is_compound_model_entity_id(compound_id) is True
    assert is_homodimer_model_entity_id(compound_id) is False
    assert parse_compound_model_entity_id(compound_id) == ("AF-ABC", "AF-DEF")
    assert normalized.startswith("AF-ABC")


def test_legacy_compound_normalizes_leading_separator_only() -> None:
    assert normalize_model_entity_id("AF_ABC_AF_DEF") == "AF-ABC_AF_DEF"


@pytest.mark.parametrize(
    "malformed",
    [
        "",
        "AF-123",
        "AF-000000000000000",
        "AF-00000000000000001",
        "AF-000000000000000X",
        "AF-0000000000000001-extra",
        "prefix-AF-0000000000000001",
        "AF_ABC",
        "AF_ABC_AF_",
        "AF__0000000000000001",
        "AFDB_AFDB_AF_0000000000000001",
        "AF-0000000000000001-unknown.pdb",
    ],
)
def test_malformed_ids_are_rejected(malformed: str) -> None:
    with pytest.raises(ValueError):
        normalize_model_entity_id(malformed)


def test_empty_and_non_string_values_fail() -> None:
    with pytest.raises(ValueError):
        normalize_model_entity_id("")
    with pytest.raises(ValueError):
        normalize_model_entity_id(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        is_homodimer_model_entity_id("")
    with pytest.raises(ValueError):
        is_compound_model_entity_id("")


def test_predicate_helpers_return_false_for_malformed() -> None:
    assert is_homodimer_model_entity_id("AF-123") is False
    assert is_compound_model_entity_id("AF-ABC") is False


def test_parse_compound_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        parse_compound_model_entity_id("not-a-model-id")


def test_normalize_model_entity_id_accepts_pdb_assembly() -> None:
    """PDB assembly identities pass through normalization unchanged."""
    assert normalize_model_entity_id("pdb_5snm_assembly_1") == "pdb_5snm_assembly_1"


def test_parse_compound_returns_none_for_pdb_assembly() -> None:
    """PDB assembly identities are not compound — parse returns None, not ValueError."""
    assert parse_compound_model_entity_id("pdb_5snm_assembly_1") is None


def test_is_homodimer_returns_false_for_pdb_assembly() -> None:
    assert is_homodimer_model_entity_id("pdb_5snm_assembly_1") is False


def test_is_pdb_assembly_model_entity_id_accepts_pdb_stems() -> None:
    assert is_pdb_assembly_model_entity_id("pdb_5snm_assembly_1") is True
    assert is_pdb_assembly_model_entity_id("pdb_1abc_assembly_3") is True


def test_is_pdb_assembly_model_entity_id_rejects_uppercase() -> None:
    """Uppercase PDB assembly identities are rejected (lowercase-only regex)."""
    assert is_pdb_assembly_model_entity_id("PDB_5SNM_ASSEMBLY_1") is False


def test_is_compound_returns_false_for_pdb_assembly() -> None:
    assert is_compound_model_entity_id("pdb_5snm_assembly_1") is False


def test_contract_module_does_not_import_runtime() -> None:
    source_path = Path(model_identity.__file__)  # type: ignore[arg-type]
    assert "orchestration.runtime" not in source_path.read_text()
