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

"""Additive HumanSTRING grammar; existing AF/PDB policy remains frozen."""

from __future__ import annotations

import pytest

from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.model_identity import (
    KNOWN_SUFFIXES,
    is_compound_model_entity_id,
    is_homodimer_model_entity_id,
    is_human_string_model_entity_id,
    is_pdb_assembly_model_entity_id,
    normalize_model_entity_id,
    parse_compound_model_entity_id,
    parse_human_string_model_entity_id,
)

VALID = ("homo_P12345", "homo_A0A024RBG1", "hetero_Q9Y6K9_P12345", "hetero_A0A024RBG1_Q9Y6K9")


@pytest.mark.parametrize("model_id", VALID)
def test_exact_human_identity_and_suffixes(model_id: str) -> None:
    expected = tuple(model_id.split("_")[1:])
    for suffix in ("", *KNOWN_SUFFIXES):
        value = model_id + suffix
        assert normalize_model_entity_id(value) == model_id
        assert parse_human_string_model_entity_id(value) == expected
        assert is_human_string_model_entity_id(value)
        assert not is_homodimer_model_entity_id(value)
        assert not is_compound_model_entity_id(value)
        assert not is_pdb_assembly_model_entity_id(value)
        assert parse_compound_model_entity_id(value) is None
    record = MsaSetConsumption("sha256:" + "a" * 64, 1, (f"a3ms/{model_id}.a3m",), True)
    assert record.to_mapping()["member_a3m_paths"] == [f"a3ms/{model_id}.a3m"]


@pytest.mark.parametrize(
    "value",
    [
        "homo_p12345",
        "homo_P\uff11\uff12\uff13\uff14\uff15",
        "homo_P12345-2",
        "homo_P12345_extra",
        "homo_P12345 ",
        "hetero_P12345_P12345",
        "hetero_P12345",
        "hetero_P12345_Q9Y6K9_A0A024RBG1",
        "homo_123456",
        "homo_A00000",
        "homo_A0A024RBG12",
        "homo_O0A024RBG1",
        "../homo_P12345",
        "/homo_P12345",
        "homo_P12345/child",
        "AFDB_homo_P12345",
        "homo_P12345_model",
        "AFDB_hetero_P12345_Q9Y6K9_model",
        "homo_P12345-model_v1.pdb-meta_v1.json",
        "homo_P12345\n",
    ],
)
def test_malformed_human_identity_rejected(value: str) -> None:
    assert not is_human_string_model_entity_id(value)
    with pytest.raises(ValueError):
        normalize_model_entity_id(value)
    with pytest.raises(ValueError):
        parse_human_string_model_entity_id(value)


@pytest.mark.parametrize(
    "stem", ["homo_P12345-model_v1.pdb", "AFDB_homo_P12345", "AF-0000000000000001", "hetero_P12345_Q9Y6K9_meta"]
)
def test_member_gate_does_not_accept_filename_aliases_or_new_af_forms(stem: str) -> None:
    with pytest.raises(ValueError):
        MsaSetConsumption("sha256:" + "a" * 64, 1, (f"a3ms/{stem}.a3m",), True)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("AFDB_AF_0000000000000001_model", "AF-0000000000000001"),
        ("AF_ABC_AF_DEF", "AF-ABC_AF_DEF"),
        ("pdb_7amq_assembly_1-model_v1.pdb", "pdb_7amq_assembly_1"),
    ],
)
def test_legacy_normalization_unchanged(value: str, expected: str) -> None:
    assert normalize_model_entity_id(value) == expected
    assert not is_human_string_model_entity_id(value)
    assert parse_human_string_model_entity_id(value) is None
