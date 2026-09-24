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

"""Reject HumanSTRING at frozen postprocessing discovery without losing a subset."""

from __future__ import annotations

import pytest

from bspp.orchestration.runtime.worker.model_inventory import discover_model_ids, is_model_id, parse_compound_model_id


@pytest.mark.parametrize(
    "member",
    [
        "homo_P12345",
        "homo_P12345/",
        "./homo_P12345/ranked_0.pdb",
        "hetero_P12345_Q9Y6K9-model_v1.pdb",
        "hetero_P12345_Q9Y6K9-meta_v1.json",
        "hetero_P12345_Q9Y6K9\\ranked_0.pdb",
    ],
)
@pytest.mark.parametrize("mixed", [False, True])
def test_public_discovery_explicitly_rejects_human_string(member: str, mixed: bool) -> None:
    members = ["AF-0000000000000001/ranked_0.pdb", member] if mixed else [member]
    with pytest.raises(ValueError, match="HumanSTRING postprocessing is unsupported"):
        discover_model_ids(members)


@pytest.mark.parametrize(
    "member",
    [
        "misc/homo_P12345/ranked_0.pdb",
        "homo_P12345.txt",
        "homo_invalid/ranked_0.pdb",
        "AFDB_homo_P12345-model_v1.pdb",
        "hetero_P12345_P12345-meta_v1.json",
        "../homo_P12345/ranked_0.pdb",
        "homo_P12345-model_v1.pdb/irrelevant",
        "homo_P12345-model_v1.pdb-meta_v1.json",
    ],
)
def test_unrelated_names_keep_frozen_ignore_behavior(member: str) -> None:
    assert discover_model_ids([member, "pdb_7amq_assembly_1/ranked_0.pdb"]) == ("pdb_7amq_assembly_1",)


def test_human_names_do_not_enable_frozen_component_mapping() -> None:
    for model_id in ("homo_P12345", "hetero_P12345_Q9Y6K9"):
        assert not is_model_id(model_id)
        assert parse_compound_model_id(model_id) is None
