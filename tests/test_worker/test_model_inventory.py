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

from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from bspp.orchestration.runtime.worker.model_inventory import (
    canonical_link_model_id,
    discover_model_ids,
    filter_archive_members_by_allowlist,
    filter_model_ids_by_allowlist,
    is_compound_model_id,
    is_model_id,
    parse_compound_model_id,
    plan_reassigned_model_id_mapping,
    rewrite_reassigned_model_ids,
)


def _homodimer(index: int) -> str:
    return f"AF-{index:016d}"


def test_archive_allowlist_keeps_eight_and_drops_two_in_archive_order() -> None:
    model_ids = [_homodimer(index) for index in range(1, 11)]
    archive_members = [f"{model_id}/ranked_0.pdb" for model_id in model_ids]
    allowlist = [
        _homodimer(8),
        _homodimer(7),
        _homodimer(6),
        _homodimer(5),
        _homodimer(4),
        _homodimer(3),
        _homodimer(2),
        _homodimer(1),
        _homodimer(99),
    ]

    selection = filter_archive_members_by_allowlist(archive_members, allowlist)

    assert selection.kept_model_ids == tuple(model_ids[:8])
    assert selection.dropped_model_ids == tuple(model_ids[8:])
    assert selection.missing_allowlist_model_ids == (_homodimer(99),)


def test_discovery_deduplicates_duplicate_archive_members() -> None:
    first = _homodimer(1)
    second = _homodimer(2)
    compound = "AF_1001_AF_1002"

    assert discover_model_ids(
        [
            f"{first}/ranked_0.pdb",
            f"{first}/scores.json",
            PurePosixPath(f"{second}/ranked_0.pdb"),
            f"{compound}/ranked_0.pdb",
            f"{compound}/scores.json",
        ],
    ) == (first, second, compound)


def test_discovery_finds_wp8a_flat_archive_member_prefixes() -> None:
    homodimer = _homodimer(1)
    compound = "AF_1001_AF_1002"

    assert discover_model_ids(
        [
            f"{homodimer}-model_v1.pdb",
            f"{homodimer}-meta_v1.json",
            f"{compound}.merged_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb",
            f"{compound}.merged_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
        ],
    ) == (homodimer, compound)


def test_discovery_strips_afdb_prefix_from_flat_homodimer_archive_members() -> None:
    homodimer = _homodimer(1)
    compound = "AF_1001_AF_1002"

    assert discover_model_ids(
        [
            "AFDB_AF_0000000000000001-meta_v1.json",
            "AFDB_AF_0000000000000001-model_v1.pdb",
            f"AFDB_{compound}.merged_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
        ],
    ) == (homodimer, compound)


def test_discovery_normalizes_current_upstream_single_af_groups() -> None:
    assert discover_model_ids(
        [
            "AF_123-model_v1.pdb",
            "AF-456-model_v1.pdb",
            "AFDB_AF_789-meta_v1.json",
        ],
    ) == ("AF-123", "AF-456", "AF-789")


def test_discovery_preserves_current_upstream_compound_stems() -> None:
    underscore_compound = "AF_123_AF_456"
    hyphen_compound = "AF-123-AF-456"

    assert discover_model_ids(
        [
            f"{underscore_compound}.merged_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
            f"{hyphen_compound}-model_v1.pdb",
            f"{hyphen_compound}/ranked_0.pdb",
        ],
    ) == (underscore_compound, hyphen_compound)


def test_canonical_link_model_id_returns_link_ids_not_inventory_ids() -> None:
    compound = "AF_1001_AF_1002"

    assert canonical_link_model_id("AF_0000000000000001") == _homodimer(1)
    assert canonical_link_model_id("AFDB_AF_0000000000000001-meta_v1.json") == _homodimer(1)
    assert canonical_link_model_id(compound) == "AF-1001_AF_1002"
    assert is_model_id(compound)
    assert not is_model_id(canonical_link_model_id(compound))


def test_allowlist_filtering_uses_exact_boundaries() -> None:
    archive_only = _homodimer(3)
    allowlist_only = _homodimer(4)
    compound = "AF_1001_AF_1002"
    selection = filter_model_ids_by_allowlist(
        [_homodimer(1), compound, archive_only],
        [compound, _homodimer(1), "AF_1001", allowlist_only],
    )

    assert selection.kept_model_ids == (_homodimer(1), compound)
    assert selection.dropped_model_ids == (archive_only,)
    assert selection.missing_allowlist_model_ids == ("AF_1001", allowlist_only)


def test_missing_allowlist_ids_are_informational_only() -> None:
    archive_only = _homodimer(2)
    missing_allowlist_only = _homodimer(99)

    selection = filter_model_ids_by_allowlist(
        [_homodimer(1), archive_only],
        [_homodimer(1), missing_allowlist_only],
    )

    assert selection.kept_model_ids == (_homodimer(1),)
    assert selection.dropped_model_ids == (archive_only,)
    assert selection.missing_allowlist_model_ids == (missing_allowlist_only,)


def test_compound_heterodimer_ids_are_recognized_without_rewriting() -> None:
    compound = "AF_1001_AF_1002"

    assert parse_compound_model_id(compound) == ("AF-1001", "AF-1002")
    assert is_compound_model_id(compound)
    assert is_model_id(compound)
    assert discover_model_ids([f"{compound}/ranked_0.pdb"]) == (compound,)


def test_homodimer_ids_remain_exact() -> None:
    model_id = _homodimer(1)

    assert is_model_id(model_id)
    assert parse_compound_model_id(model_id) is None
    assert discover_model_ids([f"./{model_id}/ranked_0.pdb"]) == (model_id,)


def test_reassigned_model_id_mapping_accepts_tuple_and_string_rows() -> None:
    mapping = plan_reassigned_model_id_mapping(
        [
            (_homodimer(1), _homodimer(11)),
            f"{_homodimer(2)},{_homodimer(12)}",
            f"{_homodimer(3)} {_homodimer(13)}",
            "",
        ],
    )

    assert mapping == {
        _homodimer(1): _homodimer(11),
        _homodimer(2): _homodimer(12),
        _homodimer(3): _homodimer(13),
    }


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ("old,new,extra", "exactly two fields"),
        ("old new extra", "exactly two fields"),
        (("old",), "tuple rows must contain exactly two fields"),
        (("old", "new", "extra"), "tuple rows must contain exactly two fields"),
    ],
)
def test_reassigned_model_id_mapping_rejects_malformed_rows(row: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        plan_reassigned_model_id_mapping([row])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("row", "match"),
    [
        (("", _homodimer(11)), "old_id cannot contain empty model IDs"),
        ((_homodimer(1), ""), "new_id cannot contain empty model IDs"),
        (f"{_homodimer(1)},", "new_id cannot contain empty model IDs"),
    ],
)
def test_reassigned_model_id_mapping_rejects_empty_fields(row: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        plan_reassigned_model_id_mapping([row])  # type: ignore[list-item]


def test_reassigned_model_id_mapping_rejects_conflicting_old_ids() -> None:
    with pytest.raises(ValueError, match="conflicting reassigned model ID"):
        plan_reassigned_model_id_mapping(
            [
                (_homodimer(1), _homodimer(11)),
                (_homodimer(1), _homodimer(12)),
            ],
        )


def test_reassigned_model_id_mapping_rejects_duplicate_new_ids() -> None:
    with pytest.raises(ValueError, match="duplicate reassigned new model ID"):
        plan_reassigned_model_id_mapping(
            [
                (_homodimer(1), _homodimer(11)),
                (_homodimer(2), _homodimer(11)),
            ],
        )


def test_reassigned_model_id_rewrite_preserves_iterable_order() -> None:
    mapping = plan_reassigned_model_id_mapping(
        [
            (_homodimer(1), _homodimer(11)),
            (_homodimer(3), _homodimer(13)),
        ],
    )

    assert rewrite_reassigned_model_ids(
        (_homodimer(1), _homodimer(2), _homodimer(3)),
        mapping,
    ) == (_homodimer(11), _homodimer(2), _homodimer(13))


def test_non_model_archive_members_are_ignored() -> None:
    model_id = _homodimer(1)

    assert discover_model_ids(
        [
            "README.txt",
            "metadata/AF-0000000000000002/ranked_0.pdb",
            "metadata/AF-0000000000000002-model_v1.pdb",
            "__MACOSX/AF-0000000000000003/ranked_0.pdb",
            ".",
            "../AF-0000000000000004/ranked_0.pdb",
            "AF_1001/ranked_0.pdb",
            "AF-0001/ranked_0.pdb",
            f"{model_id}/ranked_0.pdb",
        ],
    ) == (model_id,)


def test_empty_model_ids_raise_for_explicit_model_id_inputs() -> None:
    with pytest.raises(ValueError, match="empty model IDs"):
        parse_compound_model_id("")

    with pytest.raises(ValueError, match="empty model IDs"):
        filter_model_ids_by_allowlist([_homodimer(1)], [""])


# --- pdb-assembly identity tests (additive) ---


def test_discovery_finds_pdb_assembly_flat_member() -> None:
    assert discover_model_ids(
        ["pdb_5snm_assembly_1-model_v1.pdb", "pdb_5snm_assembly_1-meta_v1.json"],
    ) == ("pdb_5snm_assembly_1",)


def test_discovery_finds_pdb_assembly_directory_member() -> None:
    assert discover_model_ids(
        ["pdb_5snm_assembly_1/ranked_0.pdb", "pdb_5snm_assembly_1/scores.json"],
    ) == ("pdb_5snm_assembly_1",)


def test_discovery_finds_mixed_af_and_pdb_assembly_members() -> None:
    assert discover_model_ids(
        [
            "AF-0000000000000001-model_v1.pdb",
            "pdb_5snm_assembly_1-model_v1.pdb",
            "AF_1001_AF_1002/ranked_0.pdb",
        ],
    ) == ("AF-0000000000000001", "pdb_5snm_assembly_1", "AF_1001_AF_1002")


@pytest.mark.parametrize(
    "member",
    [
        "PDB_5snm_assembly_1-model_v1.pdb",
        "pdb_5SNM_assembly_1-model_v1.pdb",
        "pdb_5snm_assembly_-model_v1.pdb",
        "pdb_5snm_assembly-model_v1.pdb",
        "pdb_5snm_assembly_1_extra-model_v1.pdb",
        "pdb_assembly_1-model_v1.pdb",
    ],
)
def test_malformed_pdb_assembly_members_rejected(member: str) -> None:
    assert discover_model_ids([member]) == ()


def test_pdb_assembly_is_model_id() -> None:
    assert is_model_id("pdb_5snm_assembly_1")


@pytest.mark.parametrize(
    "value",
    [
        "PDB_5snm_assembly_1",
        "pdb_5SNM_assembly_1",
        "pdb_5snm_assembly_",
        "pdb_5snm_assembly",
        "pdb_assembly_1",
        "pdb_5snm_assembly_1_extra",
    ],
)
def test_malformed_pdb_assembly_not_model_id(value: str) -> None:
    assert not is_model_id(value)


def test_empty_model_id_raises_value_error() -> None:
    with pytest.raises(ValueError):
        is_model_id("")


def test_legacy_af_discovery_unchanged_after_pdb_extension() -> None:
    assert discover_model_ids(
        [
            "AF-0000000000000001-model_v1.pdb",
            "AF-0000000000000002/ranked_0.pdb",
            "AF_1001_AF_1002-model_v1.pdb",
        ],
    ) == ("AF-0000000000000001", "AF-0000000000000002", "AF_1001_AF_1002")


def test_filter_model_ids_by_allowlist_with_pdb_assembly() -> None:
    selection = filter_model_ids_by_allowlist(
        ["pdb_5snm_assembly_1", "pdb_6snm_assembly_2", "AF-0000000000000001"],
        ["pdb_5snm_assembly_1"],
    )
    assert selection.kept_model_ids == ("pdb_5snm_assembly_1",)
    assert selection.dropped_model_ids == ("pdb_6snm_assembly_2", "AF-0000000000000001")
    assert selection.missing_allowlist_model_ids == ()


def test_filter_archive_members_by_allowlist_with_pdb_assembly() -> None:
    selection = filter_archive_members_by_allowlist(
        ["pdb_5snm_assembly_1-model_v1.pdb", "pdb_6snm_assembly_2-model_v1.pdb"],
        ["pdb_5snm_assembly_1"],
    )
    assert selection.kept_model_ids == ("pdb_5snm_assembly_1",)
    assert selection.dropped_model_ids == ("pdb_6snm_assembly_2",)


def test_canonical_link_model_id_pdb_assembly_pass_through() -> None:
    assert canonical_link_model_id("pdb_5snm_assembly_1") == "pdb_5snm_assembly_1"
    assert canonical_link_model_id("pdb_5snm_assembly_1-model_v1.pdb") == "pdb_5snm_assembly_1"
