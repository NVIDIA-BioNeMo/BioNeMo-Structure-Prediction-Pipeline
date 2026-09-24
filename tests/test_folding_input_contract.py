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

"""Contract validation tests for the folding-input seam (e03s03)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest

from bspp.orchestration.contract.folding_input import (
    A3mSplitRules,
    BioIRPolymer,
    BioIRRequestManifest,
    FoldingInputLayout,
    MsaSetConsumption,
    a3m_split_rules_from_mapping,
    bioir_polymer_from_mapping,
    bioir_request_manifest_from_mapping,
    folding_input_layout_from_mapping,
    msa_set_consumption_from_mapping,
    parse_merged_a3m_header,
    split_boundaries,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    MsaArtifactSetManifest,
    MsaChunkManifestReference,
    msa_artifact_set_id,
)


def make_reference(member_count: int, index: int = 0) -> MsaChunkManifestReference:
    chunk_name = f"sample_tranche{index:02d}_0000{index + 1}.fa"
    return MsaChunkManifestReference(
        chunk_name=chunk_name,
        logical_path=f"chunks/{chunk_name.removesuffix('.fa')}.json",
        sha256="a" * 64,
        member_count=member_count,
        logical_bytes=member_count * 100,
    )


def make_manifest(member_count: int = 2) -> tuple[MsaArtifactSetManifest, MsaChunkManifestReference]:
    reference = make_reference(member_count)
    logical_bytes = member_count * 100
    artifact_set_id = msa_artifact_set_id((reference,), member_count, logical_bytes)
    manifest = MsaArtifactSetManifest(
        artifact_set_id=artifact_set_id,
        chunks=(reference,),
        member_count=member_count,
        logical_bytes=logical_bytes,
    )
    return manifest, reference


def make_member_paths(count: int = 2) -> tuple[str, ...]:
    return tuple(f"a3ms/AFDB_AF-{index:016d}.a3m" for index in range(count))


def make_polymer(index: int, *, sequence: str | None = None, chain_ids: tuple[str, ...] | None = None) -> BioIRPolymer:
    return BioIRPolymer(
        chain_ids=chain_ids if chain_ids is not None else (f"chain_{index}",),
        sequence=sequence if sequence is not None else f"SEQ{index}",
        unpaired_msa=f"alignments/polymer_{index:02d}/unpaired.a3m",
    )


def make_bioir_layout(*, polymer_count: int = 2, chain_ids: tuple[str, ...] | None = None) -> FoldingInputLayout:
    return FoldingInputLayout(
        layout="bioir",
        chain_ids=chain_ids if chain_ids is not None else ("A", "B"),
        template_mode="none",
        pairing="bioir-homomer-dummy",
        polymer_count=polymer_count,
        paired_row_count=0,
        use_paired_msa=False,
        max_non_query_msa_rows=5_000,
        bioir_request="bioir-request.json",
    )


# --- FoldingInputLayout: openfold ---


def test_openfold_layout_round_trip() -> None:
    layout = FoldingInputLayout(
        layout="openfold",
        chain_ids=("seq_A", "seq_B"),
        template_mode="none",
        pairing="species-from-colabfold-headers",
    )
    assert layout.to_mapping() == {
        "schema_version": 1,
        "layout": "openfold",
        "fasta_dir": "fasta",
        "alignment_dir": "alignments",
        "template_dir": "templates",
        "chain_ids": ["seq_A", "seq_B"],
        "template_mode": "none",
        "pairing": "species-from-colabfold-headers",
    }
    assert folding_input_layout_from_mapping(layout.to_mapping()) == layout


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pairing": "bioir-homomer-dummy"},
        {"chain_ids": ()},
        {"chain_ids": ("A", "A")},
        {"chain_ids": ("A", "")},
        {"fasta_dir": "wrong"},
        {"alignment_dir": "msa"},
        {"template_dir": "tpl"},
        {"template_mode": "uniref"},
        {"polymer_count": 2},
        {"bioir_request": "bioir-request.json"},
    ],
)
def test_openfold_layout_rejects(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "layout": "openfold",
        "chain_ids": ("A", "B"),
        "template_mode": "none",
        "pairing": "species-from-colabfold-headers",
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        FoldingInputLayout(**base)  # type: ignore[arg-type]


# --- FoldingInputLayout: bioir ---


@pytest.mark.parametrize(
    "pairing",
    [
        "bioir-homomer-dummy",
        "method-c-unpaired",
        "unpaired-only-no-merged-source",
        "colabfold-merged-row-index",
    ],
)
def test_bioir_layout_accepts_all_pairing_modes(pairing: str) -> None:
    layout = FoldingInputLayout(
        layout="bioir",
        chain_ids=("A", "B"),
        template_mode="none",
        pairing=pairing,  # type: ignore[arg-type]
        polymer_count=2,
        paired_row_count=0,
        use_paired_msa=True,
        max_non_query_msa_rows=5_000,
        bioir_request="bioir-request.json",
    )
    assert folding_input_layout_from_mapping(layout.to_mapping()) == layout


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_non_query_msa_rows": -1},
        {"paired_row_count": -1},
        {"use_paired_msa": 1},
        {"bioir_request": "request.json"},
        {"polymer_count": 0},
        {"polymer_count": None},
        {"paired_row_count": None},
        {"use_paired_msa": None},
        {"max_non_query_msa_rows": None},
        {"bioir_request": None},
        {"pairing": "species-from-colabfold-headers"},
    ],
)
def test_bioir_layout_rejects(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "layout": "bioir",
        "chain_ids": ("A", "B"),
        "template_mode": "none",
        "pairing": "bioir-homomer-dummy",
        "polymer_count": 2,
        "paired_row_count": 0,
        "use_paired_msa": False,
        "max_non_query_msa_rows": 5_000,
        "bioir_request": "bioir-request.json",
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        FoldingInputLayout(**base)  # type: ignore[arg-type]


def test_bioir_layout_validate_against_request_matching() -> None:
    layout = make_bioir_layout()
    request = BioIRRequestManifest(
        input_id="input-1",
        polymers=(make_polymer(0, chain_ids=("A",)), make_polymer(1, chain_ids=("B",))),
    )
    layout.validate_against_request(request)


def test_bioir_layout_validate_against_request_wrong_polymer_count() -> None:
    layout = make_bioir_layout(polymer_count=2)
    request = BioIRRequestManifest(input_id="input-1", polymers=(make_polymer(0, chain_ids=("A",)),))
    with pytest.raises(ValueError, match="polymer_count"):
        layout.validate_against_request(request)


def test_bioir_layout_validate_against_request_wrong_chain_ids() -> None:
    layout = make_bioir_layout(chain_ids=("A", "B"))
    request = BioIRRequestManifest(
        input_id="input-1",
        polymers=(make_polymer(0, chain_ids=("A",)), make_polymer(1, chain_ids=("C",))),
    )
    with pytest.raises(ValueError, match="chain_ids"):
        layout.validate_against_request(request)


def test_openfold_layout_validate_against_request_is_illegal() -> None:
    layout = FoldingInputLayout(
        layout="openfold",
        chain_ids=("A",),
        template_mode="none",
        pairing="species-from-colabfold-headers",
    )
    request = BioIRRequestManifest(input_id="input-1", polymers=(make_polymer(0),))
    with pytest.raises(ValueError, match="only legal"):
        layout.validate_against_request(request)


# --- BioIRPolymer and BioIRRequestManifest ---


def test_bioir_request_round_trip() -> None:
    request = BioIRRequestManifest(
        input_id="input-1",
        polymers=(
            BioIRPolymer(("A",), "SEQ0", "alignments/polymer_00/unpaired.a3m"),
            BioIRPolymer(("B",), "SEQ1", "alignments/polymer_01/unpaired.a3m"),
        ),
    )
    assert request.chain_ids == ("A", "B")
    assert bioir_request_manifest_from_mapping(request.to_mapping()) == request


def test_bioir_request_polymer_path_position_binding_accepts() -> None:
    BioIRRequestManifest(
        input_id="input-1",
        polymers=(make_polymer(0), make_polymer(1)),
    )


@pytest.mark.parametrize(
    "polymers",
    [
        (make_polymer(1), make_polymer(0)),  # swapped
        (make_polymer(0), make_polymer(2)),  # skipped
        (make_polymer(0), make_polymer(0)),  # duplicate index
    ],
)
def test_bioir_request_polymer_path_position_binding_rejects(polymers: tuple[BioIRPolymer, ...]) -> None:
    with pytest.raises(ValueError, match="tuple position"):
        BioIRRequestManifest(input_id="input-1", polymers=polymers)


def test_bioir_polymer_paired_msa_must_be_sibling() -> None:
    BioIRPolymer(("A",), "SEQ", "alignments/polymer_00/unpaired.a3m", "alignments/polymer_00/paired.a3m")
    with pytest.raises(ValueError, match="sibling"):
        BioIRPolymer(("A",), "SEQ", "alignments/polymer_00/unpaired.a3m", "alignments/polymer_01/paired.a3m")


def test_bioir_polymer_paired_msa_must_match_shape() -> None:
    with pytest.raises(ValueError, match="paired"):
        BioIRPolymer(("A",), "SEQ", "alignments/polymer_00/unpaired.a3m", "alignments/polymer_00/unpaired.a3m")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_id": ""},
        {"polymers": ()},
        {"polymers": (make_polymer(0), make_polymer(1, sequence="SEQ0"))},  # duplicate sequences
        {
            "polymers": (
                make_polymer(0, chain_ids=("A",)),
                make_polymer(1, chain_ids=("A",)),
            )
        },  # duplicate chain ids
    ],
)
def test_bioir_request_rejects(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "input_id": "input-1",
        "polymers": (make_polymer(0), make_polymer(1)),
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        BioIRRequestManifest(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chain_ids": ()},
        {"chain_ids": ("A", "A")},
        {"chain_ids": ("A", "")},
        {"sequence": ""},
        {"unpaired_msa": "alignments/polymer_00/paired.a3m"},
        {"unpaired_msa": "alignments/polymer_0/unpaired.a3m"},
    ],
)
def test_bioir_polymer_rejects(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "chain_ids": ("A",),
        "sequence": "SEQ",
        "unpaired_msa": "alignments/polymer_00/unpaired.a3m",
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        BioIRPolymer(**base)  # type: ignore[arg-type]


# --- MsaSetConsumption ---


def test_msa_set_consumption_matches_manifest() -> None:
    manifest, _ = make_manifest(2)
    consumption = MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=make_member_paths(2),
        requires_paired_query_header=True,
    )
    consumption.validate_against_manifest(manifest)
    assert msa_set_consumption_from_mapping(consumption.to_mapping()) == consumption


def test_msa_set_consumption_rejects_wrong_artifact_set_id() -> None:
    manifest, _ = make_manifest(2)
    consumption = MsaSetConsumption(
        artifact_set_id="sha256:" + "b" * 64,
        expected_chunk_count=1,
        member_a3m_paths=make_member_paths(2),
        requires_paired_query_header=True,
    )
    with pytest.raises(ValueError, match="artifact_set_id"):
        consumption.validate_against_manifest(manifest)


def test_msa_set_consumption_rejects_chunk_count_mismatch() -> None:
    manifest, _ = make_manifest(2)
    consumption = MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=make_member_paths(2),
        requires_paired_query_header=True,
    )
    two_chunks = object.__new__(MsaArtifactSetManifest)
    object.__setattr__(two_chunks, "artifact_type", "bspp.msa-set/v1")
    object.__setattr__(two_chunks, "artifact_set_id", manifest.artifact_set_id)
    object.__setattr__(two_chunks, "chunks", (make_reference(1), make_reference(1, index=1)))
    object.__setattr__(two_chunks, "member_count", 2)
    object.__setattr__(two_chunks, "logical_bytes", 200)
    object.__setattr__(two_chunks, "schema_version", 1)
    with pytest.raises(ValueError, match="chunk count"):
        consumption.validate_against_manifest(two_chunks)


def test_msa_set_consumption_rejects_member_count_mismatch() -> None:
    manifest, _ = make_manifest(2)
    consumption = MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=make_member_paths(2),
        requires_paired_query_header=True,
    )
    wrong_count = object.__new__(MsaArtifactSetManifest)
    object.__setattr__(wrong_count, "artifact_type", "bspp.msa-set/v1")
    object.__setattr__(wrong_count, "artifact_set_id", manifest.artifact_set_id)
    object.__setattr__(wrong_count, "chunks", (make_reference(1),))
    object.__setattr__(wrong_count, "member_count", 3)
    object.__setattr__(wrong_count, "logical_bytes", 100)
    object.__setattr__(wrong_count, "schema_version", 1)
    with pytest.raises(ValueError, match="member_count"):
        consumption.validate_against_manifest(wrong_count)


@pytest.mark.parametrize(
    "paths",
    [
        ("a3ms/AFDB_AF-0123456789012345.a3m", "a3ms/AFDB_AF-0123456789012345.a3m"),  # duplicate
        ("a3ms/AFDB_AF-0123456789012345.a3m", "a3ms/nested/AFDB_AF-0123456789012346.a3m"),  # nested
        ("a3ms/AFDB_AF-0123456789012345.a3m", "a3ms/../AFDB_AF-0123456789012346.a3m"),  # traversal
        ("a3ms/AFDB_AF-0123456789012345.a3m", "a3ms/not-a-stem.a3m"),  # bad stem
    ],
)
def test_msa_set_consumption_rejects_invalid_member_paths(paths: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        MsaSetConsumption(
            artifact_set_id="sha256:" + "a" * 64,
            expected_chunk_count=1,
            member_a3m_paths=paths,
            requires_paired_query_header=True,
        )


@pytest.mark.parametrize(
    "artifact_set_id",
    [
        "a" * 64,
        "sha256:" + "a" * 63,
        "sha256:" + "A" * 64,
        "sha256:" + "g" * 64,
    ],
)
def test_msa_set_consumption_rejects_invalid_artifact_set_id(artifact_set_id: str) -> None:
    with pytest.raises(ValueError, match="artifact_set_id"):
        MsaSetConsumption(
            artifact_set_id=artifact_set_id,
            expected_chunk_count=1,
            member_a3m_paths=make_member_paths(1),
            requires_paired_query_header=True,
        )


def test_msa_set_consumption_accepts_false_paired_query() -> None:
    """requires_paired_query_header=False is now accepted."""
    consumption = MsaSetConsumption(
        artifact_set_id="sha256:" + "a" * 64,
        expected_chunk_count=1,
        member_a3m_paths=make_member_paths(1),
        requires_paired_query_header=False,
    )
    assert consumption.requires_paired_query_header is False


def test_msa_set_consumption_accepts_false_paired_query_with_pdb_member() -> None:
    """PDB assembly member stem with requires_paired_query_header=False is accepted."""
    consumption = MsaSetConsumption(
        artifact_set_id="sha256:" + "a" * 64,
        expected_chunk_count=1,
        member_a3m_paths=("a3ms/pdb_5snm_assembly_1.a3m",),
        requires_paired_query_header=False,
    )
    assert consumption.requires_paired_query_header is False
    assert consumption.member_a3m_paths == ("a3ms/pdb_5snm_assembly_1.a3m",)


def test_msa_set_consumption_accepts_pdb_assembly_member_stem() -> None:
    """PDB assembly member stems are now accepted by the folding intake grammar."""
    consumption = MsaSetConsumption(
        artifact_set_id="sha256:" + "a" * 64,
        expected_chunk_count=1,
        member_a3m_paths=("a3ms/pdb_5snm_assembly_1.a3m",),
        requires_paired_query_header=True,
    )
    assert consumption.member_a3m_paths == ("a3ms/pdb_5snm_assembly_1.a3m",)


def test_msa_set_consumption_rejects_non_boolean_paired_query() -> None:
    with pytest.raises(ValueError, match="boolean"):
        MsaSetConsumption(
            artifact_set_id="sha256:" + "a" * 64,
            expected_chunk_count=1,
            member_a3m_paths=make_member_paths(1),
            requires_paired_query_header=1,  # type: ignore[arg-type]
        )


def test_msa_set_consumption_rejects_non_one_chunk_count() -> None:
    with pytest.raises(ValueError, match="exactly 1"):
        MsaSetConsumption(
            artifact_set_id="sha256:" + "a" * 64,
            expected_chunk_count=2,
            member_a3m_paths=make_member_paths(1),
            requires_paired_query_header=True,
        )


# --- A3mSplitRules ---


def test_a3m_split_rules_default_round_trip() -> None:
    rules = A3mSplitRules()
    assert a3m_split_rules_from_mapping(rules.to_mapping()) == rules
    assert rules.header_line == "first-non-blank"
    assert rules.header_prefix == "#"
    assert rules.tuple_separator == "\t"
    assert rules.minimum_arity == 1
    assert rules.lengths_cardinalities_rule == "positive-integers-equal-arity"
    assert rules.lowercase_rule == "insertion-never-advances-aligned-position"
    assert rules.gap_characters == (".", "-")
    assert rules.query_record == "first-record"
    assert rules.homomer_mapping_rule == "multiset-match-of-ungapped-query-pieces"
    assert rules.chain_index_base == 1
    assert rules.output_pattern == "chain_{index}.a3m"


@pytest.mark.parametrize(
    "overrides",
    [
        {"header_line": "first-line"},
        {"header_prefix": ">"},
        {"tuple_separator": " "},
        {"minimum_arity": 2},
        {"lengths_cardinalities_rule": "positive-integers"},
        {"lowercase_rule": "insertion"},
        {"gap_characters": (".",)},
        {"query_record": "last-record"},
        {"homomer_mapping_rule": "exact-match"},
        {"chain_index_base": 0},
        {"output_pattern": "chain_{index}.fasta"},
    ],
)
def test_a3m_split_rules_reject_alteration(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        A3mSplitRules(**overrides)  # type: ignore[arg-type]


# --- parse_merged_a3m_header ---


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("#102,98\t1,1", ((102, 98), (1, 1))),
        ("#200\t2", ((200,), (2,))),
        ("#10,20,30\t1,2,3", ((10, 20, 30), (1, 2, 3))),
    ],
)
def test_parse_merged_a3m_header_accepts(line: str, expected: tuple[tuple[int, ...], tuple[int, ...]]) -> None:
    assert parse_merged_a3m_header(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "102,98\t1,1",  # missing #
        "#102,98 1,1",  # space instead of tab
        "#102,98\t1,1\t2,2",  # extra tab field
        "#102,98\t1",  # arity mismatch
        "#0\t1",  # zero
        "#-1\t1",  # negative
        "#102,98\t1,",  # empty comma component
        "#102,98\t1.0",  # non-integer
        "#\t1",  # empty lengths
        "#102,98\t",  # empty cardinalities
        "",  # blank
        "#102,98",  # missing tab
        "#102,,98\t1,1",  # empty component in lengths
        "#+102,98\t1,1",  # sign without digits
    ],
)
def test_parse_merged_a3m_header_rejects(line: str) -> None:
    with pytest.raises(ValueError):
        parse_merged_a3m_header(line)


def test_parse_merged_a3m_header_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="string"):
        parse_merged_a3m_header(123)  # type: ignore[arg-type]


# --- split_boundaries ---


def test_split_boundaries() -> None:
    assert split_boundaries((102, 98)) == (102, 200)
    assert split_boundaries((200,)) == (200,)
    assert split_boundaries((10, 20, 30)) == (10, 30, 60)


@pytest.mark.parametrize(
    "lengths",
    [
        (),
        (0,),
        (-1,),
        (True,),
        (1.5,),
        [1, 2],
        (1, 0),
        (1, True),
    ],
)
def test_split_boundaries_rejects(lengths: object) -> None:
    with pytest.raises(ValueError):
        split_boundaries(lengths)  # type: ignore[arg-type]


# --- Strict mapping loaders ---


def _records() -> list[tuple[str, object, object]]:
    openfold = FoldingInputLayout(
        layout="openfold",
        chain_ids=("A", "B"),
        template_mode="none",
        pairing="species-from-colabfold-headers",
    )
    bioir = make_bioir_layout()
    polymer = BioIRPolymer(("A",), "SEQ0", "alignments/polymer_00/unpaired.a3m")
    request = BioIRRequestManifest(input_id="input-1", polymers=(polymer,))
    manifest, _ = make_manifest(2)
    consumption = MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=make_member_paths(2),
        requires_paired_query_header=True,
    )
    rules = A3mSplitRules()
    return [
        ("openfold", openfold, folding_input_layout_from_mapping),
        ("bioir", bioir, folding_input_layout_from_mapping),
        ("polymer", polymer, bioir_polymer_from_mapping),
        ("request", request, bioir_request_manifest_from_mapping),
        ("consumption", consumption, msa_set_consumption_from_mapping),
        ("rules", rules, a3m_split_rules_from_mapping),
    ]


def test_strict_mapping_round_trips() -> None:
    for _, record, loader in _records():
        assert loader(record.to_mapping()) == record  # type: ignore[operator]


def test_strict_mapping_requires_schema_version() -> None:
    for _, record, loader in _records():
        payload = record.to_mapping()
        del payload["schema_version"]
        with pytest.raises(ValueError, match="missing explicit schema_version"):
            loader(payload)  # type: ignore[operator]


def test_strict_mapping_rejects_explicit_null_schema_version() -> None:
    for _, record, loader in _records():
        payload = record.to_mapping()
        payload["schema_version"] = None
        with pytest.raises(ValueError, match="missing explicit schema_version"):
            loader(payload)  # type: ignore[operator]


def test_strict_mapping_rejects_unsupported_schema_version() -> None:
    for _, record, loader in _records():
        payload = record.to_mapping()
        payload["schema_version"] = 2
        with pytest.raises(ValueError):
            loader(payload)  # type: ignore[operator]


def test_strict_mapping_rejects_unknown_field() -> None:
    for _, record, loader in _records():
        payload = record.to_mapping()
        payload["bogus"] = "x"
        with pytest.raises(ValueError, match="Unknown"):
            loader(payload)  # type: ignore[operator]


def test_strict_mapping_rejects_wrong_container_type() -> None:
    polymer = BioIRPolymer(("A",), "SEQ0", "alignments/polymer_00/unpaired.a3m")
    payload = polymer.to_mapping()
    payload["chain_ids"] = "A"
    with pytest.raises(ValueError):
        bioir_polymer_from_mapping(payload)


def test_strict_mapping_rejects_bool_for_int() -> None:
    bioir = make_bioir_layout()
    payload = bioir.to_mapping()
    payload["polymer_count"] = True
    with pytest.raises(ValueError):
        folding_input_layout_from_mapping(payload)


def test_strict_mapping_rejects_non_bool_use_paired_msa() -> None:
    bioir = make_bioir_layout()
    payload = bioir.to_mapping()
    payload["use_paired_msa"] = 1
    with pytest.raises(ValueError):
        folding_input_layout_from_mapping(payload)


# --- Missing required field reporting ---


def _openfold_payload() -> dict[str, object]:
    return FoldingInputLayout(
        layout="openfold",
        chain_ids=("A", "B"),
        template_mode="none",
        pairing="species-from-colabfold-headers",
    ).to_mapping()


def _bioir_payload() -> dict[str, object]:
    return make_bioir_layout().to_mapping()


def _polymer_payload() -> dict[str, object]:
    return BioIRPolymer(("A",), "SEQ0", "alignments/polymer_00/unpaired.a3m").to_mapping()


def _request_payload() -> dict[str, object]:
    return BioIRRequestManifest(
        input_id="input-1",
        polymers=(BioIRPolymer(("A",), "SEQ0", "alignments/polymer_00/unpaired.a3m"),),
    ).to_mapping()


def _consumption_payload() -> dict[str, object]:
    manifest, _ = make_manifest(2)
    return MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=make_member_paths(2),
        requires_paired_query_header=True,
    ).to_mapping()


def _rules_payload() -> dict[str, object]:
    return A3mSplitRules().to_mapping()


def _assert_missing_fields(
    payload_factory: Callable[[], dict[str, object]],
    loader: Callable[[Mapping[str, object]], object],
    record_name: str,
    required_fields: tuple[str, ...],
) -> None:
    for field in required_fields:
        payload = payload_factory()
        del payload[field]
        with pytest.raises(ValueError, match=f"Missing {record_name} field") as exc:
            loader(payload)
        assert field in str(exc.value)


def test_openfold_layout_missing_required_fields_report_missing() -> None:
    _assert_missing_fields(
        _openfold_payload,
        folding_input_layout_from_mapping,
        "FoldingInputLayout",
        ("fasta_dir", "alignment_dir", "template_dir", "chain_ids", "template_mode", "pairing"),
    )


def test_bioir_layout_missing_required_fields_report_missing() -> None:
    _assert_missing_fields(
        _bioir_payload,
        folding_input_layout_from_mapping,
        "FoldingInputLayout",
        (
            "fasta_dir",
            "alignment_dir",
            "template_dir",
            "chain_ids",
            "template_mode",
            "pairing",
            "polymer_count",
            "paired_row_count",
            "use_paired_msa",
            "max_non_query_msa_rows",
            "bioir_request",
        ),
    )


def test_bioir_polymer_missing_required_fields_report_missing() -> None:
    _assert_missing_fields(
        _polymer_payload,
        bioir_polymer_from_mapping,
        "BioIRPolymer",
        ("chain_ids", "sequence", "unpaired_msa"),
    )


def test_bioir_request_missing_required_fields_report_missing() -> None:
    _assert_missing_fields(
        _request_payload,
        bioir_request_manifest_from_mapping,
        "BioIRRequestManifest",
        ("input_id", "polymers"),
    )


def test_msa_set_consumption_missing_required_fields_report_missing() -> None:
    _assert_missing_fields(
        _consumption_payload,
        msa_set_consumption_from_mapping,
        "MsaSetConsumption",
        (
            "artifact_type",
            "artifact_set_id",
            "expected_chunk_count",
            "member_a3m_paths",
            "requires_paired_query_header",
        ),
    )


def test_a3m_split_rules_missing_required_fields_report_missing() -> None:
    _assert_missing_fields(
        _rules_payload,
        a3m_split_rules_from_mapping,
        "A3mSplitRules",
        (
            "header_line",
            "header_prefix",
            "tuple_separator",
            "minimum_arity",
            "lengths_cardinalities_rule",
            "lowercase_rule",
            "gap_characters",
            "query_record",
            "homomer_mapping_rule",
            "chain_index_base",
            "output_pattern",
        ),
    )


def test_bioir_polymer_paired_msa_null_loads() -> None:
    polymer = BioIRPolymer(("A",), "SEQ0", "alignments/polymer_00/unpaired.a3m")

    assert bioir_polymer_from_mapping(polymer.to_mapping()) == polymer


def test_openfold_layout_without_bioir_fields_loads() -> None:
    layout = FoldingInputLayout(
        layout="openfold",
        chain_ids=("A", "B"),
        template_mode="none",
        pairing="species-from-colabfold-headers",
    )
    payload = layout.to_mapping()

    assert "polymer_count" not in payload
    assert folding_input_layout_from_mapping(payload) == layout


def test_openfold_layout_rejects_bioir_fields_through_mapping() -> None:
    payload = _openfold_payload()
    payload["polymer_count"] = 2

    with pytest.raises(ValueError, match="Unknown FoldingInputLayout field"):
        folding_input_layout_from_mapping(payload)
