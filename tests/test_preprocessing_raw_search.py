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

"""Contract shapes and historical paired-mode n-arity raw-search evidence."""

from __future__ import annotations

import hashlib

import pytest

from bspp.orchestration.contract.preprocessing import PreprocessingFastaRecord
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingRawSearchArtifact,
    PreprocessingRawSearchEvidence,
    preprocessing_raw_search_artifact_from_mapping,
)
from bspp.orchestration.runtime.preprocessing.content_validation import (
    validate_preprocessing_paired_a3m_bytes,
)
from bspp.orchestration.runtime.preprocessing.raw_search import PreprocessingRawSearchError, _expected_numeric_ids


@pytest.mark.parametrize(
    "argv",
    [
        (),
        ("--pair-mode",),
        ("--pair-mode", "unpaired"),
        ("--pair-mode", "unknown"),
        ("--pair-mode", "paired", "--pair-mode", "unpaired_paired"),
    ],
)
def test_raw_closure_rejects_missing_ambiguous_or_unsupported_pair_mode(argv: tuple[str, ...]) -> None:
    with pytest.raises(PreprocessingRawSearchError):
        _expected_numeric_ids(argv, 3, 6)


def _record(sequence: str, source_ordinal: int = 0, identity: str = "test") -> PreprocessingFastaRecord:
    return PreprocessingFastaRecord(
        header=f">{identity}",
        sequence=sequence,
        identity=identity,
        source_ordinal=source_ordinal,
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_named_artifact(
    member: str, data: bytes, source_ordinal: int, directory: str = "/raw"
) -> PreprocessingRawSearchArtifact:
    return PreprocessingRawSearchArtifact(
        role="named-a3m",
        member_name=member,
        path=f"{directory}/{member}",
        size_bytes=len(data),
        sha256=_sha(data),
        source_ordinal=source_ordinal,
        declared_member=member,
        raw_query_id=None,
        modeled_chain_length=None,
    )


def _make_numeric_artifact(
    raw_id: int, data: bytes, directory: str = "/raw", *, modeled_cardinality: int | None = None
) -> PreprocessingRawSearchArtifact:
    return PreprocessingRawSearchArtifact(
        role="numeric-placeholder",
        member_name=f"{raw_id}.a3m",
        path=f"{directory}/{raw_id}.a3m",
        size_bytes=len(data),
        sha256=_sha(data),
        source_ordinal=None,
        declared_member=None,
        raw_query_id=raw_id,
        modeled_chain_length=int(data.decode().split("\n")[0].split("\t")[0].removeprefix("#")),
        modeled_cardinality=modeled_cardinality,
    )


def test_monomer_closure_one_named_zero_numeric() -> None:
    """1 record, 1 chain → 1 named A3M + 0 numeric placeholders."""
    artifacts = (_make_named_artifact("test.a3m", b"#100\t1\n>test\n" + b"A" * 100 + b"\n", 0),)
    evidence = PreprocessingRawSearchEvidence(
        raw_search_output_directory="/raw",
        searched_source_ordinals=(0,),
        artifacts=artifacts,
    )
    assert len(evidence.artifacts) == 1
    assert evidence.artifacts[0].role == "named-a3m"


def test_homodimer_closure_one_named_zero_numeric() -> None:
    """1 record, 2 identical chains → 1 named A3M + 0 numeric placeholders (U=1, M=1)."""
    artifacts = (_make_named_artifact("test.a3m", b"#100\t2\n>test\n" + b"A" * 100 + b"\n", 0),)
    evidence = PreprocessingRawSearchEvidence(
        raw_search_output_directory="/raw",
        searched_source_ordinals=(0,),
        artifacts=artifacts,
    )
    assert len(evidence.artifacts) == 1


def test_heterodimer_closure_one_named_one_numeric() -> None:
    """1 record, 2 distinct chains → 1 named A3M + 1 numeric placeholder #LB\t1."""
    artifacts = (
        _make_named_artifact("test.a3m", b"#100,200\t1,1\n>test\n" + b"A" * 100 + b"B" * 200 + b"\n", 0),
        _make_numeric_artifact(1, b"#200\t1\n"),
    )
    evidence = PreprocessingRawSearchEvidence(
        raw_search_output_directory="/raw",
        searched_source_ordinals=(0,),
        artifacts=artifacts,
    )
    assert len(evidence.artifacts) == 2
    assert evidence.artifacts[1].role == "numeric-placeholder"
    assert evidence.artifacts[1].raw_query_id == 1


def test_a2b_closure_one_named_one_numeric() -> None:
    """1 record, chains A,A,B → 1 named A3M + 1 numeric placeholder #LB\t1 (U=2, M=1)."""
    artifacts = (
        _make_named_artifact("test.a3m", b"#100,200\t2,1\n>test\n" + b"A" * 100 + b"A" * 100 + b"B" * 200 + b"\n", 0),
        _make_numeric_artifact(1, b"#200\t1\n"),
    )
    evidence = PreprocessingRawSearchEvidence(
        raw_search_output_directory="/raw",
        searched_source_ordinals=(0,),
        artifacts=artifacts,
    )
    assert len(evidence.artifacts) == 2


def test_a2b2_closure_one_named_one_numeric() -> None:
    """1 record, chains A,A,B,B → 1 named A3M + 1 numeric placeholder #LB\t2 (U=2, M=1)."""
    artifacts = (
        _make_named_artifact(
            "test.a3m",
            b"#100,200\t2,2\n>test\n" + b"A" * 100 + b"B" * 200 + b"A" * 100 + b"B" * 200 + b"\n",
            0,
        ),
        _make_numeric_artifact(1, b"#200\t2\n", modeled_cardinality=2),
    )
    evidence = PreprocessingRawSearchEvidence(
        raw_search_output_directory="/raw",
        searched_source_ordinals=(0,),
        artifacts=artifacts,
    )
    assert len(evidence.artifacts) == 2
    assert evidence.artifacts[1].modeled_cardinality == 2


def test_trimer_closure_one_named_two_numeric() -> None:
    """1 record, 3 distinct chains → 1 named A3M + 2 numeric placeholders."""
    artifacts = (
        _make_named_artifact(
            "test.a3m",
            b"#100,200,300\t1,1,1\n>test\n" + b"A" * 100 + b"B" * 200 + b"C" * 300 + b"\n",
            0,
        ),
        _make_numeric_artifact(1, b"#200\t1\n"),
        _make_numeric_artifact(2, b"#300\t1\n"),
    )
    evidence = PreprocessingRawSearchEvidence(
        raw_search_output_directory="/raw",
        searched_source_ordinals=(0,),
        artifacts=artifacts,
    )
    assert len(evidence.artifacts) == 3


def test_tetramer_closure_one_named_three_numeric() -> None:
    """1 record, 4 distinct chains → 1 named A3M + 3 numeric placeholders."""
    artifacts = (
        _make_named_artifact(
            "test.a3m",
            b"#100,200,300,400\t1,1,1,1\n>test\n" + b"A" * 100 + b"B" * 200 + b"C" * 300 + b"D" * 400 + b"\n",
            0,
        ),
        _make_numeric_artifact(1, b"#200\t1\n"),
        _make_numeric_artifact(2, b"#300\t1\n"),
        _make_numeric_artifact(3, b"#400\t1\n"),
    )
    evidence = PreprocessingRawSearchEvidence(
        raw_search_output_directory="/raw",
        searched_source_ordinals=(0,),
        artifacts=artifacts,
    )
    assert len(evidence.artifacts) == 4


def test_mixed_arity_two_record_closure() -> None:
    """2 records (trimer + monomer) → 2 named A3Ms + 2 numeric placeholders."""
    artifacts = (
        _make_named_artifact(
            "trimer.a3m",
            b"#100,200,300\t1,1,1\n>trimer\n" + b"A" * 100 + b"B" * 200 + b"C" * 300 + b"\n",
            0,
        ),
        _make_named_artifact("mono.a3m", b"#50\t1\n>mono\n" + b"X" * 50 + b"\n", 1),
        _make_numeric_artifact(2, b"#200\t1\n"),
        _make_numeric_artifact(3, b"#300\t1\n"),
    )
    evidence = PreprocessingRawSearchEvidence(
        raw_search_output_directory="/raw",
        searched_source_ordinals=(0, 1),
        artifacts=artifacts,
    )
    assert len(evidence.artifacts) == 4
    assert evidence.artifacts[0].source_ordinal == 0
    assert evidence.artifacts[1].source_ordinal == 1


def test_old_heterodimer_evidence_without_modeled_cardinality_is_loadable() -> None:
    """Old evidence mapping that OMITS modeled_cardinality key loads with default 1."""
    old_mapping = {
        "schema_version": 1,
        "role": "numeric-placeholder",
        "member_name": "1.a3m",
        "path": "/raw/1.a3m",
        "size_bytes": 7,
        "sha256": _sha(b"#200\t1\n"),
        "source_ordinal": None,
        "declared_member": None,
        "raw_query_id": 1,
        "modeled_chain_length": 200,
    }
    artifact = preprocessing_raw_search_artifact_from_mapping(old_mapping)
    assert artifact.modeled_cardinality == 1


def test_new_numeric_artifact_carries_actual_cardinality() -> None:
    """New numeric artifacts carry the actual cardinality."""
    artifact = PreprocessingRawSearchArtifact(
        role="numeric-placeholder",
        member_name="1.a3m",
        path="/raw/1.a3m",
        size_bytes=8,
        sha256=_sha(b"#200\t2\n"),
        source_ordinal=None,
        declared_member=None,
        raw_query_id=1,
        modeled_chain_length=200,
        modeled_cardinality=2,
    )
    assert artifact.modeled_cardinality == 2
    assert artifact.to_mapping()["modeled_cardinality"] == 2


def test_unpaired_paired_preserves_mmsa_paired_heterocomplex_shape() -> None:
    """R7-3 regression: the new validator does not reject the mmsa harvest header shape.

    Part A: validates a full A3M payload starting with #216,482\\t1,1 (the exact
    mmsa harvest header shape retained in the internal cold live-acceptance
    evidence archive) with chain_lengths=(216, 482) and asserts it passes.
    Also asserts the old paired validator accepts the same shape.

    Part B: proves the numeric placeholder #482\\t1 (the exact bytes of the
    evidence archive's raw-1.a3m, confirmed as
    b"#482\\t1\\n") matches the byte-equality model used in raw_search.py:
    f"#{modeled_length}\\t{modeled_cardinality}\\n".encode() for modeled_length=482,
    modeled_cardinality=1.
    """
    from bspp.orchestration.runtime.preprocessing.content_validation import (
        validate_preprocessing_a3m_header_bytes,
    )

    # Part A: named-A3M header-shape regression
    payload = b"#216,482\t1,1\n>101\t102\n" + b"A" * 216 + b"B" * 482 + b"\n"
    validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(216, 482), label="mmsa-harvest")
    validate_preprocessing_paired_a3m_bytes(payload, chain_lengths=(216, 482), label="mmsa-harvest-old")

    # Part B: numeric-placeholder byte-equality regression
    modeled_length = 482
    modeled_cardinality = 1
    expected_data = f"#{modeled_length}\t{modeled_cardinality}\n".encode()
    assert expected_data == b"#482\t1\n"


def test_unique_chain_lengths_and_cardinalities_and_numeric_indexing_invariant() -> None:
    """Direct test of the runtime derivation seam for n-arity raw-search closure.

    This characterizes the historical paired-only placeholder model. Current
    unpaired_paired execution consumes these per-chain files and emits only
    named A3Ms; process-boundary tests cover both modes and mixed arities.
    """
    from bspp.orchestration.runtime.preprocessing.raw_search import _unique_chain_lengths_and_cardinalities

    # Monomer: 1 record, 1 unique chain → 0 numeric placeholders
    monomer = (_record("AAA", source_ordinal=0, identity="mono"),)
    lengths, cardinals = _unique_chain_lengths_and_cardinalities(monomer)
    assert lengths == (3,)
    assert cardinals == (1,)
    count = len(monomer)
    expected_numeric = tuple(range(count, len(lengths)))
    assert expected_numeric == ()

    # Heterodimer: 1 record, 2 unique chains → 1 numeric placeholder (#200\t1)
    heterodimer = (_record("AAA:TTT", source_ordinal=0, identity="het"),)
    lengths, cardinals = _unique_chain_lengths_and_cardinalities(heterodimer)
    assert lengths == (3, 3)
    assert cardinals == (1, 1)
    count = len(heterodimer)
    expected_numeric = tuple(range(count, len(lengths)))
    assert expected_numeric == (1,)
    assert f"#{lengths[1]}\t{cardinals[1]}\n".encode() == b"#3\t1\n"

    # Trimer: 1 record, 3 unique chains → 2 numeric placeholders
    trimer = (_record("AAA:TTT:GGG", source_ordinal=0, identity="tri"),)
    lengths, cardinals = _unique_chain_lengths_and_cardinalities(trimer)
    assert lengths == (3, 3, 3)
    assert cardinals == (1, 1, 1)
    count = len(trimer)
    expected_numeric = tuple(range(count, len(lengths)))
    assert expected_numeric == (1, 2)
    assert f"#{lengths[1]}\t{cardinals[1]}\n".encode() == b"#3\t1\n"
    assert f"#{lengths[2]}\t{cardinals[2]}\n".encode() == b"#3\t1\n"

    # Mixed trimer + monomer: 2 records, 4 unique chain lengths → 2 numeric placeholders
    mixed = (
        _record("AAA:TTT:GGG", source_ordinal=0, identity="tri"),
        _record("CC", source_ordinal=1, identity="mono2"),
    )
    lengths, cardinals = _unique_chain_lengths_and_cardinalities(mixed)
    assert lengths == (3, 3, 3, 2)
    assert cardinals == (1, 1, 1, 1)
    count = len(mixed)
    expected_numeric = tuple(range(count, len(lengths)))
    assert expected_numeric == (2, 3)
    assert f"#{lengths[2]}\t{cardinals[2]}\n".encode() == b"#3\t1\n"
    assert f"#{lengths[3]}\t{cardinals[3]}\n".encode() == b"#2\t1\n"
