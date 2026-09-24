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

"""Real ColabFold A3M envelope validation tests."""

from __future__ import annotations

import pytest

from bspp.orchestration.runtime.preprocessing.content_validation import (
    validate_preprocessing_a3m_bytes,
    validate_preprocessing_a3m_header_bytes,
    validate_preprocessing_paired_a3m_bytes,
)


def test_monomer_and_colabfold_multimer_a3ms_are_accepted() -> None:
    validate_preprocessing_a3m_bytes(b">101\nAAAA\n", label="monomer")
    validate_preprocessing_a3m_bytes(
        b"#36,49\t1,1\n>101\t102\nAAAABBBB\n>hit\tpaired\nCCCCDDDD\n",
        label="multimer",
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"#36,49 1,1\n>101\nAAAA\n",
        b"#36,49\t1\n>101\nAAAA\n",
        b"#36,0\t1,1\n>101\nAAAA\n",
        b"#36,49\t1,1\n#36,49\t1,1\n>101\nAAAA\n",
        b"#36,49\t1,1\n",
    ],
)
def test_malformed_colabfold_multimer_metadata_is_rejected(payload: bytes) -> None:
    with pytest.raises(ValueError):
        validate_preprocessing_a3m_bytes(payload, label="malformed")


def _full_a3m(header: bytes, sequence: bytes) -> bytes:
    """Build a full A3M payload with a # header line plus > records and sequence."""
    return header + b"\n" + sequence + b"\n"


def test_validate_a3m_header_bytes_accepts_monomer() -> None:
    payload = b"#414\t1\n>101\n" + b"A" * 414 + b"\n"
    validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(414,), label="monomer")


def test_validate_a3m_header_bytes_accepts_homomer() -> None:
    payload = b"#414\t2\n>101\t102\n" + b"A" * 414 + b"\n"
    validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(414, 414), label="homomer")


def test_validate_a3m_header_bytes_accepts_heteromer() -> None:
    payload = b"#100,200\t1,1\n>101\t102\n" + b"A" * 100 + b"B" * 200 + b"\n"
    validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(100, 200), label="heteromer")


def test_validate_a3m_header_bytes_accepts_trimer() -> None:
    payload = b"#100,200,300\t1,1,1\n>101\t102\t103\n" + b"A" * 100 + b"B" * 200 + b"C" * 300 + b"\n"
    validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(100, 200, 300), label="trimer")


def test_validate_a3m_header_bytes_accepts_a2b_heteromer() -> None:
    payload = b"#100,200\t2,1\n>101\t102\t103\n" + b"A" * 100 + b"A" * 100 + b"B" * 200 + b"\n"
    validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(100, 100, 200), label="a2b")


def test_validate_a3m_header_bytes_accepts_a2b2_interleaved() -> None:
    payload = b"#100,200\t2,2\n>101\t102\n" + b"A" * 100 + b"B" * 200 + b"A" * 100 + b"B" * 200 + b"\n"
    validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(100, 200, 100, 200), label="a2b2")


def test_validate_a3m_header_bytes_accepts_swapped_order_heteromer() -> None:
    """Multiset relaxation: #200,100\t1,1 matches (100,200) order-insensitively."""
    payload = b"#200,100\t1,1\n>101\t102\n" + b"A" * 100 + b"B" * 200 + b"\n"
    validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(100, 200), label="swapped")


def test_validate_a3m_header_bytes_rejects_cardinality_mismatch() -> None:
    payload = b"#100,200\t2,1\n>101\t102\n" + b"A" * 100 + b"B" * 200 + b"\n"
    with pytest.raises(ValueError, match="does not match expected chain lengths"):
        validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(100, 200), label="mismatch")


def test_validate_a3m_header_bytes_rejects_homomer_mismatch() -> None:
    payload = b"#414\t2\n>101\t102\n" + b"A" * 414 + b"\n"
    with pytest.raises(ValueError, match="does not match expected chain lengths"):
        validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(414,), label="homomer-mismatch")


def test_validate_a3m_header_bytes_rejects_wrong_lengths() -> None:
    payload = b"#100,200\t1,1\n>101\t102\n" + b"A" * 100 + b"B" * 200 + b"\n"
    with pytest.raises(ValueError, match="does not match expected chain lengths"):
        validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(100, 100), label="wrong-lengths")


def test_validate_a3m_header_bytes_rejects_unbounded_cardinality_without_expansion() -> None:
    """Artifact-controlled cardinality is rejected before expansion (no MemoryError path)."""
    payload = b"#1\t1000000000\n>101\nA\n"
    with pytest.raises(ValueError, match="does not match expected chain lengths"):
        validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(1,), label="huge-cardinality")


def test_validate_a3m_header_bytes_rejects_cardinality_overflow_mid_header() -> None:
    payload = b"#100,200\t1,2\n>101\t102\n" + b"A" * 100 + b"B" * 200 + b"\n"
    with pytest.raises(ValueError, match="does not match expected chain lengths"):
        validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(100, 200), label="mid-overflow")


def test_validate_a3m_header_bytes_preserves_mmsa_paired_heterocomplex_shape() -> None:
    """R7-3 regression: the new validator does not reject the mmsa harvest header shape #216,482\t1,1."""
    payload = b"#216,482\t1,1\n>101\t102\n" + b"A" * 216 + b"B" * 482 + b"\n"
    validate_preprocessing_a3m_header_bytes(payload, chain_lengths=(216, 482), label="mmsa-harvest")
    # Old paired validator also accepts the same shape
    validate_preprocessing_paired_a3m_bytes(payload, chain_lengths=(216, 482), label="mmsa-harvest-old")
