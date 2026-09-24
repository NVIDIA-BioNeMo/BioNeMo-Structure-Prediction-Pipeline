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

"""Prediction archive-bundle contract tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from bspp.orchestration.contract.prediction_bundle import (
    MAX_PAIRS_PER_ARCHIVE,
    PredictionArchiveBundle,
    prediction_archive_bundle_from_mapping,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION

_VALID_NAME = "bspp_260903_1234_a00001.tar.lz4"
_VALID_SHA256 = "a" * 64
_REQUIRED_BUNDLE_FIELDS = (
    "bundle_name",
    "member_ids",
    "member_count",
    "sha256",
    "size_bytes",
)


def _bundle(**overrides: object) -> PredictionArchiveBundle:
    fields: dict[str, object] = {
        "bundle_name": _VALID_NAME,
        "member_ids": ("member-a", "member-b"),
        "member_count": 2,
        "sha256": _VALID_SHA256,
        "size_bytes": 100,
        "created_at": None,
    }
    fields.update(overrides)
    return PredictionArchiveBundle(**fields)


def _mapping(**overrides: object) -> dict[str, object]:
    mapping = _bundle().to_mapping()
    mapping.update(overrides)
    return mapping


def test_valid_bundle_name_accepted() -> None:
    assert _bundle().bundle_name == _VALID_NAME


@pytest.mark.parametrize("letter", ["a", "z"])
def test_boundary_lowercase_node_letters_accepted(letter: str) -> None:
    bundle = _bundle(bundle_name=f"bspp_260903_1234_{letter}00001.tar.lz4")

    assert bundle.bundle_name.endswith(f"_{letter}00001.tar.lz4")


def test_batch_digits_with_leading_zeroes_accepted() -> None:
    bundle = _bundle(bundle_name="bspp_260903_1234_a00000.tar.lz4")

    assert bundle.bundle_name == "bspp_260903_1234_a00000.tar.lz4"


@pytest.mark.parametrize(
    "bundle_name",
    [
        "bspp_260903_1234_A00001.tar.lz4",  # uppercase node letter
        "bspp_260903_1234_00001.tar.lz4",  # missing node letter
        "bspp_260903_1234_ab00001.tar.lz4",  # multiple node letters
        "bspp_260903_1234_a0001.tar.lz4",  # four batch digits
        "bspp_260903_1234_a000001.tar.lz4",  # six batch digits
        "bspp_260903_1234_a00001.tar.gz",  # incorrect extension
        "bspp_260903_1234_a00001.tar.lz4x",  # extra trailing characters
    ],
)
def test_invalid_bundle_names_rejected(bundle_name: str) -> None:
    with pytest.raises(ValueError, match="bundle_name"):
        _bundle(bundle_name=bundle_name)


def test_empty_member_tuple_rejected() -> None:
    with pytest.raises(ValueError, match="member_ids"):
        _bundle(member_ids=(), member_count=0)


def test_empty_member_string_rejected() -> None:
    with pytest.raises(ValueError, match="member_ids"):
        _bundle(member_ids=("member-a", ""))


def test_non_string_member_rejected_through_mapping() -> None:
    with pytest.raises(ValueError, match="member_ids"):
        prediction_archive_bundle_from_mapping(_mapping(member_ids=["member-a", 42]))


def test_duplicate_member_ids_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        _bundle(member_ids=("member-a", "member-a"))


def test_mismatched_member_count_rejected() -> None:
    with pytest.raises(ValueError, match="member_count"):
        _bundle(member_count=3)


def test_5001_member_bundle_accepted() -> None:
    member_ids = tuple(f"member-{index}" for index in range(MAX_PAIRS_PER_ARCHIVE + 1))

    bundle = _bundle(member_ids=member_ids, member_count=len(member_ids))

    assert bundle.member_count == MAX_PAIRS_PER_ARCHIVE + 1
    assert len(bundle.member_ids) == MAX_PAIRS_PER_ARCHIVE + 1


def test_arbitrary_non_empty_member_syntax_accepted() -> None:
    bundle = _bundle(member_ids=("not-a-real-identity-###", "AFDB_AF-0000000000000000"))

    assert len(bundle.member_ids) == 2


def test_valid_lowercase_sha256_accepted() -> None:
    assert _bundle(sha256="0123456789abcdef" * 4).sha256 == "0123456789abcdef" * 4


@pytest.mark.parametrize(
    "sha256",
    [
        "A" * 64,  # uppercase hex
        "a" * 63,  # short
        "a" * 65,  # long
        "g" * 64,  # non-hex
    ],
)
def test_invalid_sha256_rejected(sha256: str) -> None:
    with pytest.raises(ValueError, match="sha256"):
        _bundle(sha256=sha256)


@pytest.mark.parametrize("size_bytes", [0, -1])
def test_non_positive_size_rejected(size_bytes: int) -> None:
    with pytest.raises(ValueError, match="size_bytes"):
        _bundle(size_bytes=size_bytes)


def test_boolean_size_rejected() -> None:
    with pytest.raises(ValueError, match="size_bytes"):
        _bundle(size_bytes=True)


def test_null_created_at_accepted() -> None:
    assert _bundle(created_at=None).created_at is None


def test_valid_utc_created_at_accepted() -> None:
    bundle = _bundle(created_at="2026-09-03T12:34:56Z")

    assert bundle.created_at == "2026-09-03T12:34:56Z"


@pytest.mark.parametrize(
    "created_at",
    [
        "2026-09-03T12:34:56+05:00",  # non-UTC offset
        "2026-09-03T12:34:56",  # absent Z suffix
        "not-a-timestamp",  # malformed
    ],
)
def test_invalid_created_at_rejected(created_at: str) -> None:
    with pytest.raises(ValueError, match="created_at"):
        _bundle(created_at=created_at)


def test_to_mapping_includes_explicit_schema_version() -> None:
    mapping = _bundle().to_mapping()

    assert mapping["schema_version"] == CURRENT_CONTRACT_SCHEMA_VERSION
    assert set(mapping) == {
        "schema_version",
        "bundle_name",
        "member_ids",
        "member_count",
        "sha256",
        "size_bytes",
        "created_at",
    }


def test_round_trip_preserves_equality_and_tuple_immutability() -> None:
    bundle = _bundle(created_at="2026-09-03T12:34:56Z")

    parsed = prediction_archive_bundle_from_mapping(bundle.to_mapping())

    assert parsed == bundle
    assert isinstance(parsed.member_ids, tuple)
    assert parsed.member_ids == ("member-a", "member-b")


def test_missing_schema_version_rejected() -> None:
    mapping = _mapping()
    del mapping["schema_version"]

    with pytest.raises(ValueError, match="schema_version"):
        prediction_archive_bundle_from_mapping(mapping)


def test_explicit_null_schema_version_rejected() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        prediction_archive_bundle_from_mapping(_mapping(schema_version=None))


def test_unsupported_schema_version_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported PredictionArchiveBundle schema_version 2"):
        prediction_archive_bundle_from_mapping(_mapping(schema_version=2))


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown PredictionArchiveBundle field"):
        prediction_archive_bundle_from_mapping(_mapping(extra=True))


def test_missing_required_field_rejected() -> None:
    mapping = _mapping()
    del mapping["sha256"]

    with pytest.raises(ValueError, match="sha256"):
        prediction_archive_bundle_from_mapping(mapping)


@pytest.mark.parametrize("field", _REQUIRED_BUNDLE_FIELDS)
def test_missing_required_field_reports_missing_message(field: str) -> None:
    mapping = _mapping()
    del mapping[field]

    with pytest.raises(ValueError, match="Missing PredictionArchiveBundle field"):
        prediction_archive_bundle_from_mapping(mapping)


def test_missing_multiple_required_fields_report_sorted_message() -> None:
    mapping = _mapping()
    del mapping["sha256"]
    del mapping["size_bytes"]

    with pytest.raises(ValueError, match=r"Missing PredictionArchiveBundle field\(s\): sha256, size_bytes"):
        prediction_archive_bundle_from_mapping(mapping)


def test_created_at_null_loads_through_mapping() -> None:
    parsed = prediction_archive_bundle_from_mapping(_mapping(created_at=None))

    assert parsed.created_at is None


def test_created_at_omitted_loads_through_mapping() -> None:
    mapping = _mapping()
    del mapping["created_at"]

    parsed = prediction_archive_bundle_from_mapping(mapping)

    assert parsed.created_at is None


def test_wrong_member_container_type_rejected() -> None:
    with pytest.raises(ValueError, match="member_ids"):
        prediction_archive_bundle_from_mapping(_mapping(member_ids="member-a"))


def test_wrong_member_element_type_rejected() -> None:
    with pytest.raises(ValueError, match="member_ids"):
        prediction_archive_bundle_from_mapping(_mapping(member_ids=["member-a", None]))


def test_boolean_member_count_rejected_through_mapping() -> None:
    with pytest.raises(ValueError, match="member_count"):
        prediction_archive_bundle_from_mapping(_mapping(member_count=True))


def test_round_trip_does_not_mutate_source_mapping() -> None:
    mapping = _mapping()
    snapshot = deepcopy(mapping)

    prediction_archive_bundle_from_mapping(mapping)

    assert mapping == snapshot
