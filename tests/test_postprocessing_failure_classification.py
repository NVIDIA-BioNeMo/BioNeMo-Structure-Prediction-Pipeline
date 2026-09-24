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

"""Tests for the contract-owned pure group-A/group-B failure classifier."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from bspp.orchestration.contract.postprocessing_failure_classification import (
    PostprocessingFailureClassification,
    PostprocessingTransportFailureObservation,
    classify_transport_failure,
    postprocessing_failure_classification_from_mapping,
    postprocessing_transport_failure_observation_from_mapping,
)
from bspp.orchestration.contract.versioning import UnsupportedSchemaVersionError


@pytest.mark.parametrize(
    ("failure_kind", "expected_group"),
    [
        ("timeout", "A"),
        ("connection-reset", "A"),
        ("temporary-dns", "A"),
        ("http-500", "A"),
        ("http-502", "A"),
        ("http-503", "A"),
        ("http-504", "A"),
        ("other", "B"),
    ],
)
def test_classify_transport_failure_maps_audited_outcomes(failure_kind: str, expected_group: str) -> None:
    observation = PostprocessingTransportFailureObservation(failure_kind=failure_kind, failure_detail="x")

    result = classify_transport_failure(observation)

    assert result.group == expected_group
    assert result.failure_kind == failure_kind
    if expected_group == "A":
        assert result.reason == f"audited {failure_kind}"
    else:
        assert result.reason == "not a group-A transport outcome"


@pytest.mark.parametrize(
    "failure_kind",
    [
        "timeout",
        "connection-reset",
        "temporary-dns",
        "http-500",
        "http-502",
        "http-503",
        "http-504",
        "other",
    ],
)
def test_observation_round_trip(failure_kind: str) -> None:
    observation = PostprocessingTransportFailureObservation(failure_kind=failure_kind, failure_detail="socket timeout")

    loaded = postprocessing_transport_failure_observation_from_mapping(observation.to_mapping())

    assert loaded == observation


@pytest.mark.parametrize(
    "failure_kind",
    [
        "timeout",
        "connection-reset",
        "temporary-dns",
        "http-500",
        "http-502",
        "http-503",
        "http-504",
        "other",
    ],
)
def test_classification_round_trip(failure_kind: str) -> None:
    observation = PostprocessingTransportFailureObservation(failure_kind=failure_kind, failure_detail="x")
    result = classify_transport_failure(observation)

    loaded = postprocessing_failure_classification_from_mapping(result.to_mapping())

    assert loaded == result


def test_observation_loader_rejects_unknown_field() -> None:
    payload = {"schema_version": 1, "failure_kind": "timeout", "failure_detail": "x", "extra": "nope"}

    with pytest.raises(ValueError, match="unknown fields"):
        postprocessing_transport_failure_observation_from_mapping(payload)


def test_classification_loader_rejects_unknown_field() -> None:
    payload = {
        "schema_version": 1,
        "group": "A",
        "failure_kind": "timeout",
        "reason": "audited timeout",
        "extra": "nope",
    }

    with pytest.raises(ValueError, match="unknown fields"):
        postprocessing_failure_classification_from_mapping(payload)


def test_observation_loader_rejects_invalid_failure_kind() -> None:
    payload = {"schema_version": 1, "failure_kind": "oom", "failure_detail": "x"}

    with pytest.raises(ValueError, match="failure_kind"):
        postprocessing_transport_failure_observation_from_mapping(payload)


def test_direct_construction_rejects_invalid_failure_kind() -> None:
    with pytest.raises(ValueError, match="failure_kind"):
        PostprocessingTransportFailureObservation(failure_kind="oom", failure_detail="x")


def test_classification_loader_rejects_invalid_group() -> None:
    payload = {"schema_version": 1, "group": "C", "failure_kind": "timeout", "reason": "x"}

    with pytest.raises(ValueError, match="group"):
        postprocessing_failure_classification_from_mapping(payload)


def test_direct_construction_rejects_invalid_group() -> None:
    with pytest.raises(ValueError, match="group"):
        PostprocessingFailureClassification(group="C", failure_kind="timeout", reason="x")


@pytest.mark.parametrize("blank", ["", "  "])
def test_observation_rejects_blank_failure_detail(blank: str) -> None:
    with pytest.raises(ValueError, match="failure_detail"):
        PostprocessingTransportFailureObservation(failure_kind="timeout", failure_detail=blank)


@pytest.mark.parametrize("blank", ["", "  "])
def test_classification_rejects_blank_reason(blank: str) -> None:
    with pytest.raises(ValueError, match="reason"):
        PostprocessingFailureClassification(group="A", failure_kind="timeout", reason=blank)


def test_observation_loader_rejects_unsupported_schema_version() -> None:
    payload = {"schema_version": 2, "failure_kind": "timeout", "failure_detail": "x"}

    with pytest.raises(UnsupportedSchemaVersionError):
        postprocessing_transport_failure_observation_from_mapping(payload)


def test_classification_loader_rejects_unsupported_schema_version() -> None:
    payload = {"schema_version": 2, "group": "A", "failure_kind": "timeout", "reason": "x"}

    with pytest.raises(UnsupportedSchemaVersionError):
        postprocessing_failure_classification_from_mapping(payload)


def test_observation_is_immutable() -> None:
    observation = PostprocessingTransportFailureObservation(failure_kind="timeout", failure_detail="x")

    with pytest.raises(FrozenInstanceError):
        observation.failure_kind = "other"  # type: ignore[misc]


def test_classification_is_immutable() -> None:
    classification = PostprocessingFailureClassification(group="A", failure_kind="timeout", reason="x")

    with pytest.raises(FrozenInstanceError):
        classification.group = "B"  # type: ignore[misc]


def test_classify_transport_failure_is_deterministic() -> None:
    observation = PostprocessingTransportFailureObservation(failure_kind="http-503", failure_detail="x")

    first = classify_transport_failure(observation)
    second = classify_transport_failure(observation)

    assert first == second
