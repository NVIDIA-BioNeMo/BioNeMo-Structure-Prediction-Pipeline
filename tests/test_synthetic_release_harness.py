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

from pathlib import Path

import pytest

from bspp.orchestration.contract.submission_evidence import (
    submission_expectation_from_bytes,
    submission_result_from_bytes,
)
from tests.support.release_harness import SyntheticReleaseHarness


def test_synthetic_release_accepts_then_authorizes_exactly_one_publication(tmp_path: Path) -> None:
    harness = SyntheticReleaseHarness.create(tmp_path)
    attempt = harness.root / "submissions" / "3-process" / "attempt-1"
    expectation = submission_expectation_from_bytes((attempt / "expectation.json").read_bytes())
    result = submission_result_from_bytes((attempt / "result.json").read_bytes())
    assert result.token == expectation.token

    acceptance = harness.validate()
    assert acceptance.ok
    with pytest.raises(ValueError, match="absent"):
        harness.publish(acceptance.acceptance_sha256)

    harness.approve(acceptance.acceptance_sha256)
    harness.publish(acceptance.acceptance_sha256)
    with pytest.raises(ValueError, match="consumed"):
        harness.publish(acceptance.acceptance_sha256)


def test_synthetic_release_rejects_stale_and_destination_mismatched_approval(tmp_path: Path) -> None:
    harness = SyntheticReleaseHarness.create(tmp_path)
    acceptance = harness.validate()
    harness.approve(acceptance.acceptance_sha256)

    with pytest.raises(ValueError, match="stale"):
        harness.publish("f" * 64)
    with pytest.raises(ValueError, match="stale"):
        harness.publish(acceptance.acceptance_sha256, destination="s3://different/location")


@pytest.mark.parametrize(
    ("mutation", "issue"),
    [
        ("inventory", "candidate inventory mismatch"),
        ("extra_member", "unexpected archive member"),
        ("missing_member", "missing archive member"),
        ("digest_mismatch", "local package digest mismatches"),
        ("semantic", "semantic mismatches"),
        ("terminal_failure", "terminal failures"),
        ("missing_execution", "missing execution results"),
        ("index", "provenance tree verification failed"),
        ("extra_evidence", "provenance tree verification failed"),
        ("symlink_evidence", "provenance tree verification failed"),
        ("upload", "external uploads must be zero"),
    ],
)
def test_synthetic_release_rejects_adversarial_mutations(tmp_path: Path, mutation: str, issue: str) -> None:
    harness = SyntheticReleaseHarness.create(tmp_path)
    harness.mutate(mutation)

    assert any(issue in value for value in harness.validate().issues)
