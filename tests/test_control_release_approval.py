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

import json
from pathlib import Path

from click.testing import CliRunner

from bspp.orchestration.contract.release_acceptance import publication_approval_from_mapping
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.release_approval import execute_approved_publication
from tests.support.release_harness import SyntheticReleaseHarness

RUNSPEC_SHA256 = "a" * 64
DESTINATION = "s3://release/location"


def _write_acceptance(path: Path, harness: SyntheticReleaseHarness) -> None:
    path.write_text(json.dumps(harness.evidence.to_mapping(), sort_keys=True, separators=(",", ":")) + "\n")


def test_release_approve_publication_revalidates_and_creates_exact_approval(tmp_path: Path) -> None:
    harness = SyntheticReleaseHarness.create(tmp_path)
    acceptance_path = tmp_path / "acceptance.json"
    approval_path = tmp_path / "approval.json"
    _write_acceptance(acceptance_path, harness)

    result = CliRunner().invoke(
        cli,
        [
            "release",
            "approve-publication",
            str(acceptance_path),
            "--evidence-root",
            str(harness.root),
            "--expected-runspec-sha256",
            RUNSPEC_SHA256,
            "--destination",
            DESTINATION,
            "--approval",
            str(approval_path),
        ],
    )

    assert result.exit_code == 0, result.output
    approval = publication_approval_from_mapping(json.loads(approval_path.read_text()))
    assert approval.acceptance_sha256 == harness.validate().acceptance_sha256
    assert approval.destination == DESTINATION


def test_release_approve_publication_rejects_failed_acceptance_without_approval(tmp_path: Path) -> None:
    harness = SyntheticReleaseHarness.create(tmp_path)
    harness.mutate("upload")
    acceptance_path = tmp_path / "acceptance.json"
    approval_path = tmp_path / "approval.json"
    _write_acceptance(acceptance_path, harness)

    result = CliRunner().invoke(
        cli,
        [
            "release",
            "approve-publication",
            str(acceptance_path),
            "--evidence-root",
            str(harness.root),
            "--expected-runspec-sha256",
            RUNSPEC_SHA256,
            "--destination",
            DESTINATION,
            "--approval",
            str(approval_path),
        ],
    )

    assert result.exit_code != 0
    assert "external uploads must be zero" in result.output
    assert not approval_path.exists()


def test_publication_adapter_consumes_approval_before_external_write(tmp_path: Path) -> None:
    harness = SyntheticReleaseHarness.create(tmp_path)
    acceptance_path = tmp_path / "acceptance.json"
    approval_path = tmp_path / "approval.json"
    _write_acceptance(acceptance_path, harness)
    result = CliRunner().invoke(
        cli,
        [
            "release",
            "approve-publication",
            str(acceptance_path),
            "--evidence-root",
            str(harness.root),
            "--expected-runspec-sha256",
            RUNSPEC_SHA256,
            "--destination",
            DESTINATION,
            "--approval",
            str(approval_path),
        ],
    )
    assert result.exit_code == 0, result.output
    calls: list[str] = []

    execute_approved_publication(
        approval_path,
        acceptance_sha256=harness.validate().acceptance_sha256,
        destination=DESTINATION,
        publish=lambda destination: calls.append(destination),
    )
    assert calls == [DESTINATION]

    try:
        execute_approved_publication(
            approval_path,
            acceptance_sha256=harness.validate().acceptance_sha256,
            destination=DESTINATION,
            publish=lambda destination: calls.append(destination),
        )
    except ValueError as error:
        assert "consumed" in str(error)
    else:
        raise AssertionError("consumed publication approval was reused")
    assert calls == [DESTINATION]
