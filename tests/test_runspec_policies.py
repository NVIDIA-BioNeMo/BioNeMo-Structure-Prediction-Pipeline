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

"""Tests for removable Phase 1 RunSpec policy guardrails."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from bspp.orchestration.contract.runspec_policies import (
    Phase1PolicyError,
    enforce_phase1_policies,
    evaluate_phase1_policies,
    parse_slurm_time_seconds,
    policy_results_for,
)
from tests.runspec_workflow_helpers import (
    load_workflow_runspec,
    load_workflow_runspec_baked,
    workflow_runspec_data,
)

RunSpecData = dict[str, object]
PolicyMutator = Callable[[RunSpecData, Path], None]


def test_phase1_policies_allow_ready_workflow_spec(tmp_path: Path) -> None:
    spec = load_workflow_runspec(tmp_path)

    results = evaluate_phase1_policies(spec)

    assert [result.code for result in results] == [
        "BSPP-P1-001",
        "BSPP-P1-002",
        "BSPP-P1-003",
        "BSPP-P1-004",
        "BSPP-P1-005",
        "BSPP-P1-006",
        "BSPP-P1-007",
        "BSPP-P1-008",
    ]
    assert all(result.ok for result in results)


def test_policy_results_for_legacy_spec_matches_phase1_results(tmp_path: Path) -> None:
    spec = load_workflow_runspec(tmp_path)

    assert policy_results_for(spec) == evaluate_phase1_policies(spec)


@pytest.mark.parametrize(
    ("code", "mutate"),
    [
        ("BSPP-P1-001", lambda data, _tmp_path: _storage(data).update({"allow_production_prefixes": True})),
        (
            "BSPP-P1-002",
            lambda data, _tmp_path: (
                _storage(data).update({"gcs_destination_prefix": "gs://isolated-test/postprocessed/run1/"}),
                _secrets(data).update({"gcs_credentials_ref": "env:bspp/gcs"}),
            ),
        ),
        ("BSPP-P1-003", lambda data, tmp_path: _enable_local_tar_self_upload(data, tmp_path)),
        ("BSPP-P1-004", lambda data, _tmp_path: _container(data).update({"image": "relative-image.sqsh"})),
        ("BSPP-P1-005", lambda data, tmp_path: _container(data).update({"image": str(tmp_path / "image.sif")})),
        ("BSPP-P1-006", lambda data, _tmp_path: _remove_mount_target(data, "/workspace/bspp-orchestration")),
        ("BSPP-P1-007", lambda data, _tmp_path: _remove_mount_target(data, "/workspace/AFDB-Integration-Kit")),
        (
            "BSPP-P1-008",
            lambda data, _tmp_path: _resource(data, "acceptance_tar_payload_parity").update({"time": "00:31:00"}),
        ),
    ],
)
def test_phase1_policy_blocks_by_stable_code(
    tmp_path: Path,
    code: str,
    mutate: PolicyMutator,
) -> None:
    data = workflow_runspec_data(tmp_path)
    mutate(data, tmp_path)
    spec = load_workflow_runspec(tmp_path, data)

    blocked_codes = [result.code for result in evaluate_phase1_policies(spec) if not result.ok]

    assert blocked_codes == [code]


@pytest.mark.parametrize(
    "code",
    [
        "BSPP-P1-001",
        "BSPP-P1-002",
        "BSPP-P1-003",
        "BSPP-P1-004",
        "BSPP-P1-005",
        "BSPP-P1-006",
        "BSPP-P1-007",
        "BSPP-P1-008",
    ],
)
def test_phase1_policy_allows_ready_spec_by_stable_code(tmp_path: Path, code: str) -> None:
    spec = load_workflow_runspec(tmp_path)

    result = {item.code: item for item in evaluate_phase1_policies(spec)}[code]

    assert result.ok is True


def test_phase1_policy_allows_absent_parity_resource(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    resources = data["resources"]
    assert isinstance(resources, dict)
    resources.pop("acceptance_tar_payload_parity")
    spec = load_workflow_runspec(tmp_path, data)

    result = {item.code: item for item in evaluate_phase1_policies(spec)}["BSPP-P1-008"]

    assert result.ok is True
    assert result.details["resource_present"] is False


def test_phase1_policy_p1_007_passes_in_baked_mode(tmp_path: Path) -> None:
    spec = load_workflow_runspec_baked(tmp_path)

    result = {item.code: item for item in evaluate_phase1_policies(spec)}["BSPP-P1-007"]

    assert result.ok is True
    assert result.details["baked_mode"] is True
    assert "image-baked" in result.message


def test_phase1_policy_blocks_invalid_parity_time(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    _resource(data, "acceptance_tar_payload_parity")["time"] = "30:00"
    spec = load_workflow_runspec(tmp_path, data)

    result = {item.code: item for item in evaluate_phase1_policies(spec)}["BSPP-P1-008"]

    assert result.ok is False
    assert "invalid" in result.message


def test_enforce_phase1_policies_aggregates_blocked_codes(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    _storage(data).update(
        {
            "allow_production_prefixes": True,
            "gcs_destination_prefix": "gs://isolated-test/postprocessed/run1/",
        }
    )
    _secrets(data)["gcs_credentials_ref"] = "env:bspp/gcs"
    spec = load_workflow_runspec(tmp_path, data)

    with pytest.raises(Phase1PolicyError) as exc_info:
        enforce_phase1_policies(spec)

    assert [failure.code for failure in exc_info.value.failures] == ["BSPP-P1-001", "BSPP-P1-002"]
    assert "BSPP-P1-001, BSPP-P1-002" in str(exc_info.value)


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("00:30:00", 1800),
        ("1-00:00:00", 86400),
        ("24:00:00", 86400),
    ],
)
def test_parse_slurm_time_seconds(value: str, seconds: int) -> None:
    assert parse_slurm_time_seconds(value) == seconds


def _storage(data: RunSpecData) -> dict[str, object]:
    value = data["storage"]
    assert isinstance(value, dict)
    return value


def _secrets(data: RunSpecData) -> dict[str, object]:
    value = data["secrets"]
    assert isinstance(value, dict)
    return value


def _container(data: RunSpecData) -> dict[str, object]:
    value = data["container"]
    assert isinstance(value, dict)
    return value


def _resource(data: RunSpecData, name: str) -> dict[str, object]:
    resources = data["resources"]
    assert isinstance(resources, dict)
    resource = resources[name]
    assert isinstance(resource, dict)
    return resource


def _enable_local_tar_self_upload(data: RunSpecData, tmp_path: Path) -> None:
    _storage(data).update(
        {
            "upload_mode": "tar",
            "local_tar_dir": str(tmp_path / "project" / "output" / "local-tars"),
            "local_tar_manifest_csv": str(tmp_path / "project" / "output" / "local-tars.csv"),
        }
    )
    worker = data["worker"]
    assert isinstance(worker, dict)
    worker["self_upload"] = True


def _remove_mount_target(data: RunSpecData, target: str) -> None:
    container = _container(data)
    mounts = container["mounts"]
    assert isinstance(mounts, list)
    container["mounts"] = [mount for mount in mounts if isinstance(mount, dict) and mount.get("target") != target]
