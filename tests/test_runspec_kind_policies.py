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

"""Tests for Run Kind policy guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest

from bspp.orchestration.contract.runspec_policies import (
    Phase1PolicyResult,
    RunKindPolicyContext,
    evaluate_run_kind_policies,
    policy_results_for,
)
from tests.runspec_workflow_helpers import load_workflow_runspec, workflow_runspec_data


def test_dev_allows_partial_workflow_and_skipped_acceptance(tmp_path: Path) -> None:
    data = _with_run_kind(workflow_runspec_data(tmp_path), "dev")
    _set_acceptance_gates(data, run=False)
    spec = load_workflow_runspec(tmp_path, data)

    evaluation = evaluate_run_kind_policies(spec, RunKindPolicyContext(run_kind="dev"))

    assert _failed_codes(evaluation.results) == []
    assert evaluation.requires_confirmation is False
    assert evaluation.requires_runtime_qualification is False
    assert evaluation.requires_committed_source is False


def test_dev_allows_nonproduction_gcs_destination_with_credentials(tmp_path: Path) -> None:
    data = _with_run_kind(workflow_runspec_data(tmp_path), "dev")
    _storage(data)["gcs_destination_prefix"] = "gs://example-gcs-bucket/users/test/postprocessed/dev-run/"
    secrets = data["secrets"]
    assert isinstance(secrets, dict)
    secrets["gcs_credentials_ref"] = "env:bspp/gcs"
    spec = load_workflow_runspec(tmp_path, data)

    evaluation = evaluate_run_kind_policies(spec, RunKindPolicyContext(run_kind="dev"))

    assert _failed_codes(evaluation.results) == []


def test_canary_nonproduction_prefix_does_not_require_acceptance_gates(tmp_path: Path) -> None:
    data = _with_run_kind(workflow_runspec_data(tmp_path), "canary")
    _set_acceptance_gates(data, run=False)
    spec = load_workflow_runspec(tmp_path, data)

    evaluation = evaluate_run_kind_policies(spec, RunKindPolicyContext(run_kind="canary"))

    assert _failed_codes(evaluation.results) == []
    assert evaluation.requires_confirmation is False
    assert evaluation.requires_runtime_qualification is False


def test_production_local_tar_rejects_missing_acceptance_gates(tmp_path: Path) -> None:
    data = _with_run_kind(workflow_runspec_data(tmp_path), "production")
    _enable_local_tar(data, tmp_path)
    _set_workflow_step_run(data, "acceptance-tar-payload-parity", False)
    spec = load_workflow_runspec(tmp_path, data)

    evaluation = evaluate_run_kind_policies(spec, RunKindPolicyContext(run_kind="production"))

    assert _failed_codes(evaluation.results) == ["BSPP-RK-003"]


def test_production_local_tar_with_all_acceptance_gates_is_allowed(tmp_path: Path) -> None:
    data = _with_run_kind(workflow_runspec_data(tmp_path), "production")
    _enable_local_tar(data, tmp_path)
    spec = load_workflow_runspec(tmp_path, data)

    evaluation = evaluate_run_kind_policies(spec, RunKindPolicyContext(run_kind="production"))

    assert _failed_codes(evaluation.results) == []
    assert evaluation.requires_confirmation is True
    assert evaluation.requires_runtime_qualification is True
    assert evaluation.requires_committed_source is True


def test_production_files_mode_does_not_require_acceptance_gates(tmp_path: Path) -> None:
    data = _with_run_kind(workflow_runspec_data(tmp_path), "production")
    _set_acceptance_gates(data, run=False)
    spec = load_workflow_runspec(tmp_path, data)

    evaluation = evaluate_run_kind_policies(spec, RunKindPolicyContext(run_kind="production"))

    assert _failed_codes(evaluation.results) == []
    assert evaluation.requires_committed_source is True


def test_dev_source_policy_blocks_dirty_without_explicit_opt_in(tmp_path: Path) -> None:
    data = _with_run_kind(workflow_runspec_data(tmp_path), "dev")
    spec = load_workflow_runspec(tmp_path, data)

    dirty_default = evaluate_run_kind_policies(spec, RunKindPolicyContext(run_kind="dev", source_state="dirty"))
    dirty_allowed = evaluate_run_kind_policies(
        spec,
        RunKindPolicyContext(run_kind="dev", source_state="dirty", allow_dirty_source=True),
    )

    assert _failed_codes(dirty_default.results) == ["BSPP-RK-004"]
    assert _failed_codes(dirty_allowed.results) == []


@pytest.mark.parametrize("run_kind", ["canary", "production"])
def test_committed_source_policy_blocks_dirty_for_canary_and_production(tmp_path: Path, run_kind: str) -> None:
    data = _with_run_kind(workflow_runspec_data(tmp_path), run_kind)
    spec = load_workflow_runspec(tmp_path, data)

    dirty = evaluate_run_kind_policies(spec, RunKindPolicyContext(run_kind=run_kind, source_state="dirty"))
    committed = evaluate_run_kind_policies(spec, RunKindPolicyContext(run_kind=run_kind, source_state="committed"))
    unknown = evaluate_run_kind_policies(spec, RunKindPolicyContext(run_kind=run_kind, source_state="unknown"))

    assert _failed_codes(dirty.results) == ["BSPP-RK-004"]
    assert _failed_codes(committed.results) == []
    assert committed.requires_committed_source is True
    assert _failed_codes(unknown.results) == []
    assert unknown.requires_committed_source is True


def test_run_kind_results_preserve_universal_phase1_checks_without_legacy_prefix_checks(tmp_path: Path) -> None:
    data = _with_run_kind(workflow_runspec_data(tmp_path), "canary")
    spec = load_workflow_runspec(tmp_path, data)

    codes = [result.code for result in policy_results_for(spec)]

    assert "BSPP-P1-001" not in codes
    assert "BSPP-P1-002" not in codes
    assert codes[-6:] == [
        "BSPP-P1-003",
        "BSPP-P1-004",
        "BSPP-P1-005",
        "BSPP-P1-006",
        "BSPP-P1-007",
        "BSPP-P1-008",
    ]


def _with_run_kind(data: dict[str, object], run_kind: str) -> dict[str, object]:
    data["run_kind"] = run_kind
    return data


def _storage(data: dict[str, object]) -> dict[str, object]:
    value = data["storage"]
    assert isinstance(value, dict)
    return value


def _workflow_steps(data: dict[str, object]) -> list[dict[str, object]]:
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    steps = workflow["steps"]
    assert isinstance(steps, list)
    assert all(isinstance(step, dict) for step in steps)
    return steps


def _set_workflow_step_run(data: dict[str, object], step_name: str, run: bool) -> None:
    for step in _workflow_steps(data):
        if step["name"] == step_name:
            step["run"] = run
            return
    raise AssertionError(f"missing workflow step {step_name}")


def _set_acceptance_gates(data: dict[str, object], *, run: bool) -> None:
    for step_name in (
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    ):
        _set_workflow_step_run(data, step_name, run)


def _enable_local_tar(data: dict[str, object], tmp_path: Path) -> None:
    _storage(data).update(
        {
            "upload_mode": "tar",
            "local_tar_dir": str(tmp_path / "project" / "output" / "local-tars"),
            "local_tar_manifest_csv": str(tmp_path / "project" / "output" / "local-tars.csv"),
        }
    )


def _failed_codes(results: tuple[Phase1PolicyResult, ...]) -> list[str]:
    return [result.code for result in results if result.blocker and not result.ok]
