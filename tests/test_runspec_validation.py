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

"""Tests for static active workflow RunSpec validation."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from bspp.orchestration.contract.runspec import load_runspec
from bspp.orchestration.contract.runspec_validation import validate_active_workflow_static
from bspp.orchestration.runtime.cli import cli
from tests.runspec_workflow_helpers import (
    load_workflow_runspec,
    load_workflow_runspec_baked,
    workflow_runspec_data,
    write_workflow_runspec,
)


def test_static_validation_accepts_ready_active_workflow(tmp_path: Path) -> None:
    spec = load_workflow_runspec(tmp_path)

    result = validate_active_workflow_static(spec)

    assert result.ok is True
    assert result.issues == ()


def test_static_validation_is_noop_for_non_active_runspec(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    data.pop("workflow")
    data.pop("submission")
    data.pop("acceptance")
    spec = load_runspec(write_workflow_runspec(tmp_path, data))

    result = validate_active_workflow_static(spec)

    assert result.ok is True
    assert result.issues == ()


def test_static_validation_accepts_baked_mode_without_toolkit_path_or_mount(tmp_path: Path) -> None:
    spec = load_workflow_runspec_baked(tmp_path)

    result = validate_active_workflow_static(spec)

    assert result.ok is True
    assert not any(issue.code == "BSPP-STATIC-003" for issue in result.issues)


def test_static_validation_baked_mode_does_not_issue_static003(tmp_path: Path) -> None:
    spec = load_workflow_runspec_baked(tmp_path)

    result = validate_active_workflow_static(spec)

    codes = {issue.code for issue in result.issues}
    assert "BSPP-STATIC-003" not in codes


def test_static_validation_rejects_out_of_order_present_steps(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    workflow["steps"] = [
        {"name": "preflight", "run": True},
        {"name": "analysis-finalize", "run": True, "mode": "submit-and-monitor"},
        {"name": "slurm", "run": True, "mode": "submit-and-monitor"},
    ]
    spec = load_workflow_runspec(tmp_path, data)

    result = validate_active_workflow_static(spec)

    assert result.ok is False
    assert [issue.code for issue in result.issues] == ["BSPP-STATIC-001"]
    assert "slurm" in result.blockers[0]


def test_static_validation_rejects_missing_orchestration_source_to_target_mount(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    container = data["container"]
    paths = data["paths"]
    assert isinstance(container, dict)
    assert isinstance(paths, dict)
    container["mounts"] = [
        {"source": str(tmp_path / "wrong-orchestration"), "target": "/workspace/bspp-orchestration"},
        {"source": paths["afdb_toolkit_repo"], "target": "/workspace/AFDB-Integration-Kit"},
    ]
    spec = load_workflow_runspec(tmp_path, data)

    result = validate_active_workflow_static(spec)

    assert result.ok is False
    assert any(issue.code == "BSPP-STATIC-002" for issue in result.issues)


def test_static_validation_rejects_missing_toolkit_source_to_target_mount(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    container = data["container"]
    paths = data["paths"]
    assert isinstance(container, dict)
    assert isinstance(paths, dict)
    container["mounts"] = [
        {"source": paths["orchestration_repo"], "target": "/workspace/bspp-orchestration"},
        {"source": str(tmp_path / "wrong-toolkit"), "target": "/workspace/AFDB-Integration-Kit"},
    ]
    spec = load_workflow_runspec(tmp_path, data)

    result = validate_active_workflow_static(spec)

    assert result.ok is False
    assert any(issue.code == "BSPP-STATIC-003" for issue in result.issues)


def test_static_validation_rejects_tar_parity_without_acceptance_config(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    data.pop("acceptance")
    _set_workflow_steps(
        data,
        [
            {"name": "preflight", "run": False},
            {"name": "acceptance-tar-payload-parity", "run": True, "mode": "submit-and-monitor"},
        ],
    )
    spec = load_workflow_runspec(tmp_path, data)

    result = validate_active_workflow_static(spec)

    assert result.ok is False
    assert [issue.code for issue in result.issues] == ["BSPP-STATIC-004"]
    assert "acceptance.baseline_output_dir" in result.issues[0].details["missing"]


def test_static_validation_rejects_semantic_without_resource(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    resources = data["resources"]
    assert isinstance(resources, dict)
    resources.pop("acceptance_semantic")
    _set_workflow_steps(
        data,
        [
            {"name": "preflight", "run": False},
            {"name": "acceptance-semantic", "run": True, "mode": "submit-and-monitor"},
        ],
    )
    spec = load_workflow_runspec(tmp_path, data)

    result = validate_active_workflow_static(spec)

    assert result.ok is False
    assert [issue.code for issue in result.issues] == ["BSPP-STATIC-005"]
    assert "resources.acceptance_semantic" in result.issues[0].details["missing"]


def test_static_validation_rejects_verify_evidence_without_enabled_comparator(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    _set_workflow_steps(
        data,
        [
            {"name": "preflight", "run": False},
            {"name": "acceptance-tar-payload-parity", "run": False, "mode": "submit-and-monitor"},
            {"name": "acceptance-verify-evidence", "run": True},
        ],
    )
    spec = load_workflow_runspec(tmp_path, data)

    result = validate_active_workflow_static(spec)

    assert result.ok is False
    assert [issue.code for issue in result.issues] == ["BSPP-STATIC-006"]


def test_runspec_validate_cli_rejects_static_validation_failure(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    container = data["container"]
    assert isinstance(container, dict)
    container["mounts"] = []
    path = write_workflow_runspec(tmp_path, data)

    result = CliRunner().invoke(cli, ["runspec", "validate", str(path)])

    assert result.exit_code != 0
    assert "BSPP-STATIC-002" in result.output
    assert "BSPP-STATIC-003" in result.output


def test_runspec_validate_cli_accepts_static_valid_active_workflow(tmp_path: Path) -> None:
    path = write_workflow_runspec(tmp_path, workflow_runspec_data(tmp_path))

    result = CliRunner().invoke(cli, ["runspec", "validate", str(path)])

    assert result.exit_code == 0
    assert "RunSpec dry run" in result.output


def _set_workflow_steps(data: dict[str, object], steps: list[dict[str, object]]) -> None:
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    workflow["steps"] = steps
