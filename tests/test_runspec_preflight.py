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

"""Tests for workflow-only RunSpec preflight reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.runspec_preflight import (
    ActiveWorkflowSafetyError,
    build_runspec_preflight_report,
    enforce_active_workflow_safety,
    render_runspec_preflight_report,
    write_runspec_preflight_reports,
)
from tests.runspec_workflow_helpers import (
    load_workflow_runspec,
    load_workflow_runspec_baked,
    workflow_runspec_data,
    write_workflow_runspec,
)


def test_build_runspec_preflight_report_ready(tmp_path: Path) -> None:
    spec = load_workflow_runspec(tmp_path)

    report = build_runspec_preflight_report(spec, hostname="login01.example-cluster")
    rendered = json.loads(render_runspec_preflight_report(report))

    assert report.schema_version == 1
    assert report.ready is True
    assert report.blockers == ()
    assert report.hostname_matches_cluster is True
    assert rendered["dataset"] == "ds1"
    assert rendered["run_id"] == "run1"
    assert rendered["workflow_summary"]["enabled_steps"] == [
        "preflight",
        "recipe",
        "preprocess",
        "slurm",
        "analysis-finalize",
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    ]


def test_hostname_mismatch_is_evidence_only(tmp_path: Path) -> None:
    spec = load_workflow_runspec(tmp_path)

    report = build_runspec_preflight_report(spec, hostname="login01.other-cluster")

    assert report.hostname_matches_cluster is False
    assert report.ready is True
    assert report.blockers == ()


def test_preflight_report_collects_static_and_phase1_blockers(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    storage = data["storage"]
    container = data["container"]
    assert isinstance(storage, dict)
    assert isinstance(container, dict)
    storage["allow_production_prefixes"] = True
    container["mounts"] = []
    spec = load_workflow_runspec(tmp_path, data)

    report = build_runspec_preflight_report(spec, hostname="login01.example-cluster")

    assert report.ready is False
    assert any("BSPP-STATIC-002" in blocker for blocker in report.blockers)
    assert any("BSPP-STATIC-003" in blocker for blocker in report.blockers)
    assert any("BSPP-P1-001" in blocker for blocker in report.blockers)
    assert any("BSPP-P1-006" in blocker for blocker in report.blockers)
    assert any("BSPP-P1-007" in blocker for blocker in report.blockers)


def test_preflight_report_serializes_static_mount_blocker_details(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    container = data["container"]
    assert isinstance(container, dict)
    container["mounts"] = []
    spec = load_workflow_runspec(tmp_path, data)

    report = build_runspec_preflight_report(spec, hostname="login01.example-cluster")
    rendered = json.loads(render_runspec_preflight_report(report))

    assert rendered["ready"] is False
    static_issues = rendered["static_validation"]["issues"]
    assert any(issue["code"] == "BSPP-STATIC-002" for issue in static_issues)
    assert all(isinstance(value, str) for issue in static_issues for value in issue["details"].values())


def test_write_runspec_preflight_reports_under_submission_evidence(tmp_path: Path) -> None:
    spec = load_workflow_runspec(tmp_path)
    assert spec.submission is not None
    report = build_runspec_preflight_report(spec, hostname="login01.example-cluster")

    json_path, text_path = write_runspec_preflight_reports(report, spec.submission.evidence_dir)

    assert json_path == spec.submission.evidence_dir / "preflight" / "preflight_report.json"
    assert text_path == spec.submission.evidence_dir / "preflight" / "preflight_report.txt"
    assert json.loads(json_path.read_text())["ready"] is True
    assert "schema_version: 1" in text_path.read_text()


def test_runspec_preflight_cli_prints_json_and_writes_reports(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    path = write_workflow_runspec(tmp_path, data)

    result = CliRunner().invoke(cli, ["runspec", "preflight", str(path), "--write-report", "--strict"])

    assert result.exit_code == 0
    rendered = json.loads(result.output)
    assert rendered["ready"] is True
    evidence_dir = Path(str(_submission(data)["evidence_dir"]))
    assert (evidence_dir / "preflight" / "preflight_report.json").exists()
    assert (evidence_dir / "preflight" / "preflight_report.txt").exists()


def test_runspec_preflight_cli_strict_exits_nonzero_on_blockers(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    storage = data["storage"]
    assert isinstance(storage, dict)
    storage["allow_production_prefixes"] = True
    path = write_workflow_runspec(tmp_path, data)

    result = CliRunner().invoke(cli, ["runspec", "preflight", str(path), "--strict"])

    assert result.exit_code == 1
    rendered = json.loads(result.output)
    assert rendered["ready"] is False
    assert any("BSPP-P1-001" in blocker for blocker in rendered["blockers"])


def test_run_kind_preflight_uses_rk_prefix_policy_instead_of_legacy_phase1_prefix_blocker(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    data["run_kind"] = "canary"
    storage = data["storage"]
    assert isinstance(storage, dict)
    storage["s3_output_prefix"] = "s3://example-bucket/postprocessed/canary/"
    spec = load_workflow_runspec(tmp_path, data)

    report = build_runspec_preflight_report(spec, hostname="login01.example-cluster")
    rendered = json.loads(render_runspec_preflight_report(report))

    codes = [result["code"] for result in rendered["phase1_policies"]]
    assert report.ready is True
    assert "BSPP-RK-001" not in codes
    assert "BSPP-P1-001" not in codes
    assert "BSPP-P1-002" not in codes
    enforce_active_workflow_safety(spec)


def test_enforce_active_workflow_safety_raises_for_failing_run_kind_policy(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    data["run_kind"] = "production"
    _enable_local_tar(data, tmp_path)
    _set_workflow_step_run(data, "acceptance-tar-payload-parity", False)
    spec = load_workflow_runspec(tmp_path, data)

    with pytest.raises(ActiveWorkflowSafetyError) as exc_info:
        enforce_active_workflow_safety(spec)

    assert [result.code for result in exc_info.value.phase1_policies if not result.ok] == ["BSPP-RK-003"]


def test_runspec_preflight_requires_active_workflow(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    data.pop("workflow")
    data.pop("submission")
    data.pop("acceptance")
    path = write_workflow_runspec(tmp_path, data)

    result = CliRunner().invoke(cli, ["runspec", "preflight", str(path)])

    assert result.exit_code != 0
    assert "requires an active workflow RunSpec" in result.output


def test_native_preflight_command_is_removed() -> None:
    result = CliRunner().invoke(cli, ["runspec", "native-preflight", "--help"])

    assert result.exit_code != 0
    assert "No such command 'native-preflight'" in result.output


def test_enforce_active_workflow_safety_aggregates_static_and_policy_blockers(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    storage = data["storage"]
    container = data["container"]
    assert isinstance(storage, dict)
    assert isinstance(container, dict)
    storage["allow_production_prefixes"] = True
    container["mounts"] = []
    spec = load_workflow_runspec(tmp_path, data)

    with pytest.raises(ActiveWorkflowSafetyError) as exc_info:
        enforce_active_workflow_safety(spec)

    assert any("BSPP-STATIC-002" in blocker for blocker in exc_info.value.static_validation.blockers)
    assert [result.code for result in exc_info.value.phase1_policies if not result.ok] == [
        "BSPP-P1-001",
        "BSPP-P1-006",
        "BSPP-P1-007",
    ]


def test_enforce_active_workflow_safety_accepts_ready_spec(tmp_path: Path) -> None:
    spec = load_workflow_runspec(tmp_path)

    enforce_active_workflow_safety(spec)


def test_preflight_baked_mode_passes_all_checks(tmp_path: Path) -> None:
    spec = load_workflow_runspec_baked(tmp_path)

    report = build_runspec_preflight_report(spec, hostname="login01.example-cluster")
    assert report.ready is True
    assert report.blockers == ()
    enforce_active_workflow_safety(spec)


def _submission(data: dict[str, object]) -> dict[str, object]:
    value = data["submission"]
    assert isinstance(value, dict)
    return value


def _enable_local_tar(data: dict[str, object], tmp_path: Path) -> None:
    storage = data["storage"]
    assert isinstance(storage, dict)
    storage.update(
        {
            "upload_mode": "tar",
            "local_tar_dir": str(tmp_path / "project" / "output" / "local-tars"),
            "local_tar_manifest_csv": str(tmp_path / "project" / "output" / "local-tars.csv"),
        }
    )


def _set_workflow_step_run(data: dict[str, object], step_name: str, run: bool) -> None:
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    steps = workflow["steps"]
    assert isinstance(steps, list)
    for step in steps:
        assert isinstance(step, dict)
        if step["name"] == step_name:
            step["run"] = run
            return
    raise AssertionError(f"missing workflow step {step_name}")
