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

"""Tests for file-defined workflow template expansion."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bspp.orchestration.control.workflows import load_workflow_template


def test_load_workflow_template_reads_relative_file_and_preserves_step_order(tmp_path: Path) -> None:
    plan_path = tmp_path / "run-plan.yaml"
    plan_path.write_text("run_kind: dev\n")
    template_path = tmp_path / "templates" / "archive.yaml"
    template_path.parent.mkdir()
    _write_workflow_template(
        template_path,
        [
            {"name": "preflight", "run": True},
            {"name": "recipe", "run": True},
            {"name": "preprocess", "run": True},
            {"name": "slurm", "run": True, "mode": "submit-and-monitor"},
        ],
    )

    workflow = load_workflow_template(plan_path, "templates/archive.yaml")

    steps = workflow["steps"]
    assert isinstance(steps, list)
    assert [step["name"] for step in steps] == ["preflight", "recipe", "preprocess", "slurm"]


def test_load_workflow_template_is_deterministic(tmp_path: Path) -> None:
    plan_path = tmp_path / "run-plan.yaml"
    plan_path.write_text("run_kind: dev\n")
    template_path = tmp_path / "workflow.yaml"
    _write_workflow_template(template_path, [{"name": "preflight", "run": False}])

    first = load_workflow_template(plan_path, "workflow.yaml")
    second = load_workflow_template(plan_path, "workflow.yaml")

    assert first == second


def test_load_workflow_template_rejects_environment_interpolation(tmp_path: Path) -> None:
    plan_path = tmp_path / "run-plan.yaml"
    plan_path.write_text("run_kind: dev\n")
    _write_workflow_template(
        tmp_path / "workflow.yaml",
        [{"name": "slurm", "run": True, "mode": "monitor-existing", "job_id": "$SLURM_JOB_ID"}],
    )

    with pytest.raises(ValueError, match=r"environment interpolation.*workflow\.steps\.0\.job_id"):
        load_workflow_template(plan_path, "workflow.yaml")


def _write_workflow_template(path: Path, steps: list[dict[str, object]]) -> None:
    path.write_text(yaml.safe_dump({"workflow": {"steps": steps}}, sort_keys=False))
