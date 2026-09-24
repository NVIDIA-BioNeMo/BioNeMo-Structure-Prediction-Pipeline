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

"""Workflow template loading for Run Plan expansion."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from bspp.orchestration.contract.runplan import reject_environment_interpolation
from bspp.orchestration.contract.runspec import WorkflowSpec


def load_workflow_template(run_plan_path: Path, workflow_template: str) -> dict[str, object]:
    """Load a workflow template path relative to the Run Plan file."""
    template_path = Path(workflow_template)
    if template_path.is_absolute():
        msg = "workflow_template must be a file path relative to the Run Plan file"
        raise ValueError(msg)
    data = yaml.safe_load((run_plan_path.parent / template_path).read_bytes())
    if not isinstance(data, Mapping):
        msg = f"Expected workflow template YAML mapping in {run_plan_path.parent / template_path}"
        raise TypeError(msg)
    reject_environment_interpolation(data, context="workflow template")
    if set(data) != {"workflow"}:
        msg = "Workflow templates must contain only top-level workflow"
        raise ValueError(msg)
    workflow = data["workflow"]
    if not isinstance(workflow, Mapping):
        msg = "Workflow templates require workflow mapping"
        raise TypeError(msg)
    if set(workflow) != {"steps"}:
        msg = "Workflow templates own workflow.steps only"
        raise ValueError(msg)
    return WorkflowSpec.model_validate(workflow).model_dump(mode="json", exclude_none=True)


__all__ = ["load_workflow_template"]
