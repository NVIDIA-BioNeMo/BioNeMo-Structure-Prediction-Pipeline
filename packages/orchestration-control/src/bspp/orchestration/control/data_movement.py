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

"""Control-side data-movement planning module.

This module owns the **planning** layer for operator-initiated data movement.
It dispatches to the contract-level
:func:`~bspp.orchestration.contract.operator_data_movement.build_plan_referenced_transfer_plan`
and
:func:`~bspp.orchestration.contract.operator_data_movement.build_manual_transfer_plan`
functions. It imports only from ``contract.*`` — never from ``runtime.*`` —
so the control/runtime boundary is preserved.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from bspp.orchestration.contract.operator_data_movement import (
    OperatorTransferPlan,
    build_manual_transfer_plan,
    build_plan_referenced_transfer_plan,
)


def plan_data_movement(
    *,
    phase_plan_path: Path | None = None,
    source: str | None = None,
    destination: str | None = None,
    size_bytes: int | None = None,
    sha256: str | None = None,
    s3_prefix: str | None = None,
    override_prefix: str | None = None,
    dry_run: bool = True,
) -> OperatorTransferPlan:
    """Dispatch to plan-referenced or manual transfer-plan builders.

    In plan-referenced mode (``phase_plan_path`` provided), ``s3_prefix``
    is required unless ``override_prefix`` is given.

    In manual mode (``source``/``destination``/``size_bytes``/``sha256`` provided),
    ``s3_prefix`` is not required and ``override_prefix`` is not allowed.
    """
    if phase_plan_path is not None:
        manual_options = (source, destination, size_bytes, sha256)
        if any(value is not None for value in manual_options):
            raise ValueError(
                "--phase-plan is mutually exclusive with manual-mode options "
                "(--source, --destination, --size-bytes, --sha256)"
            )
        return build_plan_referenced_transfer_plan(
            phase_plan_path=phase_plan_path,
            s3_prefix=s3_prefix,
            override_prefix=override_prefix,
            dry_run=dry_run,
        )
    # Manual mode
    if source is None or destination is None or size_bytes is None or sha256 is None:
        raise ValueError("manual mode requires --source, --destination, --size-bytes, and --sha256")
    return build_manual_transfer_plan(
        source=source,
        destination=destination,
        size_bytes=size_bytes,
        sha256=sha256,
        s3_prefix=s3_prefix,
        override_prefix=override_prefix,
        dry_run=dry_run,
    )


def render_operator_transfer_plan_yaml(plan: OperatorTransferPlan) -> str:
    """Render an OperatorTransferPlan as YAML."""
    return yaml.safe_dump(plan.to_mapping(), sort_keys=True, default_flow_style=False)


def render_operator_transfer_plan_json(plan: OperatorTransferPlan) -> str:
    """Render an OperatorTransferPlan as JSON."""
    return plan.to_json()


__all__ = [
    "plan_data_movement",
    "render_operator_transfer_plan_json",
    "render_operator_transfer_plan_yaml",
]
