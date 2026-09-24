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

"""Shared Control State contract records and parsing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from bspp.orchestration.contract.versioning import validate_schema_version

ControlStatus = Literal["materialized", "running", "completed", "failed", "blocked"]
StepStateStatus = Literal["pending", "running", "completed", "skipped", "failed", "blocked"]
TerminalStepStateStatus = Literal["completed", "skipped", "failed", "blocked"]
EventType = Literal["materialized", "step-submitted", "step-finished"]


@dataclass(frozen=True)
class ControlStepState:
    """Current state for one workflow step."""

    index: int
    name: str
    run: bool
    status: StepStateStatus
    job_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    last_observed_status: str | None = None


@dataclass(frozen=True)
class ControlState:
    """Fast current status for one materialized RunSpec."""

    schema_version: int
    status: ControlStatus
    dataset: str
    run_id: str
    cluster: str
    evidence_dir: Path
    source_runspec: Path | None
    source_hash: str | None
    materialized_at: str
    updated_at: str
    steps: tuple[ControlStepState, ...]

    def to_redacted_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready state."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "dataset": self.dataset,
            "run_id": self.run_id,
            "cluster": self.cluster,
            "evidence_dir": str(self.evidence_dir),
            "source_runspec": str(self.source_runspec) if self.source_runspec is not None else None,
            "source_hash": self.source_hash,
            "materialized_at": self.materialized_at,
            "updated_at": self.updated_at,
            "steps": [asdict(step) for step in self.steps],
        }


@dataclass(frozen=True)
class ControlStateValidation:
    """Validation result comparing current state with event reconstruction."""

    ok: bool
    issues: tuple[str, ...] = ()


def control_state_from_payload(payload: dict[str, object]) -> ControlState:
    """Parse and validate a Control State JSON object."""
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="ControlState")
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        msg = "Control State payload requires steps"
        raise ValueError(msg)
    return ControlState(
        schema_version=schema_version,
        status=_control_status(_required_str(payload.get("status"), "status")),
        dataset=_required_str(payload.get("dataset"), "dataset"),
        run_id=_required_str(payload.get("run_id"), "run_id"),
        cluster=_required_str(payload.get("cluster"), "cluster"),
        evidence_dir=Path(_required_str(payload.get("evidence_dir"), "evidence_dir")),
        source_runspec=_optional_path(payload.get("source_runspec")),
        source_hash=_optional_str(payload.get("source_hash")),
        materialized_at=_required_str(payload.get("materialized_at"), "materialized_at"),
        updated_at=_required_str(payload.get("updated_at"), "updated_at"),
        steps=tuple(control_step_state_from_payload(step) for step in raw_steps),
    )


def control_step_state_from_payload(payload: object) -> ControlStepState:
    """Parse and validate one Control State step object."""
    if not isinstance(payload, dict):
        msg = "Control State step payload must be an object"
        raise ValueError(msg)
    return ControlStepState(
        index=_required_int(payload.get("index"), "index"),
        name=_required_str(payload.get("name"), "name"),
        run=_required_bool(payload.get("run"), "run"),
        status=_step_status(_required_str(payload.get("status"), "status")),
        job_id=_optional_str(payload.get("job_id")),
        started_at=_optional_str(payload.get("started_at")),
        finished_at=_optional_str(payload.get("finished_at")),
        last_observed_status=_optional_str(payload.get("last_observed_status")),
    )


def _required_str(value: object, field_name: str) -> str:
    if isinstance(value, str):
        return value
    msg = f"{field_name} must be a string"
    raise ValueError(msg)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msg = "expected optional string"
    raise ValueError(msg)


def _required_int(value: object, field_name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    msg = f"{field_name} must be an integer"
    raise ValueError(msg)


def _required_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    msg = f"{field_name} must be a boolean"
    raise ValueError(msg)


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str):
        return Path(value)
    msg = "expected optional path string"
    raise ValueError(msg)


def _control_status(value: str) -> ControlStatus:
    if value in {"materialized", "running", "completed", "failed", "blocked"}:
        return cast(ControlStatus, value)
    msg = f"unknown Control State status {value!r}"
    raise ValueError(msg)


def _step_status(value: str) -> StepStateStatus:
    if value in {"pending", "running", "completed", "skipped", "failed", "blocked"}:
        return cast(StepStateStatus, value)
    msg = f"unknown Control State step status {value!r}"
    raise ValueError(msg)


__all__ = [
    "ControlState",
    "ControlStateValidation",
    "ControlStatus",
    "ControlStepState",
    "EventType",
    "StepStateStatus",
    "TerminalStepStateStatus",
    "control_state_from_payload",
    "control_step_state_from_payload",
]
