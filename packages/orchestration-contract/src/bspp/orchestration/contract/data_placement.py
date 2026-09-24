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

"""Shared data-placement contract records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

DataPlacementStage = Literal["s3", "gcs", "gcs-direct"]
DataPlacementTool = Literal["dm", "s5cmd", "gcloud"]

DATA_PLACEMENT_TOOLS_BY_STAGE: dict[DataPlacementStage, tuple[DataPlacementTool, ...]] = {
    "s3": ("s5cmd", "dm"),
    "gcs": ("gcloud", "dm"),
    "gcs-direct": ("gcloud",),
}


@dataclass(frozen=True)
class DataPlacementRecord:
    """Portable evidence record for one data-placement stage."""

    stage: DataPlacementStage
    tool: DataPlacementTool
    dataset: str
    source: str
    destination: str
    payload_bytes_moved: bool
    evidence_path: str | None = None
    commands: tuple[tuple[str, ...], ...] = ()
    job_id: str | None = None
    terminal_status: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic YAML/JSON-ready data."""
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "tool": self.tool,
            "dataset": self.dataset,
            "source": self.source,
            "destination": self.destination,
            "payload_bytes_moved": self.payload_bytes_moved,
            "evidence_path": self.evidence_path,
            "commands": [list(command) for command in self.commands],
            "job_id": self.job_id,
            "terminal_status": self.terminal_status,
        }


def data_placement_record_from_mapping(payload: Mapping[str, object]) -> DataPlacementRecord:
    """Parse a data-placement evidence record from a YAML/JSON mapping."""
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="DataPlacement")
    stage = _required_stage(payload, "stage")
    tool = _required_tool(payload, "tool")
    validate_data_placement_tool_for_stage(tool, stage)
    return DataPlacementRecord(
        schema_version=schema_version,
        stage=stage,
        tool=tool,
        dataset=_required_str(payload, "dataset"),
        source=_required_str(payload, "source"),
        destination=_required_str(payload, "destination"),
        payload_bytes_moved=_required_bool(payload, "payload_bytes_moved"),
        evidence_path=_optional_str(payload, "evidence_path"),
        commands=_optional_commands(payload, "commands"),
        job_id=_optional_str(payload, "job_id"),
        terminal_status=_optional_str(payload, "terminal_status"),
    )


def validate_data_placement_tool_for_stage(tool: DataPlacementTool, stage: DataPlacementStage) -> None:
    """Raise ValueError if a data-placement tool is invalid for a stage."""
    allowed = DATA_PLACEMENT_TOOLS_BY_STAGE[stage]
    if tool not in allowed:
        msg = f"Tool {tool!r} is not supported for stage {stage!r}. Allowed: {', '.join(allowed)}."
        raise ValueError(msg)


def _required_stage(payload: Mapping[str, object], key: str) -> DataPlacementStage:
    value = _required_str(payload, key)
    if value not in DATA_PLACEMENT_TOOLS_BY_STAGE:
        msg = f"{key} must be one of: {', '.join(DATA_PLACEMENT_TOOLS_BY_STAGE)}"
        raise ValueError(msg)
    return value


def _required_tool(payload: Mapping[str, object], key: str) -> DataPlacementTool:
    value = _required_str(payload, key)
    if value not in {"dm", "s5cmd", "gcloud"}:
        msg = f"{key} must be one of: dm, s5cmd, gcloud"
        raise ValueError(msg)
    return cast("DataPlacementTool", value)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None or isinstance(value, str):
        return value
    msg = f"{key} must be a string or null"
    raise ValueError(msg)


def _optional_commands(payload: Mapping[str, object], key: str) -> tuple[tuple[str, ...], ...]:
    value = payload.get(key, ())
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    commands: list[tuple[str, ...]] = []
    for index, command in enumerate(value):
        if not isinstance(command, list | tuple) or not all(isinstance(part, str) for part in command):
            msg = f"{key}[{index}] must be a list of strings"
            raise ValueError(msg)
        commands.append(tuple(command))
    return tuple(commands)


def _required_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean"
        raise ValueError(msg)
    return value


__all__ = [
    "DATA_PLACEMENT_TOOLS_BY_STAGE",
    "DataPlacementRecord",
    "DataPlacementStage",
    "DataPlacementTool",
    "data_placement_record_from_mapping",
    "validate_data_placement_tool_for_stage",
]
