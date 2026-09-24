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

"""User-authored Run Plan schema and loading helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import StrictStr, field_validator, model_validator

from bspp.orchestration.contract.config_models import FrozenConfigModel
from bspp.orchestration.contract.runspec import (
    AcceptanceSpec,
    AnalysisMetadataSpec,
    DatasetSpec,
    ObjectStorageSpec,
    ReferenceSpec,
    RunDataPlacementSpec,
    RunKind,
    RunSecrets,
    ValidationPolicy,
    WorkerSpec,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_ENV_INTERPOLATION_PATTERN = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")
_SINGLE_CLUSTER_MESSAGE = "Run Plans support a single target_cluster only; multi-cluster input is not supported"


class RunPlan(FrozenConfigModel):
    """User-authored run intent before cluster-specific resolution.

    ``run_kind`` is validated here and emitted into the concrete RunSpec.
    """

    schema_version: Literal[1] = CURRENT_CONTRACT_SCHEMA_VERSION
    run_kind: RunKind
    target_cluster: StrictStr
    workflow_template: StrictStr
    dataset: DatasetSpec
    references: ReferenceSpec
    worker: WorkerSpec
    storage: ObjectStorageSpec
    data_placement: RunDataPlacementSpec | None = None
    analysis_metadata: AnalysisMetadataSpec | None = None
    validation: ValidationPolicy | None = None
    acceptance: AcceptanceSpec | None = None
    secrets: RunSecrets

    @model_validator(mode="before")
    @classmethod
    def _reject_multi_cluster_shapes(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        data["schema_version"] = validate_schema_version(data.get("schema_version"), record_name="RunPlan")
        if "targets" in data or "target_clusters" in data:
            raise ValueError(_SINGLE_CLUSTER_MESSAGE)
        target_cluster = data.get("target_cluster")
        if isinstance(target_cluster, list | tuple):
            raise ValueError(_SINGLE_CLUSTER_MESSAGE)
        return data

    @field_validator("target_cluster", "workflow_template")
    @classmethod
    def _validate_non_empty_string(cls, value: str) -> str:
        if value == "":
            msg = "Expected non-empty string"
            raise ValueError(msg)
        return value


def load_runplan(path: Path) -> RunPlan:
    """Load and validate a YAML Run Plan."""
    data = yaml.safe_load(path.read_bytes())
    if not isinstance(data, dict):
        msg = f"Expected Run Plan YAML mapping in {path}"
        raise TypeError(msg)
    reject_environment_interpolation(data, context="Run Plan")
    return RunPlan.model_validate(data)


def reject_environment_interpolation(value: object, *, context: str, path: Sequence[str] = ()) -> None:
    """Reject shell-style ``$VAR`` and ``${VAR}`` strings in static configuration."""
    if isinstance(value, str):
        if _ENV_INTERPOLATION_PATTERN.search(value):
            location = ".".join(path) if path else "<root>"
            msg = f"{context} does not support environment interpolation at {location}: {value!r}"
            raise ValueError(msg)
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            reject_environment_interpolation(key, context=context, path=(*path, "<key>"))
            reject_environment_interpolation(nested, context=context, path=(*path, str(key)))
        return
    if isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            reject_environment_interpolation(nested, context=context, path=(*path, str(index)))


__all__ = [
    "RunKind",
    "RunPlan",
    "load_runplan",
    "reject_environment_interpolation",
]
