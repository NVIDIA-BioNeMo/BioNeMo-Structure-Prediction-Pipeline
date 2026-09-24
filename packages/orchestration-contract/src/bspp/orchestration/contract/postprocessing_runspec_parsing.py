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

"""Version-neutral parsers for postprocessing RunSpec value objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from bspp.orchestration.contract._postprocessing_validation import (
    _fields,
    _mapping,
    _str,
    _str_list,
)
from bspp.orchestration.contract.phase import phase_slurm_resources_from_mapping
from bspp.orchestration.contract.postprocessing_action_contract import PostprocessingRuntimeAction
from bspp.orchestration.contract.postprocessing_logical_identity import (
    InputVerificationKind,
    LogicalInputEntry,
    PhysicalInputLocator,
)
from bspp.orchestration.contract.versioning import validate_schema_version


def postprocessing_runtime_action_from_mapping(payload: Mapping[str, object]) -> PostprocessingRuntimeAction:
    _fields(
        payload,
        {
            "schema_version",
            "action_kind",
            "action_id",
            "step_name",
            "dependencies",
            "resources",
            "normalized_array",
            "expected_task_indexes",
        },
        "PostprocessingRuntimeAction",
    )
    dependencies = _str_list(payload, "dependencies")
    indexes = payload.get("expected_task_indexes")
    if not isinstance(indexes, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in indexes):
        raise ValueError("expected_task_indexes must be a list of integers")
    normalized = payload.get("normalized_array")
    if normalized is not None and (not isinstance(normalized, str) or not normalized):
        raise ValueError("normalized_array must be null or non-empty string")
    if _str(payload, "action_kind") != "postprocessing-step":
        raise ValueError("unsupported postprocessing action kind")
    return PostprocessingRuntimeAction(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingRuntimeAction"
        ),
        action_id=_str(payload, "action_id"),
        step_name=_str(payload, "step_name"),
        dependencies=dependencies,
        resources=phase_slurm_resources_from_mapping(_mapping(payload, "resources")),
        normalized_array=normalized,
        expected_task_indexes=tuple(indexes),
    )


def logical_input_entry_from_mapping(payload: Mapping[str, object]) -> LogicalInputEntry:
    _fields(
        payload,
        {
            "schema_version",
            "name",
            "verification_kind",
            "authority",
            "content_sha256",
            "size_bytes",
            "member_identity",
            "expected_content_sha256",
            "expected_size_bytes",
        },
        "LogicalInputEntry",
    )
    kind = _str(payload, "verification_kind")
    if kind not in {"local-content-sha256-v1", "authority-declared-content-v1"}:
        raise ValueError(f"unsupported logical input verification kind: {kind!r}")
    digest = payload.get("content_sha256")
    size = payload.get("size_bytes")
    identity = payload.get("member_identity")
    expected_digest = payload.get("expected_content_sha256")
    expected_size = payload.get("expected_size_bytes")
    if digest is not None and not isinstance(digest, str):
        raise ValueError("logical input content_sha256 must be a string when present")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool)):
        raise ValueError("logical input size_bytes must be an integer when present")
    if identity is not None and (not isinstance(identity, str) or not identity):
        raise ValueError("logical input member_identity must be non-empty when present")
    if expected_digest is not None and not isinstance(expected_digest, str):
        raise ValueError("logical input expected_content_sha256 must be a string when present")
    if expected_size is not None and (not isinstance(expected_size, int) or isinstance(expected_size, bool)):
        raise ValueError("logical input expected_size_bytes must be an integer when present")
    return LogicalInputEntry(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="LogicalInputEntry"),
        name=_str(payload, "name"),
        verification_kind=cast("InputVerificationKind", kind),
        authority=_str(payload, "authority"),
        content_sha256=digest,
        size_bytes=size,
        member_identity=identity,
        expected_content_sha256=expected_digest,
        expected_size_bytes=expected_size,
    )


def physical_input_locator_from_mapping(payload: Mapping[str, object]) -> PhysicalInputLocator:
    _fields(payload, {"schema_version", "name", "locator", "purpose"}, "PhysicalInputLocator")
    return PhysicalInputLocator(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PhysicalInputLocator"),
        name=_str(payload, "name"),
        locator=_str(payload, "locator"),
        purpose=_str(payload, "purpose"),
    )


__all__ = [
    "logical_input_entry_from_mapping",
    "physical_input_locator_from_mapping",
    "postprocessing_runtime_action_from_mapping",
]
