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

"""Raw postprocessing acceptance evidence capture contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from bspp.orchestration.contract._postprocessing_validation import _fields, _int, _mapping_list, _schema, _str
from bspp.orchestration.contract.phase import (
    canonical_mapping_digest,
    validate_phase_attempt_id,
    validate_phase_run_id,
)
from bspp.orchestration.contract.postprocessing_acceptance_policy import AcceptanceStepName
from bspp.orchestration.contract.postprocessing_phase_ids import postprocessing_action_id
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PostprocessingArtifactBinding:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if (
            not self.path
            or self.path.startswith("/")
            or ".." in self.path.split("/")
            or _SHA256.fullmatch(self.sha256) is None
            or not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("postprocessing acceptance artifact binding is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class PostprocessingAcceptanceCapture:
    """Raw exit and immutable report bindings captured without deciding acceptance."""

    phase_run_id: str
    attempt_id: str
    action_id: str
    step_name: AcceptanceStepName
    policy_sha256: str
    raw_exit_code: int
    raw_stdout: PostprocessingArtifactBinding
    raw_stderr: PostprocessingArtifactBinding
    reports: tuple[PostprocessingArtifactBinding, ...]
    completed_at: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        if self.action_id != postprocessing_action_id(self.step_name):
            raise ValueError("acceptance capture action id does not match its step")
        if _SHA256.fullmatch(self.policy_sha256) is None:
            raise ValueError("acceptance capture policy digest must be lowercase SHA-256")
        if (
            not isinstance(self.raw_exit_code, int)
            or isinstance(self.raw_exit_code, bool)
            or not 0 <= self.raw_exit_code <= 255
        ):
            raise ValueError("acceptance capture raw exit code must be in 0..255")
        report_paths = tuple(item.path for item in self.reports)
        if (
            not report_paths
            or tuple(sorted(report_paths)) != report_paths
            or len(set(report_paths)) != len(report_paths)
        ):
            raise ValueError("acceptance capture reports must be a non-empty sorted tuple")
        if not self.completed_at:
            raise ValueError("acceptance capture completed_at must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "action_id": self.action_id,
            "step_name": self.step_name,
            "policy_sha256": self.policy_sha256,
            "raw_exit_code": self.raw_exit_code,
            "raw_stdout": self.raw_stdout.to_mapping(),
            "raw_stderr": self.raw_stderr.to_mapping(),
            "reports": [item.to_mapping() for item in self.reports],
            "completed_at": self.completed_at,
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


def postprocessing_acceptance_capture_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingAcceptanceCapture:
    _fields(
        payload,
        {
            "schema_version",
            "phase_run_id",
            "attempt_id",
            "action_id",
            "step_name",
            "policy_sha256",
            "raw_exit_code",
            "raw_stdout",
            "raw_stderr",
            "reports",
            "completed_at",
        },
        "PostprocessingAcceptanceCapture",
    )
    step = _str(payload, "step_name")
    if step not in {
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    }:
        raise ValueError(f"unsupported acceptance capture step: {step!r}")
    reports: list[PostprocessingArtifactBinding] = []
    for item in _mapping_list(payload, "reports"):
        _fields(item, {"path", "sha256", "size_bytes"}, "PostprocessingAcceptanceCaptureReport")
        reports.append(_artifact_from_mapping(item, record="PostprocessingAcceptanceCaptureReport"))
    return PostprocessingAcceptanceCapture(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingAcceptanceCapture"
        ),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        action_id=_str(payload, "action_id"),
        step_name=cast("AcceptanceStepName", step),
        policy_sha256=_str(payload, "policy_sha256"),
        raw_exit_code=_int(payload, "raw_exit_code"),
        raw_stdout=_artifact_from_mapping(payload.get("raw_stdout"), record="PostprocessingAcceptanceRawStdout"),
        raw_stderr=_artifact_from_mapping(payload.get("raw_stderr"), record="PostprocessingAcceptanceRawStderr"),
        reports=tuple(reports),
        completed_at=_str(payload, "completed_at"),
    )


def _artifact_from_mapping(value: object, *, record: str) -> PostprocessingArtifactBinding:
    if not isinstance(value, Mapping):
        raise ValueError(f"{record} must be a mapping")
    _fields(value, {"path", "sha256", "size_bytes"}, record)
    return PostprocessingArtifactBinding(
        path=_str(value, "path"),
        sha256=_str(value, "sha256"),
        size_bytes=_int(value, "size_bytes"),
    )


__all__ = [
    "PostprocessingAcceptanceCapture",
    "PostprocessingArtifactBinding",
    "postprocessing_acceptance_capture_from_mapping",
]
