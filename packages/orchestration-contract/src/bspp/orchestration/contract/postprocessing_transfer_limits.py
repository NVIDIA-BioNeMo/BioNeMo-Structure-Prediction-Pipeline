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

"""Focused postprocessing contracts extracted from postprocessing_finalization_bundle.py."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")


POSTPROCESSING_FINALIZATION_FIXED_PATHS = (
    "acceptance/adjudication.json",
    "acceptance/bundle.json",
    "acceptance/captures/acceptance-semantic.json",
    "acceptance/captures/acceptance-tar-payload-parity.json",
    "acceptance/captures/acceptance-verify-evidence.json",
    "acceptance/reports/acceptance-evidence-report.json",
    "acceptance/reports/semantic-acceptance-summary.json",
    "acceptance/reports/tar-payload-parity-report.json",
    "aggregate-action-evidence.json",
    "inputs/runtime-input-attestations.json",
    "outputs/artifact-locations.json",
    "outputs/artifact-set-root.json",
    "outputs/output-handoff.json",
    "outputs/scientific-output-root.json",
)


@dataclass(frozen=True)
class PostprocessingEvidenceTransferLimitsV1:
    """Non-configurable limits applied before any indexed descendant fetch."""

    max_indexed_files: int = 96
    max_relative_path_depth: int = 8
    max_relative_path_utf8_bytes: int = 512
    max_path_component_utf8_bytes: int = 128
    max_file_bytes: int = 16_777_216
    max_aggregate_bytes: int = 134_217_728
    max_tar_manifests: int = 64
    max_members_per_tar_manifest: int = 16_384
    max_total_tar_members: int = 131_072
    permitted_suffixes: tuple[str, ...] = (".json",)
    layout_kind: Literal["postprocessing-finalization-layout-v1"] = "postprocessing-finalization-layout-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name=type(self).__name__)
        if (
            self.max_indexed_files,
            self.max_relative_path_depth,
            self.max_relative_path_utf8_bytes,
            self.max_path_component_utf8_bytes,
            self.max_file_bytes,
            self.max_aggregate_bytes,
            self.max_tar_manifests,
            self.max_members_per_tar_manifest,
            self.max_total_tar_members,
            self.permitted_suffixes,
            self.layout_kind,
        ) != (
            96,
            8,
            512,
            128,
            16_777_216,
            134_217_728,
            64,
            16_384,
            131_072,
            (".json",),
            "postprocessing-finalization-layout-v1",
        ):
            raise ValueError("postprocessing evidence transfer limits v1 are immutable")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "layout_kind": self.layout_kind,
            "max_indexed_files": self.max_indexed_files,
            "max_relative_path_depth": self.max_relative_path_depth,
            "max_relative_path_utf8_bytes": self.max_relative_path_utf8_bytes,
            "max_path_component_utf8_bytes": self.max_path_component_utf8_bytes,
            "max_file_bytes": self.max_file_bytes,
            "max_aggregate_bytes": self.max_aggregate_bytes,
            "max_tar_manifests": self.max_tar_manifests,
            "max_members_per_tar_manifest": self.max_members_per_tar_manifest,
            "max_total_tar_members": self.max_total_tar_members,
            "permitted_suffixes": list(self.permitted_suffixes),
        }


POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1 = PostprocessingEvidenceTransferLimitsV1()


def validate_postprocessing_bundle_relative_path(value: str) -> str:
    """Validate one normalized JSON-only path under the finalization root."""
    limits = POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or unicodedata.normalize("NFC", value) != value
        or _CONTROL_CHARACTER.search(value) is not None
        or len(value.encode("utf-8")) > limits.max_relative_path_utf8_bytes
    ):
        raise ValueError("postprocessing finalization member path is unsafe or noncanonical")
    parts = value.split("/")
    if (
        len(parts) > limits.max_relative_path_depth
        or any(part in {"", ".", ".."} for part in parts)
        or any(len(part.encode("utf-8")) > limits.max_path_component_utf8_bytes for part in parts)
        or PurePosixPath(value).as_posix() != value
        or not value.endswith(limits.permitted_suffixes)
    ):
        raise ValueError("postprocessing finalization member path exceeds v1 layout limits")
    return value


def _limits_from_mapping(payload: Mapping[str, object]) -> PostprocessingEvidenceTransferLimitsV1:
    expected = POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1.to_mapping()
    if dict(payload) != expected:
        raise ValueError("postprocessing handoff index transfer limits differ from immutable v1 constants")
    return POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1


__all__ = [
    "POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1",
    "POSTPROCESSING_FINALIZATION_FIXED_PATHS",
    "PostprocessingEvidenceTransferLimitsV1",
    "validate_postprocessing_bundle_relative_path",
]
