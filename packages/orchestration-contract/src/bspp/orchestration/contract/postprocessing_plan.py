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

"""Focused postprocessing contracts extracted from phase_postprocessing.py."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract._postprocessing_validation import (
    _fields,
    _int,
    _mapping,
    _nonnegative,
    _schema,
    _sha,
    _str,
)
from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_autorequeue_policy import (
    POSTPROCESSING_AUTOREQUEUE_DISABLED,
    PostprocessingAutorequeuePolicy,
    postprocessing_autorequeue_policy_from_mapping,
)
from bspp.orchestration.contract.runplan import reject_environment_interpolation
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

PostprocessingPhaseKind = Literal["postprocessing"]


_OUTPUT_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class LocalAuthorityDocument:
    """A locally readable authored document verified before materialization."""

    path: str
    sha256: str
    size_bytes: int
    document_kind: Literal[
        "legacy-run-plan",
        "acceptance-policy",
        "logical-input-inventory",
        "runtime-qualification",
    ]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not self.path:
            raise ValueError("local authority document path must be non-empty")
        _sha(self.sha256, "local authority document SHA-256")
        _nonnegative(self.size_bytes, "local authority document size")
        if self.document_kind not in {
            "legacy-run-plan",
            "acceptance-policy",
            "logical-input-inventory",
            "runtime-qualification",
        }:
            raise ValueError(f"unsupported local authority document kind: {self.document_kind!r}")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "document_kind": self.document_kind,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class PostprocessingPhasePlan:
    """Separate phase-oriented intent referencing the unchanged legacy Run Plan."""

    target_cluster: str
    output_namespace: str
    legacy_run_plan: LocalAuthorityDocument
    acceptance_policy: LocalAuthorityDocument
    logical_input_inventory: LocalAuthorityDocument
    runtime_qualification: LocalAuthorityDocument
    phase_kind: PostprocessingPhaseKind = "postprocessing"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION
    autorequeue_policy: PostprocessingAutorequeuePolicy = POSTPROCESSING_AUTOREQUEUE_DISABLED

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.phase_kind != "postprocessing":
            raise ValueError("postprocessing Phase Plan must declare phase_kind 'postprocessing'")
        if not self.target_cluster:
            raise ValueError("postprocessing Phase Plan target_cluster must be non-empty")
        if _OUTPUT_NAMESPACE.fullmatch(self.output_namespace) is None:
            raise ValueError("postprocessing output_namespace must be a portable non-empty name")
        if self.legacy_run_plan.document_kind != "legacy-run-plan":
            raise ValueError("legacy_run_plan must reference a legacy-run-plan document")
        if self.acceptance_policy.document_kind != "acceptance-policy":
            raise ValueError("acceptance_policy must reference an acceptance-policy document")
        if self.logical_input_inventory.document_kind != "logical-input-inventory":
            raise ValueError("logical_input_inventory must reference a logical-input-inventory document")
        if self.runtime_qualification.document_kind != "runtime-qualification":
            raise ValueError("runtime_qualification must reference a runtime-qualification document")

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "target_cluster": self.target_cluster,
            "output_namespace": self.output_namespace,
            "legacy_run_plan": self.legacy_run_plan.to_mapping(),
            "acceptance_policy": self.acceptance_policy.to_mapping(),
            "logical_input_inventory": self.logical_input_inventory.to_mapping(),
            "runtime_qualification": self.runtime_qualification.to_mapping(),
        }
        if self.autorequeue_policy != POSTPROCESSING_AUTOREQUEUE_DISABLED:
            mapping["autorequeue_policy"] = self.autorequeue_policy.to_mapping()
        return mapping

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


def postprocessing_phase_plan_from_mapping(payload: Mapping[str, object]) -> PostprocessingPhasePlan:
    reject_environment_interpolation(payload, context="Postprocessing Phase Plan")
    _fields(
        payload,
        {
            "schema_version",
            "phase_kind",
            "target_cluster",
            "output_namespace",
            "legacy_run_plan",
            "acceptance_policy",
            "logical_input_inventory",
            "runtime_qualification",
            "autorequeue_policy",
        },
        "PostprocessingPhasePlan",
    )
    if _str(payload, "phase_kind") != "postprocessing":
        raise ValueError("postprocessing Phase Plan must declare phase_kind 'postprocessing'")
    return PostprocessingPhasePlan(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PostprocessingPhasePlan"),
        target_cluster=_str(payload, "target_cluster"),
        output_namespace=_str(payload, "output_namespace"),
        legacy_run_plan=_local_document(_mapping(payload, "legacy_run_plan")),
        acceptance_policy=_local_document(_mapping(payload, "acceptance_policy")),
        logical_input_inventory=_local_document(_mapping(payload, "logical_input_inventory")),
        runtime_qualification=_local_document(_mapping(payload, "runtime_qualification")),
        autorequeue_policy=(
            postprocessing_autorequeue_policy_from_mapping(_mapping(payload, "autorequeue_policy"))
            if payload.get("autorequeue_policy") is not None
            else POSTPROCESSING_AUTOREQUEUE_DISABLED
        ),
    )


def _local_document(payload: Mapping[str, object]) -> LocalAuthorityDocument:
    _fields(payload, {"schema_version", "document_kind", "path", "sha256", "size_bytes"}, "LocalAuthorityDocument")
    kind = _str(payload, "document_kind")
    if kind not in {
        "legacy-run-plan",
        "acceptance-policy",
        "logical-input-inventory",
        "runtime-qualification",
    }:
        raise ValueError(f"unsupported local authority document kind: {kind!r}")
    return LocalAuthorityDocument(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="LocalAuthorityDocument"),
        document_kind=cast(
            "Literal['legacy-run-plan', 'acceptance-policy', 'logical-input-inventory', 'runtime-qualification']",
            kind,
        ),
        path=_str(payload, "path"),
        sha256=_str(payload, "sha256"),
        size_bytes=_int(payload, "size_bytes"),
    )


__all__ = [
    "LocalAuthorityDocument",
    "PostprocessingPhaseKind",
    "PostprocessingPhasePlan",
    "postprocessing_phase_plan_from_mapping",
]
