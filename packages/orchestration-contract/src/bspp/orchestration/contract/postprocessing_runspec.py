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

from collections.abc import Mapping

from bspp.orchestration.contract._postprocessing_validation import _fields, _mapping, _str
from bspp.orchestration.contract.postprocessing_execution_parsing import (
    postprocessing_cluster_snapshot_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runspec_v1 import (
    HistoricalPostprocessingPhaseRunSpecV1,
    _historical_v1_payload_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runspec_v2 import (
    PostprocessingPhaseRunSpec,
    _postprocessing_phase_runspec_v2_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runspec_v3 import (
    POSTPROCESSING_RUNSPEC_V3_KIND,
    PostprocessingPhaseRunSpecV3,
    postprocessing_phase_runspec_v3_from_mapping,
)
from bspp.orchestration.contract.runplan import reject_environment_interpolation
from bspp.orchestration.contract.versioning import validate_schema_version

ReadablePostprocessingPhaseRunSpec = (
    PostprocessingPhaseRunSpecV3 | PostprocessingPhaseRunSpec | HistoricalPostprocessingPhaseRunSpecV1
)
ExecutablePostprocessingPhaseRunSpec = PostprocessingPhaseRunSpecV3 | PostprocessingPhaseRunSpec


def read_postprocessing_phase_runspec_from_mapping(
    payload: Mapping[str, object],
) -> ReadablePostprocessingPhaseRunSpec:
    """Read V2 for execution or the superseded V1 family for inspection only."""
    reject_environment_interpolation(payload, context="Postprocessing Phase RunSpec")
    if payload.get("runspec_kind") == POSTPROCESSING_RUNSPEC_V3_KIND:
        return postprocessing_phase_runspec_v3_from_mapping(payload)
    _fields(
        payload,
        {
            "schema_version",
            "phase_kind",
            "phase_run_id",
            "attempt_id",
            "phase_plan_digest",
            "materialized_at",
            "cluster",
            "payload",
        },
        "PostprocessingPhaseRunSpec",
    )
    if _str(payload, "phase_kind") != "postprocessing":
        raise ValueError("postprocessing Phase RunSpec must declare phase_kind 'postprocessing'")
    nested_payload = _mapping(payload, "payload")
    scientific_identity = _mapping(nested_payload, "scientific_identity")
    logical_inputs = _mapping(nested_payload, "logical_inputs")
    if (
        scientific_identity.get("identity_kind") == "postprocessing-scientific-identity-v1"
        and logical_inputs.get("manifest_kind") == "postprocessing-logical-inputs-v1"
    ):
        return HistoricalPostprocessingPhaseRunSpecV1(
            schema_version=validate_schema_version(
                payload.get("schema_version"), record_name="HistoricalPostprocessingPhaseRunSpecV1"
            ),
            phase_run_id=_str(payload, "phase_run_id"),
            attempt_id=_str(payload, "attempt_id"),
            phase_plan_digest=_str(payload, "phase_plan_digest"),
            materialized_at=_str(payload, "materialized_at"),
            cluster=postprocessing_cluster_snapshot_from_mapping(_mapping(payload, "cluster")),
            payload=_historical_v1_payload_from_mapping(nested_payload),
        )
    return _postprocessing_phase_runspec_v2_from_mapping(payload)


def postprocessing_phase_runspec_from_mapping(payload: Mapping[str, object]) -> ExecutablePostprocessingPhaseRunSpec:
    """Load an executable V2/V3 RunSpec, rejecting historical V1 authority."""
    runspec = read_postprocessing_phase_runspec_from_mapping(payload)
    if isinstance(runspec, HistoricalPostprocessingPhaseRunSpecV1):
        raise ValueError(
            "historical postprocessing V1 authority is read-only; rematerialize from the Phase Plan to execute"
        )
    return runspec


__all__ = [
    "ExecutablePostprocessingPhaseRunSpec",
    "ReadablePostprocessingPhaseRunSpec",
    "postprocessing_phase_runspec_from_mapping",
    "read_postprocessing_phase_runspec_from_mapping",
]
