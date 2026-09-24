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

"""Compatibility exports for focused postprocessing contract modules."""

from __future__ import annotations

from bspp.orchestration.contract.postprocessing_action_contract import (
    PostprocessingActionKind,
    PostprocessingActionSemanticsV2,
    PostprocessingActionSemanticV2,
    PostprocessingRuntimeAction,
    normalize_slurm_array,
    postprocessing_action_graph_digest,
    postprocessing_action_semantics_digest,
    validate_postprocessing_action_graph,
)
from bspp.orchestration.contract.postprocessing_execution import (
    PostprocessingAttemptPaths,
    PostprocessingClusterSnapshot,
    PostprocessingCredentialMountSnapshot,
    PostprocessingExecutionProjection,
    PostprocessingPhaseExecutionIdentity,
    PostprocessingRuntimeImagePolicy,
    PostprocessingRuntimeSourceKind,
    ProjectionKind,
    QualifiedPostprocessingRuntimeSelection,
)
from bspp.orchestration.contract.postprocessing_identity_v3 import (
    PostprocessingActionSemanticsV3,
    PostprocessingActionSemanticV3,
    PostprocessingDatasetScopeV3,
    PostprocessingScientificIdentityV3,
)
from bspp.orchestration.contract.postprocessing_logical_identity import (
    InputVerificationKind,
    LogicalInputEntry,
    LogicalInputManifest,
    PhysicalInputLocator,
    PostprocessingDatasetScopeV2,
    PostprocessingLogicalInputIdentityManifestV2,
    PostprocessingLogicalInputIdentityV2,
    PostprocessingScientificIdentityV1,
    PostprocessingScientificIdentityV2,
    PostprocessingScientificParametersV2,
    PostprocessingSemanticField,
)
from bspp.orchestration.contract.postprocessing_phase_ids import POSTPROCESSING_STEP_ORDINALS, postprocessing_action_id
from bspp.orchestration.contract.postprocessing_plan import (
    LocalAuthorityDocument,
    PostprocessingPhaseKind,
    PostprocessingPhasePlan,
    postprocessing_phase_plan_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runspec import (
    ExecutablePostprocessingPhaseRunSpec,
    ReadablePostprocessingPhaseRunSpec,
    postprocessing_phase_runspec_from_mapping,
    read_postprocessing_phase_runspec_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runspec_v1 import HistoricalPostprocessingPhaseRunSpecV1
from bspp.orchestration.contract.postprocessing_runspec_v2 import (
    PostprocessingPhaseRunSpec,
    PostprocessingPhaseRunSpecPayload,
)
from bspp.orchestration.contract.postprocessing_runspec_v3 import (
    POSTPROCESSING_RUNSPEC_V3_KIND,
    PostprocessingPhaseRunSpecPayloadV3,
    PostprocessingPhaseRunSpecV3,
)

__all__ = [
    "POSTPROCESSING_RUNSPEC_V3_KIND",
    "POSTPROCESSING_STEP_ORDINALS",
    "ExecutablePostprocessingPhaseRunSpec",
    "HistoricalPostprocessingPhaseRunSpecV1",
    "InputVerificationKind",
    "LocalAuthorityDocument",
    "LogicalInputEntry",
    "LogicalInputManifest",
    "PhysicalInputLocator",
    "PostprocessingActionKind",
    "PostprocessingActionSemanticV2",
    "PostprocessingActionSemanticV3",
    "PostprocessingActionSemanticsV2",
    "PostprocessingActionSemanticsV3",
    "PostprocessingAttemptPaths",
    "PostprocessingClusterSnapshot",
    "PostprocessingCredentialMountSnapshot",
    "PostprocessingDatasetScopeV2",
    "PostprocessingDatasetScopeV3",
    "PostprocessingExecutionProjection",
    "PostprocessingLogicalInputIdentityManifestV2",
    "PostprocessingLogicalInputIdentityV2",
    "PostprocessingPhaseExecutionIdentity",
    "PostprocessingPhaseKind",
    "PostprocessingPhasePlan",
    "PostprocessingPhaseRunSpec",
    "PostprocessingPhaseRunSpecPayload",
    "PostprocessingPhaseRunSpecPayloadV3",
    "PostprocessingPhaseRunSpecV3",
    "PostprocessingRuntimeAction",
    "PostprocessingRuntimeImagePolicy",
    "PostprocessingRuntimeSourceKind",
    "PostprocessingScientificIdentityV1",
    "PostprocessingScientificIdentityV2",
    "PostprocessingScientificIdentityV3",
    "PostprocessingScientificParametersV2",
    "PostprocessingSemanticField",
    "ProjectionKind",
    "QualifiedPostprocessingRuntimeSelection",
    "ReadablePostprocessingPhaseRunSpec",
    "normalize_slurm_array",
    "postprocessing_action_graph_digest",
    "postprocessing_action_id",
    "postprocessing_action_semantics_digest",
    "postprocessing_phase_plan_from_mapping",
    "postprocessing_phase_runspec_from_mapping",
    "read_postprocessing_phase_runspec_from_mapping",
    "validate_postprocessing_action_graph",
]
