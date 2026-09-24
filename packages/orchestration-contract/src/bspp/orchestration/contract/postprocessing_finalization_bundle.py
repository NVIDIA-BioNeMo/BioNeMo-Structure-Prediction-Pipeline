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

from bspp.orchestration.contract.postprocessing_action09_bundle import (
    PostprocessingAction09AssemblyWitness,
    PostprocessingBundleMemberIdentity,
    PostprocessingFinalizationHandoffIndex,
    postprocessing_action09_assembly_witness_from_mapping,
    postprocessing_handoff_index_from_mapping,
)
from bspp.orchestration.contract.postprocessing_artifact_locations import (
    PostprocessingArtifactLocationSet,
    postprocessing_artifact_location_set_from_mapping,
)
from bspp.orchestration.contract.postprocessing_bundle_manifest import (
    PostprocessingScientificOutputRoot,
    PostprocessingSmallOutputIdentity,
    PostprocessingTarManifest,
    PostprocessingTarManifestReference,
    PostprocessingTarMemberIdentity,
    postprocessing_scientific_output_root_from_mapping,
    postprocessing_tar_manifest_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runtime_evidence import (
    PostprocessingCompletedRuntimeActionEvidence,
    PostprocessingRuntimeActionEvidenceAggregate,
    PostprocessingRuntimeTaskEvidence,
    postprocessing_runtime_action_evidence_aggregate_from_mapping,
    postprocessing_runtime_task_evidence_from_mapping,
)
from bspp.orchestration.contract.postprocessing_transfer_limits import (
    POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1,
    POSTPROCESSING_FINALIZATION_FIXED_PATHS,
    PostprocessingEvidenceTransferLimitsV1,
    validate_postprocessing_bundle_relative_path,
)

__all__ = [
    "POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1",
    "POSTPROCESSING_FINALIZATION_FIXED_PATHS",
    "PostprocessingAction09AssemblyWitness",
    "PostprocessingArtifactLocationSet",
    "PostprocessingBundleMemberIdentity",
    "PostprocessingCompletedRuntimeActionEvidence",
    "PostprocessingEvidenceTransferLimitsV1",
    "PostprocessingFinalizationHandoffIndex",
    "PostprocessingRuntimeActionEvidenceAggregate",
    "PostprocessingRuntimeTaskEvidence",
    "PostprocessingScientificOutputRoot",
    "PostprocessingSmallOutputIdentity",
    "PostprocessingTarManifest",
    "PostprocessingTarManifestReference",
    "PostprocessingTarMemberIdentity",
    "postprocessing_action09_assembly_witness_from_mapping",
    "postprocessing_artifact_location_set_from_mapping",
    "postprocessing_handoff_index_from_mapping",
    "postprocessing_runtime_action_evidence_aggregate_from_mapping",
    "postprocessing_runtime_task_evidence_from_mapping",
    "postprocessing_scientific_output_root_from_mapping",
    "postprocessing_tar_manifest_from_mapping",
    "validate_postprocessing_bundle_relative_path",
]
