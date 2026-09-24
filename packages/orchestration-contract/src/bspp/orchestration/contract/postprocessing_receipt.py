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

from bspp.orchestration.contract.postprocessing_artifacts import (
    PostprocessingEvidenceArtifact,
    PostprocessingLogicalArtifactSet,
    PostprocessingScientificOutputInventory,
    postprocessing_artifact_inventory_digest,
    postprocessing_artifact_set_id,
    postprocessing_scientific_output_inventory_from_mapping,
)
from bspp.orchestration.contract.postprocessing_attestations import (
    PostprocessingRuntimeInputAttestation,
    PostprocessingRuntimeInputAttestationSet,
    PostprocessingRuntimeQualificationAttestation,
    postprocessing_runtime_input_attestation_set_from_mapping,
)
from bspp.orchestration.contract.postprocessing_handoff import (
    PostprocessingActionReceiptEvidence,
    PostprocessingArtifactPlacement,
    PostprocessingOutputHandoff,
    PostprocessingTaskReceiptEvidence,
    PostprocessingVerifiedArtifactLocation,
    postprocessing_handoff_id,
    postprocessing_output_handoff_from_mapping,
    postprocessing_verified_artifact_location_from_mapping,
)
from bspp.orchestration.contract.postprocessing_phase_receipt import (
    PostprocessingFinalizedPayload,
    PostprocessingPhaseReceipt,
    postprocessing_finalized_payload_from_mapping,
    postprocessing_receipt_id,
)

__all__ = [
    "PostprocessingActionReceiptEvidence",
    "PostprocessingArtifactPlacement",
    "PostprocessingEvidenceArtifact",
    "PostprocessingFinalizedPayload",
    "PostprocessingLogicalArtifactSet",
    "PostprocessingOutputHandoff",
    "PostprocessingPhaseReceipt",
    "PostprocessingRuntimeInputAttestation",
    "PostprocessingRuntimeInputAttestationSet",
    "PostprocessingRuntimeQualificationAttestation",
    "PostprocessingScientificOutputInventory",
    "PostprocessingTaskReceiptEvidence",
    "PostprocessingVerifiedArtifactLocation",
    "postprocessing_artifact_inventory_digest",
    "postprocessing_artifact_set_id",
    "postprocessing_finalized_payload_from_mapping",
    "postprocessing_handoff_id",
    "postprocessing_output_handoff_from_mapping",
    "postprocessing_receipt_id",
    "postprocessing_runtime_input_attestation_set_from_mapping",
    "postprocessing_scientific_output_inventory_from_mapping",
    "postprocessing_verified_artifact_location_from_mapping",
]
