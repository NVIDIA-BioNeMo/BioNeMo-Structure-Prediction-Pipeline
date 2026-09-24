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

"""Compatibility exports for focused postprocessing acceptance contracts."""

from __future__ import annotations

from bspp.orchestration.contract.postprocessing_acceptance_adjudication import (
    PostprocessingAcceptanceAdjudication,
    PostprocessingReconciliationResult,
    PostprocessingResidualCardinality,
    postprocessing_acceptance_adjudication_from_mapping,
)
from bspp.orchestration.contract.postprocessing_acceptance_capture import (
    PostprocessingAcceptanceCapture,
    PostprocessingArtifactBinding,
    postprocessing_acceptance_capture_from_mapping,
)
from bspp.orchestration.contract.postprocessing_acceptance_diagnostics import (
    PostprocessingAcceptanceEvaluation,
    canonical_allowance_id,
    diagnostic_allowance_reference,
    evaluate_postprocessing_acceptance_reports,
    project_acceptance_evidence_issues,
    project_unallowlisted_occurrences,
    safe_canonical_json_projection,
)
from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    AcceptanceStepName,
    PostprocessingAcceptancePolicySnapshot,
    PostprocessingBaselineReportBinding,
    PostprocessingCompletionExitContract,
    PostprocessingCrossReportReconciliation,
    PostprocessingRawExitReportOutcome,
    PostprocessingResidualAllowance,
    ResidualMatchKind,
    postprocessing_acceptance_policy_from_mapping,
)
from bspp.orchestration.contract.postprocessing_acceptance_reference import (
    PostprocessingAcceptanceSnapshotReference,
)

__all__ = [
    "AcceptanceStepName",
    "PostprocessingAcceptanceAdjudication",
    "PostprocessingAcceptanceCapture",
    "PostprocessingAcceptanceEvaluation",
    "PostprocessingAcceptancePolicySnapshot",
    "PostprocessingAcceptanceSnapshotReference",
    "PostprocessingArtifactBinding",
    "PostprocessingBaselineReportBinding",
    "PostprocessingCompletionExitContract",
    "PostprocessingCrossReportReconciliation",
    "PostprocessingRawExitReportOutcome",
    "PostprocessingReconciliationResult",
    "PostprocessingResidualAllowance",
    "PostprocessingResidualCardinality",
    "ResidualMatchKind",
    "canonical_allowance_id",
    "diagnostic_allowance_reference",
    "evaluate_postprocessing_acceptance_reports",
    "postprocessing_acceptance_adjudication_from_mapping",
    "postprocessing_acceptance_capture_from_mapping",
    "postprocessing_acceptance_policy_from_mapping",
    "project_acceptance_evidence_issues",
    "project_unallowlisted_occurrences",
    "safe_canonical_json_projection",
]
