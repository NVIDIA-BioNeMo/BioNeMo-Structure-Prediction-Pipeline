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

"""Architecture locks for the focused postprocessing contract modules."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_CONTRACT_ROOT = (
    Path(__file__).parents[1] / "packages" / "orchestration-contract" / "src" / "bspp" / "orchestration" / "contract"
)
_PACKAGE_ROOT = Path(__file__).parents[1] / "packages"
_FACADE_EXPORTS = {
    "phase_postprocessing": {
        "postprocessing_plan": ("LocalAuthorityDocument", "PostprocessingPhasePlan"),
        "postprocessing_logical_identity": (
            "LogicalInputEntry",
            "PostprocessingLogicalInputIdentityManifestV2",
            "PostprocessingScientificIdentityV2",
        ),
        "postprocessing_action_contract": ("PostprocessingRuntimeAction", "PostprocessingActionSemanticsV2"),
        "postprocessing_execution": ("PostprocessingExecutionProjection", "QualifiedPostprocessingRuntimeSelection"),
        "postprocessing_runspec_v1": ("HistoricalPostprocessingPhaseRunSpecV1",),
        "postprocessing_runspec_v2": ("PostprocessingPhaseRunSpec",),
        "postprocessing_runspec": (
            "postprocessing_phase_runspec_from_mapping",
            "read_postprocessing_phase_runspec_from_mapping",
        ),
    },
    "postprocessing_receipt": {
        "postprocessing_artifacts": ("PostprocessingLogicalArtifactSet",),
        "postprocessing_attestations": ("PostprocessingRuntimeInputAttestationSet",),
        "postprocessing_handoff": ("PostprocessingOutputHandoff",),
        "postprocessing_phase_receipt": ("PostprocessingPhaseReceipt", "PostprocessingFinalizedPayload"),
    },
    "postprocessing_finalization_bundle": {
        "postprocessing_transfer_limits": ("PostprocessingEvidenceTransferLimitsV1",),
        "postprocessing_runtime_evidence": ("PostprocessingRuntimeActionEvidenceAggregate",),
        "postprocessing_bundle_manifest": ("PostprocessingScientificOutputRoot", "PostprocessingTarManifest"),
        "postprocessing_artifact_locations": ("PostprocessingArtifactLocationSet",),
        "postprocessing_action09_bundle": ("PostprocessingFinalizationHandoffIndex",),
    },
    "postprocessing_lifecycle": {
        "postprocessing_submission_events": ("PostprocessingSubmissionIntendedPayload",),
        "postprocessing_terminal_events": ("PostprocessingActionTerminalObservedPayload",),
        "postprocessing_cancellation_events": ("PostprocessingCancelledPayload",),
        "postprocessing_retry_events": ("PostprocessingAttemptRetriedPayload",),
        "postprocessing_event": ("PostprocessingPhaseEvent", "postprocessing_phase_event_from_mapping"),
    },
    "postprocessing_acceptance": {
        "postprocessing_acceptance_policy": (
            "PostprocessingAcceptancePolicySnapshot",
            "PostprocessingRawExitReportOutcome",
        ),
        "postprocessing_acceptance_reference": ("PostprocessingAcceptanceSnapshotReference",),
        "postprocessing_acceptance_capture": (
            "PostprocessingAcceptanceCapture",
            "PostprocessingArtifactBinding",
        ),
        "postprocessing_acceptance_adjudication": (
            "PostprocessingAcceptanceAdjudication",
            "PostprocessingReconciliationResult",
            "PostprocessingResidualCardinality",
        ),
        "postprocessing_acceptance_diagnostics": (
            "PostprocessingAcceptanceEvaluation",
            "canonical_allowance_id",
            "diagnostic_allowance_reference",
            "evaluate_postprocessing_acceptance_reports",
            "project_acceptance_evidence_issues",
            "project_unallowlisted_occurrences",
            "safe_canonical_json_projection",
        ),
    },
}


@pytest.mark.parametrize(("facade_name", "owners"), _FACADE_EXPORTS.items())
def test_contract_facades_reexport_canonical_objects(
    facade_name: str,
    owners: dict[str, tuple[str, ...]],
) -> None:
    facade = importlib.import_module(f"bspp.orchestration.contract.{facade_name}")
    tree = ast.parse((_CONTRACT_ROOT / f"{facade_name}.py").read_text())
    assert not any(isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) for node in tree.body)
    for owner_name, names in owners.items():
        owner = importlib.import_module(f"bspp.orchestration.contract.{owner_name}")
        for name in names:
            assert getattr(facade, name) is getattr(owner, name)


def test_production_modules_never_import_compatibility_facades() -> None:
    forbidden = {*_FACADE_EXPORTS, "postprocessing_authority"}
    violations: list[str] = []
    for path in _PACKAGE_ROOT.rglob("*.py"):
        if path.stem in forbidden:
            continue
        tree = ast.parse(path.read_text())
        imported = {
            node.module.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported.update(
            alias.name.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        if imported & forbidden:
            violations.append(f"{path.relative_to(_PACKAGE_ROOT)}: {sorted(imported & forbidden)!r}")
    assert not violations, "\n".join(violations)


def test_v1_modules_never_import_v2_implementations() -> None:
    violations: list[str] = []
    for path in _PACKAGE_ROOT.rglob("*_v1.py"):
        tree = ast.parse(path.read_text())
        imported = {
            node.module.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported.update(
            alias.name.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        v2_imports = sorted(
            name
            for name in imported
            if name.endswith("_v2") or name in {"postprocessing_authority", "postprocessing_phase_rendering"}
        )
        private_live_imports = sorted(
            f"{node.module}.{alias.name}"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and not node.module.endswith("._postprocessing_validation")
            for alias in node.names
            if alias.name.startswith("_")
        )
        v2_imports.extend(private_live_imports)
        if v2_imports:
            violations.append(f"{path.relative_to(_PACKAGE_ROOT)}: {v2_imports!r}")
    assert not violations, "\n".join(violations)


def test_authority_facade_preserves_v2_object_identity() -> None:
    facade = importlib.import_module("bspp.orchestration.control.postprocessing_authority")
    owner = importlib.import_module("bspp.orchestration.control.postprocessing_authority_v2")
    assert facade.validate_postprocessing_authority is owner.validate_postprocessing_authority
