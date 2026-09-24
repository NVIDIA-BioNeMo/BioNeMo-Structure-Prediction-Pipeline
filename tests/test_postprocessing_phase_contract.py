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

"""Focused contract tests for the additive postprocessing Phase family."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bspp.orchestration.contract.phase_postprocessing import (
    LogicalInputEntry,
    PostprocessingAttemptPaths,
    PostprocessingCredentialMountSnapshot,
    PostprocessingExecutionProjection,
    PostprocessingPhaseExecutionIdentity,
    QualifiedPostprocessingRuntimeSelection,
    normalize_slurm_array,
    postprocessing_action_id,
)
from bspp.orchestration.contract.postprocessing_acceptance import (
    PostprocessingAcceptanceAdjudication,
    PostprocessingAcceptancePolicySnapshot,
    PostprocessingBaselineReportBinding,
    PostprocessingCompletionExitContract,
    PostprocessingCrossReportReconciliation,
    PostprocessingRawExitReportOutcome,
    PostprocessingReconciliationResult,
    PostprocessingResidualAllowance,
    PostprocessingResidualCardinality,
)
from bspp.orchestration.contract.runplan import RunPlan
from bspp.orchestration.contract.runspec import (
    AcceptanceSpec,
    AnalysisMetadataSpec,
    DataMoverSelectionSpec,
    DatasetSpec,
    HighQualityFromTarsSpec,
    HqChunkPublicationSpec,
    ObjectStorageSpec,
    ReferenceArtifact,
    ReferenceSpec,
    RunDataPlacementSpec,
    RunSecrets,
    SecretRef,
    ValidationPolicy,
    WorkerSpec,
    WorkflowStepSpec,
)
from bspp.orchestration.control.postprocessing_identity import POSTPROCESSING_FIELD_RULES

RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
ATTEMPT_ID = "attempt-0001"
SHA = "1" * 64


def test_credential_mount_snapshot_contains_only_canonical_locator_authority() -> None:
    snapshot = PostprocessingCredentialMountSnapshot(
        aws_shared_credentials_file="/home/operator/.aws/credentials",
        aws_config_file="/home/operator/.aws/config",
    )

    assert snapshot.to_mapping() == {
        "schema_version": 1,
        "aws_shared_credentials_file": "/home/operator/.aws/credentials",
        "aws_config_file": "/home/operator/.aws/config",
    }
    assert PostprocessingCredentialMountSnapshot(None, None).to_mapping()["aws_config_file"] is None
    with pytest.raises(ValueError, match="both present or both null"):
        PostprocessingCredentialMountSnapshot("/home/operator/.aws/credentials", None)
    for unsafe in ("relative/credentials", "/home/../credentials", "/home/credentials,other"):
        with pytest.raises(ValueError, match="canonical absolute POSIX"):
            PostprocessingCredentialMountSnapshot(unsafe, "/home/operator/.aws/config")
    with pytest.raises(ValueError, match="must be distinct"):
        PostprocessingCredentialMountSnapshot("/credentials", "/credentials")


def test_every_legacy_run_plan_field_has_an_explicit_identity_classification() -> None:
    rules = tuple(rule for rule in POSTPROCESSING_FIELD_RULES if rule.scope == "run-plan")
    assert {rule.pattern[0] for rule in rules if len(rule.pattern) == 1} == set(RunPlan.model_fields)
    component_models = {
        "dataset": DatasetSpec,
        "references": ReferenceSpec,
        "references.artifacts.*": ReferenceArtifact,
        "worker": WorkerSpec,
        "storage": ObjectStorageSpec,
        "data_placement": RunDataPlacementSpec,
        "data_placement.*": DataMoverSelectionSpec,
        "analysis_metadata": AnalysisMetadataSpec,
        "analysis_metadata.high_quality_from_tars": HighQualityFromTarsSpec,
        "analysis_metadata.high_quality_from_tars.publication": HqChunkPublicationSpec,
        "validation": ValidationPolicy,
        "acceptance": AcceptanceSpec,
        "secrets": RunSecrets,
        "secrets.*": SecretRef,
    }
    for component, model in component_models.items():
        prefix = tuple(component.split("."))
        for field_name in model.model_fields:
            concrete = (*prefix, field_name)
            matches = tuple(
                rule
                for rule in rules
                if len(rule.pattern) == len(concrete)
                and all(
                    expected == "*" or expected == actual
                    for expected, actual in zip(rule.pattern, concrete, strict=True)
                )
            )
            assert len(matches) == 1, ".".join(concrete)
    workflow_rules = tuple(rule for rule in POSTPROCESSING_FIELD_RULES if rule.scope == "workflow-step")
    workflow_prefix = ("workflow", "steps", "*")
    assert {
        rule.pattern[-1] for rule in workflow_rules if rule.pattern[:-1] == workflow_prefix and rule.kind == "leaf"
    } == set(WorkflowStepSpec.model_fields)


def test_postprocessing_action_ids_are_exact_and_permanent() -> None:
    assert postprocessing_action_id("preflight") == "postprocessing-01-preflight"
    assert (
        postprocessing_action_id("acceptance-tar-payload-parity") == "postprocessing-06-acceptance-tar-payload-parity"
    )
    assert postprocessing_action_id("acceptance-adjudication") == "postprocessing-09-acceptance-adjudication"


def test_slurm_array_normalization_enumerates_exact_sorted_tasks() -> None:
    assert normalize_slurm_array("853-857:2%02") == ("853-857:2%2", (853, 855, 857))
    with pytest.raises(ValueError, match="duplicate task"):
        normalize_slurm_array("1-3,2")
    with pytest.raises(ValueError, match="100000-task"):
        normalize_slurm_array("0-100000")


def test_authority_declared_logical_input_requires_expected_content_facts() -> None:
    entry = LogicalInputEntry(
        name="baseline",
        verification_kind="authority-declared-content-v1",
        authority="input-inventory.json",
        member_identity="baseline-task853-v1",
        expected_content_sha256=SHA,
        expected_size_bytes=42,
    )
    assert entry.to_mapping()["expected_size_bytes"] == 42
    with pytest.raises(ValueError, match="member identity"):
        replace(entry, member_identity=None)


def test_projection_binds_full_versioned_identity_mapping_and_digest() -> None:
    runtime = _runtime()
    substitutions = PostprocessingAttemptPaths(
        legacy_run_id="task853-run",
        output_dir="/output/task853-run",
        evidence_dir="/output/task853-run/evidence",
        staging_dir="/staging/task853-run",
        object_prefix="s3://bucket/task853-run/",
    )
    identity = PostprocessingPhaseExecutionIdentity(
        phase_plan_digest=SHA,
        logical_input_manifest_digest=SHA,
        scientific_identity_digest=SHA,
        action_semantics_digest=SHA,
        acceptance_semantic_digest=SHA,
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        qualified_runtime_digest=runtime.digest,
        output_namespace="task853",
        substitutions=substitutions,
    )
    projection = PostprocessingExecutionProjection(
        document_location=f"attempts/{ATTEMPT_ID}/legacy-runspec.yaml",
        document_sha256=SHA,
        document_size_bytes=10,
        legacy_schema_version=1,
        phase_identity=identity,
        phase_identity_digest=identity.digest,
    )
    assert projection.to_mapping()["phase_identity"] == identity.to_mapping()
    with pytest.raises(ValueError, match="does not match its mapping"):
        replace(projection, phase_identity_digest="2" * 64)


def test_acceptance_policy_preserves_residual_composition_and_reconciliation() -> None:
    policy = PostprocessingAcceptancePolicySnapshot(
        baseline_id="task853-fixed-fork",
        baseline_version="c8f824d",
        policy_schema="bspp-postprocessing-acceptance",
        policy_version="1",
        residual_allowances=(
            PostprocessingResidualAllowance(
                report="semantic.json",
                json_pointer="/findings/known",
                match_kind="json-pointer-count",
                expected_value="known-residual",
                cardinality_kind="exact",
                required_count=20,
                permitted_min=20,
                permitted_max=20,
            ),
        ),
        completion_exit_contracts=tuple(
            PostprocessingCompletionExitContract(
                step_name=step,
                allowed_raw_exit_codes=(0, 1),
                report_paths=(report,),
                report_schema=schema,
                report_schema_version="1",
                outcome_report_path=report,
                outcome_json_pointer="/ok",
                raw_exit_report_outcomes=(
                    PostprocessingRawExitReportOutcome(raw_exit_code=0, report_ok=True),
                    PostprocessingRawExitReportOutcome(raw_exit_code=1, report_ok=False),
                ),
            )
            for step, report, schema in (
                ("acceptance-tar-payload-parity", "parity.json", "tar-payload-parity"),
                ("acceptance-semantic", "semantic.json", "semantic-acceptance"),
                ("acceptance-verify-evidence", "verification.json", "acceptance-verification"),
            )
        ),
        baseline_report_bindings=(
            PostprocessingBaselineReportBinding(
                report="semantic.json",
                baseline_locator_json_pointer="/baseline_dir",
            ),
        ),
        cross_report_reconciliations=(
            PostprocessingCrossReportReconciliation(
                left_report="semantic.json",
                left_json_pointer="/candidate_run",
                right_report="verification.json",
                right_json_pointer="/candidate_run",
            ),
        ),
    )
    assert policy.policy_id.endswith(policy.semantic_digest)
    with pytest.raises(ValueError, match="result does not match"):
        PostprocessingAcceptanceAdjudication(
            phase_run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
            policy_id=policy.policy_id,
            policy_sha256=SHA,
            capture_digests=(SHA, SHA, SHA),
            non_allowlisted_errors=0,
            residual_cardinalities=(
                PostprocessingResidualCardinality(
                    allowance_id="semantic:/findings/known",
                    observed=19,
                    permitted_min=20,
                    permitted_max=20,
                ),
            ),
            reconciliation_results=(
                PostprocessingReconciliationResult(reconciliation_id="candidate-run", matched=True),
            ),
            adjudicated_at="2026-09-03T00:00:00Z",
            result="passed",
        )


def _runtime() -> QualifiedPostprocessingRuntimeSelection:
    return QualifiedPostprocessingRuntimeSelection(
        tuple_id=SHA,
        qualification_location=f"attempts/{ATTEMPT_ID}/runtime-qualification.json",
        qualification_sha256=SHA,
        qualification_size_bytes=10,
        qualified_at="2026-09-02T00:00:00Z",
        expires_at="2026-09-09T00:00:00Z",
        image_path="/images/bspp.sqsh",
        image_sha256=SHA,
        image_size_bytes=1024,
        image_policy="digest-checked",
        source_kind="override",
        source_revision="a" * 40,
        source_package_path="/packages/orchestration.tar",
        toolkit_package_path="/packages/toolkit.tar",
        runtime_ipsae_binary_path="/runtime/ipsae.py",
        runtime_ipsae_binary_sha256=SHA,
        runtime_ipsae_binary_size_bytes=1024,
        source_identity_digest=SHA,
        source_package_identity_digest=SHA,
        toolkit_identity_digest=SHA,
        bootstrap_sha256=SHA,
        runtime_component_identity_digest=SHA,
    )
