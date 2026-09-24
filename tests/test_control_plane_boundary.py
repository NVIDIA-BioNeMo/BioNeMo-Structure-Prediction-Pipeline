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

"""Physical distribution boundary checks."""

from __future__ import annotations

import configparser
import hashlib
import json
import os
import re
import subprocess
import sys
import sysconfig
import textwrap
import tomllib
import zipfile
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner

from bspp.orchestration.contract.runspec_policies import provenance_governed

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SRC = ROOT / "packages" / "orchestration-contract" / "src"
CONTROL_SRC = ROOT / "packages" / "orchestration-control" / "src"
RUNTIME_SRC = ROOT / "packages" / "orchestration-runtime" / "src"
SOURCE_ROOTS = {
    "contract": CONTRACT_SRC,
    "control": CONTROL_SRC,
    "runtime": RUNTIME_SRC,
}
MMSA_BASELINE = "419813dbb5a3949e5e16f289f974d9f95e94bf01"
OPENFOLD_BASELINE = "3864d0eda67e70979b8e48f00ed6a08f9e71c59e"
EXPECTED_CONTRACT_EXPORTS = {
    "DATABASE_CACHE_ROOT",
    "DATABASE_REPLICA_LEASE_TARGET",
    "DATABASE_SOURCE_ROOT",
    "CURRENT_CONTRACT_SCHEMA_VERSION",
    "DATA_PLACEMENT_TOOLS_BY_STAGE",
    "DatabaseAlias",
    "DatabaseAccessPolicy",
    "DatabaseAcceptanceCacheProfile",
    "DatabaseCacheIdentityLockObservation",
    "DatabaseCacheIdentityLockSummary",
    "DatabaseCacheMaintenanceDescriptor",
    "DatabaseCacheMaintenanceEvidence",
    "DatabaseMemberKind",
    "DatabaseCacheMountFacts",
    "DatabaseCacheRemovedEntry",
    "DatabaseCapacityDecision",
    "DatabaseCapacityFallbackResult",
    "DatabaseCapacityGate",
    "DatabaseDirectResult",
    "DatabaseMountDescriptor",
    "DatabasePlacementBranch",
    "DatabasePlacementBranchKind",
    "DatabasePlacementFailureClassification",
    "DatabasePlacementFailureEvidence",
    "DatabasePlacementOutcomeKind",
    "DatabasePlacementResult",
    "DatabasePostScienceEvidence",
    "DatabasePostScienceObservation",
    "DatabasePostScienceObservationFailure",
    "DatabaseProfileStagingSnapshot",
    "DatabaseReplicaColdFailureClassification",
    "DatabaseReplicaColdFailureEvidence",
    "DatabaseReplicaColdResult",
    "DatabaseReplicaCopyEvidence",
    "DatabaseReplicaLeaseAcquisition",
    "DatabaseReplicaLeaseEvidence",
    "DatabaseReplicaLeaseFailureClassification",
    "DatabaseReplicaLeaseFailureEvidence",
    "DatabaseReplicaLeaseTerminalEvidence",
    "DatabaseReplicaManifest",
    "DatabaseReplicaMember",
    "DatabaseReplicaResult",
    "DatabaseReplicaWarmLeaseFailureEvidence",
    "DatabaseReplicaWarmLeaseTerminalEvidence",
    "DatabaseReplicaWarmResult",
    "DatabaseRole",
    "DatabaseRsyncOutcome",
    "DatabaseSetDeclaration",
    "DatabaseSetIdentity",
    "DatabaseSetProvisioningEvidence",
    "DatabaseSetSelection",
    "DatabaseSourceManifest",
    "DatabaseSourceMember",
    "DatabaseSourceMountFacts",
    "DatabaseSourceObservation",
    "DatabaseWarmCacheMountFacts",
    "DeclaredDatabaseMember",
    "DeclaredDatabaseRole",
    "SUPPORTED_CONTRACT_SCHEMA_VERSIONS",
    "ArchiveBatchPlan",
    "ArchiveManifestRecord",
    "ArchiveMember",
    "ArchivePlanOptions",
    "CheckpointLayout",
    "DataMoverSelectionSpec",
    "DataPlacementRecord",
    "DataPlacementStage",
    "DataPlacementTool",
    "FailureDisposition",
    "FoldingArchiveLifecycleDecision",
    "FoldingArchiveLifecycleRequest",
    "FoldingArchivePlan",
    "FoldingCheckpointInputs",
    "FoldingCheckpointState",
    "FoldingCompletionRecord",
    "FoldingFailureRecord",
    "FoldingIndex",
    "FoldingIndexLifecycleDecision",
    "FoldingIndexLifecycleRequest",
    "FoldingIndexRecord",
    "FoldingLifecycleConfig",
    "FoldingLifecycleState",
    "FoldingManifestEvidence",
    "FoldingMergeLifecycleDecision",
    "FoldingMergeLifecycleRequest",
    "FoldingPreflightCheck",
    "FoldingPreflightLifecycleDecision",
    "FoldingPreflightLifecycleRequest",
    "FoldingQueueArtifact",
    "FoldingQueueAssignment",
    "FoldingQueueConfig",
    "FoldingQueueLayout",
    "FoldingQueuePlan",
    "FoldingQueueRenderResult",
    "FoldingQueueStrategy",
    "FoldingQueueWriteDeclaration",
    "FoldingRemainingWorkPlan",
    "FoldingResultAssociation",
    "FoldingResultInventory",
    "FoldingResumeLifecycleDecision",
    "FoldingResumeLifecycleRequest",
    "FoldingStatusEvidenceKind",
    "FoldingStatusLifecycleDecision",
    "FoldingStatusLifecycleRequest",
    "FoldingSubmitLifecycleDecision",
    "FoldingSubmitLifecycleRequest",
    "OperatorTransferEvidence",
    "OperatorTransferItem",
    "OperatorTransferPlan",
    "PhaseActionDispatchIntendedEvent",
    "PhaseActionDispatchIntendedPayload",
    "PhaseActionDispatchRejectedEvent",
    "PhaseActionDispatchRejectedPayload",
    "PhaseActionSubmissionPlan",
    "PhaseActionSubmissionView",
    "PhaseActionSubmittedEvent",
    "PhaseActionSubmittedPayload",
    "PhaseSubmissionIntendedEvent",
    "PhaseSubmissionIntendedPayload",
    "PhaseSubmissionLifecycleView",
    "PreflightAction",
    "PreflightName",
    "PreflightStatus",
    "PreexistingChecksum",
    "PreprocessingRawSearchArtifact",
    "PreprocessingRawSearchEvidence",
    "PreprocessingDatabasePlacementCommandFailureClassification",
    "PreprocessingDatabasePlacementCommandFailureEvidence",
    "PreprocessingDatabasePlacementEvidence",
    "PreprocessingStagedDatabasePlacementEvidence",
    "PreprocessingDatabaseBinding",
    "PREPROCESSING_ADAPTER_VERSION",
    "PREPROCESSING_RUNTIME_COMMAND",
    "PREPROCESSING_RUNTIME_CONTRACT_ID",
    "PreprocessingRuntimeGpuEvidence",
    "PreprocessingRuntimeImageEvidence",
    "PreprocessingRuntimeImageIdentity",
    "PreprocessingRuntimeQualificationRecord",
    "PreprocessingRuntimeQualificationTuple",
    "PreprocessingRuntimeSmokeEvidence",
    "PreprocessingRuntimeSourceEvidence",
    "PreprocessingRuntimeToolEvidence",
    "ProvisioningPlan",
    "ProvisioningDisposition",
    "QualifiedPreprocessingRuntimeSelection",
    "ResolvedSecret",
    "ResultKind",
    "ResultStatus",
    "RunDataPlacementSpec",
    "RunPlan",
    "RunSpec",
    "SELECTED_DATABASE_ROOT",
    "RuntimeImageCacheCheckRecord",
    "RuntimeImagePlan",
    "RuntimeImageProvisionRecord",
    "RuntimeQualificationCheck",
    "RuntimeQualificationRecord",
    "SecretRef",
    "SeamTransferDecision",
    "SourceBundleArchiveVerification",
    "SourceBundleBuildRecord",
    "SourceBundleManifestEntry",
    "SourceBundlePlan",
    "SourceBundleStageRecord",
    "SourcePackageIdentity",
    "SourcePackageManifestEntry",
    "StaticValidationError",
    "StaticValidationIssue",
    "StaticValidationResult",
    "UnmatchedFoldingResult",
    "UnmatchedReason",
    "UnsupportedSchemaVersionError",
    "archive_batch_plan_from_mapping",
    "archive_manifest_record_from_mapping",
    "archive_member_from_mapping",
    "archive_plan_from_mapping",
    "archive_plan_options_from_mapping",
    "build_manual_transfer_plan",
    "build_plan_referenced_transfer_plan",
    "build_preprocessing_database_binding",
    "canonical_database_cache_maintenance_evidence_bytes",
    "canonical_database_cache_maintenance_profile_bytes",
    "canonical_database_capacity_fallback_result_bytes",
    "canonical_database_direct_result_bytes",
    "canonical_database_placement_failure_evidence_bytes",
    "canonical_database_placement_result_bytes",
    "canonical_database_replica_cold_failure_evidence_bytes",
    "canonical_database_replica_cold_result_bytes",
    "canonical_database_replica_manifest_bytes",
    "canonical_database_replica_result_bytes",
    "canonical_database_replica_warm_result_bytes",
    "canonical_database_source_manifest_bytes",
    "database_branch_kinds",
    "database_cache_maintenance_evidence_from_mapping",
    "database_cache_identity_lock_observations_digest",
    "database_capacity_fallback_result_digest",
    "database_capacity_fallback_result_from_mapping",
    "database_direct_result_digest",
    "database_direct_result_from_mapping",
    "database_mount_descriptor_from_mapping",
    "database_placement_branch_from_mapping",
    "database_placement_failure_evidence_digest",
    "database_placement_failure_evidence_from_mapping",
    "database_placement_result_digest",
    "database_placement_result_from_mapping",
    "database_replica_cold_failure_evidence_digest",
    "database_replica_cold_failure_evidence_from_mapping",
    "database_replica_cold_result_digest",
    "database_replica_cold_result_from_mapping",
    "database_replica_manifest_digest",
    "database_replica_manifest_from_mapping",
    "database_replica_lease_evidence_from_mapping",
    "database_replica_result_digest",
    "database_replica_result_from_mapping",
    "database_replica_warm_result_digest",
    "database_replica_warm_result_from_mapping",
    "database_post_science_evidence_from_mapping",
    "database_post_science_observation_from_mapping",
    "database_post_science_observation_failure_from_mapping",
    "database_profile_staging_snapshot_from_mapping",
    "database_set_declaration_from_mapping",
    "database_set_provisioning_evidence_from_mapping",
    "database_source_manifest_digest",
    "database_source_manifest_from_mapping",
    "database_source_manifest_names",
    "database_set_selection_from_mapping",
    "database_warm_cache_mount_facts_from_mapping",
    "empty_database_cache_entries_digest",
    "preprocessing_database_binding_from_mapping",
    "data_placement_record_from_mapping",
    "folding_archive_lifecycle_decision_from_mapping",
    "folding_archive_lifecycle_request_from_mapping",
    "folding_checkpoint_inputs_from_mapping",
    "folding_checkpoint_state_from_mapping",
    "folding_completion_record_from_mapping",
    "folding_failure_record_from_mapping",
    "folding_index_from_mapping",
    "folding_index_lifecycle_decision_from_mapping",
    "folding_index_lifecycle_request_from_mapping",
    "folding_index_record_from_mapping",
    "folding_lifecycle_config_from_mapping",
    "folding_manifest_evidence_from_mapping",
    "folding_merge_lifecycle_decision_from_mapping",
    "folding_merge_lifecycle_request_from_mapping",
    "folding_preflight_check_from_mapping",
    "folding_preflight_lifecycle_decision_from_mapping",
    "folding_preflight_lifecycle_request_from_mapping",
    "folding_queue_artifact_from_mapping",
    "folding_queue_assignment_from_mapping",
    "folding_queue_config_from_mapping",
    "folding_queue_plan_from_mapping",
    "folding_queue_render_result_from_mapping",
    "folding_queue_write_declaration_from_mapping",
    "folding_remaining_work_plan_from_mapping",
    "folding_result_association_from_mapping",
    "folding_result_inventory_from_mapping",
    "folding_resume_lifecycle_decision_from_mapping",
    "folding_resume_lifecycle_request_from_mapping",
    "folding_status_lifecycle_decision_from_mapping",
    "folding_status_lifecycle_request_from_mapping",
    "folding_submit_lifecycle_decision_from_mapping",
    "folding_submit_lifecycle_request_from_mapping",
    "load_runplan",
    "load_runspec",
    "load_database_cache_maintenance_evidence",
    "load_database_set_declaration",
    "make_folding_index",
    "phase_action_dispatch_intended_event_from_mapping",
    "phase_action_dispatch_intended_payload_from_mapping",
    "phase_action_dispatch_rejected_event_from_mapping",
    "phase_action_dispatch_rejected_payload_from_mapping",
    "phase_action_scheduler_correlation_token",
    "phase_action_submission_identity_mapping",
    "phase_action_submission_plan_from_mapping",
    "phase_action_submission_view_from_mapping",
    "phase_action_submitted_event_from_mapping",
    "phase_action_submitted_payload_from_mapping",
    "phase_submission_id",
    "phase_submission_intended_event_from_mapping",
    "phase_submission_intended_payload_from_mapping",
    "phase_submission_lifecycle_view_from_mapping",
    "preprocessing_chunk_action_evidence_from_mapping",
    "preprocessing_runtime_gpu_evidence_from_mapping",
    "preprocessing_runtime_image_evidence_from_mapping",
    "preprocessing_runtime_image_identity_from_mapping",
    "preprocessing_runtime_qualification_record_from_mapping",
    "preprocessing_runtime_qualification_tuple_from_mapping",
    "preprocessing_runtime_smoke_evidence_from_mapping",
    "preprocessing_runtime_source_evidence_from_mapping",
    "preprocessing_runtime_tool_evidence_from_mapping",
    "preprocessing_runtime_tuple_id",
    "preprocessing_raw_search_artifact_from_mapping",
    "preprocessing_raw_search_evidence_from_mapping",
    "provisioning_plan_from_mapping",
    "qualified_preprocessing_runtime_selection_from_mapping",
    "reject_profile_inheritance",
    "removed_database_cache_entries_digest",
    "render_dry_run",
    "resolve_execution_tool",
    "resolve_seam_transfer_decision",
    "resolve_transfer_destination",
    "runspec_from_mapping",
    "runtime_image_plan_from_mapping",
    "seam_transfer_decision_from_mapping",
    "select_database_acceptance_cache_profile",
    "source_bundle_manifest_digest",
    "source_bundle_plan_from_mapping",
    "source_package_identity_from_mapping",
    "source_package_identity_from_source_bundle",
    "unmatched_folding_result_from_mapping",
    "validate_active_workflow_static",
    "validate_data_placement_tool_for_stage",
    "validate_schema_version",
    "verify_source_package",
    # Stage-10 handoff-contract re-exports (e03s01-e03s03).
    "A3mSplitRules",
    "BioIRPairingMode",
    "BioIRPolymer",
    "BioIRRequestManifest",
    "CANONICAL_META_SUFFIX",
    "CANONICAL_MODEL_SUFFIX",
    "FoldingInputLayout",
    "KNOWN_SUFFIXES",
    "LayoutKind",
    "MASTER_PARQUET_PROJECTION",
    "MAX_PAIRS_PER_ARCHIVE",
    "MAX_PROTEINS_PER_SHARD",
    "MasterParquetColumn",
    "MasterParquetDtype",
    "MsaSetConsumption",
    "OpenFoldPairingMode",
    "PredictionArchiveBundle",
    "PredictionPair",
    "PredictionScoresPayload",
    "TemplateMode",
    "a3m_split_rules_from_mapping",
    "bioir_polymer_from_mapping",
    "bioir_request_manifest_from_mapping",
    "folding_input_layout_from_mapping",
    "is_compound_model_entity_id",
    "is_homodimer_model_entity_id",
    "is_human_string_model_entity_id",
    "is_pdb_assembly_model_entity_id",
    "master_parquet_column_from_mapping",
    "msa_set_consumption_from_mapping",
    "normalize_model_entity_id",
    "operator_transfer_evidence_from_json",
    "operator_transfer_evidence_from_mapping",
    "operator_transfer_evidence_id",
    "operator_transfer_item_from_mapping",
    "operator_transfer_plan_from_json",
    "operator_transfer_plan_from_mapping",
    "parse_compound_model_entity_id",
    "parse_human_string_model_entity_id",
    "parse_merged_a3m_header",
    "prediction_archive_bundle_from_mapping",
    "prediction_pair_from_mapping",
    "prediction_scores_payload_from_mapping",
    "split_boundaries",
    "validate_projection_columns",
}


def test_provenance_governance_is_centralized_by_run_kind() -> None:
    assert provenance_governed("dev") is False
    assert provenance_governed("canary") is True
    assert provenance_governed("production") is True


RUNTIME_HEAVY_DEPENDENCIES = {
    "afdb-toolkit",
    "duckdb",
    "google-cloud-storage",
    "numpy",
    "orjson",
    "pyarrow",
    "rich",
    "submitit",
    "zstandard",
}
RUNTIME_HEAVY_IMPORTS = {
    "afdb_integration_kit",
    "boto3",
    "botocore",
    "duckdb",
    "google",
    "numpy",
    "orjson",
    "pyarrow",
    "rich",
    "submitit",
    "torch",
    "zstandard",
}
FORBIDDEN_RUNTIME_MODULES = {
    "bspp.orchestration.runtime",
}
OLD_DIRECT_RUNTIME_MODULES = (
    "bspp.orchestration.cli",
    "bspp.orchestration.cluster",
    "bspp.orchestration.config",
    "bspp.orchestration.constants",
    "bspp.orchestration.data_movement",
    "bspp.orchestration.discovery",
    "bspp.orchestration.extraction",
    "bspp.orchestration.inputs",
    "bspp.orchestration.postprocessing",
    "bspp.orchestration.runspec_preflight",
    "bspp.orchestration.runspec_runner",
    "bspp.orchestration.slurm",
    "bspp.orchestration.status",
    "bspp.orchestration.validation",
    "bspp.orchestration.worker",
)
NEW_RUNTIME_MODULES = (
    "bspp.orchestration.runtime",
    "bspp.orchestration.runtime.cli",
    "bspp.orchestration.runtime.cluster",
    "bspp.orchestration.runtime.config",
    "bspp.orchestration.runtime.constants",
    "bspp.orchestration.runtime.contrib",
    "bspp.orchestration.runtime.data",
    "bspp.orchestration.runtime.data_movement",
    "bspp.orchestration.runtime.discovery",
    "bspp.orchestration.runtime.extraction",
    "bspp.orchestration.runtime.folding",
    "bspp.orchestration.runtime.inputs",
    "bspp.orchestration.runtime.postprocessing",
    "bspp.orchestration.runtime.preprocessing",
    "bspp.orchestration.runtime.runspec_preflight",
    "bspp.orchestration.runtime.slurm",
    "bspp.orchestration.runtime.status",
    "bspp.orchestration.runtime.toolkit",
    "bspp.orchestration.runtime.validation",
    "bspp.orchestration.runtime.worker",
)


def test_control_plane_and_contract_imports_do_not_import_runtime_heavy_modules() -> None:
    source = """
        import importlib
        import json
        import pkgutil
        import sys

        import bspp.orchestration.contract
        import bspp.orchestration.control

        def import_surface(package):
            for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                importlib.import_module(module.name)

        import_surface(bspp.orchestration.contract)
        import_surface(bspp.orchestration.control)

        expected_contract_exports = __EXPECTED_CONTRACT_EXPORTS__
        export_drift = sorted(expected_contract_exports ^ set(bspp.orchestration.contract.__all__))
        missing_exports = sorted(
            name for name in expected_contract_exports if not hasattr(bspp.orchestration.contract, name)
        )

        heavy_modules = {
            "afdb_integration_kit",
            "boto3",
            "botocore",
            "duckdb",
            "google",
            "numpy",
            "orjson",
            "pyarrow",
            "rich",
            "submitit",
            "torch",
            "zstandard",
        }
        loaded = sorted(name for name in heavy_modules if name in sys.modules)
        print(json.dumps({"export_drift": export_drift, "loaded": loaded, "missing_exports": missing_exports}))
        """
    result = _run_python(
        source.replace("__EXPECTED_CONTRACT_EXPORTS__", repr(EXPECTED_CONTRACT_EXPORTS)),
        CONTRACT_SRC,
        CONTROL_SRC,
    )

    assert result.stdout.strip() == '{"export_drift": [], "loaded": [], "missing_exports": []}', result.stderr


def test_control_plane_and_contract_imports_are_safe_when_runtime_dependencies_are_unavailable() -> None:
    result = _run_python(
        """
        import importlib
        import importlib.abc
        import json
        import pkgutil
        import sys

        blocked = {
            "afdb_integration_kit",
            "boto3",
            "botocore",
            "duckdb",
            "google",
            "numpy",
            "orjson",
            "pyarrow",
            "rich",
            "submitit",
            "torch",
            "zstandard",
        }

        class BlockRuntimeHeavyImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", maxsplit=1)[0] in blocked:
                    raise ImportError(f"blocked runtime-heavy import: {fullname}")
                return None

        sys.meta_path.insert(0, BlockRuntimeHeavyImports())

        import bspp.orchestration.contract
        import bspp.orchestration.control

        def import_surface(package):
            for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                importlib.import_module(module.name)

        import_surface(bspp.orchestration.contract)
        import_surface(bspp.orchestration.control)

        try:
            importlib.import_module("pyarrow")
        except ImportError as exc:
            runtime_import_error = str(exc)
        else:
            runtime_import_error = ""

        print(json.dumps({"runtime_import_error": runtime_import_error}))
        """,
        CONTRACT_SRC,
        CONTROL_SRC,
    )

    payload = result.stdout.strip()
    assert result.returncode == 0, result.stderr
    assert "blocked runtime-heavy import:" in payload


def test_bsppctl_help_exposes_control_commands_without_runtime_package_imports() -> None:
    result = _run_python(
        """
        import importlib.abc
        import json
        import sys

        from click.testing import CliRunner

        blocked = {
            "bspp.orchestration.runtime",
        }

        class BlockRuntimePackageImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == module or fullname.startswith(f"{module}.") for module in blocked):
                    raise ImportError(f"blocked runtime-package import: {fullname}")
                return None

        sys.meta_path.insert(0, BlockRuntimePackageImports())

        from bspp.orchestration.control.cli import cli

        runner = CliRunner()
        outputs = {
            "top": runner.invoke(cli, ["--help"]),
            "phase": runner.invoke(cli, ["phase", "--help"]),
            "runtime": runner.invoke(cli, ["runtime", "--help"]),
            "release": runner.invoke(cli, ["release", "--help"]),
            "prepare-benchmark": runner.invoke(cli, ["prepare-benchmark", "--help"]),
            "validate-run": runner.invoke(cli, ["validate-run", "--help"]),
            "run": runner.invoke(cli, ["run", "--help"]),

            "plan": runner.invoke(cli, ["plan", "--help"]),
            "provision": runner.invoke(cli, ["provision", "--help"]),
            "data": runner.invoke(cli, ["data", "--help"]),
        }
        loaded = sorted(name for name in blocked if name in sys.modules)
        print(
            json.dumps(
                {
                    "exit_codes": {name: result.exit_code for name, result in outputs.items()},
                    "loaded": loaded,
                    "top": outputs["top"].output,
                    "phase": outputs["phase"].output,
                    "runtime": outputs["runtime"].output,
                    "data": outputs["data"].output,
                    "release": outputs["release"].output,
                    "run": outputs["run"].output,
                    "plan": outputs["plan"].output,
                    "provision": outputs["provision"].output,
                },
                sort_keys=True,
            )
        )
        """,
        CONTRACT_SRC,
        CONTROL_SRC,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["exit_codes"] == {
        "data": 0,
        "phase": 0,
        "plan": 2,
        "prepare-benchmark": 0,
        "provision": 2,
        "release": 0,
        "run": 2,
        "runtime": 0,
        "top": 0,
        "validate-run": 0,
    }
    assert payload["loaded"] == []
    assert "phase" in payload["top"]
    assert "runtime" in payload["top"]
    assert "release" in payload["top"]

    assert "prepare-benchmark" in payload["top"]
    assert "validate-run" in payload["top"]
    assert "\n  plan " not in payload["top"]
    assert "\n  provision " not in payload["top"]
    assert "\n  run " not in payload["top"]
    assert "data" in payload["top"]
    assert "materialize" in payload["phase"]
    assert "submit" in payload["phase"]
    assert "finalize" in payload["phase"]
    assert "qualify" in payload["runtime"]
    assert "approve-publication" in payload["release"]

    assert "No such command" in payload["run"]
    assert "No such command" in payload["plan"]
    assert "No such command" in payload["provision"]
    assert "plan" in payload["data"]


def test_control_plane_help_uses_control_plane_entrypoint() -> None:
    from bspp.orchestration.control.cli import cli

    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Usage: bsppctl" in result.output
    assert "BSPP Control Plane" in result.output


def test_project_dependency_sets_preserve_physical_install_boundaries() -> None:
    root = tomllib.loads((ROOT / "pyproject.toml").read_text())
    contract = _project("orchestration-contract")
    control = _project("orchestration-control")
    runtime = _project("orchestration-runtime")

    assert "project" not in root
    assert sorted(root["tool"]["uv"]["workspace"]["members"]) == [
        "packages/orchestration-contract",
        "packages/orchestration-control",
        "packages/orchestration-runtime",
    ]
    assert contract["name"] == "bspp-orchestration-contract"
    assert set(_dependency_names(contract["dependencies"])) == {"pydantic", "pyyaml"}
    assert control["name"] == "bspp-orchestration-control"
    assert set(_dependency_names(control["dependencies"])) == {
        "bspp-orchestration-contract",
        "click",
        "pydantic",
        "pyyaml",
        "tabulate",
    }
    assert "bspp-orchestration-runtime" not in set(_dependency_names(control["dependencies"]))
    assert set(_dependency_names(control["dependencies"])).isdisjoint(RUNTIME_HEAVY_DEPENDENCIES)
    assert runtime["name"] == "bspp-orchestration-runtime"
    runtime_dependencies = set(_dependency_names(runtime["dependencies"]))
    assert "bspp-orchestration-contract" in runtime_dependencies
    assert "bspp-orchestration-control" not in runtime_dependencies
    assert runtime_dependencies >= (RUNTIME_HEAVY_DEPENDENCIES - {"afdb-toolkit"})
    assert "afdb-toolkit" not in runtime_dependencies
    assert "scripts" not in contract
    assert control["scripts"] == {"bsppctl": "bspp.orchestration.control.cli:cli"}
    assert runtime["scripts"] == {"bspp-orchestration-runtime": "bspp.orchestration.runtime.cli:cli"}


def test_declared_package_surfaces_make_dependency_boundaries_enforceable() -> None:
    from tests.support.package_boundaries import (
        CONTROL_PACKAGE_IMPORTS,
        PACKAGE_SURFACES,
        check_boundary_imports,
    )
    from tests.support.package_boundaries import (
        RUNTIME_HEAVY_IMPORTS as DECLARED_RUNTIME_HEAVY_IMPORTS,
    )

    assert set(PACKAGE_SURFACES) == {"contract", "control", "runtime"}
    assert set(DECLARED_RUNTIME_HEAVY_IMPORTS) == RUNTIME_HEAVY_IMPORTS
    assert PACKAGE_SURFACES["contract"].distribution_name == "bspp-orchestration-contract"
    assert PACKAGE_SURFACES["control"].distribution_name == "bspp-orchestration-control"
    assert PACKAGE_SURFACES["runtime"].distribution_name == "bspp-orchestration-runtime"
    assert PACKAGE_SURFACES["contract"].import_roots == (
        "bspp.orchestration.contract",
        "bspp.orchestration.contract.config_models",
        "bspp.orchestration.contract.control_state",
        "bspp.orchestration.contract.data_placement",
        "bspp.orchestration.contract.database_set_provisioning",
        "bspp.orchestration.contract.phase",
        "bspp.orchestration.contract.phase_state",
        "bspp.orchestration.contract.provisioning",
        "bspp.orchestration.contract.runplan",
        "bspp.orchestration.contract.runspec",
        "bspp.orchestration.contract.runspec_policies",
        "bspp.orchestration.contract.runspec_validation",
        "bspp.orchestration.contract.runtime_qualification",
        "bspp.orchestration.contract.secrets",
        "bspp.orchestration.contract.versioning",
    )
    assert PACKAGE_SURFACES["runtime"].import_roots == ("bspp.orchestration.runtime",)
    assert set(PACKAGE_SURFACES["contract"].forbidden_imports) >= FORBIDDEN_RUNTIME_MODULES
    assert set(PACKAGE_SURFACES["control"].forbidden_imports) >= FORBIDDEN_RUNTIME_MODULES
    assert set(PACKAGE_SURFACES["runtime"].forbidden_imports) >= set(CONTROL_PACKAGE_IMPORTS)

    violations = check_boundary_imports(SOURCE_ROOTS)

    assert violations == []


def test_declared_package_surfaces_cover_top_level_runtime_modules() -> None:
    from tests.support.package_boundaries import PACKAGE_SURFACES

    package_root = RUNTIME_SRC / "bspp" / "orchestration" / "runtime"
    top_level_modules = {
        "bspp.orchestration.runtime"
        if path.name == "__init__.py"
        else f"bspp.orchestration.runtime.{path.stem}"
        if path.is_file()
        else f"bspp.orchestration.runtime.{path.name}"
        for path in package_root.iterdir()
        if path.name != "__pycache__"
    }
    expected_top_level_modules = set(NEW_RUNTIME_MODULES)
    declared_roots = PACKAGE_SURFACES["runtime"].import_roots

    assert top_level_modules == expected_top_level_modules
    assert declared_roots == ("bspp.orchestration.runtime",)
    assert all(
        module == "bspp.orchestration.runtime" or module.startswith("bspp.orchestration.runtime.")
        for module in top_level_modules
    )


def test_runtime_source_tree_is_under_runtime_package() -> None:
    package_root = RUNTIME_SRC / "bspp" / "orchestration"

    assert sorted(path.name for path in package_root.iterdir() if path.name != "__pycache__") == ["runtime"]


def test_runtime_pythonpath_resolves_new_roots_and_not_old_direct_roots() -> None:
    result = _run_python(
        f"""
        import importlib.util
        import json

        old_roots = {OLD_DIRECT_RUNTIME_MODULES!r}
        new_roots = {NEW_RUNTIME_MODULES!r}

        print(
            json.dumps(
                {{
                    "old_present": sorted(name for name in old_roots if importlib.util.find_spec(name) is not None),
                    "new_missing": sorted(name for name in new_roots if importlib.util.find_spec(name) is None),
                }},
                sort_keys=True,
            )
        )
        """,
        RUNTIME_SRC,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"new_missing": [], "old_present": []}


def test_workspace_namespace_imports_merge_across_package_roots() -> None:
    result = _run_python(
        """
        import json

        import bspp
        import bspp.orchestration
        import bspp.orchestration.runtime.cli
        import bspp.orchestration.contract
        import bspp.orchestration.control

        print(
            json.dumps(
                {
                    "bspp_is_namespace": getattr(bspp, "__file__", None) is None,
                    "bspp_paths": sorted(str(path) for path in bspp.__path__),
                    "orchestration_is_namespace": getattr(bspp.orchestration, "__file__", None) is None,
                    "orchestration_paths": sorted(str(path) for path in bspp.orchestration.__path__),
                },
                sort_keys=True,
            )
        )
        """,
        CONTRACT_SRC,
        CONTROL_SRC,
        RUNTIME_SRC,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["bspp_is_namespace"] is True
    assert payload["orchestration_is_namespace"] is True
    assert payload["bspp_paths"] == [
        str(CONTRACT_SRC / "bspp"),
        str(CONTROL_SRC / "bspp"),
        str(RUNTIME_SRC / "bspp"),
    ]
    assert payload["orchestration_paths"] == [
        str(CONTRACT_SRC / "bspp" / "orchestration"),
        str(CONTROL_SRC / "bspp" / "orchestration"),
        str(RUNTIME_SRC / "bspp" / "orchestration"),
    ]


def test_old_underscore_namespace_is_not_importable() -> None:
    result = _run_python(
        """
        import importlib.util
        import json

        old_namespace = "bspp" + "_orchestration"
        print(json.dumps({"old_namespace_present": importlib.util.find_spec(old_namespace) is not None}))
        """,
        CONTRACT_SRC,
        CONTROL_SRC,
        RUNTIME_SRC,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"old_namespace_present": False}


def test_control_only_pythonpath_cannot_import_runtime_modules() -> None:
    result = _run_python(
        """
        import importlib.util
        import json

        import bspp.orchestration.contract
        import bspp.orchestration.control

        def module_present(name):
            try:
                return importlib.util.find_spec(name) is not None
            except ModuleNotFoundError:
                return False

        print(
            json.dumps(
                {
                    "runtime_cli_present": module_present("bspp.orchestration.runtime.cli"),
                    "runtime_package_present": module_present("bspp.orchestration.runtime.worker"),
                },
                sort_keys=True,
            )
        )
        """,
        CONTRACT_SRC,
        CONTROL_SRC,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"runtime_cli_present": False, "runtime_package_present": False}


def test_runtime_pythonpath_cannot_import_control_modules() -> None:
    result = _run_python(
        """
        import importlib.util
        import json

        import bspp.orchestration.runtime.cli
        import bspp.orchestration.contract

        print(
            json.dumps(
                {
                    "control_present": importlib.util.find_spec("bspp.orchestration.control") is not None,
                },
                sort_keys=True,
            )
        )
        """,
        CONTRACT_SRC,
        RUNTIME_SRC,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"control_present": False}


@pytest.fixture(scope="module")
def _built_wheel_installations(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    build_root = tmp_path_factory.mktemp("isolated-wheels")
    dist = build_root / "dist"
    dist.mkdir()
    result = subprocess.run(
        ["uv", "build", "--offline", "--all-packages", "--out-dir", str(dist)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    wheels = {path.name.split("-0.1.0-", maxsplit=1)[0].replace("_", "-"): path for path in dist.glob("*.whl")}
    assert set(wheels) == {
        "bspp-orchestration-contract",
        "bspp-orchestration-control",
        "bspp-orchestration-runtime",
    }
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in wheels.values()}
    print("WHEEL_SHA256 " + json.dumps(hashes, sort_keys=True))

    target_root = build_root / "targets"
    target_root.mkdir()
    matrices = {
        "contract": (wheels["bspp-orchestration-contract"],),
        "control": (wheels["bspp-orchestration-contract"], wheels["bspp-orchestration-control"]),
        "runtime": (wheels["bspp-orchestration-contract"], wheels["bspp-orchestration-runtime"]),
    }
    targets: dict[str, Path] = {}
    for name, selected_wheels in matrices.items():
        target = target_root / name
        install = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(target),
                "--no-deps",
                "--no-index",
                "--offline",
                *(str(path) for path in selected_wheels),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert install.returncode == 0, install.stderr
        targets[name] = target

    probe_cwd = build_root / "probe-cwd"
    probe_cwd.mkdir()
    return {
        "hashes": hashes,
        "probe_cwd": probe_cwd,
        "targets": targets,
        "wheels": wheels,
    }


def test_built_wheels_contain_only_owned_surfaces(_built_wheel_installations: dict[str, object]) -> None:
    wheels = _built_wheel_installations["wheels"]
    hashes = _built_wheel_installations["hashes"]
    assert isinstance(wheels, dict)
    assert isinstance(hashes, dict)
    assert set(hashes) == {path.name for path in wheels.values()}
    assert all(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values())

    contract_names = _wheel_names(wheels["bspp-orchestration-contract"])
    control_names = _wheel_names(wheels["bspp-orchestration-control"])
    runtime_names = _wheel_names(wheels["bspp-orchestration-runtime"])

    for names in (contract_names, control_names, runtime_names):
        assert "bspp/__init__.py" not in names
        assert "bspp/orchestration/__init__.py" not in names
        assert not any(".git/" in name for name in names)
        assert not any("consensus/openfold-baseline" in name or "consensus/mmsa-baseline" in name for name in names)

    assert _console_scripts(wheels["bspp-orchestration-contract"]) == {}

    assert _python_modules(contract_names) <= {"bspp/orchestration/contract"}
    assert any(name.startswith("bspp/orchestration/contract/") for name in contract_names)
    assert not any(name.startswith("bspp/orchestration/control/") for name in contract_names)
    assert not any(name.startswith("bspp/orchestration/worker/") for name in contract_names)
    assert not any(name.startswith("bspp/orchestration/runtime/") for name in contract_names)

    assert any(name.startswith("bspp/orchestration/control/") for name in control_names)
    assert "bspp/orchestration/control/data/cluster_profile_templates.yaml" in control_names
    assert not any(name.startswith("bspp/orchestration/worker/") for name in control_names)
    assert not any(name.startswith("bspp/orchestration/slurm/") for name in control_names)
    assert not any(name.startswith("bspp/orchestration/runtime/") for name in control_names)
    assert _console_scripts(wheels["bspp-orchestration-control"]) == {"bsppctl": "bspp.orchestration.control.cli:cli"}

    assert all(
        name.startswith("bspp/orchestration/runtime/")
        for name in runtime_names
        if name.startswith("bspp/orchestration/") and name.endswith(".py")
    )
    assert any(name.startswith("bspp/orchestration/runtime/worker/") for name in runtime_names)
    assert any(name.startswith("bspp/orchestration/runtime/slurm/") for name in runtime_names)
    assert "bspp/orchestration/runtime/folding/__init__.py" in runtime_names
    assert "bspp/orchestration/runtime/preprocessing/__init__.py" in runtime_names
    assert "bspp/orchestration/runtime/data/clusters.yaml" in runtime_names
    assert "bspp/orchestration/runtime/data/recipe_template.yaml" in runtime_names
    for path in (
        "bspp/orchestration/cli.py",
        "bspp/orchestration/cluster.py",
        "bspp/orchestration/config.py",
        "bspp/orchestration/constants.py",
        "bspp/orchestration/discovery.py",
        "bspp/orchestration/runspec_preflight.py",
        "bspp/orchestration/runspec_runner.py",
        "bspp/orchestration/status.py",
    ):
        assert path not in runtime_names
    for package in (
        "data_movement",
        "extraction",
        "inputs",
        "postprocessing",
        "slurm",
        "validation",
        "worker",
    ):
        assert not any(name.startswith(f"bspp/orchestration/{package}/") for name in runtime_names)
    assert not any(name.startswith("bspp/orchestration/control/") for name in runtime_names)
    assert not any(name.startswith("bspp/orchestration/contract/") for name in runtime_names)
    assert _console_scripts(wheels["bspp-orchestration-runtime"]) == {
        "bspp-orchestration-runtime": "bspp.orchestration.runtime.cli:cli"
    }


def test_installed_contract_wheel_is_isolated(_built_wheel_installations: dict[str, object]) -> None:
    payload = _probe_installed_target(_built_wheel_installations, "contract")

    assert payload["forbidden_present"] == []
    assert payload["provenance"] == {"mmsa": True, "openfold": True}


def test_installed_control_wheel_is_isolated_from_runtime(_built_wheel_installations: dict[str, object]) -> None:
    payload = _probe_installed_target(_built_wheel_installations, "control")

    assert payload["forbidden_present"] == []
    assert payload["provenance"] == {"mmsa": True, "openfold": True}


def test_installed_runtime_wheel_is_isolated_from_control(_built_wheel_installations: dict[str, object]) -> None:
    payload = _probe_installed_target(_built_wheel_installations, "runtime")

    assert payload["forbidden_present"] == []
    assert payload["provenance"] == {"mmsa": True, "openfold": True}


def test_tracked_sources_do_not_reference_ignored_review_snapshots() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    forbidden = (
        b".git/" + b"consensus/" + b"openfold-baseline",
        b".git/" + b"consensus/" + b"mmsa-baseline",
    )
    violations = [
        os.fsdecode(relative)
        for relative in tracked
        if relative and any(fragment in (ROOT / os.fsdecode(relative)).read_bytes() for fragment in forbidden)
    ]

    assert violations == []


def _probe_installed_target(installation: dict[str, object], mode: str) -> dict[str, object]:
    targets = installation["targets"]
    probe_cwd = installation["probe_cwd"]
    assert isinstance(targets, dict)
    assert isinstance(probe_cwd, Path)
    target = targets[mode]
    assert isinstance(target, Path)

    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean = subprocess.run(
        [sys.executable, "-S", "-c", "import json, sys; print(json.dumps(sys.path))"],
        cwd=probe_cwd,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert clean.returncode == 0, clean.stderr
    clean_path = json.loads(clean.stdout)
    dependency_roots = tuple(
        dict.fromkeys(str(Path(sysconfig.get_paths()[name]).resolve()) for name in ("purelib", "platlib"))
    )
    interpreter_roots = tuple(
        dict.fromkeys(str(Path(sysconfig.get_paths()[name]).resolve()) for name in ("stdlib", "platstdlib"))
    )
    expected_path = [clean_path[0], str(target), *dependency_roots, *clean_path[1:]]
    expected_path_after = ["/opt/bspp/lib", *expected_path] if mode == "control" else expected_path
    env = clean_env | {
        "BSPP_DEPENDENCY_ROOTS": os.pathsep.join(dependency_roots),
        "BSPP_INTERPRETER_ROOTS": os.pathsep.join(interpreter_roots),
        "BSPP_EXPECTED_PATH": json.dumps(expected_path),
        "BSPP_MMSA_BASELINE": MMSA_BASELINE,
        "BSPP_OPENFOLD_BASELINE": OPENFOLD_BASELINE,
        "BSPP_PROBE_MODE": mode,
        "BSPP_PROJECT_ROOT": str(ROOT.parent.resolve()),
        "BSPP_TARGET": str(target),
        "PYTHONPATH": os.pathsep.join((str(target), *dependency_roots)),
    }
    probe = subprocess.run(
        [sys.executable, "-S", "-c", textwrap.dedent(_INSTALLED_TARGET_PROBE)],
        cwd=probe_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    payload = cast(dict[str, object], json.loads(probe.stdout))
    assert payload["sys_path_before"] == expected_path
    assert payload["sys_path_after"] == expected_path_after
    assert payload["bspp_paths"] == [str(target / "bspp")]
    assert payload["orchestration_paths"] == [str(target / "bspp" / "orchestration")]
    assert payload["bspp_origin_violations"] == []
    assert payload["project_origin_violations"] == []
    assert payload["afdb_integration_kit_loaded"] is False
    return payload


_INSTALLED_TARGET_PROBE = """
import importlib
import importlib.util
import json
import os
import pkgutil
import sys
from pathlib import Path

before = list(sys.path)
mode = os.environ["BSPP_PROBE_MODE"]
target = Path(os.environ["BSPP_TARGET"]).resolve()
dependency_roots = tuple(Path(path).resolve() for path in os.environ["BSPP_DEPENDENCY_ROOTS"].split(os.pathsep))
interpreter_roots = tuple(Path(path).resolve() for path in os.environ["BSPP_INTERPRETER_ROOTS"].split(os.pathsep))
project_root = Path(os.environ["BSPP_PROJECT_ROOT"]).resolve()

import bspp
import bspp.orchestration
import bspp.orchestration.contract

contract_modules = (
    "bspp.orchestration.contract.preprocessing",
    "bspp.orchestration.contract.preprocessing_action",
    "bspp.orchestration.contract.preprocessing_execution",
    "bspp.orchestration.contract.preprocessing_state",
    "bspp.orchestration.contract.folding_archive",
    "bspp.orchestration.contract.folding_checkpoint",
    "bspp.orchestration.contract.folding_index",
    "bspp.orchestration.contract.folding_lifecycle",
    "bspp.orchestration.contract.folding_queue",
)
loaded_contract_modules = tuple(importlib.import_module(name) for name in contract_modules)

if mode == "control":
    control = importlib.import_module("bspp.orchestration.control")
    for module in pkgutil.walk_packages(control.__path__, control.__name__ + "."):
        importlib.import_module(module.name)
elif mode == "runtime":
    importlib.import_module("bspp.orchestration.runtime.preprocessing")
    importlib.import_module("bspp.orchestration.runtime.preprocessing.planning")
    importlib.import_module("bspp.orchestration.runtime.preprocessing.commands")
    execution_module = importlib.import_module("bspp.orchestration.runtime.preprocessing.execution")
    importlib.import_module("bspp.orchestration.runtime.preprocessing.state")
    importlib.import_module("bspp.orchestration.runtime.preprocessing.retry")
    importlib.import_module("bspp.orchestration.runtime.folding")
    if not str(Path(execution_module.__file__).resolve()).endswith("/preprocessing/execution.py"):
        raise AssertionError(f"runtime execution did not load active source: {execution_module.__file__}")

def module_present(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False

forbidden = {
    "contract": ("bspp.orchestration.control", "bspp.orchestration.runtime"),
    "control": ("bspp.orchestration.runtime",),
    "runtime": ("bspp.orchestration.control",),
}[mode]

def below(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True

bspp_origin_violations = []
project_origin_violations = []
for name, module in tuple(sys.modules.items()):
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        continue
    resolved = Path(module_file).resolve()
    if name.startswith("bspp.") and not below(resolved, target):
        bspp_origin_violations.append(f"{name}:{resolved}")
    if below(resolved, project_root) and not below(resolved, target) and not any(
        below(resolved, root) for root in (*dependency_roots, *interpreter_roots)
    ):
        project_origin_violations.append(f"{name}:{resolved}")

if mode == "runtime":
    preprocessing_doc = importlib.import_module("bspp.orchestration.runtime.preprocessing").__doc__ or ""
    folding_doc = importlib.import_module("bspp.orchestration.runtime.folding").__doc__ or ""
else:
    preprocessing_doc = loaded_contract_modules[0].__doc__ or ""
    folding_doc = loaded_contract_modules[4].__doc__ or ""

print(json.dumps({
    "afdb_integration_kit_loaded": "afdb_integration_kit" in sys.modules,
    "bspp_origin_violations": sorted(bspp_origin_violations),
    "bspp_paths": list(bspp.__path__),
    "forbidden_present": sorted(name for name in forbidden if module_present(name)),
    "orchestration_paths": list(bspp.orchestration.__path__),
    "project_origin_violations": sorted(project_origin_violations),
    "provenance": {
        "mmsa": os.environ["BSPP_MMSA_BASELINE"] in preprocessing_doc,
        "openfold": os.environ["BSPP_OPENFOLD_BASELINE"] in folding_doc,
    },
    "sys_path_after": list(sys.path),
    "sys_path_before": before,
}, sort_keys=True))
"""


def _run_python(source: str, *roots: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    dependency_roots = {
        Path(sysconfig.get_paths()["purelib"]),
        Path(sysconfig.get_paths()["platlib"]),
    }
    env["PYTHONPATH"] = os.pathsep.join(str(root) for root in (*roots, *sorted(dependency_roots)))
    return subprocess.run(
        [sys.executable, "-S", "-c", textwrap.dedent(source)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _project(member: str) -> dict[str, object]:
    return tomllib.loads((ROOT / "packages" / member / "pyproject.toml").read_text())["project"]


def _dependency_names(dependencies: list[str]) -> list[str]:
    names: list[str] = []
    for dependency in dependencies:
        name = dependency.split(";", maxsplit=1)[0]
        name = name.split("[", maxsplit=1)[0]
        for separator in ("<", ">", "=", "!", "~", " "):
            name = name.split(separator, maxsplit=1)[0]
        names.append(name.lower().replace("_", "-"))
    return names


def _wheel_names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as wheel:
        return set(wheel.namelist())


def _python_modules(names: set[str]) -> set[str]:
    return {str(Path(name).parent) for name in names if name.startswith("bspp/orchestration/") and name.endswith(".py")}


def _console_scripts(path: Path) -> dict[str, str]:
    names = _wheel_names(path)
    entry_points_path = next((name for name in names if name.endswith(".dist-info/entry_points.txt")), None)
    if entry_points_path is None:
        return {}
    with zipfile.ZipFile(path) as wheel:
        payload = wheel.read(entry_points_path).decode()
    parser = configparser.ConfigParser()
    parser.read_string(payload)
    if not parser.has_section("console_scripts"):
        return {}
    return dict(parser.items("console_scripts"))
