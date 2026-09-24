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

"""Public #90 contracts for direct Database Placement action evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from bspp.orchestration.contract.database_placement import (
    DATABASE_SOURCE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
)
from bspp.orchestration.contract.database_placement_result import (
    DatabasePlacementFailureEvidence,
    DatabasePlacementResult,
    DatabasePostScienceObservation,
    DatabasePostScienceObservationFailure,
    DatabaseSourceMountFacts,
    DatabaseSourceObservation,
    canonical_database_placement_failure_evidence_bytes,
    database_placement_failure_evidence_from_mapping,
    database_placement_result_digest,
)
from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseSetIdentity,
    DatabaseSourceMember,
)
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingDatabasePlacementCommandFailureEvidence,
    PreprocessingDatabasePlacementEvidence,
)


def _failure() -> DatabasePlacementFailureEvidence:
    return DatabasePlacementFailureEvidence(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        phase_runspec_digest="a" * 64,
        action_id="preprocessing-chunk-000000",
        database_set=DatabaseSetIdentity(identifier="bspp-search", version="2026-08"),
        requested_policy=DatabaseAccessPolicy.DIRECT,
        source_manifest_sha256="b" * 64,
        source_mount=None,
        science_started=False,
        classification="source-manifest-invalid",
        error="staged manifest digest did not match RunSpec authority",
    )


def _observation(*, size_bytes: int = 7) -> DatabaseSourceObservation:
    return DatabaseSourceObservation(
        source_container_root=DATABASE_SOURCE_ROOT,
        source_manifest_sha256="b" * 64,
        members=(
            DatabaseSourceMember(
                role="primary",
                database_name="primary",
                logical_name="primary",
                source_path="primary",
                source_kind="regular",
                resolved_path="primary",
                resolved_kind="regular",
                size_bytes=size_bytes,
                mtime_ns=123,
                alias_topology=(),
                preexisting_checksum=None,
            ),
            DatabaseSourceMember(
                role="metagenomic",
                database_name="metagenomic",
                logical_name="metagenomic",
                source_path="metagenomic",
                source_kind="regular",
                resolved_path="metagenomic",
                resolved_kind="regular",
                size_bytes=7,
                mtime_ns=123,
                alias_topology=(),
                preexisting_checksum=None,
            ),
        ),
    )


def _result() -> DatabasePlacementResult:
    return DatabasePlacementResult(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        phase_runspec_digest="a" * 64,
        action_id="preprocessing-chunk-000000",
        database_set=DatabaseSetIdentity(identifier="bspp-search", version="2026-08"),
        requested_policy=DatabaseAccessPolicy.DIRECT,
        source_manifest_sha256="b" * 64,
        branch_kind="direct-requested",
        outcome=DatabasePlacementOutcomeKind.DIRECT_REQUESTED,
        selected_container_root=SELECTED_DATABASE_ROOT,
        verification="metadata-verified",
        source_mount=DatabaseSourceMountFacts(
            mount_id=42,
            parent_mount_id=31,
            device_major=8,
            device_minor=2,
            mount_root="/",
            mount_point=DATABASE_SOURCE_ROOT,
            filesystem_type="lustre",
            mount_source="server:/database",
            mount_options=("nodev", "ro"),
            super_options=("flock", "ro"),
            read_only=True,
        ),
        pre_science_observation=_observation(),
    )


def test_database_placement_failure_evidence_round_trips_exact_canonical_shape() -> None:
    failure = _failure()

    assert database_placement_failure_evidence_from_mapping(failure.to_mapping()) == failure
    assert canonical_database_placement_failure_evidence_bytes(failure).endswith(b"\n")

    unknown = deepcopy(failure.to_mapping())
    unknown["database_placement_failure"]["detail"] = "unbounded"
    with pytest.raises(ValueError, match="fields are invalid"):
        database_placement_failure_evidence_from_mapping(unknown)


@pytest.mark.parametrize(
    "classification",
    [
        "authority-invalid",
        "source-manifest-invalid",
        "source-mount-invalid",
        "source-inventory-invalid",
        "result-publication-failed",
    ],
)
def test_database_placement_failure_taxonomy_is_closed(classification: str) -> None:
    payload = _failure().to_mapping()
    payload["database_placement_failure"]["classification"] = classification
    assert database_placement_failure_evidence_from_mapping(payload).classification == classification


def test_database_placement_failure_rejects_unbounded_or_started_science() -> None:
    payload = _failure().to_mapping()
    payload["database_placement_failure"]["classification"] = "disk-full"
    with pytest.raises(ValueError, match="classification"):
        database_placement_failure_evidence_from_mapping(payload)

    payload = _failure().to_mapping()
    payload["database_placement_failure"]["science_started"] = True
    with pytest.raises(ValueError, match="science_started"):
        database_placement_failure_evidence_from_mapping(payload)

    payload = _failure().to_mapping()
    payload["database_placement_failure"]["error"] = "x" * 2049
    with pytest.raises(ValueError, match="bounded"):
        database_placement_failure_evidence_from_mapping(payload)


def test_action_placement_success_binds_result_digest_and_post_observation() -> None:
    result = _result()
    placement = PreprocessingDatabasePlacementEvidence(
        result=result,
        result_digest=database_placement_result_digest(result),
        failure=None,
        science_started=True,
        post_science_observation=DatabasePostScienceObservation.from_pre_science(_observation()),
    )

    assert placement.result is result
    assert placement.post_science_observation is not None
    assert placement.post_science_observation.matches(result.pre_science_observation)
    assert PreprocessingDatabasePlacementEvidence.from_mapping(placement.to_mapping()) == placement


def test_action_placement_retains_drifted_post_observation_but_not_as_success() -> None:
    result = _result()
    placement = PreprocessingDatabasePlacementEvidence(
        result=result,
        result_digest=database_placement_result_digest(result),
        failure=None,
        science_started=True,
        post_science_observation=DatabasePostScienceObservation.from_pre_science(_observation(size_bytes=8)),
    )

    assert placement.post_science_observation is not None
    assert not placement.post_science_observation.matches(result.pre_science_observation)


def test_action_placement_retains_bounded_post_science_observation_failure() -> None:
    result = _result()
    observation_failure = DatabasePostScienceObservationFailure(
        source_container_root=DATABASE_SOURCE_ROOT,
        source_manifest_sha256=result.source_manifest_sha256,
        error="selected Database Source root is unavailable",
    )
    placement = PreprocessingDatabasePlacementEvidence(
        result=result,
        result_digest=database_placement_result_digest(result),
        failure=None,
        science_started=True,
        post_science_observation=observation_failure,
    )

    loaded = PreprocessingDatabasePlacementEvidence.from_mapping(placement.to_mapping())

    assert loaded == placement
    assert loaded.post_science_observation == observation_failure
    assert loaded.post_science_observation.verification == "observation-failed"


def test_action_placement_pre_science_failure_excludes_result_and_post_observation() -> None:
    placement = PreprocessingDatabasePlacementEvidence(
        result=None,
        result_digest=None,
        failure=_failure(),
        science_started=False,
        post_science_observation=None,
    )

    assert PreprocessingDatabasePlacementEvidence.from_mapping(placement.to_mapping()) == placement

    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(placement, result=_result())


def test_action_placement_command_failure_retains_exact_result_after_nonzero_exit() -> None:
    result = _result()
    evidence = PreprocessingDatabasePlacementCommandFailureEvidence(
        phase_run_id=result.phase_run_id,
        attempt_id=result.attempt_id,
        phase_runspec_digest=result.phase_runspec_digest,
        action_id=result.action_id,
        database_set=result.database_set,
        requested_policy=result.requested_policy,
        source_manifest_sha256=result.source_manifest_sha256,
        placement_process_status=23,
        result=result,
        result_digest=database_placement_result_digest(result),
        science_started=False,
        classification="nonzero-after-result",
        error="Database Placement exited nonzero after publishing its exact Result",
    )

    assert PreprocessingDatabasePlacementCommandFailureEvidence.from_mapping(evidence.to_mapping()) == evidence


def _command_failure() -> PreprocessingDatabasePlacementCommandFailureEvidence:
    result = _result()
    return PreprocessingDatabasePlacementCommandFailureEvidence(
        phase_run_id=result.phase_run_id,
        attempt_id=result.attempt_id,
        phase_runspec_digest=result.phase_runspec_digest,
        action_id=result.action_id,
        database_set=result.database_set,
        requested_policy=result.requested_policy,
        source_manifest_sha256=result.source_manifest_sha256,
        placement_process_status=23,
        result=result,
        result_digest=database_placement_result_digest(result),
        science_started=False,
        classification="nonzero-after-result",
        error="Database Placement exited nonzero after publishing its exact Result",
    )


@pytest.mark.parametrize("status", [True, -1, 256])
def test_action_placement_command_failure_rejects_invalid_process_status(status: object) -> None:
    with pytest.raises(ValueError, match="status"):
        replace(_command_failure(), placement_process_status=status)


def test_action_placement_command_failure_rejects_zero_changed_digest_or_outer_identity() -> None:
    evidence = _command_failure()
    with pytest.raises(ValueError, match="nonzero"):
        replace(evidence, placement_process_status=0)
    with pytest.raises(ValueError, match="digest"):
        replace(evidence, result_digest="f" * 64)
    with pytest.raises(ValueError, match="outer trusted identity"):
        replace(evidence, phase_runspec_digest="f" * 64)


def test_action_placement_reconciliation_failure_excludes_result_and_bounds_diagnostic() -> None:
    evidence = _command_failure()
    with pytest.raises(ValueError, match="cannot retain"):
        replace(evidence, classification="evidence-reconciliation-failed")
    reconciled = replace(
        evidence,
        classification="evidence-reconciliation-failed",
        placement_process_status=0,
        result=None,
        result_digest=None,
    )
    assert PreprocessingDatabasePlacementCommandFailureEvidence.from_mapping(reconciled.to_mapping()) == reconciled
    with pytest.raises(ValueError, match="bounded"):
        replace(reconciled, error="x" * 2049)
    with pytest.raises(ValueError, match="bounded"):
        replace(reconciled, error="invalid\x00authority")
    with pytest.raises(ValueError, match="science_started"):
        replace(reconciled, science_started=True)
