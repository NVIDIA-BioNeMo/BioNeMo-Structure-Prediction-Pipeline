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

"""Strict staged Database Replica Lease action-evidence contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bspp.orchestration.contract.database_placement import (
    DATABASE_CACHE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
)
from bspp.orchestration.contract.database_placement_result import (
    DatabasePostScienceObservation,
    database_placement_result_digest,
)
from bspp.orchestration.contract.database_replica import (
    DatabaseCapacityGate,
    DatabaseReplicaColdFailureEvidence,
    DatabaseReplicaColdResult,
    DatabaseReplicaWarmResult,
    DatabaseWarmCacheMountFacts,
    canonical_database_replica_cold_failure_evidence_bytes,
    canonical_database_replica_cold_result_bytes,
    canonical_database_replica_manifest_bytes,
    database_replica_cold_result_digest,
    database_replica_manifest_digest,
    database_replica_result_from_mapping,
    database_replica_warm_result_digest,
)
from bspp.orchestration.contract.database_replica_lease import (
    DatabaseReplicaLeaseFailureEvidence,
    DatabaseReplicaLeaseTerminalEvidence,
    DatabaseReplicaWarmLeaseTerminalEvidence,
)
from bspp.orchestration.contract.database_set_provisioning import DatabaseSetIdentity
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingDatabasePlacementEvidence,
    PreprocessingStagedDatabasePlacementEvidence,
    preprocessing_chunk_action_evidence_from_mapping,
)
from tests.support.database_cold_replica import _cache_mount, _capacity, _manifest, _source_mount
from tests.support.preprocessing_execution import (
    configure_preprocessing_fakes,
    invoke_preprocessing_execution,
    preprocessing_execution_fixture,
    skip_preprocessing_server_warmup,
)
from tests.test_preprocessing_database_direct_action import _observation, _result


def _cold_result(*, phase_runspec_digest: str = "b" * 64) -> DatabaseReplicaColdResult:
    manifest = _manifest()
    return DatabaseReplicaColdResult(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        phase_runspec_digest=phase_runspec_digest,
        action_id="preprocessing-chunk-000000",
        database_set=manifest.database_set,
        requested_policy=DatabaseAccessPolicy.STAGE_REQUIRED,
        source_manifest_sha256=manifest.source_manifest_sha256,
        branch_kind="staged",
        outcome=DatabasePlacementOutcomeKind.REPLICA_COLD,
        selected_container_root=SELECTED_DATABASE_ROOT,
        replica_container_root=f"{DATABASE_CACHE_ROOT}/replicas/{'a' * 64}",
        verification="metadata-verified",
        source_mount=_source_mount(),
        cache_mount=_cache_mount(),
        capacity_gate=_capacity(),
        replica_manifest_sha256=database_replica_manifest_digest(manifest),
        copy_evidence=manifest.copy_evidence,
    )


def _cold_failure() -> DatabaseReplicaColdFailureEvidence:
    return DatabaseReplicaColdFailureEvidence(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        phase_runspec_digest="b" * 64,
        action_id="preprocessing-chunk-000000",
        database_set=DatabaseSetIdentity(identifier="bspp-search", version="2026-08"),
        requested_policy=DatabaseAccessPolicy.STAGE_REQUIRED,
        source_manifest_sha256="a" * 64,
        source_mount=_source_mount(),
        cache_mount=_cache_mount(),
        capacity_gate=DatabaseCapacityGate(
            available_user_bytes=127,
            allocated_replica_bytes=64,
            reserved_bytes=64,
            required_bytes=128,
            decision="insufficient",
        ),
        science_started=False,
        classification="insufficient-capacity",
        error="insufficient user-available capacity",
    )


def _warm_result(*, phase_runspec_digest: str = "b" * 64) -> DatabaseReplicaWarmResult:
    cold_mount = _cache_mount()
    manifest = _manifest()
    return DatabaseReplicaWarmResult(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        phase_runspec_digest=phase_runspec_digest,
        action_id="preprocessing-chunk-000000",
        database_set=manifest.database_set,
        requested_policy=DatabaseAccessPolicy.STAGE_REQUIRED,
        source_manifest_sha256=manifest.source_manifest_sha256,
        branch_kind="staged",
        outcome=DatabasePlacementOutcomeKind.REPLICA_WARM,
        selected_container_root=SELECTED_DATABASE_ROOT,
        replica_container_root=f"{DATABASE_CACHE_ROOT}/replicas/{'a' * 64}",
        verification="metadata-verified",
        cache_mount=DatabaseWarmCacheMountFacts(
            mount_id=cold_mount.mount_id,
            parent_mount_id=cold_mount.parent_mount_id,
            device_major=cold_mount.device_major,
            device_minor=cold_mount.device_minor,
            mount_root=cold_mount.mount_root,
            mount_point=cold_mount.mount_point,
            filesystem_type=cold_mount.filesystem_type,
            mount_source=cold_mount.mount_source,
            mount_options=cold_mount.mount_options,
            super_options=cold_mount.super_options,
            read_write=True,
        ),
        replica_manifest_sha256=database_replica_manifest_digest(manifest),
    )


def _warm_terminal(result: DatabaseReplicaWarmResult) -> DatabaseReplicaWarmLeaseTerminalEvidence:
    return DatabaseReplicaWarmLeaseTerminalEvidence(
        database_replica_warm_result_digest=database_replica_warm_result_digest(result),
        source_manifest_sha256=result.source_manifest_sha256,
        replica_manifest_sha256=result.replica_manifest_sha256,
        selected_container_root=result.selected_container_root,
        lease_target="/run/bspp/database/replica-lease.lock",
        verification="metadata-verified",
        acquisition="shared-nonblocking",
        kernel_started=True,
        held_through_kernel_exit=True,
    )


def _terminal_mapping(result: DatabaseReplicaColdResult) -> dict[str, object]:
    return {
        "schema_version": 1,
        "lease_outcome_kind": "terminal",
        "database_replica_cold_result_digest": database_replica_cold_result_digest(result),
        "source_manifest_sha256": result.source_manifest_sha256,
        "replica_manifest_sha256": result.replica_manifest_sha256,
        "selected_container_root": result.selected_container_root,
        "lease_target": "/run/bspp/database/replica-lease.lock",
        "verification": "metadata-verified",
        "acquisition": "shared-nonblocking",
        "kernel_started": True,
        "held_through_kernel_exit": True,
    }


def _terminal(result: DatabaseReplicaColdResult) -> DatabaseReplicaLeaseTerminalEvidence:
    return DatabaseReplicaLeaseTerminalEvidence(
        database_replica_cold_result_digest=database_replica_cold_result_digest(result),
        source_manifest_sha256=result.source_manifest_sha256,
        replica_manifest_sha256=result.replica_manifest_sha256,
        selected_container_root=result.selected_container_root,
        lease_target="/run/bspp/database/replica-lease.lock",
        verification="metadata-verified",
        acquisition="shared-nonblocking",
        kernel_started=True,
        held_through_kernel_exit=True,
    )


def _lease_failure(result: DatabaseReplicaColdResult) -> DatabaseReplicaLeaseFailureEvidence:
    return DatabaseReplicaLeaseFailureEvidence(
        database_replica_cold_result_digest=database_replica_cold_result_digest(result),
        source_manifest_sha256=result.source_manifest_sha256,
        replica_manifest_sha256=result.replica_manifest_sha256,
        selected_container_root=result.selected_container_root,
        lease_target="/run/bspp/database/replica-lease.lock",
        verification="metadata-verified",
        classification="replica-revalidation-failed",
        error="replica member metadata changed during handoff",
    )


def test_staged_action_mapping_strict_loads_as_a_separate_concrete_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work")
    configure_preprocessing_fakes(fixture, monkeypatch)
    skip_preprocessing_server_warmup(monkeypatch)
    invocation = invoke_preprocessing_execution(fixture)
    assert invocation.exit_code == 0, invocation.output
    mapping = json.loads(fixture.evidence_path.read_text())
    direct = preprocessing_chunk_action_evidence_from_mapping(mapping)
    result = _cold_result(phase_runspec_digest=direct.phase_runspec_digest)
    result = DatabaseReplicaColdResult(
        **{
            **result.__dict__,
            "phase_run_id": direct.phase_run_id,
            "attempt_id": direct.attempt_id,
            "action_id": direct.action_id,
        }
    )
    mapping["database_placement"] = {
        "schema_version": 1,
        "result": result.to_mapping(),
        "result_digest": database_replica_cold_result_digest(result),
        "failure": None,
        "lease_outcome": _terminal_mapping(result),
        "science_started": True,
    }

    loaded = preprocessing_chunk_action_evidence_from_mapping(mapping)

    assert type(loaded.database_placement).__name__ == "PreprocessingStagedDatabasePlacementEvidence"
    assert loaded.database_placement.to_mapping() == mapping["database_placement"]


def test_staged_policy_failure_action_rejects_payload_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work")
    configure_preprocessing_fakes(fixture, monkeypatch)
    skip_preprocessing_server_warmup(monkeypatch)
    invocation = invoke_preprocessing_execution(fixture)
    assert invocation.exit_code == 0, invocation.output
    direct = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    failure = replace(
        _cold_failure(),
        phase_run_id=direct.phase_run_id,
        attempt_id=direct.attempt_id,
        phase_runspec_digest=direct.phase_runspec_digest,
        action_id=direct.action_id,
    )
    failed = replace(
        direct,
        placement_process_status=19,
        outcome="failed",
        command_outcomes=(),
        paired_evidence=replace(direct.paired_evidence, record_lines=None, log_lines=None),
        archive_evidence=replace(
            direct.archive_evidence,
            tar_size_bytes=None,
            lz4_size_bytes=None,
            tar_members=None,
        ),
        output_hashes=(),
        error="Database Placement failed before science",
        database_placement=PreprocessingStagedDatabasePlacementEvidence(None, None, failure, None, False),
        raw_search_evidence=None,
    )

    with pytest.raises(ValueError, match="payload-free"):
        replace(failed, paired_evidence=replace(failed.paired_evidence, log_lines=("forged",)))
    with pytest.raises(ValueError, match="payload-free"):
        replace(failed, archive_evidence=replace(failed.archive_evidence, lz4_size_bytes=1))


@pytest.mark.parametrize("retained", ["output_hashes", "raw_search_evidence"])
def test_failed_started_staged_action_rejects_any_accepted_science_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retained: str,
) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work")
    configure_preprocessing_fakes(fixture, monkeypatch)
    skip_preprocessing_server_warmup(monkeypatch)
    invocation = invoke_preprocessing_execution(fixture)
    assert invocation.exit_code == 0, invocation.output
    mapping = json.loads(fixture.evidence_path.read_text())
    direct = preprocessing_chunk_action_evidence_from_mapping(mapping)
    result = _cold_result(phase_runspec_digest=direct.phase_runspec_digest)
    result = DatabaseReplicaColdResult(
        **{
            **result.__dict__,
            "phase_run_id": direct.phase_run_id,
            "attempt_id": direct.attempt_id,
            "action_id": direct.action_id,
        }
    )
    mapping["database_placement"] = {
        "schema_version": 1,
        "result": result.to_mapping(),
        "result_digest": database_replica_cold_result_digest(result),
        "failure": None,
        "lease_outcome": _terminal_mapping(result),
        "science_started": True,
    }
    mapping["outcome"] = "failed"
    mapping["error"] = "post-science staged failure"
    if retained == "output_hashes":
        mapping.pop("raw_search_evidence")
    else:
        mapping["output_hashes"] = []

    with pytest.raises(ValueError, match="failed staged"):
        preprocessing_chunk_action_evidence_from_mapping(mapping)


def test_direct_and_issue_91_canonical_bytes_remain_exact() -> None:
    direct_result = _result()
    direct = PreprocessingDatabasePlacementEvidence(
        result=direct_result,
        result_digest=database_placement_result_digest(direct_result),
        failure=None,
        science_started=True,
        post_science_observation=DatabasePostScienceObservation.from_pre_science(_observation()),
    )
    direct_bytes = (json.dumps(direct.to_mapping(), sort_keys=True, separators=(",", ":")) + "\n").encode()

    assert (
        hashlib.sha256(direct_bytes).hexdigest() == "26ff82c4073f8df6948608c28a09fb71f4e69138eaa5b7b803f2e09e59e02f5c"
    )
    assert hashlib.sha256(canonical_database_replica_cold_result_bytes(_cold_result())).hexdigest() == (
        "ae6618c2be83915d626c785a1ab3f3f17f7c3e70c3d0e827ba2edaddebc2c2a3"
    )
    assert hashlib.sha256(canonical_database_replica_cold_failure_evidence_bytes(_cold_failure())).hexdigest() == (
        "de4c94a70b3e97af05387249e4788c486f777e5d70bbe6058765835d7440753a"
    )
    assert hashlib.sha256(canonical_database_replica_manifest_bytes(_manifest())).hexdigest() == (
        "69ba03bb3e138b6872ab1c3f0ee20833cc2eda4997b525387da2903c6a1113d6"
    )
    result = _cold_result()
    wrappers = (
        PreprocessingStagedDatabasePlacementEvidence(None, None, _cold_failure(), None, False),
        PreprocessingStagedDatabasePlacementEvidence(
            result,
            database_replica_cold_result_digest(result),
            None,
            _lease_failure(result),
            False,
        ),
        PreprocessingStagedDatabasePlacementEvidence(
            result,
            database_replica_cold_result_digest(result),
            None,
            _terminal(result),
            True,
        ),
    )
    wrapper_hashes = tuple(
        hashlib.sha256(
            (json.dumps(item.to_mapping(), sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        for item in wrappers
    )
    assert wrapper_hashes == (
        "7979aa22c659c6cee2ffe825cf373d4a292842cf3e874f0e0f3cb0c84b2c87c4",
        "736c143c17bc6431ee261244162e15cc7abe77c32974350e0fd184b35bc396d3",
        "088ab1bf8fd820a6b259438b3dce34ec25da2ec945d2ff85695c3fd36117afa8",
    )
    lease_hashes = tuple(
        hashlib.sha256(
            (json.dumps(item.to_mapping(), sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        for item in (_lease_failure(result), _terminal(result))
    )
    assert lease_hashes == (
        "f737bfa9d8836f5a3b1f9d36c65c07e7b34839ce773b0ea6e3866ba483f856f0",
        "7454241de4cfc2ccdf3a7c0af95f3387ff15784a67d8f37b5f6d6c04765f2733",
    )


def test_staged_wrapper_accepts_exact_three_state_machine() -> None:
    result = _cold_result()
    cold_failure = PreprocessingStagedDatabasePlacementEvidence(
        result=None,
        result_digest=None,
        failure=_cold_failure(),
        lease_outcome=None,
        science_started=False,
    )
    lease_failure = PreprocessingStagedDatabasePlacementEvidence(
        result=result,
        result_digest=database_replica_cold_result_digest(result),
        failure=None,
        lease_outcome=_lease_failure(result),
        science_started=False,
    )
    terminal = PreprocessingStagedDatabasePlacementEvidence(
        result=result,
        result_digest=database_replica_cold_result_digest(result),
        failure=None,
        lease_outcome=_terminal(result),
        science_started=True,
    )

    for evidence in (cold_failure, lease_failure, terminal):
        assert PreprocessingStagedDatabasePlacementEvidence.from_mapping(evidence.to_mapping()) == evidence


def test_warm_staged_wrapper_round_trips_and_rejects_cold_lease_family() -> None:
    result = _warm_result()
    evidence = PreprocessingStagedDatabasePlacementEvidence(
        result=result,
        result_digest=database_replica_warm_result_digest(result),
        failure=None,
        lease_outcome=_warm_terminal(result),
        science_started=True,
    )

    assert PreprocessingStagedDatabasePlacementEvidence.from_mapping(evidence.to_mapping()) == evidence
    cold_lease = replace(
        _terminal(_cold_result()),
        database_replica_cold_result_digest=database_replica_warm_result_digest(result),
        source_manifest_sha256=result.source_manifest_sha256,
        replica_manifest_sha256=result.replica_manifest_sha256,
        selected_container_root=result.selected_container_root,
    )
    with pytest.raises(ValueError, match="warm Result requires the warm lease"):
        replace(evidence, lease_outcome=cold_lease)


@pytest.mark.parametrize(
    "forgery",
    [
        "missing-gpuserver",
        "extra-command",
        "wrong-gpuserver-outcome",
        "retained-output-hashes",
        "retained-raw-search",
    ],
)
def test_warm_no_kernel_action_rejects_every_non_exact_failed_to_start_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forgery: str,
) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work")
    configure_preprocessing_fakes(fixture, monkeypatch)
    skip_preprocessing_server_warmup(monkeypatch)
    invocation = invoke_preprocessing_execution(fixture)
    assert invocation.exit_code == 0, invocation.output
    mapping = json.loads(fixture.evidence_path.read_text())
    direct = preprocessing_chunk_action_evidence_from_mapping(mapping)
    result = replace(
        _warm_result(phase_runspec_digest=direct.phase_runspec_digest),
        phase_run_id=direct.phase_run_id,
        attempt_id=direct.attempt_id,
        action_id=direct.action_id,
    )
    terminal = replace(
        _warm_terminal(result),
        kernel_started=False,
        held_through_kernel_exit=False,
    )
    original_commands = mapping["command_outcomes"]
    original_output_hashes = mapping["output_hashes"]
    original_raw_search = mapping["raw_search_evidence"]
    assert isinstance(original_commands, list)
    assert isinstance(original_output_hashes, list)
    assert isinstance(original_raw_search, dict)
    failed_to_start = {
        **original_commands[0],
        "disposition": "failed-to-start",
        "return_code": None,
    }
    mapping.update(
        {
            "outcome": "failed",
            "command_outcomes": [failed_to_start],
            "output_hashes": [],
            "error": "gpuserver failed to start",
            "database_placement": PreprocessingStagedDatabasePlacementEvidence(
                result=result,
                result_digest=database_replica_warm_result_digest(result),
                failure=None,
                lease_outcome=terminal,
                science_started=False,
            ).to_mapping(),
        }
    )
    mapping.pop("raw_search_evidence")
    if forgery == "missing-gpuserver":
        mapping["command_outcomes"] = []
    elif forgery == "extra-command":
        mapping["command_outcomes"] = [failed_to_start, original_commands[1]]
    elif forgery == "wrong-gpuserver-outcome":
        mapping["command_outcomes"] = [
            {
                **failed_to_start,
                "disposition": "completed",
                "return_code": 0,
            }
        ]
    elif forgery == "retained-output-hashes":
        mapping["output_hashes"] = original_output_hashes
    elif forgery == "retained-raw-search":
        mapping["raw_search_evidence"] = original_raw_search

    with pytest.raises(
        ValueError,
        match="no-kernel staged evidence requires the exact failed-to-start action shape",
    ):
        preprocessing_chunk_action_evidence_from_mapping(mapping)


@pytest.mark.parametrize(
    "mapping",
    [
        {},
        {"unknown_database_replica_result": {}},
        {
            "database_replica_cold_result": {},
            "database_replica_warm_result": {},
        },
    ],
    ids=["zero-envelope", "unknown-envelope", "multiple-envelopes"],
)
def test_database_replica_result_union_rejects_non_exact_envelopes(mapping: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="one known exact envelope"):
        database_replica_result_from_mapping(mapping)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda evidence, result: replace(evidence, result=result),
        lambda evidence, result: replace(evidence, lease_outcome=_terminal(result)),
        lambda evidence, result: replace(evidence, science_started=True),
    ],
)
def test_staged_cold_failure_rejects_mixed_result_lease_or_science(
    mutation: object,
) -> None:
    result = _cold_result()
    evidence = PreprocessingStagedDatabasePlacementEvidence(
        result=None,
        result_digest=None,
        failure=_cold_failure(),
        lease_outcome=None,
        science_started=False,
    )
    assert callable(mutation)
    with pytest.raises(ValueError):
        mutation(evidence, result)


def test_staged_result_accepts_pre_lease_state_without_lease_outcome() -> None:
    """A cold Result with no lease outcome and no science is a valid pre-lease state (F9)."""
    result = _cold_result()
    digest = database_replica_cold_result_digest(result)
    evidence = PreprocessingStagedDatabasePlacementEvidence(result, digest, None, None, False)
    assert evidence.result == result
    assert evidence.lease_outcome is None
    assert evidence.science_started is False
    assert PreprocessingStagedDatabasePlacementEvidence.from_mapping(evidence.to_mapping()) == evidence


def test_staged_result_rejects_missing_wrong_phase_or_forged_lease() -> None:
    result = _cold_result()
    digest = database_replica_cold_result_digest(result)
    with pytest.raises(ValueError, match="subsequent lease outcome"):
        PreprocessingStagedDatabasePlacementEvidence(result, digest, None, None, True)
    with pytest.raises(ValueError, match="before science"):
        PreprocessingStagedDatabasePlacementEvidence(result, digest, None, _lease_failure(result), True)
    with pytest.raises(ValueError, match="science_started"):
        PreprocessingStagedDatabasePlacementEvidence(result, digest, None, _terminal(result), False)
    forged = replace(_terminal(result), replica_manifest_sha256="f" * 64)
    with pytest.raises(ValueError, match="exact cold Result"):
        PreprocessingStagedDatabasePlacementEvidence(result, digest, None, forged, True)


@pytest.mark.parametrize(
    ("kernel_started", "held_through_kernel_exit"),
    [(True, False), (False, True)],
)
def test_terminal_lease_evidence_rejects_crossed_kernel_lifetime_booleans(
    kernel_started: bool,
    held_through_kernel_exit: bool,
) -> None:
    result = _cold_result()
    with pytest.raises(ValueError, match="kernel"):
        replace(
            _terminal(result),
            kernel_started=kernel_started,
            held_through_kernel_exit=held_through_kernel_exit,
        )


def test_staged_wrapper_requires_science_started_to_match_terminal_kernel_started() -> None:
    result = _cold_result()
    no_kernel = replace(_terminal(result), kernel_started=False, held_through_kernel_exit=False)
    evidence = PreprocessingStagedDatabasePlacementEvidence(
        result=result,
        result_digest=database_replica_cold_result_digest(result),
        failure=None,
        lease_outcome=no_kernel,
        science_started=False,
    )
    assert PreprocessingStagedDatabasePlacementEvidence.from_mapping(evidence.to_mapping()) == evidence
    with pytest.raises(ValueError, match="science_started"):
        replace(evidence, science_started=True)
