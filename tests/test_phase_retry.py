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

"""Immutable successor Phase Attempt retry tests."""

from __future__ import annotations

import json
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from bspp.orchestration.contract.database_placement import (
    DatabaseAccessPolicy,
    PreprocessingDatabaseBinding,
)
from bspp.orchestration.contract.database_set_provisioning import (
    canonical_database_source_manifest_bytes,
)
from bspp.orchestration.contract.phase_retry import (
    PhaseAttemptRetriedEvent,
    PhaseAttemptRetriedPayload,
    phase_input_set_identity_digest,
    phase_retry_id,
    phase_scientific_identity_digest,
)
from bspp.orchestration.contract.phase_state import PhaseAttempt
from bspp.orchestration.contract.phase_submission import PhaseSubmissionIntendedEvent
from bspp.orchestration.contract.preprocessing_runtime import (
    preprocessing_runtime_qualification_record_from_mapping,
    preprocessing_runtime_tuple_id,
)
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore, PhaseAuthorityValidation
from bspp.orchestration.control.phase_cancellation import cancel_phase
from bspp.orchestration.control.phase_finalization import finalize_phase
from bspp.orchestration.control.phase_materialization import materialize_phase
from bspp.orchestration.control.phase_rendering import render_phase_submission_intent
from bspp.orchestration.control.phase_resume import resume_phase
from bspp.orchestration.control.phase_retry import retry_phase
from bspp.orchestration.control.phase_status import status_phase
from bspp.orchestration.control.phase_submission import _submission_intended_at, submit_phase
from bspp.orchestration.control.preprocessing_runtime_qualification import (
    preprocessing_runtime_qualification_path,
    preprocessing_runtime_qualification_tuple,
)
from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.control.transport import CommandResult
from tests.test_phase_cancellation import CancellationRunner
from tests.test_phase_materialization import (
    FIXED_RUN_ID,
    FIXED_TIME,
    MaterializationFixture,
    _file_bytes,
    _fixture,
)
from tests.test_phase_resume import _record_submission_state, _record_terminal
from tests.test_phase_submission import LocalSubmissionRunner

RETRY_TIME = datetime(2026, 8, 20, 13, 0, tzinfo=UTC)


def _failed_materialized_run(tmp_path: Path) -> tuple[MaterializationFixture, Path]:
    fixture = _fixture(tmp_path, policy=DatabaseAccessPolicy.DIRECT)
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    _record_submission_state(authority_root, FIXED_RUN_ID, "submitted")
    _record_terminal(authority_root, FIXED_RUN_ID, state="FAILED", exit_code="1:0", outcome="failed")
    return fixture, authority_root


def _set_gpu_array(fixture: MaterializationFixture, value: str) -> None:
    profile_mapping = yaml.safe_load(fixture.profile_path.read_text())
    profile_mapping["clusters"]["example-cluster"]["resources"] = {"gpu_worker": {"array": value}}
    fixture.profile_path.write_text(yaml.safe_dump(profile_mapping, sort_keys=True))


def _leave_terminal_retry_event(
    fixture: MaterializationFixture,
    authority_root: Path,
) -> PhaseAuthorityValidation:
    def interrupt_after_event(_validation: object) -> None:
        raise OSError("stop after Retry event")

    with pytest.raises(OSError, match="stop after Retry event"):
        retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: RETRY_TIME,
            authority_store=PhaseAuthorityStore(
                authority_root,
                after_retry_event_publish=interrupt_after_event,
            ),
        )
    return PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)


def _raw_retry_event(
    authority: PhaseAuthorityValidation,
    *,
    sequence: int,
    predecessor_attempt_id: str,
    successor_ordinal: int,
) -> PhaseAttemptRetriedEvent:
    successor_id = f"attempt-{successor_ordinal:04d}"
    materialized_at = "2026-08-21T13:00:00.000000Z"
    successor_runspec = replace(
        authority.phase_runspec,
        attempt_id=successor_id,
        materialized_at=materialized_at,
        payload=replace(
            authority.phase_runspec.payload,
            database=replace(
                authority.phase_runspec.payload.database,
                source_manifest_projection=f"attempts/{successor_id}/database-source-manifest.json",
            ),
        ),
    )
    successor = PhaseAttempt(
        attempt_id=successor_id,
        ordinal=successor_ordinal,
        phase_runspec_location=f"attempts/{successor_id}/phase-runspec.json",
        phase_runspec_digest=successor_runspec.digest,
        created_at=materialized_at,
    )
    input_digest = phase_input_set_identity_digest(authority.phase_plan)
    scientific_digest = phase_scientific_identity_digest(authority.phase_plan)
    retry_identity = phase_retry_id(
        phase_run_id=FIXED_RUN_ID,
        predecessor_attempt_id=predecessor_attempt_id,
        successor_attempt_id=successor_id,
        predecessor_outcome="failed",
        predecessor_phase_runspec_digest=authority.phase_runspec.digest,
        phase_plan_digest=authority.phase_plan.digest,
        input_set_identity_digest=input_digest,
        scientific_identity_digest=scientific_digest,
        selected_cluster_profile=successor_runspec.cluster.profile_name,
        successor_phase_runspec_digest=successor_runspec.digest,
    )
    payload = PhaseAttemptRetriedPayload(
        retry_id=retry_identity,
        predecessor_attempt_id=predecessor_attempt_id,
        predecessor_phase_runspec_digest=authority.phase_runspec.digest,
        predecessor_outcome="failed",
        phase_plan_digest=authority.phase_plan.digest,
        input_set_identity_digest=input_digest,
        scientific_identity_digest=scientific_digest,
        selected_cluster_profile=successor_runspec.cluster.profile_name,
        successor_attempt=successor,
        successor_phase_runspec=successor_runspec,
    )
    return PhaseAttemptRetriedEvent(
        sequence=sequence,
        phase_run_id=FIXED_RUN_ID,
        attempt_id=successor_id,
        occurred_at=materialized_at,
        payload=payload,
    )


def test_failed_attempt_retries_to_clean_active_successor_and_preserves_predecessor_bytes(tmp_path: Path) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    run_root = authority_root / FIXED_RUN_ID
    before = _file_bytes(run_root)
    predecessor_sentinel = tmp_path / "predecessor-work" / "evidence" / "sentinel.json"
    predecessor_sentinel.parent.mkdir(parents=True)
    predecessor_sentinel.write_bytes(b'{"attempt":"attempt-0001","immutable":true}\n')
    sentinel_before = predecessor_sentinel.read_bytes()
    profile = resolve_cluster_profile("example-cluster", config_path=fixture.profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=fixture.source_repo)
    qualification_path = preprocessing_runtime_qualification_path(
        profile,
        tuple_id=preprocessing_runtime_tuple_id(qualification_tuple),
    )
    qualification_before = {qualification_path: qualification_path.read_bytes()}

    result = retry_phase(
        FIXED_RUN_ID,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: RETRY_TIME,
    )

    assert result.predecessor_attempt_id == "attempt-0001"
    assert result.successor_attempt_id == "attempt-0002"
    assert result.status == "materialized"
    validation = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert validation.current_attempt.attempt_id == validation.lifecycle.current_attempt_id == "attempt-0002"
    assert validation.current_runspec_projection_complete
    assert validation.submission is None
    assert validation.terminal_observations == ()
    assert validation.cancellation is None
    assert len(validation.prior_attempts) == 1
    assert validation.prior_attempts[0].outcome == "failed"
    assert validation.prior_attempts[0].submission is not None
    for relative, expected in before.items():
        assert (run_root / relative).read_bytes() == expected
    assert predecessor_sentinel.read_bytes() == sentinel_before
    assert {path: path.read_bytes() for path in qualification_before} == qualification_before
    assert (run_root / "attempts/attempt-0002/phase-runspec.json").is_file()
    assert (run_root / "attempts/attempt-0002/database-source-manifest.json").read_bytes() == (
        canonical_database_source_manifest_bytes(validation.phase_runspec.payload.database.source_manifest)
    )

    report = status_phase(FIXED_RUN_ID, authority_root=authority_root)
    assert report.attempt_id == "attempt-0002"
    assert [item.attempt_id for item in report.attempt_history] == ["attempt-0001", "attempt-0002"]
    assert report.requested_job_ids == ()


def test_cancelled_attempt_is_retryable_only_after_durable_cancellation_completion(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    cancelled = cancel_phase(FIXED_RUN_ID, authority_root=authority_root, clock=lambda: RETRY_TIME)
    assert cancelled.status == "cancelled"

    result = retry_phase(
        FIXED_RUN_ID,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: RETRY_TIME,
    )
    validation = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert result.successor_attempt_id == "attempt-0002"
    assert validation.prior_attempts[0].outcome == "cancelled"
    assert validation.prior_attempts[0].cancellation is not None
    assert validation.cancellation is None


def test_definitive_dispatch_rejection_is_retryable_without_resubmission(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, policy=DatabaseAccessPolicy.DIRECT)
    profiles = yaml.safe_load(fixture.profile_path.read_text())
    paths = profiles["clusters"]["example-cluster"]["paths"]
    paths["project_root"] = str(tmp_path / "project")
    paths["output_root"] = str(tmp_path / "output")
    paths["staging_root"] = str(tmp_path / "staging")
    fixture.profile_path.write_text(yaml.safe_dump(profiles, sort_keys=True))
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    runner = LocalSubmissionRunner(
        authority_root,
        FIXED_RUN_ID,
        sbatch_result=CommandResult((), 1, "", "partition rejected\n"),
    )
    with pytest.raises(ValueError, match="partition rejected"):
        submit_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            clock=lambda: RETRY_TIME,
            runner=runner,
        )
    assert len([call for call in runner.calls if call[0] == "sbatch"]) == 1

    retried = retry_phase(
        FIXED_RUN_ID,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: RETRY_TIME,
    )
    assert retried.successor_attempt_id == "attempt-0002"
    assert PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID).submission is None


def test_retry_can_select_rotated_qualified_image_without_changing_plan_identity(tmp_path: Path) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    old_profile = resolve_cluster_profile("example-cluster", config_path=fixture.profile_path)
    old_tuple = preprocessing_runtime_qualification_tuple(old_profile, source_repo=fixture.source_repo)
    old_record_path = preprocessing_runtime_qualification_path(
        old_profile,
        tuple_id=preprocessing_runtime_tuple_id(old_tuple),
    )
    old_record = preprocessing_runtime_qualification_record_from_mapping(json.loads(old_record_path.read_text()))
    profile_mapping = yaml.safe_load(fixture.profile_path.read_text())
    rotated = deepcopy(profile_mapping["clusters"]["example-cluster"])
    rotated["preprocessing_runtime"]["cluster_image_path"] = "/images/preprocessing-v2.sqsh"
    rotated["preprocessing_runtime"]["cluster_image_sha256"] = "2" * 64
    rotated["preprocessing_runtime"]["oci_digest"] = "sha256:" + "3" * 64
    profile_mapping["clusters"]["example-cluster"] = rotated
    fixture.profile_path.write_text(yaml.safe_dump(profile_mapping, sort_keys=True))

    new_profile = resolve_cluster_profile("example-cluster", config_path=fixture.profile_path)
    new_tuple = preprocessing_runtime_qualification_tuple(new_profile, source_repo=fixture.source_repo)
    new_record = replace(
        old_record,
        tuple_id=preprocessing_runtime_tuple_id(new_tuple),
        qualification_tuple=new_tuple,
        smoke_evidence=replace(
            old_record.smoke_evidence,
            image=replace(
                old_record.smoke_evidence.image,
                cluster_image_sha256="2" * 64,
                oci_digest="sha256:" + "3" * 64,
            ),
        ),
    )
    new_record_path = preprocessing_runtime_qualification_path(new_profile, tuple_id=new_record.tuple_id)
    new_record_path.parent.mkdir(parents=True, exist_ok=True)
    new_record_path.write_text(json.dumps(new_record.to_mapping(), indent=2, sort_keys=True) + "\n")

    result = retry_phase(
        FIXED_RUN_ID,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        profile_name="example-cluster",
        source_repo=fixture.source_repo,
        clock=lambda: RETRY_TIME,
    )

    authority = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert result.selected_cluster_profile == "example-cluster"
    assert authority.phase_plan.target_cluster == "example-cluster"
    assert authority.phase_runspec.phase_plan_digest == authority.phase_plan.digest
    assert authority.phase_runspec.cluster.profile_name == "example-cluster"
    assert authority.phase_runspec.cluster.runtime_image == "/images/preprocessing-v2.sqsh"
    assert authority.phase_runspec.payload.actions[0].payload.site.container_image == "/images/preprocessing-v2.sqsh"


def test_retry_rematerializes_only_profile_sourced_paths_and_resources(tmp_path: Path) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    before = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID).phase_runspec
    profile_mapping = yaml.safe_load(fixture.profile_path.read_text())
    profile = profile_mapping["clusters"]["example-cluster"]
    profile["paths"]["project_root"] = "/rotated/project"
    profile["paths"]["output_root"] = "/rotated/output"
    profile["paths"]["staging_root"] = "/rotated/staging"
    profile["resources"] = {
        "gpu_worker": {
            "partition": "rotated_gpu",
            "cpus_per_task": 12,
            "memory": "96G",
            "time": "02:30:00",
        }
    }
    fixture.profile_path.write_text(yaml.safe_dump(profile_mapping, sort_keys=True))

    retry_phase(
        FIXED_RUN_ID,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: RETRY_TIME,
    )

    successor = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID).phase_runspec
    assert successor.cluster.project_root == "/rotated/project"
    assert successor.cluster.output_root == "/rotated/output"
    assert successor.cluster.staging_root == "/rotated/staging"
    resources = successor.payload.actions[0].resources
    assert (resources.partition, resources.cpus_per_task, resources.memory, resources.time) == (
        "rotated_gpu",
        12,
        "96G",
        "02:30:00",
    )
    assert successor.phase_plan_digest == before.phase_plan_digest
    assert successor.input_location == before.input_location
    assert successor.payload.work_plan == before.payload.work_plan


def test_retry_freezes_fresh_database_site_authority_and_recovers_from_event_only(tmp_path: Path) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    before = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID).phase_runspec.payload.database
    rotated_manifest = replace(
        before.source_manifest,
        source_root="/rotated/databases",
        members=tuple(replace(member, mtime_ns=member.mtime_ns + 1) for member in before.source_manifest.members),
    )
    rotated_manifest_path = tmp_path / "rotated-database-source-manifest.json"
    rotated_manifest_path.write_bytes(canonical_database_source_manifest_bytes(rotated_manifest))
    profile_mapping = yaml.safe_load(fixture.profile_path.read_text())
    profile = profile_mapping["clusters"]["example-cluster"]
    profile["database_sets"][0]["manifest_path"] = str(rotated_manifest_path)
    profile["database_cache_root"] = "/rotated/cache"
    profile["database_cache_unix_user"] = "retry-user"
    profile["database_cache_filesystem_type"] = "xfs"
    profile["database_cache_reserve_bytes"] = 4096
    profile["database_lock_wait_seconds"] = 120
    fixture.profile_path.write_text(yaml.safe_dump(profile_mapping, sort_keys=True))

    def interrupt_after_event(validation: PhaseAuthorityValidation) -> None:
        assert validation.current_runspec_projection_complete is False
        raise OSError("stop after fresh-authority Retry event")

    with pytest.raises(OSError, match="fresh-authority Retry event"):
        retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: RETRY_TIME,
            authority_store=PhaseAuthorityStore(
                authority_root,
                after_retry_event_publish=interrupt_after_event,
            ),
        )

    fixture.profile_path.unlink()
    rotated_manifest_path.unlink()
    retry_phase(
        FIXED_RUN_ID,
        authority_root=authority_root,
        config_path=tmp_path / "must-not-read-profile.yaml",
        source_repo=tmp_path / "must-not-read-source",
        clock=lambda: (_ for _ in ()).throw(AssertionError("clock must not run")),
    )
    complete = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    successor = complete.phase_runspec.payload.database
    assert successor.database_set == before.database_set
    assert successor.requested_policy == before.requested_policy
    assert successor.selected_container_root == before.selected_container_root
    assert successor.primary_database_name == before.primary_database_name
    assert successor.metagenomic_database_name == before.metagenomic_database_name
    assert successor.source_manifest == rotated_manifest
    assert successor.source_manifest_sha256 != before.source_manifest_sha256
    assert successor.staging is not None
    assert successor.staging.cache_root == "/rotated/cache"
    assert successor.staging.unix_user == "retry-user"
    assert successor.staging.expected_filesystem_type == "xfs"
    assert successor.staging.reserve_bytes == 4096
    assert successor.staging.lock_wait_seconds == 120

    def frozen_branch_authority(database: PreprocessingDatabaseBinding) -> tuple[object, ...]:
        branches = database.branches
        return tuple(
            (
                branch.branch_kind,
                branch.authorized_outcomes,
                branch.gpuserver_argv,
                branch.search_argv,
                tuple(
                    (mount.target, mount.purpose, mount.read_only)
                    for mounts in (
                        branch.placement_mounts,
                        branch.scientific_mounts,
                        branch.finalization_mounts,
                    )
                    for mount in mounts
                ),
            )
            for branch in branches
        )

    assert frozen_branch_authority(successor) == frozen_branch_authority(before)
    projection = complete.authority_path / successor.source_manifest_projection
    assert projection.read_bytes() == canonical_database_source_manifest_bytes(rotated_manifest)
    assert stat.S_IMODE(projection.stat().st_mode) == 0o444


def test_active_retry_rejects_before_clock_config_or_filesystem_changes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    before = _file_bytes(authority_root / FIXED_RUN_ID)

    with pytest.raises(ValueError, match="active Phase Attempt boundary"):
        retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=tmp_path / "missing.yaml",
            clock=lambda: (_ for _ in ()).throw(AssertionError("clock must not run")),
        )
    assert _file_bytes(authority_root / FIXED_RUN_ID) == before


def test_retry_rejects_array_resources_without_successor_publication(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _set_gpu_array(fixture, "0-3%2")
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    cancelled = cancel_phase(FIXED_RUN_ID, authority_root=authority_root, clock=lambda: RETRY_TIME)
    assert cancelled.status == "cancelled"
    before = _file_bytes(authority_root / FIXED_RUN_ID)

    with pytest.raises(ValueError, match="Retry does not support Slurm arrays"):
        retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: RETRY_TIME,
        )

    assert _file_bytes(authority_root / FIXED_RUN_ID) == before
    assert not (authority_root / FIXED_RUN_ID / "attempts" / "attempt-0002").exists()


@pytest.mark.parametrize("submission_state", ["planned", "dispatching", "submitted"])
def test_submitting_and_submitted_attempts_reject_retry_before_operational_resolution(
    tmp_path: Path,
    submission_state: str,
) -> None:
    fixture = _fixture(tmp_path, policy=DatabaseAccessPolicy.DIRECT)
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    _record_submission_state(authority_root, FIXED_RUN_ID, submission_state)
    before = _file_bytes(authority_root / FIXED_RUN_ID)

    with pytest.raises(ValueError, match="active Phase Attempt boundary"):
        retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=tmp_path / "missing.yaml",
            clock=lambda: (_ for _ in ()).throw(AssertionError("clock must not run")),
        )
    assert _file_bytes(authority_root / FIXED_RUN_ID) == before


def test_retry_event_before_projection_recovers_without_resolving_operational_inputs(tmp_path: Path) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)

    def interrupt_after_event(validation: PhaseAuthorityValidation) -> None:
        assert validation.current_runspec_projection_complete is False
        raise OSError("simulated Retry projection interruption")

    interrupted_store = PhaseAuthorityStore(authority_root, after_retry_event_publish=interrupt_after_event)
    with pytest.raises(OSError, match="projection interruption"):
        retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: RETRY_TIME,
            authority_store=interrupted_store,
        )

    incomplete = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert incomplete.current_attempt.attempt_id == "attempt-0002"
    assert not incomplete.current_runspec_projection_complete
    report = status_phase(FIXED_RUN_ID, authority_root=authority_root)
    assert not report.runspec_projection_complete
    with pytest.raises(ValueError, match="rerunning Phase Retry"):
        submit_phase(FIXED_RUN_ID, authority_root=authority_root, runner=lambda _argv: _unexpected_effect())
    with pytest.raises(ValueError, match="rerunning Phase Retry"):
        resume_phase(FIXED_RUN_ID, authority_root=authority_root, runner=lambda _argv: _unexpected_effect())
    with pytest.raises(ValueError, match="rerunning Phase Retry"):
        cancel_phase(FIXED_RUN_ID, authority_root=authority_root, runner=lambda _argv: _unexpected_effect())
    with pytest.raises(ValueError, match="rerunning Phase Retry"):
        finalize_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            scheduler_evidence_path=tmp_path / "must-not-read-scheduler.json",
            action_evidence_path=tmp_path / "must-not-read-action.json",
            handoff_path=tmp_path / "must-not-read-handoff",
        )

    recovered = retry_phase(
        FIXED_RUN_ID,
        authority_root=authority_root,
        config_path=tmp_path / "not-consulted.yaml",
        clock=lambda: (_ for _ in ()).throw(AssertionError("clock must not run")),
    )
    assert recovered.successor_attempt_id == "attempt-0002"
    complete = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert complete.current_runspec_projection_complete
    assert (
        complete.authority_path / complete.phase_runspec.payload.database.source_manifest_projection
    ).read_bytes() == canonical_database_source_manifest_bytes(complete.phase_runspec.payload.database.source_manifest)
    assert (
        stat.S_IMODE(
            (complete.authority_path / complete.phase_runspec.payload.database.source_manifest_projection)
            .stat()
            .st_mode
        )
        == 0o444
    )
    with pytest.raises(ValueError, match="active Phase Attempt boundary"):
        retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=tmp_path / "not-consulted.yaml",
            clock=lambda: (_ for _ in ()).throw(AssertionError("clock must not run")),
        )


def test_retry_pre_event_fault_leaves_no_event_attempt_or_staging_bytes(tmp_path: Path) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    before = _file_bytes(authority_root / FIXED_RUN_ID)

    def interrupt_before_event(_staged_event: Path) -> None:
        raise OSError("stop before Retry event publication")

    with pytest.raises(OSError, match="before Retry event"):
        retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: RETRY_TIME,
            authority_store=PhaseAuthorityStore(
                authority_root,
                before_event_publish=interrupt_before_event,
            ),
        )

    authority = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert authority.current_attempt.attempt_id == "attempt-0001"
    assert _file_bytes(authority.authority_path) == before
    assert not list(authority_root.glob(".phase-runspec-*.json"))
    assert not (authority.authority_path / "attempts" / "attempt-0002").exists()


@pytest.mark.parametrize(
    ("fault_hook", "projection_complete"),
    [
        ("after_retry_attempt_directory_create", False),
        ("after_retry_runspec_link", True),
        ("before_retry_projection_replay", True),
    ],
)
def test_retry_projection_fault_windows_recover_to_one_exact_authority(
    tmp_path: Path,
    fault_hook: str,
    projection_complete: bool,
) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    observed_paths: list[Path] = []

    def interrupt(path: Path) -> None:
        observed_paths.append(path)
        raise OSError(f"stop at {fault_hook}")

    with pytest.raises(OSError, match=fault_hook):
        retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: RETRY_TIME,
            authority_store=PhaseAuthorityStore(
                authority_root,
                **{fault_hook: interrupt},
            ),
        )

    assert len(observed_paths) == 1
    store = PhaseAuthorityStore(authority_root)
    interrupted = store.validate(FIXED_RUN_ID)
    assert interrupted.current_attempt.attempt_id == "attempt-0002"
    assert interrupted.current_runspec_projection_complete is projection_complete
    assert len([event for event in interrupted.events if event.event_type == "phase-attempt-retried"]) == 1
    if projection_complete:
        before = _file_bytes(interrupted.authority_path)
        first = store.publish_current_attempt_runspec_projection(FIXED_RUN_ID)
        second = store.publish_current_attempt_runspec_projection(FIXED_RUN_ID)
        assert first == second
        assert _file_bytes(interrupted.authority_path) == before
        with pytest.raises(ValueError, match="active Phase Attempt boundary"):
            retry_phase(
                FIXED_RUN_ID,
                authority_root=authority_root,
                config_path=tmp_path / "not-consulted.yaml",
                clock=lambda: (_ for _ in ()).throw(AssertionError("clock must not run")),
            )
    else:
        recovered = retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=tmp_path / "not-consulted.yaml",
            clock=lambda: (_ for _ in ()).throw(AssertionError("clock must not run")),
        )
        assert recovered.successor_attempt_id == "attempt-0002"
    complete = store.validate(FIXED_RUN_ID)
    projection = complete.authority_path / complete.current_attempt.phase_runspec_location
    manifest_projection = complete.authority_path / complete.phase_runspec.payload.database.source_manifest_projection
    expected = (json.dumps(complete.phase_runspec.to_mapping(), indent=2, sort_keys=True) + "\n").encode()
    assert projection.read_bytes() == expected
    assert manifest_projection.read_bytes() == canonical_database_source_manifest_bytes(
        complete.phase_runspec.payload.database.source_manifest
    )
    assert not list(authority_root.glob(".phase-runspec-*.json"))


def test_terminal_retry_event_accepts_only_its_latest_canonical_partial_projection(tmp_path: Path) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)

    def interrupt_after_event(_validation: object) -> None:
        raise OSError("stop after Retry event")

    with pytest.raises(OSError, match="stop after Retry event"):
        retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: RETRY_TIME,
            authority_store=PhaseAuthorityStore(
                authority_root,
                after_retry_event_publish=interrupt_after_event,
            ),
        )
    attempt_directory = authority_root / FIXED_RUN_ID / "attempts" / "attempt-0002"
    attempt_directory.mkdir()
    directory_only = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert not directory_only.current_runspec_projection_complete
    (attempt_directory / "phase-runspec.json").write_text("{}\n")
    with pytest.raises(ValueError):
        PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    with pytest.raises(ValueError):
        PhaseAuthorityStore(authority_root).publish_current_attempt_runspec_projection(FIXED_RUN_ID)


def test_typed_retry_prepass_rejects_undeclared_attempt_directory(tmp_path: Path) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)

    def interrupt_after_event(_validation: object) -> None:
        raise OSError("stop after Retry event")

    with pytest.raises(OSError, match="stop after Retry event"):
        retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: RETRY_TIME,
            authority_store=PhaseAuthorityStore(
                authority_root,
                after_retry_event_publish=interrupt_after_event,
            ),
        )
    (authority_root / FIXED_RUN_ID / "attempts" / "attempt-9999").mkdir()
    with pytest.raises(ValueError, match="layout mismatch"):
        PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)


@pytest.mark.parametrize(
    "tamper",
    ["envelope", "payload", "embedded-attempt", "embedded-runspec"],
)
def test_typed_retry_prepass_rejects_raw_retry_tampering(tmp_path: Path, tamper: str) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    incomplete = _leave_terminal_retry_event(fixture, authority_root)
    event_path = next((incomplete.authority_path / "events").glob("*-phase-attempt-retried.json"))
    mapping = json.loads(event_path.read_text())
    if tamper == "envelope":
        mapping["attempt_id"] = "attempt-0003"
    elif tamper == "payload":
        mapping["payload"]["phase_plan_digest"] = "0" * 64
    elif tamper == "embedded-attempt":
        mapping["payload"]["successor_attempt"]["phase_runspec_location"] = "attempts/attempt-9999/phase-runspec.json"
    else:
        mapping["payload"]["successor_phase_runspec"]["attempt_id"] = "attempt-0003"
    event_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError):
        PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)


@pytest.mark.parametrize("chain_fault", ["duplicate", "gapped"])
def test_typed_retry_prepass_rejects_duplicate_and_gapped_attempt_chains(
    tmp_path: Path,
    chain_fault: str,
) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    incomplete = _leave_terminal_retry_event(fixture, authority_root)
    first_retry = next(event for event in incomplete.events if isinstance(event, PhaseAttemptRetriedEvent))
    sequence = len(incomplete.events) + 1
    event = (
        replace(first_retry, sequence=sequence)
        if chain_fault == "duplicate"
        else _raw_retry_event(
            incomplete,
            sequence=sequence,
            predecessor_attempt_id="attempt-0003",
            successor_ordinal=4,
        )
    )
    event_path = incomplete.authority_path / "events" / f"{sequence:06d}-phase-attempt-retried.json"
    event_path.write_text(json.dumps(event.to_mapping(), indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="contiguous predecessor/successor chain"):
        PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)


def test_missing_latest_retry_projection_rejects_when_a_later_event_exists(tmp_path: Path) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    incomplete = _leave_terminal_retry_event(fixture, authority_root)
    attempt = incomplete.current_attempt
    intent = render_phase_submission_intent(
        incomplete.phase_runspec,
        phase_runspec_location=attempt.phase_runspec_location,
        phase_runspec_document_sha256="0" * 64,
    )
    sequence = len(incomplete.events) + 1
    later = PhaseSubmissionIntendedEvent(
        sequence=sequence,
        phase_run_id=FIXED_RUN_ID,
        attempt_id=attempt.attempt_id,
        occurred_at="2026-08-20T13:01:00.000000Z",
        payload=intent,
    )
    event_path = incomplete.authority_path / "events" / f"{sequence:06d}-phase-submission-intended.json"
    event_path.write_text(json.dumps(later.to_mapping(), indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="layout mismatch"):
        PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)


def test_full_authority_rejects_tampered_initial_phase_run_snapshot(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    snapshot_path = authority_root / FIXED_RUN_ID / "phase-run.json"
    snapshot = json.loads(snapshot_path.read_text())
    initial_attempt = snapshot["attempts"][0]
    initial_attempt["attempt_id"] = "attempt-0002"
    initial_attempt["ordinal"] = 2
    initial_attempt["phase_runspec_location"] = "attempts/attempt-0002/phase-runspec.json"
    snapshot["current_attempt_id"] = "attempt-0002"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="initial Phase Run"):
        PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)


def test_failed_successor_can_retry_to_contiguous_attempt_three(tmp_path: Path) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    retry_phase(
        FIXED_RUN_ID,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: RETRY_TIME,
    )
    _record_submission_state(authority_root, FIXED_RUN_ID, "submitted", job_id="654321")
    _record_terminal(authority_root, FIXED_RUN_ID, state="FAILED", exit_code="1:0", outcome="failed")
    failed = status_phase(FIXED_RUN_ID, authority_root=authority_root)
    assert failed.attempt_id == "attempt-0002"
    assert failed.attempt_status == failed.status == "failed"
    third_time = datetime(2026, 8, 21, 13, 0, tzinfo=UTC)
    result = retry_phase(
        FIXED_RUN_ID,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: third_time,
    )

    validation = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert result.successor_attempt_id == "attempt-0003"
    assert validation.current_attempt.ordinal == 3
    assert [item.attempt.attempt_id for item in validation.prior_attempts] == ["attempt-0001", "attempt-0002"]
    assert [item.outcome for item in validation.prior_attempts] == ["failed", "failed"]
    (authority_root / FIXED_RUN_ID / "attempts" / "attempt-0002" / "phase-runspec.json").unlink()
    with pytest.raises(ValueError, match="layout mismatch"):
        PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)


def test_successor_cancellation_scopes_repeated_action_and_terminal_ids_to_attempt_two(tmp_path: Path) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    retry_phase(
        FIXED_RUN_ID,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: RETRY_TIME,
    )
    _record_submission_state(authority_root, FIXED_RUN_ID, "submitted", job_id="654321")
    first = cancel_phase(
        FIXED_RUN_ID,
        authority_root=authority_root,
        clock=lambda: RETRY_TIME,
        runner=CancellationRunner(accounting_states=[("RUNNING", None)]),
    )
    cancelling = status_phase(FIXED_RUN_ID, authority_root=authority_root)
    assert first.status == "cancelling"
    assert cancelling.attempt_id == "attempt-0002"
    assert cancelling.attempt_status == cancelling.status == "cancelling"
    with pytest.raises(ValueError, match="cancelling Phase Attempt boundary"):
        retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=tmp_path / "missing.yaml",
            clock=lambda: (_ for _ in ()).throw(AssertionError("clock must not run")),
        )

    result = cancel_phase(
        FIXED_RUN_ID,
        authority_root=authority_root,
        clock=lambda: RETRY_TIME,
        runner=CancellationRunner(accounting_states=[("CANCELLED", "0:15")]),
    )

    authority = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    cancelled = status_phase(FIXED_RUN_ID, authority_root=authority_root)
    assert result.status == "cancelled"
    assert result.attempt_id == "attempt-0002"
    assert cancelled.attempt_id == "attempt-0002"
    assert cancelled.attempt_status == cancelled.status == "cancelled"
    assert result.target_job_ids == ("654321",)
    assert authority.lifecycle.current_attempt_id == "attempt-0002"
    assert authority.cancellation is not None and authority.cancellation.attempt_id == "attempt-0002"
    assert authority.prior_attempts[0].terminal_observations[0].job_id == "123456"
    assert authority.terminal_observations[0].job_id == "654321"


@pytest.mark.parametrize("operation", ["submit", "resume", "cancel"])
def test_successor_dispatch_correlation_uses_its_own_durable_intent_timestamp(
    tmp_path: Path,
    operation: str,
) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    retry_phase(
        FIXED_RUN_ID,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: RETRY_TIME,
    )
    successor_intended_at = "2026-08-20T14:00:00.000000Z"
    _record_submission_state(
        authority_root,
        FIXED_RUN_ID,
        "dispatching",
        intended_at=successor_intended_at,
    )

    authority = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert _submission_intended_at(authority) == successor_intended_at
    assert authority.prior_attempts[0].submission is not None
    assert authority.submission is not None
    assert authority.prior_attempts[0].submission.submission_id != authority.submission.submission_id

    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        if argv[0] in {"squeue", "sacct"}:
            return CommandResult(argv, 0, json.dumps({"jobs": []}), "")
        raise AssertionError(f"unexpected external effect: {argv}")

    if operation == "submit":
        with pytest.raises(ValueError, match="remains uncertain"):
            submit_phase(FIXED_RUN_ID, authority_root=authority_root, clock=lambda: RETRY_TIME, runner=runner)
    elif operation == "resume":
        with pytest.raises(ValueError, match="remains uncertain"):
            resume_phase(FIXED_RUN_ID, authority_root=authority_root, clock=lambda: RETRY_TIME, runner=runner)
    else:
        result = cancel_phase(FIXED_RUN_ID, authority_root=authority_root, clock=lambda: RETRY_TIME, runner=runner)
        assert result.status == "cancelling"

    correlation_sacct = next(call for call in calls if call[0] == "sacct" and any("--name=" in arg for arg in call))
    assert "--starttime=2026-08-20T14:00:00" in correlation_sacct


def test_concurrent_retry_creates_one_successor_and_one_effect_free_loser(tmp_path: Path) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    start = threading.Barrier(2)

    def invoke() -> object:
        start.wait()
        try:
            return retry_phase(
                FIXED_RUN_ID,
                authority_root=authority_root,
                config_path=fixture.profile_path,
                source_repo=fixture.source_repo,
                clock=lambda: RETRY_TIME,
            )
        except ValueError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(lambda _index: invoke(), range(2)))

    successes = [item for item in outcomes if not isinstance(item, ValueError)]
    rejections = [item for item in outcomes if isinstance(item, ValueError)]
    assert len(successes) == len(rejections) == 1
    assert "active Phase Attempt boundary" in str(rejections[0])
    authority = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert authority.current_attempt.attempt_id == "attempt-0002"
    assert len(authority.prior_attempts) == 1
    assert len([event for event in authority.events if event.event_type == "phase-attempt-retried"]) == 1
    assert sorted(path.name for path in (authority.authority_path / "attempts").iterdir()) == [
        "attempt-0001",
        "attempt-0002",
    ]


@pytest.mark.parametrize("operation", ["submit", "resume", "cancel", "finalize"])
def test_retry_serializes_after_failed_attempt_operations_without_external_effects(
    tmp_path: Path,
    operation: str,
) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    operation_entered = threading.Event()
    release_operation = threading.Event()
    retry_started = threading.Event()

    class BlockingValidationStore(PhaseAuthorityStore):
        def validate(self, phase_run_id: str) -> PhaseAuthorityValidation:
            operation_entered.set()
            if not release_operation.wait(timeout=5):
                raise AssertionError("test did not release blocked lifecycle validation")
            return super().validate(phase_run_id)

    operation_store = BlockingValidationStore(authority_root)

    def reject_effect(_argv: tuple[str, ...]) -> CommandResult:
        raise AssertionError(f"{operation} must reject the failed Attempt before scheduler effects")

    def invoke_operation() -> object:
        try:
            if operation == "submit":
                return submit_phase(
                    FIXED_RUN_ID,
                    authority_root=authority_root,
                    authority_store=operation_store,
                    runner=reject_effect,
                )
            if operation == "resume":
                return resume_phase(
                    FIXED_RUN_ID,
                    authority_root=authority_root,
                    authority_store=operation_store,
                    runner=reject_effect,
                )
            if operation == "cancel":
                return cancel_phase(
                    FIXED_RUN_ID,
                    authority_root=authority_root,
                    authority_store=operation_store,
                    runner=reject_effect,
                )
            return finalize_phase(
                FIXED_RUN_ID,
                authority_root=authority_root,
                authority_store=operation_store,
                scheduler_evidence_path=tmp_path / "must-not-read-scheduler.json",
                action_evidence_path=tmp_path / "must-not-read-action.json",
                handoff_path=tmp_path / "must-not-read-handoff",
            )
        except Exception as exc:
            return exc

    def invoke_retry() -> object:
        retry_started.set()
        return retry_phase(
            FIXED_RUN_ID,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: RETRY_TIME,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        operation_future = pool.submit(invoke_operation)
        assert operation_entered.wait(timeout=5)
        retry_future = pool.submit(invoke_retry)
        assert retry_started.wait(timeout=5)
        assert not retry_future.done()
        release_operation.set()
        operation_outcome = operation_future.result(timeout=5)
        retry_outcome = retry_future.result(timeout=5)

    assert isinstance(operation_outcome, ValueError)
    assert getattr(retry_outcome, "successor_attempt_id", None) == "attempt-0002"
    authority = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert authority.current_attempt.attempt_id == "attempt-0002"
    assert authority.submission is None
    assert authority.cancellation is None


def test_phase_retry_cli_emits_stable_scheduler_free_result(tmp_path: Path) -> None:
    fixture, authority_root = _failed_materialized_run(tmp_path)
    result = CliRunner().invoke(
        cli,
        (
            "--config",
            str(fixture.profile_path),
            "phase",
            "retry",
            FIXED_RUN_ID,
            "--authority-root",
            str(authority_root),
            "--source-repo",
            str(fixture.source_repo),
        ),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["successor_attempt_id"] == "attempt-0002"
    assert payload["status"] == "materialized"
    assert "job_id" not in payload


def _unexpected_effect() -> CommandResult:
    raise AssertionError("operation must reject an incomplete Retry projection before remote effects")
