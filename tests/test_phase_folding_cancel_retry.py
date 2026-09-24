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

"""Contract and live-lifecycle tests for folding cancel + retry closure.

These tests pin the two seams that e08s03 left fail-closed: the generalized
``_ACTION_ID`` grammar and the family-dispatched Retry successor loader on the
contract side, and the live generic cancel/retry transitions on the control
side. Preprocessing and postprocessing acceptance/rejection stay byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from bspp.orchestration.contract.folding_carry_forward import (
    FoldingCarryForwardRecord,
    FoldingCarryForwardReference,
)
from bspp.orchestration.contract.folding_execution import FoldingBackendAssetsSnapshot
from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.phase import (
    FoldingPhasePlan,
    FoldingPhasePlanPayload,
    FoldingPhaseRunSpec,
    PhaseRunSpec,
    canonical_mapping_digest,
)
from bspp.orchestration.contract.phase_cancellation import (
    _ACTION_ID as _CANCELLATION_ACTION_ID,
)
from bspp.orchestration.contract.phase_cancellation import (
    PhaseCancellationActionTarget,
    PhaseCancellationCompletedEvent,
    PhaseCancellationIntendedEvent,
)
from bspp.orchestration.contract.phase_reconciliation import (
    _ACTION_ID as _RECONCILIATION_ACTION_ID,
)
from bspp.orchestration.contract.phase_reconciliation import (
    FoldingActionTerminalObservationView,
    FoldingActionTerminalObservedEvent,
    PhaseActionTerminalObservedEvent,
)
from bspp.orchestration.contract.phase_retry import (
    PhaseAttemptRetriedEvent,
    phase_attempt_retried_payload_from_mapping,
)
from bspp.orchestration.contract.phase_submission import (
    PhaseActionSatisfiedWithoutDispatchEvent,
    PhaseActionSubmittedEvent,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    BundledMemberVerification,
    MsaArtifactSetManifest,
    MsaChunkManifestReference,
    VerifiedLocalBundledArtifactLocation,
    msa_artifact_set_id,
    verified_local_bundled_artifact_location_id,
)
from bspp.orchestration.control.folding_phase_adapter import render_folding_submission_intent
from bspp.orchestration.control.folding_phase_types import folding_qualification_tuple_id
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore
from bspp.orchestration.control.phase_cancellation import cancel_phase
from bspp.orchestration.control.phase_materialization import materialize_phase
from bspp.orchestration.control.phase_resume import resume_phase
from bspp.orchestration.control.phase_retry import retry_phase
from bspp.orchestration.control.phase_status import status_phase
from bspp.orchestration.control.phase_submission import submit_phase
from bspp.orchestration.control.transport import CommandResult
from tests.test_phase_folding_lifecycle import (
    FIXED_RUN_ID,
    RUNTIME_IMAGE,
    FoldingCancellationRunner,
    FoldingSubmissionRunner,
    _materialize_folding,
    _phase_plan,
    _record_folding_terminal_failure,
    _write_profile,
)
from tests.test_phase_retry_contract import _contract_fixture

_FIXED_TIME = datetime(2026, 9, 11, 12, 0, 0, tzinfo=UTC)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_action_id_folding_acceptance_matrix() -> None:
    accepted = (
        "preprocessing-chunk-000001",
        "msa-flatten-000001",
        "split-000001",
        "preprocess-000001",
        "fold-000001",
        "canonical-pair-000001",
    )
    rejected = ("bogus-000001", "fold-00001", "fold-0000001", "preprocessing-chunk-0000010")
    for pattern in (_CANCELLATION_ACTION_ID, _RECONCILIATION_ACTION_ID):
        for action_id in accepted:
            assert pattern.fullmatch(action_id) is not None, action_id
        for action_id in rejected:
            assert pattern.fullmatch(action_id) is None, action_id

    target = PhaseCancellationActionTarget(action_id="fold-000001", initial_submission_status="not-submitted")
    assert target.action_id == "fold-000001"
    with pytest.raises(ValueError, match="cancellation target action id is invalid"):
        PhaseCancellationActionTarget(action_id="bogus-000001", initial_submission_status="not-submitted")


def test_retry_payload_family_dispatch(tmp_path: Path) -> None:
    event, _plan, _predecessor = _contract_fixture(tmp_path / "preprocessing")
    loaded = phase_attempt_retried_payload_from_mapping(event.payload.to_mapping())
    assert loaded == event.payload
    assert isinstance(loaded.successor_phase_runspec, PhaseRunSpec)

    postprocessing = deepcopy(event.payload.to_mapping())
    postprocessing["successor_phase_runspec"]["phase_kind"] = "postprocessing"
    with pytest.raises(ValueError):
        phase_attempt_retried_payload_from_mapping(postprocessing)

    folding_root = tmp_path / "folding"
    folding_root.mkdir()
    authority_root, phase_run_id = _materialize_folding(folding_root)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _record_folding_terminal_failure(authority_root, phase_run_id)
    retry_phase(
        phase_run_id,
        authority_root=authority_root,
        config_path=_write_profile(folding_root),
        clock=lambda: _FIXED_TIME,
    )
    validation = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    retry_event = next(event for event in validation.events if isinstance(event, PhaseAttemptRetriedEvent))
    folding_loaded = phase_attempt_retried_payload_from_mapping(retry_event.payload.to_mapping())
    assert isinstance(folding_loaded.successor_phase_runspec, FoldingPhaseRunSpec)


def test_folding_cancellation_drives_generic_append_only_transition(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    authority_path = authority_root / phase_run_id
    before = _tree_bytes(authority_path)
    runner = FoldingCancellationRunner()

    result = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=runner,
    )

    assert result.status == "cancelled"
    assert len([call for call in runner.calls if call[0] == "scancel"]) == 5
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert sum(isinstance(event, PhaseCancellationIntendedEvent) for event in replay.events) == 1
    assert sum(isinstance(event, PhaseCancellationCompletedEvent) for event in replay.events) == 1
    terminal_events = [event for event in replay.events if isinstance(event, PhaseActionTerminalObservedEvent)]
    assert len(terminal_events) == 5
    for relative, content in before.items():
        assert (authority_path / relative).read_bytes() == content


def test_folding_retry_materializes_one_clean_immutable_successor(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _record_folding_terminal_failure(authority_root, phase_run_id)
    authority_path = authority_root / phase_run_id
    before = _tree_bytes(authority_path)

    result = retry_phase(
        phase_run_id,
        authority_root=authority_root,
        config_path=_write_profile(tmp_path),
        clock=lambda: _FIXED_TIME,
    )

    assert result.successor_attempt_id == "attempt-0002"
    validation = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert isinstance(validation.phase_runspec, FoldingPhaseRunSpec)
    assert validation.current_attempt.attempt_id == "attempt-0002"
    assert validation.submission is None
    assert len(validation.prior_attempts) == 1
    assert validation.prior_attempts[0].outcome == "failed"
    assert sum(isinstance(event, PhaseAttemptRetriedEvent) for event in validation.events) == 1
    for relative, content in before.items():
        assert (authority_path / relative).read_bytes() == content
    assert (authority_path / "attempts" / "attempt-0002" / "phase-runspec.json").is_file()


def test_folding_retry_writes_successor_fold_shard_projection(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _record_folding_terminal_failure(authority_root, phase_run_id)

    retry_phase(
        phase_run_id,
        authority_root=authority_root,
        config_path=_write_profile(tmp_path),
        clock=lambda: _FIXED_TIME,
    )

    authority_path = authority_root / phase_run_id
    successor_projection = authority_path / "attempts" / "attempt-0002" / "fold-shard-projection.json"
    assert successor_projection.is_file()
    assert not successor_projection.is_symlink()

    validation = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert isinstance(validation.phase_runspec, FoldingPhaseRunSpec)
    binding = validation.phase_runspec.payload.fold_shard_projection
    assert binding is not None
    assert binding.location == "attempts/attempt-0002/fold-shard-projection.json"

    from bspp.orchestration.control.folding_shard import (
        fold_shard_projection_document_bytes,
        fold_shard_projection_from_runspec,
    )

    expected_projection, expected_binding = fold_shard_projection_from_runspec(
        validation.phase_plan, validation.phase_runspec
    )
    assert expected_binding == binding
    assert successor_projection.read_bytes() == fold_shard_projection_document_bytes(expected_projection)


def test_folding_retry_resolves_changed_assets_into_successor(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _record_folding_terminal_failure(authority_root, phase_run_id)
    authority_path = authority_root / phase_run_id
    predecessor = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    predecessor_digest = predecessor.phase_runspec.digest
    before = _tree_bytes(authority_path)

    profile_b = tmp_path / "profiles-b.yaml"
    profile_b.write_text(
        yaml.safe_dump(
            {
                "clusters": {
                    "example-cluster": {
                        "owner": "tester",
                        "transport": "local-slurm",
                        "project_root": str(tmp_path / "project"),
                        "output_root": str(tmp_path / "output"),
                        "staging_root": str(tmp_path / "staging"),
                        "orchestration_repo": str(tmp_path / "orchestration"),
                        "image": "registry/bspp-runtime:latest",
                        "folding_backend_images": [
                            {"backend": "openfold-cli", "image": "registry/bspp-folding-openfold-cli:latest"},
                        ],
                        "folding_backend_assets": [
                            {
                                "backend": "openfold-cli",
                                "chain_manifest_csv": "/assets/chains-b.csv",
                                "openfold_model_dir": "/assets/models-b",
                            }
                        ],
                        "extra_mounts": [
                            {"source": "/assets/chains-b.csv", "target": "/assets/chains-b.csv", "read_only": True},
                            {"source": "/assets/models-b", "target": "/assets/models-b", "read_only": True},
                        ],
                    }
                }
            },
            sort_keys=True,
        )
    )

    result = retry_phase(
        phase_run_id,
        authority_root=authority_root,
        config_path=profile_b,
        clock=lambda: _FIXED_TIME,
    )

    assert result.successor_attempt_id == "attempt-0002"
    validation = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert isinstance(validation.phase_runspec, FoldingPhaseRunSpec)
    # The successor retains the exact predecessor Plan root manifest while
    # resolving a fresh backend-assets snapshot.
    assert predecessor.phase_plan.payload.msa_set_manifest is not None
    assert validation.phase_runspec.payload.msa_set_manifest == predecessor.phase_plan.payload.msa_set_manifest
    assert validation.phase_runspec.cluster.backend_assets == FoldingBackendAssetsSnapshot(
        backend="openfold-cli",
        chain_manifest_csv="/assets/chains-b.csv",
        openfold_model_dir="/assets/models-b",
    )
    assert validation.phase_runspec.digest != predecessor_digest
    for relative, content in before.items():
        assert (authority_path / relative).read_bytes() == content

    intent_a = render_folding_submission_intent(
        phase_runspec=predecessor.phase_runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    intent_b = render_folding_submission_intent(
        phase_runspec=validation.phase_runspec,
        phase_runspec_location="attempts/attempt-0002/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    assert intent_a.qualification_tuple_id != intent_b.qualification_tuple_id


def _write_packed_profile(tmp_path: Path) -> Path:
    profile_path = tmp_path / "profiles-packed.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "clusters": {
                    "example-cluster": {
                        "owner": "tester",
                        "transport": "local-slurm",
                        "project_root": str(tmp_path / "project"),
                        "output_root": str(tmp_path / "output"),
                        "staging_root": str(tmp_path / "staging"),
                        "orchestration_repo": str(tmp_path / "orchestration"),
                        "image": RUNTIME_IMAGE,
                        "folding_backend_images": [
                            {"backend": "openfold-cli", "image": "registry/bspp-folding-openfold-cli:latest"},
                        ],
                        "folding_backend_assets": [
                            {
                                "backend": "openfold-cli",
                                "chain_manifest_csv": "/assets/chains.csv",
                                "openfold_model_dir": "/assets/models",
                            }
                        ],
                        "extra_mounts": [
                            {"source": "/assets/chains.csv", "target": "/assets/chains.csv", "read_only": True},
                            {"source": "/assets/models", "target": "/assets/models", "read_only": True},
                        ],
                        "resources": {
                            "gpu_worker": {
                                "partition": "gpu",
                                "cpus_per_task": 8,
                                "memory": "64G",
                                "time": "01:00:00",
                                "gres": None,
                                "nodes": 3,
                                "tasks_per_node": 2,
                                "gpus_per_task": 1,
                                "max_parallel": 2,
                            }
                        },
                    }
                }
            },
            sort_keys=True,
        )
    )
    return profile_path


def _materialize_packed_folding(tmp_path: Path) -> tuple[Path, str]:
    profile_path = _write_packed_profile(tmp_path)
    plan = _phase_plan(tmp_path)
    plan_path = tmp_path / "folding-phase-plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan.to_mapping(), sort_keys=True))
    authority_root = tmp_path / "authority"
    result = materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: _FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    return authority_root, result.phase_run_id


class FoldingArrayObservationRunner:
    """Synthesize exact sacct child rows for the packed fold array (parent 1004)."""

    def __init__(self, *, missing_child: bool = False, failed_task: int | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.missing_child = missing_child
        self.failed_task = failed_task

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "squeue":
            return CommandResult(argv, 0, json.dumps({"jobs": []}), "")
        if argv[0] == "sacct":
            job_ids = argv[argv.index("-j") + 1].split(",")
            rows: list[dict[str, object]] = []
            for parent in job_ids:
                if parent != "1004":
                    continue
                for index in (0, 1, 2):
                    if self.missing_child and index == 2:
                        continue
                    state = "FAILED" if self.failed_task == index else "COMPLETED"
                    exit_code = "1:0" if self.failed_task == index else "0:0"
                    rows.append(
                        {
                            "job_id_raw": f"{parent}_{index}",
                            "state": state,
                            "exit_code": exit_code,
                            "restarts": 0,
                        }
                    )
            return CommandResult(argv, 0, json.dumps({"jobs": rows}), "")
        raise AssertionError(f"unexpected command: {argv}")


def test_packed_fold_submission_declares_expected_task_indexes(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_packed_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )

    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert authority.submission is not None
    fold_view = next(item for item in authority.submission.actions if item.action_id == "fold-000001")
    assert fold_view.plan.expected_task_indexes == (0, 1, 2)
    assert fold_view.job_id == "1004"

    submitted = next(
        event
        for event in authority.events
        if isinstance(event, PhaseActionSubmittedEvent) and event.payload.action_id == "fold-000001"
    )
    assert submitted.payload.expected_task_indexes == (0, 1, 2)
    assert submitted.payload.job_id == "1004"
    # Non-fold actions stay scalar (no expected task indexes).
    msa_view = next(item for item in authority.submission.actions if item.action_id == "msa-flatten-000001")
    assert msa_view.plan.expected_task_indexes == ()


def test_packed_fold_array_success_yields_succeeded_terminal(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_packed_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )

    resumed = resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingArrayObservationRunner(),
    )

    assert resumed.outcome == "reconciled"
    assert "fold-000001" in resumed.terminal_action_ids
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    array_events = [event for event in authority.events if isinstance(event, FoldingActionTerminalObservedEvent)]
    assert len(array_events) == 1
    assert array_events[0].payload.parent_job_id == "1004"
    assert array_events[0].payload.expected_task_indexes == (0, 1, 2)
    assert array_events[0].payload.outcome == "succeeded"
    assert len(authority.array_terminal_observations) == 1
    assert authority.array_terminal_observations[0].outcome == "succeeded"
    # The attempt stays active: the canonical-pair action can still dispatch next cycle.
    assert authority.lifecycle.attempt_status == "materialized"
    assert authority.lifecycle.run_status == "materialized"


def test_packed_fold_array_failure_yields_failed_lifecycle(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_packed_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )

    resumed = resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingArrayObservationRunner(failed_task=1),
    )

    assert resumed.outcome == "failed"
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    array_events = [event for event in authority.events if isinstance(event, FoldingActionTerminalObservedEvent)]
    assert len(array_events) == 1
    assert array_events[0].payload.outcome == "failed"
    assert authority.lifecycle.attempt_status == "failed"
    assert authority.lifecycle.run_status == "failed"


def test_packed_fold_array_missing_child_stays_unresolved(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_packed_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )

    resumed = resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingArrayObservationRunner(missing_child=True),
    )

    assert resumed.outcome == "no-op"
    assert "fold-000001" not in resumed.terminal_action_ids
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert not [event for event in authority.events if isinstance(event, FoldingActionTerminalObservedEvent)]
    assert authority.array_terminal_observations == ()
    # No collapse to the parent row: the fold action has no scalar or array terminal view.
    fold_result = next(item for item in resumed.actions if item.action_id == "fold-000001")
    assert fold_result.terminal is None
    assert authority.lifecycle.attempt_status == "materialized"


class EmptyObservationRunner:
    """Return empty squeue/sacct rows so Status reports durable terminal facts only."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "squeue":
            return CommandResult(argv, 0, json.dumps({"jobs": []}), "")
        if argv[0] == "sacct":
            return CommandResult(argv, 0, json.dumps({"jobs": []}), "")
        raise AssertionError(f"unexpected command: {argv}")


@pytest.mark.parametrize("failed_task", [None, 1])
def test_status_projects_packed_fold_array_terminal_observation(tmp_path: Path, failed_task: int | None) -> None:
    authority_root, phase_run_id = _materialize_packed_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingArrayObservationRunner(failed_task=failed_task),
    )

    report = status_phase(
        phase_run_id,
        authority_root=authority_root,
        runner=EmptyObservationRunner(),
    )
    fold_status = next(item for item in report.actions if item.action_id == "fold-000001")
    assert isinstance(fold_status.terminal, FoldingActionTerminalObservationView)
    assert fold_status.terminal.parent_job_id == "1004"
    assert fold_status.terminal.expected_task_indexes == (0, 1, 2)
    assert [task.task_index for task in fold_status.terminal.tasks] == [0, 1, 2]
    assert [task.scheduler_job_id for task in fold_status.terminal.tasks] == ["1004_0", "1004_1", "1004_2"]
    expected_outcome = "failed" if failed_task is not None else "succeeded"
    assert fold_status.terminal.outcome == expected_outcome
    assert all(
        task.state == ("FAILED" if task.task_index == failed_task else "COMPLETED")
        for task in fold_status.terminal.tasks
    )
    assert all(
        task.exit_code == ("1:0" if task.task_index == failed_task else "0:0") for task in fold_status.terminal.tasks
    )

    # Scalar actions keep their scalar terminal shape (None before observation).
    msa_status = next(item for item in report.actions if item.action_id == "msa-flatten-000001")
    assert msa_status.terminal is None


def test_scalar_fold_keeps_existing_scalar_terminal_path(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    submitted = submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    assert [(item.action_id, item.job_id) for item in submitted.actions] == [
        ("msa-flatten-000001", "1001"),
        ("split-000001", "1002"),
        ("preprocess-000001", "1003"),
        ("fold-000001", "1004"),
        ("canonical-pair-000001", "1005"),
    ]

    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    fold_view = next(item for item in authority.submission.actions if item.action_id == "fold-000001")
    assert fold_view.plan.expected_task_indexes == ()
    submitted_event = next(
        event
        for event in authority.events
        if isinstance(event, PhaseActionSubmittedEvent) and event.payload.action_id == "fold-000001"
    )
    assert submitted_event.payload.expected_task_indexes == ()
    assert "expected_task_indexes" not in submitted_event.payload.to_mapping()
    assert "expected_task_indexes" not in fold_view.plan.to_mapping()


def _write_packed_profile_without_cap(tmp_path: Path) -> Path:
    """Packed profile with max_parallel omitted (uncapped array)."""
    profile_path = tmp_path / "profiles-packed-uncapped.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "clusters": {
                    "example-cluster": {
                        "owner": "tester",
                        "transport": "local-slurm",
                        "project_root": str(tmp_path / "project"),
                        "output_root": str(tmp_path / "output"),
                        "staging_root": str(tmp_path / "staging"),
                        "orchestration_repo": str(tmp_path / "orchestration"),
                        "image": RUNTIME_IMAGE,
                        "folding_backend_images": [
                            {"backend": "openfold-cli", "image": "registry/bspp-folding-openfold-cli:latest"},
                        ],
                        "folding_backend_assets": [
                            {
                                "backend": "openfold-cli",
                                "chain_manifest_csv": "/assets/chains.csv",
                                "openfold_model_dir": "/assets/models",
                            }
                        ],
                        "extra_mounts": [
                            {"source": "/assets/chains.csv", "target": "/assets/chains.csv", "read_only": True},
                            {"source": "/assets/models", "target": "/assets/models", "read_only": True},
                        ],
                        "resources": {
                            "gpu_worker": {
                                "partition": "gpu",
                                "cpus_per_task": 8,
                                "memory": "64G",
                                "time": "01:00:00",
                                "gres": None,
                                "nodes": 3,
                                "tasks_per_node": 2,
                                "gpus_per_task": 1,
                            }
                        },
                    }
                }
            },
            sort_keys=True,
        )
    )
    return profile_path


def test_packed_materialization_without_max_parallel_renders_uncapped_array(tmp_path: Path) -> None:
    profile_path = _write_packed_profile_without_cap(tmp_path)
    plan = _phase_plan(tmp_path)
    plan_path = tmp_path / "folding-phase-plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan.to_mapping(), sort_keys=True))
    authority_root = tmp_path / "authority"
    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: _FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )

    authority = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert isinstance(authority.phase_runspec, FoldingPhaseRunSpec)
    fold_action = next(action for action in authority.phase_runspec.payload.actions if action.action_kind == "fold")
    assert fold_action.resources.max_parallel is None
    assert fold_action.resources.array == "0-2"

    intent = render_folding_submission_intent(
        phase_runspec=authority.phase_runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    fold_plan = next(plan for plan in intent.actions if plan.action_id == "fold-000001")
    assert "#SBATCH --array=0-2" in fold_plan.script_body
    assert "%" not in fold_plan.script_body.split("--array=0-2")[1].split("\n")[0]


# --- e13s10: automatic folding carry-forward derivation ---

_TWO_TARGET_MEMBERS = (
    "a3ms/AFDB_AF-0000000000000001.a3m",
    "a3ms/AFDB_AF-0000000000000002.a3m",
)
_TWO_TARGET_IDS = ("AF-0000000000000001", "AF-0000000000000002")


def _two_target_bundled_location(tmp_path: Path, artifact_set_id: str) -> VerifiedLocalBundledArtifactLocation:
    bundle_path = tmp_path / "msa-set" / "msa-set.tar.lz4"
    tar_path = tmp_path / "msa-set" / "msa-set.tar"
    bundle_path.parent.mkdir(parents=True)
    bundle_bytes = b"bspp-fixture-lz4-bundle\n"
    tar_bytes = b"bspp-fixture-tar\n"
    bundle_path.write_bytes(bundle_bytes)
    tar_path.write_bytes(tar_bytes)
    members = tuple(
        BundledMemberVerification(
            logical_path=path,
            member_name=Path(path).name,
            raw_member_name=Path(path).name,
            size_bytes=1,
            sha256="b" * 64,
        )
        for path in _TWO_TARGET_MEMBERS
    )
    location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=Path(bundle_path).as_uri(),
        tar_size_bytes=len(tar_bytes),
        tar_sha256=hashlib.sha256(tar_bytes).hexdigest(),
        lz4_size_bytes=len(bundle_bytes),
        lz4_sha256=hashlib.sha256(bundle_bytes).hexdigest(),
        raw_tar_members=tuple(Path(path).name for path in _TWO_TARGET_MEMBERS),
        members=members,
    )
    return VerifiedLocalBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=Path(bundle_path).as_uri(),
        tar_size_bytes=len(tar_bytes),
        tar_sha256=hashlib.sha256(tar_bytes).hexdigest(),
        lz4_size_bytes=len(bundle_bytes),
        lz4_sha256=hashlib.sha256(bundle_bytes).hexdigest(),
        raw_tar_members=tuple(Path(path).name for path in _TWO_TARGET_MEMBERS),
        members=members,
        verified_at="2026-01-01T00:00:00Z",
    )


def _two_target_plan(tmp_path: Path) -> FoldingPhasePlan:
    member_lengths = (200, 100)
    chunk = MsaChunkManifestReference(
        chunk_name="foo_tranche00_00001.fa",
        logical_path="chunks/foo_tranche00_00001.json",
        sha256="f" * 64,
        member_count=2,
        logical_bytes=2,
    )
    manifest = MsaArtifactSetManifest(
        artifact_set_id=msa_artifact_set_id((chunk,), 2, 2, member_lengths=member_lengths),
        chunks=(chunk,),
        member_count=2,
        logical_bytes=2,
        member_lengths=member_lengths,
    )
    msa_set = MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=_TWO_TARGET_MEMBERS,
        requires_paired_query_header=True,
    )
    return FoldingPhasePlan(
        target_cluster="example-cluster",
        input_location=_two_target_bundled_location(tmp_path, manifest.artifact_set_id),
        payload=FoldingPhasePlanPayload(msa_set=msa_set, backend="openfold-cli", msa_set_manifest=manifest),
    )


def _write_two_worker_profile(tmp_path: Path) -> Path:
    profile_path = tmp_path / "profiles-two-worker.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "clusters": {
                    "example-cluster": {
                        "owner": "tester",
                        "transport": "local-slurm",
                        "project_root": str(tmp_path / "project"),
                        "output_root": str(tmp_path / "output"),
                        "staging_root": str(tmp_path / "staging"),
                        "orchestration_repo": str(tmp_path / "orchestration"),
                        "image": RUNTIME_IMAGE,
                        "folding_backend_images": [
                            {"backend": "openfold-cli", "image": "registry/bspp-folding-openfold-cli:latest"},
                        ],
                        "folding_backend_assets": [
                            {
                                "backend": "openfold-cli",
                                "chain_manifest_csv": "/assets/chains.csv",
                                "openfold_model_dir": "/assets/models",
                            }
                        ],
                        "extra_mounts": [
                            {"source": "/assets/chains.csv", "target": "/assets/chains.csv", "read_only": True},
                            {"source": "/assets/models", "target": "/assets/models", "read_only": True},
                        ],
                        "resources": {
                            "gpu_worker": {
                                "partition": "gpu",
                                "cpus_per_task": 8,
                                "memory": "64G",
                                "time": "01:00:00",
                                "gres": None,
                                "nodes": 1,
                                "tasks_per_node": 2,
                                "gpus_per_task": 1,
                                "max_parallel": 1,
                            }
                        },
                    }
                }
            },
            sort_keys=True,
        )
    )
    return profile_path


def _materialize_two_target_folding(tmp_path: Path) -> tuple[Path, str, Path]:
    profile_path = _write_two_worker_profile(tmp_path)
    plan = _two_target_plan(tmp_path)
    plan_path = tmp_path / "folding-phase-plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan.to_mapping(), sort_keys=True))
    authority_root = tmp_path / "authority"
    result = materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: _FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    return authority_root, result.phase_run_id, profile_path


def _folding_journal_path(
    tmp_path: Path,
    authority_root: Path,
    phase_run_id: str,
    *,
    rank: int,
    adopted: bool = False,
) -> Path:
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    runspec = authority.phase_runspec
    assert isinstance(runspec, FoldingPhaseRunSpec)
    fold_action = next(action for action in runspec.payload.actions if action.action_kind == "fold")
    journal_name = "adopted.jsonl" if adopted else "journal.jsonl"
    return (
        tmp_path
        / "project"
        / "bspp-phase-runs"
        / phase_run_id
        / runspec.attempt_id
        / "actions"
        / fold_action.action_id
        / "ranks"
        / str(rank)
        / journal_name
    )


def _write_folding_journal_event(
    tmp_path: Path,
    authority_root: Path,
    phase_run_id: str,
    *,
    rank: int,
    target_id: str,
    adopted: bool = False,
    predecessor_digest: str = "d" * 64,
    sequence_sha256: str = "e" * 64,
    carry_record_digest: str | None = None,
) -> None:
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    runspec = authority.phase_runspec
    assert isinstance(runspec, FoldingPhaseRunSpec)
    fold_action = next(action for action in runspec.payload.actions if action.action_kind == "fold")
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    kernel_image = next(value for key, value in fold_action.payload.params if key == "kernel_image")
    qualification_id = folding_qualification_tuple_id(
        backend=runspec.payload.backend,
        kernel_image=kernel_image,
        cluster_snapshot_digest=canonical_mapping_digest(runspec.cluster.to_mapping()),
    )
    outputs: list[dict[str, object]] = []
    for suffix, content in (
        ("structure.pdb", f"{runspec.attempt_id}-{rank}-{target_id}-structure\n"),
        ("scores.json", f"{runspec.attempt_id}-{rank}-{target_id}-scores\n"),
    ):
        output_path = tmp_path / "outputs" / f"{runspec.attempt_id}-{rank}-{target_id}-{suffix}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_bytes = content.encode()
        output_path.write_bytes(output_bytes)
        outputs.append(
            {
                "path": str(output_path),
                "size": len(output_bytes),
                "sha256": hashlib.sha256(output_bytes).hexdigest(),
            }
        )
    event: dict[str, object] = {
        "schema_version": 1,
        "phase_run_id": phase_run_id,
        "attempt_id": runspec.attempt_id,
        "rank": rank,
        "fold_action_id": fold_action.action_id,
        "fold_action_digest": canonical_mapping_digest(fold_action.to_mapping()),
        "shard_projection_sha256": binding.sha256,
        "shard_projection_worker_count": binding.worker_count,
        "shard_projection_lpt_version": binding.lpt_version,
        "predecessor_digest": predecessor_digest,
        "target_id": target_id,
        "sequence_sha256": sequence_sha256,
        "description": f"a3ms/{target_id}.a3m",
        "backend": runspec.payload.backend,
        "qualification_tuple_id": qualification_id,
        "outputs": outputs,
    }
    if adopted:
        carry = authority.current_carry_forward
        assert isinstance(carry, FoldingCarryForwardRecord)
        event["event_kind"] = "adopted"
        event["source_attempt_id"] = carry.source_attempt_id
        event["carry_record_digest"] = carry_record_digest if carry_record_digest is not None else carry.digest
    journal_path = _folding_journal_path(
        tmp_path,
        authority_root,
        phase_run_id,
        rank=rank,
        adopted=adopted,
    )
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _write_empty_folding_journal(
    tmp_path: Path,
    authority_root: Path,
    phase_run_id: str,
    *,
    rank: int,
    adopted: bool = False,
) -> None:
    journal_path = _folding_journal_path(
        tmp_path,
        authority_root,
        phase_run_id,
        rank=rank,
        adopted=adopted,
    )
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_bytes(b"")


def test_folding_retry_derives_partial_carry_record(tmp_path: Path) -> None:
    authority_root, phase_run_id, profile_path = _materialize_two_target_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _write_folding_journal_event(tmp_path, authority_root, phase_run_id, rank=0, target_id="AF-0000000000000001")
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=1)
    _record_folding_terminal_failure(authority_root, phase_run_id)

    result = retry_phase(
        phase_run_id,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: _FIXED_TIME,
    )

    assert result.successor_attempt_id == "attempt-0002"
    assert result.attempt_carry_forward_id is not None
    validation = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert isinstance(validation.phase_runspec, FoldingPhaseRunSpec)
    reference = validation.phase_runspec.carry_forward
    assert isinstance(reference, FoldingCarryForwardReference)
    assert reference.location == "attempts/attempt-0002/folding-carry-forward.json"
    record = validation.current_carry_forward
    assert isinstance(record, FoldingCarryForwardRecord)
    assert [item.target_id for item in record.content] == ["AF-0000000000000001"]
    assert record.source_attempt_id == "attempt-0001"
    assert record.target_attempt_id == "attempt-0002"
    assert record.ancestor_closure == ()
    assert (authority_root / phase_run_id / "attempts" / "attempt-0002" / "folding-carry-forward.json").is_file()


def test_folding_retry_zero_result_scan_omits_carry_record(tmp_path: Path) -> None:
    authority_root, phase_run_id, profile_path = _materialize_two_target_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=0)
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=1)
    _record_folding_terminal_failure(authority_root, phase_run_id)

    result = retry_phase(
        phase_run_id,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: _FIXED_TIME,
    )

    assert result.successor_attempt_id == "attempt-0002"
    assert result.attempt_carry_forward_id is None
    validation = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert isinstance(validation.phase_runspec, FoldingPhaseRunSpec)
    assert validation.phase_runspec.carry_forward is None
    assert validation.current_carry_forward is None
    assert not (authority_root / phase_run_id / "attempts" / "attempt-0002" / "folding-carry-forward.json").exists()


def test_folding_retry_missing_journal_is_scan_error(tmp_path: Path) -> None:
    authority_root, phase_run_id, profile_path = _materialize_two_target_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    # Rank 0 has no native journal; rank 1 is empty. Missing native journal is
    # a scan error, never an empty successful scan.
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=1)
    _record_folding_terminal_failure(authority_root, phase_run_id)

    with pytest.raises(ValueError):
        retry_phase(
            phase_run_id,
            authority_root=authority_root,
            config_path=profile_path,
            clock=lambda: _FIXED_TIME,
        )

    assert not (authority_root / phase_run_id / "attempts" / "attempt-0002").exists()


def test_folding_retry_malformed_journal_is_scan_error(tmp_path: Path) -> None:
    authority_root, phase_run_id, profile_path = _materialize_two_target_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    malformed = _folding_journal_path(tmp_path, authority_root, phase_run_id, rank=0)
    malformed.parent.mkdir(parents=True, exist_ok=True)
    # A malformed non-final line is a scan error; only a torn trailing append
    # (the final line) is ignored.
    malformed.write_bytes(b'{"schema_version": 1, "phase_run_id": "trunc\n\n')
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=1)
    _record_folding_terminal_failure(authority_root, phase_run_id)

    with pytest.raises(ValueError, match="malformed folding journal"):
        retry_phase(
            phase_run_id,
            authority_root=authority_root,
            config_path=profile_path,
            clock=lambda: _FIXED_TIME,
        )

    assert not (authority_root / phase_run_id / "attempts" / "attempt-0002").exists()


def test_folding_retry_complete_malformed_final_line_is_scan_error(tmp_path: Path) -> None:
    authority_root, phase_run_id, profile_path = _materialize_two_target_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    malformed = _folding_journal_path(tmp_path, authority_root, phase_run_id, rank=0)
    malformed.parent.mkdir(parents=True, exist_ok=True)
    # A complete (newline-terminated) malformed final line is a scan error, not
    # a torn append: only an unterminated trailing partial line may be ignored.
    malformed.write_bytes(b'{"schema_version": 1, "phase_run_id": "trunc"\n')
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=1)
    _record_folding_terminal_failure(authority_root, phase_run_id)

    with pytest.raises(ValueError, match="malformed folding journal"):
        retry_phase(
            phase_run_id,
            authority_root=authority_root,
            config_path=profile_path,
            clock=lambda: _FIXED_TIME,
        )

    assert not (authority_root / phase_run_id / "attempts" / "attempt-0002").exists()


def test_folding_retry_malformed_digest_is_scan_error(tmp_path: Path) -> None:
    authority_root, phase_run_id, profile_path = _materialize_two_target_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _write_folding_journal_event(
        tmp_path,
        authority_root,
        phase_run_id,
        rank=0,
        target_id="AF-0000000000000001",
        predecessor_digest="not-a-sha256",
    )
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=1)
    _record_folding_terminal_failure(authority_root, phase_run_id)

    with pytest.raises(ValueError, match="predecessor_digest must be a lowercase SHA-256"):
        retry_phase(
            phase_run_id,
            authority_root=authority_root,
            config_path=profile_path,
            clock=lambda: _FIXED_TIME,
        )

    assert not (authority_root / phase_run_id / "attempts" / "attempt-0002").exists()


def test_folding_retry_inconsistent_predecessor_digest_is_scan_error(tmp_path: Path) -> None:
    authority_root, phase_run_id, profile_path = _materialize_two_target_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _write_folding_journal_event(
        tmp_path,
        authority_root,
        phase_run_id,
        rank=0,
        target_id="AF-0000000000000001",
        predecessor_digest="a" * 64,
    )
    _write_folding_journal_event(
        tmp_path,
        authority_root,
        phase_run_id,
        rank=1,
        target_id="AF-0000000000000002",
        predecessor_digest="b" * 64,
    )
    _record_folding_terminal_failure(authority_root, phase_run_id)

    with pytest.raises(ValueError, match="disagree on the predecessor handoff digest"):
        retry_phase(
            phase_run_id,
            authority_root=authority_root,
            config_path=profile_path,
            clock=lambda: _FIXED_TIME,
        )

    assert not (authority_root / phase_run_id / "attempts" / "attempt-0002").exists()


def test_folding_retry_adopted_provenance_mismatch_is_scan_error(tmp_path: Path) -> None:
    authority_root, phase_run_id, profile_path = _materialize_two_target_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _write_folding_journal_event(tmp_path, authority_root, phase_run_id, rank=0, target_id="AF-0000000000000001")
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=1)
    _record_folding_terminal_failure(authority_root, phase_run_id)
    retry_phase(
        phase_run_id,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: _FIXED_TIME,
    )

    # attempt-0002 adopts target 001 with a forged carry_record_digest; the
    # adopted provenance must bind the predecessor carry record exactly.
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _write_folding_journal_event(
        tmp_path,
        authority_root,
        phase_run_id,
        rank=0,
        target_id="AF-0000000000000001",
        adopted=True,
        carry_record_digest="f" * 64,
    )
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=0)
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=1)
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=1, adopted=True)
    _record_folding_terminal_failure(authority_root, phase_run_id)

    with pytest.raises(ValueError, match="carry_record_digest does not bind"):
        retry_phase(
            phase_run_id,
            authority_root=authority_root,
            config_path=profile_path,
            clock=lambda: _FIXED_TIME,
        )

    assert not (authority_root / phase_run_id / "attempts" / "attempt-0003").exists()


def test_folding_retry_ancestor_closure_leads_with_predecessor(tmp_path: Path) -> None:
    authority_root, phase_run_id, profile_path = _materialize_two_target_folding(tmp_path)

    # attempt-0001 folds target 001 (rank 0) only; rank 1 is empty.
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _write_folding_journal_event(tmp_path, authority_root, phase_run_id, rank=0, target_id="AF-0000000000000001")
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=1)
    _record_folding_terminal_failure(authority_root, phase_run_id)

    first = retry_phase(
        phase_run_id,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: _FIXED_TIME,
    )
    assert first.successor_attempt_id == "attempt-0002"
    first_validation = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    first_record = first_validation.current_carry_forward
    assert isinstance(first_record, FoldingCarryForwardRecord)
    assert first_record.ancestor_closure == ()

    # attempt-0002 adopts carried target 001 (rank 0) and folds target 002 (rank 1).
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _write_folding_journal_event(
        tmp_path, authority_root, phase_run_id, rank=0, target_id="AF-0000000000000001", adopted=True
    )
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=0)
    _write_folding_journal_event(tmp_path, authority_root, phase_run_id, rank=1, target_id="AF-0000000000000002")
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=1, adopted=True)
    _record_folding_terminal_failure(authority_root, phase_run_id)

    second = retry_phase(
        phase_run_id,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: _FIXED_TIME,
    )
    assert second.successor_attempt_id == "attempt-0003"
    second_validation = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    second_record = second_validation.current_carry_forward
    assert isinstance(second_record, FoldingCarryForwardRecord)
    assert [item.target_id for item in second_record.content] == [
        "AF-0000000000000001",
        "AF-0000000000000002",
    ]
    assert second_record.ancestor_closure[0].folding_carry_forward_id == first_record.folding_carry_forward_id
    assert second_record.ancestor_closure[0].digest == first_record.digest
    assert len(second_record.ancestor_closure) == 1


def test_full_carry_skips_fold_dispatch_and_satisfies(tmp_path: Path) -> None:
    authority_root, phase_run_id, profile_path = _materialize_two_target_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _write_folding_journal_event(tmp_path, authority_root, phase_run_id, rank=0, target_id="AF-0000000000000001")
    _write_folding_journal_event(tmp_path, authority_root, phase_run_id, rank=1, target_id="AF-0000000000000002")
    _record_folding_terminal_failure(authority_root, phase_run_id)

    retry_phase(
        phase_run_id,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: _FIXED_TIME,
    )
    validation = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert isinstance(validation.current_carry_forward, FoldingCarryForwardRecord)
    assert len(validation.current_carry_forward.content) == 2

    runner = FoldingSubmissionRunner()
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=runner,
    )

    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.submission is not None
    fold_view = next(item for item in replay.submission.actions if item.action_id == "fold-000001")
    assert fold_view.status == "satisfied"
    assert fold_view.job_id is None
    satisfied = [event for event in replay.events if isinstance(event, PhaseActionSatisfiedWithoutDispatchEvent)]
    assert len(satisfied) == 1
    assert satisfied[0].payload.action_id == "fold-000001"
    assert satisfied[0].payload.carried_target_ids == (
        "AF-0000000000000001",
        "AF-0000000000000002",
    )
    assert satisfied[0].payload.phase_runspec_digest == replay.phase_runspec.digest
    assert satisfied[0].payload.carry_record_digest == validation.current_carry_forward.digest
    assert replay.submission.status == "submitted"
    canonical_view = next(item for item in replay.submission.actions if item.action_id == "canonical-pair-000001")
    assert canonical_view.status == "submitted"
    # The satisfied fold dependency contributes no Slurm job id, so canonical-pair
    # dispatches without --dependency=afterok:<fold>.
    assert canonical_view.dependency_job_ids == ()
    dispatch_events = [
        event
        for event in replay.events
        if getattr(event, "event_type", None) == "phase-action-submitted"
        and event.payload.submission_id == replay.submission.submission_id
    ]
    assert {event.payload.action_id for event in dispatch_events} == {
        "msa-flatten-000001",
        "split-000001",
        "preprocess-000001",
        "canonical-pair-000001",
    }


def test_partial_carry_submits_unchanged_full_array(tmp_path: Path) -> None:
    authority_root, phase_run_id, profile_path = _materialize_two_target_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _write_folding_journal_event(tmp_path, authority_root, phase_run_id, rank=0, target_id="AF-0000000000000001")
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=1)
    _record_folding_terminal_failure(authority_root, phase_run_id)

    predecessor = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert predecessor.submission is not None
    pred_fold = next(item for item in predecessor.submission.actions if item.action_id == "fold-000001")
    pred_array = next(line for line in pred_fold.plan.script_body.splitlines() if line.startswith("#SBATCH --array="))

    retry_phase(
        phase_run_id,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: _FIXED_TIME,
    )
    validation = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert isinstance(validation.current_carry_forward, FoldingCarryForwardRecord)
    assert len(validation.current_carry_forward.content) == 1

    runner = FoldingSubmissionRunner()
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=runner,
    )

    successor = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert successor.submission is not None
    succ_fold = next(item for item in successor.submission.actions if item.action_id == "fold-000001")
    # Partial carry still dispatches the fold action (no rebalancing) with the
    # unchanged full N-element array.
    assert succ_fold.status == "submitted"
    assert succ_fold.plan.expected_task_indexes == pred_fold.plan.expected_task_indexes == (0,)
    succ_array = next(line for line in succ_fold.plan.script_body.splitlines() if line.startswith("#SBATCH --array="))
    assert succ_array == pred_array


def test_folding_retry_topology_mismatch_fails_closed(tmp_path: Path) -> None:
    authority_root, phase_run_id, _ = _materialize_two_target_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: _FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _write_folding_journal_event(tmp_path, authority_root, phase_run_id, rank=0, target_id="AF-0000000000000001")
    _write_empty_folding_journal(tmp_path, authority_root, phase_run_id, rank=1)
    _record_folding_terminal_failure(authority_root, phase_run_id)

    changed_profile = tmp_path / "profiles-changed-topology.yaml"
    changed_profile.write_text(
        yaml.safe_dump(
            {
                "clusters": {
                    "example-cluster": {
                        "owner": "tester",
                        "transport": "local-slurm",
                        "project_root": str(tmp_path / "project"),
                        "output_root": str(tmp_path / "output"),
                        "staging_root": str(tmp_path / "staging"),
                        "orchestration_repo": str(tmp_path / "orchestration"),
                        "image": RUNTIME_IMAGE,
                        "folding_backend_images": [
                            {"backend": "openfold-cli", "image": "registry/bspp-folding-openfold-cli:latest"},
                        ],
                        "folding_backend_assets": [
                            {
                                "backend": "openfold-cli",
                                "chain_manifest_csv": "/assets/chains.csv",
                                "openfold_model_dir": "/assets/models",
                            }
                        ],
                        "extra_mounts": [
                            {"source": "/assets/chains.csv", "target": "/assets/chains.csv", "read_only": True},
                            {"source": "/assets/models", "target": "/assets/models", "read_only": True},
                        ],
                        "resources": {
                            "gpu_worker": {
                                "partition": "gpu",
                                "cpus_per_task": 8,
                                "memory": "64G",
                                "time": "01:00:00",
                                "gres": None,
                                "nodes": 2,
                                "tasks_per_node": 1,
                                "gpus_per_task": 1,
                                "max_parallel": 1,
                            }
                        },
                    }
                }
            },
            sort_keys=True,
        )
    )

    with pytest.raises(ValueError, match="topology must be invariant"):
        retry_phase(
            phase_run_id,
            authority_root=authority_root,
            config_path=changed_profile,
            clock=lambda: _FIXED_TIME,
        )

    assert not (authority_root / phase_run_id / "attempts" / "attempt-0002").exists()


def test_preprocessing_carry_forward_contract_unchanged(tmp_path: Path) -> None:
    from bspp.orchestration.contract.phase_carry_forward import (
        AttemptCarryForwardRecord,
        attempt_carry_forward_record_from_mapping,
    )

    event, _plan, _predecessor = _contract_fixture(tmp_path / "preprocessing")
    loaded = phase_attempt_retried_payload_from_mapping(event.payload.to_mapping())
    assert loaded == event.payload
    assert isinstance(loaded.successor_phase_runspec, PhaseRunSpec)
    assert loaded.carry_forward_record is None

    # The folding record is a sibling schema, not a subclass of the
    # preprocessing record, so the Retry loader's family dispatch stays
    # unambiguous; the preprocessing loader itself is unchanged.
    assert not issubclass(FoldingCarryForwardRecord, AttemptCarryForwardRecord)
    assert not issubclass(AttemptCarryForwardRecord, FoldingCarryForwardRecord)
    with pytest.raises(ValueError, match="missing"):
        attempt_carry_forward_record_from_mapping({})
