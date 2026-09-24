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

"""Cross-story folding packing integration tests (e13 coherence round 0).

These tests walk the full producer -> transport -> mount -> Runtime consumer
path for the canonical fold shard projection: materialization stages the
authority-relative projection beside every staged folding RunSpec, packed fold
and canonical-pair actions mount that exact file read-only, and Runtime loads
the same staged sidecar with digest/size/worker-count binding. Scalar authority
stages the projection immutably but never mounts or reads it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_shard import FOLD_SHARD_PROJECTION_FILENAME
from bspp.orchestration.contract.phase import FoldingPhaseRunSpec
from bspp.orchestration.contract.phase_reconciliation import FoldingActionTerminalObservationView
from bspp.orchestration.control import transport as transport_mod
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore
from bspp.orchestration.control.phase_cancellation import cancel_phase
from bspp.orchestration.control.phase_resume import resume_phase
from bspp.orchestration.control.phase_status import status_phase
from bspp.orchestration.control.phase_submission import submit_phase
from bspp.orchestration.runtime.folding.executor import (
    FoldingExecutorError,
    _load_packed_shard_selection,
)
from tests.test_phase_folding_cancel_retry import (
    EmptyObservationRunner,
    FoldingArrayObservationRunner,
    FoldingCancellationRunner,
    _materialize_packed_folding,
    _tree_bytes,
)
from tests.test_phase_folding_lifecycle import (
    FIXED_TIME,
    FoldingSubmissionRunner,
    _materialize_folding,
)


def _record_staging(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, str, str, str]]:
    """Record every ``stage_immutable_artifact`` call while still executing it."""
    calls: list[tuple[Path, str, str, str]] = []
    original = transport_mod.RemoteSlurmTransport.stage_immutable_artifact

    def recording(self, local_path, target_path, *, expected_sha256, staging_token):
        calls.append((Path(local_path), target_path, expected_sha256, staging_token))
        return original(
            self,
            local_path,
            target_path,
            expected_sha256=expected_sha256,
            staging_token=staging_token,
        )

    monkeypatch.setattr(transport_mod.RemoteSlurmTransport, "stage_immutable_artifact", recording)
    return calls


def _cluster_paths(runspec: FoldingPhaseRunSpec) -> tuple[Path, Path]:
    staging_root = Path(runspec.cluster.staging_root)
    cluster_runspec_path = (
        staging_root / "bspp-phase-runs" / runspec.phase_run_id / runspec.attempt_id / "phase-runspec.json"
    )
    cluster_projection_path = cluster_runspec_path.parent / FOLD_SHARD_PROJECTION_FILENAME
    return cluster_runspec_path, cluster_projection_path


def test_packed_projection_producer_transport_mount_runtime_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root, phase_run_id = _materialize_packed_folding(tmp_path)
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert isinstance(authority.phase_runspec, FoldingPhaseRunSpec)
    runspec = authority.phase_runspec
    binding = runspec.payload.fold_shard_projection
    assert binding is not None and binding.worker_count > 1

    cluster_runspec_path, cluster_projection_path = _cluster_paths(runspec)
    authority_projection = authority.authority_path / binding.location

    stage_calls = _record_staging(monkeypatch)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )

    # Transport: the projection is staged beside the RunSpec under its canonical
    # basename, byte-identical to the authority document, and bound by SHA-256/size.
    projection_calls = [call for call in stage_calls if call[1] == str(cluster_projection_path)]
    assert projection_calls
    assert cluster_projection_path.read_bytes() == authority_projection.read_bytes()
    assert hashlib.sha256(cluster_projection_path.read_bytes()).hexdigest() == binding.sha256
    assert len(cluster_projection_path.read_bytes()) == binding.size_bytes
    for local_path, target, expected_sha256, staging_token in projection_calls:
        assert local_path == authority_projection
        assert target == str(cluster_projection_path)
        assert expected_sha256 == binding.sha256
        # The projection uses the same action staging token as the RunSpec sidecar.
        assert any(call[1] == str(cluster_runspec_path) and call[3] == staging_token for call in stage_calls)

    # Mounts: packed fold and canonical-pair mount the exact file read-only;
    # unrelated actions do not expose the projection at all.
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.submission is not None
    by_id = {action.action_id: action for action in replay.submission.actions}
    mount_fragment = f"{cluster_projection_path}:{cluster_projection_path}:ro"
    assert mount_fragment in by_id["fold-000001"].plan.script_body
    assert mount_fragment in by_id["canonical-pair-000001"].plan.script_body
    for action_id in ("msa-flatten-000001", "split-000001", "preprocess-000001"):
        assert str(cluster_projection_path) not in by_id[action_id].plan.script_body

    # Runtime consumer: rank selection succeeds against the actually staged
    # RunSpec + transported sidecar and reconciles the binding worker count.
    fold_action = next(action for action in runspec.payload.actions if action.action_kind == "fold")
    selection = _load_packed_shard_selection(cluster_runspec_path, runspec, fold_action, rank=0)
    assert selection.worker_count == fold_action.resources.workers == binding.worker_count

    # Corrupting or removing the sidecar makes packed execution fail closed.
    cluster_projection_path.write_bytes(b"corrupted")
    with pytest.raises(FoldingExecutorError, match="SHA-256 mismatch"):
        _load_packed_shard_selection(cluster_runspec_path, runspec, fold_action, rank=0)
    cluster_projection_path.unlink()
    with pytest.raises(FoldingExecutorError, match="cannot read fold shard projection"):
        _load_packed_shard_selection(cluster_runspec_path, runspec, fold_action, rank=0)


def test_scalar_projection_is_staged_but_never_mounted_or_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert isinstance(authority.phase_runspec, FoldingPhaseRunSpec)
    runspec = authority.phase_runspec
    binding = runspec.payload.fold_shard_projection
    assert binding is not None and binding.worker_count == 1

    cluster_runspec_path, cluster_projection_path = _cluster_paths(runspec)

    stage_calls = _record_staging(monkeypatch)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )

    # The projection may be immutably staged for scalar authority...
    assert cluster_projection_path.is_file()
    assert any(call[1] == str(cluster_projection_path) for call in stage_calls)

    # ...but no script gains a projection mount.
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert replay.submission is not None
    for action in replay.submission.actions:
        assert str(cluster_projection_path) not in action.plan.script_body

    # Scalar Runtime dispatch never reads the projection: the packed-only loader
    # rejects the scalar fold topology before touching the staged sidecar.
    fold_action = next(action for action in runspec.payload.actions if action.action_kind == "fold")
    assert not fold_action.resources.is_packed
    with pytest.raises(FoldingExecutorError, match="packed fold execution requires a packed fold action topology"):
        _load_packed_shard_selection(cluster_runspec_path, runspec, fold_action, rank=0)


def test_packed_array_resume_then_cancel_status_prefers_exact_array_evidence(tmp_path: Path) -> None:
    """Status stays total over the reachable resume(array) -> cancel(scalar) state.

    Cancellation writes one scalar parent-row terminal fact per bound action,
    while Resume already recorded the exact parent-plus-task array evidence for
    the packed fold. The public read path must project that overlap onto the
    array view without mutating authority, rejecting replay, or
    raising on an ordinary reachable composition.
    """
    authority_root, phase_run_id = _materialize_packed_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )

    resumed = resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: FIXED_TIME,
        runner=FoldingArrayObservationRunner(),
    )
    assert resumed.outcome == "reconciled"
    assert "fold-000001" in resumed.terminal_action_ids

    cancelled = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: FIXED_TIME,
        runner=FoldingCancellationRunner(),
    )
    assert cancelled.status == "cancelled"

    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert sum(item.action_id == "fold-000001" for item in replay.terminal_observations) == 1
    assert [item.action_id for item in replay.array_terminal_observations] == ["fold-000001"]

    before = _tree_bytes(authority_root / phase_run_id)
    report = status_phase(
        phase_run_id,
        authority_root=authority_root,
        runner=EmptyObservationRunner(),
    )
    assert _tree_bytes(authority_root / phase_run_id) == before

    assert report.status == "cancelled"
    fold_status = next(item for item in report.actions if item.action_id == "fold-000001")
    assert isinstance(fold_status.terminal, FoldingActionTerminalObservationView)
    assert fold_status.terminal.parent_job_id == "1004"
    assert fold_status.terminal.expected_task_indexes == (0, 1, 2)
    assert [task.task_index for task in fold_status.terminal.tasks] == [0, 1, 2]
    assert fold_status.terminal.outcome == "succeeded"
    mapping = fold_status.to_mapping()
    assert mapping["terminal"]["parent_job_id"] == "1004"
    assert mapping["terminal"]["expected_task_indexes"] == [0, 1, 2]
    assert "job_id" not in mapping["terminal"]

    second = status_phase(
        phase_run_id,
        authority_root=authority_root,
        runner=EmptyObservationRunner(),
    )
    assert second == report
    assert _tree_bytes(authority_root / phase_run_id) == before
