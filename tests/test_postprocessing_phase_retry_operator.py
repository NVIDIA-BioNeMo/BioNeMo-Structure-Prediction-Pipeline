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

"""Public restart semantics for the explicit postprocessing Retry coordinator."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import bspp.orchestration.control.postprocessing_phase_operator as operator_module
import bspp.orchestration.control.postprocessing_phase_retry_operator as retry_operator_module
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.phase_materialization import materialize_phase
from bspp.orchestration.control.postprocessing_authority_store import (
    postprocessing_phase_coordinator_lock,
)
from bspp.orchestration.control.postprocessing_authority_v2 import validate_postprocessing_authority
from bspp.orchestration.control.postprocessing_phase_cancellation import cancel_postprocessing_phase
from bspp.orchestration.control.postprocessing_phase_operator import (
    PostprocessingPhaseOperatorDependencies,
)
from bspp.orchestration.control.postprocessing_phase_retry import (
    PostprocessingRetryProjectionStore,
    PostprocessingRetryResult,
    PreparedPostprocessingRetry,
    apply_prepared_postprocessing_retry,
    prepare_postprocessing_retry,
    retry_postprocessing_phase,
)
from bspp.orchestration.control.postprocessing_phase_retry_operator import (
    PostprocessingPhaseRetryOperatorDependencies,
    retry_postprocessing_phase_to_completion,
)
from tests.test_postprocessing_phase_materialization import (
    NOW,
    RUN_ID,
    _materialized_authority,
    _retry_resolution,
)

OTHER_RUN_ID = "phase-run-fedcba9876543210fedcba9876543210"


class _AcceptedLifecycle:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def status(self, _phase_run_id: str, **_kwargs: object) -> object:
        self.calls.append("status")
        return SimpleNamespace(
            phase_run_id=RUN_ID,
            attempt_id="attempt-0002",
            status="accepted",
            details={"actions": []},
        )

    def fetch(self, _phase_run_id: str, **kwargs: object) -> None:
        self.calls.append("fetch")
        destination = kwargs["destination"]
        assert isinstance(destination, Path)
        destination.mkdir(parents=True, exist_ok=True)

    def export(self, _phase_run_id: str, **kwargs: object) -> None:
        self.calls.append("export")
        output = kwargs["output"]
        assert isinstance(output, Path)
        output.write_text("{}\n")

    def finalize(self, _phase_run_id: str, **_kwargs: object) -> object:
        self.calls.append("finalize")
        return SimpleNamespace(phase_run_id=RUN_ID, attempt_id="attempt-0002")

    def diagnostics(self, _phase_run_id: str, **kwargs: object) -> object:
        self.calls.append("diagnostics")
        root = kwargs["diagnostics_root"]
        assert isinstance(root, Path)
        root.mkdir(parents=True, exist_ok=True)
        output = root / "summary.json"
        output.write_text("{}\n")
        return SimpleNamespace(output=output)


def _dependencies(lifecycle: _AcceptedLifecycle, *, prepare: object, apply: object) -> object:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unexpected submission or Resume")

    lifecycle_dependencies = PostprocessingPhaseOperatorDependencies(
        materialize=forbidden,
        submit=getattr(lifecycle, "submit", forbidden),
        status=lifecycle.status,
        resume=getattr(lifecycle, "resume", forbidden),
        fetch=lifecycle.fetch,
        export_scheduler=lifecycle.export,
        finalize=lifecycle.finalize,
        diagnostics=lifecycle.diagnostics,
        wall_clock=lambda: NOW,
        monotonic_clock=lambda: 1.0,
        sleeper=lambda _seconds: None,
        interrupted=lambda: False,
        phase_run_id_factory=forbidden,
        bind_initial_attempt=forbidden,
        emit=lambda _event: None,
    )
    return PostprocessingPhaseRetryOperatorDependencies(
        lifecycle=lifecycle_dependencies,
        prepare_retry=prepare,
        apply_retry=apply,
    )


class _PartialSubmissionLifecycle(_AcceptedLifecycle):
    def __init__(self) -> None:
        super().__init__()
        self.assigned = False

    def status(self, _phase_run_id: str, **_kwargs: object) -> object:
        self.calls.append("status")
        return SimpleNamespace(
            phase_run_id=RUN_ID,
            attempt_id="attempt-0002",
            status="accepted" if self.assigned else "submitted",
            details={
                "actions": [
                    {
                        "action_id": "action",
                        "durable_status": "submitted" if self.assigned else "dispatching",
                        "job_id": "12" if self.assigned else None,
                        "terminal": {} if self.assigned else None,
                    }
                ]
            },
        )

    def submit(self, _phase_run_id: str, **_kwargs: object) -> object:
        self.calls.append("submit")
        if self.assigned:
            raise AssertionError("partial Retry submission was repeated")
        self.assigned = True
        return SimpleNamespace(phase_run_id=RUN_ID, attempt_id="attempt-0002")


class _FailedLifecycle(_AcceptedLifecycle):
    def status(self, _phase_run_id: str, **_kwargs: object) -> object:
        self.calls.append("status")
        return SimpleNamespace(
            phase_run_id=RUN_ID,
            attempt_id="attempt-0002",
            status="failed",
            details={"actions": []},
        )


def _prepare_retry(
    tmp_path: Path,
    *,
    authority_root: Path,
    phase_run_id: str,
) -> PreparedPostprocessingRetry:
    predecessor = validate_postprocessing_authority(authority_root, phase_run_id)
    return prepare_postprocessing_retry(
        phase_run_id,
        authority_root=authority_root,
        config_path=tmp_path / "profiles.yaml",
        source_repo=tmp_path / "source-repo",
        clock=lambda: NOW,
        runtime_resolver=_retry_resolution(tmp_path=tmp_path, authority=predecessor),
    )


def _materialize_second_phase(tmp_path: Path, *, authority_root: Path) -> None:
    materialize_phase(
        tmp_path / "postprocessing-phase-plan.yaml",
        authority_root=authority_root,
        config_path=tmp_path / "profiles.yaml",
        source_repo=tmp_path / "source-repo",
        clock=lambda: NOW,
        phase_run_id_factory=lambda: OTHER_RUN_ID,
    )
    cancel_postprocessing_phase(OTHER_RUN_ID, authority_root=authority_root, clock=lambda: NOW)


def _expected_retry_result(prepared: PreparedPostprocessingRetry) -> PostprocessingRetryResult:
    payload = prepared.payload
    successor_attempt_id = prepared.successor_attempt_id
    return PostprocessingRetryResult(
        phase_run_id=prepared.phase_run_id,
        predecessor_attempt_id=prepared.predecessor_attempt_id,
        successor_attempt_id=successor_attempt_id,
        retry_id=prepared.retry_id,
        phase_runspec_digest=payload.successor_phase_runspec.digest,
        phase_runspec_location=f"attempts/{successor_attempt_id}/phase-runspec.json",
        phase_plan_digest=payload.phase_plan_digest,
        logical_input_manifest_digest=payload.logical_input_manifest_digest,
        selected_cluster_profile=payload.successor_phase_runspec.cluster.profile_name,
    )


def test_retry_coordinator_rejects_fresh_prepared_transition_for_another_phase_before_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    _materialize_second_phase(tmp_path, authority_root=authority_root)
    prepared = _prepare_retry(tmp_path, authority_root=authority_root, phase_run_id=OTHER_RUN_ID)
    monkeypatch.setattr(retry_operator_module, "validate_runtime_qualification_source", lambda *_args, **_kwargs: None)
    execution_root = tmp_path / "retry-operation"
    lifecycle = _AcceptedLifecycle()
    apply_calls: list[str] = []

    result = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            lifecycle,
            prepare=lambda *_args, **_kwargs: prepared,
            apply=lambda *_args, **_kwargs: apply_calls.append("apply"),
        ),
    )

    assert result.status == "configuration-mismatch"
    assert not (execution_root / "operation-intent.json").exists()
    assert apply_calls == []
    assert lifecycle.calls == []
    assert validate_postprocessing_authority(authority_root, OTHER_RUN_ID).attempt_id == "attempt-0001"


def test_retry_coordinator_rejects_persisted_transition_for_another_phase_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    _materialize_second_phase(tmp_path, authority_root=authority_root)
    prepared = _prepare_retry(tmp_path, authority_root=authority_root, phase_run_id=OTHER_RUN_ID)
    monkeypatch.setattr(retry_operator_module, "validate_runtime_qualification_source", lambda *_args, **_kwargs: None)
    execution_root = tmp_path / "retry-operation"
    first_lifecycle = _AcceptedLifecycle()
    first = retry_postprocessing_phase_to_completion(
        OTHER_RUN_ID,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            first_lifecycle,
            prepare=lambda *_args, **_kwargs: prepared,
            apply=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("stop after intent")),
        ),
    )
    assert first.status == "lifecycle-failed"
    assert (execution_root / "operation-intent.json").is_file()
    assert first_lifecycle.calls == []
    first_lifecycle.calls.clear()
    apply_calls: list[str] = []

    restarted = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            first_lifecycle,
            prepare=lambda *_args, **_kwargs: pytest.fail("persisted Retry prepared again"),
            apply=lambda *_args, **_kwargs: apply_calls.append("apply"),
        ),
    )

    assert restarted.status == "configuration-mismatch"
    assert apply_calls == []
    assert first_lifecycle.calls == []
    assert validate_postprocessing_authority(authority_root, OTHER_RUN_ID).attempt_id == "attempt-0001"


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    (
        ("phase_run_id", OTHER_RUN_ID),
        ("phase_runspec_digest", "b" * 64),
        ("phase_plan_digest", "b" * 64),
        ("logical_input_manifest_digest", "b" * 64),
        ("selected_cluster_profile", "different-cluster"),
        ("phase_runspec_location", "attempts/attempt-0001/phase-runspec.json"),
        ("status", "accepted"),
    ),
)
def test_retry_coordinator_rejects_inexact_apply_result_before_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_field: str,
    changed_value: str,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    prepared = _prepare_retry(tmp_path, authority_root=authority_root, phase_run_id=RUN_ID)
    monkeypatch.setattr(retry_operator_module, "validate_runtime_qualification_source", lambda *_args, **_kwargs: None)
    invalid = replace(_expected_retry_result(prepared), **{changed_field: changed_value})
    lifecycle = _AcceptedLifecycle()

    result = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=tmp_path / "retry-operation",
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            lifecycle,
            prepare=lambda *_args, **_kwargs: prepared,
            apply=lambda *_args, **_kwargs: invalid,
        ),
    )

    assert result.status == "configuration-mismatch"
    assert lifecycle.calls == []


def test_retry_coordinator_rejects_incomplete_apply_result_before_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    prepared = _prepare_retry(tmp_path, authority_root=authority_root, phase_run_id=RUN_ID)
    monkeypatch.setattr(retry_operator_module, "validate_runtime_qualification_source", lambda *_args, **_kwargs: None)
    lifecycle = _AcceptedLifecycle()
    incomplete = SimpleNamespace(
        predecessor_attempt_id=prepared.predecessor_attempt_id,
        successor_attempt_id=prepared.successor_attempt_id,
        retry_id=prepared.retry_id,
    )

    result = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=tmp_path / "retry-operation",
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            lifecycle,
            prepare=lambda *_args, **_kwargs: prepared,
            apply=lambda *_args, **_kwargs: incomplete,
        ),
    )

    assert result.status == "configuration-mismatch"
    assert lifecycle.calls == []


def test_retry_coordinator_persists_exact_transition_then_drives_only_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    authority_path = authority_root / RUN_ID
    before = {
        path.relative_to(authority_path): path.read_bytes() for path in authority_path.rglob("*") if path.is_file()
    }
    resolver = _retry_resolution(tmp_path=tmp_path, authority=predecessor)
    monkeypatch.setattr(
        "bspp.orchestration.control.postprocessing_phase_retry_operator.validate_runtime_qualification_source",
        lambda *_args, **_kwargs: None,
    )
    lifecycle = _AcceptedLifecycle()

    result = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=tmp_path / "retry-operation",
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            lifecycle,
            prepare=lambda *args, **kwargs: prepare_postprocessing_retry(*args, **kwargs, runtime_resolver=resolver),
            apply=apply_prepared_postprocessing_retry,
        ),
    )

    assert result.status == "accepted"
    assert result.predecessor_attempt_id == "attempt-0001"
    assert result.successor_attempt_id == "attempt-0002"
    assert result.recovery_command is None
    assert lifecycle.calls == ["status", "fetch", "export", "finalize"]
    intent = json.loads((tmp_path / "retry-operation/operation-intent.json").read_bytes())
    assert intent["operation_kind"] == "postprocessing-phase-retry-operator-v1"
    assert intent["transition"]["successor_attempt_id"] == "attempt-0002"
    assert intent["qualification"]["document"]
    assert validate_postprocessing_authority(authority_root, RUN_ID).attempt_id == "attempt-0002"
    for relative, payload in before.items():
        assert (authority_path / relative).read_bytes() == payload


def test_retry_coordinator_restart_after_intent_does_not_create_attempt_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    resolver = _retry_resolution(tmp_path=tmp_path, authority=predecessor)
    monkeypatch.setattr(
        "bspp.orchestration.control.postprocessing_phase_retry_operator.validate_runtime_qualification_source",
        lambda *_args, **_kwargs: None,
    )
    execution_root = tmp_path / "retry-operation"

    def crash_before_retry(*_args: object, **_kwargs: object) -> object:
        raise OSError("simulated interruption before Retry")

    first_lifecycle = _AcceptedLifecycle()
    first = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            first_lifecycle,
            prepare=lambda *args, **kwargs: prepare_postprocessing_retry(*args, **kwargs, runtime_resolver=resolver),
            apply=crash_before_retry,
        ),
    )
    assert first.status == "lifecycle-failed"
    assert (execution_root / "operation-intent.json").is_file()
    assert first_lifecycle.calls == []
    assert validate_postprocessing_authority(authority_root, RUN_ID).attempt_id == "attempt-0001"

    def forbidden_prepare(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("restart reselected current qualification")

    restarted = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            _AcceptedLifecycle(),
            prepare=forbidden_prepare,
            apply=apply_prepared_postprocessing_retry,
        ),
    )

    assert restarted.status == "accepted"
    assert restarted.successor_attempt_id == "attempt-0002"
    assert validate_postprocessing_authority(authority_root, RUN_ID).attempt_id == "attempt-0002"


def test_retry_coordinator_restart_recovers_exact_event_before_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    resolver = _retry_resolution(tmp_path=tmp_path, authority=predecessor)
    monkeypatch.setattr(
        "bspp.orchestration.control.postprocessing_phase_retry_operator.validate_runtime_qualification_source",
        lambda *_args, **_kwargs: None,
    )
    execution_root = tmp_path / "retry-operation"

    def interrupt_projection(prepared: object, **kwargs: object) -> object:
        def fail_projection(_path: Path, _payload: bytes) -> None:
            raise OSError("simulated projection interruption")

        return apply_prepared_postprocessing_retry(
            prepared,
            authority_root=kwargs["authority_root"],
            projection_store=PostprocessingRetryProjectionStore(publish_exact=fail_projection),
        )

    first_lifecycle = _AcceptedLifecycle()
    first = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            first_lifecycle,
            prepare=lambda *args, **kwargs: prepare_postprocessing_retry(*args, **kwargs, runtime_resolver=resolver),
            apply=interrupt_projection,
        ),
    )
    incomplete = validate_postprocessing_authority(authority_root, RUN_ID)
    assert first.status == "lifecycle-failed"
    assert first_lifecycle.calls == []
    assert incomplete.attempt_id == "attempt-0002"
    assert incomplete.current_attempt_projection_complete is False

    restarted_lifecycle = _AcceptedLifecycle()
    restarted = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            restarted_lifecycle,
            prepare=lambda *_args, **_kwargs: pytest.fail("qualification was reselected"),
            apply=apply_prepared_postprocessing_retry,
        ),
    )
    assert restarted.status == "accepted"
    assert validate_postprocessing_authority(authority_root, RUN_ID).current_attempt_projection_complete is True


def test_retry_coordinator_changed_config_fails_before_retry_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    resolver = _retry_resolution(tmp_path=tmp_path, authority=predecessor)
    monkeypatch.setattr(
        "bspp.orchestration.control.postprocessing_phase_retry_operator.validate_runtime_qualification_source",
        lambda *_args, **_kwargs: None,
    )
    execution_root = tmp_path / "retry-operation"
    first = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            _AcceptedLifecycle(),
            prepare=lambda *args, **kwargs: prepare_postprocessing_retry(*args, **kwargs, runtime_resolver=resolver),
            apply=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("stop before Retry")),
        ),
    )
    assert first.status == "lifecycle-failed"
    with (tmp_path / "profiles.yaml").open("a") as handle:
        handle.write("\n# changed\n")

    restarted = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            _AcceptedLifecycle(),
            prepare=lambda *_args, **_kwargs: pytest.fail("prepared again"),
            apply=lambda *_args, **_kwargs: pytest.fail("Retry applied after config mismatch"),
        ),
    )
    assert restarted.status == "configuration-mismatch"
    assert validate_postprocessing_authority(authority_root, RUN_ID).attempt_id == "attempt-0001"


def test_retry_coordinator_rejects_different_manual_successor_without_attempt_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    resolver = _retry_resolution(tmp_path=tmp_path, authority=predecessor)
    monkeypatch.setattr(
        "bspp.orchestration.control.postprocessing_phase_retry_operator.validate_runtime_qualification_source",
        lambda *_args, **_kwargs: None,
    )
    execution_root = tmp_path / "retry-operation"
    first_lifecycle = _AcceptedLifecycle()
    retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            first_lifecycle,
            prepare=lambda *args, **kwargs: prepare_postprocessing_retry(*args, **kwargs, runtime_resolver=resolver),
            apply=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("stop before Retry")),
        ),
    )
    assert first_lifecycle.calls == []
    retry_postprocessing_phase(
        RUN_ID,
        authority_root=authority_root,
        config_path=tmp_path / "profiles.yaml",
        source_repo=tmp_path / "source-repo",
        clock=lambda: NOW + timedelta(seconds=1),
        runtime_resolver=resolver,
    )

    restarted_lifecycle = _AcceptedLifecycle()
    restarted = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            restarted_lifecycle,
            prepare=lambda *_args, **_kwargs: pytest.fail("prepared again"),
            apply=apply_prepared_postprocessing_retry,
        ),
    )
    assert restarted.status == "lifecycle-failed"
    assert restarted_lifecycle.calls == []
    assert validate_postprocessing_authority(authority_root, RUN_ID).attempt_id == "attempt-0002"
    assert not (authority_root / RUN_ID / "attempts/attempt-0003").exists()


def test_retry_coordinator_changed_source_checkout_is_configuration_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    resolver = _retry_resolution(tmp_path=tmp_path, authority=predecessor)
    execution_root = tmp_path / "retry-operation"
    monkeypatch.setattr(retry_operator_module, "validate_runtime_qualification_source", lambda *_args, **_kwargs: None)
    first = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            _AcceptedLifecycle(),
            prepare=lambda *args, **kwargs: prepare_postprocessing_retry(*args, **kwargs, runtime_resolver=resolver),
            apply=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("stop before Retry")),
        ),
    )
    assert first.status == "lifecycle-failed"

    def changed_source(*_args: object, **_kwargs: object) -> None:
        raise ValueError("source checkout differs from persisted qualification")

    monkeypatch.setattr(retry_operator_module, "validate_runtime_qualification_source", changed_source)
    restarted = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            _AcceptedLifecycle(),
            prepare=lambda *_args, **_kwargs: pytest.fail("prepared again"),
            apply=lambda *_args, **_kwargs: pytest.fail("Retry applied after source mismatch"),
        ),
    )

    assert restarted.status == "configuration-mismatch"
    assert validate_postprocessing_authority(authority_root, RUN_ID).attempt_id == "attempt-0001"


def test_retry_coordinator_recovers_partial_successor_submission_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    resolver = _retry_resolution(tmp_path=tmp_path, authority=predecessor)
    monkeypatch.setattr(retry_operator_module, "validate_runtime_qualification_source", lambda *_args, **_kwargs: None)
    lifecycle = _PartialSubmissionLifecycle()

    result = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=tmp_path / "retry-operation",
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            lifecycle,
            prepare=lambda *args, **kwargs: prepare_postprocessing_retry(*args, **kwargs, runtime_resolver=resolver),
            apply=apply_prepared_postprocessing_retry,
        ),
    )

    assert result.status == "accepted"
    assert lifecycle.calls == ["status", "submit", "status", "fetch", "export", "finalize"]


def test_retry_coordinator_failed_successor_captures_diagnostics_without_attempt_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    resolver = _retry_resolution(tmp_path=tmp_path, authority=predecessor)
    monkeypatch.setattr(retry_operator_module, "validate_runtime_qualification_source", lambda *_args, **_kwargs: None)
    lifecycle = _FailedLifecycle()

    result = retry_postprocessing_phase_to_completion(
        RUN_ID,
        authority_root=authority_root,
        execution_root=tmp_path / "retry-operation",
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=_dependencies(
            lifecycle,
            prepare=lambda *args, **kwargs: prepare_postprocessing_retry(*args, **kwargs, runtime_resolver=resolver),
            apply=apply_prepared_postprocessing_retry,
        ),
    )

    assert result.status == "lifecycle-failed"
    assert result.diagnostics == str(tmp_path / "retry-operation/diagnostics/summary.json")
    assert lifecycle.calls == ["status", "diagnostics"]
    assert not (authority_root / RUN_ID / "attempts/attempt-0003").exists()


def test_retry_and_operator_v1_contend_on_the_same_phase_lock(
    tmp_path: Path,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    prepared = False

    def forbidden_prepare(*_args: object, **_kwargs: object) -> object:
        nonlocal prepared
        prepared = True
        raise AssertionError("Retry prepared while operator-v1 owned the Phase lock")

    with postprocessing_phase_coordinator_lock(authority_root, RUN_ID):
        result = retry_postprocessing_phase_to_completion(
            RUN_ID,
            authority_root=authority_root,
            execution_root=tmp_path / "retry-operation",
            source_repo=tmp_path / "source-repo",
            config_path=tmp_path / "profiles.yaml",
            dependencies=_dependencies(
                _AcceptedLifecycle(),
                prepare=forbidden_prepare,
                apply=lambda *_args, **_kwargs: pytest.fail("Retry applied during contention"),
            ),
        )

    assert result.status == "lock-conflict"
    assert prepared is False
    assert not (tmp_path / "retry-operation/operation-intent.json").exists()


def test_real_operator_v1_binding_rejects_durable_retry_successor(
    tmp_path: Path,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    execution_root = tmp_path / "operator-v1"
    status_calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unexpected operator lifecycle mutation")

    def status(_phase_run_id: str, **kwargs: object) -> object:
        status_calls.append("status")
        authority = validate_postprocessing_authority(kwargs["authority_root"], RUN_ID)
        return SimpleNamespace(
            phase_run_id=RUN_ID,
            attempt_id=authority.attempt_id,
            status="cancelled",
            details={"actions": []},
        )

    lifecycle = PostprocessingPhaseOperatorDependencies(
        materialize=forbidden,
        submit=forbidden,
        status=status,
        resume=forbidden,
        fetch=forbidden,
        export_scheduler=forbidden,
        finalize=forbidden,
        diagnostics=lambda *_args, **_kwargs: SimpleNamespace(output=tmp_path / "diagnostics.json"),
        wall_clock=lambda: NOW,
        monotonic_clock=lambda: 1.0,
        sleeper=lambda _seconds: None,
        interrupted=lambda: False,
        phase_run_id_factory=lambda: RUN_ID,
        bind_initial_attempt=operator_module.DEFAULT_POSTPROCESSING_PHASE_OPERATOR_DEPENDENCIES.bind_initial_attempt,
        emit=lambda _event: None,
    )
    first = operator_module.run_postprocessing_phase(
        tmp_path / "postprocessing-phase-plan.yaml",
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=lifecycle,
    )
    assert first.status == "cancelled"
    assert status_calls == ["status"]

    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    retry_postprocessing_phase(
        RUN_ID,
        authority_root=authority_root,
        config_path=tmp_path / "profiles.yaml",
        source_repo=tmp_path / "source-repo",
        clock=lambda: NOW + timedelta(seconds=1),
        runtime_resolver=_retry_resolution(tmp_path=tmp_path, authority=predecessor),
    )
    restarted = operator_module.run_postprocessing_phase(
        tmp_path / "postprocessing-phase-plan.yaml",
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=tmp_path / "source-repo",
        config_path=tmp_path / "profiles.yaml",
        dependencies=lifecycle,
    )

    assert restarted.status == "configuration-mismatch"
    assert restarted.attempt_id == "attempt-0001"
    assert status_calls == ["status"]
    assert "attempt-0002" in (restarted.detail or "")


def test_retry_postprocessing_cli_is_a_thin_operator_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    source_repo = tmp_path / "source"
    source_repo.mkdir()
    config_path = tmp_path / "profiles.yaml"
    config_path.write_text("clusters: {}\n")
    captured: dict[str, object] = {}

    def retry_stub(phase_run_id: str, **kwargs: object) -> object:
        captured["phase_run_id"] = phase_run_id
        captured.update(kwargs)
        return SimpleNamespace(status="accepted", render_json=lambda: "{}\n")

    monkeypatch.setattr(retry_operator_module, "retry_postprocessing_phase_to_completion", retry_stub)
    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config_path),
            "phase",
            "retry-postprocessing",
            RUN_ID,
            "--authority-root",
            str(authority_root),
            "--execution-root",
            str(tmp_path / "retry-operation"),
            "--source-repo",
            str(source_repo),
            "--poll-interval",
            "2",
            "--timeout",
            "60",
        ],
    )

    assert result.exit_code == 0
    assert captured["phase_run_id"] == RUN_ID
    assert captured["authority_root"] == authority_root
    assert captured["execution_root"] == tmp_path / "retry-operation"
    assert captured["source_repo"] == source_repo
    assert captured["config_path"] == config_path
    assert captured["poll_interval_seconds"] == 2.0
    assert captured["timeout_seconds"] == 60.0
