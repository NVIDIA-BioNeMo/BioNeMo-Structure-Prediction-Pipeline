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

"""Restart and path-ownership tests for the postprocessing Phase operator."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import bspp.orchestration.control.postprocessing_phase_operator as operator_module
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.postprocessing_authority_store import (
    postprocessing_phase_coordinator_lock,
)
from bspp.orchestration.control.postprocessing_phase_operator import (
    PostprocessingPhaseOperatorDependencies,
    run_postprocessing_phase,
)

PHASE_RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
PHASE_PLAN_DIGEST = "a" * 64


@dataclass
class _FakeLifecycle:
    calls: list[tuple[str, dict[str, object]]]
    crash_after: str | None = None
    crashed: bool = False

    def _state_path(self, authority_root: Path) -> Path:
        return authority_root / PHASE_RUN_ID / "fake-state.json"

    def _state(self, authority_root: Path) -> dict[str, object]:
        return json.loads(self._state_path(authority_root).read_bytes())

    def _save(self, authority_root: Path, state: dict[str, object]) -> None:
        self._state_path(authority_root).write_text(json.dumps(state))

    def _crash(self, boundary: str) -> None:
        if self.crash_after == boundary and not self.crashed:
            self.crashed = True
            raise OSError(f"simulated crash after {boundary}")

    def materialize(self, _plan: Path, **kwargs: object) -> object:
        self.calls.append(("materialize", kwargs))
        self._crash("intent")
        phase_run_id = kwargs["phase_run_id_factory"]()
        authority_root = kwargs["authority_root"]
        assert isinstance(authority_root, Path)
        (authority_root / phase_run_id).mkdir(parents=True)
        (authority_root / phase_run_id / "phase-run.json").write_text(
            json.dumps(
                {
                    "phase_run_id": phase_run_id,
                    "phase_plan_digest": PHASE_PLAN_DIGEST,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        self._save(authority_root, {"status": "materialized", "terminal": False})
        self._crash("materialize")
        return SimpleNamespace(phase_run_id=phase_run_id, attempt_id="attempt-0001")

    def submit(self, _phase_run_id: str, **kwargs: object) -> object:
        self.calls.append(("submit", kwargs))
        authority_root = kwargs["authority_root"]
        assert isinstance(authority_root, Path)
        state = self._state(authority_root)
        if state["status"] != "materialized":
            raise AssertionError("submit repeated outside durable materialized state")
        state["status"] = "submitted"
        self._save(authority_root, state)
        self._crash("submit")
        return self._status(authority_root)

    def status(self, _phase_run_id: str, **kwargs: object) -> object:
        self.calls.append(("status", kwargs))
        authority_root = kwargs["authority_root"]
        assert isinstance(authority_root, Path)
        return self._status(authority_root)

    def resume(self, _phase_run_id: str, **kwargs: object) -> object:
        self.calls.append(("resume", kwargs))
        authority_root = kwargs["authority_root"]
        assert isinstance(authority_root, Path)
        state = self._state(authority_root)
        if state["status"] != "submitted":
            raise AssertionError("resume outside durable submitted state")
        state["terminal"] = True
        self._save(authority_root, state)
        self._crash("resume")
        return self._status(authority_root)

    def fetch(self, _phase_run_id: str, **kwargs: object) -> object:
        self.calls.append(("fetch", kwargs))
        authority_root = kwargs["authority_root"]
        assert isinstance(authority_root, Path)
        state = self._state(authority_root)
        if state["status"] not in {"submitted", "accepted"} or not state["terminal"]:
            raise AssertionError("fetch before durable terminal completion")
        destination = kwargs["destination"]
        assert isinstance(destination, Path)
        aggregate = destination / "aggregate-action-evidence.json"
        adjudication = destination / "acceptance/adjudication.json"
        if destination.exists():
            if aggregate.read_text() != "{}\n" or adjudication.read_text() != "{}\n":
                raise AssertionError("fetch did not validate its existing exact destination")
        else:
            (destination / "acceptance").mkdir(parents=True)
            aggregate.write_text("{}\n")
            adjudication.write_text("{}\n")
        self._crash("fetch")
        return SimpleNamespace(destination=destination)

    def export(self, _phase_run_id: str, **kwargs: object) -> object:
        self.calls.append(("export", kwargs))
        output = kwargs["output"]
        assert isinstance(output, Path)
        if not (output.parent / "handoff").is_dir():
            raise AssertionError("scheduler export preceded exact handoff validation")
        if output.exists():
            if output.read_text() != "{}\n":
                raise AssertionError("export did not validate its create-once output")
        else:
            output.write_text("{}\n")
        self._crash("export")
        return SimpleNamespace(output=output)

    def finalize(self, _phase_run_id: str, **kwargs: object) -> object:
        self.calls.append(("finalize", kwargs))
        authority_root = kwargs["authority_root"]
        assert isinstance(authority_root, Path)
        state = self._state(authority_root)
        if state["status"] == "accepted":
            if not Path(kwargs["scheduler_evidence_path"]).is_file():
                raise AssertionError("accepted restart omitted scheduler evidence validation")
        elif state["status"] == "submitted" and state["terminal"]:
            state["status"] = "accepted"
            self._save(authority_root, state)
        else:
            raise AssertionError("finalize outside durable terminal completion")
        self._crash("finalize")
        return SimpleNamespace(
            phase_run_id=PHASE_RUN_ID,
            attempt_id=str(state.get("attempt_id", "attempt-0001")),
        )

    def diagnostics(self, _phase_run_id: str, **kwargs: object) -> object:
        self.calls.append(("diagnostics", kwargs))
        root = kwargs["diagnostics_root"]
        assert isinstance(root, Path)
        root.mkdir(exist_ok=True)
        output = root / "summary.json"
        output.write_text("{}\n")
        return SimpleNamespace(output=output)

    def _status(self, authority_root: Path) -> object:
        state = self._state(authority_root)
        return SimpleNamespace(
            phase_run_id=PHASE_RUN_ID,
            attempt_id=str(state.get("attempt_id", "attempt-0001")),
            status=state["status"],
            details={
                "actions": [
                    {
                        "action_id": "action",
                        "job_id": "11" if state["status"] != "materialized" else None,
                        "terminal": {} if state["terminal"] else None,
                    }
                ]
            },
        )


@dataclass
class _PartialSubmissionLifecycle(_FakeLifecycle):
    partial_stage: str = "submission-intent"
    submit_interrupted: bool = False

    def submit(self, _phase_run_id: str, **kwargs: object) -> object:
        self.calls.append(("submit", kwargs))
        authority_root = kwargs["authority_root"]
        assert isinstance(authority_root, Path)
        state = self._state(authority_root)
        if state["status"] == "materialized":
            rows: list[dict[str, object]]
            if self.partial_stage == "submission-intent":
                rows = [
                    {"durable_status": "planned", "job_id": None},
                    {"durable_status": "planned", "job_id": None},
                ]
            elif self.partial_stage == "dispatch-intent":
                rows = [
                    {"durable_status": "dispatching", "job_id": None},
                    {"durable_status": "planned", "job_id": None},
                ]
            elif self.partial_stage == "partial-assignments":
                rows = [
                    {"durable_status": "submitted", "job_id": "11"},
                    {"durable_status": "planned", "job_id": None},
                ]
            else:
                raise AssertionError(f"unexpected partial submission stage {self.partial_stage}")
            state.update(status="submitted", actions=rows)
            self._save(authority_root, state)
            if not self.submit_interrupted:
                self.submit_interrupted = True
                raise OSError(f"simulated crash after {self.partial_stage}")
        if state["status"] != "submitted" or state.get("terminal"):
            raise AssertionError("submit repeated after complete durable initial assignment")
        rows = state.get("actions")
        if not isinstance(rows, list) or not any(item.get("job_id") is None for item in rows):
            raise AssertionError("submit repeated after complete durable initial assignment")
        state["actions"] = [
            {"durable_status": "submitted", "job_id": "11"},
            {"durable_status": "submitted", "job_id": "12"},
        ]
        self._save(authority_root, state)
        return self._status(authority_root)

    def _status(self, authority_root: Path) -> object:
        state = self._state(authority_root)
        rows = state.get("actions")
        if not isinstance(rows, list):
            rows = [
                {"durable_status": "not-submitted", "job_id": None},
                {"durable_status": "not-submitted", "job_id": None},
            ]
        actions = [
            {
                "action_id": f"action-{index}",
                "durable_status": row["durable_status"],
                "job_id": row["job_id"],
                "terminal": {} if state["terminal"] else None,
            }
            for index, row in enumerate(rows, start=1)
        ]
        return SimpleNamespace(
            phase_run_id=PHASE_RUN_ID,
            attempt_id=str(state.get("attempt_id", "attempt-0001")),
            status=state["status"],
            details={"actions": actions},
        )


@dataclass
class _DriftingLifecycle(_FakeLifecycle):
    drift_boundary: str = "status"
    drift_field: str = "phase_run_id"

    def _drift(self, result: object) -> object:
        values = vars(result).copy()
        values[self.drift_field] = (
            "phase-run-fedcba9876543210fedcba9876543210" if self.drift_field == "phase_run_id" else "attempt-9999"
        )
        return SimpleNamespace(**values)

    def status(self, phase_run_id: str, **kwargs: object) -> object:
        result = super().status(phase_run_id, **kwargs)
        return self._drift(result) if self.drift_boundary == "status" else result

    def submit(self, phase_run_id: str, **kwargs: object) -> object:
        result = super().submit(phase_run_id, **kwargs)
        return self._drift(result) if self.drift_boundary == "submit" else result

    def resume(self, phase_run_id: str, **kwargs: object) -> object:
        result = super().resume(phase_run_id, **kwargs)
        return self._drift(result) if self.drift_boundary == "resume" else result

    def finalize(self, phase_run_id: str, **kwargs: object) -> object:
        result = super().finalize(phase_run_id, **kwargs)
        return self._drift(result) if self.drift_boundary == "finalize" else result


def _dependencies(fake: _FakeLifecycle, *, emit: list[dict[str, object]]) -> PostprocessingPhaseOperatorDependencies:
    return PostprocessingPhaseOperatorDependencies(
        materialize=fake.materialize,
        submit=fake.submit,
        status=fake.status,
        resume=fake.resume,
        fetch=fake.fetch,
        export_scheduler=fake.export,
        finalize=fake.finalize,
        diagnostics=fake.diagnostics,
        wall_clock=lambda: datetime(2026, 9, 3, tzinfo=UTC),
        monotonic_clock=lambda: 1.0,
        sleeper=lambda _seconds: None,
        interrupted=lambda: False,
        phase_run_id_factory=lambda: PHASE_RUN_ID,
        bind_initial_attempt=lambda _root, **_kwargs: "attempt-0001",
        emit=lambda event: emit.append(dict(event)),
    )


@pytest.fixture
def operator_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path, Path]:
    plan = tmp_path / "phase-plan.yaml"
    plan.write_text("phase_kind: postprocessing\n")
    config = tmp_path / "profiles.yaml"
    config.write_text("clusters: {}\n")
    source = tmp_path / "source"
    source.mkdir()
    authority = tmp_path / "authority"
    execution = tmp_path / "execution"

    def intent_identity(**kwargs: object) -> dict[str, object]:
        plan_document = kwargs["plan_document"]
        config_document = kwargs["config_document"]
        assert isinstance(plan_document, bytes)
        assert isinstance(config_document, bytes)
        return {
            "format_version": 1,
            "operation_kind": "postprocessing-phase-operator-v1",
            "phase_plan": {
                "path": str(kwargs["plan_path"]),
                "sha256": hashlib.sha256(plan_document).hexdigest(),
                "size_bytes": len(plan_document),
                "phase_plan_digest": PHASE_PLAN_DIGEST,
            },
            "authority_root": str(kwargs["authority_root"]),
            "execution_root": str(kwargs["execution_root"]),
            "source": {"repository": str(kwargs["source_repo"])},
            "profile": {
                "config_path": str(kwargs["config_path"]),
                "config_sha256": hashlib.sha256(config_document).hexdigest(),
                "config_size_bytes": len(config_document),
            },
            "options": {
                "poll_interval_seconds": kwargs["poll_interval_seconds"],
                "timeout_seconds": kwargs["timeout_seconds"],
            },
        }

    monkeypatch.setattr(operator_module, "_intent_identity", intent_identity)
    return plan, authority, execution, source, config


def test_operator_happy_path_owns_exact_derived_finalization_paths(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
) -> None:
    plan, authority, execution, source, config = operator_inputs
    calls: list[tuple[str, dict[str, object]]] = []
    transitions: list[dict[str, object]] = []
    fake = _FakeLifecycle(calls)

    result = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        poll_interval_seconds=2,
        timeout_seconds=60,
        dependencies=_dependencies(fake, emit=transitions),
    )

    assert result.status == "accepted"
    assert [name for name, _kwargs in calls] == [
        "materialize",
        "status",
        "submit",
        "status",
        "resume",
        "status",
        "fetch",
        "export",
        "finalize",
    ]
    assert {path.name for path in execution.iterdir()} == {
        "operation-intent.json",
        "current.json",
        "operation.lock",
        "handoff",
        "scheduler-evidence.json",
    }
    finalize = calls[-1][1]
    assert finalize["scheduler_evidence_path"] == execution / "scheduler-evidence.json"
    assert finalize["aggregate_action_evidence_path"] == execution / "handoff/aggregate-action-evidence.json"
    assert finalize["handoff_path"] == execution / "handoff"
    assert finalize["acceptance_adjudication_path"] == execution / "handoff/acceptance/adjudication.json"
    assert json.loads((execution / "current.json").read_bytes())["state"] == "accepted"
    assert [item["state"] for item in transitions] == [
        "materializing",
        "submitting",
        "polling",
        "fetching",
        "exporting-scheduler-evidence",
        "finalizing",
        "accepted",
    ]
    materialize = calls[0][1]
    assert materialize["phase_plan_document"] == plan.read_bytes()
    assert materialize["config_document"] == config.read_bytes()


@pytest.mark.parametrize(
    ("mutation_boundary", "intent_published"),
    (("before-intent-publication", False), ("before-materialization", True)),
)
def test_operator_input_mutation_between_intent_seams_fails_closed(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    mutation_boundary: str,
    intent_published: bool,
) -> None:
    plan, authority, execution, source, config = operator_inputs
    original_plan = plan.read_bytes()
    original_config = config.read_bytes()
    calls: list[tuple[str, dict[str, object]]] = []

    if mutation_boundary == "before-intent-publication":
        intent_identity = operator_module._intent_identity

        def mutate_after_identity(**kwargs: object) -> dict[str, object]:
            identity = intent_identity(**kwargs)
            plan.write_bytes(original_plan + b"# changed after capture\n")
            return identity

        monkeypatch.setattr(operator_module, "_intent_identity", mutate_after_identity)
    else:
        write_intent = operator_module._write_create_once_json

        def mutate_after_intent(path: Path, payload: Mapping[str, object]) -> None:
            write_intent(path, payload)
            if path.name == "operation-intent.json":
                config.write_bytes(original_config + b"# changed after publication\n")

        monkeypatch.setattr(operator_module, "_write_create_once_json", mutate_after_intent)

    result = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=_dependencies(_FakeLifecycle(calls), emit=[]),
    )

    assert result.status == "configuration-mismatch"
    assert calls == []
    assert not (authority / PHASE_RUN_ID).exists()
    intent_path = execution / "operation-intent.json"
    assert intent_path.exists() is intent_published
    if intent_published:
        persisted = json.loads(intent_path.read_bytes())
        assert persisted["phase_plan"]["sha256"] == hashlib.sha256(original_plan).hexdigest()
        assert persisted["profile"]["config_sha256"] == hashlib.sha256(original_config).hexdigest()


def test_operator_restart_reuses_create_once_intent_and_skips_materialization(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
) -> None:
    plan, authority, execution, source, config = operator_inputs
    first_calls: list[tuple[str, dict[str, object]]] = []
    first = _FakeLifecycle(first_calls)
    dependencies = _dependencies(first, emit=[])
    assert (
        run_postprocessing_phase(
            plan,
            authority_root=authority,
            execution_root=execution,
            source_repo=source,
            config_path=config,
            dependencies=dependencies,
        ).status
        == "accepted"
    )
    intent = (execution / "operation-intent.json").read_bytes()

    restart_calls: list[tuple[str, dict[str, object]]] = []
    restart = _FakeLifecycle(restart_calls)
    result = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=_dependencies(restart, emit=[]),
    )

    assert result.status == "accepted"
    assert [name for name, _kwargs in restart_calls] == ["status", "fetch", "export", "finalize"]
    assert (execution / "operation-intent.json").read_bytes() == intent


def test_operator_v1_restart_does_not_adopt_a_manually_retried_successor(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
) -> None:
    plan, authority, execution, source, config = operator_inputs
    first = _FakeLifecycle([])
    assert (
        run_postprocessing_phase(
            plan,
            authority_root=authority,
            execution_root=execution,
            source_repo=source,
            config_path=config,
            dependencies=_dependencies(first, emit=[]),
        ).status
        == "accepted"
    )
    state_path = authority / PHASE_RUN_ID / "fake-state.json"
    state = json.loads(state_path.read_bytes())
    state["attempt_id"] = "attempt-0002"
    state_path.write_text(json.dumps(state))

    calls: list[tuple[str, dict[str, object]]] = []
    result = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=_dependencies(_FakeLifecycle(calls), emit=[]),
    )

    assert result.status == "configuration-mismatch"
    assert result.attempt_id == "attempt-0001"
    assert calls == [("status", {"authority_root": authority})]
    assert "attempt-0001" in (result.detail or "")
    assert "attempt-0002" in (result.detail or "")


@pytest.mark.parametrize("partial_stage", ["submission-intent", "dispatch-intent", "partial-assignments"])
def test_operator_restart_reenters_submit_for_incomplete_durable_initial_submission(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
    partial_stage: str,
) -> None:
    plan, authority, execution, source, config = operator_inputs
    calls: list[tuple[str, dict[str, object]]] = []
    fake = _PartialSubmissionLifecycle(calls=calls, partial_stage=partial_stage)
    dependencies = _dependencies(fake, emit=[])

    first = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=dependencies,
    )
    recovered = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=dependencies,
    )
    calls_before_complete_restart = len(calls)
    complete_restart = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=dependencies,
    )

    assert first.status == "lifecycle-failed"
    assert recovered.status == complete_restart.status == "accepted"
    assert sum(name == "submit" for name, _kwargs in calls) == 2
    assert [name for name, _kwargs in calls[calls_before_complete_restart:]] == [
        "status",
        "fetch",
        "export",
        "finalize",
    ]


@pytest.mark.parametrize("boundary", ("status", "submit", "resume", "finalize"))
@pytest.mark.parametrize("identity_field", ("phase_run_id", "attempt_id"))
def test_shared_attempt_driver_rejects_phase_or_attempt_drift_at_every_return_boundary(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
    boundary: str,
    identity_field: str,
) -> None:
    plan, authority, execution, source, config = operator_inputs
    calls: list[tuple[str, dict[str, object]]] = []
    lifecycle = _DriftingLifecycle(
        calls=calls,
        drift_boundary=boundary,
        drift_field=identity_field,
    )

    result = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=_dependencies(lifecycle, emit=[]),
    )

    assert result.status == "configuration-mismatch"
    assert "diagnostics" not in [name for name, _kwargs in calls]
    if boundary == "resume":
        assert [name for name, _kwargs in calls].count("status") == 2
        assert calls[-1][0] == "resume"


@pytest.mark.parametrize("boundary", ["intent", "materialize", "submit", "resume", "fetch", "export", "finalize"])
def test_operator_recovers_each_durable_boundary_without_invalid_repetition(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
    boundary: str,
) -> None:
    plan, authority, execution, source, config = operator_inputs
    calls: list[tuple[str, dict[str, object]]] = []
    fake = _FakeLifecycle(calls, crash_after=boundary)
    dependencies = _dependencies(fake, emit=[])

    first = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=dependencies,
    )
    recovered = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=dependencies,
    )

    assert first.status == "lifecycle-failed"
    assert recovered.status == "accepted"
    assert sum(name == "submit" for name, _kwargs in calls) == 1
    assert json.loads((execution / "current.json").read_bytes())["state"] == "accepted"


def test_operator_timeout_is_distinct_captures_warning_only_diagnostics_and_never_cancels(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
) -> None:
    plan, authority, execution, source, config = operator_inputs
    calls: list[tuple[str, dict[str, object]]] = []
    fake = _FakeLifecycle(calls)
    tick = iter((0.0, 0.1, 0.2, 0.8, 0.9, 1.0))
    dependencies = _dependencies(fake, emit=[])
    dependencies = PostprocessingPhaseOperatorDependencies(
        **{**dependencies.__dict__, "monotonic_clock": lambda: next(tick)}
    )

    result = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        timeout_seconds=0.5,
        dependencies=dependencies,
    )

    assert result.status == "timed-out"
    assert result.diagnostics == str(execution / "diagnostics/summary.json")
    assert [name for name, _kwargs in calls] == ["materialize", "diagnostics"]
    assert result.recovery_command is not None


def test_operator_changed_restart_options_fail_against_create_once_intent(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, authority, execution, source, config = operator_inputs
    calls: list[tuple[str, dict[str, object]]] = []
    fake = _FakeLifecycle(calls)
    dependencies = _dependencies(fake, emit=[])
    dependencies = PostprocessingPhaseOperatorDependencies(**{**dependencies.__dict__, "interrupted": lambda: True})
    assert (
        run_postprocessing_phase(
            plan,
            authority_root=authority,
            execution_root=execution,
            source_repo=source,
            config_path=config,
            timeout_seconds=60,
            dependencies=dependencies,
        ).status
        == "interrupted"
    )

    mismatch = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        timeout_seconds=61,
        dependencies=_dependencies(fake, emit=[]),
    )

    assert mismatch.status == "configuration-mismatch"
    assert calls == []
    assert mismatch.recovery_command is not None
    tokens = shlex.split(mismatch.recovery_command)
    assert tokens == [
        "bsppctl",
        "--config",
        str(config.resolve()),
        "phase",
        "run-postprocessing",
        str(plan.resolve()),
        "--authority-root",
        str(authority.resolve()),
        "--execution-root",
        str(execution.resolve()),
        "--source-repo",
        str(source.resolve()),
        "--poll-interval",
        "30",
        "--timeout",
        "60",
    ]
    captured: dict[str, object] = {}

    def run_stub(phase_plan: Path, **kwargs: object) -> object:
        captured["phase_plan"] = phase_plan
        captured.update(kwargs)
        return SimpleNamespace(status="accepted", render_json=lambda: "{}\n")

    monkeypatch.setattr(operator_module, "run_postprocessing_phase", run_stub)
    parsed = CliRunner().invoke(cli, tokens[1:])

    assert parsed.exit_code == 0
    assert captured["phase_plan"] == plan.resolve()
    assert captured["timeout_seconds"] == 60.0


def test_operator_rejects_malicious_persisted_phase_run_id_before_path_use(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
) -> None:
    plan, authority, execution, source, config = operator_inputs
    calls: list[tuple[str, dict[str, object]]] = []
    fake = _FakeLifecycle(calls)
    interrupted = _dependencies(fake, emit=[])
    interrupted = PostprocessingPhaseOperatorDependencies(**{**interrupted.__dict__, "interrupted": lambda: True})
    assert (
        run_postprocessing_phase(
            plan,
            authority_root=authority,
            execution_root=execution,
            source_repo=source,
            config_path=config,
            dependencies=interrupted,
        ).status
        == "interrupted"
    )
    intent_path = execution / "operation-intent.json"
    intent = json.loads(intent_path.read_bytes())
    intent["phase_run_id"] = "../../escaped"
    intent_path.write_text(json.dumps(intent, indent=2, sort_keys=True) + "\n")

    result = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=_dependencies(fake, emit=[]),
    )

    assert result.status == "configuration-mismatch"
    assert calls == []
    assert not (operator_inputs[0].parent / "escaped").exists()


def test_operator_rejects_preexisting_authority_for_a_different_phase_plan(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
) -> None:
    plan, authority, execution, source, config = operator_inputs
    calls: list[tuple[str, dict[str, object]]] = []
    fake = _FakeLifecycle(calls)
    interrupted = _dependencies(fake, emit=[])
    interrupted = PostprocessingPhaseOperatorDependencies(**{**interrupted.__dict__, "interrupted": lambda: True})
    assert (
        run_postprocessing_phase(
            plan,
            authority_root=authority,
            execution_root=execution,
            source_repo=source,
            config_path=config,
            dependencies=interrupted,
        ).status
        == "interrupted"
    )
    authority_path = authority / PHASE_RUN_ID
    authority_path.mkdir(parents=True)
    (authority_path / "phase-run.json").write_text(
        json.dumps(
            {"phase_run_id": PHASE_RUN_ID, "phase_plan_digest": "b" * 64},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    result = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=_dependencies(fake, emit=[]),
    )

    assert result.status == "configuration-mismatch"
    assert calls == []
    assert "does not match the persisted Phase Plan identity" in (result.detail or "")


def test_operator_interrupt_is_distinct_and_never_invokes_cancellation(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
) -> None:
    plan, authority, execution, source, config = operator_inputs
    calls: list[tuple[str, dict[str, object]]] = []
    fake = _FakeLifecycle(calls)
    dependencies = _dependencies(fake, emit=[])
    dependencies = PostprocessingPhaseOperatorDependencies(**{**dependencies.__dict__, "interrupted": lambda: True})

    result = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=dependencies,
    )

    assert result.status == "interrupted"
    assert "no cancellation was requested" in (result.detail or "")
    assert calls == []
    assert result.recovery_command is not None


def test_operator_recovery_command_has_one_execution_root_and_round_trips_through_cli(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, authority, execution, source, config = operator_inputs
    fake = _FakeLifecycle([])
    dependencies = _dependencies(fake, emit=[])
    dependencies = PostprocessingPhaseOperatorDependencies(**{**dependencies.__dict__, "interrupted": lambda: True})
    interrupted = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        poll_interval_seconds=2,
        timeout_seconds=60,
        dependencies=dependencies,
    )
    assert interrupted.recovery_command is not None
    tokens = shlex.split(interrupted.recovery_command)
    assert tokens == [
        "bsppctl",
        "--config",
        str(config.resolve()),
        "phase",
        "run-postprocessing",
        str(plan.resolve()),
        "--authority-root",
        str(authority.resolve()),
        "--execution-root",
        str(execution.resolve()),
        "--source-repo",
        str(source.resolve()),
        "--poll-interval",
        "2",
        "--timeout",
        "60",
    ]
    assert tokens.count(str(execution.resolve())) == 1
    captured: dict[str, object] = {}

    def run_stub(phase_plan: Path, **kwargs: object) -> object:
        captured["phase_plan"] = phase_plan
        captured.update(kwargs)
        return SimpleNamespace(status="accepted", render_json=lambda: "{}\n")

    monkeypatch.setattr(operator_module, "run_postprocessing_phase", run_stub)
    parsed = CliRunner().invoke(cli, tokens[1:])

    assert parsed.exit_code == 0
    assert captured["phase_plan"] == plan.resolve()
    assert captured["execution_root"] == execution.resolve()


def test_operator_lock_conflict_returns_without_lifecycle_effects(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
) -> None:
    plan, authority, execution, source, config = operator_inputs
    execution.mkdir()
    descriptor = os.open(execution / "operation.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    calls: list[tuple[str, dict[str, object]]] = []
    try:
        result = run_postprocessing_phase(
            plan,
            authority_root=authority,
            execution_root=execution,
            source_repo=source,
            config_path=config,
            dependencies=_dependencies(_FakeLifecycle(calls), emit=[]),
        )
    finally:
        os.close(descriptor)

    assert result.status == "lock-conflict"
    assert calls == []
    assert not (execution / "operation-intent.json").exists()


def test_operator_v1_holds_the_common_phase_coordinator_lock(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
) -> None:
    plan, authority, execution, source, config = operator_inputs
    calls: list[tuple[str, dict[str, object]]] = []

    with postprocessing_phase_coordinator_lock(authority, PHASE_RUN_ID):
        result = run_postprocessing_phase(
            plan,
            authority_root=authority,
            execution_root=execution,
            source_repo=source,
            config_path=config,
            dependencies=_dependencies(_FakeLifecycle(calls), emit=[]),
        )

    assert result.status == "lock-conflict"
    assert calls == []
    assert not (authority / PHASE_RUN_ID).exists()


def test_operator_reports_durable_cancelling_without_implicit_cancel(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
) -> None:
    plan, authority, execution, source, config = operator_inputs
    calls: list[tuple[str, dict[str, object]]] = []
    fake = _FakeLifecycle(calls, crash_after="materialize")
    dependencies = _dependencies(fake, emit=[])
    assert (
        run_postprocessing_phase(
            plan,
            authority_root=authority,
            execution_root=execution,
            source_repo=source,
            config_path=config,
            dependencies=dependencies,
        ).status
        == "lifecycle-failed"
    )
    fake.crash_after = None
    fake._save(authority, {"status": "cancelling", "terminal": False})
    calls.clear()

    result = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=dependencies,
    )

    assert result.status == "cancelling"
    assert result.explicit_cancel_command == (f"bsppctl phase cancel {PHASE_RUN_ID} --authority-root {authority}")
    assert [name for name, _kwargs in calls] == ["status"]


def test_diagnostics_failure_is_warning_only_and_preserves_lifecycle_result(
    operator_inputs: tuple[Path, Path, Path, Path, Path],
) -> None:
    plan, authority, execution, source, config = operator_inputs
    calls: list[tuple[str, dict[str, object]]] = []
    fake = _FakeLifecycle(calls, crash_after="materialize")
    dependencies = _dependencies(fake, emit=[])

    def fail_diagnostics(_phase_run_id: str, **_kwargs: object) -> object:
        raise OSError("diagnostics unavailable")

    dependencies = PostprocessingPhaseOperatorDependencies(**{**dependencies.__dict__, "diagnostics": fail_diagnostics})

    result = run_postprocessing_phase(
        plan,
        authority_root=authority,
        execution_root=execution,
        source_repo=source,
        config_path=config,
        dependencies=dependencies,
    )

    assert result.status == "lifecycle-failed"
    assert result.diagnostics == "diagnostics warning: diagnostics unavailable"


def test_run_postprocessing_cli_exposes_only_operator_inputs() -> None:
    result = CliRunner().invoke(cli, ["phase", "run-postprocessing", "--help"])

    assert result.exit_code == 0
    for option in (
        "--authority-root",
        "--execution-root",
        "--source-repo",
        "--poll-interval",
        "--timeout",
    ):
        assert option in result.output
    for forbidden in (
        "--scheduler-evidence",
        "--handoff",
        "--aggregate-action-evidence",
        "--acceptance-adjudication",
        "--diagnostics-root",
    ):
        assert forbidden not in result.output


def test_retry_postprocessing_help_exposes_explicit_existing_phase_interface() -> None:
    result = CliRunner().invoke(cli, ["phase", "retry-postprocessing", "--help"])

    assert result.exit_code == 0
    assert "PHASE_RUN_ID" in result.output
    for option in ("--authority-root", "--execution-root", "--source-repo", "--poll-interval", "--timeout"):
        assert option in result.output
    assert "--profile" not in result.output


def test_run_postprocessing_cli_is_a_thin_operator_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tmp_path / "phase-plan.yaml"
    plan.write_text("phase_kind: postprocessing\n")
    config = tmp_path / "profiles.yaml"
    config.write_text("clusters: {}\n")
    source = tmp_path / "source"
    source.mkdir()
    captured: dict[str, object] = {}

    def run_stub(phase_plan: Path, **kwargs: object) -> object:
        captured["phase_plan"] = phase_plan
        captured.update(kwargs)
        return SimpleNamespace(status="accepted", render_json=lambda: "{}\n")

    monkeypatch.setattr(operator_module, "run_postprocessing_phase", run_stub)
    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config),
            "phase",
            "run-postprocessing",
            str(plan),
            "--authority-root",
            str(tmp_path / "authority"),
            "--execution-root",
            str(tmp_path / "execution"),
            "--source-repo",
            str(source),
            "--poll-interval",
            "2",
            "--timeout",
            "60",
        ],
    )

    assert result.exit_code == 0
    assert captured["phase_plan"] == plan
    assert captured["config_path"] == config
    assert captured["poll_interval_seconds"] == 2.0
    assert captured["timeout_seconds"] == 60.0
    assert "scheduler_evidence_path" not in captured
    assert "handoff_path" not in captured


def test_operator_dependency_surface_has_no_cancel_or_environment_preparation_hooks() -> None:
    assert set(PostprocessingPhaseOperatorDependencies.__dataclass_fields__) == {
        "materialize",
        "submit",
        "status",
        "resume",
        "fetch",
        "export_scheduler",
        "finalize",
        "diagnostics",
        "wall_clock",
        "monotonic_clock",
        "sleeper",
        "interrupted",
        "phase_run_id_factory",
        "bind_initial_attempt",
        "emit",
    }
    source = Path(operator_module.__file__).read_text()
    assert "cancel_postprocessing_phase" not in source
    assert "subprocess" not in source


def test_operator_dependency_constructor_keeps_the_pre_binding_v1_call_shape() -> None:
    dependencies = _dependencies(_FakeLifecycle([]), emit=[])
    pre_binding_fields = {
        name: value for name, value in dependencies.__dict__.items() if name != "bind_initial_attempt"
    }

    compatible = PostprocessingPhaseOperatorDependencies(**pre_binding_fields)

    assert callable(compatible.bind_initial_attempt)


def test_compact_progress_keeps_stored_gate_order_and_observation_precedence() -> None:
    observed = SimpleNamespace(
        status="submitted",
        details={
            "actions": [
                {"action_id": "already-terminal", "job_id": "10", "terminal": {}},
                {
                    "action_id": "gate-action",
                    "durable_status": "submitted",
                    "job_id": "11",
                    "terminal": None,
                    "scheduler": {"state": ""},
                    "tasks": [
                        {"task_index": 1, "observation_status": "observed", "state": "COMPLETED"},
                        {"task_index": 2, "observation_status": "not-applicable", "state": None},
                        {"task_index": 3, "observation_status": "missing", "state": "PENDING"},
                    ],
                },
                {"action_id": "later-action", "durable_status": "planned", "job_id": None, "terminal": None},
            ]
        },
    )

    assert operator_module._compact_progress(observed) == (
        "authority_status=submitted assigned=2/3 terminal=1/3 gate=gate-action task_index=3 state=PENDING"
    )

    observed.details["actions"][1]["scheduler"] = {"state": "RUNNING"}
    assert operator_module._compact_progress(observed) == (
        "authority_status=submitted assigned=2/3 terminal=1/3 gate=gate-action state=RUNNING"
    )

    observed.details["actions"][1]["scheduler"] = None
    observed.details["actions"][1]["tasks"] = [
        {"task_index": 1, "observation_status": "observed", "state": "COMPLETED"},
        {"task_index": 2, "observation_status": "not-applicable", "state": None},
    ]
    assert operator_module._compact_progress(observed) == (
        "authority_status=submitted assigned=2/3 terminal=1/3 gate=gate-action task_index=1 state=COMPLETED"
    )

    observed.details["actions"][1]["tasks"] = []
    assert operator_module._compact_progress(observed) == (
        "authority_status=submitted assigned=2/3 terminal=1/3 gate=gate-action state=submitted"
    )


def test_transition_writer_emits_detail_changes_and_exact_300_second_heartbeats(tmp_path: Path) -> None:
    emitted: list[dict[str, object]] = []
    monotonic = iter((0.0, 1.0, 2.0, 301.0, 302.0))
    wall_clock = iter(datetime(2026, 9, 3, 0, 0, second, tzinfo=UTC) for second in range(5))
    dependencies = _dependencies(_FakeLifecycle([]), emit=emitted)
    dependencies = PostprocessingPhaseOperatorDependencies(
        **{
            **dependencies.__dict__,
            "monotonic_clock": lambda: next(monotonic),
            "wall_clock": lambda: next(wall_clock),
        }
    )
    writer = operator_module._TransitionWriter(tmp_path, PHASE_RUN_ID, dependencies)

    writer.record("polling", attempt_id="attempt-0001", detail="same\x00detail")
    writer.record("polling", attempt_id="attempt-0001", detail="samedetail")
    current_after_duplicate = json.loads((tmp_path / "current.json").read_bytes())
    writer.record("polling", attempt_id="attempt-0001", detail="changed")
    writer.record("polling", attempt_id="attempt-0001", detail="changed")
    writer.record("polling", attempt_id="attempt-0001", detail="changed")

    assert [item["detail"] for item in emitted] == ["samedetail", "changed", "changed"]
    assert current_after_duplicate["detail"] == "samedetail"
    assert current_after_duplicate["updated_at"] == "2026-09-03T00:00:01Z"
    assert json.loads((tmp_path / "current.json").read_bytes())["updated_at"] == "2026-09-03T00:00:04Z"
