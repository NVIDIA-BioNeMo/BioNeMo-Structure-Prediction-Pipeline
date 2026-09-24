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

"""Literal historical V1 readability, rendering, and mutation-barrier locks."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_runspec import read_postprocessing_phase_runspec_from_mapping
from bspp.orchestration.contract.postprocessing_runspec_v1 import HistoricalPostprocessingPhaseRunSpecV1
from bspp.orchestration.control.phase_cancellation import cancel_phase
from bspp.orchestration.control.phase_finalization import finalize_phase
from bspp.orchestration.control.phase_resume import resume_phase
from bspp.orchestration.control.phase_retry import retry_phase
from bspp.orchestration.control.phase_status import status_phase
from bspp.orchestration.control.phase_submission import submit_phase
from bspp.orchestration.control.postprocessing_authority_reader import (
    HISTORICAL_V1_MUTATION_ERROR,
    read_postprocessing_authority,
)
from bspp.orchestration.control.postprocessing_evidence_transfer import (
    fetch_postprocessing_finalization_evidence,
)
from bspp.orchestration.control.postprocessing_phase_cancellation import cancel_postprocessing_phase
from bspp.orchestration.control.postprocessing_phase_finalization import finalize_postprocessing_phase
from bspp.orchestration.control.postprocessing_phase_observation import (
    resume_postprocessing_phase,
    status_postprocessing_phase,
)
from bspp.orchestration.control.postprocessing_phase_rendering_v1 import (
    HISTORICAL_POSTPROCESSING_RENDERERS,
    PostprocessingRenderInput,
    historical_postprocessing_renderer,
    render_historical_postprocessing_action_script_v1,
)
from bspp.orchestration.control.postprocessing_phase_retry import retry_postprocessing_phase
from bspp.orchestration.control.postprocessing_phase_submission import submit_postprocessing_phase
from bspp.orchestration.control.postprocessing_phase_types import (
    HistoricalPostprocessingAuthorityV1,
    PostprocessingSubmissionState,
)
from bspp.orchestration.control.postprocessing_scheduler_evidence import (
    export_postprocessing_scheduler_evidence,
)
from bspp.orchestration.control.transport import CommandResult

RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
_FIXTURE = Path(__file__).parent / "fixtures" / "postprocessing_phase" / "historical_v1"
_RUNSPEC = _FIXTURE / "attempts" / "attempt-0001" / "phase-runspec.json"


def test_literal_v1_fixture_manifest_freezes_complete_inventory() -> None:
    manifest = _read_json(_FIXTURE / "MANIFEST.json")
    declared = manifest["files"]
    assert isinstance(declared, list)
    actual_paths = tuple(
        path.relative_to(_FIXTURE).as_posix()
        for path in sorted(_FIXTURE.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    )
    declared_paths = tuple(item["path"] for item in declared)
    assert declared_paths == tuple(sorted(declared_paths)) == actual_paths
    for item in declared:
        payload = (_FIXTURE / item["path"]).read_bytes()
        assert item["size_bytes"] == len(payload)
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()


def test_literal_v1_runspec_is_fully_typed_and_round_trips() -> None:
    raw = _read_json(_RUNSPEC)
    runspec = read_postprocessing_phase_runspec_from_mapping(raw)

    assert isinstance(runspec, HistoricalPostprocessingPhaseRunSpecV1)
    assert runspec.to_mapping() == raw
    assert runspec.payload.action_semantics_digest == "f90eb0aadb1072dc67b11c7343bdea5bd216e41ba7aff1c87abff05dee2b4550"
    assert runspec.payload.actions[0].action_id == "postprocessing-01-preflight"
    assert runspec.payload.logical_inputs.entries[0].member_identity
    assert runspec.payload.execution_projection.phase_identity.substitutions == runspec.payload.attempt_paths


@pytest.mark.parametrize(
    ("event_limit", "expected_status", "expected_jobs"),
    ((1, "materialized", 0), (4, "submitted", 1), (None, "accepted", 9)),
)
def test_literal_event_prefixes_read_materialized_active_and_accepted_authority(
    tmp_path: Path,
    event_limit: int | None,
    expected_status: str,
    expected_jobs: int,
) -> None:
    authority_root = _literal_authority(tmp_path, event_limit=event_limit)
    authority = read_postprocessing_authority(authority_root, RUN_ID)

    assert isinstance(authority, HistoricalPostprocessingAuthorityV1)
    assert authority.status == expected_status
    jobs = authority.submission_state.job_ids_by_action() if authority.submission_state else {}
    assert len(jobs) == expected_jobs
    assert authority.sealed is (expected_status == "accepted")


@pytest.mark.parametrize("event_limit", (1, 4, None))
def test_v1_status_projection_is_scheduler_free_through_both_routes(
    tmp_path: Path,
    event_limit: int | None,
) -> None:
    authority_root = _literal_authority(tmp_path, event_limit=event_limit)
    authority = read_postprocessing_authority(authority_root, RUN_ID)
    before = _tree_bytes(authority_root)
    transport_calls: list[tuple[str, ...]] = []

    def fail_transport(argv: tuple[str, ...]) -> CommandResult:
        transport_calls.append(argv)
        raise AssertionError("historical V1 status reached scheduler transport")

    direct = status_postprocessing_phase(RUN_ID, authority_root=authority_root, runner=fail_transport)
    routed = status_phase(RUN_ID, authority_root=authority_root, runner=fail_transport)

    assert direct.to_mapping() == routed.to_mapping()
    assert direct.details["contract_family"] == "postprocessing-runspec-v1"
    assert direct.details["read_only"] is True
    assert direct.details["scheduler"] is None
    assert direct.details["warnings"] == []
    json_before_table = direct.render_json()
    action_rows = _cast_mapping_list(direct.details["actions"])
    assert len(action_rows) == 9
    terminals = {terminal.action_id: terminal for terminal in authority.terminal_payloads}
    jobs = authority.submission_state.job_ids_by_action() if authority.submission_state else {}
    for row in action_rows:
        action_id = _cast_string(row["action_id"])
        assert row["scheduler"] is None
        tasks = _cast_mapping_list(row["tasks"])
        terminal = terminals.get(action_id)
        if terminal is not None:
            assert tasks == [
                {
                    "task_index": task.task_index,
                    "scheduler_job_id": task.scheduler_job_id,
                    "observation_status": "observed",
                    "state": task.state,
                    "exit_code": task.exit_code,
                    "source": task.source,
                }
                for task in terminal.tasks
            ]
        else:
            expected_state = "missing" if action_id in jobs else "not-applicable"
            assert {task["observation_status"] for task in tasks} == {expected_state}
            assert all(task["state"] is None and task["exit_code"] is None and task["source"] is None for task in tasks)
    table = direct.render_table()
    assert "phase_run_id:" in table
    assert (
        "scheduler_status=missing scheduler_state=missing scheduler_source=missing scheduler_exit_code=missing" in table
    )
    if event_limit is not None:
        assert "state=null source=null exit_code=null" in table
    assert direct.render_json() == json_before_table
    assert not transport_calls
    assert _tree_bytes(authority_root) == before


def test_v1_renderer_matches_every_literal_script_golden(tmp_path: Path) -> None:
    authority_root = _literal_authority(tmp_path)
    authority = read_postprocessing_authority(authority_root, RUN_ID)
    render_input = PostprocessingRenderInput(runspec=authority.runspec, legacy_runspec=authority.legacy_runspec)

    for action in authority.runspec.payload.actions:
        expected = (_FIXTURE / "scripts" / f"{action.action_id}.sbatch").read_text()
        actual = render_historical_postprocessing_action_script_v1(render_input, action)
        assert actual == expected
        syntax = subprocess.run(("bash", "-n"), input=actual, text=True, capture_output=True, check=False)
        assert syntax.returncode == 0, syntax.stderr


def test_v1_renderer_registry_is_immutable() -> None:
    with pytest.raises(TypeError):
        HISTORICAL_POSTPROCESSING_RENDERERS[2] = render_historical_postprocessing_action_script_v1  # type: ignore[index]
    with pytest.raises(TypeError):
        del HISTORICAL_POSTPROCESSING_RENDERERS[1]  # type: ignore[attr-defined]
    assert historical_postprocessing_renderer(1) is render_historical_postprocessing_action_script_v1


def test_v1_semantic_tampering_is_rejected_after_surrounding_authority_is_rehashed(tmp_path: Path) -> None:
    authority_root = _literal_authority(tmp_path, event_limit=1)
    run_root = authority_root / RUN_ID
    runspec_path = run_root / "attempts/attempt-0001/phase-runspec.json"
    runspec = _read_json(runspec_path)
    runspec["payload"]["action_semantics_digest"] = "0" * 64
    _write_json(runspec_path, runspec)
    runspec_digest = canonical_mapping_digest(runspec)
    phase_run_path = run_root / "phase-run.json"
    phase_run = _read_json(phase_run_path)
    phase_run["attempts"][0]["phase_runspec_digest"] = runspec_digest
    _write_json(phase_run_path, phase_run)
    event_path = run_root / "events/000001-phase-materialized.json"
    event = _read_json(event_path)
    event["payload"]["phase_runspec_digest"] = runspec_digest
    _write_json(event_path, event)

    with pytest.raises(ValueError, match="action semantics digest differs from its stored preimage"):
        read_postprocessing_authority(authority_root, RUN_ID)


def test_v1_corruption_and_unknown_renderer_are_rejected(tmp_path: Path) -> None:
    corrupted_root = _literal_authority(tmp_path / "corrupt", event_limit=1)
    legacy = corrupted_root / RUN_ID / "attempts/attempt-0001/legacy-runspec.yaml"
    legacy.write_bytes(legacy.read_bytes() + b"# corruption\n")
    with pytest.raises(ValueError, match="bytes differ from the declared immutable identity"):
        read_postprocessing_authority(corrupted_root, RUN_ID)

    renderer_root = _literal_authority(tmp_path / "renderer", event_limit=2)
    event_path = renderer_root / RUN_ID / "events/000002-phase-submission-intended.json"
    event = _read_json(event_path)
    event["payload"]["actions"][0]["renderer_contract_version"] = 99
    _write_json(event_path, event)
    with pytest.raises(ValueError, match="unsupported postprocessing renderer contract version"):
        read_postprocessing_authority(renderer_root, RUN_ID)
    with pytest.raises(ValueError, match="unsupported historical postprocessing renderer contract version"):
        historical_postprocessing_renderer(99)


def test_every_v1_mutation_entry_point_fails_before_locks_writes_or_transport(tmp_path: Path) -> None:
    authority_root = _literal_authority(tmp_path, event_limit=1)
    nowhere = tmp_path / "must-not-exist"
    transport_calls: list[tuple[str, ...]] = []

    def fail_transport(argv: tuple[str, ...]) -> CommandResult:
        transport_calls.append(argv)
        raise AssertionError("historical V1 mutation reached transport")

    operations: tuple[tuple[str, Callable[[], object]], ...] = (
        (
            "postprocessing-submit",
            lambda: submit_postprocessing_phase(RUN_ID, authority_root=authority_root, runner=fail_transport),
        ),
        (
            "postprocessing-resume",
            lambda: resume_postprocessing_phase(RUN_ID, authority_root=authority_root, runner=fail_transport),
        ),
        (
            "postprocessing-cancel",
            lambda: cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, runner=fail_transport),
        ),
        (
            "postprocessing-retry",
            lambda: retry_postprocessing_phase(
                RUN_ID, authority_root=authority_root, config_path=nowhere, runner=fail_transport
            ),
        ),
        (
            "postprocessing-finalize",
            lambda: finalize_postprocessing_phase(
                RUN_ID,
                authority_root=authority_root,
                scheduler_evidence_path=nowhere,
                aggregate_action_evidence_path=nowhere,
                handoff_path=nowhere,
                acceptance_adjudication_path=nowhere,
            ),
        ),
        (
            "postprocessing-evidence-fetch",
            lambda: fetch_postprocessing_finalization_evidence(
                RUN_ID, authority_root=authority_root, destination=nowhere, runner=fail_transport
            ),
        ),
        (
            "postprocessing-evidence-export",
            lambda: export_postprocessing_scheduler_evidence(RUN_ID, authority_root=authority_root, output=nowhere),
        ),
        (
            "phase-submit",
            lambda: submit_phase(RUN_ID, authority_root=authority_root, runner=fail_transport),
        ),
        (
            "phase-resume",
            lambda: resume_phase(RUN_ID, authority_root=authority_root, runner=fail_transport),
        ),
        (
            "phase-cancel",
            lambda: cancel_phase(RUN_ID, authority_root=authority_root, runner=fail_transport),
        ),
        ("phase-retry", lambda: retry_phase(RUN_ID, authority_root=authority_root, config_path=nowhere)),
        ("phase-finalize", lambda: finalize_phase(RUN_ID, authority_root=authority_root)),
    )
    before = _tree_bytes(authority_root)
    for operation, invoke in operations:
        lock_path = authority_root / RUN_ID / ".operation.lock"
        with pytest.raises(ValueError, match="historical postprocessing V1 authority is read-only") as raised:
            invoke()
        assert str(raised.value) == HISTORICAL_V1_MUTATION_ERROR, operation
        assert not lock_path.exists(), operation
        assert not nowhere.exists(), operation
        assert not transport_calls, operation
        assert _tree_bytes(authority_root) == before, operation


def test_typed_replay_has_no_mapping_emulation_or_event_indexing() -> None:
    assert not issubclass(PostprocessingSubmissionState, Mapping)
    assert "__getitem__" not in PostprocessingSubmissionState.__dict__
    assert callable(PostprocessingSubmissionState.job_ids_by_action)
    roots = (
        "postprocessing_event_replay.py",
        "postprocessing_phase_lifecycle.py",
        "postprocessing_scheduler_evidence.py",
    )
    source_root = Path(__file__).parents[1] / "packages/orchestration-control/src/bspp/orchestration/control"
    for name in roots:
        tree = ast.parse((source_root / name).read_text())
        for node in ast.walk(tree):
            assert not (
                isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "event"
            ), name
            assert not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"event", "terminal", "receipt_action"}
                and node.func.attr in {"get", "to_mapping"}
            ), name


def _literal_authority(tmp_path: Path, *, event_limit: int | None = None) -> Path:
    authority_root = tmp_path / "authority"
    run_root = authority_root / RUN_ID
    shutil.copytree(_FIXTURE, run_root, ignore=shutil.ignore_patterns("MANIFEST.json", "scripts"))
    if event_limit is not None:
        for path in sorted((run_root / "events").glob("*.json")):
            if int(path.name.split("-", 1)[0]) > event_limit:
                path.unlink()
    return authority_root


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _cast_mapping_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise TypeError("expected a list of mappings")
    return list(value)


def _cast_string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("expected a string")
    return value
