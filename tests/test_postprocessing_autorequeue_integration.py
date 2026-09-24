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

"""Cross-story autorequeue integration tests (coherence round 0).

These are the committed replacements for the throwaway runtime-lane script.
Each test pins one seam that no single story owned; they are intentionally
kept on typed fakes, disposable tmp_path fixtures, and the public Runtime
boundary so the suite stays independent of the mutable sibling checkouts.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_autorequeue_contract import (
    AUTOREQUEUE_ACTION_ID_ENV,
    AUTOREQUEUE_COMMAND_DIGEST_ENV,
    AUTOREQUEUE_ENV_NAMES,
    AUTOREQUEUE_PHASE_RUNSPEC_ENV,
    AUTOREQUEUE_RESTART_COUNT_FILE_ENV,
    POSTPROCESSING_RESTART_COUNT_FILENAME,
    postprocessing_restart_classification_filename,
    postprocessing_restart_evidence_relative_dir,
)
from bspp.orchestration.contract.postprocessing_autorequeue_policy import (
    PostprocessingAutorequeuePolicy,
)
from bspp.orchestration.contract.postprocessing_event import postprocessing_phase_event_from_mapping
from bspp.orchestration.contract.postprocessing_lifecycle import (
    PostprocessingActionTerminalObservedPayload,
)
from bspp.orchestration.contract.postprocessing_phase_ids import POSTPROCESSING_ACTION_IDS
from bspp.orchestration.control.monitoring import SlurmJobRecord, parse_sacct_json
from bspp.orchestration.control.phase_materialization import materialize_phase
from bspp.orchestration.control.postprocessing_authority_v2 import validate_postprocessing_authority
from bspp.orchestration.control.postprocessing_phase_adapter import (
    PostprocessingRenderInput,
    cancel_postprocessing_phase,
    render_postprocessing_action_script,
    resume_postprocessing_phase,
    submit_postprocessing_phase,
)
from bspp.orchestration.control.postprocessing_phase_lifecycle import _complete_action_task_set
from bspp.orchestration.control.postprocessing_phase_materialization import require_autorequeue_cap_for_policy
from bspp.orchestration.control.postprocessing_phase_retry import retry_postprocessing_phase
from bspp.orchestration.control.postprocessing_runtime_qualification import replay_postprocessing_runtime
from bspp.orchestration.control.postprocessing_scheduler_evidence import (
    _successful_task,
    export_postprocessing_scheduler_evidence,
)
from bspp.orchestration.control.transport import (
    CommandResult,
    RemoteSlurmTransport,
    SlurmAction,
    SlurmSubmission,
)
from bspp.orchestration.runtime.postprocessing import phase_artifacts as phase_artifacts_mod
from bspp.orchestration.runtime.postprocessing import scientific_output_snapshot as snapshot_mod
from bspp.orchestration.runtime.postprocessing.autorequeue_boundary import (
    handle_autorequeue_transport_failure,
)
from bspp.orchestration.runtime.postprocessing.failure_adapter import (
    TransportFailure,
)
from bspp.orchestration.runtime.postprocessing.finalization_bundle import (
    generate_scientific_output_root,
    publish_action09_finalization_bundle,
    record_successful_action_task,
)
from bspp.orchestration.runtime.postprocessing.phase_artifacts import attest_runtime_inputs
from bspp.orchestration.runtime.postprocessing.restart_classification import record_restart_classification
from bspp.orchestration.runtime.postprocessing.runtime_action_evidence import _load_prior_runtime_actions
from bspp.orchestration.runtime.postprocessing.transfer import copy_files_to_scratch
from tests.test_postprocessing_autorequeue_policy import (
    _refresh_runtime_qualification_reference,
    _write_autorequeue_cap,
)
from tests.test_postprocessing_finalization_bundle_runtime import (
    RuntimeFixture,
    _acceptance,
    _prepared_finalization,
    _write_tar,
)
from tests.test_postprocessing_finalization_bundle_runtime import (
    _fixture as _runtime_fixture,
)
from tests.test_postprocessing_phase_materialization import (
    NOW,
    RUN_ID,
    _finalizable_authority,
    _finalize_fixture,
    _fixture,
    _materialized_authority,
    _observation,
    _patch_transport,
    _render_input,
    _retry_resolution,
)
from tests.test_postprocessing_restart_classification import (
    _classification,
    _sha,
)
from tests.test_postprocessing_restart_classification import (
    _write_runspec as _write_v2_runspec,
)

_REPO = Path(__file__).parents[1]
_NOW_STR = "2026-09-03T12:00:00Z"
_HISTORICAL_V1 = _REPO / "tests" / "fixtures" / "postprocessing_phase" / "historical_v1"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    assert isinstance(payload, dict)
    return payload


def _enabled_cap_authority(tmp_path: Path) -> object:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    _write_autorequeue_cap(tmp_path)
    payload = yaml.safe_load(plan_path.read_text())
    payload = _refresh_runtime_qualification_reference(payload, tmp_path)
    payload["autorequeue_policy"] = {
        "schema_version": 1,
        "mode": "enabled",
        "action_ids": ["postprocessing-01-preflight", "postprocessing-04-slurm"],
    }
    plan_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    authority_root = tmp_path / "authority"
    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )
    return validate_postprocessing_authority(authority_root, RUN_ID)


def _enabled_runtime_fixture(tmp_path: Path) -> RuntimeFixture:
    authority = _enabled_cap_authority(tmp_path)
    runspec = authority.runspec
    attempt_root = authority.authority_path / "attempts" / runspec.attempt_id
    output_root = Path(runspec.payload.attempt_paths.output_dir)
    evidence_root = Path(runspec.payload.attempt_paths.evidence_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    batch_tar = output_root / "local_tars/shard_1/batch_0.tar"
    metadata_tar = output_root / "local_tars/metadata/shard_1_metadata.tar"
    _write_tar(batch_tar, (("b.cif.zst", b"beta"), ("a.json.zst", b"alpha")))
    _write_tar(metadata_tar, (("metadata.parquet", b"metadata"),))
    analysis = output_root / "analysis/analysis_metadata.parquet"
    analysis.parent.mkdir(parents=True)
    analysis.write_bytes(b"parquet fixture")
    (output_root / "local_tars.csv").write_text(
        "tar_type,tar_name,shard_id,size_bytes\n"
        f"batch,batch_0.tar,1,{batch_tar.stat().st_size}\n"
        f"metadata,shard_1_metadata.tar,1,{metadata_tar.stat().st_size}\n"
    )
    return RuntimeFixture(
        authority=authority,
        runspec_path=attempt_root / "phase-runspec.json",
        projection_path=authority.authority_path / runspec.payload.execution_projection.document_location,
        policy_path=authority.authority_path / runspec.payload.acceptance_policy.location,
        output_root=output_root,
        evidence_root=evidence_root,
        batch_tar=batch_tar,
        metadata_tar=metadata_tar,
    )


def _prepare_enabled_finalization(tmp_path: Path) -> RuntimeFixture:
    fixture = _enabled_runtime_fixture(tmp_path)
    attest_runtime_inputs(
        phase_runspec_path=fixture.runspec_path,
        qualification_path=fixture.runspec_path.parent / "runtime-qualification.json",
        evidence_root=fixture.evidence_root,
        output_path=fixture.evidence_root / "phase-inputs/runtime-input-attestations.json",
        observed_at=_NOW_STR,
        probe_runner=lambda _locator: {"access_kind": "fixture-stat-v1"},
    )
    _acceptance(fixture)
    generate_scientific_output_root(phase_runspec_path=fixture.runspec_path, workers=2)
    for action_number, action in enumerate(fixture.authority.runspec.payload.actions[:-1], start=1):
        for task_index in action.expected_task_indexes or (None,):
            parent_job_id = str(7000 + action_number)
            scheduler_job_id = parent_job_id if task_index is None else f"{parent_job_id}_{task_index}"
            record_successful_action_task(
                phase_runspec_path=fixture.runspec_path,
                action_id=action.action_id,
                command_digest=hashlib.sha256(action.action_id.encode()).hexdigest(),
                scheduler_job_id=scheduler_job_id,
                task_index=task_index,
                completed_at=_NOW_STR,
            )
    return fixture


def _cap_less_retry_resolver(*, tmp_path: Path, authority: object) -> object:
    base = _retry_resolution(tmp_path=tmp_path, authority=authority)

    class CapLessResolver:
        def resolve(self, **values: object) -> tuple[object, object, bytes]:
            profile, _selection, qualification_document = base.resolve(**values)
            payload = json.loads(qualification_document)
            payload.pop("autorequeue_cap", None)
            stripped = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            profile_name, selection = replay_postprocessing_runtime(stripped, attempt_id=values["attempt_id"])
            assert profile_name == profile.name
            return profile, selection, stripped

    return CapLessResolver()


def _action(*, indexes: tuple[int, ...]) -> object:
    return type("ArrayAction", (), {"expected_task_indexes": indexes})()


def _record(job_id: str, *, restarts: object) -> SlurmJobRecord:
    return SlurmJobRecord(
        job_id=job_id,
        source="sacct",
        requested_job_id=job_id.split("_", 1)[0],
        state="COMPLETED",
        exit_code="0:0",
        restarts=restarts,
    )


def test_t1_v3_payload_disabled_policy_digest_invisible(tmp_path: Path) -> None:
    authority = validate_postprocessing_authority(_materialized_authority(tmp_path), RUN_ID)
    payload = authority.runspec.payload
    disabled_mapping = payload.to_mapping()
    assert "autorequeue_policy" not in disabled_mapping

    enabled_payload = replace(
        payload,
        autorequeue_policy=PostprocessingAutorequeuePolicy(
            mode="enabled",
            action_ids=("postprocessing-01-preflight",),
        ),
    )
    enabled_mapping = enabled_payload.to_mapping()
    assert "autorequeue_policy" in enabled_mapping
    assert canonical_mapping_digest(enabled_mapping) != canonical_mapping_digest(disabled_mapping)


def test_t2_historical_terminal_observed_events_round_trip_byte_exactly() -> None:
    for index in range(21, 30):
        path = _HISTORICAL_V1 / "events" / f"{index:06d}-phase-action-terminal-observed.json"
        raw = _read_json(path)
        assert postprocessing_phase_event_from_mapping(raw).to_mapping() == raw


def test_t3_requeue_then_success_finalizes_action09(tmp_path: Path) -> None:
    fixture = _prepare_enabled_finalization(tmp_path)
    action_id = POSTPROCESSING_ACTION_IDS["preflight"]
    record_restart_classification(
        phase_runspec_path=fixture.runspec_path,
        action_id=action_id,
        command_digest=_sha("preflight-command"),
        classification=_classification(),
    )
    publish_action09_finalization_bundle(
        phase_runspec_path=fixture.runspec_path,
        execution_projection_path=fixture.projection_path,
        acceptance_policy_path=fixture.policy_path,
        command_digest="9" * 64,
        workers=2,
        assembled_at=_NOW_STR,
    )
    restart_root = fixture.evidence_root / "phase-actions/restarts" / action_id
    assert (restart_root / "restart-0000000000-classification.json").is_file()
    runtime_restart = (
        fixture.evidence_root / "phase-actions/runtime" / action_id / "restart-0000000000-classification.json"
    )
    assert not runtime_restart.exists()


def test_t4_success_tree_stays_exactly_closed(tmp_path: Path) -> None:
    fixture = _prepared_finalization(tmp_path)
    action_id = fixture.authority.runspec.payload.actions[0].action_id
    extra = fixture.evidence_root / "phase-actions/runtime" / action_id / "extra.txt"
    extra.write_text("bogus")
    with pytest.raises(ValueError, match="missing, extra, or unsafe"):
        _load_prior_runtime_actions(fixture.authority.runspec, fixture.evidence_root)


def test_t5_restart_subtree_invisible_to_output_walkers(tmp_path: Path) -> None:
    fixture = _runtime_fixture(tmp_path)
    baseline_phase = phase_artifacts_mod._regular_output_files(fixture.output_root, fixture.evidence_root)
    baseline_snapshot = snapshot_mod._regular_output_files(fixture.output_root, fixture.evidence_root)

    restart_dir = fixture.evidence_root / "phase-actions/restarts" / POSTPROCESSING_ACTION_IDS["preflight"]
    restart_dir.mkdir(parents=True)
    (restart_dir / "restart-0000000000-classification.json").write_text("{}")

    assert phase_artifacts_mod._regular_output_files(fixture.output_root, fixture.evidence_root) == baseline_phase
    assert snapshot_mod._regular_output_files(fixture.output_root, fixture.evidence_root) == baseline_snapshot


def test_t6_transport_failure_survives_exception_machinery() -> None:
    import contextlib
    import copy
    import pickle

    @contextlib.contextmanager
    def _raise_through_context() -> object:
        try:
            yield
        finally:
            pass

    failure = TransportFailure(kind="timeout", detail="connection timed out")
    with pytest.raises(TransportFailure) as exc_info, _raise_through_context():
        raise failure
    assert exc_info.value is failure
    assert str(exc_info.value) == "connection timed out"
    assert copy.copy(failure).detail == "connection timed out"
    restored = pickle.loads(pickle.dumps(failure))
    assert restored.kind == "timeout"
    assert restored.detail == "connection timed out"
    assert str(restored) == "connection timed out"


def test_t7_io_with_retry_audited_errno_logs_and_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    (src_dir / "a.txt").write_text("a")

    def _econnreset(*_args: object, **_kwargs: object) -> object:
        raise OSError(errno.ECONNRESET, "connection reset by peer")

    monkeypatch.setattr(shutil, "copy2", _econnreset)
    assert copy_files_to_scratch(src_dir, dst_dir, ["a.txt"]) == 0


def test_t8_end_to_end_exit_85_from_rendered_action_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority = _enabled_cap_authority(tmp_path)
    listed = authority.runspec.payload.actions[0]
    script = render_postprocessing_action_script(_render_input(authority), listed, renderer_contract_version=3)
    assert "#SBATCH --requeue" in script
    assert "SLURM_RESTART_COUNT" in script
    assert "BSPP_AUTOREQUEUE_RESTART_COUNT_FILE" in script
    assert "from bspp.orchestration.runtime.cli import main; main()" in script
    syntax = subprocess.run(("bash", "-n"), input=script, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr

    boundary_root = tmp_path / "boundary"
    boundary_root.mkdir()
    runspec_path = _write_v2_runspec(boundary_root, max_batch_requeue=2)
    action_id = POSTPROCESSING_ACTION_IDS["preflight"]
    evidence_root = boundary_root / "evidence"
    restart_count_file = evidence_root / "phase-actions/restarts" / action_id / "restart-count"
    restart_count_file.parent.mkdir(parents=True)
    restart_count_file.write_text("0")
    monkeypatch.setenv("BSPP_AUTOREQUEUE_PHASE_RUNSPEC", str(runspec_path))
    monkeypatch.setenv("BSPP_AUTOREQUEUE_ACTION_ID", action_id)
    monkeypatch.setenv("BSPP_AUTOREQUEUE_COMMAND_DIGEST", _sha("command"))
    monkeypatch.setenv("BSPP_AUTOREQUEUE_RESTART_COUNT_FILE", str(restart_count_file))

    entry = (
        "from bspp.orchestration.runtime import cli as cli_mod\n"
        "from bspp.orchestration.runtime.postprocessing.failure_adapter import TransportFailure\n"
        "def boom():\n"
        "    raise TransportFailure(kind='timeout', detail='connection timed out')\n"
        "cli_mod.cli = boom\n"
        "cli_mod.main()\n"
    )
    result = subprocess.run([sys.executable, "-c", entry], env={**os.environ}, capture_output=True, text=True)
    assert result.returncode == 85, result.stderr

    records = sorted((evidence_root / "phase-actions/restarts" / action_id).glob("restart-*.json"))
    assert [path.name for path in records] == ["restart-0000000000-classification.json"]
    assert not (evidence_root / "phase-actions/runtime" / action_id).exists()


def test_t9_restart_ordinal_carries_across_incarnations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runspec_path = _write_v2_runspec(tmp_path, max_batch_requeue=2)
    action_id = POSTPROCESSING_ACTION_IDS["preflight"]
    evidence_root = tmp_path / "evidence"
    restart_count_file = evidence_root / "phase-actions/restarts" / action_id / "restart-count"
    restart_count_file.parent.mkdir(parents=True)
    restart_count_file.write_text("0")
    monkeypatch.setenv("BSPP_AUTOREQUEUE_PHASE_RUNSPEC", str(runspec_path))
    monkeypatch.setenv("BSPP_AUTOREQUEUE_ACTION_ID", action_id)
    monkeypatch.setenv("BSPP_AUTOREQUEUE_COMMAND_DIGEST", _sha("command"))
    monkeypatch.setenv("BSPP_AUTOREQUEUE_RESTART_COUNT_FILE", str(restart_count_file))

    failure = TransportFailure(kind="timeout", detail="connection timed out")
    with pytest.raises(SystemExit) as first:
        handle_autorequeue_transport_failure(failure)
    assert first.value.code == 85

    restart_count_file.write_text("1")
    with pytest.raises(SystemExit) as second:
        handle_autorequeue_transport_failure(failure)
    assert second.value.code == 85

    records = sorted((evidence_root / "phase-actions/restarts" / action_id).glob("restart-*.json"))
    assert [path.name for path in records] == [
        "restart-0000000000-classification.json",
        "restart-0000000001-classification.json",
    ]


@pytest.mark.parametrize("kind", ["oom", "credential-error", "unknown"])
def test_t10_group_b_never_exits_85(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    runspec_path = _write_v2_runspec(tmp_path, max_batch_requeue=2)
    action_id = POSTPROCESSING_ACTION_IDS["preflight"]
    evidence_root = tmp_path / "evidence"
    restart_count_file = evidence_root / "phase-actions/restarts" / action_id / "restart-count"
    restart_count_file.parent.mkdir(parents=True)
    restart_count_file.write_text("0")
    monkeypatch.setenv("BSPP_AUTOREQUEUE_PHASE_RUNSPEC", str(runspec_path))
    monkeypatch.setenv("BSPP_AUTOREQUEUE_ACTION_ID", action_id)
    monkeypatch.setenv("BSPP_AUTOREQUEUE_COMMAND_DIGEST", _sha("command"))
    monkeypatch.setenv("BSPP_AUTOREQUEUE_RESTART_COUNT_FILE", str(restart_count_file))

    failure = TransportFailure(kind=kind, detail="non-audited failure")  # type: ignore[arg-type]
    assert handle_autorequeue_transport_failure(failure) is None
    restart_dir = evidence_root / "phase-actions/restarts" / action_id
    assert sorted(path.name for path in restart_dir.glob("restart-*.json")) == []


def test_t11_disabled_and_unlisted_renders_have_no_autorequeue_lines(tmp_path: Path) -> None:
    authority = validate_postprocessing_authority(_materialized_authority(tmp_path), RUN_ID)
    for action in authority.runspec.payload.actions:
        script = render_postprocessing_action_script(_render_input(authority), action, renderer_contract_version=3)
        assert "BSPP_AUTOREQUEUE_" not in script
        assert "restart-count" not in script
        assert "from bspp.orchestration.runtime.cli import main; main()" not in script

    enabled = PostprocessingAutorequeuePolicy(mode="enabled", action_ids=("postprocessing-01-preflight",))
    enabled_payload = replace(authority.runspec.payload, autorequeue_policy=enabled)
    enabled_runspec = replace(authority.runspec, payload=enabled_payload)
    render_input = PostprocessingRenderInput(runspec=enabled_runspec, legacy_runspec=authority.legacy_runspec)
    unlisted = authority.runspec.payload.actions[1]
    script = render_postprocessing_action_script(render_input, unlisted, renderer_contract_version=3)
    assert "BSPP_AUTOREQUEUE_" not in script
    assert "restart-count" not in script
    assert "from bspp.orchestration.runtime.cli import main; main()" not in script


def test_t12_action09_rejected_at_policy_boundary() -> None:
    with pytest.raises(ValueError, match="Actions 01--08"):
        PostprocessingAutorequeuePolicy(
            mode="enabled",
            action_ids=(POSTPROCESSING_ACTION_IDS["acceptance-adjudication"],),
        )
    for action_id in tuple(POSTPROCESSING_ACTION_IDS.values())[:-1]:
        PostprocessingAutorequeuePolicy(mode="enabled", action_ids=(action_id,))


def test_t13_retry_reasserts_cap(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    _write_autorequeue_cap(tmp_path)
    payload = yaml.safe_load(plan_path.read_text())
    payload = _refresh_runtime_qualification_reference(payload, tmp_path)
    payload["autorequeue_policy"] = {
        "schema_version": 1,
        "mode": "enabled",
        "action_ids": ["postprocessing-01-preflight", "postprocessing-04-slurm"],
    }
    plan_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    authority_root = tmp_path / "authority"
    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    resolver = _cap_less_retry_resolver(tmp_path=tmp_path, authority=predecessor)
    with pytest.raises(ValueError, match="autorequeue"):
        retry_postprocessing_phase(
            RUN_ID,
            authority_root=authority_root,
            config_path=profile_path,
            source_repo=source_repo,
            clock=lambda: NOW,
            runtime_resolver=resolver,
        )


def test_t13_shared_cap_helper_fails_closed() -> None:
    from bspp.orchestration.contract.postprocessing_execution import QualifiedPostprocessingRuntimeSelection

    cap_less = QualifiedPostprocessingRuntimeSelection(
        tuple_id="1" * 64,
        qualification_location="/qualification.json",
        qualification_sha256="1" * 64,
        qualification_size_bytes=1,
        qualified_at=_NOW_STR,
        expires_at=_NOW_STR,
        image_path="/image.sqsh",
        image_sha256="1" * 64,
        image_size_bytes=1,
        image_policy="digest-checked",
        source_kind="baked",
        source_revision="a" * 40,
        source_package_path="/source.tar",
        toolkit_package_path=None,
        runtime_ipsae_binary_path="/ipsae",
        runtime_ipsae_binary_sha256="1" * 64,
        runtime_ipsae_binary_size_bytes=1,
        source_identity_digest="1" * 64,
        source_package_identity_digest="1" * 64,
        toolkit_identity_digest="1" * 64,
        bootstrap_sha256="1" * 64,
        runtime_component_identity_digest="1" * 64,
        requeue_exit=None,
        max_batch_requeue=None,
    )
    enabled = PostprocessingAutorequeuePolicy(mode="enabled", action_ids=("postprocessing-01-preflight",))
    with pytest.raises(ValueError, match="autorequeue"):
        require_autorequeue_cap_for_policy(policy=enabled, qualified_runtime=cap_less)
    disabled = PostprocessingAutorequeuePolicy(mode="disabled", action_ids=())
    assert require_autorequeue_cap_for_policy(policy=disabled, qualified_runtime=cap_less) is None


def test_t14_alternative_restarts_spelling_and_disabled_tolerance() -> None:
    records = parse_sacct_json(
        json.dumps(
            {
                "jobs": [
                    {"job_id_raw": "1001", "state": "COMPLETED", "exit_code": "0:0", "restart_cnt": 0},
                ]
            }
        ),
        requested=("1001",),
    )
    assert [record.restarts for record in records] == [0]

    action = _action(indexes=())
    record = _record("1001", restarts=None)
    disabled = _complete_action_task_set(action, "1001", (record,), autorequeue_enabled=False)
    assert disabled is not None
    assert [item.restarts for item in disabled] == [None]
    assert _complete_action_task_set(action, "1001", (record,), autorequeue_enabled=True) is None


def test_t15_disabled_policy_restarts_none_survives_resume_export_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume_root = tmp_path / "resume"
    resume_root.mkdir()
    authority_root = _materialized_authority(resume_root)
    submitted: list[SlurmAction] = []

    def submit_action(_transport: RemoteSlurmTransport, action: SlurmAction) -> SlurmSubmission:
        submitted.append(action)
        job_id = str(5000 + len(submitted))
        result = CommandResult(argv=("sbatch",), returncode=0, stdout=f"{job_id}\n", stderr="")
        return SlurmSubmission(job_id=job_id, command=result.argv, result=result)

    _patch_transport(monkeypatch, submit_action=submit_action)
    submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    action_ids = tuple(action.action_id for action in authority.runspec.payload.actions)
    job_by_action = dict(zip(action_ids, (str(5001 + index) for index in range(len(action_ids))), strict=True))
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "query_observation_best_effort",
        lambda _transport, _job_ids, **_kwargs: _observation(authority, job_by_action=job_by_action, restarts=None),
    )
    resumed = resume_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    assert resumed.details["newly_terminal_action_ids"] == list(action_ids)

    replayed = validate_postprocessing_authority(authority_root, RUN_ID)
    assert len(replayed.terminal_payloads) == len(action_ids)
    for payload in replayed.terminal_payloads:
        assert isinstance(payload, PostprocessingActionTerminalObservedPayload)
        assert all(task.restarts is None for task in payload.tasks)

    event_files = sorted((authority.authority_path / "events").glob("*-phase-action-terminal-observed.json"))
    assert len(event_files) == len(action_ids)
    for path in event_files:
        raw = _read_json(path)
        tasks = raw["payload"]["tasks"]
        assert isinstance(tasks, list)
        for task in tasks:
            assert isinstance(task, dict)
            assert "restarts" not in task

    scheduler_path = tmp_path / "scheduler-evidence.json"
    export_postprocessing_scheduler_evidence(RUN_ID, authority_root=authority_root, output=scheduler_path)
    scheduler_raw = _read_json(scheduler_path)
    actions = scheduler_raw["actions"]
    assert isinstance(actions, list)
    for action in actions:
        assert isinstance(action, dict)
        for task in action["tasks"]:
            assert isinstance(task, dict)
            assert "restarts" not in task

    finalize_root = tmp_path / "finalize"
    finalize_root.mkdir()
    finalizable = _finalizable_authority(finalize_root, monkeypatch, restarts=None)
    finalized = _finalize_fixture(finalizable)
    assert finalized.phase_receipt_id


def test_t16_frozen_pre_epic_terminal_event_projects_successfully() -> None:
    path = _HISTORICAL_V1 / "events" / "000021-phase-action-terminal-observed.json"
    event = postprocessing_phase_event_from_mapping(_read_json(path))
    assert isinstance(event.payload, PostprocessingActionTerminalObservedPayload)
    for task in event.payload.tasks:
        assert task.restarts is None
        receipt = _successful_task(task)
        receipt_mapping = receipt.to_mapping()
        assert "restarts" not in receipt_mapping
        assert receipt_mapping == task.to_mapping()


def test_t17_enabled_policy_still_requires_exact_restarts_to_go_terminal() -> None:
    action = _action(indexes=())
    record = _record("1001", restarts=None)
    assert _complete_action_task_set(action, "1001", (record,), autorequeue_enabled=True) is None

    disabled = _complete_action_task_set(action, "1001", (record,), autorequeue_enabled=False)
    assert disabled is not None
    assert len(disabled) == 1
    assert disabled[0].restarts is None
    receipt = _successful_task(disabled[0])
    assert receipt.restarts is None
    assert "restarts" not in receipt.to_mapping()


def test_t18_contract_owned_handshake_is_the_single_source_for_both_planes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled_root = tmp_path / "enabled"
    enabled_root.mkdir()
    enabled = _enabled_cap_authority(enabled_root)
    array_action = enabled.runspec.payload.actions[3]
    assert array_action.expected_task_indexes
    script = render_postprocessing_action_script(_render_input(enabled), array_action, renderer_contract_version=3)
    for env_name in AUTOREQUEUE_ENV_NAMES:
        assert env_name in script
    evidence_dir = enabled.runspec.payload.attempt_paths.evidence_dir
    restart_count_path = (
        f"{evidence_dir}/{postprocessing_restart_evidence_relative_dir(array_action.action_id)}"
        f"/{POSTPROCESSING_RESTART_COUNT_FILENAME}"
    )
    assert f"{restart_count_path}${{SLURM_ARRAY_TASK_ID}}" in script

    boundary_root = tmp_path / "boundary"
    boundary_root.mkdir()
    runspec_path = _write_v2_runspec(boundary_root, max_batch_requeue=2)
    action_id = POSTPROCESSING_ACTION_IDS["preflight"]
    evidence_root = boundary_root / "evidence"
    restart_count_file = (
        evidence_root / postprocessing_restart_evidence_relative_dir(action_id) / POSTPROCESSING_RESTART_COUNT_FILENAME
    )
    restart_count_file.parent.mkdir(parents=True)
    restart_count_file.write_text("0")
    monkeypatch.setenv(AUTOREQUEUE_PHASE_RUNSPEC_ENV, str(runspec_path))
    monkeypatch.setenv(AUTOREQUEUE_ACTION_ID_ENV, action_id)
    monkeypatch.setenv(AUTOREQUEUE_COMMAND_DIGEST_ENV, _sha("command"))
    monkeypatch.setenv(AUTOREQUEUE_RESTART_COUNT_FILE_ENV, str(restart_count_file))
    failure = TransportFailure(kind="timeout", detail="connection timed out")
    with pytest.raises(SystemExit) as exited:
        handle_autorequeue_transport_failure(failure)
    assert exited.value.code == 85
    record_path = (
        evidence_root
        / postprocessing_restart_evidence_relative_dir(action_id)
        / postprocessing_restart_classification_filename(None, 0)
    )
    assert record_path.is_file()
    restart_dir = evidence_root / postprocessing_restart_evidence_relative_dir(action_id)
    assert [path.name for path in sorted(restart_dir.glob("restart-*.json"))] == [
        "restart-0000000000-classification.json"
    ]

    disabled_root = tmp_path / "disabled"
    disabled_root.mkdir()
    disabled = validate_postprocessing_authority(_materialized_authority(disabled_root), RUN_ID)
    for action in disabled.runspec.payload.actions:
        disabled_script = render_postprocessing_action_script(
            _render_input(disabled), action, renderer_contract_version=3
        )
        assert not any(env_name in disabled_script for env_name in AUTOREQUEUE_ENV_NAMES)
        assert POSTPROCESSING_RESTART_COUNT_FILENAME not in disabled_script
