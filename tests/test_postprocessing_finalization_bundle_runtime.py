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

"""Focused Runtime tests for scientific roots and Action 09 publication."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import multiprocessing
import os
import subprocess
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

import bspp.orchestration.runtime.postprocessing.finalization_io as finalization_io
import bspp.orchestration.runtime.postprocessing.phase_artifacts as phase_artifacts
from bspp.orchestration.contract.postprocessing_finalization_bundle import (
    POSTPROCESSING_FINALIZATION_FIXED_PATHS,
    postprocessing_action09_assembly_witness_from_mapping,
    postprocessing_handoff_index_from_mapping,
    postprocessing_runtime_action_evidence_aggregate_from_mapping,
    postprocessing_scientific_output_root_from_mapping,
    postprocessing_tar_manifest_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runspec_v3 import PostprocessingPhaseRunSpecV3
from bspp.orchestration.control.postprocessing_phase_adapter import validate_postprocessing_authority
from bspp.orchestration.runtime.postprocessing.finalization_bundle import (
    generate_scientific_output_root,
    publish_action09_finalization_bundle,
    record_successful_action_task,
)
from bspp.orchestration.runtime.postprocessing.finalization_io import (
    LOCAL_FINALIZATION_PUBLICATION_STORE,
    FinalizationPublicationStore,
)
from bspp.orchestration.runtime.postprocessing.phase_acceptance import (
    adjudicate_acceptance,
    capture_acceptance,
)
from bspp.orchestration.runtime.postprocessing.phase_artifacts import attest_runtime_inputs
from bspp.orchestration.runtime.postprocessing.scientific_output_snapshot import (
    LocalScientificSnapshotIO,
)
from tests.test_postprocessing_phase_materialization import _materialized_authority


def test_runtime_publication_falls_back_on_einval_and_never_replaces(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()

    def unsupported(*_args: object) -> None:
        raise OSError(errno.EINVAL, "forced unsupported capability")

    finalization_io._rename_directory_no_replace(source, destination, renameat2=unsupported)
    assert destination.is_dir()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    with pytest.raises(FileExistsError):
        finalization_io._rename_directory_no_replace(replacement, destination, renameat2=unsupported)
    assert replacement.is_dir()


def test_runtime_publication_forced_fallback_has_one_multiprocess_winner(tmp_path: Path) -> None:
    destination = tmp_path / "winner"
    sources = (tmp_path / "source-a", tmp_path / "source-b")
    for index, source in enumerate(sources):
        source.mkdir()
        (source / "identity").write_text(str(index))
    start = multiprocessing.Event()
    results: multiprocessing.Queue[str] = multiprocessing.Queue()

    def publish(source: Path) -> None:
        def unsupported(*_args: object) -> None:
            raise OSError(errno.EINVAL, "forced unsupported capability")

        start.wait(5)
        try:
            finalization_io._rename_directory_no_replace(source, destination, renameat2=unsupported)
        except FileExistsError:
            results.put("loser")
        else:
            results.put("winner")

    processes = tuple(multiprocessing.Process(target=publish, args=(source,)) for source in sources)
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    assert sorted(results.get(timeout=2) for _ in processes) == ["loser", "winner"]
    assert destination.is_dir()
    assert sum(source.exists() for source in sources) == 1


NOW = "2026-09-03T12:00:00Z"


def test_runtime_runspec_loaders_accept_v3(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    phase_artifact_runspec = phase_artifacts._load_runspec(fixture.runspec_path)
    finalization_runspec = finalization_io._load_runspec(fixture.runspec_path)

    assert isinstance(phase_artifact_runspec, PostprocessingPhaseRunSpecV3)
    assert finalization_runspec == phase_artifact_runspec


def test_object_probe_uses_resolved_profile_credentials_and_redacts_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    call_count = 0
    timeout_mode = False
    monkeypatch.setenv("PATH", "/fixture/bin")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-value")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://swiftstack.test")

    def run(argv: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal call_count, timeout_mode
        call_count += 1
        if timeout_mode:
            raise subprocess.TimeoutExpired(argv, 120)
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(argv, 0, stdout="2026 object\n", stderr="")

    monkeypatch.setattr(phase_artifacts.subprocess, "run", run)
    proof = phase_artifacts._probe_locator("s3://example-bucket/input/")
    assert captured["argv"] == (
        "s5cmd",
        "--endpoint-url",
        "https://swiftstack.test",
        "ls",
        "s3://example-bucket/input/*",
    )
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["AWS_ACCESS_KEY_ID"] == "access-value"
    assert env["AWS_SECRET_ACCESS_KEY"] == "secret-value"
    assert env["S3_ENDPOINT_URL"] == "https://swiftstack.test"
    assert env["PATH"] == "/fixture/bin"
    assert proof["access_kind"] == "s5cmd-listing-v1"

    monkeypatch.delenv("AWS_ACCESS_KEY_ID")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")
    monkeypatch.delenv("S3_ENDPOINT_URL")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "missing-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "missing-config"))
    with pytest.raises(ValueError, match="credentials are unavailable") as missing:
        phase_artifacts._probe_locator("s3://example-bucket/input/")
    assert "secret-value" not in str(missing.value)
    assert call_count == 1

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-value")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://swiftstack.test")
    timeout_mode = True
    with pytest.raises(ValueError, match="probe timed out") as timeout:
        phase_artifacts._probe_locator("s3://example-bucket/input/")
    assert "secret-value" not in str(timeout.value)


@dataclass(frozen=True)
class RuntimeFixture:
    authority: object
    runspec_path: Path
    projection_path: Path
    policy_path: Path
    output_root: Path
    evidence_root: Path
    batch_tar: Path
    metadata_tar: Path


def test_scientific_root_is_worker_deterministic_and_separates_metadata_tar(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    first = generate_scientific_output_root(phase_runspec_path=fixture.runspec_path, workers=1)
    second = generate_scientific_output_root(phase_runspec_path=fixture.runspec_path, workers=4)

    assert second == first
    assert len(first.tar_manifests) == 1
    assert {item.path for item in first.small_outputs} == {
        "analysis/analysis_metadata.parquet",
        "local_tars.csv",
        "local_tars/metadata/shard_1_metadata.tar",
    }
    phase_output = fixture.evidence_root / "phase-output"
    root = postprocessing_scientific_output_root_from_mapping(_mapping(phase_output / "scientific-output-root.json"))
    reference = root.tar_manifests[0]
    manifest = postprocessing_tar_manifest_from_mapping(
        _mapping(phase_output / reference.path.removeprefix("outputs/"))
    )
    assert manifest.tar_path == "local_tars/shard_1/batch_0.tar"
    assert [(item.path, item.size_bytes) for item in manifest.members] == [("a.json.zst", 5), ("b.cif.zst", 4)]


def test_scientific_root_rejects_unsafe_tar_member_and_detects_tar_mutation(tmp_path: Path) -> None:
    unsafe = _fixture(tmp_path / "unsafe", batch_members=(("../escape", b"bad"),))
    with pytest.raises(ValueError, match="unsafe"):
        generate_scientific_output_root(phase_runspec_path=unsafe.runspec_path)

    mutated = _fixture(tmp_path / "mutated")
    real_fstat = os.fstat
    calls = 0

    def mutating_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        try:
            target = Path(f"/proc/self/fd/{descriptor}").resolve()
        except FileNotFoundError:
            return real_fstat(descriptor)
        if target == mutated.batch_tar:
            calls += 1
            if calls == 2:
                with mutated.batch_tar.open("ab") as handle:
                    handle.write(b"mutation")
        return real_fstat(descriptor)

    class MutatingSnapshotIO(LocalScientificSnapshotIO):
        def fstat(self, descriptor: int) -> os.stat_result:
            return mutating_fstat(descriptor)

    with pytest.raises(ValueError, match="changed while"):
        generate_scientific_output_root(
            phase_runspec_path=mutated.runspec_path,
            snapshot_io=MutatingSnapshotIO(),
        )


def test_action09_requires_every_prior_runtime_task(tmp_path: Path) -> None:
    fixture = _prepared_finalization(tmp_path, omit_last_task=True)

    with pytest.raises(ValueError, match="missing, extra, or unsafe"):
        _publish(fixture)
    assert not (fixture.evidence_root / "phase-finalization").exists()


def test_existing_runtime_input_attestation_reconciles_full_qualification(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = fixture.evidence_root / "phase-inputs/runtime-input-attestations.json"
    qualification = fixture.runspec_path.parent / "runtime-qualification.json"
    attest_runtime_inputs(
        phase_runspec_path=fixture.runspec_path,
        qualification_path=qualification,
        evidence_root=fixture.evidence_root,
        output_path=output,
        observed_at=NOW,
        probe_runner=lambda _locator: {"access_kind": "fixture-stat-v1"},
    )
    stale = _mapping(output)
    runtime_qualification = stale["runtime_qualification"]
    assert isinstance(runtime_qualification, dict)
    runtime_qualification["record_sha256"] = "0" * 64
    _write_json(output, stale)

    with pytest.raises(ValueError, match="existing Runtime Qualification attestation differs"):
        attest_runtime_inputs(
            phase_runspec_path=fixture.runspec_path,
            qualification_path=qualification,
            evidence_root=fixture.evidence_root,
            output_path=output,
            observed_at=NOW,
            probe_runner=lambda _locator: {"access_kind": "fixture-stat-v1"},
        )


def test_action09_publication_is_atomic_and_exactly_idempotent(tmp_path: Path) -> None:
    fixture = _prepared_finalization(tmp_path)
    destination = fixture.evidence_root / "phase-finalization"
    real_publish = LOCAL_FINALIZATION_PUBLICATION_STORE.rename_directory_no_replace

    def interrupted(source: Path, target: Path) -> None:
        if Path(target) == destination:
            raise OSError("simulated publication crash")
        real_publish(source, target)

    with pytest.raises(OSError, match="publication crash"):
        _publish(
            fixture,
            publication_store=FinalizationPublicationStore(rename_directory_no_replace=interrupted),
        )
    assert not destination.exists()
    assert not tuple(fixture.evidence_root.glob(".phase-finalization.assembly-*"))

    first = _publish(fixture)
    snapshot = {path.relative_to(destination).as_posix(): path.read_bytes() for path in destination.rglob("*.json")}
    second = _publish(fixture)
    assert second == first
    assert snapshot == {
        path.relative_to(destination).as_posix(): path.read_bytes() for path in destination.rglob("*.json")
    }

    index = postprocessing_handoff_index_from_mapping(_mapping(destination / "handoff-index.json"))
    dynamic = {item.path for item in index.members if item.path.startswith("outputs/tar-manifests/")}
    assert set(item.path for item in index.members) == set(POSTPROCESSING_FINALIZATION_FIXED_PATHS) | dynamic
    assert len(dynamic) == 1
    witness = postprocessing_action09_assembly_witness_from_mapping(_mapping(destination / "acceptance/bundle.json"))
    assert witness.publication_claim == "none"
    aggregate = postprocessing_runtime_action_evidence_aggregate_from_mapping(
        _mapping(destination / "aggregate-action-evidence.json")
    )
    assert tuple(item.action_id for item in aggregate.completed_actions) == tuple(
        item.action_id for item in fixture.authority.runspec.payload.actions[:-1]
    )
    assert all(item.action_id != witness.action_id for item in aggregate.completed_actions)


def test_action09_uses_staged_manifests_without_rehashing_scientific_payloads(tmp_path: Path) -> None:
    fixture = _prepared_finalization(tmp_path)

    class NoTarReopenSnapshotIO(LocalScientificSnapshotIO):
        def open_tar(self, **_kwargs: object) -> tarfile.TarFile:
            raise AssertionError("Action 09 must not reopen or rehash scientific payloads")

    _publish(fixture, snapshot_io=NoTarReopenSnapshotIO())


def test_action09_rejects_mode_change_after_action05(tmp_path: Path) -> None:
    fixture = _prepared_finalization(tmp_path)
    fixture.batch_tar.chmod(0o600 if fixture.batch_tar.stat().st_mode & 0o777 != 0o600 else 0o640)

    with pytest.raises(ValueError, match="scientific output changed"):
        _publish(fixture)


@pytest.mark.parametrize("mutation", ["missing", "extra", "boolean", "negative"])
def test_action09_rejects_invalid_source_fingerprint_mode(tmp_path: Path, mutation: str) -> None:
    fixture = _prepared_finalization(tmp_path)
    fingerprint_path = fixture.evidence_root / "phase-output/source-fingerprints.json"
    fingerprint = dict(_mapping(fingerprint_path))
    files = fingerprint["files"]
    assert isinstance(files, list) and files and isinstance(files[0], dict)
    first = files[0]
    if mutation == "missing":
        del first["mode"]
    elif mutation == "extra":
        first["invented"] = 1
    elif mutation == "boolean":
        first["mode"] = True
    else:
        first["mode"] = -1
    fingerprint_path.write_text(json.dumps(fingerprint, separators=(",", ":"), sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="fingerprint"):
        _publish(fixture)


def test_action09_concurrent_publish_accepts_only_the_identical_winner(tmp_path: Path) -> None:
    fixture = _prepared_finalization(tmp_path)
    destination = fixture.evidence_root / "phase-finalization"
    real_publish = LOCAL_FINALIZATION_PUBLICATION_STORE.rename_directory_no_replace
    barrier = threading.Barrier(2)

    def synchronized(source: Path, target: Path) -> None:
        barrier.wait(timeout=5)
        real_publish(source, target)

    publication_store = FinalizationPublicationStore(rename_directory_no_replace=synchronized)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda _index: _publish(fixture, publication_store=publication_store),
                range(2),
            )
        )

    assert results[0] == results[1]
    assert destination.is_dir()
    assert not tuple(fixture.evidence_root.glob(".phase-finalization.assembly-*"))


def _prepared_finalization(tmp_path: Path, *, omit_last_task: bool = False) -> RuntimeFixture:
    fixture = _fixture(tmp_path)
    attest_runtime_inputs(
        phase_runspec_path=fixture.runspec_path,
        qualification_path=fixture.runspec_path.parent / "runtime-qualification.json",
        evidence_root=fixture.evidence_root,
        output_path=fixture.evidence_root / "phase-inputs/runtime-input-attestations.json",
        observed_at=NOW,
        probe_runner=lambda _locator: {"access_kind": "fixture-stat-v1"},
    )
    _acceptance(fixture)
    generate_scientific_output_root(phase_runspec_path=fixture.runspec_path, workers=2)
    actions = fixture.authority.runspec.payload.actions[:-1]
    final_action = actions[-1]
    final_task = (final_action.expected_task_indexes or (None,))[-1]
    for action_number, action in enumerate(actions, start=1):
        for task_index in action.expected_task_indexes or (None,):
            if omit_last_task and action == final_action and task_index == final_task:
                continue
            parent_job_id = str(7000 + action_number)
            scheduler_job_id = parent_job_id if task_index is None else f"{parent_job_id}_{task_index}"
            record_successful_action_task(
                phase_runspec_path=fixture.runspec_path,
                action_id=action.action_id,
                command_digest=hashlib.sha256(action.action_id.encode()).hexdigest(),
                scheduler_job_id=scheduler_job_id,
                task_index=task_index,
                completed_at=NOW,
            )
    return fixture


def _publish(
    fixture: RuntimeFixture,
    *,
    publication_store: FinalizationPublicationStore | None = None,
    snapshot_io: LocalScientificSnapshotIO | None = None,
) -> object:
    return publish_action09_finalization_bundle(
        phase_runspec_path=fixture.runspec_path,
        execution_projection_path=fixture.projection_path,
        acceptance_policy_path=fixture.policy_path,
        command_digest="9" * 64,
        workers=2,
        assembled_at=NOW,
        publication_store=publication_store,
        **({"snapshot_io": snapshot_io} if snapshot_io is not None else {}),
    )


def _fixture(
    tmp_path: Path,
    *,
    batch_members: tuple[tuple[str, bytes], ...] = (("b.cif.zst", b"beta"), ("a.json.zst", b"alpha")),
) -> RuntimeFixture:
    tmp_path.mkdir(parents=True, exist_ok=True)
    authority_root = _materialized_authority(tmp_path)
    authority = validate_postprocessing_authority(authority_root, "phase-run-0123456789abcdef0123456789abcdef")
    runspec = authority.runspec
    attempt_root = authority.authority_path / "attempts" / runspec.attempt_id
    output_root = Path(runspec.payload.attempt_paths.output_dir)
    evidence_root = Path(runspec.payload.attempt_paths.evidence_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    batch_tar = output_root / "local_tars/shard_1/batch_0.tar"
    metadata_tar = output_root / "local_tars/metadata/shard_1_metadata.tar"
    _write_tar(batch_tar, batch_members)
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


def _acceptance(fixture: RuntimeFixture) -> None:
    authority = fixture.authority
    baseline = next(
        item.locator for item in authority.runspec.payload.physical_inputs if item.name == "baseline-output"
    )
    parity_path = "acceptance/tar_payload_parity/tar_payload_parity_report.json"
    semantic_path = "acceptance/semantic_acceptance/semantic_acceptance_summary.json"
    verify_path = "acceptance/verify_evidence/acceptance_evidence_report.json"
    _write_json(
        fixture.evidence_root / parity_path,
        {
            "baseline_dir": baseline,
            "candidate_dir": str(fixture.output_root),
            "relative_dir": "local_tars",
            "candidate_tar_count": 1,
            "ok": True,
            "inventory_errors": [],
            "baseline_only_tars": [],
            "candidate_only_tars": [],
            "payload_mismatch_count": 0,
            "error_count": 0,
            "files": [{"relative_path": "shard_1/batch_0.tar"}],
        },
    )
    _write_json(
        fixture.evidence_root / semantic_path,
        {
            "baseline_dir": baseline,
            "candidate_dir": str(fixture.output_root),
            "ok": False,
            "errors": ["known-task853-residual"],
        },
    )
    _write_json(
        fixture.evidence_root / verify_path,
        {
            "schema_version": 1,
            "ok": False,
            "parity_report_path": parity_path,
            "semantic_report_path": semantic_path,
            "issues": [{"check": "semantic-acceptance", "message": "ok is not true", "report_path": semantic_path}],
        },
    )
    for step, raw_exit in zip(
        ("acceptance-tar-payload-parity", "acceptance-semantic", "acceptance-verify-evidence"),
        (0, 1, 1),
        strict=True,
    ):
        stdout = fixture.evidence_root / f"phase-acceptance/raw/{step}.stdout"
        stderr = fixture.evidence_root / f"phase-acceptance/raw/{step}.stderr"
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text(f"{step} stdout\n")
        stderr.write_text(f"{step} stderr\n")
        action = next(item for item in authority.runspec.payload.actions if item.step_name == step)
        capture_acceptance(
            policy_path=fixture.policy_path,
            expected_policy_sha256=authority.runspec.payload.acceptance_policy.sha256,
            evidence_root=fixture.evidence_root,
            phase_run_id=authority.phase_run_id,
            attempt_id=authority.attempt_id,
            action_id=action.action_id,
            step_name=step,
            raw_exit_code=raw_exit,
            raw_stdout_path=stdout,
            raw_stderr_path=stderr,
            output_path=fixture.evidence_root / f"phase-acceptance/{step}-capture.json",
            completed_at=NOW,
        )
    adjudication = adjudicate_acceptance(
        policy_path=fixture.policy_path,
        expected_policy_sha256=authority.runspec.payload.acceptance_policy.sha256,
        evidence_root=fixture.evidence_root,
        phase_run_id=authority.phase_run_id,
        attempt_id=authority.attempt_id,
        output_path=fixture.evidence_root / "phase-acceptance/adjudication.json",
        adjudicated_at=NOW,
        phase_runspec_path=fixture.runspec_path,
        expected_phase_runspec_digest=authority.runspec.digest,
    )
    assert adjudication.result == "passed"


def _write_tar(path: Path, members: tuple[tuple[str, bytes], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
