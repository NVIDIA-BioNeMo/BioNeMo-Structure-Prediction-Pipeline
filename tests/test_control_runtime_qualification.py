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

"""Tests for Runtime Qualification records and gates."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from bspp.orchestration.contract.runtime_qualification import (
    RuntimeQualificationCheck,
    RuntimeQualificationRecord,
    RuntimeQualificationSnapshot,
)
from bspp.orchestration.contract.source_package import build_source_package
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.execution_bootstrap import PIXI_PYTHON_PATH, ImageIdentity
from bspp.orchestration.control.postprocessing_runtime_qualification import (
    replay_postprocessing_runtime,
    resolve_current_postprocessing_runtime,
)
from bspp.orchestration.control.profiles import ResolvedClusterProfile, resolve_cluster_profile
from bspp.orchestration.control.runtime_qualification import (
    _MAX_QUALIFICATION_RESULT_BYTES,
    _REMOTE_ATTEMPT_CLEANUP,
    _REMOTE_IDENTITY_PRODUCER,
    _REMOTE_RESULT_PROBE,
    _TOOLKIT_PACKAGE_CONTAINER,
    BAKED_TOOLKIT_COMMIT,
    _qualification_tuple,
    _read_bounded_stable_json,
    _remote_image_identity,
    _selected_source_kind,
    _toolkit_revision,
    _tuple_id,
    _valid_selected_source_identity,
    check_runtime_qualification,
    qualify_runtime,
    render_runtime_qualification_smoke_script,
)
from bspp.orchestration.control.source_package_staging import build_and_stage_governed_source_package
from bspp.orchestration.control.transport import CommandResult
from tests.runtime_ipsae_fixtures import (
    selected_source_identity,
    write_publication_compatibility_artifact,
    write_runtime_ipsae_artifacts,
)
from tests.support.transport_argv import maybe_unwrap_remote_command
from tests.test_control_source_package_staging import FakeStageRunner
from tests.test_control_transport import RecordingRunner


def _argv_parts(argv: tuple[str, ...]) -> list[str]:
    """Argv parts with any base64-wrapped ssh login-shell payload unwrapped."""
    return [maybe_unwrap_remote_command(part) for part in argv]


class FakeSubmitRunner:
    def __init__(self, job_ids: list[str]) -> None:
        self.job_ids = job_ids
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        return CommandResult(argv=argv, returncode=0, stdout=f"{self.job_ids.pop(0)}\n", stderr="")


class LostSubmissionResponseRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        raise TimeoutError("scheduler response lost after acceptance")


def test_runtime_qualification_check_enforces_current_snapshot_equivalence(tmp_path: Path) -> None:
    document = b"{}\n"
    snapshot = RuntimeQualificationSnapshot(
        document=document,
        sha256=hashlib.sha256(document).hexdigest(),
        size_bytes=len(document),
    )

    RuntimeQualificationCheck(False, "missing", None, tmp_path / "missing.json")
    with pytest.raises(ValueError, match="current status and snapshot presence differ"):
        RuntimeQualificationCheck(True, "current", "a" * 64, tmp_path / "current.json")
    with pytest.raises(ValueError, match="current status and snapshot presence differ"):
        RuntimeQualificationCheck(False, "invalid-record", "a" * 64, tmp_path / "invalid.json", snapshot)


class CompletedTokenReconciliationRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "squeue":
            return CommandResult(argv, 0, "", "")
        if argv[0] == "sacct":
            name = next(value.removeprefix("--name=") for value in argv if value.startswith("--name="))
            return CommandResult(argv, 0, f"4242|{name}|tester|COMPLETED\n", "")
        raise AssertionError(f"restart must not resubmit: {argv}")


class CompletedJobRunner:
    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        if argv[0] == "squeue":
            return CommandResult(argv, 0, json.dumps({"jobs": []}), "")
        if argv[0] == "sacct":
            return CommandResult(
                argv,
                0,
                json.dumps({"jobs": [{"job_id_raw": "8801", "state": "COMPLETED", "exit_code": "0:0"}]}),
                "",
            )
        raise AssertionError(argv)


def record_runtime_qualification_smoke_success(
    *,
    profile_name: str,
    config_path: Path,
    source_repo: Path,
    now: datetime | None = None,
) -> RuntimeQualificationRecord:
    """Create an authentic promoted record through the production lifecycle."""
    qualified_at = now or datetime.now(UTC)
    record = qualify_runtime(
        profile_name=profile_name,
        config_path=config_path,
        source_repo=source_repo,
        now=qualified_at,
        runner=FakeSubmitRunner(["8801"]),
    )
    payload = json.loads(record.path.read_text())
    qualification_tuple = payload["tuple"]
    result_path = Path(payload["smoke_job"]["result_path"])
    result_path.write_text(
        json.dumps(
            {
                "tuple_id": record.tuple_id,
                "job_id": "8801",
                "status": "succeeded",
                "attempt_token": payload["smoke_job"]["attempt_token"],
                "python": "3.12",
                "gpu": "synthetic gpu",
                "bootstrap_sha256": "7" * 64,
                "source_package_identity": qualification_tuple["source_package_identity"],
                "toolkit_package_identity": qualification_tuple.get("toolkit_package_identity"),
                "image_identity": qualification_tuple["image_identity"],
                "selected_source_identity": qualification_tuple["selected_source"],
                "runtime_ipsae": write_runtime_ipsae_artifacts(
                    result_path,
                    source_revision=_toolkit_revision(qualification_tuple),
                ),
                "publication_compatibility": write_publication_compatibility_artifact(result_path),
            }
        )
    )
    check = check_runtime_qualification(
        profile_name=profile_name,
        config_path=config_path,
        source_repo=source_repo,
        now=qualified_at,
        runner=CompletedJobRunner(),
    )
    assert check.current
    return RuntimeQualificationRecord(tuple_id=record.tuple_id, path=record.path, status="qualified")


def test_baked_record_with_divergent_smoke_revision_reaches_current(tmp_path: Path) -> None:
    """Option A regression: a baked record whose smoke reports the image's actual
    baked commit (different from the public placeholder) still promotes to current."""
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    # Rewrite the profile to baked mode (no afdb_toolkit_repo).
    profiles = yaml.safe_load(config_path.read_text())
    del profiles["clusters"]["example-cluster"]["afdb_toolkit_repo"]
    config_path.write_text(yaml.safe_dump(profiles, sort_keys=False))
    qualified_at = _instant("2026-06-06T12:00:00Z")
    record = qualify_runtime(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=qualified_at,
        runner=FakeSubmitRunner(["8801"]),
    )
    payload = json.loads(record.path.read_text())
    qualification_tuple = payload["tuple"]
    assert qualification_tuple["selected_source"]["source_kind"] == "baked"
    result_path = Path(payload["smoke_job"]["result_path"])
    image_baked_revision = "b" * 40  # the image's actual baked commit, != placeholder
    assert image_baked_revision != qualification_tuple["selected_source"]["revision"]
    result_path.write_text(
        json.dumps(
            {
                "tuple_id": record.tuple_id,
                "job_id": "8801",
                "status": "succeeded",
                "attempt_token": payload["smoke_job"]["attempt_token"],
                "python": "3.12",
                "gpu": "synthetic gpu",
                "bootstrap_sha256": "7" * 64,
                "source_package_identity": qualification_tuple["source_package_identity"],
                "toolkit_package_identity": qualification_tuple.get("toolkit_package_identity"),
                "image_identity": qualification_tuple["image_identity"],
                "selected_source_identity": {
                    **qualification_tuple["selected_source"],
                    "revision": image_baked_revision,
                },
                "runtime_ipsae": write_runtime_ipsae_artifacts(result_path, source_revision=image_baked_revision),
                "publication_compatibility": write_publication_compatibility_artifact(result_path),
            }
        )
    )
    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=qualified_at,
        runner=CompletedJobRunner(),
    )
    assert check.current


def test_runtime_qualify_submits_smoke_job_and_records_pending_tuple(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    (tmp_path / "toolkit" / "ordinary-untracked.tmp").write_text("irrelevant\n")
    runner = FakeSubmitRunner(["8801"])

    record = qualify_runtime(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
        runner=runner,
    )

    record_path = tmp_path / "qualifications" / "example-cluster" / f"{record.tuple_id}.json"
    assert record.path == record_path
    assert record.job_id == "8801"
    attempt_token = json.loads(record_path.read_text())["smoke_job"]["attempt_token"]
    attempt_root = tmp_path / "qualifications" / "example-cluster" / "attempts" / record.tuple_id / attempt_token
    assert record.script_path == attempt_root / "smoke.sbatch"
    assert runner.calls == [("sbatch", "--parsable", str(record.script_path))]
    assert record_path.exists()
    payload = json.loads(record_path.read_text())
    assert payload["schema_version"] == 1
    assert payload["profile"] == "example-cluster"
    assert payload["status"] == "submitted"
    assert payload["evidence_status"] == "pending"
    assert payload["submitted_at"] == "2026-06-06T12:00:00Z"
    assert payload["qualified_at"] is None
    assert payload["expires_at"] == "2026-06-13T12:00:00Z"
    assert payload["tuple_id"] == record.tuple_id
    assert len(payload["tuple_id"]) == 64
    assert payload["tuple"]["cluster_profile"] == "example-cluster"
    assert payload["tuple"]["scheduling_class"] == "gpu_worker"
    assert payload["tuple"]["execution_runtime_image"] == str(tmp_path / "images" / "bspp.sqsh")
    assert payload["tuple"]["runtime_facts"] == {"gpu_worker_gres": "gpu:1"}
    assert payload["tuple"]["source_bundle_id"].startswith("bspp-orchestration-")
    assert payload["tuple"]["source_bundle_path"].startswith(str(tmp_path / "bundles"))
    assert payload["tuple"]["toolkit_source"] == str(tmp_path / "toolkit")
    assert payload["smoke_job"]["job_id"] == "8801"
    assert payload["smoke_job"]["script_path"] == str(record.script_path)
    script = record.script_path.read_text()
    assert f"--container-image={tmp_path}/images/bspp.sqsh" in script
    lines = script.splitlines()
    srun_index = next(index for index, line in enumerate(lines) if line.startswith("srun "))
    assert "/usr/bin/python3 -I -S -c" in lines[srun_index - 1]
    assert all(command not in script for command in ("sha256sum", "wc -c", "test ! -L"))
    assert "env -i" in script
    assert "/usr/bin/python3 -I -S" in script
    assert "entrypoint.sh" not in script
    assert "--identity-record /run/bspp/runtime-qualification.json" in script
    assert f"{payload['tuple']['source_bundle_path']}:/run/bspp/source-package.tar:ro" in script
    assert f"{payload['tuple']['toolkit_package_path']}:/run/bspp/toolkit-package.tar:ro" in script
    assert f"{payload['smoke_job']['input_path']}:/run/bspp/runtime-qualification.json:ro" in script
    assert f"{tmp_path}/toolkit:/workspace/AFDB-Integration-Kit" not in script
    assert "#SBATCH --gres=gpu:1" in script
    assert 'tar -xaf "$BSPP_SOURCE_BUNDLE"' not in script


def test_runtime_qualification_smoke_writes_the_persisted_result_filename(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record = qualify_runtime(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        runner=FakeSubmitRunner(["8801"]),
    )
    payload = json.loads(record.path.read_text())
    result_path = Path(payload["smoke_job"]["remote_result_path"])

    assert f"/run/bspp/result/{result_path.name}" in record.script_path.read_text()


def test_smoke_result_atomically_promotes_exact_submitted_attempt(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record = qualify_runtime(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
        runner=FakeSubmitRunner(["8801"]),
    )
    payload = json.loads(record.path.read_text())
    tuple_payload = payload["tuple"]
    result_path = Path(payload["smoke_job"]["result_path"])
    result_path.write_text(
        json.dumps(
            {
                "tuple_id": record.tuple_id,
                "job_id": "8801",
                "status": "succeeded",
                "attempt_token": payload["smoke_job"]["attempt_token"],
                "python": "3.12",
                "gpu": "synthetic gpu",
                "bootstrap_sha256": "7" * 64,
                "source_package_identity": tuple_payload["source_package_identity"],
                "toolkit_package_identity": tuple_payload["toolkit_package_identity"],
                "image_identity": tuple_payload["image_identity"],
                "selected_source_identity": tuple_payload["selected_source"],
                "runtime_ipsae": write_runtime_ipsae_artifacts(
                    result_path,
                    source_revision=tuple_payload["toolkit_package_identity"]["commit"],
                ),
                "publication_compatibility": write_publication_compatibility_artifact(result_path),
            }
        )
    )
    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:01:00Z"),
        runner=CompletedJobRunner(),
    )
    assert check.current
    promoted = json.loads(record.path.read_text())
    assert promoted["status"] == "qualified"
    assert promoted["smoke_evidence"]["job_id"] == "8801"


def test_runtime_qualify_restart_reuses_submitted_attempt_without_duplicate_submit(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)

    class ActiveRunner(FakeSubmitRunner):
        def __call__(self, argv: tuple[str, ...]) -> CommandResult:
            if argv[0] == "squeue":
                self.calls.append(argv)
                return CommandResult(argv, 0, json.dumps({"jobs": [{"job_id": 8801, "job_state": ["RUNNING"]}]}), "")
            if argv[0] == "sacct":
                self.calls.append(argv)
                return CommandResult(argv, 0, json.dumps({"jobs": []}), "")
            return super().__call__(argv)

    runner = ActiveRunner(["8801"])
    first = qualify_runtime(
        profile_name="example-cluster", config_path=config_path, source_repo=source_repo, runner=runner
    )
    second = qualify_runtime(
        profile_name="example-cluster", config_path=config_path, source_repo=source_repo, runner=runner
    )
    assert second.path == first.path
    assert second.job_id == "8801"
    assert len([call for call in runner.calls if call[:2] == ("sbatch", "--parsable")]) == 1


def test_submitted_attempt_missing_from_scheduler_accounting_remains_ambiguous_without_resubmit(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    original = qualify_runtime(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        runner=FakeSubmitRunner(["8801"]),
    )
    original_payload = original.path.read_bytes()

    reconciliation = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
        ]
    )
    with pytest.raises(ValueError, match=r"ambiguous.*accounting"):
        qualify_runtime(
            profile_name="example-cluster",
            config_path=config_path,
            source_repo=source_repo,
            runner=reconciliation,
        )

    assert [call[0] for call in reconciliation.calls] == ["squeue", "sacct"]
    assert not any(call[0] == "sbatch" for call in reconciliation.calls)
    assert original.path.read_bytes() == original_payload


def test_runtime_qualification_module_does_not_export_synthetic_success_recorder() -> None:
    from bspp.orchestration.control import runtime_qualification

    assert not hasattr(runtime_qualification, "record_runtime_qualification_smoke_success")


def test_runtime_qualify_ambiguous_sbatch_leaves_tokenized_submitting_record(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    runner = FakeSubmitRunner(["not-a-job-id"])
    with pytest.raises(ValueError, match="could not parse"):
        qualify_runtime(profile_name="example-cluster", config_path=config_path, source_repo=source_repo, runner=runner)
    records = tuple((tmp_path / "qualifications" / "example-cluster").glob("*.json"))
    assert len(records) == 1
    payload = json.loads(records[0].read_text())
    assert payload["status"] == "submitting"
    token = payload["smoke_job"]["attempt_token"]
    assert len(token) == 32
    assert token in Path(payload["smoke_job"]["script_path"]).read_text()
    reconcile = FakeSubmitRunner(["4242\n4243"])
    with pytest.raises(ValueError, match="ambiguous"):
        qualify_runtime(
            profile_name="example-cluster",
            config_path=config_path,
            source_repo=source_repo,
            runner=reconcile,
        )
    assert len(reconcile.calls) == 1
    assert reconcile.calls[0][0] == "squeue"


def test_accepted_submission_completed_before_restart_reconciles_accounting_without_resubmit(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    first = FakeSubmitRunner(["accepted-without-id"])
    with pytest.raises(ValueError, match="could not parse"):
        qualify_runtime(profile_name="example-cluster", config_path=config_path, source_repo=source_repo, runner=first)
    record_path = next((tmp_path / "qualifications" / "example-cluster").glob("*.json"))
    token = json.loads(record_path.read_text())["smoke_job"]["attempt_token"]
    accounting = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout="", stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=f"4242|bspp_rq_example-cluster_{token}|tester|COMPLETED\n",
                stderr="",
            ),
        ]
    )
    reconciled = qualify_runtime(
        profile_name="example-cluster", config_path=config_path, source_repo=source_repo, runner=accounting
    )
    assert reconciled.job_id == "4242"
    assert json.loads(record_path.read_text())["status"] == "submitted"
    assert not any(call[0] == "sbatch" for call in accounting.calls)


def test_submitting_attempt_with_accounting_lag_remains_ambiguous_without_resubmit(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    with pytest.raises(ValueError, match="could not parse"):
        qualify_runtime(
            profile_name="example-cluster",
            config_path=config_path,
            source_repo=source_repo,
            runner=FakeSubmitRunner(["accepted-without-id"]),
        )

    reconciliation = RecordingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout="", stderr=""),
            CommandResult(argv=(), returncode=0, stdout="", stderr=""),
        ]
    )
    with pytest.raises(ValueError, match=r"ambiguous.*submission"):
        qualify_runtime(
            profile_name="example-cluster",
            config_path=config_path,
            source_repo=source_repo,
            runner=reconciliation,
        )

    assert [call[0] for call in reconciliation.calls] == ["squeue", "sacct"]
    record_path = next((tmp_path / "qualifications" / "example-cluster").glob("*.json"))
    assert json.loads(record_path.read_text())["status"] == "submitting"


@pytest.mark.parametrize(
    "forged_field",
    [
        "record_path",
        "script_path",
        "input_path",
        "result_path",
        "remote_script_path",
        "remote_input_path",
        "remote_result_path",
    ],
)
def test_check_rejects_forged_submitted_attempt_paths_before_result_access(tmp_path: Path, forged_field: str) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record = qualify_runtime(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        runner=FakeSubmitRunner(["8801"]),
    )
    outside = tmp_path / "outside-result.json"
    outside.write_text('{"marker":"preserve"}\n')
    payload = json.loads(record.path.read_text())
    payload["smoke_job"][forged_field] = str(outside)
    record.path.write_text(json.dumps(payload))

    class NoSchedulerRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def __call__(self, argv: tuple[str, ...]) -> CommandResult:
            self.calls.append(argv)
            raise AssertionError(f"forged attempt must not reach transport: {argv}")

    runner = NoSchedulerRunner()
    check = check_runtime_qualification(
        profile_name="example-cluster", config_path=config_path, source_repo=source_repo, runner=runner
    )

    assert not check.current
    assert check.reason == "invalid-record"
    assert runner.calls == []
    assert outside.read_text() == '{"marker":"preserve"}\n'


def test_remote_result_probe_rejects_symlink_and_oversize_before_fetch(tmp_path: Path) -> None:
    regular = tmp_path / "result.json"
    regular.write_bytes(b"x" * (_MAX_QUALIFICATION_RESULT_BYTES + 1))
    command = (sys.executable, "-I", "-S", "-c", _REMOTE_RESULT_PROBE)
    oversize = subprocess.run(
        (*command, str(regular), str(_MAX_QUALIFICATION_RESULT_BYTES)), capture_output=True, text=True
    )
    assert oversize.returncode != 0
    assert "oversize result" in oversize.stderr
    regular.write_text("{}\n")
    link = tmp_path / "result-link.json"
    link.symlink_to(regular)
    symlink = subprocess.run(
        (*command, str(link), str(_MAX_QUALIFICATION_RESULT_BYTES)), capture_output=True, text=True
    )
    assert symlink.returncode != 0
    assert "unsafe result" in symlink.stderr


def test_remote_identity_producer_uses_portable_streaming_hash() -> None:
    assert "hashlib.file_digest" not in _REMOTE_IDENTITY_PRODUCER


def test_bounded_stable_result_reader_rejects_symlink_and_oversize(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text('{"ok":true}\n')
    assert _read_bounded_stable_json(result) == {"ok": True}
    link = tmp_path / "alias.json"
    link.symlink_to(result)
    with pytest.raises(ValueError, match="unsafe"):
        _read_bounded_stable_json(link)
    result.write_bytes(b"x" * (_MAX_QUALIFICATION_RESULT_BYTES + 1))
    with pytest.raises(ValueError, match="oversize"):
        _read_bounded_stable_json(result)


def test_snapshot_path_verifier_is_bounded_and_rejects_fifo_symlink_and_oversize(tmp_path: Path) -> None:
    from bspp.orchestration.control import runtime_qualification as runtime_module

    verifier = runtime_module.verify_runtime_qualification_snapshot_path
    document = b'{"status":"qualified"}\n'
    expected_sha256 = hashlib.sha256(document).hexdigest()
    record = tmp_path / "record.json"
    record.write_bytes(document)
    verifier(record, expected_sha256=expected_sha256, expected_size_bytes=len(document))

    link = tmp_path / "record-link.json"
    link.symlink_to(record)
    with pytest.raises(ValueError, match="regular file"):
        verifier(link, expected_sha256=expected_sha256, expected_size_bytes=len(document))

    record.write_bytes(document + b"forged")
    with pytest.raises(ValueError, match="size"):
        verifier(record, expected_sha256=expected_sha256, expected_size_bytes=len(document))

    fifo = tmp_path / "record.fifo"
    fifo.parent.mkdir(parents=True, exist_ok=True)
    fifo.unlink(missing_ok=True)
    os.mkfifo(fifo)
    command = (
        sys.executable,
        "-c",
        (
            "import sys; from pathlib import Path; "
            "from bspp.orchestration.control.runtime_qualification import "
            "verify_runtime_qualification_snapshot_path as verify; "
            "verify(Path(sys.argv[1]), expected_sha256=sys.argv[2], expected_size_bytes=int(sys.argv[3]))"
        ),
        str(fifo),
        hashlib.sha256(b"").hexdigest(),
        "0",
    )
    result = subprocess.run(command, capture_output=True, text=True, timeout=2)
    assert result.returncode != 0
    assert "regular file" in result.stderr


def test_remote_attempt_cleanup_rejects_intermediate_symlink_and_preserves_external_directory(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "preserve"
    marker.write_text("safe\n")
    configured = tmp_path / "configured"
    configured.symlink_to(external, target_is_directory=True)
    tuple_id = "a" * 64
    token = "b" * 32
    target = configured / "example-cluster" / "attempts" / tuple_id / token
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-c",
            _REMOTE_ATTEMPT_CLEANUP,
            str(target),
            str(configured),
            "example-cluster",
            tuple_id,
            token,
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert marker.read_text() == "safe\n"


def test_ssh_runtime_qualification_requires_staged_identity_without_local_remote_path_write(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    payload = yaml.safe_load(config_path.read_text())
    remote_root = tmp_path / "must-not-be-created" / "remote-bundles"
    cluster = payload["clusters"]["example-cluster"]
    cluster["transport"] = "ssh"
    cluster["ssh_target"] = "login.invalid"
    cluster["source_bundle_root"] = str(remote_root)
    cluster["runtime_qualification_control_root"] = str(tmp_path / "control-qualifications")
    config_path.write_text(yaml.safe_dump(payload))
    runner = FakeSubmitRunner(["8801"])
    with pytest.raises(ValueError, match="SSH Runtime Qualification requires --source-package-identity"):
        qualify_runtime(profile_name="example-cluster", config_path=config_path, source_repo=source_repo, runner=runner)
    assert runner.calls == []
    assert not remote_root.exists()


@pytest.mark.parametrize("field", ["commit", "tree"])
def test_runtime_qualification_rejects_source_package_identity_stale_against_checkout(
    tmp_path: Path, field: str
) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    profile = replace(
        resolve_cluster_profile("example-cluster", config_path=config_path),
        afdb_toolkit_repo=None,
        governed_package_root=str(tmp_path / "governed"),
    )
    commit = _git(source_repo, "rev-parse", "HEAD")
    tree = _git(source_repo, "rev-parse", "HEAD^{tree}")
    candidate = build_source_package(
        source_repo,
        tmp_path / "candidate.tar",
        commit=commit,
        tree=tree,
        tracked_git=True,
    )
    target = tmp_path / "governed" / "orchestration" / commit / f"{candidate.package_sha256}.tar"
    target.parent.mkdir(parents=True)
    candidate.package_path.replace(target)
    identity = replace(candidate, package_path=target, **{field: "0" * 40})

    with pytest.raises(ValueError, match=f"source package {field} does not match clean source checkout"):
        _qualification_tuple(profile, source_repo=source_repo, source_package_identity=identity)


def test_runtime_qualification_rejects_source_package_outside_current_commit_digest_path(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    profile = replace(
        resolve_cluster_profile("example-cluster", config_path=config_path),
        afdb_toolkit_repo=None,
        governed_package_root=str(tmp_path / "governed"),
    )
    commit = _git(source_repo, "rev-parse", "HEAD")
    tree = _git(source_repo, "rev-parse", "HEAD^{tree}")
    candidate = build_source_package(
        source_repo,
        tmp_path / "candidate.tar",
        commit=commit,
        tree=tree,
        tracked_git=True,
    )
    stale_target = tmp_path / "governed" / "orchestration" / ("0" * 40) / f"{candidate.package_sha256}.tar"
    stale_target.parent.mkdir(parents=True)
    candidate.package_path.replace(stale_target)
    identity = replace(candidate, package_path=stale_target)

    with pytest.raises(ValueError, match="current checkout digest-addressed path"):
        _qualification_tuple(profile, source_repo=source_repo, source_package_identity=identity)


def test_runtime_qualification_rejects_forged_manifest_for_current_commit_and_tree(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    profile = replace(
        resolve_cluster_profile("example-cluster", config_path=config_path),
        afdb_toolkit_repo=None,
        governed_package_root=str(tmp_path / "governed"),
    )
    commit = _git(source_repo, "rev-parse", "HEAD")
    tree = _git(source_repo, "rev-parse", "HEAD^{tree}")
    forged_root = tmp_path / "forged"
    forged_payload = forged_root / "packages" / "orchestration-runtime" / "src" / "forged.py"
    forged_payload.parent.mkdir(parents=True)
    forged_payload.write_text("FORGED = True\n")
    candidate = build_source_package(
        forged_root,
        tmp_path / "candidate.tar",
        commit=commit,
        tree=tree,
        package_role="orchestration",
    )
    target = tmp_path / "governed" / "orchestration" / commit / f"{candidate.package_sha256}.tar"
    target.parent.mkdir(parents=True)
    candidate.package_path.replace(target)
    identity = replace(candidate, package_path=target)

    with pytest.raises(ValueError, match="manifest does not match clean source checkout"):
        _qualification_tuple(profile, source_repo=source_repo, source_package_identity=identity)


def test_ssh_runtime_qualification_consumes_exact_staged_identity(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    payload = yaml.safe_load(config_path.read_text())
    remote_root = tmp_path / "not-local" / "bundles"
    cluster = payload["clusters"]["example-cluster"]
    cluster["transport"] = "ssh"
    cluster["ssh_target"] = "login.invalid"
    cluster["source_bundle_root"] = str(remote_root)
    cluster["runtime_qualification_control_root"] = str(tmp_path / "control-qualifications")
    config_path.write_text(yaml.safe_dump(payload))
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    commit = _git(source_repo, "rev-parse", "HEAD")
    identity = build_and_stage_governed_source_package(
        source_repo,
        build_dir=tmp_path / "build",
        target_path=remote_root / f"bspp-orchestration-{commit}.tar",
        profile=profile,
        runner=FakeStageRunner("unused"),
        verify_remote_digest=False,
    )
    identity_path = tmp_path / "build" / "source-package-identity.json"

    class VerifyThenSubmitRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def __call__(self, argv: tuple[str, ...]) -> CommandResult:
            self.calls.append(argv)
            if any("squeue --json" in part for part in _argv_parts(argv)):
                return CommandResult(argv, 0, json.dumps({"jobs": []}), "")
            if any("sacct --json" in part for part in _argv_parts(argv)):
                return CommandResult(
                    argv,
                    0,
                    json.dumps({"jobs": [{"job_id_raw": "8801", "state": "COMPLETED", "exit_code": "0:0"}]}),
                    "",
                )
            if any("source package policy" in part for part in _argv_parts(argv)):
                return CommandResult(argv, 0, "OK\n", "")
            if any("python3" in part and str(tmp_path / "toolkit") in part for part in _argv_parts(argv)):
                remote_payload = {
                    "image_identity": {
                        "format_version": 1,
                        "policy": "digest-checked",
                        "path": str((tmp_path / "images" / "bspp.sqsh").absolute()),
                        "size_bytes": len(b"synthetic-runtime-image"),
                        "sha256": __import__("hashlib").sha256(b"synthetic-runtime-image").hexdigest(),
                    },
                    "toolkit_package_identity": {
                        "format_version": 1,
                        "format": "bspp-tar-v1",
                        "verifier": "safe-tar-v1",
                        "package_path": str(
                            tmp_path / "qualifications/packages/toolkit/" / ("b" * 40) / f"{'c' * 64}.tar"
                        ),
                        "package_size_bytes": 10240,
                        "package_sha256": "c" * 64,
                        "manifest_sha256": "d" * 64,
                        "commit": "b" * 40,
                        "tree": "e" * 40,
                        "package_role": "toolkit",
                        "policy_version": 1,
                    },
                }
                return CommandResult(argv, 0, json.dumps(remote_payload), "")
            if any("package identity mismatch" in part for part in _argv_parts(argv)):
                return CommandResult(
                    argv,
                    0,
                    json.dumps({"size_bytes": identity.package_size_bytes, "sha256": identity.package_sha256}),
                    "",
                )
            return CommandResult(argv, 0, "8801\n", "")

    runner = VerifyThenSubmitRunner()
    record = qualify_runtime(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        source_package_identity_path=identity_path,
        runner=runner,
    )
    payload = json.loads(record.path.read_text())
    assert payload["tuple"]["source_package_identity"] == identity.to_mapping()
    assert record.path.is_relative_to(tmp_path / "control-qualifications")
    remote_script = payload["smoke_job"]["remote_script_path"]
    remote_result = Path(payload["smoke_job"]["remote_result_path"])
    script = record.script_path.read_text()
    assert remote_script.startswith(str(tmp_path / "qualifications"))
    assert f"{remote_result.parent}:/run/bspp/result" in script
    assert "/run/bspp/result/result.json" in script
    assert "/run/bspp/result/smoke.json" not in script
    sbatch_calls = [call for call in runner.calls if any("sbatch" in part for part in call)]
    assert sbatch_calls and remote_script in sbatch_calls[-1][-1]
    assert str(payload["smoke_job"]["script_path"]) not in sbatch_calls[-1][-1]
    assert not remote_root.exists()


def test_ssh_runtime_qualification_uses_remote_producer_for_nonlocal_image_and_toolkit(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    cluster = config["clusters"]["example-cluster"]
    cluster["transport"] = "ssh"
    cluster["ssh_target"] = "login.invalid"
    cluster["image"] = "/remote/images/runtime.sqsh"
    cluster["afdb_toolkit_repo"] = "/remote/repos/toolkit"
    cluster["governed_package_root"] = "/remote/governed/packages"
    cluster["runtime_qualification_control_root"] = str(tmp_path / "control-qualifications")
    config_path.write_text(yaml.safe_dump(config))
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    commit = _git(source_repo, "rev-parse", "HEAD")
    source_identity = build_and_stage_governed_source_package(
        source_repo,
        build_dir=tmp_path / "build",
        target_path=Path(cluster["source_bundle_root"]) / f"bspp-orchestration-{commit}.tar",
        profile=profile,
        runner=FakeStageRunner("unused"),
        verify_remote_digest=False,
    )

    class RemoteOnlyRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def __call__(self, argv: tuple[str, ...]) -> CommandResult:
            self.calls.append(argv)
            if any("squeue --json" in part for part in _argv_parts(argv)):
                return CommandResult(argv, 0, json.dumps({"jobs": []}), "")
            if any("sacct --json" in part for part in _argv_parts(argv)):
                return CommandResult(
                    argv,
                    0,
                    json.dumps({"jobs": [{"job_id_raw": "8801", "state": "COMPLETED", "exit_code": "0:0"}]}),
                    "",
                )
            if any("source package policy" in part for part in _argv_parts(argv)):
                return CommandResult(argv, 0, "OK\n", "")
            if any("/remote/repos/toolkit" in part and "python3" in part for part in _argv_parts(argv)):
                payload = {
                    "image_identity": {
                        "format_version": 1,
                        "policy": "digest-checked",
                        "path": "/remote/images/runtime.sqsh",
                        "size_bytes": 5,
                        "sha256": "a" * 64,
                    },
                    "toolkit_package_identity": {
                        "format_version": 1,
                        "format": "bspp-tar-v1",
                        "verifier": "safe-tar-v1",
                        "package_path": "/remote/governed/packages/toolkit/" + "b" * 40 + "/" + "c" * 64 + ".tar",
                        "package_size_bytes": 10240,
                        "package_sha256": "c" * 64,
                        "manifest_sha256": "d" * 64,
                        "commit": "b" * 40,
                        "tree": "e" * 40,
                        "package_role": "toolkit",
                        "policy_version": 1,
                    },
                }
                return CommandResult(argv, 0, json.dumps(payload), "")
            if any("package identity mismatch" in part for part in _argv_parts(argv)):
                return CommandResult(
                    argv,
                    0,
                    json.dumps(
                        {
                            "size_bytes": source_identity.package_size_bytes,
                            "sha256": source_identity.package_sha256,
                        }
                    ),
                    "",
                )
            return CommandResult(argv, 0, "8801\n", "")

    runner = RemoteOnlyRunner()
    record = qualify_runtime(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        source_package_identity_path=tmp_path / "build" / "source-package-identity.json",
        runner=runner,
    )
    record_payload = json.loads(record.path.read_text())
    payload = record_payload["tuple"]
    assert payload["image_identity"]["path"] == "/remote/images/runtime.sqsh"
    assert payload["toolkit_package_identity"]["package_path"].startswith("/remote/governed/packages/toolkit/")
    assert any(
        call[0:2] == ("ssh", "login.invalid") and any("/remote/repos/toolkit" in part for part in _argv_parts(call))
        for call in runner.calls
    )
    result_path = Path(record_payload["smoke_job"]["result_path"])
    result_path.write_text(
        json.dumps(
            {
                "tuple_id": record.tuple_id,
                "job_id": "8801",
                "status": "succeeded",
                "attempt_token": record_payload["smoke_job"]["attempt_token"],
                "python": "3.12",
                "gpu": "remote gpu",
                "bootstrap_sha256": "7" * 64,
                "source_package_identity": payload["source_package_identity"],
                "toolkit_package_identity": payload["toolkit_package_identity"],
                "image_identity": payload["image_identity"],
                "selected_source_identity": payload["selected_source"],
                "runtime_ipsae": write_runtime_ipsae_artifacts(
                    result_path,
                    source_revision=payload["toolkit_package_identity"]["commit"],
                ),
                "publication_compatibility": write_publication_compatibility_artifact(result_path),
            }
        )
    )
    check = check_runtime_qualification(
        profile_name="example-cluster", config_path=config_path, source_repo=source_repo, runner=runner
    )
    assert check.current


def test_remote_image_identity_runs_image_only_producer_over_ssh(tmp_path: Path) -> None:
    """Baked SSH profiles identify the runtime image through a remote producer, not local stat."""
    config_path = _write_profiles(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    cluster = config["clusters"]["example-cluster"]
    cluster["transport"] = "ssh"
    cluster["ssh_target"] = "login.invalid"
    cluster["image"] = "/remote/images/runtime.sqsh"
    cluster["runtime_qualification_control_root"] = str(tmp_path / "control-qualifications")
    config_path.write_text(yaml.safe_dump(config))
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)

    class ImageOnlyRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def __call__(self, argv: tuple[str, ...]) -> CommandResult:
            self.calls.append(argv)
            return CommandResult(
                argv,
                0,
                json.dumps(
                    {
                        "image_identity": {
                            "format_version": 1,
                            "policy": "digest-checked",
                            "path": "/remote/images/runtime.sqsh",
                            "size_bytes": 5,
                            "sha256": "a" * 64,
                        }
                    }
                ),
                "",
            )

    runner = ImageOnlyRunner()
    identity = _remote_image_identity(profile, runner=runner)
    assert identity.path == Path("/remote/images/runtime.sqsh")
    assert identity.sha256 == "a" * 64
    assert identity.policy == "digest-checked"
    assert runner.calls
    assert runner.calls[0][0:2] == ("ssh", "login.invalid")
    assert any("/remote/images/runtime.sqsh" in part for part in _argv_parts(runner.calls[0]))


def test_ssh_baked_tuple_routes_image_identity_to_remote_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Baked SSH qualification uses the remote image producer, never local identify_runtime_image."""
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    cluster = config["clusters"]["example-cluster"]
    cluster["transport"] = "ssh"
    cluster["ssh_target"] = "login.invalid"
    cluster["image"] = "/remote/images/runtime.sqsh"
    del cluster["afdb_toolkit_repo"]
    cluster["governed_package_root"] = "/remote/governed/packages"
    cluster["runtime_qualification_control_root"] = str(tmp_path / "control-qualifications")
    config_path.write_text(yaml.safe_dump(config))
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    commit = _git(source_repo, "rev-parse", "HEAD")
    source_identity = build_and_stage_governed_source_package(
        source_repo,
        build_dir=tmp_path / "build",
        target_path=Path(cluster["source_bundle_root"]) / f"bspp-orchestration-{commit}.tar",
        profile=profile,
        runner=FakeStageRunner("unused"),
        verify_remote_digest=False,
    )

    routed: list[str] = []
    fake_image_identity = ImageIdentity(1, "digest-checked", Path("/remote/images/runtime.sqsh"), 5, "a" * 64)

    def fake_remote_image_identity(profile: ResolvedClusterProfile, *, runner: object) -> ImageIdentity:
        routed.append("remote")
        return fake_image_identity

    monkeypatch.setattr(
        "bspp.orchestration.control.runtime_qualification._remote_image_identity",
        fake_remote_image_identity,
    )
    monkeypatch.setattr(
        "bspp.orchestration.control.runtime_qualification._verify_staged_source_package",
        lambda *_args, **_kwargs: None,
    )

    qualification_tuple = _qualification_tuple(
        profile,
        source_repo=source_repo,
        source_package_identity=source_identity,
        runner=FakeSubmitRunner(["8801"]),
    )
    assert routed == ["remote"]
    assert qualification_tuple["image_identity"]["path"] == "/remote/images/runtime.sqsh"
    assert qualification_tuple["selected_source"]["source_kind"] == "baked"
    assert "toolkit_package_identity" not in qualification_tuple


def test_remote_toolkit_producer_matches_local_canonical_package_bytes(tmp_path: Path) -> None:
    toolkit = _init_git_repo(tmp_path / "toolkit")
    (toolkit / "ordinary.tmp").write_text("must not influence package\n")
    image = tmp_path / "runtime.sqsh"
    image.write_bytes(b"image")
    commit = _git(toolkit, "rev-parse", "HEAD")
    tree = _git(toolkit, "rev-parse", "HEAD^{tree}")
    local = build_source_package(
        toolkit,
        tmp_path / "local.tar",
        commit=commit,
        tree=tree,
        tracked_git=True,
        governed_runtime_only=False,
        allow_untracked=True,
    )
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-c",
            _REMOTE_IDENTITY_PRODUCER,
            str(toolkit),
            str(image),
            str(tmp_path / "remote-packages"),
            "digest-checked",
            "",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    remote = json.loads(result.stdout)["toolkit_package_identity"]
    assert remote["package_sha256"] == local.package_sha256
    assert Path(remote["package_path"]).read_bytes() == local.package_path.read_bytes()


def test_remote_identity_producer_honors_trusted_cache_policy(tmp_path: Path) -> None:
    toolkit = _init_git_repo(tmp_path / "toolkit")
    image = tmp_path / "image.sqsh"
    cache = tmp_path / "cache.sqsh"
    image.write_bytes(b"same-image")
    cache.write_bytes(b"same-image")
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-c",
            _REMOTE_IDENTITY_PRODUCER,
            str(toolkit),
            str(image),
            str(tmp_path / "packages"),
            "trusted-cache",
            str(cache),
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    identity = json.loads(result.stdout)["image_identity"]
    assert identity["policy"] == "trusted-cache"
    assert identity["path"] == str(cache.resolve())


@pytest.mark.parametrize("change", ["modified", "staged-addition", "deletion"])
def test_remote_toolkit_producer_rejects_tracked_checkout_changes(tmp_path: Path, change: str) -> None:
    toolkit = _init_git_repo(tmp_path / "toolkit")
    image = tmp_path / "runtime.sqsh"
    image.write_bytes(b"image")
    tracked = toolkit / "README.md"
    if change == "modified":
        tracked.write_text("modified\n")
    elif change == "staged-addition":
        (toolkit / "added.py").write_text("added\n")
        _git(toolkit, "add", "added.py")
    else:
        tracked.unlink()
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-c",
            _REMOTE_IDENTITY_PRODUCER,
            str(toolkit),
            str(image),
            str(tmp_path / "remote-packages"),
            "digest-checked",
            "",
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "dirty tracked toolkit" in result.stderr


def test_runtime_qualification_smoke_success_record_is_current(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)

    record = record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )

    payload = json.loads(record.path.read_text())
    assert payload["status"] == "qualified"
    assert payload["evidence_status"] == "qualified"
    assert payload["qualified_at"] == "2026-06-06T12:00:00Z"
    assert payload["expires_at"] == "2026-06-13T12:00:00Z"
    assert payload["smoke_evidence"]["gpu"] == "synthetic gpu"
    expected_bytes = record.path.read_bytes()

    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )

    assert check.current is True
    assert check.reason == "current"
    assert check.snapshot is not None
    assert check.snapshot.document == expected_bytes
    assert check.snapshot.size_bytes == len(expected_bytes)
    assert check.snapshot.sha256 == hashlib.sha256(expected_bytes).hexdigest()


def test_bsppctl_runtime_qualify_prints_record_summary(tmp_path: Path, monkeypatch) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record_path = tmp_path / "qualifications" / "example-cluster" / "tuple.json"
    script_path = tmp_path / "qualifications" / "example-cluster" / "tuple.smoke.sbatch"

    def fake_qualify_runtime(*args: object, **kwargs: object) -> RuntimeQualificationRecord:
        del args, kwargs
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profile": "example-cluster",
                    "status": "submitted",
                    "evidence_status": "pending",
                    "tuple_id": "a" * 64,
                    "tuple": {},
                    "expires_at": "2026-06-13T12:00:00Z",
                    "smoke_job": {
                        "job_id": "8801",
                        "script_path": str(script_path),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return RuntimeQualificationRecord(
            tuple_id="a" * 64,
            path=record_path,
            status="submitted",
            job_id="8801",
            script_path=script_path,
        )

    monkeypatch.setattr(
        "bspp.orchestration.control.runtime_qualification.qualify_runtime",
        fake_qualify_runtime,
    )

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config_path),
            "runtime",
            "qualify",
            "--profile",
            "example-cluster",
            "--source-repo",
            str(source_repo),
        ],
    )

    assert result.exit_code == 0, result.output
    summary = yaml.safe_load(result.output)
    assert summary["runtime_qualification"]["profile"] == "example-cluster"
    assert summary["runtime_qualification"]["status"] == "submitted"
    assert len(summary["runtime_qualification"]["tuple_id"]) == 64
    record_path = Path(summary["runtime_qualification"]["record_path"])
    assert record_path.exists()
    assert record_path.parent == tmp_path / "qualifications" / "example-cluster"
    assert summary["runtime_qualification"]["expires_at"].endswith("Z")
    assert summary["runtime_qualification"]["job_id"] == "8801"
    assert summary["runtime_qualification"]["script_path"] == str(script_path)


def test_bsppctl_runtime_resolve_reports_exact_current_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record = record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )
    assert check.snapshot is not None
    monkeypatch.setattr(
        "bspp.orchestration.control.runtime_qualification.check_runtime_qualification",
        lambda **_kwargs: check,
    )

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config_path),
            "runtime",
            "resolve",
            "--profile",
            "example-cluster",
            "--source-repo",
            str(source_repo),
        ],
    )

    assert result.exit_code == 0, result.output
    summary = yaml.safe_load(result.output)["runtime_qualification"]
    assert summary["current"] is True
    assert summary["status"] == "qualified"
    assert summary["tuple_id"] == record.tuple_id
    assert summary["record_path"] == str(record.path)
    assert summary["record_sha256"] == check.snapshot.sha256
    assert summary["record_size_bytes"] == check.snapshot.size_bytes


def test_bsppctl_runtime_resolve_reports_noncurrent_and_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record_path = tmp_path / "qualifications" / "example-cluster" / f"{'a' * 64}.json"
    check = RuntimeQualificationCheck(False, "missing", "a" * 64, record_path)
    monkeypatch.setattr(
        "bspp.orchestration.control.runtime_qualification.check_runtime_qualification",
        lambda **_kwargs: check,
    )

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config_path),
            "runtime",
            "resolve",
            "--profile",
            "example-cluster",
            "--source-repo",
            str(source_repo),
        ],
    )

    assert result.exit_code == 1
    assert yaml.safe_load(result.output) == {
        "runtime_qualification": {
            "current": False,
            "reason": "missing",
            "tuple_id": "a" * 64,
            "record_path": str(record_path),
        }
    }


@pytest.mark.parametrize("mutation", ["missing-smoke", "forged-smoke"])
def test_qualified_record_requires_full_authentic_smoke_evidence(tmp_path: Path, mutation: str) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record = record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    payload = json.loads(record.path.read_text())
    if mutation == "missing-smoke":
        del payload["smoke_evidence"]
    else:
        payload["smoke_evidence"]["job_id"] = "9999"
    record.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )

    assert check.current is False
    assert check.reason == "invalid-record"
    assert check.snapshot is None


def test_existing_promoted_schema_v1_without_publication_compatibility_remains_current(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record = record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    payload = json.loads(record.path.read_text())
    del payload["smoke_evidence"]["publication_compatibility"]
    record.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )

    assert check.current
    assert check.reason == "current"
    assert check.snapshot is not None


@pytest.mark.parametrize("mutation", ["non-string", "wrong-attempt", "unsafe-record"])
def test_rootless_replay_rejects_forged_smoke_job_paths(tmp_path: Path, mutation: str) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record = record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    payload = json.loads(record.path.read_text())
    if mutation == "non-string":
        payload["smoke_job"]["script_path"] = 7
    elif mutation == "wrong-attempt":
        payload["smoke_job"]["result_path"] = str(tmp_path / "outside" / "result.json")
    else:
        payload["smoke_job"]["record_path"] = str(record.path.parent / ".." / record.path.name)
    document = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()

    with pytest.raises(ValueError, match="Runtime Qualification"):
        replay_postprocessing_runtime(document, attempt_id="attempt-0002")


def test_snapshot_capture_accepts_settled_valid_same_tuple_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record = record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    replacement = json.loads(record.path.read_text())
    replacement["attempt_history"].append({"status": "superseded"})
    replacement_bytes = (json.dumps(replacement, indent=2, sort_keys=True) + "\n").encode()
    from bspp.orchestration.control import runtime_qualification as runtime_module

    original_read = runtime_module._read_bounded_stable_bytes
    calls = 0

    def replace_between_reads(path: Path) -> tuple[bytes, str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_bytes(replacement_bytes)
        return original_read(path)

    monkeypatch.setattr(runtime_module, "_read_bounded_stable_bytes", replace_between_reads)
    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )

    assert check.current
    assert check.snapshot is not None
    assert check.snapshot.document == replacement_bytes
    assert calls == 2


def test_snapshot_capture_fails_closed_after_bounded_unstable_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    from bspp.orchestration.control import runtime_qualification as runtime_module

    original_read = runtime_module._read_bounded_stable_bytes
    calls = 0

    def race_after_initial_read(path: Path) -> tuple[bytes, str]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("simulated concurrent replacement")
        return original_read(path)

    monkeypatch.setattr(runtime_module, "_read_bounded_stable_bytes", race_after_initial_read)
    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )

    assert check.current is False
    assert check.reason == "invalid-record"
    assert check.snapshot is None
    assert calls == 4


def test_initial_authority_capture_retries_a_settled_same_tuple_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    from bspp.orchestration.control import runtime_qualification as runtime_module

    original_read = runtime_module._read_bounded_stable_bytes
    calls = 0

    def race_during_initial_capture(path: Path) -> tuple[bytes, str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("simulated concurrent replacement")
        return original_read(path)

    monkeypatch.setattr(runtime_module, "_read_bounded_stable_bytes", race_during_initial_capture)
    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )

    assert check.current is True
    assert check.snapshot is not None
    assert calls == 3


def test_initial_authority_capture_normalizes_bounded_os_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    from bspp.orchestration.control import runtime_qualification as runtime_module

    calls = 0

    def inaccessible_authority(_path: Path) -> tuple[bytes, str]:
        nonlocal calls
        calls += 1
        raise OSError("simulated path-stat failure")

    monkeypatch.setattr(runtime_module, "_read_bounded_stable_bytes", inaccessible_authority)
    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )

    assert check.current is False
    assert check.reason == "invalid-record"
    assert check.snapshot is None
    assert calls == 3


def test_initial_authority_path_stat_error_is_structured_invalid_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record = record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    original_lstat = Path.lstat

    def fail_authority_stat(path: Path) -> os.stat_result:
        if path == record.path:
            raise OSError("simulated authority lstat failure")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_authority_stat)
    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )

    assert check.current is False
    assert check.reason == "invalid-record"
    assert check.snapshot is None


def test_runtime_qualification_changes_to_tuple_identity_invalidate_record(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    changed_config_path = _write_profiles(tmp_path, gpu_worker_gres="gpu:2")

    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=changed_config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )

    assert check.current is False
    assert check.reason == "tuple-mismatch"
    assert check.record_path.parent == tmp_path / "qualifications" / "example-cluster"
    assert not check.record_path.exists()


def test_runtime_qualification_expires_at_profile_duration_boundary(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )

    current = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-13T11:59:59Z"),
    )
    stale = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-13T12:00:00Z"),
    )

    assert current.current is True
    assert current.reason == "current"
    assert stale.current is False
    assert stale.reason == "stale"


def test_runtime_qualification_unsupported_schema_version_is_not_current(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record = record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    payload = json.loads(record.path.read_text())
    payload["schema_version"] = 2
    record.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )

    assert check.current is False
    assert check.reason == "unsupported-schema-version"
    assert check.record_path == record.path


def test_postprocessing_retry_resolver_consumes_checked_snapshot_without_path_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    record = record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )
    assert check.snapshot is not None
    record.path.write_text("{}\n")
    monkeypatch.setattr(
        "bspp.orchestration.control.postprocessing_runtime_qualification.check_runtime_qualification",
        lambda **_kwargs: check,
    )

    _profile, selection, document = resolve_current_postprocessing_runtime(
        attempt_id="attempt-0002",
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        observed_at=_instant("2026-06-06T13:00:00Z"),
    )

    assert document == check.snapshot.document
    assert selection.qualification_sha256 == check.snapshot.sha256
    assert selection.qualification_size_bytes == check.snapshot.size_bytes


def test_baked_qualification_tuple_includes_selected_source_with_baked_kind_and_root(tmp_path: Path) -> None:
    """Task 1: Baked mode tuple has selected_source with baked kind, root, and image_identity."""
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    baked_profile = replace(profile, afdb_toolkit_repo=None)
    assert _selected_source_kind(baked_profile) == "baked"
    qualification_tuple = _qualification_tuple(baked_profile, source_repo=source_repo)
    selected = qualification_tuple["selected_source"]
    assert selected["source_kind"] == "baked"
    assert selected["root"] == "/opt/afdb-toolkit"
    assert selected["revision"] == BAKED_TOOLKIT_COMMIT
    assert "image_identity" in selected
    assert "toolkit_package_identity" not in selected
    assert "toolkit_package_identity" not in qualification_tuple
    assert "toolkit_package_path" not in qualification_tuple
    assert "toolkit_source" not in qualification_tuple


def test_override_qualification_tuple_retains_toolkit_package_identity(tmp_path: Path) -> None:
    """Task 1: Override mode tuple has selected_source with override kind and toolkit_package_identity."""
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    assert _selected_source_kind(profile) == "override"
    qualification_tuple = _qualification_tuple(profile, source_repo=source_repo)
    selected = qualification_tuple["selected_source"]
    assert selected["source_kind"] == "override"
    assert selected["root"] == "/workspace/AFDB-Integration-Kit"
    assert "toolkit_package_identity" in selected
    assert selected["toolkit_package_identity"] == qualification_tuple["toolkit_package_identity"]
    assert "toolkit_package_path" in qualification_tuple
    assert "toolkit_source" in qualification_tuple


def test_baked_mode_omits_toolkit_package_from_tuple(tmp_path: Path) -> None:
    """Task 1: Baked mode tuple omits toolkit_package_identity and toolkit_package_path."""
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    baked_profile = replace(profile, afdb_toolkit_repo=None)
    qualification_tuple = _qualification_tuple(baked_profile, source_repo=source_repo)
    assert "toolkit_package_identity" not in qualification_tuple
    assert "toolkit_package_path" not in qualification_tuple
    assert "toolkit_source" not in qualification_tuple
    assert "selected_source" in qualification_tuple
    assert qualification_tuple["selected_source"]["source_kind"] == "baked"


def test_baked_smoke_script_uses_opt_afdb_toolkit_and_omits_toolkit_package_mount(tmp_path: Path) -> None:
    """Task 2: Baked smoke script uses /opt/afdb-toolkit and omits toolkit package mount."""
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    baked_profile = replace(profile, afdb_toolkit_repo=None)
    qualification_tuple = _qualification_tuple(baked_profile, source_repo=source_repo)
    tuple_id = _tuple_id(qualification_tuple)
    record_path = tmp_path / "qualifications" / "example-cluster" / f"{tuple_id}.json"
    script = render_runtime_qualification_smoke_script(
        profile=baked_profile,
        qualification_tuple=qualification_tuple,
        tuple_id=tuple_id,
        record_path=record_path,
        expires_hours=168,
    )
    assert "/opt/afdb-toolkit" in script
    assert _TOOLKIT_PACKAGE_CONTAINER not in script
    assert "--toolkit-package" not in script
    assert BAKED_TOOLKIT_COMMIT in script


def test_override_smoke_script_preserves_existing_toolkit_package_behavior(tmp_path: Path) -> None:
    """Task 2: Override smoke script preserves existing toolkit package behavior."""
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    qualification_tuple = _qualification_tuple(profile, source_repo=source_repo)
    tuple_id = _tuple_id(qualification_tuple)
    record_path = tmp_path / "qualifications" / "example-cluster" / f"{tuple_id}.json"
    script = render_runtime_qualification_smoke_script(
        profile=profile,
        qualification_tuple=qualification_tuple,
        tuple_id=tuple_id,
        record_path=record_path,
        expires_hours=168,
    )
    assert "{BSPP_TOOLKIT_ROOT}" in script
    assert "--toolkit-package" in script
    assert _TOOLKIT_PACKAGE_CONTAINER in script


def test_smoke_script_runs_authenticated_publication_compatibility_before_ipsae(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    qualification_tuple = _qualification_tuple(profile, source_repo=source_repo)
    tuple_id = _tuple_id(qualification_tuple)
    script = render_runtime_qualification_smoke_script(
        profile=profile,
        qualification_tuple=qualification_tuple,
        tuple_id=tuple_id,
        record_path=tmp_path / "qualifications" / "example-cluster" / f"{tuple_id}.json",
        expires_hours=168,
    )
    srun_lines = [line for line in script.splitlines() if line.startswith("srun ")]
    assert len(srun_lines) == 2
    rendered_argv = []
    for line in srun_lines:
        tokens = shlex.split(line)
        marker = tokens.index("--exec-argv-json")
        rendered_argv.append(json.loads(tokens[marker + 1]))
    compatibility_argv, ipsae_argv = rendered_argv
    assert compatibility_argv == [
        str(PIXI_PYTHON_PATH),
        "-I",
        "-c",
        compatibility_argv[3],
        "{BSPP_SOURCE_ROOT}",
        "bspp.orchestration.runtime.postprocessing.publication_compatibility",
        "--root",
        "/run/bspp/result/publication-compatibility",
        "--output",
        "/run/bspp/result/publication-compatibility.json",
    ]
    assert "-S" not in compatibility_argv
    assert "packages/orchestration-contract/src" in compatibility_argv[3]
    assert "packages/orchestration-runtime/src" in compatibility_argv[3]
    assert ipsae_argv[:4] == [str(PIXI_PYTHON_PATH), "-I", "-S", "-c"]
    assert "afdb_integration_kit/ipsae" in ipsae_argv[4]


def _historical_publication_compatibility_launcher() -> str:
    return (
        "import runpy,sys;from pathlib import Path;"
        "source=Path(sys.argv[1]);module=sys.argv[2];"
        "sys.path[:0]=[str(source/'packages/orchestration-contract/src'),"
        "str(source/'packages/orchestration-runtime/src')];"
        "sys.argv=[module,*sys.argv[3:]];runpy.run_module(module,run_name='__main__')"
    )


def _run_real_publication_compatibility(
    tmp_path: Path, *, launcher: str, flags: tuple[str, ...]
) -> subprocess.CompletedProcess[str]:
    source = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve() / "publication-compatibility"
    output = tmp_path.resolve() / "publication-compatibility.json"
    return subprocess.run(
        (
            sys.executable,
            *flags,
            "-c",
            launcher,
            str(source),
            "bspp.orchestration.runtime.postprocessing.publication_compatibility",
            "--root",
            str(root),
            "--output",
            str(output),
        ),
        check=False,
        capture_output=True,
        text=True,
    )


def test_historical_runpy_launcher_with_no_site_fails_on_qualified_dependency(tmp_path: Path) -> None:
    completed = _run_real_publication_compatibility(
        tmp_path, launcher=_historical_publication_compatibility_launcher(), flags=("-I", "-S")
    )
    assert completed.returncode != 0
    assert "ModuleNotFoundError: No module named 'yaml'" in completed.stderr


def test_historical_runpy_launcher_with_site_fails_spawn_identity(tmp_path: Path) -> None:
    completed = _run_real_publication_compatibility(
        tmp_path, launcher=_historical_publication_compatibility_launcher(), flags=("-I",)
    )
    assert completed.returncode != 0
    assert "PicklingError" in completed.stderr
    assert "attribute lookup _worker on __main__ failed" in completed.stderr


def test_production_launcher_executes_real_multiprocess_publication_compatibility(tmp_path: Path) -> None:
    from bspp.orchestration.control.runtime_qualification import _publication_compatibility_launcher

    completed = _run_real_publication_compatibility(
        tmp_path, launcher=_publication_compatibility_launcher(), flags=("-I",)
    )
    assert completed.returncode == 0, completed.stderr
    root = tmp_path.resolve() / "publication-compatibility"
    evidence = json.loads((tmp_path.resolve() / "publication-compatibility.json").read_text())
    assert evidence == {
        "check": "postprocessing-directory-publication-v1",
        "fallback_errno": 22,
        "output_identity": {
            "path": "published/payload.json",
            "sha256": "bf04841124813309efa145d98b02da3eae3e3dd0ec4009a709c25d55af684c05",
            "size_bytes": 35,
        },
        "process_count": 2,
        "published_directory_count": 1,
        "schema_version": 1,
        "status": "passed",
    }
    assert not tuple(root.glob("stage-*"))


def test_publication_compatibility_launcher_prefers_authenticated_source_and_qualified_dependencies(
    tmp_path: Path,
) -> None:
    from bspp.orchestration.control.runtime_qualification import _publication_compatibility_launcher

    source = tmp_path / "authenticated-source"
    contract_src = source / "packages/orchestration-contract/src"
    runtime_src = source / "packages/orchestration-runtime/src"
    contract_src.mkdir(parents=True)
    module_name = "bspp.orchestration.runtime.postprocessing.fixture_qualification"
    module = runtime_src / Path(*module_name.split(".")).with_suffix(".py")
    module.parent.mkdir(parents=True)
    (runtime_src / "bspp/orchestration/runtime/__init__.py").write_text("")
    (module.parent / "__init__.py").write_text("")
    output = tmp_path / "observed.json"
    module.write_text(
        "import json,sys,yaml\n"
        "from pathlib import Path\n"
        "def main(argv):\n"
        " Path(argv[1]).write_text(json.dumps({"
        "'module_name':__name__,'module_file':__file__,'dependency_file':yaml.__file__,"
        "'argv':list(argv),'paths':sys.path[:2],'all_paths':sys.path},sort_keys=True)+'\\n')\n"
        " return 0\n"
    )
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    (hostile / "yaml.py").write_text("raise RuntimeError('hostile yaml imported')\n")
    hostile_module = hostile / Path(*module_name.split(".")).with_suffix(".py")
    hostile_module.parent.mkdir(parents=True)
    hostile_module.write_text("raise RuntimeError('hostile module imported')\n")
    user_site = (
        hostile / "userbase" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )
    user_site.mkdir(parents=True)
    (user_site / "yaml.py").write_text("raise RuntimeError('hostile user-site yaml imported')\n")
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-c",
            _publication_compatibility_launcher(),
            str(source),
            module_name,
            "--output",
            str(output),
        ),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(hostile), "PYTHONUSERBASE": str(hostile / "userbase")},
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(output.read_text())
    assert observed["module_name"] == module_name
    assert Path(observed["module_file"]) == module
    assert Path(observed["dependency_file"]).name == "__init__.py"
    assert hostile not in Path(observed["dependency_file"]).parents
    assert observed["argv"] == ["--output", str(output)]
    assert observed["paths"] == [str(contract_src), str(runtime_src)]
    assert str(hostile) not in observed["all_paths"]
    assert str(user_site) not in observed["all_paths"]


@pytest.mark.parametrize("mutation", ("missing", "malformed", "oversize", "mismatch"))
def test_invalid_publication_compatibility_prevents_promotion(tmp_path: Path, mutation: str) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    submitted = qualify_runtime(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
        runner=FakeSubmitRunner(["8801"]),
    )
    payload = json.loads(submitted.path.read_text())
    qualification_tuple = payload["tuple"]
    result_path = Path(payload["smoke_job"]["result_path"])
    compatibility = write_publication_compatibility_artifact(result_path)
    artifact = result_path.parent / "publication-compatibility.json"
    if mutation == "malformed":
        compatibility["check"] = "wrong-check"
    elif mutation == "oversize":
        document = b"x" * 4097
        artifact.write_bytes(document)
        compatibility["artifact"] = {
            "path": "publication-compatibility.json",
            "sha256": hashlib.sha256(document).hexdigest(),
            "size_bytes": len(document),
        }
    elif mutation == "mismatch":
        artifact.write_text("{}\n")
    result_path.write_text(
        json.dumps(
            {
                "tuple_id": submitted.tuple_id,
                "job_id": "8801",
                "status": "succeeded",
                "attempt_token": payload["smoke_job"]["attempt_token"],
                "python": "3.12",
                "gpu": "synthetic gpu",
                "bootstrap_sha256": "7" * 64,
                "source_package_identity": qualification_tuple["source_package_identity"],
                "toolkit_package_identity": qualification_tuple.get("toolkit_package_identity"),
                "image_identity": qualification_tuple["image_identity"],
                "selected_source_identity": qualification_tuple["selected_source"],
                "runtime_ipsae": write_runtime_ipsae_artifacts(
                    result_path, source_revision=_toolkit_revision(qualification_tuple)
                ),
                **({} if mutation == "missing" else {"publication_compatibility": compatibility}),
            }
        )
    )
    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:01:00Z"),
        runner=CompletedJobRunner(),
    )
    assert not check.current
    assert check.reason == "not-qualified"


def test_baked_smoke_result_promotes_with_selected_source_identity(tmp_path: Path) -> None:
    """Task 2: Baked qualified record with selected_source_identity is current."""
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    # Override example-cluster user config to baked mode by omitting afdb_toolkit_repo
    config = yaml.safe_load(config_path.read_text())
    example_cluster_profile = dict(config["clusters"]["example-cluster"])
    del example_cluster_profile["afdb_toolkit_repo"]
    config["clusters"]["example-cluster"] = example_cluster_profile
    config_path.write_text(yaml.safe_dump(config))

    qualification_tuple = _qualification_tuple(
        resolve_cluster_profile("example-cluster", config_path=config_path), source_repo=source_repo
    )
    tuple_id = _tuple_id(qualification_tuple)
    record = record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    assert record.tuple_id == tuple_id

    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )
    assert check.current
    promoted = json.loads(record.path.read_text())
    assert promoted["status"] == "qualified"
    assert promoted["smoke_evidence"]["selected_source_identity"]["source_kind"] == "baked"


def test_smoke_result_rejects_selected_source_mismatch(tmp_path: Path) -> None:
    """Task 2: Baked and override profiles produce different tuples (selected_source mismatch)."""
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    # Create a baked record by omitting afdb_toolkit_repo
    config = yaml.safe_load(config_path.read_text())
    baked_example_cluster_profile = dict(config["clusters"]["example-cluster"])
    del baked_example_cluster_profile["afdb_toolkit_repo"]
    config["clusters"]["example-cluster"] = baked_example_cluster_profile
    config_path.write_text(yaml.safe_dump(config))

    baked_profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    qualification_tuple = _qualification_tuple(baked_profile, source_repo=source_repo)
    tuple_id = _tuple_id(qualification_tuple)
    record = record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    assert record.tuple_id == tuple_id

    # Restore override afdb_toolkit_repo — tuple now mismatches baked record
    config = yaml.safe_load(config_path.read_text())
    config["clusters"]["example-cluster"]["afdb_toolkit_repo"] = str(tmp_path / "toolkit")
    config_path.write_text(yaml.safe_dump(config))

    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )
    assert not check.current
    assert check.reason == "tuple-mismatch"


def test_baked_smoke_python_attests_provenance(tmp_path: Path) -> None:
    """Task 2: In-container smoke Python validates baked provenance file."""
    from bspp.orchestration.control.runtime_qualification import _runtime_ipsae_smoke_python

    code = _runtime_ipsae_smoke_python()
    assert "source_kind" in code
    assert "expected_revision" in code
    assert "baked toolkit provenance attestation" in code
    assert "provenance.json" in code
    assert "selected_source_identity" in code


def test_smoke_python_compiles_and_hashes_explicit_bootstrap_path(tmp_path: Path) -> None:
    """The in-container smoke program hashes the exact bootstrap path passed by the renderer."""
    from bspp.orchestration.control.runtime_qualification import _runtime_ipsae_smoke_python

    code = _runtime_ipsae_smoke_python()
    compile(code, "<smoke>", "exec")
    assert "def sha(path):" in code
    assert "bootstrap_sha256=sha(Path(sys.argv[15]))" in code


def test_override_mode_preserves_toolkit_package_identity_validation(tmp_path: Path) -> None:
    """Task 3: Override mode still validates toolkit_package_identity in result."""
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    qualification_tuple = _qualification_tuple(profile, source_repo=source_repo)
    tuple_id = _tuple_id(qualification_tuple)
    record = record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    assert record.tuple_id == tuple_id

    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
        runner=CompletedJobRunner(),
    )
    assert check.current


def test_baked_tuple_change_invalidates_existing_record(tmp_path: Path) -> None:
    """Task 3: Changing the baked profile (different image) invalidates existing record."""
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    # Override example-cluster user config to baked mode
    config = yaml.safe_load(config_path.read_text())
    baked_example_cluster_profile = dict(config["clusters"]["example-cluster"])
    del baked_example_cluster_profile["afdb_toolkit_repo"]
    config["clusters"]["example-cluster"] = baked_example_cluster_profile
    config_path.write_text(yaml.safe_dump(config))

    baked_profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    qualification_tuple = _qualification_tuple(baked_profile, source_repo=source_repo)
    tuple_id = _tuple_id(qualification_tuple)
    record = record_runtime_qualification_smoke_success(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T12:00:00Z"),
    )
    assert record.tuple_id == tuple_id

    # Change the baked profile's image while keeping it baked
    different_image = tmp_path / "images" / "different.sqsh"
    different_image.write_bytes(b"different-runtime-image")
    config = yaml.safe_load(config_path.read_text())
    config["clusters"]["example-cluster"]["image"] = str(different_image)
    config_path.write_text(yaml.safe_dump(config))

    check = check_runtime_qualification(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        now=_instant("2026-06-06T13:00:00Z"),
    )
    assert not check.current
    assert check.reason == "tuple-mismatch"


def test_baked_to_override_transition_produces_different_tuple(tmp_path: Path) -> None:
    """Task 3: Baked and override modes produce different tuple IDs."""
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    baked_profile = replace(profile, afdb_toolkit_repo=None)
    baked_tuple = _qualification_tuple(baked_profile, source_repo=source_repo)
    override_tuple = _qualification_tuple(profile, source_repo=source_repo)
    assert _tuple_id(baked_tuple) != _tuple_id(override_tuple)


def test_valid_selected_source_identity_rejects_mismatched_fields() -> None:
    """_valid_selected_source_identity: source_kind/root must match; baked mode
    accepts a self-reported 40-hex revision (image-sha256 binding, Option A)."""
    baked = selected_source_identity(source_kind="baked")
    override = selected_source_identity(source_kind="override")
    assert _valid_selected_source_identity(baked, baked)
    assert not _valid_selected_source_identity(baked, override)
    # Baked mode: the tuple's revision is only a public placeholder; the smoke's
    # self-reported 40-hex revision is authoritative.
    assert _valid_selected_source_identity(
        {"source_kind": "baked", "root": "/opt/afdb-toolkit", "revision": "a" * 40},
        {"source_kind": "baked", "root": "/opt/afdb-toolkit", "revision": "b" * 40},
    )
    # Baked mode still rejects a non-40-hex self-reported revision and a root mismatch.
    assert not _valid_selected_source_identity(
        {"source_kind": "baked", "root": "/opt/afdb-toolkit", "revision": "not-a-sha"},
        {"source_kind": "baked", "root": "/opt/afdb-toolkit", "revision": "b" * 40},
    )
    assert not _valid_selected_source_identity(
        {"source_kind": "baked", "root": "/other", "revision": "a" * 40},
        {"source_kind": "baked", "root": "/opt/afdb-toolkit", "revision": "b" * 40},
    )
    assert not _valid_selected_source_identity(None, baked)
    assert not _valid_selected_source_identity(baked, None)


def test_toolkit_revision_falls_back_to_selected_source_for_baked(tmp_path: Path) -> None:
    """Task 3: _toolkit_revision reads from selected_source when toolkit_package_identity is absent."""
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    baked_profile = replace(profile, afdb_toolkit_repo=None)
    qualification_tuple = _qualification_tuple(baked_profile, source_repo=source_repo)
    assert "toolkit_package_identity" not in qualification_tuple
    assert _toolkit_revision(qualification_tuple) == BAKED_TOOLKIT_COMMIT


def _init_git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "config", "user.email", "tester@example.com")
    _git(path, "config", "user.name", "Tester")
    (path / "README.md").write_text("hello\n")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    return path


def _write_profiles(
    tmp_path: Path,
    *,
    image: str | None = None,
    gpu_worker_gres: str = "gpu:1",
    include_runtime_qualification: bool = True,
) -> Path:
    image_path = Path(image) if image is not None else tmp_path / "images" / "bspp.sqsh"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        image_path.write_bytes(b"synthetic-runtime-image")
    toolkit_path = tmp_path / "toolkit"
    if not toolkit_path.exists():
        _init_git_repo(toolkit_path)
    cluster = {
        "owner": "tester",
        "project_root": str(tmp_path / "project"),
        "output_root": str(tmp_path / "output"),
        "staging_root": str(tmp_path / "staging"),
        "afdb_toolkit_repo": str(toolkit_path),
        "orchestration_repo": str(tmp_path / "orchestration"),
        "image": str(image_path),
        "transport": "local-slurm",
        "source_bundle_root": str(tmp_path / "bundles"),
        "runtime_image_cache_root": str(tmp_path / "image-cache"),
        "account": "user-account",
        "resources": {
            "gpu_worker": {
                "partition": "gpu",
                "cpus_per_task": 30,
                "memory": "128G",
                "time": "04:00:00",
                "gres": gpu_worker_gres,
            }
        },
    }
    if include_runtime_qualification:
        cluster.update(
            {
                "runtime_qualification_root": str(tmp_path / "qualifications"),
                "runtime_qualification_expires_hours": 168,
            }
        )
    return _write_yaml(
        tmp_path / "profiles.yaml",
        {
            "clusters": {
                "example-cluster": cluster,
            }
        },
    )


def _write_workflow(path: Path) -> None:
    _write_yaml(path, {"workflow": {"steps": [{"name": "preflight", "run": True}]}})


def _write_acceptance_workflow(path: Path) -> None:
    _write_yaml(
        path,
        {
            "workflow": {
                "steps": [
                    {"name": "preflight", "run": True},
                    {"name": "acceptance-tar-payload-parity", "run": True, "mode": "submit-and-monitor"},
                    {"name": "acceptance-semantic", "run": True, "mode": "submit-and-monitor"},
                    {"name": "acceptance-verify-evidence", "run": True},
                ]
            }
        },
    )


def _runplan_data() -> dict[str, object]:
    return {
        "run_kind": "dev",
        "target_cluster": "example-cluster",
        "workflow_template": "workflow.yaml",
        "dataset": {
            "name": "ds1",
            "run_id": "run1",
            "mode": "archive",
            "array": "0-3",
            "archive_source": "tracking",
        },
        "references": {
            "master_parquet": "/refs/master.parquet",
            "tracking_parquet": "/refs/tracking.parquet",
            "manifest_csv": "/refs/manifest.csv",
            "uniprot_duckdb": "/refs/uniprot.duckdb",
        },
        "worker": {
            "stages": "metadata_export",
            "workers": 24,
            "batch_size": 500,
            "shards_per_archive": 2,
            "self_upload": False,
            "local_scratch": True,
            "scratch_dir": "/dev/shm",
            "s5cmd_path": "s5cmd",
            "upload_slots": 4,
        },
        "storage": {
            "s3_archive_prefix": "s3://example-bucket/structures/",
            "s3_output_prefix": "s3://example-bucket/users/test/postprocessed/ds1/",
            "allow_production_prefixes": False,
        },
        "validation": {"expected_archives": 4},
        "secrets": {"s3_credentials_ref": "env:bspp/s3"},
    }


def _write_yaml(path: Path, data: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _instant(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", "-C", str(repo), *args), capture_output=True, text=True, check=True)
    return result.stdout.strip()
