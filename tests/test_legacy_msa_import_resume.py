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

"""Control-only receipt-loss recovery: exact identity, no resubmission or payload I/O."""

from __future__ import annotations

import json
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.preprocessing_handoff import (
    BundledMemberVerification,
    MsaArtifactMember,
    MsaArtifactSetManifest,
    MsaChunkManifest,
    MsaChunkManifestReference,
    PreprocessingHandoffBundle,
    VerifiedLocalBundledArtifactLocation,
    msa_artifact_set_id,
    preprocessing_content_validation_evidence_from_mapping,
    preprocessing_content_validation_evidence_id,
    verified_local_bundled_artifact_location_id,
)
from bspp.orchestration.control import legacy_msa_import as subject
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.monitoring import SlurmJobState
from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.control.transport import CommandResult, RemoteSlurmTransport
from tests.support.transport_argv import unwrap_remote_command

_JOB = "18957638"


def _bundle(lengths=None):
    """Typed metadata fixture only; no claimed archive is created or opened."""
    member = MsaArtifactMember(
        record_identity="AFDB_AF-0000000000000001",
        source_ordinal=0,
        source_header=">AFDB_AF-0000000000000001",
        logical_path="a3ms/member.a3m",
        size_bytes=25,
        sha256="a" * 64,
    )
    chunk = MsaChunkManifest(
        chunk_name="sample_tranche00_00001.fa", members=(member,), member_count=1, logical_bytes=25
    )
    ref = MsaChunkManifestReference(
        chunk_name="sample_tranche00_00001.fa",
        logical_path="chunks/sample_tranche00_00001.json",
        sha256=chunk.digest,
        member_count=1,
        logical_bytes=25,
    )
    aset = MsaArtifactSetManifest(
        artifact_set_id=msa_artifact_set_id((ref,), 1, 25, member_lengths=lengths),
        chunks=(ref,),
        member_count=1,
        logical_bytes=25,
        member_lengths=lengths,
    )
    location_fields = dict(
        artifact_set_id=aset.artifact_set_id,
        tar_path="/payload/bundle.tar",
        bundle_path="/payload/bundle.tar.lz4",
        bundle_uri="file:///payload/bundle.tar.lz4",
        tar_size_bytes=2048,
        tar_sha256="b" * 64,
        lz4_size_bytes=100,
        lz4_sha256="c" * 64,
        raw_tar_members=("./member.a3m",),
        members=(
            BundledMemberVerification(
                logical_path=member.logical_path,
                member_name="member.a3m",
                raw_member_name="./member.a3m",
                size_bytes=member.size_bytes,
                sha256=member.sha256,
            ),
        ),
    )
    location = VerifiedLocalBundledArtifactLocation(
        artifact_location_id=verified_local_bundled_artifact_location_id(**location_fields),
        verified_at="2026-09-01T00:00:00.000000Z",
        **location_fields,
    )
    body = dict(
        schema_version=1,
        phase_run_id="phase-run-" + "a" * 32,
        attempt_id="attempt-0001",
        phase_runspec_digest="d" * 64,
        action_id="preprocessing-chunk-000001",
        action_evidence_digest="e" * 64,
        chunk_name=chunk.chunk_name,
        started_at="2026-09-01T00:00:00.000000Z",
        finished_at="2026-09-01T00:00:00.000000Z",
        outcome="passed",
        artifact_set_id=aset.artifact_set_id,
        artifact_location_id=location.artifact_location_id,
        chunk_manifest_digest=chunk.digest,
        artifact_set_manifest_digest=canonical_mapping_digest(aset.to_mapping()),
        member_count=1,
        logical_bytes=25,
        lz4_command_digest="f" * 64,
        lz4_return_code=0,
        decompressed_tar_size_bytes=2048,
        decompressed_tar_sha256="b" * 64,
        error=None,
    )
    body["evidence_id"] = preprocessing_content_validation_evidence_id(body)
    validation = preprocessing_content_validation_evidence_from_mapping(body)
    return PreprocessingHandoffBundle(chunk, aset, location, validation)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    profile_path = tmp_path / "profiles.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "clusters": {
                    "example-cluster": {
                        "owner": "tester",
                        "project_root": str(tmp_path / "project"),
                        "output_root": str(tmp_path / "output"),
                        "staging_root": str(tmp_path / "staging"),
                        "orchestration_repo": "/original-source",
                        "image": "/original-image.sqsh",
                        "transport": "local-slurm",
                        "account": "user-account",
                    }
                }
            }
        )
    )
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    handoff, output = tmp_path / "handoff", tmp_path / "enriched"
    legacy, enriched = _bundle(), _bundle((5,))
    records = {}
    for root, bundle in ((handoff, legacy), (output, enriched)):
        records[root / "artifact-set.json"] = bundle.artifact_set.to_mapping()
        records[root / "artifact-location.json"] = bundle.artifact_location.to_mapping()
        records[root / "content-validation.json"] = bundle.content_validation.to_mapping()
        records[root / "chunks/sample_tranche00_00001.json"] = bundle.chunk_manifest.to_mapping()
    fetched = []

    def fetch(transport, path):
        fetched.append(path)
        return records[path]

    monkeypatch.setattr(subject, "_fetch_record", fetch)
    rendered = subject.render_legacy_msa_import_submission(
        cluster_profile=profile,
        handoff_root=handoff,
        output_dir=output,
        script_path=tmp_path / "fixture.sbatch",
        payload_paths=(Path("/payload/bundle.tar"), Path("/payload/bundle.tar.lz4")),
    )
    script = rendered.script_path.read_bytes()
    script_path = output.with_name(output.name + "-slurm") / "legacy-msa-import.sbatch"
    context = SimpleNamespace(
        profile=profile,
        profile_path=profile_path,
        handoff=handoff,
        output=output,
        records=records,
        fetched=fetched,
        script=script,
        script_path=script_path,
        script_fetches=0,
        script_mutation=False,
        job_name=rendered.job_name,
    )

    def stable(transport, path, *, remote_root, maximum_bytes):
        assert path == script_path and remote_root == script_path.parent and maximum_bytes == 256 * 1024
        context.script_fetches += 1
        return context.script + (b"changed" if context.script_mutation and context.script_fetches > 1 else b"")

    monkeypatch.setattr(RemoteSlurmTransport, "fetch_stable_artifact", stable)
    return context


class ReadOnlyRunner:
    def __init__(self, context):
        self.context = context
        self.calls = []
        self.owner = "tester\n"
        self.identity = f"{_JOB}|{context.job_name}|tester|sbatch --parsable {shlex.quote(str(context.script_path))}\n"
        self.identity_returncode = 0
        self.state, self.exit_code = "COMPLETED", "0:0"
        self.unavailable = False

    def __call__(self, argv):
        args = tuple(shlex.split(unwrap_remote_command(argv[-1]))) if argv[0] == "ssh" else argv
        self.calls.append(args)
        if args == ("id", "-un"):
            return CommandResult(argv, 0, self.owner, "")
        if args[:3] == ("env", "TZ=UTC", "sacct"):
            assert args[3:] == (
                "--duplicates",
                "-X",
                "--parsable2",
                "--noheader",
                "--jobs",
                _JOB,
                "--format=JobIDRaw,JobName%128,User%128,SubmitLine%4096",
            )
            return CommandResult(argv, self.identity_returncode, self.identity, "")
        if args[0] == "squeue":
            return CommandResult(argv, 1, "", "slurm_load_jobs error: Invalid job id specified")
        if args[0] == "sacct":
            if self.unavailable:
                return CommandResult(argv, 1, "", "accounting unavailable")
            if "--json" in args:
                sparse = {
                    "job_id": int(_JOB),
                    "state": {"current": [self.state]},
                    "exit_code": {
                        "return_code": {"set": True, "number": 0},
                        "signal": {"id": {"set": False, "number": 0}},
                    },
                }
                return CommandResult(argv, 0, json.dumps({"jobs": [sparse]}), "")
            if any("Restarts" in arg for arg in args):
                return CommandResult(argv, 1, "", 'sacct: error: Invalid field requested: "Restarts"')
            return CommandResult(argv, 0, f"{_JOB}|{_JOB}|{self.state}|{self.exit_code}\n", "")
        pytest.fail(f"unexpected command, possible mutation: {args}")


def _resume(context, runner, **kwargs):
    return subject.resume_legacy_msa_import(
        job_id=kwargs.pop("job_id", _JOB),
        cluster_profile=kwargs.pop("cluster_profile", context.profile),
        handoff_root=context.handoff,
        output_dir=context.output,
        runner=runner,
        poll_interval_seconds=0.001,
        timeout_seconds=0.001,
        sleeper=lambda _: None,
        **kwargs,
    )


@pytest.mark.parametrize("kind", ["ssh", "local-slurm"])
def test_resume_expired_queue_real_transport_fallback_and_strict_results(prepared, kind):
    prepared.profile = replace(prepared.profile, transport=kind, ssh_target="example-host" if kind == "ssh" else None)
    runner = ReadOnlyRunner(prepared)
    result = _resume(prepared, runner)
    assert json.loads(result.render_json()) == {
        "job_id": _JOB,
        "artifact_set_id": prepared.records[prepared.output / "artifact-set.json"]["artifact_set_id"],
        "artifact_location_id": prepared.records[prepared.output / "artifact-location.json"]["artifact_location_id"],
        "member_lengths": [5],
    }
    assert prepared.script_fetches == 2
    assert len(prepared.fetched) == 7
    assert not prepared.output.exists() and not prepared.script_path.parent.exists()
    assert sum(call[0] == "squeue" for call in runner.calls) == 2
    assert any("Restarts" in arg for call in runner.calls for arg in call)


@pytest.mark.parametrize("job", ["", "0", "01", "12_0", "12.batch", " 12", "\uff11\uff12", "12;true"])
def test_invalid_id_precedes_all_transport(prepared, job):
    runner = ReadOnlyRunner(prepared)
    with pytest.raises(ValueError, match="scalar job id"):
        _resume(prepared, runner, job_id=job)
    assert not runner.calls and not prepared.fetched


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-id",
        "wrong-owner",
        "wrong-name",
        "duplicate",
        "absent",
        "oversize",
        "nonzero",
        "wrong-path",
        "extra-argument",
        "truncated",
        "malformed",
        "owner-session",
    ],
)
def test_identity_refusals(prepared, mutation):
    runner = ReadOnlyRunner(prepared)
    if mutation == "wrong-id":
        runner.identity = runner.identity.replace(_JOB, "18957244")
    if mutation == "wrong-owner":
        runner.identity = runner.identity.replace("|tester|", "|foreign|")
    if mutation == "wrong-name":
        runner.identity = runner.identity.replace(prepared.job_name, "other")
    if mutation == "duplicate":
        runner.identity *= 2
    if mutation == "absent":
        runner.identity = ""
    if mutation == "oversize":
        runner.identity = "x" * 65537
    if mutation == "nonzero":
        runner.identity_returncode = 1
    if mutation == "wrong-path":
        runner.identity = runner.identity.replace("enriched-slurm", "another-slurm")
    if mutation == "extra-argument":
        runner.identity = runner.identity.rstrip() + " --export=ALL\n"
    if mutation == "truncated":
        runner.identity = runner.identity[:50] + "+\n"
    if mutation == "malformed":
        runner.identity = runner.identity.rstrip() + "'\n"
    if mutation == "owner-session":
        runner.owner = "foreign\n"
    with pytest.raises(ValueError, match="resume"):
        _resume(prepared, runner)
    assert not any(p.parent == prepared.output for p in prepared.fetched)


@pytest.mark.parametrize("change", ["script", "image", "source", "lz4", "after-poll"])
def test_expected_script_and_final_recheck(prepared, change):
    kwargs = {}
    if change == "script":
        prepared.script += b"changed"
    if change == "image":
        prepared.profile = replace(prepared.profile, image="/wrong-image.sqsh")
    if change == "source":
        prepared.profile = replace(prepared.profile, orchestration_repo="/wrong-source")
    if change == "lz4":
        kwargs["lz4_executable"] = "other-lz4"
    if change == "after-poll":
        prepared.script_mutation = True
    with pytest.raises(ValueError, match="staged script differs"):
        _resume(prepared, ReadOnlyRunner(prepared), **kwargs)
    assert not any(p.parent == prepared.output for p in prepared.fetched)


@pytest.mark.parametrize(("state", "exit_code"), [("FAILED", "15:0"), ("CANCELLED", "0:0"), ("COMPLETED", "1:0")])
def test_failed_native_accounting_is_never_accepted(prepared, state, exit_code):
    runner = ReadOnlyRunner(prepared)
    runner.state, runner.exit_code = state, exit_code
    with pytest.raises(ValueError, match="exact sacct COMPLETED"):
        _resume(prepared, runner)
    assert not any(p.parent == prepared.output for p in prepared.fetched)


def test_both_scheduler_sources_unavailable_times_out(prepared):
    runner = ReadOnlyRunner(prepared)
    runner.unavailable = True
    with pytest.raises(ValueError, match="timed out"):
        _resume(prepared, runner)
    assert not any(p.parent == prepared.output for p in prepared.fetched)


@pytest.mark.parametrize(
    "state",
    [
        SlurmJobState(_JOB, "COMPLETED", "squeue", "0:0"),
        SlurmJobState("18957244", "COMPLETED", "sacct", "0:0"),
        SlurmJobState(_JOB, "COMPLETED", "sacct", None),
    ],
)
def test_strict_gate_rejects_wrong_source_id_or_unknown_exit(prepared, monkeypatch, state):
    monkeypatch.setattr(subject, "_wait_for_terminal_state", lambda *a, **k: state)
    with pytest.raises(ValueError, match="exact sacct COMPLETED"):
        _resume(prepared, ReadOnlyRunner(prepared))


def test_existing_result_validation_still_rejects_missing_lengths(prepared):
    prepared.records[prepared.output / "artifact-set.json"] = prepared.records[prepared.handoff / "artifact-set.json"]
    with pytest.raises(ValueError, match="missing member_lengths"):
        _resume(prepared, ReadOnlyRunner(prepared))


@pytest.mark.parametrize("resume", [False, True])
def test_public_cli_dispatch_preserves_json_shape(prepared, monkeypatch, resume):
    calls = []
    result = _resume(prepared, ReadOnlyRunner(prepared))

    def capture(**kwargs):
        calls.append(kwargs)
        return result

    def forbidden(**kwargs):
        pytest.fail("wrong CLI branch")

    monkeypatch.setattr(subject, "resume_legacy_msa_import" if resume else "submit_legacy_msa_import", capture)
    monkeypatch.setattr(subject, "submit_legacy_msa_import" if resume else "resume_legacy_msa_import", forbidden)
    args = [
        "--config",
        str(prepared.profile_path),
        "legacy-msa-import",
        "--profile",
        "example-cluster",
        "--handoff-root",
        str(prepared.handoff),
        "--output-dir",
        str(prepared.output),
    ]
    if resume:
        args += ["--resume-job-id", _JOB]
    response = CliRunner().invoke(cli, args)
    assert response.exit_code == 0, response.output
    assert json.loads(response.output) == json.loads(result.render_json())
    assert calls[0].get("job_id") == (_JOB if resume else None)
