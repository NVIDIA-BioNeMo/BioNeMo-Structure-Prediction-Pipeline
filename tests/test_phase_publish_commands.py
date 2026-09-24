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

"""Tests for the phase publish commands (F12 seam-transport publish boundary).

Tests the control→runtime boundary classification:
- CLUSTER-BOUND: publish-preprocessing and publish-folding route through
  SSH/sbatch transport to a Slurm job + runtime container (never local
  subprocess).
- LOCAL: derive-seam-parquets retains local subprocess delegation (regression
  guard against rerouting).
- Boundary classification is asserted testably.

Mock only external boundaries (transport, scheduler) per repo conventions.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.phase import (
    FoldingPhasePlan,
    FoldingPhasePlanPayload,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    BundledMemberVerification,
    MsaArtifactSetManifest,
    MsaChunkManifestReference,
    VerifiedLocalBundledArtifactLocation,
    msa_artifact_set_id,
    verified_local_bundled_artifact_location_id,
)
from bspp.orchestration.control.cli import cli as control_cli
from bspp.orchestration.control.phase_materialization import materialize_phase
from bspp.orchestration.control.transport import CommandResult
from tests.test_phase_materialization import FIXED_RUN_ID, FIXED_TIME, _fixture

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_s5cmd(tmp_path: Path) -> Path:
    """Create a fake s5cmd executable that exits 0 regardless of arguments."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    s5cmd = bin_dir / "s5cmd"
    s5cmd.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(0)\n")
    s5cmd.chmod(0o755)
    return bin_dir


def _set_dummy_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set dummy AWS env vars so load_credentials_from_env() succeeds."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:0")


def _write_publish_profiles(tmp_path: Path) -> Path:
    """Write a Cluster Profile with control_cpu resources and credential mounts."""
    cluster = {
        "owner": "tester",
        "transport": "local-slurm",
        "project_root": str(tmp_path / "project"),
        "output_root": str(tmp_path / "output"),
        "staging_root": str(tmp_path / "staging"),
        "orchestration_repo": str(tmp_path / "orchestration"),
        "image": str(tmp_path / "images" / "bspp.sqsh"),
        "account": "user-account",
        "postprocessing_credential_mounts": {
            "aws_shared_credentials_file": str(tmp_path / "aws" / "credentials"),
            "aws_config_file": str(tmp_path / "aws" / "config"),
        },
    }
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump({"clusters": {"example-cluster": cluster}}, sort_keys=False))
    return path


def _materialize_preprocessing_authority(
    tmp_path: Path,
    *,
    transport: str = "publish-to-s3",
    s3_prefix: str | None = "s3://bucket/msa",
) -> tuple[Path, str, Path, Path]:
    """Materialize a preprocessing phase and return (authority_root, phase_run_id, handoff_dir, profile_path)."""
    fixture = _fixture(tmp_path)
    # Modify the fixture profile: use a writable staging_root and add
    # postprocessing_credential_mounts so the same profile can be used for
    # publish without a profile-name mismatch.
    profile_mapping = yaml.safe_load(fixture.profile_path.read_text())
    cluster = profile_mapping["clusters"]["example-cluster"]
    cluster["postprocessing_credential_mounts"] = {
        "aws_shared_credentials_file": str(tmp_path / "aws" / "credentials"),
        "aws_config_file": str(tmp_path / "aws" / "config"),
    }
    # Use a writable staging_root so publish can create directories.
    cluster["paths"]["staging_root"] = str(tmp_path / "staging")
    fixture.profile_path.write_text(yaml.safe_dump(profile_mapping, sort_keys=True))
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    runspec_path = authority_root / FIXED_RUN_ID / "attempts" / "attempt-0001" / "phase-runspec.json"
    mapping = json.loads(runspec_path.read_text())
    mapping["payload"]["transport"] = transport
    if s3_prefix is not None:
        mapping["payload"]["s3_publish_prefix"] = s3_prefix
    runspec_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    plan_path = authority_root / FIXED_RUN_ID / "phase-plan.json"
    plan_mapping = json.loads(plan_path.read_text())
    plan_mapping["payload"]["transport"] = transport
    if s3_prefix is not None:
        plan_mapping["payload"]["s3_publish_prefix"] = s3_prefix
    else:
        plan_mapping["payload"].pop("s3_publish_prefix", None)
    plan_path.write_text(json.dumps(plan_mapping, indent=2, sort_keys=True) + "\n")
    from bspp.orchestration.contract.phase import phase_plan_from_mapping

    plan = phase_plan_from_mapping(plan_mapping)
    new_plan_digest = plan.digest
    mapping["phase_plan_digest"] = new_plan_digest
    runspec_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    from bspp.orchestration.contract.phase import phase_runspec_from_mapping

    runspec = phase_runspec_from_mapping(mapping)
    new_digest = runspec.digest
    phase_run_path = authority_root / FIXED_RUN_ID / "phase-run.json"
    phase_run_mapping = json.loads(phase_run_path.read_text())
    phase_run_mapping["attempts"][0]["phase_runspec_digest"] = new_digest
    phase_run_mapping["phase_plan_digest"] = new_plan_digest
    phase_run_path.write_text(json.dumps(phase_run_mapping, indent=2, sort_keys=True) + "\n")
    event_path = authority_root / FIXED_RUN_ID / "events" / "000001-phase-materialized.json"
    event_mapping = json.loads(event_path.read_text())
    event_mapping["payload"]["phase_run"] = phase_run_mapping
    event_mapping["payload"]["phase_runspec"] = mapping
    event_path.write_text(json.dumps(event_mapping, indent=2, sort_keys=True) + "\n")
    handoff_dir = tmp_path / "handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = (tmp_path / "bundle.tar.lz4").absolute()
    bundle_path.write_bytes(b"x" * 64)
    tar_path = (tmp_path / "bundle.tar").absolute()
    tar_path.write_bytes(b"y" * 100)
    lz4_sha256 = hashlib.sha256(b"x" * 64).hexdigest()
    tar_sha256 = "c" * 64
    member_sha256 = "b" * 64
    member_name = "AFDB_AF-0000000000000001.a3m"
    artifact_set_id = "sha256:" + "a" * 64
    member = BundledMemberVerification(
        logical_path=f"a3ms/{member_name}",
        member_name=member_name,
        raw_member_name=f"./{member_name}",
        size_bytes=100,
        sha256=member_sha256,
    )
    bundle_uri = bundle_path.as_uri()
    raw_tar_members = (".", f"./{member_name}")
    location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=bundle_uri,
        tar_size_bytes=100,
        tar_sha256=tar_sha256,
        lz4_size_bytes=64,
        lz4_sha256=lz4_sha256,
        raw_tar_members=raw_tar_members,
        members=(member,),
    )
    location = VerifiedLocalBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=bundle_uri,
        tar_size_bytes=100,
        tar_sha256=tar_sha256,
        lz4_size_bytes=64,
        lz4_sha256=lz4_sha256,
        raw_tar_members=raw_tar_members,
        members=(member,),
        verified_at="2026-09-01T00:00:00.000000Z",
    )
    (handoff_dir / "artifact-location.json").write_text(
        json.dumps(location.to_mapping(), indent=2, sort_keys=True) + "\n"
    )
    return authority_root, FIXED_RUN_ID, handoff_dir, fixture.profile_path


# ---------------------------------------------------------------------------
# Folding authority fixture helpers
# ---------------------------------------------------------------------------

_FOLDING_RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
_FOLDING_TIME = datetime(2026, 9, 11, 12, 0, 0, tzinfo=UTC)
_FOLDING_MEMBER_NAME = "AFDB_AF-0000000000000001.a3m"
_FOLDING_MEMBER_PATH = f"a3ms/{_FOLDING_MEMBER_NAME}"
_FOLDING_RUNTIME_IMAGE = "registry/bspp-runtime:latest"


def _folding_manifest() -> MsaArtifactSetManifest:
    chunk = MsaChunkManifestReference(
        chunk_name="folding_tranche00_00001.fa",
        logical_path="chunks/folding_tranche00_00001.json",
        sha256="f" * 64,
        member_count=1,
        logical_bytes=1,
    )
    return MsaArtifactSetManifest(
        artifact_set_id=msa_artifact_set_id((chunk,), 1, 1, member_lengths=(1,)),
        chunks=(chunk,),
        member_count=1,
        logical_bytes=1,
        member_lengths=(1,),
    )


def _materialize_folding_authority(
    tmp_path: Path,
    *,
    transport: str = "publish-to-s3",
    s3_prediction_prefix: str | None = "s3://bucket/predictions",
) -> tuple[Path, str, Path]:
    """Materialize a folding phase and return (authority_root, phase_run_id, profile_path)."""
    manifest = _folding_manifest()
    artifact_set_id = manifest.artifact_set_id
    bundle_path = tmp_path / "msa-set" / "msa-set.tar.lz4"
    tar_path = tmp_path / "msa-set" / "msa-set.tar"
    bundle_path.parent.mkdir(parents=True)
    bundle_bytes = b"bspp-fixture-lz4-bundle\n"
    tar_bytes = b"bspp-fixture-tar\n"
    bundle_path.write_bytes(bundle_bytes)
    tar_path.write_bytes(tar_bytes)
    lz4_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    tar_sha256 = hashlib.sha256(tar_bytes).hexdigest()
    members = (
        BundledMemberVerification(
            logical_path=_FOLDING_MEMBER_PATH,
            member_name=_FOLDING_MEMBER_NAME,
            raw_member_name=_FOLDING_MEMBER_NAME,
            size_bytes=1,
            sha256="b" * 64,
        ),
    )
    location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=Path(bundle_path).as_uri(),
        tar_size_bytes=len(tar_bytes),
        tar_sha256=tar_sha256,
        lz4_size_bytes=len(bundle_bytes),
        lz4_sha256=lz4_sha256,
        raw_tar_members=(_FOLDING_MEMBER_NAME,),
        members=members,
    )
    location = VerifiedLocalBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=Path(bundle_path).as_uri(),
        tar_size_bytes=len(tar_bytes),
        tar_sha256=tar_sha256,
        lz4_size_bytes=len(bundle_bytes),
        lz4_sha256=lz4_sha256,
        raw_tar_members=(_FOLDING_MEMBER_NAME,),
        members=members,
        verified_at="2026-01-01T00:00:00Z",
    )
    msa_set = MsaSetConsumption(
        artifact_set_id=artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=(_FOLDING_MEMBER_PATH,),
        requires_paired_query_header=True,
    )
    plan = FoldingPhasePlan(
        target_cluster="example-cluster",
        input_location=location,
        payload=FoldingPhasePlanPayload(
            msa_set=msa_set,
            backend="openfold-cli",
            msa_set_manifest=manifest,
            transport=transport,
            s3_prediction_prefix=s3_prediction_prefix,
        ),
    )
    plan_path = tmp_path / "folding-phase-plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan.to_mapping(), sort_keys=True))
    profile_path = _write_folding_publish_profile(tmp_path)
    authority_root = tmp_path / "folding-authority"
    result = materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: _FOLDING_TIME,
        phase_run_id_factory=lambda: _FOLDING_RUN_ID,
    )
    return authority_root, result.phase_run_id, profile_path


def _write_folding_publish_profile(tmp_path: Path) -> Path:
    """Write a folding Cluster Profile with control_cpu and credential mounts."""
    cluster = {
        "owner": "tester",
        "transport": "local-slurm",
        "project_root": str(tmp_path / "project"),
        "output_root": str(tmp_path / "output"),
        "staging_root": str(tmp_path / "staging"),
        "orchestration_repo": str(tmp_path / "orchestration"),
        "image": _FOLDING_RUNTIME_IMAGE,
        "account": "user-account",
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
        "postprocessing_credential_mounts": {
            "aws_shared_credentials_file": str(tmp_path / "aws" / "credentials"),
            "aws_config_file": str(tmp_path / "aws" / "config"),
        },
    }
    path = tmp_path / "folding-profiles.yaml"
    path.write_text(yaml.safe_dump({"clusters": {"example-cluster": cluster}}, sort_keys=True))
    return path


def _make_prediction_bundle_files(tmp_path: Path) -> tuple[Path, Path, str, int, str]:
    """Create a prediction bundle file and its JSON record.

    Returns (bundles_json, local_paths_json, sha256, size, bundle_name).
    """
    bundle_name = "bspp_260901_1200_a00001.tar.lz4"
    bundle_bytes = b"prediction-bundle-fixture\n"
    bundle_path = tmp_path / bundle_name
    bundle_path.write_bytes(bundle_bytes)
    actual_sha = hashlib.sha256(bundle_bytes).hexdigest()
    size = len(bundle_bytes)
    bundle_record = {
        "schema_version": 1,
        "bundle_name": bundle_name,
        "member_ids": ["AF-0000000000000001"],
        "member_count": 1,
        "sha256": actual_sha,
        "size_bytes": size,
        "created_at": "2026-09-01T12:00:00.000000Z",
    }
    bundles_json = tmp_path / "bundles.json"
    bundles_json.write_text(json.dumps([bundle_record], indent=2, sort_keys=True) + "\n")
    local_paths_json = tmp_path / "local-paths.json"
    local_paths_json.write_text(json.dumps([str(bundle_path)], indent=2, sort_keys=True) + "\n")
    return bundles_json, local_paths_json, actual_sha, size, bundle_name


# ---------------------------------------------------------------------------
# Mock transport runner for cluster-side publish tests
# ---------------------------------------------------------------------------


class PublishFetchRunner:
    """Mock runner that returns fake sbatch/squeue/sacct and creates evidence on fetch."""

    def __init__(
        self,
        responses: list[CommandResult],
        evidence_contents: dict[str, str],
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.responses = responses
        self.evidence_contents = evidence_contents

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        if argv and argv[0] == "cp":
            source = Path(argv[1])
            dest = Path(argv[2])
            filename = source.name
            if filename in self.evidence_contents:
                dest.write_text(self.evidence_contents[filename])
                return CommandResult(argv=argv, returncode=0, stdout="", stderr="")
            return CommandResult(argv=argv, returncode=1, stdout="", stderr=f"file not found: {source}")
        response = self.responses.pop(0)
        return CommandResult(argv=argv, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr)


def _completed_job_responses(job_id: str = "4242") -> list[CommandResult]:
    """Return the sequence of CommandResults for a successful sbatch → terminal cycle."""
    return [
        CommandResult(argv=(), returncode=0, stdout=f"{job_id}\n", stderr=""),
        CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
        CommandResult(
            argv=(),
            returncode=0,
            stdout=json.dumps(
                {
                    "jobs": [
                        {
                            "job_id_raw": job_id,
                            "state": {"current": "COMPLETED"},
                            "exit_code": {"return_code": 0, "signal": 0},
                        }
                    ]
                }
            ),
            stderr="",
        ),
    ]


# ---------------------------------------------------------------------------
# Preprocessing publish tests
# ---------------------------------------------------------------------------


def test_publish_preprocessing_cluster_side_submission(
    tmp_path: Path,
) -> None:
    """publish-preprocessing renders a correct sbatch script and routes through sbatch transport (F12).

    This test exercises the REAL render path (``_render_publish_script``) and
    asserts the rendered sbatch script's executable, mounts, and structure.
    The scheduler and transport are mocked for submission/polling/fetch, but
    the rendered script is never fabricated — it is read from disk and asserted.
    """
    authority_root, phase_run_id, handoff_dir, profile_path = _materialize_preprocessing_authority(tmp_path)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    remote_artifact = json.dumps({"artifact_location_id": "sha256:" + "a" * 64}, sort_keys=True)
    remote_msa = json.dumps(
        {"phase_run_id": phase_run_id, "attempt_id": "attempt-0001", "object_key": "s3://bucket/msa/test.tar.lz4"},
        sort_keys=True,
    )
    runner = PublishFetchRunner(
        _completed_job_responses(),
        {
            "artifact-location-remote.json": remote_artifact,
            "msa-set-upload-evidence.json": remote_msa,
        },
    )

    from bspp.orchestration.control.phase_publish import publish_preprocessing_phase
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    result = publish_preprocessing_phase(
        phase_run_id,
        authority_root=authority_root,
        handoff_path=handoff_dir,
        evidence_dir=evidence_dir,
        cluster_profile=profile,
        runner=runner,
        poll_interval_seconds=0.01,
    )
    assert result.job_id == "4242"
    assert "artifact-location-remote.json" in result.evidence_files
    assert "msa-set-upload-evidence.json" in result.evidence_files
    assert (evidence_dir / "artifact-location-remote.json").is_file()
    assert (evidence_dir / "msa-set-upload-evidence.json").is_file()

    # --- Render-and-assert: read the actual sbatch script from disk ---
    sbatch_files = list(evidence_dir.glob("bspp_publish_*.sbatch"))
    assert len(sbatch_files) == 1, f"expected exactly one sbatch script, found {len(sbatch_files)}"
    script_text = sbatch_files[0].read_text()

    # (a) Executable line must be the correct runtime entry point
    assert "bspp-orchestration-runtime phase publish-preprocessing" in script_text, (
        f"expected 'bspp-orchestration-runtime phase publish-preprocessing' in script; got:\n{script_text}"
    )
    assert "bspp-orchestration runtime" not in script_text, (
        "the old wrong command 'bspp-orchestration runtime ...' must not appear"
    )

    # (b) Mounts: evidence dir must be writable (no :ro); inputs/credentials/repo must be :ro
    mounts_line = None
    for line in script_text.splitlines():
        if "--container-mounts=" in line:
            mounts_line = line
            break
    assert mounts_line is not None, f"could not find --container-mounts line in script:\n{script_text}"
    # Strip trailing line-continuation backslash and whitespace.
    mounts_raw = mounts_line.split("--container-mounts=", 1)[1].rstrip(" \\")
    # If shlex.quote wrapped the value, strip the single quotes.
    mounts_raw = mounts_raw.strip().strip("'")
    mount_segments = [seg.strip() for seg in mounts_raw.split(",")]
    mounts_arg = mounts_raw
    # Evidence dir should be writable — no :ro suffix
    evidence_segs = [seg for seg in mount_segments if "/evidence" in seg and "bspp-publish" in seg]
    assert evidence_segs, f"could not find evidence dir mount in {mounts_arg}"
    for seg in evidence_segs:
        assert not seg.endswith(":ro"), f"evidence dir must be writable (no :ro): {seg}"
    # Credential mounts must be :ro
    cred_segs = [
        seg for seg in mount_segments if "/workspace/bspp-aws/credentials" in seg or "/workspace/bspp-aws/config" in seg
    ]
    assert cred_segs, f"could not find credential mounts in {mounts_arg}"
    for seg in cred_segs:
        assert seg.endswith(":ro"), f"credential mount must be read-only (:ro): {seg}"
    # Input JSON mount must be :ro
    input_segs = [seg for seg in mount_segments if "/input.json" in seg]
    assert input_segs, f"could not find input.json mount in {mounts_arg}"
    for seg in input_segs:
        assert seg.endswith(":ro"), f"input JSON mount must be read-only (:ro): {seg}"
    # Orchestration repo mount must be :ro
    repo_segs = [seg for seg in mount_segments if "/workspace/bspp-orchestration" in seg]
    assert repo_segs, f"could not find orchestration repo mount in {mounts_arg}"
    for seg in repo_segs:
        assert seg.endswith(":ro"), f"orchestration repo mount must be read-only (:ro): {seg}"
    # Bundle parent mounts must be :ro (if any)
    known = evidence_segs + cred_segs + input_segs + repo_segs
    bundle_segs = [seg for seg in mount_segments if seg not in known]
    for seg in bundle_segs:
        assert seg.endswith(":ro"), f"bundle parent mount must be read-only (:ro): {seg}"

    # The sbatch command was issued (not a local subprocess)
    sbatch_calls = [c for c in runner.calls if (c and c[0] in ("sbatch",)) or (len(c) > 0 and "sbatch" in c)]
    assert len(sbatch_calls) > 0, "expected at least one sbatch submission"
    # No sys.executable subprocess was used
    python_calls = [c for c in runner.calls if c and c[0] in ("python", "python3") and "-m" in c]
    assert len(python_calls) == 0, "publish must not invoke the runtime CLI via local python subprocess"
    # Validate authority is not corrupted
    from bspp.orchestration.control.phase_authority import PhaseAuthorityStore

    PhaseAuthorityStore(authority_root).validate(phase_run_id)


def test_publish_preprocessing_rejects_local_transport(
    tmp_path: Path,
) -> None:
    """publish-preprocessing on a RunSpec with transport='local' raises ClickException."""
    authority_root, phase_run_id, handoff_dir, profile_path = _materialize_preprocessing_authority(
        tmp_path, transport="local", s3_prefix=None
    )
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(
        control_cli,
        [
            "--config",
            str(profile_path),
            "phase",
            "publish-preprocessing",
            phase_run_id,
            "--authority-root",
            str(authority_root),
            "--handoff-path",
            str(handoff_dir),
            "--evidence-dir",
            str(evidence_dir),
            "--profile",
            "example-cluster",
        ],
    )
    assert result.exit_code != 0
    assert "local" in result.output or "transport" in result.output


def test_publish_preprocessing_on_folding_authority_raises(tmp_path: Path) -> None:
    """publish-preprocessing on a folding authority raises ClickException (criterion #20)."""
    authority_root, phase_run_id, _profile_path = _materialize_folding_authority(tmp_path)
    profile_path = _write_folding_publish_profile(tmp_path)
    handoff_dir = tmp_path / "handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    (handoff_dir / "artifact-location.json").write_text("{}")
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(
        control_cli,
        [
            "--config",
            str(profile_path),
            "phase",
            "publish-preprocessing",
            phase_run_id,
            "--authority-root",
            str(authority_root),
            "--handoff-path",
            str(handoff_dir),
            "--evidence-dir",
            str(evidence_dir),
            "--profile",
            "example-cluster",
        ],
    )
    assert result.exit_code != 0
    assert "not a preprocessing phase run" in result.output


# ---------------------------------------------------------------------------
# Folding publish tests
# ---------------------------------------------------------------------------


def test_publish_folding_cluster_side_submission(
    tmp_path: Path,
) -> None:
    """publish-folding renders a correct sbatch script and routes through sbatch transport (F12).

    This test exercises the REAL render path (``_render_publish_script``) and
    asserts the rendered sbatch script's executable, mounts, and structure.
    The scheduler and transport are mocked for submission/polling/fetch, but
    the rendered script is never fabricated — it is read from disk and asserted.
    """
    authority_root, phase_run_id, profile_path = _materialize_folding_authority(tmp_path)
    bundles_json, local_paths_json, actual_sha, _size, bundle_name = _make_prediction_bundle_files(tmp_path)
    evidence_dir = tmp_path / "folding-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    remote_evidence = json.dumps(
        [{"bundle_name": bundle_name, "sha256": actual_sha, "operator_attested": True}],
        sort_keys=True,
    )
    runner = PublishFetchRunner(
        _completed_job_responses(),
        {"prediction-bundle-upload-evidence.json": remote_evidence},
    )

    from bspp.orchestration.control.phase_publish import publish_folding_phase
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    result = publish_folding_phase(
        phase_run_id,
        authority_root=authority_root,
        bundles_path=bundles_json,
        local_paths_json=local_paths_json,
        evidence_dir=evidence_dir,
        cluster_profile=profile,
        runner=runner,
        poll_interval_seconds=0.01,
    )
    assert result.job_id == "4242"
    assert "prediction-bundle-upload-evidence.json" in result.evidence_files
    assert (evidence_dir / "prediction-bundle-upload-evidence.json").is_file()

    # --- Render-and-assert: read the actual sbatch script from disk ---
    sbatch_files = list(evidence_dir.glob("bspp_publish_*.sbatch"))
    assert len(sbatch_files) == 1, f"expected exactly one sbatch script, found {len(sbatch_files)}"
    script_text = sbatch_files[0].read_text()

    # (a) Executable line must be the correct runtime entry point
    assert "bspp-orchestration-runtime phase publish-folding" in script_text, (
        f"expected 'bspp-orchestration-runtime phase publish-folding' in script; got:\n{script_text}"
    )
    assert "bspp-orchestration runtime" not in script_text, (
        "the old wrong command 'bspp-orchestration runtime ...' must not appear"
    )

    # (b) Mounts: evidence dir must be writable (no :ro); inputs/credentials/repo must be :ro
    mounts_line = None
    for line in script_text.splitlines():
        if "--container-mounts=" in line:
            mounts_line = line
            break
    assert mounts_line is not None, f"could not find --container-mounts line in script:\n{script_text}"
    # Strip trailing line-continuation backslash and whitespace.
    mounts_arg = mounts_line.split("--container-mounts=", 1)[1].rstrip(" \\")
    # If shlex.quote wrapped the value, strip the single quotes.
    mounts_arg = mounts_arg.strip().strip("'")
    mount_segments = [seg.strip() for seg in mounts_arg.split(",")]
    # Evidence dir should be writable — no :ro suffix
    evidence_segs = [seg for seg in mount_segments if "/evidence" in seg and "bspp-publish" in seg]
    assert evidence_segs, f"could not find evidence dir mount in {mounts_arg}"
    for seg in evidence_segs:
        assert not seg.endswith(":ro"), f"evidence dir must be writable (no :ro): {seg}"
    # Credential mounts must be :ro
    cred_segs = [
        seg for seg in mount_segments if "/workspace/bspp-aws/credentials" in seg or "/workspace/bspp-aws/config" in seg
    ]
    assert cred_segs, f"could not find credential mounts in {mounts_arg}"
    for seg in cred_segs:
        assert seg.endswith(":ro"), f"credential mount must be read-only (:ro): {seg}"
    # Input JSON mount must be :ro
    input_segs = [seg for seg in mount_segments if "/input.json" in seg]
    assert input_segs, f"could not find input.json mount in {mounts_arg}"
    for seg in input_segs:
        assert seg.endswith(":ro"), f"input JSON mount must be read-only (:ro): {seg}"
    # Orchestration repo mount must be :ro
    repo_segs = [seg for seg in mount_segments if "/workspace/bspp-orchestration" in seg]
    assert repo_segs, f"could not find orchestration repo mount in {mounts_arg}"
    for seg in repo_segs:
        assert seg.endswith(":ro"), f"orchestration repo mount must be read-only (:ro): {seg}"
    # Bundle parent mounts must be :ro (if any)
    known = evidence_segs + cred_segs + input_segs + repo_segs
    bundle_segs = [seg for seg in mount_segments if seg not in known]
    for seg in bundle_segs:
        assert seg.endswith(":ro"), f"bundle parent mount must be read-only (:ro): {seg}"

    # The sbatch command was issued (not a local subprocess)
    sbatch_calls = [c for c in runner.calls if (c and c[0] in ("sbatch",)) or (len(c) > 0 and "sbatch" in c)]
    assert len(sbatch_calls) > 0, "expected at least one sbatch submission"
    # No sys.executable subprocess was used
    python_calls = [c for c in runner.calls if c and c[0] in ("python", "python3") and "-m" in c]
    assert len(python_calls) == 0, "publish must not invoke the runtime CLI via local python subprocess"
    # Validate authority is not corrupted
    from bspp.orchestration.control.phase_authority import PhaseAuthorityStore

    PhaseAuthorityStore(authority_root).validate(phase_run_id)


def test_publish_folding_on_preprocessing_authority_raises(
    tmp_path: Path,
) -> None:
    """publish-folding on a preprocessing authority raises ClickException (criterion #20)."""
    authority_root, phase_run_id, _handoff_dir, _profile_path = _materialize_preprocessing_authority(tmp_path)
    profile_path = _write_publish_profiles(tmp_path)
    bundles_json, local_paths_json, _sha, _size, _name = _make_prediction_bundle_files(tmp_path)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(
        control_cli,
        [
            "--config",
            str(profile_path),
            "phase",
            "publish-folding",
            phase_run_id,
            "--authority-root",
            str(authority_root),
            "--bundles",
            str(bundles_json),
            "--local-paths",
            str(local_paths_json),
            "--evidence-dir",
            str(evidence_dir),
            "--profile",
            "example-cluster",
        ],
    )
    assert result.exit_code != 0
    assert "not a folding phase run" in result.output


def test_publish_folding_refuses_missing_prediction_prefix(tmp_path: Path) -> None:
    """publish-folding refuses a folding RunSpec with no s3_prediction_prefix (deferred enforcement)."""
    authority_root, phase_run_id, profile_path = _materialize_folding_authority(tmp_path, s3_prediction_prefix=None)
    bundles_json, local_paths_json, _sha, _size, _name = _make_prediction_bundle_files(tmp_path)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(
        control_cli,
        [
            "--config",
            str(profile_path),
            "phase",
            "publish-folding",
            phase_run_id,
            "--authority-root",
            str(authority_root),
            "--bundles",
            str(bundles_json),
            "--local-paths",
            str(local_paths_json),
            "--evidence-dir",
            str(evidence_dir),
            "--profile",
            "example-cluster",
        ],
    )
    assert result.exit_code != 0
    assert "s3_prediction_prefix" in result.output


def test_publish_folding_refuses_local_transport(tmp_path: Path) -> None:
    """publish-folding refuses a folding RunSpec whose sealed transport is not publish-to-s3."""
    authority_root, phase_run_id, profile_path = _materialize_folding_authority(
        tmp_path, transport="local", s3_prediction_prefix=None
    )
    bundles_json, local_paths_json, _sha, _size, _name = _make_prediction_bundle_files(tmp_path)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(
        control_cli,
        [
            "--config",
            str(profile_path),
            "phase",
            "publish-folding",
            phase_run_id,
            "--authority-root",
            str(authority_root),
            "--bundles",
            str(bundles_json),
            "--local-paths",
            str(local_paths_json),
            "--evidence-dir",
            str(evidence_dir),
            "--profile",
            "example-cluster",
        ],
    )
    assert result.exit_code != 0
    assert "not 'publish-to-s3'" in result.output


def test_publish_folding_failed_job_raises(tmp_path: Path) -> None:
    """publish-folding raises when the Slurm job fails (not COMPLETED 0:0)."""
    authority_root, phase_run_id, profile_path = _materialize_folding_authority(tmp_path)
    bundles_json, local_paths_json, _sha, _size, _name = _make_prediction_bundle_files(tmp_path)
    evidence_dir = tmp_path / "folding-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    failed_responses = [
        CommandResult(argv=(), returncode=0, stdout="9999\n", stderr=""),
        CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
        CommandResult(
            argv=(),
            returncode=0,
            stdout=json.dumps(
                {
                    "jobs": [
                        {
                            "job_id_raw": "9999",
                            "state": {"current": "FAILED"},
                            "exit_code": {"return_code": 1, "signal": 0},
                        }
                    ]
                }
            ),
            stderr="",
        ),
    ]
    runner = PublishFetchRunner(failed_responses, {})

    from bspp.orchestration.control.phase_publish import publish_folding_phase
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    with pytest.raises(ValueError, match="FAILED"):
        publish_folding_phase(
            phase_run_id,
            authority_root=authority_root,
            bundles_path=bundles_json,
            local_paths_json=local_paths_json,
            evidence_dir=evidence_dir,
            cluster_profile=profile,
            runner=runner,
            poll_interval_seconds=0.01,
        )


# ---------------------------------------------------------------------------
# Runtime CLI direct tests (unchanged — the runtime CLI is not modified)
# ---------------------------------------------------------------------------


def test_publish_folding_bundles_option_overrides_input_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime CLI --bundles option overrides payload bundles from --input-json (criterion #19)."""
    _authority_root, phase_run_id, _profile_path = _materialize_folding_authority(tmp_path)
    bundles_json, local_paths_json, actual_sha, _size, bundle_name = _make_prediction_bundle_files(tmp_path)
    evidence_dir = tmp_path / "folding-evidence-override"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    bin_dir = _make_fake_s5cmd(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    _set_dummy_aws_env(monkeypatch)

    bad_bundle_record = {
        "schema_version": 1,
        "bundle_name": "bspp_260901_1200_a00002.tar.lz4",
        "member_ids": ["AF-0000000000000002"],
        "member_count": 1,
        "sha256": "e" * 64,
        "size_bytes": 999,
        "created_at": "2026-09-01T12:00:00.000000Z",
    }
    input_payload = {
        "bundles": [bad_bundle_record],
        "local_paths": json.loads(local_paths_json.read_text()),
        "s3_prefix": "s3://bucket/predictions",
        "phase_run_id": phase_run_id,
        "attempt_id": "attempt-0001",
    }
    input_json = evidence_dir / "input.json"
    input_json.write_text(json.dumps(input_payload, indent=2, sort_keys=True) + "\n")

    from bspp.orchestration.runtime.cli import cli as runtime_cli

    runner = CliRunner()
    result = runner.invoke(
        runtime_cli,
        [
            "phase",
            "publish-folding",
            "--input-json",
            str(input_json),
            "--evidence-dir",
            str(evidence_dir),
            "--bundles",
            str(bundles_json),
        ],
    )
    assert result.exit_code == 0, f"stdout={result.stdout}, exception={result.exception}"
    evidence_data = json.loads((evidence_dir / "prediction-bundle-upload-evidence.json").read_text())
    assert evidence_data[0]["bundle_name"] == bundle_name
    assert evidence_data[0]["sha256"] == actual_sha


def test_no_test_hook_in_production_runtime_cli() -> None:
    """Grep the runtime CLI and seam transport source for test-only transfer bypass — none found (criterion #19)."""
    import re

    base = (
        Path(__file__).resolve().parent.parent
        / "packages"
        / "orchestration-runtime"
        / "src"
        / "bspp"
        / "orchestration"
        / "runtime"
    )
    sources = {
        "cli.py": base / "cli.py",
        "seam_transport.py": base / "folding" / "seam_transport.py",
    }
    forbidden_patterns = [
        r"transfer_fixture",
        r"BSPP_TEST_TRANSFER",
        r"test.*transfer.*bypass",
        r"env.*var.*transfer",
    ]
    for label, path in sources.items():
        source = path.read_text()
        for pattern in forbidden_patterns:
            assert re.search(pattern, source) is None, f"Found forbidden test hook pattern {pattern!r} in {label}"


# ---------------------------------------------------------------------------
# Boundary classification regression tests
# ---------------------------------------------------------------------------


def test_derive_seam_parquets_still_local_subprocess() -> None:
    """derive-seam-parquets retains local subprocess delegation (regression: not rerouted).

    The derive-seam-parquets command reads already-fetched evidence files on the
    workstation and writes parquets — it does NOT read cluster paths and does
    NOT need S3 access.  Local subprocess.run is the correct boundary.
    See ADR-0075 § "Boundary classification for the publish family".
    """
    cli_path = (
        Path(__file__).resolve().parent.parent
        / "packages"
        / "orchestration-control"
        / "src"
        / "bspp"
        / "orchestration"
        / "control"
        / "cli.py"
    )
    source = cli_path.read_text()
    # Find the derive-seam-parquets command section
    start = source.index("def phase_derive_seam_parquets_cmd")
    end = source.index('\n\n@cli.group("runtime")', start)
    derive_source = source[start:end]
    assert "subprocess.run" in derive_source, "derive-seam-parquets must still use local subprocess.run"
    assert "sys.executable" in derive_source, "derive-seam-parquets must still delegate via sys.executable"
    assert "--profile" not in derive_source, "derive-seam-parquets must not take --profile (it is a local transform)"


def test_publish_preprocessing_uses_transport_not_subprocess() -> None:
    """publish-preprocessing does NOT use local subprocess.run (regression: rerouted to cluster-side).

    See ADR-0075 § "Boundary classification for the publish family".
    """
    cli_path = (
        Path(__file__).resolve().parent.parent
        / "packages"
        / "orchestration-control"
        / "src"
        / "bspp"
        / "orchestration"
        / "control"
        / "cli.py"
    )
    source = cli_path.read_text()
    start = source.index("def phase_publish_preprocessing_cmd")
    end = source.index('\n@phase_group.command("publish-folding")', start)
    pub_source = source[start:end]
    assert "subprocess.run" not in pub_source, "publish-preprocessing must not use local subprocess.run"
    assert "sys.executable" not in pub_source, "publish-preprocessing must not invoke via sys.executable"
    assert "publish_preprocessing_phase" in pub_source, (
        "publish-preprocessing must delegate to publish_preprocessing_phase"
    )


def test_publish_folding_uses_transport_not_subprocess() -> None:
    """publish-folding does NOT use local subprocess.run (regression: rerouted to cluster-side).

    See ADR-0075 § "Boundary classification for the publish family".
    """
    cli_path = (
        Path(__file__).resolve().parent.parent
        / "packages"
        / "orchestration-control"
        / "src"
        / "bspp"
        / "orchestration"
        / "control"
        / "cli.py"
    )
    source = cli_path.read_text()
    start = source.index("def phase_publish_folding_cmd")
    end = source.index('\n\n\n@phase_group.command("derive-seam-parquets")', start)
    pub_source = source[start:end]
    assert "subprocess.run" not in pub_source, "publish-folding must not use local subprocess.run"
    assert "sys.executable" not in pub_source, "publish-folding must not invoke via sys.executable"
    assert "publish_folding_phase" in pub_source, "publish-folding must delegate to publish_folding_phase"


def test_publish_requires_credential_mounts(tmp_path: Path) -> None:
    """publish-preprocessing raises when the profile lacks postprocessing_credential_mounts."""
    authority_root, phase_run_id, handoff_dir, profile_path = _materialize_preprocessing_authority(tmp_path)
    # Remove credential mounts from the profile to trigger the check.
    profile_mapping = yaml.safe_load(profile_path.read_text())
    profile_mapping["clusters"]["example-cluster"].pop("postprocessing_credential_mounts", None)
    profile_path.write_text(yaml.safe_dump(profile_mapping, sort_keys=True))
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    from bspp.orchestration.control.phase_publish import publish_preprocessing_phase
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    with pytest.raises(ValueError, match="postprocessing_credential_mounts"):
        publish_preprocessing_phase(
            phase_run_id,
            authority_root=authority_root,
            handoff_path=handoff_dir,
            evidence_dir=evidence_dir,
            cluster_profile=profile,
            runner=PublishFetchRunner(_completed_job_responses(), {}),
            poll_interval_seconds=0.01,
        )


# ---------------------------------------------------------------------------
# Profile mismatch (P1) tests
# ---------------------------------------------------------------------------


def test_publish_preprocessing_rejects_profile_mismatch(tmp_path: Path) -> None:
    """publish-preprocessing refuses a profile whose staging_root differs from the frozen snapshot."""
    authority_root, phase_run_id, handoff_dir, _profile_path = _materialize_preprocessing_authority(tmp_path)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Write a DIFFERENT profile with a different staging_root — same profile name
    # but different field values that must match the frozen snapshot.
    mismatch_profile_path = tmp_path / "mismatch-profiles.yaml"
    cluster = {
        "owner": "tester",
        "transport": "local-slurm",
        "project_root": str(tmp_path / "project"),
        "output_root": str(tmp_path / "output"),
        "staging_root": str(tmp_path / "WRONG-staging"),
        "orchestration_repo": str(tmp_path / "orchestration"),
        "image": str(tmp_path / "images" / "bspp.sqsh"),
        "account": "user-account",
        "postprocessing_credential_mounts": {
            "aws_shared_credentials_file": str(tmp_path / "aws" / "credentials"),
            "aws_config_file": str(tmp_path / "aws" / "config"),
        },
    }
    mismatch_profile_path.write_text(yaml.safe_dump({"clusters": {"example-cluster": cluster}}, sort_keys=False))

    from bspp.orchestration.control.phase_publish import publish_preprocessing_phase
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    profile = resolve_cluster_profile("example-cluster", config_path=mismatch_profile_path)
    with pytest.raises(ValueError, match="does not match the frozen RunSpec cluster snapshot"):
        publish_preprocessing_phase(
            phase_run_id,
            authority_root=authority_root,
            handoff_path=handoff_dir,
            evidence_dir=evidence_dir,
            cluster_profile=profile,
            runner=PublishFetchRunner(_completed_job_responses(), {}),
            poll_interval_seconds=0.01,
        )


def test_publish_folding_rejects_profile_mismatch(tmp_path: Path) -> None:
    """publish-folding refuses a profile whose staging_root differs from the frozen snapshot."""
    authority_root, phase_run_id, _profile_path = _materialize_folding_authority(tmp_path)
    bundles_json, local_paths_json, _sha, _size, _name = _make_prediction_bundle_files(tmp_path)
    evidence_dir = tmp_path / "folding-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Write a DIFFERENT profile with a different staging_root.
    mismatch_profile_path = tmp_path / "mismatch-folding-profiles.yaml"
    cluster = {
        "owner": "tester",
        "transport": "local-slurm",
        "project_root": str(tmp_path / "project"),
        "output_root": str(tmp_path / "output"),
        "staging_root": str(tmp_path / "WRONG-staging"),
        "orchestration_repo": str(tmp_path / "orchestration"),
        "image": _FOLDING_RUNTIME_IMAGE,
        "account": "user-account",
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
        "postprocessing_credential_mounts": {
            "aws_shared_credentials_file": str(tmp_path / "aws" / "credentials"),
            "aws_config_file": str(tmp_path / "aws" / "config"),
        },
    }
    mismatch_profile_path.write_text(yaml.safe_dump({"clusters": {"example-cluster": cluster}}, sort_keys=True))

    from bspp.orchestration.control.phase_publish import publish_folding_phase
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    profile = resolve_cluster_profile("example-cluster", config_path=mismatch_profile_path)
    with pytest.raises(ValueError, match="does not match the frozen RunSpec cluster snapshot"):
        publish_folding_phase(
            phase_run_id,
            authority_root=authority_root,
            bundles_path=bundles_json,
            local_paths_json=local_paths_json,
            evidence_dir=evidence_dir,
            cluster_profile=profile,
            runner=PublishFetchRunner(_completed_job_responses(), {}),
            poll_interval_seconds=0.01,
        )


def test_publish_preprocessing_rejects_image_mismatch(tmp_path: Path) -> None:
    """publish-preprocessing refuses a profile whose qualified preprocessing image differs from frozen runtime_image.

    For preprocessing, the frozen runtime_image is the qualified
    preprocessing_runtime.cluster_image_path (phase_attempt_materialization.py:169-180),
    NOT profile.image (paths.image).  This test mutates the correct field —
    preprocessing_runtime.cluster_image_path — and asserts the refusal fires
    BEFORE any sbatch renders.
    """
    import copy

    authority_root, phase_run_id, handoff_dir, profile_path = _materialize_preprocessing_authority(tmp_path)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Read the materialized profile and mutate ONLY the qualified preprocessing image.
    profile_mapping = yaml.safe_load(profile_path.read_text())
    mismatch_mapping = copy.deepcopy(profile_mapping)
    mismatch_mapping["clusters"]["example-cluster"]["preprocessing_runtime"]["cluster_image_path"] = (
        "/attacker/unfrozen-runtime.sqsh"
    )
    mismatch_profile_path = tmp_path / "image-mismatch-profiles.yaml"
    mismatch_profile_path.write_text(yaml.safe_dump(mismatch_mapping, sort_keys=True))

    from bspp.orchestration.control.phase_publish import publish_preprocessing_phase
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    profile = resolve_cluster_profile("example-cluster", config_path=mismatch_profile_path)
    runner = PublishFetchRunner(_completed_job_responses(), {})
    with pytest.raises(ValueError, match="does not match the frozen RunSpec cluster snapshot"):
        publish_preprocessing_phase(
            phase_run_id,
            authority_root=authority_root,
            handoff_path=handoff_dir,
            evidence_dir=evidence_dir,
            cluster_profile=profile,
            runner=runner,
            poll_interval_seconds=0.01,
        )
    # Assert no submission occurred — no sbatch calls, no sbatch file rendered.
    assert not any(c and c[0] == "sbatch" for c in runner.calls)
    assert not list(evidence_dir.glob("bspp_publish_*.sbatch"))


def test_publish_folding_rejects_image_mismatch(tmp_path: Path) -> None:
    """publish-folding refuses a profile whose image differs from the frozen runtime_image.

    The refusal fires BEFORE any sbatch renders — no submission occurs.
    """
    import copy

    authority_root, phase_run_id, profile_path = _materialize_folding_authority(tmp_path)
    bundles_json, local_paths_json, _sha, _size, _name = _make_prediction_bundle_files(tmp_path)
    evidence_dir = tmp_path / "folding-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Read the materialized profile and mutate ONLY the image field.
    profile_mapping = yaml.safe_load(profile_path.read_text())
    mismatch_mapping = copy.deepcopy(profile_mapping)
    mismatch_mapping["clusters"]["example-cluster"]["image"] = "/attacker/unfrozen-runtime.sqsh"
    mismatch_profile_path = tmp_path / "image-mismatch-folding-profiles.yaml"
    mismatch_profile_path.write_text(yaml.safe_dump(mismatch_mapping, sort_keys=True))

    from bspp.orchestration.control.phase_publish import publish_folding_phase
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    profile = resolve_cluster_profile("example-cluster", config_path=mismatch_profile_path)
    runner = PublishFetchRunner(_completed_job_responses(), {})
    with pytest.raises(ValueError, match="does not match the frozen RunSpec cluster snapshot"):
        publish_folding_phase(
            phase_run_id,
            authority_root=authority_root,
            bundles_path=bundles_json,
            local_paths_json=local_paths_json,
            evidence_dir=evidence_dir,
            cluster_profile=profile,
            runner=runner,
            poll_interval_seconds=0.01,
        )
    # Assert no submission occurred — no sbatch calls, no sbatch file rendered.
    assert not any(c and c[0] == "sbatch" for c in runner.calls)
    assert not list(evidence_dir.glob("bspp_publish_*.sbatch"))


def test_publish_preprocessing_accepts_distinct_images(tmp_path: Path) -> None:
    """publish-preprocessing accepts a legitimate profile where paths.image differs
    from the qualified preprocessing runtime image.

    The fixture profile has paths.image=/images/postprocessing.sqsh (the general
    runtime image) while preprocessing_runtime.cluster_image_path=/images/preprocessing.sqsh
    (the qualified preprocessing image).  The frozen runtime_image is the
    preprocessing image.  The profile validation must compare the correct field
    pair (preprocessing_runtime.cluster_image_path vs frozen runtime_image) and
    ACCEPT this legitimate distinct-image configuration.

    This is the case the bent fixture (which forced paths.image = cluster_image_path)
    hid in round 2.
    """
    authority_root, phase_run_id, handoff_dir, profile_path = _materialize_preprocessing_authority(tmp_path)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    from bspp.orchestration.control.phase_publish import publish_preprocessing_phase
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    # Verify the legitimate distinct-image configuration is present.
    assert profile.image != profile.preprocessing_runtime.cluster_image_path
    remote_artifact = json.dumps({"artifact_location_id": "sha256:" + "a" * 64}, sort_keys=True)
    remote_msa = json.dumps(
        {
            "phase_run_id": phase_run_id,
            "attempt_id": "attempt-0001",
            "object_key": "s3://bucket/msa/test.tar.lz4",
        },
        sort_keys=True,
    )
    runner = PublishFetchRunner(
        _completed_job_responses(),
        {
            "artifact-location-remote.json": remote_artifact,
            "msa-set-upload-evidence.json": remote_msa,
        },
    )
    result = publish_preprocessing_phase(
        phase_run_id,
        authority_root=authority_root,
        handoff_path=handoff_dir,
        evidence_dir=evidence_dir,
        cluster_profile=profile,
        runner=runner,
        poll_interval_seconds=0.01,
    )
    assert result.job_id == "4242"
    assert "artifact-location-remote.json" in result.evidence_files
    # The sbatch script was rendered — the frozen runtime_image was used.
    sbatch_files = list(evidence_dir.glob("bspp_publish_*.sbatch"))
    assert len(sbatch_files) == 1
