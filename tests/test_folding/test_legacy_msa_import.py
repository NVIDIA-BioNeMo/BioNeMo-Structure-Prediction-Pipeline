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

"""Tests for the legacy-MSA import job.

These tests exercise the runtime import worker over a tiny real A3M tar/lz4
bundle fixture and round-trip the Control submission through a mocked
transport/runner.  The legacy records and payload bytes are asserted to remain
byte-identical across every import.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.preprocessing_handoff import (
    BundledMemberVerification,
    MsaArtifactMember,
    MsaArtifactSetManifest,
    MsaChunkManifest,
    MsaChunkManifestReference,
    PreprocessingContentValidationEvidence,
    PreprocessingHandoffBundle,
    VerifiedLocalBundledArtifactLocation,
    msa_artifact_set_id,
    preprocessing_content_validation_evidence_id,
    verified_local_bundled_artifact_location_id,
)
from bspp.orchestration.control.legacy_msa_import import (
    _fetch_record,
    render_legacy_msa_import_submission,
    submit_legacy_msa_import,
)
from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.control.transport import CommandResult, RemoteSlurmTransport
from bspp.orchestration.runtime.folding.legacy_msa_import import (
    LegacyMsaImportError,
    _read_record,
    run_legacy_msa_import,
)

pytestmark = pytest.mark.skipif(shutil.which("lz4") is None, reason="lz4 binary required for real tar/lz4 fixtures")

_VERIFIED_AT = "2026-09-01T00:00:00.000000Z"
_CHUNK_NAME = "sample_tranche00_00001.fa"

_MEMBER_BYTES = {
    "AFDB_AF-0000000000000001.a3m": b"#2,3\t1,1\n>AFDB_AF-0000000000000001\nAAAAA\n",
    "AFDB_AF-0000000000000002.a3m": b"#3,4\t1,1\n>AFDB_AF-0000000000000002\nGGGGGGG\n",
}
_EXPECTED_LENGTHS = (5, 7)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_tar(entries: list[tuple[str, bytes]]) -> tuple[bytes, str, int, tuple[str, ...]]:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    raw = buffer.getvalue()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        raw_names = tuple(header.name for header in archive.getmembers())
    return raw, _sha256(raw), len(raw), raw_names


def _make_tar(members: dict[str, bytes]) -> tuple[bytes, str, int, tuple[str, ...]]:
    return _write_tar([(f"./{name}", data) for name, data in members.items()])


def _make_lz4(tar_bytes: bytes, lz4_path: Path) -> Path:
    tar_path = lz4_path.with_suffix(".tar")
    tar_path.write_bytes(tar_bytes)
    subprocess.run([shutil.which("lz4"), str(tar_path), str(lz4_path)], check=True, capture_output=True)
    return lz4_path


def _chunk_manifest(members_data: dict[str, bytes]) -> MsaChunkManifest:
    members = tuple(
        MsaArtifactMember(
            record_identity=name.removesuffix(".a3m"),
            source_ordinal=ordinal,
            source_header=f">{name.removesuffix('.a3m')}",
            logical_path=f"a3ms/{name}",
            size_bytes=len(data),
            sha256=_sha256(data),
        )
        for ordinal, (name, data) in enumerate(members_data.items())
    )
    return MsaChunkManifest(
        chunk_name=_CHUNK_NAME,
        members=members,
        member_count=len(members),
        logical_bytes=sum(member.size_bytes for member in members),
    )


def _chunk_reference(chunk_manifest: MsaChunkManifest) -> MsaChunkManifestReference:
    return MsaChunkManifestReference(
        chunk_name=chunk_manifest.chunk_name,
        logical_path=f"chunks/{chunk_manifest.chunk_name.removesuffix('.fa')}.json",
        sha256=chunk_manifest.digest,
        member_count=chunk_manifest.member_count,
        logical_bytes=chunk_manifest.logical_bytes,
    )


def _artifact_set(
    reference: MsaChunkManifestReference,
    member_count: int,
    logical_bytes: int,
    *,
    member_lengths: tuple[int, ...] | None = None,
) -> MsaArtifactSetManifest:
    return MsaArtifactSetManifest(
        artifact_set_id=msa_artifact_set_id((reference,), member_count, logical_bytes, member_lengths=member_lengths),
        chunks=(reference,),
        member_count=member_count,
        logical_bytes=logical_bytes,
        member_lengths=member_lengths,
    )


def _bundled_members(members_data: dict[str, bytes]) -> tuple[BundledMemberVerification, ...]:
    return tuple(
        BundledMemberVerification(
            logical_path=f"a3ms/{name}",
            member_name=name,
            raw_member_name=f"./{name}",
            size_bytes=len(data),
            sha256=_sha256(data),
        )
        for name, data in members_data.items()
    )


def _make_location(
    tmp_path: Path,
    artifact_set: MsaArtifactSetManifest,
    members_data: dict[str, bytes],
    tar_bytes: bytes,
    raw_names: tuple[str, ...],
    *,
    member_overrides: tuple[BundledMemberVerification, ...] | None = None,
) -> VerifiedLocalBundledArtifactLocation:
    tar_path = (tmp_path / "bundle.tar").absolute()
    tar_path.write_bytes(tar_bytes)
    lz4_path = (tmp_path / "bundle.tar.lz4").absolute()
    _make_lz4(tar_bytes, lz4_path)
    lz4_bytes = lz4_path.read_bytes()
    members = _bundled_members(members_data) if member_overrides is None else member_overrides
    bundle_uri = lz4_path.as_uri()
    location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=artifact_set.artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(lz4_path),
        bundle_uri=bundle_uri,
        tar_size_bytes=len(tar_bytes),
        tar_sha256=_sha256(tar_bytes),
        lz4_size_bytes=len(lz4_bytes),
        lz4_sha256=_sha256(lz4_bytes),
        raw_tar_members=raw_names,
        members=members,
    )
    return VerifiedLocalBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=artifact_set.artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(lz4_path),
        bundle_uri=bundle_uri,
        tar_size_bytes=len(tar_bytes),
        tar_sha256=_sha256(tar_bytes),
        lz4_size_bytes=len(lz4_bytes),
        lz4_sha256=_sha256(lz4_bytes),
        raw_tar_members=raw_names,
        members=members,
        verified_at=_VERIFIED_AT,
    )


def _make_validation(
    artifact_set: MsaArtifactSetManifest,
    location: VerifiedLocalBundledArtifactLocation,
    chunk_manifest: MsaChunkManifest,
    *,
    tar_size: int,
    tar_sha256: str,
) -> PreprocessingContentValidationEvidence:
    body: dict[str, object] = {
        "schema_version": 1,
        "phase_run_id": "phase-run-" + "a" * 32,
        "attempt_id": "attempt-0001",
        "phase_runspec_digest": "b" * 64,
        "action_id": "preprocessing-chunk-000001",
        "action_evidence_digest": "c" * 64,
        "chunk_name": chunk_manifest.chunk_name,
        "started_at": _VERIFIED_AT,
        "finished_at": _VERIFIED_AT,
        "outcome": "passed",
        "artifact_set_id": artifact_set.artifact_set_id,
        "artifact_location_id": location.artifact_location_id,
        "chunk_manifest_digest": chunk_manifest.digest,
        "artifact_set_manifest_digest": canonical_mapping_digest(artifact_set.to_mapping()),
        "member_count": artifact_set.member_count,
        "logical_bytes": artifact_set.logical_bytes,
        "lz4_command_digest": "d" * 64,
        "lz4_return_code": 0,
        "decompressed_tar_size_bytes": tar_size,
        "decompressed_tar_sha256": tar_sha256,
        "error": None,
    }
    return PreprocessingContentValidationEvidence(
        evidence_id=preprocessing_content_validation_evidence_id(body),
        phase_run_id="phase-run-" + "a" * 32,
        attempt_id="attempt-0001",
        phase_runspec_digest="b" * 64,
        action_id="preprocessing-chunk-000001",
        action_evidence_digest="c" * 64,
        chunk_name=chunk_manifest.chunk_name,
        started_at=_VERIFIED_AT,
        finished_at=_VERIFIED_AT,
        outcome="passed",
        artifact_set_id=artifact_set.artifact_set_id,
        artifact_location_id=location.artifact_location_id,
        chunk_manifest_digest=chunk_manifest.digest,
        artifact_set_manifest_digest=canonical_mapping_digest(artifact_set.to_mapping()),
        member_count=artifact_set.member_count,
        logical_bytes=artifact_set.logical_bytes,
        lz4_command_digest="d" * 64,
        lz4_return_code=0,
        decompressed_tar_size_bytes=tar_size,
        decompressed_tar_sha256=tar_sha256,
        error=None,
    )


def _write_handoff(
    handoff_root: Path,
    bundle: PreprocessingHandoffBundle,
) -> None:
    handoff_root.mkdir(parents=True, exist_ok=True)
    chunk_relative = Path(bundle.artifact_set.chunks[0].logical_path)
    (handoff_root / chunk_relative).parent.mkdir(parents=True, exist_ok=True)
    (handoff_root / chunk_relative).write_text(
        json.dumps(bundle.chunk_manifest.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (handoff_root / "artifact-set.json").write_text(
        json.dumps(bundle.artifact_set.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (handoff_root / "artifact-location.json").write_text(
        json.dumps(bundle.artifact_location.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (handoff_root / "content-validation.json").write_text(
        json.dumps(bundle.content_validation.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_legacy_handoff(
    tmp_path: Path,
    *,
    member_lengths: tuple[int, ...] | None = None,
    member_overrides: tuple[BundledMemberVerification, ...] | None = None,
    members_data: dict[str, bytes] | None = None,
) -> tuple[Path, PreprocessingHandoffBundle]:
    members_data = _MEMBER_BYTES if members_data is None else members_data
    chunk_manifest = _chunk_manifest(members_data)
    reference = _chunk_reference(chunk_manifest)
    artifact_set = _artifact_set(
        reference,
        chunk_manifest.member_count,
        chunk_manifest.logical_bytes,
        member_lengths=member_lengths,
    )
    tar_bytes, _, _, raw_names = _make_tar(members_data)
    location = _make_location(
        tmp_path,
        artifact_set,
        members_data,
        tar_bytes,
        raw_names,
        member_overrides=member_overrides,
    )
    validation = _make_validation(
        artifact_set,
        location,
        chunk_manifest,
        tar_size=len(tar_bytes),
        tar_sha256=_sha256(tar_bytes),
    )
    bundle = PreprocessingHandoffBundle(
        chunk_manifest=chunk_manifest,
        artifact_set=artifact_set,
        artifact_location=location,
        content_validation=validation,
    )
    handoff_root = tmp_path / "handoff"
    _write_handoff(handoff_root, bundle)
    return handoff_root, bundle


def _snapshot(root: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file():
            snapshot[str(candidate.relative_to(root))] = candidate.read_bytes()
    return snapshot


def test_run_legacy_msa_import_enriches_and_rebinds(tmp_path: Path) -> None:
    handoff_root, legacy = _build_legacy_handoff(tmp_path)
    output_dir = tmp_path / "enriched"
    legacy_snapshot = _snapshot(handoff_root)
    bundle_bytes = Path(legacy.artifact_location.bundle_path).read_bytes()
    tar_bytes = Path(legacy.artifact_location.tar_path).read_bytes()

    result = run_legacy_msa_import(
        handoff_root,
        output_dir,
        lz4_argv=(str(shutil.which("lz4")), "-d", "-c"),
        clock=lambda: datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert result.artifact_set_id != legacy.artifact_set.artifact_set_id
    assert result.artifact_set_id == result.enriched_manifest.artifact_set_id
    assert result.member_lengths == _EXPECTED_LENGTHS
    assert result.enriched_manifest.member_lengths == _EXPECTED_LENGTHS
    assert result.rebound_location.artifact_set_id == result.artifact_set_id
    assert result.rebound_validation.artifact_set_id == result.artifact_set_id
    assert result.rebound_validation.artifact_location_id == result.artifact_location_id

    # Legacy records and payload bytes are byte-identical before vs after.
    assert _snapshot(handoff_root) == legacy_snapshot
    assert Path(legacy.artifact_location.bundle_path).read_bytes() == bundle_bytes
    assert Path(legacy.artifact_location.tar_path).read_bytes() == tar_bytes

    # Exactly the three result records exist in the output directory.
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "artifact-location.json",
        "artifact-set.json",
        "content-validation.json",
    ]

    # The rebound location still references the unchanged payload bytes.
    assert result.rebound_location.tar_path == legacy.artifact_location.tar_path
    assert result.rebound_location.bundle_path == legacy.artifact_location.bundle_path
    assert result.rebound_location.tar_sha256 == legacy.artifact_location.tar_sha256
    assert result.rebound_location.lz4_sha256 == legacy.artifact_location.lz4_sha256
    assert result.rebound_location.members == legacy.artifact_location.members

    # Frozen from the unchanged f24611c2 serializer/fixture: byte compatibility,
    # including indentation, key order, newline and member-length identity.
    assert _sha256((output_dir / "artifact-set.json").read_bytes()) == (
        "70c40b4a598fe8d5f6617df95cb62d9b79c8dfb44bd77a43e402804f33b4b055"
    )
    for name, mapping in {
        "artifact-location.json": result.rebound_location.to_mapping(),
        "content-validation.json": result.rebound_validation.to_mapping(),
    }.items():
        assert (output_dir / name).read_bytes() == (json.dumps(mapping, indent=2, sort_keys=True) + "\n").encode()


def test_run_legacy_msa_import_rejects_already_enriched(tmp_path: Path) -> None:
    handoff_root, _ = _build_legacy_handoff(tmp_path, member_lengths=_EXPECTED_LENGTHS)
    output_dir = tmp_path / "enriched"

    with pytest.raises(LegacyMsaImportError, match="already enriched"):
        run_legacy_msa_import(handoff_root, output_dir, lz4_argv=(str(shutil.which("lz4")), "-d", "-c"))


def test_run_legacy_msa_import_rejects_corrupt_bundle(tmp_path: Path) -> None:
    handoff_root, legacy = _build_legacy_handoff(tmp_path)
    Path(legacy.artifact_location.bundle_path).write_bytes(b"corrupted bundle bytes")
    output_dir = tmp_path / "enriched"

    with pytest.raises(LegacyMsaImportError):
        run_legacy_msa_import(handoff_root, output_dir, lz4_argv=(str(shutil.which("lz4")), "-d", "-c"))


def test_run_legacy_msa_import_rejects_corrupt_tar(tmp_path: Path) -> None:
    handoff_root, legacy = _build_legacy_handoff(tmp_path)
    Path(legacy.artifact_location.tar_path).write_bytes(b"corrupted tar bytes")
    output_dir = tmp_path / "enriched"

    with pytest.raises(LegacyMsaImportError):
        run_legacy_msa_import(handoff_root, output_dir, lz4_argv=(str(shutil.which("lz4")), "-d", "-c"))


def test_run_legacy_msa_import_rejects_corrupt_member(tmp_path: Path) -> None:
    members_data = _MEMBER_BYTES
    corrupted = tuple(
        BundledMemberVerification(
            logical_path=f"a3ms/{name}",
            member_name=name,
            raw_member_name=f"./{name}",
            size_bytes=len(data),
            sha256="0" * 64,
        )
        for name, data in members_data.items()
    )
    handoff_root, _ = _build_legacy_handoff(tmp_path, member_overrides=corrupted)
    output_dir = tmp_path / "enriched"

    with pytest.raises(LegacyMsaImportError):
        run_legacy_msa_import(handoff_root, output_dir, lz4_argv=(str(shutil.which("lz4")), "-d", "-c"))


def test_run_legacy_msa_import_accepts_empty_output_dir(tmp_path: Path) -> None:
    handoff_root, _ = _build_legacy_handoff(tmp_path)
    output_dir = tmp_path / "enriched"
    # The Control Plane pre-creates output_dir so the writable container mount
    # source exists; the worker must accept an existing-but-empty directory.
    output_dir.mkdir(parents=True, exist_ok=False)

    result = run_legacy_msa_import(handoff_root, output_dir, lz4_argv=(str(shutil.which("lz4")), "-d", "-c"))

    assert result.member_lengths == _EXPECTED_LENGTHS
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "artifact-location.json",
        "artifact-set.json",
        "content-validation.json",
    ]


def test_run_legacy_msa_import_refuses_non_empty_output_dir(tmp_path: Path) -> None:
    handoff_root, _ = _build_legacy_handoff(tmp_path)
    output_dir = tmp_path / "enriched"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "artifact-set.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(LegacyMsaImportError, match="output directory"):
        run_legacy_msa_import(handoff_root, output_dir, lz4_argv=(str(shutil.which("lz4")), "-d", "-c"))


def _write_profiles(tmp_path: Path) -> Path:
    cluster = {
        "owner": "tester",
        "project_root": str(tmp_path / "project"),
        "output_root": str(tmp_path / "output"),
        "staging_root": str(tmp_path / "staging"),
        "orchestration_repo": str(tmp_path / "orchestration"),
        "image": str(tmp_path / "images" / "bspp.sqsh"),
        "transport": "local-slurm",
        "account": "user-account",
    }
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump({"clusters": {"example-cluster": cluster}}, sort_keys=False))
    return path


class WorkerRunningRunner:
    """Runner that executes the runtime worker when the Slurm job is submitted.

    This reproduces the real ordering: Control prepares its directories and
    submits, and only then does the runtime worker run against ``output_dir``.
    """

    def __init__(
        self,
        handoff_root: Path,
        output_dir: Path,
        lz4_argv: tuple[str, ...],
        responses: list[CommandResult],
    ) -> None:
        self.handoff_root = handoff_root
        self.output_dir = output_dir
        self.lz4_argv = lz4_argv
        self.responses = responses
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        if argv and argv[0] == "cp":
            shutil.copy(argv[1], argv[2])
            return CommandResult(argv=argv, returncode=0, stdout="", stderr="")
        if argv and argv[0] == "sbatch":
            run_legacy_msa_import(self.handoff_root, self.output_dir, lz4_argv=self.lz4_argv)
            return CommandResult(argv=argv, returncode=0, stdout="4242\n", stderr="")
        response = self.responses.pop(0)
        return CommandResult(argv=argv, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr)


def test_render_legacy_msa_import_submission(tmp_path: Path) -> None:
    config_path = _write_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    handoff_root = tmp_path / "handoff"
    output_dir = tmp_path / "enriched"
    script_path = tmp_path / "sbatch" / "legacy-msa-import.sbatch"
    tar_path = tmp_path / "payload" / "bundle.tar"
    bundle_path = tmp_path / "payload" / "bundle.tar.lz4"

    rendered = render_legacy_msa_import_submission(
        cluster_profile=profile,
        handoff_root=handoff_root,
        output_dir=output_dir,
        script_path=script_path,
        payload_paths=(tar_path, bundle_path),
    )

    assert "bspp-orchestration-runtime folding legacy-msa-import" in rendered.action_command
    assert "--handoff-root" in rendered.action_command
    assert "--output-dir" in rendered.action_command
    assert rendered.job_name.startswith("bspp_legacy_msa_import_")
    assert rendered.result_dir == output_dir
    script_text = script_path.read_text(encoding="utf-8")
    assert "#SBATCH --partition=" in script_text
    assert "--account=" in script_text
    assert f"{tmp_path / 'payload'}:" in script_text
    # Slurm logs and the staged sbatch live in a sibling directory, never inside
    # the worker's publication target.
    run_dir = tmp_path / "enriched-slurm"
    assert f"#SBATCH --output={run_dir / 'legacy_msa_import.%j.out'}" in script_text
    assert f"#SBATCH --error={run_dir / 'legacy_msa_import.%j.err'}" in script_text
    assert str(output_dir) + "/" not in script_text


@pytest.mark.parametrize("large_inventory", [False, True], ids=["legacy-small", "above-one-mib"])
def test_submit_legacy_msa_import_round_trips(tmp_path: Path, large_inventory: bool) -> None:
    config_path = _write_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    members_data = (
        {
            f"AFDB_AF-{number:016d}.a3m": f"#2,{3 + number % 3}\t1,1\n>query\n{'A' * (5 + number % 3)}\n".encode()
            for number in range(4096)
        }
        if large_inventory
        else _MEMBER_BYTES
    )
    expected_lengths = tuple(5 + number % 3 for number in range(4096)) if large_inventory else _EXPECTED_LENGTHS
    handoff_root, legacy = _build_legacy_handoff(tmp_path, members_data=members_data)
    original_records = _snapshot(handoff_root)
    payload_hashes = {
        path: _sha256(Path(path).read_bytes())
        for path in (legacy.artifact_location.tar_path, legacy.artifact_location.bundle_path)
    }
    if large_inventory:
        for relative in (legacy.artifact_set.chunks[0].logical_path, "artifact-location.json"):
            assert 1024**2 < (handoff_root / relative).stat().st_size < 16 * 1024**2
    output_dir = tmp_path / "enriched"
    lz4_argv = (str(shutil.which("lz4")), "-d", "-c")
    runner = WorkerRunningRunner(
        handoff_root,
        output_dir,
        lz4_argv,
        [
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {
                                "job_id_raw": "4242",
                                "state": {"current": "COMPLETED"},
                                "exit_code": {"return_code": 0, "signal": 0},
                            }
                        ]
                    }
                ),
                stderr="",
            ),
        ],
    )

    result = submit_legacy_msa_import(
        cluster_profile=profile,
        handoff_root=handoff_root,
        output_dir=output_dir,
        runner=runner,
    )

    expected_id = msa_artifact_set_id(
        legacy.artifact_set.chunks,
        legacy.artifact_set.member_count,
        legacy.artifact_set.logical_bytes,
        member_lengths=expected_lengths,
    )
    assert result.job_id == "4242"
    assert result.artifact_set_id == expected_id
    assert result.artifact_location_id == result.artifact_location.artifact_location_id
    assert result.member_lengths == expected_lengths
    assert result.artifact_set.artifact_set_id != legacy.artifact_set.artifact_set_id
    assert json.loads(result.render_json()) == {
        "job_id": "4242",
        "artifact_set_id": expected_id,
        "artifact_location_id": result.artifact_location_id,
        "member_lengths": list(expected_lengths),
    }
    # The worker ran after Control's pre-submit setup: output_dir holds exactly
    # the three records, while the sbatch script and Slurm logs stay in the
    # sibling run directory.
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "artifact-location.json",
        "artifact-set.json",
        "content-validation.json",
    ]
    assert not (output_dir / "slurm-logs").exists()
    assert not (output_dir / "legacy-msa-import.sbatch").exists()
    assert (tmp_path / "enriched-slurm" / "legacy-msa-import.sbatch").is_file()
    assert _snapshot(handoff_root) == original_records
    assert {path: _sha256(Path(path).read_bytes()) for path in payload_hashes} == payload_hashes
    assert result.artifact_location.members == legacy.artifact_location.members
    assert result.artifact_set.chunks == legacy.artifact_set.chunks
    assert result.content_validation.artifact_set_manifest_digest == canonical_mapping_digest(
        result.artifact_set.to_mapping()
    )
    if large_inventory:
        assert (output_dir / "artifact-location.json").stat().st_size > 1024**2


def _read_record_at_boundary(reader: str, path: Path) -> Mapping[str, object]:
    if reader == "runtime":
        return _read_record(path)

    def fetched_record(argv: tuple[str, ...]) -> CommandResult:
        # Model only the external transfer outcome, including invalid outcomes.
        assert argv[0] == "cp" and argv[1] == str(path)
        destination = Path(argv[2])
        if path.is_symlink():
            destination.symlink_to(path.resolve())
        elif path.is_dir():
            destination.mkdir()
        elif path.exists():
            shutil.copyfile(path, destination)
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")

    return _fetch_record(RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=fetched_record), path)


@pytest.mark.parametrize("reader", ["control", "runtime"])
@pytest.mark.parametrize("extra_bytes", [0, 1], ids=["exact-sixteen-mib", "above-sixteen-mib"])
def test_import_record_read_limit(tmp_path: Path, reader: str, extra_bytes: int) -> None:
    record = tmp_path / "record.json"
    # The oversized case is also malformed: the size guard must win before JSON
    # parsing, whereas the exact boundary parses the unchanged object.
    prefix = b"{}" if extra_bytes == 0 else b"!{"
    record.write_bytes(prefix + b" " * (16 * 1024**2 + extra_bytes - len(prefix)))
    if extra_bytes:
        with pytest.raises(
            LegacyMsaImportError if reader == "runtime" else ValueError, match="exceeds the bounded read limit"
        ):
            _read_record_at_boundary(reader, record)
    else:
        assert _read_record_at_boundary(reader, record) == {}


@pytest.mark.parametrize("reader", ["control", "runtime"])
@pytest.mark.parametrize("content", [b"{", b"[]", b"null", b"\xff"])
def test_import_record_rejects_invalid_json_object(tmp_path: Path, reader: str, content: bytes) -> None:
    record = tmp_path / "record.json"
    record.write_bytes(content)
    with pytest.raises(
        LegacyMsaImportError if reader == "runtime" else ValueError, match=r"missing or invalid|must be a JSON object"
    ):
        _read_record_at_boundary(reader, record)


@pytest.mark.parametrize("reader", ["control", "runtime"])
@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_import_record_requires_regular_file(tmp_path: Path, reader: str, kind: str) -> None:
    record = tmp_path / "record.json"
    if kind == "directory":
        record.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target.json"
        target.write_text("{}")
        record.symlink_to(target)
    with pytest.raises(LegacyMsaImportError if reader == "runtime" else ValueError, match="regular file"):
        _read_record_at_boundary(reader, record)


@pytest.mark.parametrize(
    ("header", "query", "expected"),
    [
        ("#2,3,4\t1,1,1", "AAGGGTTTT", 9),
        ("#2,3,4,1\t1,1,1,1", "AAGGGTTTTC", 10),
        ("#2,3\t2,1", "AAGGG", 7),
        ("#2,3,4\t1,2,1", "AAGGGTTTT", 12),
    ],
)
def test_import_attests_nary_expanded_target_lengths_without_changing_payloads(
    tmp_path: Path, header: str, query: str, expected: int
) -> None:
    members = {
        "pdb_test_assembly_1.a3m": f"{header}\n>query\n{query}\n".encode(),
        "pdb_test_assembly_2.a3m": b"#3\t1\n>query\nGGG\n",
    }
    handoff_root, legacy = _build_legacy_handoff(tmp_path, members_data=members)
    original = _snapshot(tmp_path)

    result = run_legacy_msa_import(handoff_root, tmp_path / "enriched")

    assert result.member_lengths == (expected, 3)
    assert result.enriched_manifest.member_lengths == (expected, 3)
    assert result.artifact_set_id != legacy.artifact_set.artifact_set_id
    assert result.rebound_location.artifact_set_id == result.artifact_set_id
    assert result.rebound_validation.artifact_set_manifest_digest == canonical_mapping_digest(
        result.enriched_manifest.to_mapping()
    )
    assert all((tmp_path / relative).read_bytes() == data for relative, data in original.items())
