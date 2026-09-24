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

"""Track A tests for MSA-set consumption and Attempt-scoped A3M projection.

These tests exercise the additive remote location record, the
``verify_bundled_artifact_set`` verifier, ``project_msa_members``, and the
download-verify-project leg ``project_from_remote`` using real tar/lz4 fixtures.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.preprocessing_handoff import (
    BundledMemberVerification,
    MsaArtifactSetManifest,
    MsaChunkManifestReference,
    VerifiedLocalBundledArtifactLocation,
    VerifiedRemoteBundledArtifactLocation,
    msa_artifact_set_id,
    verified_local_bundled_artifact_location_from_mapping,
    verified_local_bundled_artifact_location_id,
    verified_remote_bundled_artifact_location_from_mapping,
    verified_remote_bundled_artifact_location_id,
)
from bspp.orchestration.runtime.folding.execution.errors import FoldingBackendError
from bspp.orchestration.runtime.folding.execution.msa_projection import (
    project_from_remote,
    project_msa_members,
    verify_bundled_artifact_set,
)

pytestmark = pytest.mark.skipif(shutil.which("lz4") is None, reason="lz4 binary required for real tar/lz4 fixtures")

_VERIFIED_AT = "2026-09-01T00:00:00.000000Z"
_LZ4_DEV = ("/usr/bin/lz4", "-d", "-c")

_MEMBER_BYTES = {
    "AFDB_AF-0000000000000001.a3m": b"#1,1\t1,1\n>AFDB_AF-0000000000000001\nAAAA\n",
    "AFDB_AF-0000000000000002.a3m": b"#1,1\t1,1\n>AFDB_AF-0000000000000002\nGGGG\n",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_tar(entries: list[tuple[str, bytes]]) -> tuple[bytes, str, int, tuple[str, ...]]:
    """Build an in-memory tar with a root ``.`` entry plus the given members."""
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


def _make_reference(member_count: int) -> MsaChunkManifestReference:
    chunk_name = "sample_tranche00_00001.fa"
    return MsaChunkManifestReference(
        chunk_name=chunk_name,
        logical_path=f"chunks/{chunk_name.removesuffix('.fa')}.json",
        sha256="a" * 64,
        member_count=member_count,
        logical_bytes=member_count * 100,
    )


def _make_manifest(member_count: int) -> MsaArtifactSetManifest:
    reference = _make_reference(member_count)
    logical_bytes = member_count * 100
    return MsaArtifactSetManifest(
        artifact_set_id=msa_artifact_set_id((reference,), member_count, logical_bytes),
        chunks=(reference,),
        member_count=member_count,
        logical_bytes=logical_bytes,
    )


def _make_consumption(manifest: MsaArtifactSetManifest, member_names: tuple[str, ...]) -> MsaSetConsumption:
    return MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=tuple(f"a3ms/{name}" for name in member_names),
        requires_paired_query_header=True,
    )


def _members_from_data(members: dict[str, bytes]) -> tuple[BundledMemberVerification, ...]:
    return tuple(
        BundledMemberVerification(
            logical_path=f"a3ms/{name}",
            member_name=name,
            raw_member_name=f"./{name}",
            size_bytes=len(data),
            sha256=_sha256(data),
        )
        for name, data in members.items()
    )


def _build_local_location(
    tmp_path: Path,
    manifest: MsaArtifactSetManifest,
    members: tuple[BundledMemberVerification, ...],
    tar_bytes: bytes,
    raw_names: tuple[str, ...],
    *,
    tar_sha256: str | None = None,
) -> tuple[VerifiedLocalBundledArtifactLocation, Path]:
    tar_path = (tmp_path / "bundle.tar").absolute()
    tar_path.write_bytes(tar_bytes)
    lz4_path = (tmp_path / "bundle.tar.lz4").absolute()
    _make_lz4(tar_bytes, lz4_path)
    lz4_bytes = lz4_path.read_bytes()
    resolved_tar_sha256 = _sha256(tar_bytes) if tar_sha256 is None else tar_sha256
    bundle_uri = lz4_path.as_uri()
    location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=manifest.artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(lz4_path),
        bundle_uri=bundle_uri,
        tar_size_bytes=len(tar_bytes),
        tar_sha256=resolved_tar_sha256,
        lz4_size_bytes=len(lz4_bytes),
        lz4_sha256=_sha256(lz4_bytes),
        raw_tar_members=raw_names,
        members=members,
    )
    location = VerifiedLocalBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=manifest.artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(lz4_path),
        bundle_uri=bundle_uri,
        tar_size_bytes=len(tar_bytes),
        tar_sha256=resolved_tar_sha256,
        lz4_size_bytes=len(lz4_bytes),
        lz4_sha256=_sha256(lz4_bytes),
        raw_tar_members=raw_names,
        members=members,
        verified_at=_VERIFIED_AT,
    )
    return location, lz4_path


def _make_local_location(
    tmp_path: Path,
    manifest: MsaArtifactSetManifest,
    members_data: dict[str, bytes],
) -> tuple[VerifiedLocalBundledArtifactLocation, Path]:
    tar_bytes, _, _, raw_names = _make_tar(members_data)
    return _build_local_location(tmp_path, manifest, _members_from_data(members_data), tar_bytes, raw_names)


def _make_remote_location(
    local: VerifiedLocalBundledArtifactLocation,
    *,
    bundle_uri: str = "s3://bucket/prefix/bundle.tar.lz4",
) -> VerifiedRemoteBundledArtifactLocation:
    remote_id = verified_remote_bundled_artifact_location_id(
        artifact_set_id=local.artifact_set_id,
        bundle_uri=bundle_uri,
        tar_size_bytes=local.tar_size_bytes,
        tar_sha256=local.tar_sha256,
        lz4_size_bytes=local.lz4_size_bytes,
        lz4_sha256=local.lz4_sha256,
        raw_tar_members=local.raw_tar_members,
        members=local.members,
    )
    return VerifiedRemoteBundledArtifactLocation(
        artifact_location_id=remote_id,
        artifact_set_id=local.artifact_set_id,
        bundle_uri=bundle_uri,
        tar_size_bytes=local.tar_size_bytes,
        tar_sha256=local.tar_sha256,
        lz4_size_bytes=local.lz4_size_bytes,
        lz4_sha256=local.lz4_sha256,
        raw_tar_members=local.raw_tar_members,
        members=local.members,
        verified_at=local.verified_at,
    )


def test_remote_location_round_trips(tmp_path: Path) -> None:
    manifest = _make_manifest(2)
    local, _ = _make_local_location(tmp_path, manifest, _MEMBER_BYTES)
    remote = _make_remote_location(local)

    parsed = verified_remote_bundled_artifact_location_from_mapping(remote.to_mapping())

    assert parsed == remote


def test_remote_location_rejects_file_uri(tmp_path: Path) -> None:
    manifest = _make_manifest(2)
    local, _ = _make_local_location(tmp_path, manifest, _MEMBER_BYTES)

    with pytest.raises(ValueError, match="s3://"):
        _make_remote_location(local, bundle_uri="file:///tmp/x.tar.lz4")


def test_remote_location_id_distinct_from_local(tmp_path: Path) -> None:
    manifest = _make_manifest(2)
    local, _ = _make_local_location(tmp_path, manifest, _MEMBER_BYTES)
    remote = _make_remote_location(local)

    assert remote.artifact_location_id != local.artifact_location_id


def test_verify_bundled_artifact_set_ok(tmp_path: Path) -> None:
    manifest = _make_manifest(2)
    consumption = _make_consumption(manifest, tuple(_MEMBER_BYTES))
    local, _ = _make_local_location(tmp_path, manifest, _MEMBER_BYTES)

    verify_bundled_artifact_set(consumption, manifest, local, lz4_argv=_LZ4_DEV)


def test_verify_raises_on_corrupted_member_digest(tmp_path: Path) -> None:
    manifest = _make_manifest(1)
    consumption = _make_consumption(manifest, ("AFDB_AF-0000000000000001.a3m",))
    name = "AFDB_AF-0000000000000001.a3m"
    data = _MEMBER_BYTES[name]
    tar_bytes, _, _, raw_names = _make_tar({name: data})
    corrupted = BundledMemberVerification(
        logical_path=f"a3ms/{name}",
        member_name=name,
        raw_member_name=f"./{name}",
        size_bytes=len(data),
        sha256="0" * 64,
    )
    location, _ = _build_local_location(tmp_path, manifest, (corrupted,), tar_bytes, raw_names)

    with pytest.raises(FoldingBackendError):
        verify_bundled_artifact_set(consumption, manifest, location, lz4_argv=_LZ4_DEV)


def test_verify_raises_on_wrong_tar_digest(tmp_path: Path) -> None:
    manifest = _make_manifest(1)
    consumption = _make_consumption(manifest, ("AFDB_AF-0000000000000001.a3m",))
    name = "AFDB_AF-0000000000000001.a3m"
    data = _MEMBER_BYTES[name]
    tar_bytes, _, _, raw_names = _make_tar({name: data})
    location, _ = _build_local_location(
        tmp_path, manifest, _members_from_data({name: data}), tar_bytes, raw_names, tar_sha256="0" * 64
    )

    with pytest.raises(FoldingBackendError):
        verify_bundled_artifact_set(consumption, manifest, location, lz4_argv=_LZ4_DEV)


def test_verify_raises_on_manifest_mismatch(tmp_path: Path) -> None:
    manifest = _make_manifest(1)
    one_member = {"AFDB_AF-0000000000000001.a3m": _MEMBER_BYTES["AFDB_AF-0000000000000001.a3m"]}
    local, _ = _make_local_location(tmp_path, manifest, one_member)
    mismatched = MsaSetConsumption(
        artifact_set_id="sha256:" + "b" * 64,
        expected_chunk_count=1,
        member_a3m_paths=("a3ms/AFDB_AF-0000000000000001.a3m",),
        requires_paired_query_header=True,
    )

    with pytest.raises(FoldingBackendError):
        verify_bundled_artifact_set(mismatched, manifest, local, lz4_argv=_LZ4_DEV)


def test_project_msa_members_materializes_only_declared(tmp_path: Path) -> None:
    manifest = _make_manifest(2)
    consumption = _make_consumption(manifest, tuple(_MEMBER_BYTES))
    local, _ = _make_local_location(tmp_path, manifest, _MEMBER_BYTES)
    workspace = tmp_path / "attempt"

    projected = project_msa_members(consumption, manifest, local, workspace)

    assert set(projected) == {f"a3ms/{name}" for name in _MEMBER_BYTES}
    for name, data in _MEMBER_BYTES.items():
        destination = workspace / "a3ms" / name
        assert destination.read_bytes() == data
        assert projected[f"a3ms/{name}"] == destination
    assert sorted(path.name for path in (workspace / "a3ms").iterdir()) == sorted(_MEMBER_BYTES)


def test_project_msa_members_rejects_path_traversal(tmp_path: Path) -> None:
    manifest = _make_manifest(1)
    name = "AFDB_AF-0000000000000001.a3m"
    consumption = _make_consumption(manifest, (name,))
    data = _MEMBER_BYTES[name]
    tar_bytes, _, _, _ = _write_tar([(f"./{name}", data), ("../evil.a3m", b"evil")])
    member = _members_from_data({name: data})[0]
    location, _ = _build_local_location(tmp_path, manifest, (member,), tar_bytes, (".", f"./{name}"))
    workspace = tmp_path / "attempt"

    with pytest.raises(FoldingBackendError):
        project_msa_members(consumption, manifest, location, workspace)


def test_project_msa_members_rejects_undeclared_member(tmp_path: Path) -> None:
    manifest = _make_manifest(1)
    name = "AFDB_AF-0000000000000001.a3m"
    consumption = _make_consumption(manifest, (name,))
    data = _MEMBER_BYTES[name]
    extra_name = "AFDB_AF-0000000000000002.a3m"
    tar_bytes, _, _, _ = _write_tar([(f"./{name}", data), (f"./{extra_name}", _MEMBER_BYTES[extra_name])])
    member = _members_from_data({name: data})[0]
    location, _ = _build_local_location(tmp_path, manifest, (member,), tar_bytes, (".", f"./{name}"))
    workspace = tmp_path / "attempt"

    with pytest.raises(FoldingBackendError):
        project_msa_members(consumption, manifest, location, workspace)


def test_verify_rejects_same_count_different_member_identity(tmp_path: Path) -> None:
    manifest = _make_manifest(1)
    one_member = {"AFDB_AF-0000000000000001.a3m": _MEMBER_BYTES["AFDB_AF-0000000000000001.a3m"]}
    local, _ = _make_local_location(tmp_path, manifest, one_member)
    mismatched = MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=("a3ms/AFDB_AF-0000000000000002.a3m",),
        requires_paired_query_header=True,
    )

    with pytest.raises(FoldingBackendError):
        verify_bundled_artifact_set(mismatched, manifest, local, lz4_argv=_LZ4_DEV)


def test_verify_rejects_same_members_different_order(tmp_path: Path) -> None:
    manifest = _make_manifest(2)
    local, _ = _make_local_location(tmp_path, manifest, _MEMBER_BYTES)
    mismatched = _make_consumption(manifest, tuple(reversed(tuple(_MEMBER_BYTES))))

    with pytest.raises(FoldingBackendError):
        verify_bundled_artifact_set(mismatched, manifest, local, lz4_argv=_LZ4_DEV)


def test_project_msa_members_rejects_tampered_durable_tar(tmp_path: Path) -> None:
    """A durable tar swapped after bundle verification must fail closed at projection.

    ``verify_bundled_artifact_set`` validates the lz4 bundle through a
    disposable temp tar, so it cannot see a later swap of the durable
    ``tar_path``.  The projection boundary re-hashes the exact tar it opens
    and refuses to project the tampered bytes.
    """
    manifest = _make_manifest(2)
    consumption = _make_consumption(manifest, tuple(_MEMBER_BYTES))
    local, _ = _make_local_location(tmp_path, manifest, _MEMBER_BYTES)
    verify_bundled_artifact_set(consumption, manifest, local, lz4_argv=_LZ4_DEV)
    bundle_bytes = Path(local.bundle_path).read_bytes()

    # Tamper with only the durable tar: identical member names, altered bytes.
    tampered_data = {name: b"#1,1\t1,1\n>tampered\nTTTT\n" for name in _MEMBER_BYTES}
    tampered_tar, _, _, _ = _make_tar(tampered_data)
    tar_path = Path(local.tar_path)
    tar_path.write_bytes(tampered_tar)
    workspace = tmp_path / "attempt"

    with pytest.raises(FoldingBackendError, match="durable tar does not match its location record"):
        project_msa_members(consumption, manifest, local, workspace)

    # No projection happened and the producer bundle stayed byte-identical.
    assert not (workspace / "a3ms").exists()
    assert tar_path.read_bytes() == tampered_tar
    assert Path(local.bundle_path).read_bytes() == bundle_bytes


def test_project_msa_members_rejects_member_digest_mismatch(tmp_path: Path) -> None:
    """A location record contradicting its own tar fails before that member is written."""
    manifest = _make_manifest(1)
    name = "AFDB_AF-0000000000000001.a3m"
    consumption = _make_consumption(manifest, (name,))
    data = _MEMBER_BYTES[name]
    tar_bytes, _, _, raw_names = _make_tar({name: data})
    corrupted = BundledMemberVerification(
        logical_path=f"a3ms/{name}",
        member_name=name,
        raw_member_name=f"./{name}",
        size_bytes=len(data),
        sha256="0" * 64,
    )
    location, _ = _build_local_location(tmp_path, manifest, (corrupted,), tar_bytes, raw_names)
    workspace = tmp_path / "attempt"

    with pytest.raises(FoldingBackendError, match="does not match its declared verification"):
        project_msa_members(consumption, manifest, location, workspace)

    assert not (workspace / "a3ms" / name).exists()


def test_project_msa_members_rejects_mismatched_consumption_before_mutation(tmp_path: Path) -> None:
    manifest = _make_manifest(2)
    local, _ = _make_local_location(tmp_path, manifest, _MEMBER_BYTES)
    mismatched = _make_consumption(manifest, tuple(reversed(tuple(_MEMBER_BYTES))))
    workspace = tmp_path / "attempt"

    with pytest.raises(FoldingBackendError):
        project_msa_members(mismatched, manifest, local, workspace)

    assert not (workspace / "a3ms").exists()


def test_project_from_remote_reproduces_local_handoff(tmp_path: Path) -> None:
    manifest = _make_manifest(2)
    consumption = _make_consumption(manifest, tuple(_MEMBER_BYTES))
    local, lz4_path = _make_local_location(tmp_path, manifest, _MEMBER_BYTES)
    remote = _make_remote_location(local)
    workspace = tmp_path / "remote-attempt"

    def fake_download(remote_location: VerifiedRemoteBundledArtifactLocation, destination: Path) -> None:
        assert remote_location.bundle_uri == "s3://bucket/prefix/bundle.tar.lz4"
        destination.write_bytes(lz4_path.read_bytes())

    derived, projected = project_from_remote(
        consumption,
        manifest,
        remote,
        workspace,
        download=fake_download,
        lz4_argv=_LZ4_DEV,
    )

    assert derived.artifact_set_id == local.artifact_set_id
    assert derived.tar_sha256 == local.tar_sha256
    assert derived.lz4_sha256 == local.lz4_sha256
    assert derived.members == local.members
    for name, data in _MEMBER_BYTES.items():
        assert (workspace / "a3ms" / name).read_bytes() == data
        assert projected[f"a3ms/{name}"] == workspace / "a3ms" / name
    record = json.loads((workspace / "artifact-location.json").read_text(encoding="utf-8"))
    parsed = verified_local_bundled_artifact_location_from_mapping(record)
    assert parsed == derived
