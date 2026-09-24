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

"""Track-A cross-story integration: consumption → projection → MSAResult → inputs.

A small real merged-A3M bundle (a two-chain compound) is projected through both
the local and injectable-remote legs, then prepared through both the OpenFold
and BioIR input preprocessors.  Each case ends at the strict ``layout.json``
contract loader so the full Track-A seam is exercised end to end.
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

from bspp.orchestration.contract.folding_input import (
    MsaSetConsumption,
    bioir_request_manifest_from_mapping,
    folding_input_layout_from_mapping,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    BundledMemberVerification,
    MsaArtifactSetManifest,
    MsaChunkManifestReference,
    VerifiedLocalBundledArtifactLocation,
    VerifiedRemoteBundledArtifactLocation,
    msa_artifact_set_id,
    verified_local_bundled_artifact_location_id,
    verified_remote_bundled_artifact_location_id,
)
from bspp.orchestration.runtime.folding.execution.bioir_inputs import BioIRInputPreprocessor
from bspp.orchestration.runtime.folding.execution.models import ProteinTarget
from bspp.orchestration.runtime.folding.execution.msa_preparation import prepare_projected_msa
from bspp.orchestration.runtime.folding.execution.msa_projection import (
    project_from_remote,
    project_msa_members,
    verify_bundled_artifact_set,
)
from bspp.orchestration.runtime.folding.execution.openfold_inputs import OpenFoldInputPreprocessor

pytestmark = pytest.mark.skipif(shutil.which("lz4") is None, reason="lz4 binary required for real tar/lz4 fixtures")

_VERIFIED_AT = "2026-09-01T00:00:00.000000Z"
_LZ4_DEV = ("/usr/bin/lz4", "-d", "-c")

_TARGET_ID = "AF-0000000000000001_AF-0000000000000002"
_MEMBER_NAME = "AFDB_AF-0000000000000001_AF-0000000000000002.a3m"
_MEMBER_LOGICAL_PATH = f"a3ms/{_MEMBER_NAME}"
_MERGED_A3M = "#4,3\t1,1\n>query\nACDEGGX\n>hit\nAC-EG-X\n"
_MEMBER_BYTES = {_MEMBER_NAME: _MERGED_A3M.encode("utf-8")}


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


def _make_consumption(manifest: MsaArtifactSetManifest) -> MsaSetConsumption:
    return MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=(_MEMBER_LOGICAL_PATH,),
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


def _make_local_location(
    tmp_path: Path,
    manifest: MsaArtifactSetManifest,
    members_data: dict[str, bytes],
) -> tuple[VerifiedLocalBundledArtifactLocation, Path]:
    tar_bytes, _, _, raw_names = _make_tar(members_data)
    tar_path = (tmp_path / "bundle.tar").absolute()
    tar_path.write_bytes(tar_bytes)
    lz4_path = (tmp_path / "bundle.tar.lz4").absolute()
    _make_lz4(tar_bytes, lz4_path)
    lz4_bytes = lz4_path.read_bytes()
    members = _members_from_data(members_data)
    bundle_uri = lz4_path.as_uri()
    location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=manifest.artifact_set_id,
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
    location = VerifiedLocalBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=manifest.artifact_set_id,
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
    return location, lz4_path


def _make_remote_location(local: VerifiedLocalBundledArtifactLocation) -> VerifiedRemoteBundledArtifactLocation:
    bundle_uri = "s3://bucket/prefix/bundle.tar.lz4"
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


@pytest.mark.parametrize("origin", ["local", "remote"])
@pytest.mark.parametrize("preprocessor", ["openfold", "bioir"])
def test_track_a_cross_story_chain(tmp_path: Path, origin: str, preprocessor: str) -> None:
    manifest = _make_manifest(1)
    consumption = _make_consumption(manifest)
    local, lz4_path = _make_local_location(tmp_path, manifest, _MEMBER_BYTES)

    if origin == "local":
        verify_bundled_artifact_set(consumption, manifest, local, lz4_argv=_LZ4_DEV)
        workspace = tmp_path / "local-attempt"
        projected = project_msa_members(consumption, manifest, local, workspace)
    else:
        remote = _make_remote_location(local)
        workspace = tmp_path / "remote-attempt"

        def fake_download(remote_location: VerifiedRemoteBundledArtifactLocation, destination: Path) -> None:
            destination.write_bytes(lz4_path.read_bytes())

        _derived, projected = project_from_remote(
            consumption,
            manifest,
            remote,
            workspace,
            download=fake_download,
            lz4_argv=_LZ4_DEV,
        )

    target = ProteinTarget(_TARGET_ID, "compound", ("ACDE", "GGX"))
    msa = prepare_projected_msa(consumption, projected, target, tmp_path / "split")

    assert msa.backend == "track-a-projected-msa"
    assert [chain.chain_index for chain in msa.chains] == [1, 2]
    assert [chain.query_sequence for chain in msa.chains] == ["ACDE", "GGX"]
    assert msa.metadata["artifact_set_id"] == manifest.artifact_set_id
    assert msa.metadata["selected_logical_path"] == _MEMBER_LOGICAL_PATH
    assert Path(msa.metadata["merged_source_path"]).is_file()

    output_dir = tmp_path / "prepared"
    if preprocessor == "openfold":
        prepared = OpenFoldInputPreprocessor().run(target, msa, output_dir)
        layout = folding_input_layout_from_mapping(json.loads((output_dir / "layout.json").read_text(encoding="utf-8")))
        expected_chain_ids = [f"{_TARGET_ID}_A", f"{_TARGET_ID}_B"]
        assert layout.layout == "openfold"
        assert layout.chain_ids == tuple(expected_chain_ids)
        assert layout.pairing == "species-from-colabfold-headers"
        assert prepared.metadata["chain_ids"] == expected_chain_ids
        assert "-".join(prepared.metadata["chain_ids"]) == f"{_TARGET_ID}_A-{_TARGET_ID}_B"
    else:
        prepared = BioIRInputPreprocessor(use_paired_msa=True).run(target, msa, output_dir)
        request = bioir_request_manifest_from_mapping(
            json.loads((output_dir / "bioir-request.json").read_text(encoding="utf-8"))
        )
        layout = folding_input_layout_from_mapping(json.loads((output_dir / "layout.json").read_text(encoding="utf-8")))
        layout.validate_against_request(request)
        assert layout.layout == "bioir"
        assert request.chain_ids == ("A", "B")
        assert prepared.metadata["pairing"] == "colabfold-merged-row-index"
