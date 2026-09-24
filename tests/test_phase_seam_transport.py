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

"""Tests for the phase-seam transport policy and MSA-set publication leg.

Fully offline: no ``lz4`` binary and no network access. The transfer boundary
is injected as a fake callable, so ``publish_msa_set_to_s3`` never
touches ``s5cmd`` or object storage in these tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.phase import (
    FoldingPhasePlanPayload,
    PhaseSeamTransport,
    folding_phase_plan_payload_from_mapping,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    BundledMemberVerification,
    VerifiedLocalBundledArtifactLocation,
    verified_local_bundled_artifact_location_id,
)
from bspp.orchestration.runtime.data_movement.common import TransferResult
from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer
from bspp.orchestration.runtime.folding.seam_transport import (
    SeamTransportError,
    publish_msa_set_to_s3,
    resolve_seam_transfer,
)

_VERIFIED_AT = "2026-09-01T00:00:00.000000Z"
_MEMBER_NAME = "AFDB_AF-0000000000000001.a3m"
_ARTIFACT_SET_ID = "sha256:" + "a" * 64
# Real sha256 of the fixture bundle b"x" * 64 — publication re-hashes the local
# file at publish time and fails closed on drift, so the fixture must match.
_LZ4_SHA256 = "7ce100971f64e7001e8fe5a51973ecdfe1ced42befe7ee8d5fd6219506b5393c"
_TAR_SHA256 = "c" * 64
_MEMBER_SHA256 = "b" * 64


def _make_msa() -> MsaSetConsumption:
    return MsaSetConsumption(
        artifact_set_id=_ARTIFACT_SET_ID,
        expected_chunk_count=1,
        member_a3m_paths=(f"a3ms/{_MEMBER_NAME}",),
        requires_paired_query_header=True,
    )


def _make_local_location(tmp_path: Path) -> VerifiedLocalBundledArtifactLocation:
    """Build a synthetic verified local location with a 64-byte dummy bundle."""
    bundle_path = (tmp_path / "bundle.tar.lz4").absolute()
    bundle_path.write_bytes(b"x" * 64)
    tar_path = (tmp_path / "bundle.tar").absolute()
    tar_path.write_bytes(b"y" * 100)
    member = BundledMemberVerification(
        logical_path=f"a3ms/{_MEMBER_NAME}",
        member_name=_MEMBER_NAME,
        raw_member_name=f"./{_MEMBER_NAME}",
        size_bytes=100,
        sha256=_MEMBER_SHA256,
    )
    bundle_uri = bundle_path.as_uri()
    raw_tar_members = (".", f"./{_MEMBER_NAME}")
    location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=_ARTIFACT_SET_ID,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=bundle_uri,
        tar_size_bytes=100,
        tar_sha256=_TAR_SHA256,
        lz4_size_bytes=64,
        lz4_sha256=_LZ4_SHA256,
        raw_tar_members=raw_tar_members,
        members=(member,),
    )
    return VerifiedLocalBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=_ARTIFACT_SET_ID,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=bundle_uri,
        tar_size_bytes=100,
        tar_sha256=_TAR_SHA256,
        lz4_size_bytes=64,
        lz4_sha256=_LZ4_SHA256,
        raw_tar_members=raw_tar_members,
        members=(member,),
        verified_at=_VERIFIED_AT,
    )


def _ok_transfer(src: str, dst: str) -> TransferResult:
    return TransferResult(tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=0, elapsed_s=0.0)


def _fail_transfer(src: str, dst: str) -> TransferResult:
    return TransferResult(
        tool="s5cmd",
        argv=("s5cmd", "cp", src, dst),
        returncode=1,
        elapsed_s=0.0,
        stderr_tail="boom",
    )


def test_phase_seam_transport_round_trip() -> None:
    record = PhaseSeamTransport(seam="preprocessing-to-folding", policy="publish-to-s3")
    assert record.to_mapping() == {
        "schema_version": 1,
        "seam": "preprocessing-to-folding",
        "policy": "publish-to-s3",
    }


def test_folding_payload_transport_default_and_round_trip() -> None:
    msa = _make_msa()
    payload = FoldingPhasePlanPayload(msa_set=msa, backend="openfold-cli")
    assert payload.transport == "publish-to-s3"
    assert payload.to_mapping()["transport"] == "publish-to-s3"

    local = FoldingPhasePlanPayload(msa_set=msa, backend="openfold-cli", transport="local")
    assert local.transport == "local"
    assert local.to_mapping()["transport"] == "local"
    assert folding_phase_plan_payload_from_mapping(local.to_mapping()) == local


def test_folding_payload_rejects_unknown_transport() -> None:
    with pytest.raises(ValueError):
        FoldingPhasePlanPayload(msa_set=_make_msa(), backend="openfold-cli", transport="bogus")  # type: ignore[arg-type]


def test_phase_seam_transport_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError):
        PhaseSeamTransport(seam="x", policy="bogus")  # type: ignore[arg-type]


def test_resolve_seam_transfer_local() -> None:
    resolved = resolve_seam_transfer(policy="local")
    assert resolved.object_storage_leg is False
    assert resolved.tool is None
    assert resolved.transfer is None


def test_resolve_seam_transfer_publish() -> None:
    resolved = resolve_seam_transfer(policy="publish-to-s3")
    assert resolved.object_storage_leg is True
    assert resolved.tool == "s5cmd"
    assert resolved.transfer is s3_transfer.cp

    # Cross-check consistency with the contract-level decision
    from bspp.orchestration.contract.operator_data_movement import resolve_seam_transfer_decision

    decision = resolve_seam_transfer_decision(policy="publish-to-s3")
    assert decision.object_storage_leg == resolved.object_storage_leg
    assert decision.tool == resolved.tool
    assert decision.note == resolved.note


def test_resolve_seam_transfer_rejects_unknown_policy() -> None:
    with pytest.raises(SeamTransportError):
        resolve_seam_transfer(policy="bogus")  # type: ignore[arg-type]


def test_publish_msa_set_to_s3_with_mocked_boundary(tmp_path: Path) -> None:
    location = _make_local_location(tmp_path)
    calls: list[tuple[str, str]] = []

    def recording_transfer(src: str, dst: str) -> TransferResult:
        calls.append((src, dst))
        return _ok_transfer(src, dst)

    remote, evidence = publish_msa_set_to_s3(
        location=location,
        s3_prefix="s3://bucket/msa",
        transfer=recording_transfer,
    )

    assert remote.bundle_uri == f"s3://bucket/msa/{location.lz4_sha256}.tar.lz4"
    assert remote.artifact_set_id == location.artifact_set_id
    assert remote.tar_sha256 == location.tar_sha256
    assert remote.lz4_sha256 == location.lz4_sha256
    assert remote.members == location.members

    assert evidence.object_key == remote.bundle_uri
    assert evidence.size_bytes == location.lz4_size_bytes
    assert evidence.sha256 == location.lz4_sha256
    assert evidence.transfer_result.ok is True

    assert calls == [(location.bundle_path, remote.bundle_uri)]


def test_publish_fails_closed_on_transfer_error(tmp_path: Path) -> None:
    location = _make_local_location(tmp_path)
    with pytest.raises(SeamTransportError):
        publish_msa_set_to_s3(
            location=location,
            s3_prefix="s3://bucket/msa",
            transfer=_fail_transfer,
        )


def test_publish_fails_closed_on_bad_prefix(tmp_path: Path) -> None:
    location = _make_local_location(tmp_path)
    calls: list[tuple[str, str]] = []

    def recording_transfer(src: str, dst: str) -> TransferResult:
        calls.append((src, dst))
        return _ok_transfer(src, dst)

    with pytest.raises(SeamTransportError):
        publish_msa_set_to_s3(
            location=location,
            s3_prefix="not-an-s3-prefix",
            transfer=recording_transfer,
        )
    assert calls == []


def test_publish_fails_closed_on_missing_bundle(tmp_path: Path) -> None:
    location = _make_local_location(tmp_path)
    Path(location.bundle_path).unlink()
    calls: list[tuple[str, str]] = []

    def recording_transfer(src: str, dst: str) -> TransferResult:
        calls.append((src, dst))
        return _ok_transfer(src, dst)

    with pytest.raises(SeamTransportError):
        publish_msa_set_to_s3(
            location=location,
            s3_prefix="s3://bucket/msa",
            transfer=recording_transfer,
        )
    assert calls == []


def test_publish_fails_closed_on_size_mismatch(tmp_path: Path) -> None:
    location = _make_local_location(tmp_path)
    Path(location.bundle_path).write_bytes(b"z" * 65)
    calls: list[tuple[str, str]] = []

    def recording_transfer(src: str, dst: str) -> TransferResult:
        calls.append((src, dst))
        return _ok_transfer(src, dst)

    with pytest.raises(SeamTransportError):
        publish_msa_set_to_s3(
            location=location,
            s3_prefix="s3://bucket/msa",
            transfer=recording_transfer,
        )
    assert calls == []


def test_publish_fails_closed_on_same_size_content_mutation(tmp_path: Path) -> None:
    # Same length, different bytes: a size-only preflight would publish the
    # drifted bundle under the stale recorded digest; the re-hash must refuse.
    location = _make_local_location(tmp_path)
    Path(location.bundle_path).write_bytes(b"z" * 64)
    calls: list[tuple[str, str]] = []

    def recording_transfer(src: str, dst: str) -> TransferResult:
        calls.append((src, dst))
        return _ok_transfer(src, dst)

    with pytest.raises(SeamTransportError, match="refusing to publish stale bytes"):
        publish_msa_set_to_s3(
            location=location,
            s3_prefix="s3://bucket/msa",
            transfer=recording_transfer,
        )
    assert calls == []
