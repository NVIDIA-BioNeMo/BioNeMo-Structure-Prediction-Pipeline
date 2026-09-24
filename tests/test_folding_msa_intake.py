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

"""Tests for folding MSA intake: ``download_remote_msa_bundle`` + ``prepare_folding_msa_input`` dispatch.

Covers:
* ``download_remote_msa_bundle`` (injected fake transfer): success, ``PlannedTransfer``
  rejection, non-OK fail-closed.
* ``prepare_folding_msa_input`` dispatch: local → ``project_msa_members``,
  remote → ``project_from_remote``.
* Folding backward-compat: old folding Plan payload (no prefix) constructs without error.
* Folding payload ``s3_prediction_prefix`` deferred enforcement.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.phase import (
    FoldingPhasePlanPayload,
    FoldingPhaseRunSpec,
    FoldingPhaseRunSpecPayload,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    BundledMemberVerification,
    MsaArtifactSetManifest,
    VerifiedLocalBundledArtifactLocation,
    VerifiedRemoteBundledArtifactLocation,
    verified_local_bundled_artifact_location_id,
    verified_remote_bundled_artifact_location_id,
)
from bspp.orchestration.runtime.data_movement.common import PlannedTransfer, TransferResult
from bspp.orchestration.runtime.folding.execution.msa_intake import (
    MsaIntakeError,
    download_remote_msa_bundle,
    prepare_folding_msa_input,
)

ARTIFACT_SET_ID = "sha256:" + "a" * 64
MEMBER_NAME = "AFDB_AF-0000000000000001.a3m"
MEMBER_PATH = f"a3ms/{MEMBER_NAME}"


def _ok_transfer(src: str, dst: str) -> TransferResult:
    return TransferResult(tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=0, elapsed_s=0.0)


def _fail_transfer(src: str, dst: str) -> TransferResult:
    return TransferResult(tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=1, elapsed_s=0.0, stderr_tail="boom")


def _planned_transfer(src: str, dst: str) -> PlannedTransfer:
    return PlannedTransfer(tool="s5cmd", argv=("s5cmd", "cp", src, dst), note="dry-run")


def _make_members() -> tuple[BundledMemberVerification, ...]:
    return (
        BundledMemberVerification(
            logical_path=MEMBER_PATH,
            member_name=MEMBER_NAME,
            raw_member_name=MEMBER_NAME,
            size_bytes=1,
            sha256="b" * 64,
        ),
    )


def _make_artifact_set() -> MsaArtifactSetManifest:
    """Create a minimal MsaArtifactSetManifest for dispatch testing.

    Since the projection functions are mocked in dispatch tests, this only
    needs to be a valid constructible object.
    """
    from unittest.mock import MagicMock

    return MagicMock(spec=MsaArtifactSetManifest, artifact_set_id=ARTIFACT_SET_ID)  # type: ignore[return-value]


def _make_local_location(tmp_path: Path) -> VerifiedLocalBundledArtifactLocation:
    tar_path = tmp_path / "msa-set" / "msa-set.tar"
    bundle_path = tmp_path / "msa-set" / "msa-set.tar.lz4"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tar_bytes = b"tar-content"
    bundle_bytes = b"lz4-content"
    tar_path.write_bytes(tar_bytes)
    bundle_path.write_bytes(bundle_bytes)
    members = _make_members()
    raw_tar_members = (MEMBER_NAME,)
    tar_sha = hashlib.sha256(tar_bytes).hexdigest()
    lz4_sha = hashlib.sha256(bundle_bytes).hexdigest()
    location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=ARTIFACT_SET_ID,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=bundle_path.as_uri(),
        tar_size_bytes=len(tar_bytes),
        tar_sha256=tar_sha,
        lz4_size_bytes=len(bundle_bytes),
        lz4_sha256=lz4_sha,
        raw_tar_members=raw_tar_members,
        members=members,
    )
    return VerifiedLocalBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=ARTIFACT_SET_ID,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=bundle_path.as_uri(),
        tar_size_bytes=len(tar_bytes),
        tar_sha256=tar_sha,
        lz4_size_bytes=len(bundle_bytes),
        lz4_sha256=lz4_sha,
        raw_tar_members=raw_tar_members,
        members=members,
        verified_at="2026-01-01T00:00:00Z",
    )


def _make_remote_location() -> VerifiedRemoteBundledArtifactLocation:
    members = _make_members()
    raw_tar_members = (MEMBER_NAME,)
    location_id = verified_remote_bundled_artifact_location_id(
        artifact_set_id=ARTIFACT_SET_ID,
        bundle_uri="s3://bucket/msa-set.tar.lz4",
        tar_size_bytes=100,
        tar_sha256="c" * 64,
        lz4_size_bytes=64,
        lz4_sha256="d" * 64,
        raw_tar_members=raw_tar_members,
        members=members,
    )
    return VerifiedRemoteBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=ARTIFACT_SET_ID,
        bundle_uri="s3://bucket/msa-set.tar.lz4",
        tar_size_bytes=100,
        tar_sha256="c" * 64,
        lz4_size_bytes=64,
        lz4_sha256="d" * 64,
        raw_tar_members=raw_tar_members,
        members=members,
        verified_at="2026-01-01T00:00:00Z",
    )


def _make_msa_set() -> MsaSetConsumption:
    return MsaSetConsumption(
        artifact_set_id=ARTIFACT_SET_ID,
        expected_chunk_count=1,
        member_a3m_paths=(MEMBER_PATH,),
        requires_paired_query_header=True,
    )


def _make_folding_runspec(
    location: VerifiedLocalBundledArtifactLocation | VerifiedRemoteBundledArtifactLocation,
    msa_set: MsaSetConsumption,
) -> FoldingPhaseRunSpec:
    """Create a minimal FoldingPhaseRunSpec for intake dispatch testing."""
    from bspp.orchestration.contract.phase import (
        FoldingActionPayload,
        FoldingPhaseRunSpecPayload,
        FoldingResolvedClusterSnapshot,
        FoldingRuntimeAction,
        PhaseMountSnapshot,
        PhaseSlurmResources,
    )

    resources = PhaseSlurmResources(partition="cpu", cpus_per_task=4, memory="16G", time="01:00:00")
    action = FoldingRuntimeAction(
        action_id="msa-flatten-000001",
        dependencies=(),
        resources=resources,
        payload=FoldingActionPayload(action_kind="msa-flatten", params=()),
        action_kind="msa-flatten",
    )
    cluster = FoldingResolvedClusterSnapshot(
        profile_name="folding-gpu",
        owner="bspp",
        transport="ssh",
        ssh_target="example-cluster",
        account="bspp",
        project_root="/lustre/bspp",
        staging_root="/lustre/bspp/staging",
        orchestration_repo="/lustre/bspp/orch",
        runtime_image="registry/runtime:latest",
        extra_mounts=(PhaseMountSnapshot(source="/data", target="/data"),),
    )
    return FoldingPhaseRunSpec(
        phase_run_id="phase-run-" + "a" * 32,
        attempt_id="attempt-0001",
        phase_plan_digest="e" * 64,
        materialized_at="2026-01-01T00:00:00.000000Z",
        input_location=location,
        cluster=cluster,
        payload=FoldingPhaseRunSpecPayload(
            msa_set=msa_set,
            backend="openfold-cli",
            actions=(action,),
        ),
    )


# ---------------------------------------------------------------------------
# download_remote_msa_bundle
# ---------------------------------------------------------------------------


def test_download_remote_msa_bundle_success(tmp_path: Path) -> None:
    """A successful download writes the bundle to the destination."""
    remote = _make_remote_location()
    dest = tmp_path / "bundle.tar.lz4"

    with patch(
        "bspp.orchestration.runtime.folding.execution.msa_intake.s3_transfer.cp",
        side_effect=_ok_transfer,
    ):
        download_remote_msa_bundle(remote, dest)
    # The production binding calls s3_transfer.cp; it doesn't write the file itself.
    # We only verify no exception is raised.


def test_download_remote_msa_bundle_rejects_planned_transfer(tmp_path: Path) -> None:
    """A PlannedTransfer (dry-run) is rejected."""
    remote = _make_remote_location()
    dest = tmp_path / "bundle.tar.lz4"

    with (
        patch(
            "bspp.orchestration.runtime.folding.execution.msa_intake.s3_transfer.cp",
            side_effect=_planned_transfer,
        ),
        pytest.raises(MsaIntakeError, match="executed transfer, not a dry-run plan"),
    ):
        download_remote_msa_bundle(remote, dest)


def test_download_remote_msa_bundle_non_ok_fail_closed(tmp_path: Path) -> None:
    """A non-OK TransferResult fails closed."""
    remote = _make_remote_location()
    dest = tmp_path / "bundle.tar.lz4"

    with (
        patch(
            "bspp.orchestration.runtime.folding.execution.msa_intake.s3_transfer.cp",
            side_effect=_fail_transfer,
        ),
        pytest.raises(MsaIntakeError, match="download failed with returncode 1"),
    ):
        download_remote_msa_bundle(remote, dest)


# ---------------------------------------------------------------------------
# prepare_folding_msa_input dispatch
# ---------------------------------------------------------------------------


def test_prepare_folding_msa_input_local_dispatch(tmp_path: Path) -> None:
    """Local input_location dispatches to project_msa_members (not the remote download path)."""
    from bspp.orchestration.runtime.folding.execution import msa_intake as intake_mod

    location = _make_local_location(tmp_path)
    msa_set = _make_msa_set()
    runspec = _make_folding_runspec(location, msa_set)
    workspace = tmp_path / "workspace"
    artifact_set = _make_artifact_set()

    download_called = False

    def spy_download(remote_location, destination):  # type: ignore[no-untyped-def]
        nonlocal download_called
        download_called = True
        raise AssertionError("download_remote_msa_bundle should not be called for local dispatch")

    projected_result: dict[str, Path] = {MEMBER_PATH: workspace / "a3ms" / MEMBER_NAME}

    with (
        patch.object(intake_mod, "download_remote_msa_bundle", side_effect=spy_download),
        patch.object(intake_mod, "project_msa_members", return_value=projected_result) as mock_project,
    ):
        local_loc, projected = prepare_folding_msa_input(
            runspec, workspace, artifact_set=artifact_set, lz4_argv=("/usr/bin/lz4", "-d")
        )
    assert not download_called
    assert mock_project.called
    assert local_loc is location
    assert projected is projected_result


def test_prepare_folding_msa_input_remote_dispatch(tmp_path: Path) -> None:
    """Remote input_location dispatches to project_from_remote with download_remote_msa_bundle."""
    from bspp.orchestration.runtime.folding.execution import msa_intake as intake_mod

    remote = _make_remote_location()
    msa_set = _make_msa_set()
    runspec = _make_folding_runspec(remote, msa_set)
    workspace = tmp_path / "workspace"
    artifact_set = _make_artifact_set()

    download_called = False

    def spy_download(remote_location, destination):  # type: ignore[no-untyped-def]
        nonlocal download_called
        download_called = True

    expected_local = _make_local_location(tmp_path)
    expected_projected: dict[str, Path] = {MEMBER_PATH: workspace / "a3ms" / MEMBER_NAME}

    with (
        patch.object(intake_mod, "download_remote_msa_bundle", side_effect=spy_download) as mock_download,
        patch.object(
            intake_mod, "project_from_remote", return_value=(expected_local, expected_projected)
        ) as mock_remote,
    ):
        local_loc, projected = prepare_folding_msa_input(
            runspec, workspace, artifact_set=artifact_set, lz4_argv=("/usr/bin/lz4", "-d")
        )
    assert mock_remote.called
    # Verify download_remote_msa_bundle was passed as the download kwarg
    _, kwargs = mock_remote.call_args
    assert kwargs["download"] is mock_download
    assert local_loc is expected_local
    assert projected is expected_projected


# ---------------------------------------------------------------------------
# Folding backward-compat + prediction prefix deferred enforcement
# ---------------------------------------------------------------------------


def test_folding_payload_backward_compat_no_prefix() -> None:
    """An old folding Plan payload (no prefix) constructs without error."""
    msa = _make_msa_set()
    payload = FoldingPhasePlanPayload(msa_set=msa, backend="openfold-cli")
    assert payload.s3_prediction_prefix is None
    assert payload.transport == "publish-to-s3"


def test_folding_payload_prediction_prefix_round_trip() -> None:
    """A folding Plan payload with a prediction prefix round-trips."""
    msa = _make_msa_set()
    payload = FoldingPhasePlanPayload(
        msa_set=msa, backend="openfold-cli", s3_prediction_prefix="s3://bucket/predictions"
    )
    mapping = payload.to_mapping()
    assert mapping["s3_prediction_prefix"] == "s3://bucket/predictions"
    from bspp.orchestration.contract.phase import folding_phase_plan_payload_from_mapping

    reloaded = folding_phase_plan_payload_from_mapping(mapping)
    assert reloaded == payload
    assert reloaded.s3_prediction_prefix == "s3://bucket/predictions"


def test_folding_payload_prediction_prefix_rejects_non_s3() -> None:
    """A non-s3:// prediction prefix is rejected."""
    msa = _make_msa_set()
    with pytest.raises(ValueError, match="non-empty s3:// object key prefix"):
        FoldingPhasePlanPayload(msa_set=msa, backend="openfold-cli", s3_prediction_prefix="not-s3")


def test_folding_payload_prediction_prefix_rejects_local_with_prefix() -> None:
    """transport='local' with a prediction prefix is rejected."""
    msa = _make_msa_set()
    with pytest.raises(ValueError, match="s3_prediction_prefix must be None when transport is 'local'"):
        FoldingPhasePlanPayload(
            msa_set=msa, backend="openfold-cli", transport="local", s3_prediction_prefix="s3://bucket/p"
        )


def test_folding_runspec_payload_transport_default() -> None:
    """FoldingPhaseRunSpecPayload defaults to publish-to-s3 transport."""
    from bspp.orchestration.contract.phase import (
        FoldingActionPayload,
        FoldingRuntimeAction,
        PhaseSlurmResources,
    )

    msa = _make_msa_set()
    resources = PhaseSlurmResources(partition="cpu", cpus_per_task=4, memory="16G", time="01:00:00")
    action = FoldingRuntimeAction(
        action_id="msa-flatten-000001",
        dependencies=(),
        resources=resources,
        payload=FoldingActionPayload(action_kind="msa-flatten", params=()),
        action_kind="msa-flatten",
    )
    payload = FoldingPhaseRunSpecPayload(msa_set=msa, backend="openfold-cli", actions=(action,))
    assert payload.transport == "publish-to-s3"
    assert payload.s3_prediction_prefix is None
    # Conditional serialization: transport omitted when at default
    mapping = payload.to_mapping()
    assert "transport" not in mapping
