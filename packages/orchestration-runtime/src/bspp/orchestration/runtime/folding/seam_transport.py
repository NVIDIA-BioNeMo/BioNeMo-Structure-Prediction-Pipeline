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

"""Phase-seam transport wiring and the MSA-set publication leg.

This module maps a per-seam transport policy onto the existing
data-movement adapters and publishes the content-addressed MSA-set bundle to
S3 with bounded upload evidence plus the
``VerifiedRemoteBundledArtifactLocation`` record.

The only subprocess/network surface is the injectable ``transfer`` boundary;
nothing here calls ``require_tool`` or touches ``s5cmd`` until a transfer is
actually invoked. Import this module directly (the folding package anchor
deliberately does not re-export it).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.operator_data_movement import resolve_seam_transfer_decision
from bspp.orchestration.contract.phase import PhaseSeamTransportPolicy
from bspp.orchestration.contract.prediction_bundle import PredictionArchiveBundle
from bspp.orchestration.contract.preprocessing_handoff import (
    VerifiedLocalBundledArtifactLocation,
    VerifiedRemoteBundledArtifactLocation,
    verified_remote_bundled_artifact_location_id,
)
from bspp.orchestration.runtime.data_movement.common import PlannedTransfer, TransferResult
from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer

TransferCallable = Callable[[str, str], TransferResult | PlannedTransfer]


class SeamTransportError(RuntimeError):
    """Fail-closed error surface for phase-seam transport decisions."""


@dataclass(frozen=True)
class SeamTransferResolution:
    """The resolved wiring for one phase-seam transport policy."""

    policy: PhaseSeamTransportPolicy
    object_storage_leg: bool
    tool: str | None
    transfer: Callable[..., TransferResult | PlannedTransfer] | None
    note: str


def resolve_seam_transfer(*, policy: PhaseSeamTransportPolicy) -> SeamTransferResolution:
    """Resolve a seam transport policy to its concrete data-movement wiring.

    Delegates the pure decision to
    :func:`bspp.orchestration.contract.operator_data_movement.resolve_seam_transfer_decision`
    and attaches the runtime transfer callable. ``ValueError`` from the contract
    function is re-raised as :class:`SeamTransportError` to preserve the existing
    exception contract.
    """
    try:
        decision = resolve_seam_transfer_decision(policy=policy)
    except ValueError as exc:
        raise SeamTransportError(str(exc)) from exc
    if decision.tool == "s5cmd":
        transfer: TransferCallable | None = s3_transfer.cp
    elif decision.tool is None:
        transfer = None
    else:
        raise SeamTransportError(f"unsupported transfer tool for policy {policy!r}: {decision.tool!r}")
    return SeamTransferResolution(
        policy=decision.policy,
        object_storage_leg=decision.object_storage_leg,
        tool=decision.tool,
        transfer=transfer,
        note=decision.note,
    )


@dataclass(frozen=True)
class MsaSetUploadEvidence:
    """Bounded evidence for one MSA-set bundle upload."""

    object_key: str
    size_bytes: int
    sha256: str
    transfer_result: TransferResult
    phase_run_id: str | None = None
    attempt_id: str | None = None

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "object_key": self.object_key,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "tool": self.transfer_result.tool,
            "argv": list(self.transfer_result.argv),
            "returncode": self.transfer_result.returncode,
            "elapsed_s": self.transfer_result.elapsed_s,
        }
        if self.phase_run_id is not None:
            result["phase_run_id"] = self.phase_run_id
        if self.attempt_id is not None:
            result["attempt_id"] = self.attempt_id
        return result


def _hash_bundle(path: Path) -> str:
    """Stream the bundle's sha256 (8 MiB chunks) for the publish-time re-check."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_msa_set_to_s3(
    *,
    location: VerifiedLocalBundledArtifactLocation,
    s3_prefix: str,
    transfer: TransferCallable | None = None,
    phase_run_id: str | None = None,
    attempt_id: str | None = None,
) -> tuple[VerifiedRemoteBundledArtifactLocation, MsaSetUploadEvidence]:
    """Publish the verified local MSA-set bundle to S3 and record evidence.

    The destination object key is the content address of the uploaded LZ4
    bundle (``<prefix>/<lz4_sha256>.tar.lz4``). The location record carries the
    upstream verification, but the local file is mutable: this leg re-verifies
    the bundle at publish time (size preflight plus a streamed sha256 re-hash)
    and fails closed on any drift, so the returned remote record and upload
    evidence can never attest stale content identities (TOCTOU between the
    upstream verification and this upload).
    """
    stripped_prefix = s3_prefix.rstrip("/")
    if not s3_prefix.startswith("s3://") or stripped_prefix == "s3://":
        raise SeamTransportError("s3_prefix must be a non-empty s3:// object key prefix")

    dst = f"{stripped_prefix}/{location.lz4_sha256}.tar.lz4"

    bundle_path = Path(location.bundle_path)
    if not bundle_path.exists() or bundle_path.is_symlink() or not bundle_path.is_file():
        raise SeamTransportError(f"MSA-set bundle is missing or not a regular file: {bundle_path}")
    if bundle_path.stat().st_size != location.lz4_size_bytes:
        raise SeamTransportError(
            f"MSA-set bundle size {bundle_path.stat().st_size} does not match "
            f"the verified lz4_size_bytes {location.lz4_size_bytes}"
        )
    actual_sha256 = _hash_bundle(bundle_path)
    if actual_sha256 != location.lz4_sha256:
        raise SeamTransportError(
            f"MSA-set bundle content does not match the verified lz4_sha256 {location.lz4_sha256} "
            f"(recomputed {actual_sha256}); refusing to publish stale bytes"
        )

    transfer_fn = transfer if transfer is not None else s3_transfer.cp
    result = transfer_fn(str(bundle_path), dst)
    if isinstance(result, PlannedTransfer):
        raise SeamTransportError("MSA-set publication requires an executed transfer, not a dry-run plan")
    if not result.ok:
        raise SeamTransportError(
            f"MSA-set upload failed with returncode {result.returncode}: {result.stderr_tail.strip()}"
        )

    remote_id = verified_remote_bundled_artifact_location_id(
        artifact_set_id=location.artifact_set_id,
        bundle_uri=dst,
        tar_size_bytes=location.tar_size_bytes,
        tar_sha256=location.tar_sha256,
        lz4_size_bytes=location.lz4_size_bytes,
        lz4_sha256=location.lz4_sha256,
        raw_tar_members=location.raw_tar_members,
        members=location.members,
    )
    remote = VerifiedRemoteBundledArtifactLocation(
        artifact_location_id=remote_id,
        artifact_set_id=location.artifact_set_id,
        bundle_uri=dst,
        tar_size_bytes=location.tar_size_bytes,
        tar_sha256=location.tar_sha256,
        lz4_size_bytes=location.lz4_size_bytes,
        lz4_sha256=location.lz4_sha256,
        raw_tar_members=location.raw_tar_members,
        members=location.members,
        verified_at=location.verified_at,
    )
    evidence = MsaSetUploadEvidence(
        object_key=dst,
        size_bytes=location.lz4_size_bytes,
        sha256=location.lz4_sha256,
        transfer_result=result,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
    )
    return remote, evidence


@dataclass(frozen=True)
class PredictionBundleUploadEvidence:
    """Bounded evidence for one prediction-bundle upload to S3.

    The ``operator_attested`` flag records that the bundle was supplied by
    an operator-attested source (always ``True`` for the current
    ``publish-folding`` command, which only accepts operator-attested bundles).
    """

    bundle_name: str
    object_key: str
    size_bytes: int
    sha256: str
    member_count: int
    operator_attested: bool = True
    transfer_result: TransferResult | None = None
    phase_run_id: str | None = None
    attempt_id: str | None = None

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "bundle_name": self.bundle_name,
            "object_key": self.object_key,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "member_count": self.member_count,
            "operator_attested": self.operator_attested,
        }
        if self.transfer_result is not None:
            result["tool"] = self.transfer_result.tool
            result["argv"] = list(self.transfer_result.argv)
            result["returncode"] = self.transfer_result.returncode
            result["elapsed_s"] = self.transfer_result.elapsed_s
        if self.phase_run_id is not None:
            result["phase_run_id"] = self.phase_run_id
        if self.attempt_id is not None:
            result["attempt_id"] = self.attempt_id
        return result


def publish_prediction_bundles_to_s3(
    *,
    bundles: tuple[PredictionArchiveBundle, ...],
    local_paths: tuple[str, ...],
    s3_prefix: str,
    transfer: TransferCallable | None = None,
    phase_run_id: str | None = None,
    attempt_id: str | None = None,
) -> tuple[PredictionBundleUploadEvidence, ...]:
    """Publish operator-attested prediction bundles to S3 and record evidence.

    Each bundle is uploaded to ``<prefix>/<sha256>/<bundle_name>`` (content-addressed
    by the bundle's sha256). The local file is re-hashed at publish time and fails
    closed on any drift, mirroring the MSA-set publication leg's TOCTOU guard.
    """
    if len(bundles) != len(local_paths):
        raise SeamTransportError("bundles and local_paths must have the same length")
    if not bundles:
        raise SeamTransportError("at least one prediction bundle is required")
    stripped_prefix = s3_prefix.rstrip("/")
    if not s3_prefix.startswith("s3://") or stripped_prefix == "s3://":
        raise SeamTransportError("s3_prefix must be a non-empty s3:// object key prefix")

    transfer_fn = transfer if transfer is not None else s3_transfer.cp
    all_evidence: list[PredictionBundleUploadEvidence] = []
    for bundle, local_path in zip(bundles, local_paths, strict=True):
        path = Path(local_path)
        if not path.exists() or path.is_symlink() or not path.is_file():
            raise SeamTransportError(f"prediction bundle is missing or not a regular file: {path}")
        if path.stat().st_size != bundle.size_bytes:
            raise SeamTransportError(
                f"prediction bundle {bundle.bundle_name} size {path.stat().st_size} does not match "
                f"the declared size_bytes {bundle.size_bytes}"
            )
        actual_sha256 = _hash_bundle(path)
        if actual_sha256 != bundle.sha256:
            raise SeamTransportError(
                f"prediction bundle {bundle.bundle_name} content does not match the declared sha256 "
                f"{bundle.sha256} (recomputed {actual_sha256}); refusing to publish stale bytes"
            )
        dst = f"{stripped_prefix}/{bundle.sha256}/{bundle.bundle_name}"
        result = transfer_fn(str(path), dst)
        if isinstance(result, PlannedTransfer):
            raise SeamTransportError("prediction-bundle publication requires an executed transfer, not a dry-run plan")
        if not result.ok:
            raise SeamTransportError(
                f"prediction bundle {bundle.bundle_name} upload failed with returncode {result.returncode}: "
                f"{result.stderr_tail.strip()}"
            )
        evidence = PredictionBundleUploadEvidence(
            bundle_name=bundle.bundle_name,
            object_key=dst,
            size_bytes=bundle.size_bytes,
            sha256=bundle.sha256,
            member_count=bundle.member_count,
            operator_attested=True,
            transfer_result=result,
            phase_run_id=phase_run_id,
            attempt_id=attempt_id,
        )
        all_evidence.append(evidence)
    return tuple(all_evidence)


__all__ = [
    "MsaSetUploadEvidence",
    "PredictionBundleUploadEvidence",
    "SeamTransferResolution",
    "SeamTransportError",
    "publish_msa_set_to_s3",
    "publish_prediction_bundles_to_s3",
    "resolve_seam_transfer",
]
