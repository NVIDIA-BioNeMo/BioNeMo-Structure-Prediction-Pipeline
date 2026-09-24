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

"""Verified MSA-set consumption and Attempt-scoped A3M projection.

This module consumes the frozen ``bspp.msa-set/v1`` handoff and materializes
only the declared merged A3Ms into a fresh Attempt-scoped workspace after
verifying the lz4/tar content identities and the exact member inventory
(per the documented contract).  The remote leg downloads a remote bundle through an
injectable transport, derives a local ``VerifiedLocalBundledArtifactLocation``
for the downloaded bytes, and then projects exactly as the local leg does.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.preprocessing_handoff import (
    MsaArtifactSetManifest,
    VerifiedLocalBundledArtifactLocation,
    VerifiedRemoteBundledArtifactLocation,
    verified_local_bundled_artifact_location_id,
)

from .errors import FoldingBackendError

__all__ = ["project_from_remote", "project_msa_members", "verify_bundled_artifact_set"]

_STREAM_BLOCK_SIZE = 1024 * 1024


def _normalize_tar_member(name: str) -> str:
    """Apply the repeated leading-``./`` normalization used by the handoff contract."""
    while name.startswith("./"):
        name = name[2:]
    return name or "."


def _hash_stream(handle: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while block := handle.read(_STREAM_BLOCK_SIZE):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def _hash_path(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise FoldingBackendError(f"bundled input must be a regular file: {path}")
    with path.open("rb") as handle:
        sha256, size = _hash_stream(handle)
    if size <= 0:
        raise FoldingBackendError(f"bundled input must be non-empty: {path}")
    return sha256, size


def _stream_lz4_to_file(argv: tuple[str, ...], destination: Path) -> tuple[str, int, int, str]:
    """Stream lz4 stdout into ``destination`` while hashing its bytes.

    Returns ``(sha256, size, return_code, stderr)`` for the decompressed stream.
    """
    with tempfile.TemporaryFile() as error_handle:
        try:
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=error_handle)
        except OSError as exc:
            raise FoldingBackendError(f"failed to start LZ4 decompression: {exc}") from exc
        assert process.stdout is not None
        digest = hashlib.sha256()
        size = 0
        with destination.open("wb") as handle:
            while block := process.stdout.read(_STREAM_BLOCK_SIZE):
                digest.update(block)
                size += len(block)
                handle.write(block)
        return_code = process.wait()
        error_handle.seek(0)
        error = error_handle.read().decode("utf-8", errors="replace").strip()
    return digest.hexdigest(), size, return_code, error


def _decompress_lz4_to_tar(argv: tuple[str, ...], destination: Path) -> tuple[str, int]:
    sha256, size, return_code, error = _stream_lz4_to_file(argv, destination)
    if return_code != 0:
        raise FoldingBackendError(f"LZ4 decompression failed with return code {return_code}: {error}")
    return sha256, size


def _verify_tar_members(tar_path: Path, location: VerifiedLocalBundledArtifactLocation) -> None:
    with tarfile.open(tar_path, mode="r:") as archive:
        headers = archive.getmembers()
        raw_names = tuple(header.name for header in headers)
        if raw_names != location.raw_tar_members:
            raise FoldingBackendError("tar member inventory does not match the location record")
        by_name = {_normalize_tar_member(header.name): header for header in headers if header.isfile()}
        for member in location.members:
            header = by_name.get(member.member_name)
            if header is None:
                raise FoldingBackendError(f"declared member missing from tar: {member.member_name}")
            stream = archive.extractfile(header)
            if stream is None:
                raise FoldingBackendError(f"unable to read tar member {header.name!r}")
            payload = stream.read()
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != member.size_bytes or digest != member.sha256:
                raise FoldingBackendError(f"tar member does not match its declared verification: {member.member_name}")
        for header in headers:
            if _normalize_tar_member(header.name) != "." and not header.isfile():
                raise FoldingBackendError(f"tar member is not a regular file: {header.name!r}")


def _verify_member_inventory_binding(
    consumption: MsaSetConsumption,
    location: VerifiedLocalBundledArtifactLocation,
) -> None:
    """Require the consumer's declared member order to equal the physical inventory.

    MSA member order is contract-significant: the consumer's
    ``member_a3m_paths`` must be the exact ordered sequence of the location's
    ``members[*].logical_path`` values, not merely an equal count or set.
    """
    declared = tuple(member.logical_path for member in location.members)
    if consumption.member_a3m_paths != declared:
        raise FoldingBackendError("MSA set consumption member paths do not match the bundled location member inventory")


def verify_bundled_artifact_set(
    consumption: MsaSetConsumption,
    artifact_set: MsaArtifactSetManifest,
    location: VerifiedLocalBundledArtifactLocation,
    *,
    lz4_argv: tuple[str, ...],
) -> None:
    """Verify the exact lz4/tar identities and member inventory of a bundled MSA set.

    A failed verification never mutates the producer bundle or its durable tar
    (the bundle is only read and stream-decompressed into a
    disposable temp tar.
    """
    try:
        consumption.validate_against_manifest(artifact_set)
    except ValueError as exc:
        raise FoldingBackendError(f"MSA set consumption does not match its manifest: {exc}") from exc
    if location.artifact_set_id != artifact_set.artifact_set_id:
        raise FoldingBackendError("bundled artifact location does not reference the exact Artifact Set")
    _verify_member_inventory_binding(consumption, location)
    bundle_sha256, bundle_size = _hash_path(Path(location.bundle_path))
    if bundle_sha256 != location.lz4_sha256 or bundle_size != location.lz4_size_bytes:
        raise FoldingBackendError("lz4 bundle does not match its location record")
    argv = (*lz4_argv, location.bundle_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        tar_path = Path(temp_dir) / "bundle.tar"
        tar_sha256, tar_size, return_code, error = _stream_lz4_to_file(argv, tar_path)
        if return_code != 0:
            raise FoldingBackendError(f"LZ4 decompression failed with return code {return_code}: {error}")
        if tar_sha256 != location.tar_sha256 or tar_size != location.tar_size_bytes:
            raise FoldingBackendError("decompressed tar does not match its location record")
        _verify_tar_members(tar_path, location)


def project_msa_members(
    consumption: MsaSetConsumption,
    artifact_set: MsaArtifactSetManifest,
    location: VerifiedLocalBundledArtifactLocation,
    workspace: Path,
) -> dict[str, Path]:
    """Materialize only the declared A3M members into a fresh Attempt workspace.

    The durable ``location.tar_path`` is hashed and projected through one and
    the same open file object: the whole-tar digest is compared against the
    declared ``tar_size_bytes``/``tar_sha256`` first and fails closed before
    ``workspace/a3ms`` is created, so the bytes projected are provably the
    bytes bound to the location record with no re-open window.  Each
    member payload is also checked against its declared size/SHA-256 as it is
    read, before its destination file is written.

    The projection never uses ``tarfile.extractall`` and never uses a tar member
    name as a filesystem path component beyond the pre-validated top-level
    ``member.member_name``.  ``workspace/a3ms`` is created fresh (``exist_ok``
    disabled) so a re-projection into the same workspace fails.  The consumer
    declaration is re-bound to the physical member inventory so callers cannot
    bypass verification and project a location that contradicts the consumer.
    """
    if location.artifact_set_id != artifact_set.artifact_set_id:
        raise FoldingBackendError("bundled artifact location does not reference the exact Artifact Set")
    _verify_member_inventory_binding(consumption, location)
    tar_path = Path(location.tar_path)
    if tar_path.is_symlink() or not tar_path.is_file():
        raise FoldingBackendError(f"bundled input must be a regular file: {tar_path}")
    with tar_path.open("rb") as handle:
        tar_sha256, tar_size = _hash_stream(handle)
        if tar_sha256 != location.tar_sha256 or tar_size != location.tar_size_bytes:
            raise FoldingBackendError("durable tar does not match its location record")
        handle.seek(0)
        a3ms_dir = workspace / "a3ms"
        a3ms_dir.mkdir(parents=True, exist_ok=False)
        declared_by_member_name = {member.member_name: member for member in location.members}
        written: dict[str, Path] = {}
        with tarfile.open(fileobj=handle, mode="r:") as archive:
            for header in archive.getmembers():
                name = _normalize_tar_member(header.name)
                if name == ".":
                    continue
                if not header.isfile():
                    raise FoldingBackendError(f"tar member is not a regular file: {header.name!r}")
                if name.startswith("/") or "/" in name or ".." in name.split("/"):
                    raise FoldingBackendError(f"unsafe tar member path rejected: {header.name!r}")
                member = declared_by_member_name.get(name)
                if member is None:
                    raise FoldingBackendError(f"undeclared tar member rejected: {header.name!r}")
                stream = archive.extractfile(header)
                if stream is None:
                    raise FoldingBackendError(f"unable to read tar member {header.name!r}")
                payload = stream.read()
                if len(payload) != member.size_bytes or hashlib.sha256(payload).hexdigest() != member.sha256:
                    raise FoldingBackendError(
                        f"tar member does not match its declared verification: {member.member_name}"
                    )
                destination = a3ms_dir / member.member_name
                destination.write_bytes(payload)
                written[member.logical_path] = destination
    for member in location.members:
        if member.logical_path not in written:
            raise FoldingBackendError(f"declared member was not projected: {member.member_name}")
    return written


def project_from_remote(
    consumption: MsaSetConsumption,
    artifact_set: MsaArtifactSetManifest,
    remote_location: VerifiedRemoteBundledArtifactLocation,
    workspace: Path,
    *,
    download: Callable[[VerifiedRemoteBundledArtifactLocation, Path], None],
    lz4_argv: tuple[str, ...],
) -> tuple[VerifiedLocalBundledArtifactLocation, dict[str, Path]]:
    """Download, verify, derive a local record, and project a remote MSA bundle.

    ``download`` is an injectable transport ``(remote_location, destination_path)
    -> None``; no S3 client or credentials are hard-wired here.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    bundle_path = (workspace / "bundle.tar.lz4").absolute()
    tar_path = (workspace / "bundle.tar").absolute()
    download(remote_location, bundle_path)
    bundle_uri = bundle_path.as_uri()
    local_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=remote_location.artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=bundle_uri,
        tar_size_bytes=remote_location.tar_size_bytes,
        tar_sha256=remote_location.tar_sha256,
        lz4_size_bytes=remote_location.lz4_size_bytes,
        lz4_sha256=remote_location.lz4_sha256,
        raw_tar_members=remote_location.raw_tar_members,
        members=remote_location.members,
    )
    local = VerifiedLocalBundledArtifactLocation(
        artifact_location_id=local_id,
        artifact_set_id=remote_location.artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=bundle_uri,
        tar_size_bytes=remote_location.tar_size_bytes,
        tar_sha256=remote_location.tar_sha256,
        lz4_size_bytes=remote_location.lz4_size_bytes,
        lz4_sha256=remote_location.lz4_sha256,
        raw_tar_members=remote_location.raw_tar_members,
        members=remote_location.members,
        verified_at=remote_location.verified_at,
    )
    argv = (*lz4_argv, str(bundle_path))
    _decompress_lz4_to_tar(argv, tar_path)
    verify_bundled_artifact_set(consumption, artifact_set, local, lz4_argv=lz4_argv)
    record_path = workspace / "artifact-location.json"
    record_path.write_text(json.dumps(local.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    projected = project_msa_members(consumption, artifact_set, local, workspace)
    return local, projected
