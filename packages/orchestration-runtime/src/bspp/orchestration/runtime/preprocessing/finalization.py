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

"""Finalize one durable preprocessing bundle inside the Execution Runtime."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from bspp.orchestration.contract.phase import PhaseRunSpec, canonical_mapping_digest
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingChunkActionEvidence,
    PreprocessingOutputHash,
    preprocessing_command_digest,
)
from bspp.orchestration.contract.preprocessing_execution import member_name_conforms
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
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION
from bspp.orchestration.runtime.preprocessing._action_evidence_io import load_preprocessing_action_evidence
from bspp.orchestration.runtime.preprocessing.content_validation import (
    normalize_preprocessing_tar_member,
    validate_preprocessing_a3m_header_bytes,
    validate_preprocessing_tar_headers,
)
from bspp.orchestration.runtime.preprocessing.execution import (
    reconcile_preprocessing_chunk_action_evidence_for_finalization,
)

Clock = Callable[[], datetime]


class PreprocessingFinalizationError(RuntimeError):
    """The exact durable preprocessing bundle failed acceptance."""


def finalize_preprocessing_chunk(
    runspec: PhaseRunSpec,
    evidence: PreprocessingChunkActionEvidence,
    *,
    handoff_path: Path,
    clock: Clock | None = None,
) -> PreprocessingHandoffBundle:
    """Validate durable bytes and exclusively publish one successful handoff."""
    now = clock or _utc_now
    started_at = _format_timestamp(now())
    action_evidence_digest = canonical_mapping_digest(evidence.to_mapping())
    try:
        reconcile_preprocessing_chunk_action_evidence_for_finalization(runspec, evidence)
        action = runspec.payload.actions[0]
        expected_by_name = {item.member_name: item for item in action.payload.expected_a3ms}
        records_by_ordinal = {item.source_ordinal: item for item in runspec.payload.work_plan.input.records}
        attested_by_name = _a3m_attestations(evidence)
        tar_path = Path(action.payload.package.durable_tar_path)
        tar_sha256, tar_size = _hash_path(tar_path)
        lz4_path = Path(action.payload.package.durable_lz4_path)
        lz4_sha256, lz4_size = _hash_path(lz4_path)

        raw_names: tuple[str, ...]
        validated: dict[str, tuple[int, str]] = {}
        with tarfile.open(tar_path, mode="r:") as archive:
            headers = validate_preprocessing_tar_headers(archive, action)
            raw_names = tuple(header.name for header in headers)
            if action.payload.scientific.require_afdb_model_id_stem:
                for header in headers:
                    if not header.isfile():
                        continue
                    normalized = normalize_preprocessing_tar_member(header.name)
                    if not member_name_conforms(normalized):
                        raise PreprocessingFinalizationError(
                            "published A3M member stem does not carry a discoverable "
                            f"AFDB or PDB assembly model ID: {header.name!r}"
                        )
            by_name = {normalize_preprocessing_tar_member(header.name): header for header in headers if header.isfile()}
            for expected in sorted(action.payload.expected_a3ms, key=lambda item: item.source_ordinal):
                header = by_name[expected.member_name]
                stream = archive.extractfile(header)
                if stream is None:
                    raise PreprocessingFinalizationError(f"unable to read tar member {header.name!r}")
                payload = stream.read()
                if not payload:
                    raise PreprocessingFinalizationError(f"preprocessing tar member is empty: {header.name!r}")
                record = records_by_ordinal[expected.source_ordinal]
                chain_lengths = tuple(len(chain) for chain in record.sequence.split(":"))
                validate_preprocessing_a3m_header_bytes(
                    payload,
                    chain_lengths=chain_lengths,
                    label=header.name,
                )
                digest = hashlib.sha256(payload).hexdigest()
                attested = attested_by_name[expected.member_name]
                if len(payload) != attested.size_bytes or digest != attested.sha256:
                    raise PreprocessingFinalizationError(
                        f"tar member does not match its #72 A3M attestation: {expected.member_name}"
                    )
                validated[expected.member_name] = (len(payload), digest)

        lz4_argv = (action.payload.package.lz4_argv[0], "-d", "-c", str(lz4_path))
        decompressed_sha256, decompressed_size, lz4_return_code, lz4_error = _stream_lz4_digest(lz4_argv)
        if lz4_return_code != 0:
            raise PreprocessingFinalizationError(
                f"LZ4 verification failed with return code {lz4_return_code}: {lz4_error}"
            )
        if decompressed_sha256 != tar_sha256 or decompressed_size != tar_size:
            raise PreprocessingFinalizationError("LZ4 payload does not reproduce the exact durable tar")
        if _hash_path(tar_path) != (tar_sha256, tar_size) or _hash_path(lz4_path) != (lz4_sha256, lz4_size):
            raise PreprocessingFinalizationError("durable preprocessing bundle changed during finalization")

        members = tuple(
            MsaArtifactMember(
                record_identity=expected.record_identity,
                source_ordinal=expected.source_ordinal,
                source_header=expected.source_header,
                logical_path=f"a3ms/{expected.member_name}",
                size_bytes=validated[expected.member_name][0],
                sha256=validated[expected.member_name][1],
            )
            for expected in sorted(expected_by_name.values(), key=lambda item: item.source_ordinal)
        )
        chunk_manifest = MsaChunkManifest(
            chunk_name=action.payload.chunk_name,
            members=members,
            member_count=len(members),
            logical_bytes=sum(member.size_bytes for member in members),
        )
        chunk_reference = MsaChunkManifestReference(
            chunk_name=chunk_manifest.chunk_name,
            logical_path=f"chunks/{chunk_manifest.chunk_name.removesuffix('.fa')}.json",
            sha256=chunk_manifest.digest,
            member_count=chunk_manifest.member_count,
            logical_bytes=chunk_manifest.logical_bytes,
        )
        artifact_set_id = msa_artifact_set_id(
            (chunk_reference,),
            chunk_manifest.member_count,
            chunk_manifest.logical_bytes,
        )
        artifact_set = MsaArtifactSetManifest(
            artifact_set_id=artifact_set_id,
            chunks=(chunk_reference,),
            member_count=chunk_manifest.member_count,
            logical_bytes=chunk_manifest.logical_bytes,
        )
        bundled_members = tuple(
            BundledMemberVerification(
                logical_path=member.logical_path,
                member_name=member.logical_path.removeprefix("a3ms/"),
                raw_member_name=next(
                    raw
                    for raw in raw_names
                    if normalize_preprocessing_tar_member(raw) == member.logical_path.removeprefix("a3ms/")
                ),
                size_bytes=member.size_bytes,
                sha256=member.sha256,
            )
            for member in members
        )
        finished_at = _format_timestamp(now())
        bundle_path = str(lz4_path)
        bundle_uri = lz4_path.absolute().as_uri()
        location_id = verified_local_bundled_artifact_location_id(
            artifact_set_id=artifact_set.artifact_set_id,
            tar_path=str(tar_path),
            bundle_path=bundle_path,
            bundle_uri=bundle_uri,
            tar_size_bytes=tar_size,
            tar_sha256=tar_sha256,
            lz4_size_bytes=lz4_size,
            lz4_sha256=lz4_sha256,
            raw_tar_members=raw_names,
            members=bundled_members,
        )
        location = VerifiedLocalBundledArtifactLocation(
            artifact_location_id=location_id,
            artifact_set_id=artifact_set.artifact_set_id,
            tar_path=str(tar_path),
            bundle_path=bundle_path,
            bundle_uri=bundle_uri,
            tar_size_bytes=tar_size,
            tar_sha256=tar_sha256,
            lz4_size_bytes=lz4_size,
            lz4_sha256=lz4_sha256,
            raw_tar_members=raw_names,
            members=bundled_members,
            verified_at=finished_at,
        )
        validation_body: dict[str, object] = {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "phase_run_id": runspec.phase_run_id,
            "attempt_id": runspec.attempt_id,
            "phase_runspec_digest": runspec.digest,
            "action_id": action.action_id,
            "action_evidence_digest": action_evidence_digest,
            "chunk_name": action.payload.chunk_name,
            "started_at": started_at,
            "finished_at": finished_at,
            "outcome": "passed",
            "artifact_set_id": artifact_set.artifact_set_id,
            "artifact_location_id": location.artifact_location_id,
            "chunk_manifest_digest": chunk_manifest.digest,
            "artifact_set_manifest_digest": canonical_mapping_digest(artifact_set.to_mapping()),
            "member_count": artifact_set.member_count,
            "logical_bytes": artifact_set.logical_bytes,
            "lz4_command_digest": preprocessing_command_digest(lz4_argv, ()),
            "lz4_return_code": lz4_return_code,
            "decompressed_tar_size_bytes": decompressed_size,
            "decompressed_tar_sha256": decompressed_sha256,
            "error": None,
        }
        validation = PreprocessingContentValidationEvidence(
            evidence_id=preprocessing_content_validation_evidence_id(validation_body),
            phase_run_id=runspec.phase_run_id,
            attempt_id=runspec.attempt_id,
            phase_runspec_digest=runspec.digest,
            action_id=action.action_id,
            action_evidence_digest=action_evidence_digest,
            chunk_name=action.payload.chunk_name,
            started_at=started_at,
            finished_at=finished_at,
            outcome="passed",
            artifact_set_id=artifact_set.artifact_set_id,
            artifact_location_id=location.artifact_location_id,
            chunk_manifest_digest=chunk_manifest.digest,
            artifact_set_manifest_digest=canonical_mapping_digest(artifact_set.to_mapping()),
            member_count=artifact_set.member_count,
            logical_bytes=artifact_set.logical_bytes,
            lz4_command_digest=preprocessing_command_digest(lz4_argv, ()),
            lz4_return_code=lz4_return_code,
            decompressed_tar_size_bytes=decompressed_size,
            decompressed_tar_sha256=decompressed_sha256,
            error=None,
        )
        handoff = PreprocessingHandoffBundle(
            chunk_manifest=chunk_manifest,
            artifact_set=artifact_set,
            artifact_location=location,
            content_validation=validation,
        )
        _publish_handoff(handoff_path, handoff)
        return handoff
    except Exception as exc:
        finished_at = _format_timestamp(now())
        failure_body: dict[str, object] = {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "phase_run_id": runspec.phase_run_id,
            "attempt_id": runspec.attempt_id,
            "phase_runspec_digest": runspec.digest,
            "action_id": runspec.payload.actions[0].action_id,
            "action_evidence_digest": action_evidence_digest,
            "chunk_name": runspec.payload.actions[0].payload.chunk_name,
            "started_at": started_at,
            "finished_at": finished_at,
            "outcome": "failed",
            "artifact_set_id": None,
            "artifact_location_id": None,
            "chunk_manifest_digest": None,
            "artifact_set_manifest_digest": None,
            "member_count": None,
            "logical_bytes": None,
            "lz4_command_digest": None,
            "lz4_return_code": None,
            "decompressed_tar_size_bytes": None,
            "decompressed_tar_sha256": None,
            "error": str(exc),
        }
        failed = PreprocessingContentValidationEvidence(
            evidence_id=preprocessing_content_validation_evidence_id(failure_body),
            phase_run_id=runspec.phase_run_id,
            attempt_id=runspec.attempt_id,
            phase_runspec_digest=runspec.digest,
            action_id=runspec.payload.actions[0].action_id,
            action_evidence_digest=action_evidence_digest,
            chunk_name=runspec.payload.actions[0].payload.chunk_name,
            started_at=started_at,
            finished_at=finished_at,
            outcome="failed",
            artifact_set_id=None,
            artifact_location_id=None,
            chunk_manifest_digest=None,
            artifact_set_manifest_digest=None,
            member_count=None,
            logical_bytes=None,
            lz4_command_digest=None,
            lz4_return_code=None,
            decompressed_tar_size_bytes=None,
            decompressed_tar_sha256=None,
            error=str(exc),
        )
        with suppress(FileExistsError):
            _publish_failed_handoff(handoff_path, failed)
        if isinstance(exc, PreprocessingFinalizationError):
            raise
        raise PreprocessingFinalizationError(str(exc)) from exc


def _a3m_attestations(
    evidence: PreprocessingChunkActionEvidence,
) -> dict[str, PreprocessingOutputHash]:
    result: dict[str, PreprocessingOutputHash] = {}
    for output in evidence.output_hashes:
        if output.role == "a3m":
            assert output.member_name is not None
            result[output.member_name] = output
    return result


def _hash_path(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise PreprocessingFinalizationError(f"durable finalization input must be a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    if size <= 0:
        raise PreprocessingFinalizationError(f"durable finalization input must be non-empty: {path}")
    return digest.hexdigest(), size


def _stream_lz4_digest(argv: tuple[str, ...]) -> tuple[str, int, int, str]:
    with tempfile.TemporaryFile() as error_handle:
        try:
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=error_handle)
        except OSError as exc:
            raise PreprocessingFinalizationError(f"failed to start LZ4 verification: {exc}") from exc
        assert process.stdout is not None
        digest = hashlib.sha256()
        size = 0
        while block := process.stdout.read(1024 * 1024):
            digest.update(block)
            size += len(block)
        return_code = process.wait()
        error_handle.seek(0)
        error = error_handle.read().decode("utf-8", errors="replace").strip()
    return digest.hexdigest(), size, return_code, error


def _publish_handoff(path: Path, handoff: PreprocessingHandoffBundle) -> None:
    files = {
        Path(
            f"chunks/{handoff.chunk_manifest.chunk_name.removesuffix('.fa')}.json"
        ): handoff.chunk_manifest.to_mapping(),
        Path("artifact-set.json"): handoff.artifact_set.to_mapping(),
        Path("artifact-location.json"): handoff.artifact_location.to_mapping(),
        Path("content-validation.json"): handoff.content_validation.to_mapping(),
    }
    _publish_directory(path, files)


def _publish_failed_handoff(path: Path, evidence: PreprocessingContentValidationEvidence) -> None:
    _publish_directory(path, {Path("content-validation.json"): evidence.to_mapping()})


def _publish_directory(path: Path, files: Mapping[Path, Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent))
    try:
        for relative, payload in files.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            with destination.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_tree(staging)
        lock_path = path.parent / ".preprocessing-handoff.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if os.path.lexists(path):
                    raise FileExistsError(f"refusing to replace existing preprocessing handoff: {path}")
                os.rename(staging, path)
                _fsync_directory(path.parent)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _fsync_tree(root: Path) -> None:
    directories = sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_dir()),
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    )
    for directory in (*directories, root):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("preprocessing finalization clock must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = [
    "PreprocessingFinalizationError",
    "finalize_preprocessing_chunk",
    "load_preprocessing_action_evidence",
]
