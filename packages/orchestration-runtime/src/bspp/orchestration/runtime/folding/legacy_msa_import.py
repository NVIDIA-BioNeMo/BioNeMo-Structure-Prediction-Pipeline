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

"""Runtime legacy-MSA import job for length enrichment.

This module consumes the frozen ``bspp.msa-set/v1`` preprocessing handoff and
atomically publishes an *enriched* Artifact Set manifest carrying
``member_lengths`` plus rebound location/content-validation records that still
reference the unchanged payload bytes.  The legacy handoff records, the lz4
bundle, and the durable tar are never opened for write.

The worker deliberately imports only the contract and the folding runtime; it
never imports the Control Plane.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.preprocessing_action import preprocessing_command_digest
from bspp.orchestration.contract.preprocessing_handoff import (
    MsaArtifactSetManifest,
    PreprocessingContentValidationEvidence,
    PreprocessingHandoffBundle,
    VerifiedLocalBundledArtifactLocation,
    msa_artifact_set_id,
    msa_artifact_set_manifest_from_mapping,
    msa_chunk_manifest_from_mapping,
    preprocessing_content_validation_evidence_from_mapping,
    preprocessing_content_validation_evidence_id,
    verified_local_bundled_artifact_location_from_mapping,
    verified_local_bundled_artifact_location_id,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION
from bspp.orchestration.runtime.folding.execution.a3m_split import a3m_target_length
from bspp.orchestration.runtime.folding.execution.errors import FoldingBackendError
from bspp.orchestration.runtime.folding.execution.msa_projection import (
    project_msa_members,
    verify_bundled_artifact_set,
)

Clock = Callable[[], datetime]

_STREAM_BLOCK_SIZE = 1024 * 1024
_MAX_RECORD_BYTES = 16 * 1024 * 1024  # 16 MiB per metadata record; payload verification is streamed.


class LegacyMsaImportError(RuntimeError):
    """The legacy-MSA import could not verify or publish an enriched handoff."""


@dataclass(frozen=True)
class LegacyMsaImportResult:
    """Enriched handoff identity returned by one successful import."""

    artifact_set_id: str
    artifact_location_id: str
    member_lengths: tuple[int, ...]
    enriched_manifest: MsaArtifactSetManifest
    rebound_location: VerifiedLocalBundledArtifactLocation
    rebound_validation: PreprocessingContentValidationEvidence


def run_legacy_msa_import(
    handoff_root: Path,
    output_dir: Path,
    *,
    lz4_argv: tuple[str, ...] = ("lz4", "-d", "-c"),
    clock: Clock | None = None,
) -> LegacyMsaImportResult:
    """Verify a legacy MSA handoff and publish an enriched, rebound handoff.

    ``lz4_argv`` is the executable-plus-flags prefix passed to the verifier; the
    bundle path is appended by the verifier.  It must stream the decompressed
    tar to stdout, so the default carries ``-c``.
    """
    now = clock or _utc_now
    started_at = _format_timestamp(now())
    try:
        bundle = _load_handoff(handoff_root)
    except (OSError, ValueError) as exc:
        raise LegacyMsaImportError(f"failed to load legacy MSA handoff: {exc}") from exc
    if bundle.artifact_set.has_member_lengths():
        raise LegacyMsaImportError("already enriched")

    consumption = _consumption_for(bundle)
    try:
        verify_bundled_artifact_set(
            consumption,
            bundle.artifact_set,
            bundle.artifact_location,
            lz4_argv=lz4_argv,
        )
        with tempfile.TemporaryDirectory(prefix="bspp-legacy-msa-import-") as temp_dir:
            projected = project_msa_members(
                consumption,
                bundle.artifact_set,
                bundle.artifact_location,
                Path(temp_dir),
            )
            member_lengths = tuple(
                a3m_target_length(projected[member.logical_path]) for member in bundle.chunk_manifest.members
            )
    except (FoldingBackendError, OSError, ValueError) as exc:
        raise LegacyMsaImportError(f"failed to verify or derive member lengths: {exc}") from exc

    enriched_manifest = _enrich(bundle, member_lengths)

    full_lz4_argv = (*lz4_argv, bundle.artifact_location.bundle_path)
    decompressed_sha256, decompressed_size, lz4_return_code, lz4_error = _stream_lz4_digest(full_lz4_argv)
    if lz4_return_code != 0:
        raise LegacyMsaImportError(f"LZ4 verification failed with return code {lz4_return_code}: {lz4_error}")
    if (
        decompressed_sha256 != bundle.artifact_location.tar_sha256
        or decompressed_size != bundle.artifact_location.tar_size_bytes
    ):
        raise LegacyMsaImportError("LZ4 payload does not reproduce the exact durable tar")

    finished_at = _format_timestamp(now())
    rebound_location = _rebind_location(bundle, enriched_manifest, verified_at=finished_at)
    rebound_validation = _rebind_validation(
        bundle,
        enriched_manifest,
        rebound_location,
        started_at=started_at,
        finished_at=finished_at,
        lz4_command_digest=preprocessing_command_digest(full_lz4_argv, ()),
        lz4_return_code=lz4_return_code,
        decompressed_tar_size_bytes=decompressed_size,
        decompressed_tar_sha256=decompressed_sha256,
    )

    try:
        _publish_records(
            output_dir,
            {
                Path("artifact-set.json"): enriched_manifest.to_mapping(),
                Path("artifact-location.json"): rebound_location.to_mapping(),
                Path("content-validation.json"): rebound_validation.to_mapping(),
            },
        )
    except (OSError, ValueError) as exc:
        raise LegacyMsaImportError(f"failed to publish enriched handoff: {exc}") from exc

    return LegacyMsaImportResult(
        artifact_set_id=enriched_manifest.artifact_set_id,
        artifact_location_id=rebound_location.artifact_location_id,
        member_lengths=member_lengths,
        enriched_manifest=enriched_manifest,
        rebound_location=rebound_location,
        rebound_validation=rebound_validation,
    )


def _consumption_for(bundle: PreprocessingHandoffBundle) -> MsaSetConsumption:
    return MsaSetConsumption(
        artifact_set_id=bundle.artifact_set.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=tuple(member.logical_path for member in bundle.chunk_manifest.members),
        requires_paired_query_header=True,
    )


def _load_handoff(handoff_root: Path) -> PreprocessingHandoffBundle:
    if handoff_root.is_symlink() or not handoff_root.is_dir():
        raise LegacyMsaImportError(f"missing legacy MSA handoff directory: {handoff_root}")
    artifact_set = msa_artifact_set_manifest_from_mapping(_read_record(handoff_root / "artifact-set.json"))
    reference = artifact_set.chunks[0]
    chunk_relative = Path(reference.logical_path)
    expected_files = {
        chunk_relative,
        Path("artifact-set.json"),
        Path("artifact-location.json"),
        Path("content-validation.json"),
    }
    expected_directories = {Path("chunks")}
    observed_files: set[Path] = set()
    observed_directories: set[Path] = set()
    for candidate in handoff_root.rglob("*"):
        relative = candidate.relative_to(handoff_root)
        if candidate.is_symlink():
            raise LegacyMsaImportError(f"legacy MSA handoff must not contain symlinks: {relative}")
        if candidate.is_file():
            observed_files.add(relative)
        elif candidate.is_dir():
            observed_directories.add(relative)
        else:
            raise LegacyMsaImportError(f"legacy MSA handoff contains an unsupported entry: {relative}")
    if observed_files != expected_files or observed_directories != expected_directories:
        raise LegacyMsaImportError("legacy MSA handoff must contain the exact successful four-file layout")
    return PreprocessingHandoffBundle(
        chunk_manifest=msa_chunk_manifest_from_mapping(_read_record(handoff_root / chunk_relative)),
        artifact_set=artifact_set,
        artifact_location=verified_local_bundled_artifact_location_from_mapping(
            _read_record(handoff_root / "artifact-location.json")
        ),
        content_validation=preprocessing_content_validation_evidence_from_mapping(
            _read_record(handoff_root / "content-validation.json")
        ),
    )


def _read_record(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise LegacyMsaImportError(f"legacy MSA handoff record must be a regular file: {path}")
    if path.stat().st_size > _MAX_RECORD_BYTES:
        raise LegacyMsaImportError(f"legacy MSA handoff record exceeds the bounded read limit: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LegacyMsaImportError(f"legacy MSA handoff record is missing or invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LegacyMsaImportError(f"legacy MSA handoff record must be a JSON object: {path}")
    return payload


def _enrich(bundle: PreprocessingHandoffBundle, member_lengths: tuple[int, ...]) -> MsaArtifactSetManifest:
    legacy = bundle.artifact_set
    new_id = msa_artifact_set_id(
        legacy.chunks,
        legacy.member_count,
        legacy.logical_bytes,
        member_lengths=member_lengths,
    )
    return MsaArtifactSetManifest(
        artifact_set_id=new_id,
        chunks=legacy.chunks,
        member_count=legacy.member_count,
        logical_bytes=legacy.logical_bytes,
        member_lengths=member_lengths,
    )


def _rebind_location(
    bundle: PreprocessingHandoffBundle,
    enriched_manifest: MsaArtifactSetManifest,
    *,
    verified_at: str,
) -> VerifiedLocalBundledArtifactLocation:
    legacy = bundle.artifact_location
    new_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=enriched_manifest.artifact_set_id,
        tar_path=legacy.tar_path,
        bundle_path=legacy.bundle_path,
        bundle_uri=legacy.bundle_uri,
        tar_size_bytes=legacy.tar_size_bytes,
        tar_sha256=legacy.tar_sha256,
        lz4_size_bytes=legacy.lz4_size_bytes,
        lz4_sha256=legacy.lz4_sha256,
        raw_tar_members=legacy.raw_tar_members,
        members=legacy.members,
    )
    return VerifiedLocalBundledArtifactLocation(
        artifact_location_id=new_id,
        artifact_set_id=enriched_manifest.artifact_set_id,
        tar_path=legacy.tar_path,
        bundle_path=legacy.bundle_path,
        bundle_uri=legacy.bundle_uri,
        tar_size_bytes=legacy.tar_size_bytes,
        tar_sha256=legacy.tar_sha256,
        lz4_size_bytes=legacy.lz4_size_bytes,
        lz4_sha256=legacy.lz4_sha256,
        raw_tar_members=legacy.raw_tar_members,
        members=legacy.members,
        verified_at=verified_at,
    )


def _rebind_validation(
    bundle: PreprocessingHandoffBundle,
    enriched_manifest: MsaArtifactSetManifest,
    rebound_location: VerifiedLocalBundledArtifactLocation,
    *,
    started_at: str,
    finished_at: str,
    lz4_command_digest: str,
    lz4_return_code: int,
    decompressed_tar_size_bytes: int,
    decompressed_tar_sha256: str,
) -> PreprocessingContentValidationEvidence:
    legacy = bundle.content_validation
    body: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "phase_run_id": legacy.phase_run_id,
        "attempt_id": legacy.attempt_id,
        "phase_runspec_digest": legacy.phase_runspec_digest,
        "action_id": legacy.action_id,
        "action_evidence_digest": legacy.action_evidence_digest,
        "chunk_name": legacy.chunk_name,
        "started_at": started_at,
        "finished_at": finished_at,
        "outcome": "passed",
        "artifact_set_id": enriched_manifest.artifact_set_id,
        "artifact_location_id": rebound_location.artifact_location_id,
        "chunk_manifest_digest": legacy.chunk_manifest_digest,
        "artifact_set_manifest_digest": canonical_mapping_digest(enriched_manifest.to_mapping()),
        "member_count": enriched_manifest.member_count,
        "logical_bytes": enriched_manifest.logical_bytes,
        "lz4_command_digest": lz4_command_digest,
        "lz4_return_code": lz4_return_code,
        "decompressed_tar_size_bytes": decompressed_tar_size_bytes,
        "decompressed_tar_sha256": decompressed_tar_sha256,
        "error": None,
    }
    return PreprocessingContentValidationEvidence(
        evidence_id=preprocessing_content_validation_evidence_id(body),
        phase_run_id=legacy.phase_run_id,
        attempt_id=legacy.attempt_id,
        phase_runspec_digest=legacy.phase_runspec_digest,
        action_id=legacy.action_id,
        action_evidence_digest=legacy.action_evidence_digest,
        chunk_name=legacy.chunk_name,
        started_at=started_at,
        finished_at=finished_at,
        outcome="passed",
        artifact_set_id=enriched_manifest.artifact_set_id,
        artifact_location_id=rebound_location.artifact_location_id,
        chunk_manifest_digest=legacy.chunk_manifest_digest,
        artifact_set_manifest_digest=canonical_mapping_digest(enriched_manifest.to_mapping()),
        member_count=enriched_manifest.member_count,
        logical_bytes=enriched_manifest.logical_bytes,
        lz4_command_digest=lz4_command_digest,
        lz4_return_code=lz4_return_code,
        decompressed_tar_size_bytes=decompressed_tar_size_bytes,
        decompressed_tar_sha256=decompressed_tar_sha256,
        error=None,
    )


def _stream_lz4_digest(argv: tuple[str, ...]) -> tuple[str, int, int, str]:
    with tempfile.TemporaryFile() as error_handle:
        try:
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=error_handle)
        except OSError as exc:
            raise LegacyMsaImportError(f"failed to start LZ4 verification: {exc}") from exc
        assert process.stdout is not None
        digest = hashlib.sha256()
        size = 0
        while block := process.stdout.read(_STREAM_BLOCK_SIZE):
            digest.update(block)
            size += len(block)
        return_code = process.wait()
        error_handle.seek(0)
        error = error_handle.read().decode("utf-8", errors="replace").strip()
    return digest.hexdigest(), size, return_code, error


def _publish_records(output_dir: Path, records: Mapping[Path, Mapping[str, object]]) -> None:
    # The Control Plane pre-creates ``output_dir`` so the writable container
    # mount source exists before submission.  Accept an existing directory only
    # when it is empty; any pre-existing entry (including a prior run's three
    # records) keeps the fail-closed overwrite protection.
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LegacyMsaImportError(f"failed to create output directory: {output_dir}") from exc
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise LegacyMsaImportError(f"output path is not a regular directory: {output_dir}")
    try:
        existing_entries = tuple(output_dir.iterdir())
    except OSError as exc:
        raise LegacyMsaImportError(f"failed to inspect output directory: {output_dir}") from exc
    if existing_entries:
        raise LegacyMsaImportError(f"refusing to reuse a non-empty output directory: {output_dir}")
    try:
        for relative, payload in records.items():
            destination = output_dir / relative
            encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
            fd, temporary = tempfile.mkstemp(dir=output_dir, prefix=".legacy-msa-import-", suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                if os.path.lexists(temporary):
                    os.unlink(temporary)
        _fsync_directory(output_dir)
    except OSError as exc:
        raise LegacyMsaImportError(f"failed to publish enriched handoff: {exc}") from exc


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
        raise LegacyMsaImportError("legacy MSA import clock must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = [
    "LegacyMsaImportError",
    "LegacyMsaImportResult",
    "run_legacy_msa_import",
]
