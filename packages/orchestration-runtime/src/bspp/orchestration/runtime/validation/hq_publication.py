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

"""Pure gate-A validation for HQ chunk publication evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.runspec import RunSpec, resolve_hq_chunk_publication_target
from bspp.orchestration.runtime.data_movement.s3.inventory import (
    S3InventoryObject,
    SampleHashEvidence,
    read_inventory_csv,
    read_sample_hashes_csv,
)
from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary
from bspp.orchestration.runtime.postprocessing.hq_chunks import read_hq_chunk_manifest
from bspp.orchestration.runtime.postprocessing.hq_publication import (
    HqChunkUploadRow,
    read_hq_upload_manifest,
    select_hq_publication_sample_uris,
)


@dataclass(frozen=True, slots=True)
class HqPublicationValidationReport:
    """PASS/FAIL report for pre-collected HQ publication evidence."""

    target_prefix: str
    expected_chunks: int
    uploaded_chunks: int
    remote_objects: int
    sampled_hashes: int
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_redacted_dict(self) -> dict[str, object]:
        return {
            "target_prefix": self.target_prefix,
            "expected_chunks": self.expected_chunks,
            "uploaded_chunks": self.uploaded_chunks,
            "remote_objects": self.remote_objects,
            "sampled_hashes": self.sampled_hashes,
            "ok": self.ok,
            "failures": list(self.failures),
        }


def validate_hq_chunk_publication_evidence(
    spec: RunSpec,
    *,
    chunks_dir: Path,
    upload_manifest_path: Path,
    remote_inventory: tuple[S3InventoryObject, ...],
    sampled_hashes: tuple[SampleHashEvidence, ...],
) -> HqPublicationValidationReport:
    """Compute gate-A PASS/FAIL from local manifests and pre-collected object-store evidence."""
    target_prefix = resolve_hq_chunk_publication_target(spec)
    publication = spec.analysis_metadata.high_quality_from_tars.publication
    if not publication.enabled:
        msg = "HQ chunk publication is disabled in the RunSpec"
        raise ValueError(msg)
    chunks = read_hq_chunk_manifest(chunks_dir / "hq_chunks_manifest.csv")
    upload_rows = read_hq_upload_manifest(upload_manifest_path)
    expected_by_dest = {f"{target_prefix}{chunk.tar_path.name}": chunk for chunk in chunks}
    upload_by_dest = _unique_upload_rows(upload_rows)
    remote_by_uri = {obj.uri: obj for obj in remote_inventory if obj.uri.startswith(target_prefix)}
    hashes_by_uri = {item.uri: item for item in sampled_hashes}
    required_sample_uris = select_hq_publication_sample_uris(
        tuple(expected_by_dest),
        publication.sample_download_count,
    )
    failures: list[str] = []

    if len(upload_by_dest) != len(upload_rows):
        failures.append("upload manifest contains duplicate destination_uri rows")

    for destination, chunk in sorted(expected_by_dest.items()):
        upload = upload_by_dest.get(destination)
        if upload is None:
            failures.append(f"upload manifest missing destination: {destination}")
            continue
        failures.extend(_validate_upload_row(upload, chunk_size=chunk.size_bytes, chunk_sha256=chunk.sha256))
        remote = remote_by_uri.get(destination)
        if remote is None:
            failures.append(f"remote inventory missing destination: {destination}")
        elif remote.size_bytes != chunk.size_bytes:
            failures.append(
                f"remote size mismatch for {destination}: expected={chunk.size_bytes} actual={remote.size_bytes}"
            )
        sampled = hashes_by_uri.get(destination)
        if sampled is not None:
            if sampled.sha256 != chunk.sha256:
                failures.append(f"sampled sha256 mismatch for {destination}")
            if sampled.size_bytes is not None and sampled.size_bytes != chunk.size_bytes:
                failures.append(
                    f"sampled size mismatch for {destination}: expected={chunk.size_bytes} actual={sampled.size_bytes}"
                )

    for uri in sorted(set(upload_by_dest) - set(expected_by_dest)):
        failures.append(f"upload manifest has unexpected destination: {uri}")

    for uri in sorted(remote_by_uri):
        if uri not in expected_by_dest:
            failures.append(f"foreign remote object under prefix: {uri}")

    for uri in sorted(hashes_by_uri):
        if uri not in expected_by_dest:
            failures.append(f"sampled hash evidence is not an expected destination: {uri}")

    for uri in required_sample_uris:
        if uri not in hashes_by_uri:
            failures.append(f"sampled hash evidence missing required destination: {uri}")

    return HqPublicationValidationReport(
        target_prefix=target_prefix,
        expected_chunks=len(chunks),
        uploaded_chunks=len(upload_rows),
        remote_objects=len(remote_by_uri),
        sampled_hashes=len(sampled_hashes),
        failures=tuple(failures),
    )


def validate_hq_chunk_publication_evidence_files(
    spec: RunSpec,
    *,
    chunks_dir: Path,
    upload_manifest_path: Path,
    remote_inventory_path: Path,
    sampled_hashes_path: Path,
) -> HqPublicationValidationReport:
    """Read evidence files and compute gate-A PASS/FAIL."""
    return validate_hq_chunk_publication_evidence(
        spec,
        chunks_dir=chunks_dir,
        upload_manifest_path=upload_manifest_path,
        remote_inventory=read_inventory_csv(remote_inventory_path),
        sampled_hashes=read_sample_hashes_csv(sampled_hashes_path),
    )


def render_hq_publication_validation_report(report: HqPublicationValidationReport) -> str:
    """Render a deterministic JSON validation report."""
    return report_to_json(report)


def write_hq_publication_validation_report(
    report: HqPublicationValidationReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write JSON and text gate-A validation reports."""
    json_path = write_json_report(report, output_dir / "hq_chunks_publication_gate_a.json")
    text_path = write_text_summary(report, output_dir / "hq_chunks_publication_gate_a.txt")
    return json_path, text_path


def _unique_upload_rows(rows: tuple[HqChunkUploadRow, ...]) -> dict[str, HqChunkUploadRow]:
    by_dest: dict[str, HqChunkUploadRow] = {}
    for row in rows:
        by_dest.setdefault(row.destination_uri, row)
    return by_dest


def _validate_upload_row(row: HqChunkUploadRow, *, chunk_size: int, chunk_sha256: str) -> tuple[str, ...]:
    failures: list[str] = []
    if row.size_bytes != chunk_size:
        failures.append(f"upload manifest size mismatch for {row.destination_uri}")
    if row.sha256 != chunk_sha256:
        failures.append(f"upload manifest sha256 mismatch for {row.destination_uri}")
    if row.status not in {"uploaded", "skipped_existing"}:
        failures.append(f"upload manifest destination is not proven uploaded: {row.destination_uri}")
    if row.remote_size_bytes is not None and row.remote_size_bytes != chunk_size:
        failures.append(f"upload manifest remote size mismatch for {row.destination_uri}")
    return tuple(failures)


__all__ = [
    "HqPublicationValidationReport",
    "render_hq_publication_validation_report",
    "validate_hq_chunk_publication_evidence",
    "validate_hq_chunk_publication_evidence_files",
    "write_hq_publication_validation_report",
]
