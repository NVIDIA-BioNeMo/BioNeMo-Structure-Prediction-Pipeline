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

"""Tests for pure HQ publication gate-A validation."""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.contract.runspec import runspec_from_mapping
from bspp.orchestration.runtime.data_movement.s3.inventory import S3InventoryObject, SampleHashEvidence
from bspp.orchestration.runtime.postprocessing.hq_publication import (
    HqChunkPublicationItem,
    HqChunkUploadRow,
    plan_hq_chunk_publication,
    write_hq_upload_manifest,
)
from bspp.orchestration.runtime.validation.hq_publication import validate_hq_chunk_publication_evidence
from tests.test_postprocessing.test_hq_publication import _runspec, _write_chunks


def test_hq_publication_gate_passes_matching_evidence(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary"))
    item = plan_hq_chunk_publication(spec, chunks_dir=chunks_dir).items[0]
    upload_manifest = _write_uploaded_manifest(tmp_path / "upload.csv", item)

    report = validate_hq_chunk_publication_evidence(
        spec,
        chunks_dir=chunks_dir,
        upload_manifest_path=upload_manifest,
        remote_inventory=(S3InventoryObject(uri=item.destination_uri, size_bytes=item.size_bytes),),
        sampled_hashes=(SampleHashEvidence(uri=item.destination_uri, sha256=item.sha256, size_bytes=item.size_bytes),),
    )

    assert report.ok
    assert report.expected_chunks == 1


def test_hq_publication_gate_fails_on_foreign_remote_and_hash_mismatch(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary"))
    item = plan_hq_chunk_publication(spec, chunks_dir=chunks_dir).items[0]
    upload_manifest = _write_uploaded_manifest(tmp_path / "upload.csv", item)

    report = validate_hq_chunk_publication_evidence(
        spec,
        chunks_dir=chunks_dir,
        upload_manifest_path=upload_manifest,
        remote_inventory=(
            S3InventoryObject(uri=item.destination_uri, size_bytes=item.size_bytes),
            S3InventoryObject(uri="s3://example-bucket/users/example-user/hq-canary/stray.tar", size_bytes=1),
        ),
        sampled_hashes=(SampleHashEvidence(uri=item.destination_uri, sha256="0" * 64),),
    )

    assert not report.ok
    assert any("foreign remote object" in failure for failure in report.failures)
    assert any("sampled sha256 mismatch" in failure for failure in report.failures)


def test_hq_publication_gate_requires_configured_sampled_hashes(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0", b"chunk-1"))
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary"))
    plan = plan_hq_chunk_publication(spec, chunks_dir=chunks_dir)
    upload_manifest = tmp_path / "upload.csv"
    write_hq_upload_manifest(
        upload_manifest,
        tuple(_uploaded_row(item) for item in plan.items),
    )

    report = validate_hq_chunk_publication_evidence(
        spec,
        chunks_dir=chunks_dir,
        upload_manifest_path=upload_manifest,
        remote_inventory=tuple(
            S3InventoryObject(uri=item.destination_uri, size_bytes=item.size_bytes) for item in plan.items
        ),
        sampled_hashes=(),
    )

    assert not report.ok
    assert any("sampled hash evidence missing required destination" in failure for failure in report.failures)


def test_hq_publication_gate_rejects_unexpected_upload_manifest_destination(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary"))
    item = plan_hq_chunk_publication(spec, chunks_dir=chunks_dir).items[0]
    upload_manifest = tmp_path / "upload.csv"
    write_hq_upload_manifest(
        upload_manifest,
        (
            _uploaded_row(item),
            HqChunkUploadRow(
                chunk_index=99,
                source_path=tmp_path / "extra.tar",
                destination_uri="s3://example-bucket/users/example-user/hq-canary/chunk_0099.tar",
                size_bytes=1,
                sha256="1" * 64,
                payload_sha256="2" * 64,
                status="uploaded",
                attempts=1,
                remote_size_bytes=1,
                timestamp_utc="2026-06-01T00:00:00+00:00",
            ),
        ),
    )

    report = validate_hq_chunk_publication_evidence(
        spec,
        chunks_dir=chunks_dir,
        upload_manifest_path=upload_manifest,
        remote_inventory=(S3InventoryObject(uri=item.destination_uri, size_bytes=item.size_bytes),),
        sampled_hashes=(SampleHashEvidence(uri=item.destination_uri, sha256=item.sha256, size_bytes=item.size_bytes),),
    )

    assert not report.ok
    assert any("upload manifest has unexpected destination" in failure for failure in report.failures)


def _write_uploaded_manifest(path: Path, item: HqChunkPublicationItem) -> Path:
    return write_hq_upload_manifest(
        path,
        (_uploaded_row(item),),
    )


def _uploaded_row(item: HqChunkPublicationItem) -> HqChunkUploadRow:
    return HqChunkUploadRow(
        chunk_index=item.chunk_index,
        source_path=item.source_path,
        destination_uri=item.destination_uri,
        size_bytes=item.size_bytes,
        sha256=item.sha256,
        payload_sha256=item.payload_sha256,
        status="uploaded",
        attempts=1,
        remote_size_bytes=item.size_bytes,
        timestamp_utc="2026-06-01T00:00:00+00:00",
    )
