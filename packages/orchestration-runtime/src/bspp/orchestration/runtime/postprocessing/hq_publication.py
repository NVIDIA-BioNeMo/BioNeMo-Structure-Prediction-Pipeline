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

"""Plan and evidence manifests for publishing local HQ chunk tars."""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bspp.orchestration.contract.runspec import RunSpec, resolve_hq_chunk_publication_target
from bspp.orchestration.runtime.data_movement.common import TransferResult, require_tool
from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer
from bspp.orchestration.runtime.data_movement.s3.client import S3Credentials, load_credentials_from_env
from bspp.orchestration.runtime.data_movement.s3.inventory import (
    S3InventoryObject,
    SampleHashEvidence,
    build_list_prefix_argv,
    parse_s5cmd_ls_output,
    write_inventory_csv,
    write_sample_hashes_csv,
)
from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary
from bspp.orchestration.runtime.postprocessing.failure_adapter import raise_transport_failure_if_audited
from bspp.orchestration.runtime.postprocessing.hq_chunks import HqChunkBuildChunk, read_hq_chunk_manifest

UPLOAD_MANIFEST_NAME = "hq_chunks_upload_manifest.csv"
PRE_UPLOAD_INVENTORY_NAME = "hq_chunks_remote_inventory_before.csv"
POST_UPLOAD_INVENTORY_NAME = "hq_chunks_remote_inventory_after.csv"
SAMPLED_HASHES_NAME = "hq_chunks_sampled_hashes.csv"
SAMPLED_DOWNLOAD_DIR_NAME = "sampled_downloads"

ListPrefixFn = Callable[[str], tuple[S3InventoryObject, ...]]
UploadObjectFn = Callable[[Path, str], None]
DownloadObjectFn = Callable[[str, Path], None]


class HqChunkPublicationExecutionError(RuntimeError):
    """Raised when live HQ chunk publication cannot complete safely."""


@dataclass(frozen=True, slots=True)
class HqChunkUploadRow:
    """One resumable upload-manifest row for an HQ chunk tar."""

    chunk_index: int
    source_path: Path
    destination_uri: str
    size_bytes: int
    sha256: str
    payload_sha256: str
    status: str
    attempts: int = 0
    remote_size_bytes: int | None = None
    timestamp_utc: str | None = None

    def to_redacted_dict(self) -> dict[str, object]:
        return {
            "chunk_index": self.chunk_index,
            "source_path": str(self.source_path),
            "destination_uri": self.destination_uri,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "payload_sha256": self.payload_sha256,
            "status": self.status,
            "attempts": self.attempts,
            "remote_size_bytes": self.remote_size_bytes,
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass(frozen=True, slots=True)
class HqChunkPublicationItem:
    """One planned publication object."""

    chunk_index: int
    source_path: Path
    destination_uri: str
    size_bytes: int
    sha256: str
    payload_sha256: str
    action: str
    reason: str

    def to_upload_row(self) -> HqChunkUploadRow:
        status = "skipped_existing" if self.action == "skip_existing" else "planned"
        return HqChunkUploadRow(
            chunk_index=self.chunk_index,
            source_path=self.source_path,
            destination_uri=self.destination_uri,
            size_bytes=self.size_bytes,
            sha256=self.sha256,
            payload_sha256=self.payload_sha256,
            status=status,
            remote_size_bytes=self.size_bytes if self.action == "skip_existing" else None,
            timestamp_utc=_timestamp(),
        )

    def to_redacted_dict(self) -> dict[str, object]:
        return {
            "chunk_index": self.chunk_index,
            "source_path": str(self.source_path),
            "destination_uri": self.destination_uri,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "payload_sha256": self.payload_sha256,
            "action": self.action,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class HqChunkPublicationPlan:
    """Local publication plan plus collision/resume classification."""

    target_prefix: str
    collision_policy: str
    overwrite: bool
    remote_inventory_collected: bool
    items: tuple[HqChunkPublicationItem, ...]
    failures: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def upload_count(self) -> int:
        return sum(1 for item in self.items if item.action in {"upload", "overwrite"})

    @property
    def skip_count(self) -> int:
        return sum(1 for item in self.items if item.action == "skip_existing")

    def upload_manifest_rows(self) -> tuple[HqChunkUploadRow, ...]:
        return tuple(item.to_upload_row() for item in self.items)

    def to_redacted_dict(self) -> dict[str, object]:
        return {
            "target_prefix": self.target_prefix,
            "collision_policy": self.collision_policy,
            "overwrite": self.overwrite,
            "remote_inventory_collected": self.remote_inventory_collected,
            "ok": self.ok,
            "upload_count": self.upload_count,
            "skip_count": self.skip_count,
            "failures": list(self.failures),
            "items": [item.to_redacted_dict() for item in self.items],
        }


@dataclass(frozen=True, slots=True)
class HqChunkPublicationExecutionReport:
    """Evidence summary for a live HQ chunk publication attempt."""

    target_prefix: str
    plan: HqChunkPublicationPlan
    validation_report: Any
    pre_upload_inventory_path: Path
    post_upload_inventory_path: Path
    upload_manifest_path: Path
    sampled_hashes_path: Path
    sample_download_dir: Path
    uploaded_count: int
    skipped_count: int

    @property
    def ok(self) -> bool:
        return self.plan.ok and bool(getattr(self.validation_report, "ok", False))

    def to_redacted_dict(self) -> dict[str, object]:
        validation = (
            self.validation_report.to_redacted_dict()
            if hasattr(self.validation_report, "to_redacted_dict")
            else str(self.validation_report)
        )
        return {
            "target_prefix": self.target_prefix,
            "ok": self.ok,
            "uploaded_count": self.uploaded_count,
            "skipped_count": self.skipped_count,
            "pre_upload_inventory_path": str(self.pre_upload_inventory_path),
            "post_upload_inventory_path": str(self.post_upload_inventory_path),
            "upload_manifest_path": str(self.upload_manifest_path),
            "sampled_hashes_path": str(self.sampled_hashes_path),
            "sample_download_dir": str(self.sample_download_dir),
            "plan": self.plan.to_redacted_dict(),
            "validation_report": validation,
        }


def plan_hq_chunk_publication(
    spec: RunSpec,
    *,
    chunks_dir: Path,
    upload_manifest_path: Path | None = None,
    remote_inventory: tuple[S3InventoryObject, ...] | None = None,
) -> HqChunkPublicationPlan:
    """Plan HQ chunk publication from local manifest and optional remote evidence."""
    publication = spec.analysis_metadata.high_quality_from_tars.publication
    if not publication.enabled:
        msg = "HQ chunk publication is disabled in the RunSpec"
        raise ValueError(msg)
    target_prefix = resolve_hq_chunk_publication_target(spec)
    chunks = read_hq_chunk_manifest(chunks_dir / "hq_chunks_manifest.csv")
    previous_rows = read_hq_upload_manifest(upload_manifest_path) if upload_manifest_path is not None else ()
    inventory = remote_inventory or ()
    inventory_collected = remote_inventory is not None
    previous_by_dest = {row.destination_uri: row for row in previous_rows}
    remote_by_uri = {obj.uri: obj for obj in inventory}
    items: list[HqChunkPublicationItem] = []
    failures: list[str] = []
    expected_destinations: set[str] = set()

    for chunk in chunks:
        verified = _verify_chunk(chunk)
        destination = f"{target_prefix}{verified.tar_path.name}"
        expected_destinations.add(destination)
        remote = remote_by_uri.get(destination)
        action = "upload"
        reason = "not present in provided remote inventory" if inventory_collected else "remote inventory not collected"
        if remote is not None:
            previous = previous_by_dest.get(destination)
            if _matches_prior_upload(previous, verified, remote):
                action = "skip_existing"
                reason = "matches prior upload manifest and remote inventory"
            elif publication.overwrite:
                action = "overwrite"
                reason = "destination exists and RunSpec overwrite=true"
            else:
                failures.append(f"remote destination exists without verified resume evidence: {destination}")
                action = "blocked"
                reason = "destination collision"
        items.append(
            HqChunkPublicationItem(
                chunk_index=verified.chunk_index,
                source_path=verified.tar_path,
                destination_uri=destination,
                size_bytes=verified.size_bytes,
                sha256=verified.sha256,
                payload_sha256=verified.payload_sha256,
                action=action,
                reason=reason,
            )
        )

    for remote in inventory:
        if not remote.uri.startswith(target_prefix):
            continue
        if remote.uri not in expected_destinations:
            failures.append(f"foreign object under HQ chunk publication prefix: {remote.uri}")

    return HqChunkPublicationPlan(
        target_prefix=target_prefix,
        collision_policy=publication.collision_policy,
        overwrite=publication.overwrite,
        remote_inventory_collected=inventory_collected,
        items=tuple(items),
        failures=tuple(failures),
    )


def execute_hq_chunk_publication(
    spec: RunSpec,
    *,
    chunks_dir: Path,
    output_dir: Path,
    upload_manifest_path: Path | None = None,
    list_prefix: ListPrefixFn | None = None,
    upload_object: UploadObjectFn | None = None,
    download_object: DownloadObjectFn | None = None,
    credentials: S3Credentials | None = None,
    numworkers: int | None = None,
    post_upload_inventory_attempts: int = 5,
    post_upload_inventory_poll_seconds: float = 5.0,
) -> HqChunkPublicationExecutionReport:
    """Publish HQ chunks and collect the evidence required by the validation gate."""
    publication = spec.analysis_metadata.high_quality_from_tars.publication
    if not publication.enabled:
        msg = "HQ chunk publication is disabled in the RunSpec"
        raise ValueError(msg)
    target_prefix = resolve_hq_chunk_publication_target(spec)
    workers = numworkers if numworkers is not None else publication.s5cmd_numworkers
    resolved_credentials = credentials
    if list_prefix is None or upload_object is None or download_object is None:
        resolved_credentials = resolved_credentials or load_credentials_from_env()
    if list_prefix is None:
        assert resolved_credentials is not None

        def list_prefix(prefix: str) -> tuple[S3InventoryObject, ...]:
            return list_hq_publication_remote_prefix(
                prefix,
                credentials=resolved_credentials,
                numworkers=workers,
            )

    if upload_object is None:
        assert resolved_credentials is not None

        def upload_object(source: Path, destination: str) -> None:
            _upload_s3_object(
                source,
                destination,
                credentials=resolved_credentials,
                numworkers=workers,
            )

    if download_object is None:
        assert resolved_credentials is not None

        def download_object(uri: str, destination: Path) -> None:
            _download_s3_object(
                uri,
                destination,
                credentials=resolved_credentials,
                numworkers=workers,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    pre_inventory = list_prefix(target_prefix)
    pre_inventory_path = write_inventory_csv(output_dir / PRE_UPLOAD_INVENTORY_NAME, pre_inventory)
    plan = plan_hq_chunk_publication(
        spec,
        chunks_dir=chunks_dir,
        upload_manifest_path=upload_manifest_path,
        remote_inventory=pre_inventory,
    )
    write_hq_publication_reports(plan, output_dir)
    if not plan.ok:
        msg = "HQ chunk publication plan is blocked by collision or evidence failures"
        raise HqChunkPublicationExecutionError(msg)

    attempted_rows = _execute_publication_uploads(plan, upload_object=upload_object)
    post_inventory = _collect_post_upload_inventory(
        plan,
        list_prefix=list_prefix,
        attempts=post_upload_inventory_attempts,
        poll_seconds=post_upload_inventory_poll_seconds,
    )
    post_inventory_path = write_inventory_csv(output_dir / POST_UPLOAD_INVENTORY_NAME, post_inventory)
    upload_rows = _prove_upload_rows(plan, attempted_rows=attempted_rows, post_inventory=post_inventory)
    final_upload_manifest_path = write_hq_upload_manifest(output_dir / UPLOAD_MANIFEST_NAME, upload_rows)
    sampled_hashes = _collect_sample_hashes(
        plan,
        sample_count=publication.sample_download_count,
        sample_download_dir=output_dir / SAMPLED_DOWNLOAD_DIR_NAME,
        download_object=download_object,
    )
    sampled_hashes_path = write_sample_hashes_csv(output_dir / SAMPLED_HASHES_NAME, sampled_hashes)

    from bspp.orchestration.runtime.validation.hq_publication import (
        validate_hq_chunk_publication_evidence,
        write_hq_publication_validation_report,
    )

    validation_report = validate_hq_chunk_publication_evidence(
        spec,
        chunks_dir=chunks_dir,
        upload_manifest_path=final_upload_manifest_path,
        remote_inventory=post_inventory,
        sampled_hashes=sampled_hashes,
    )
    write_hq_publication_validation_report(validation_report, output_dir)
    report = HqChunkPublicationExecutionReport(
        target_prefix=target_prefix,
        plan=plan,
        validation_report=validation_report,
        pre_upload_inventory_path=pre_inventory_path,
        post_upload_inventory_path=post_inventory_path,
        upload_manifest_path=final_upload_manifest_path,
        sampled_hashes_path=sampled_hashes_path,
        sample_download_dir=output_dir / SAMPLED_DOWNLOAD_DIR_NAME,
        uploaded_count=sum(1 for row in upload_rows if row.status == "uploaded"),
        skipped_count=sum(1 for row in upload_rows if row.status == "skipped_existing"),
    )
    write_hq_publication_execution_report(report, output_dir)
    return report


def read_hq_upload_manifest(path: Path) -> tuple[HqChunkUploadRow, ...]:
    """Read a resumable HQ chunk upload manifest CSV."""
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[HqChunkUploadRow] = []
    required = {
        "chunk_index",
        "source_path",
        "destination_uri",
        "size_bytes",
        "sha256",
        "payload_sha256",
        "status",
        "attempts",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(path, reader.fieldnames, required)
        for row_number, row in enumerate(reader, start=2):
            rows.append(_upload_row_from_csv(row, path=path, row_number=row_number))
    return tuple(rows)


def write_hq_upload_manifest(path: Path, rows: tuple[HqChunkUploadRow, ...]) -> Path:
    """Write a resumable upload manifest with atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_UPLOAD_MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "chunk_index": row.chunk_index,
                    "source_path": str(row.source_path),
                    "destination_uri": row.destination_uri,
                    "size_bytes": row.size_bytes,
                    "sha256": row.sha256,
                    "payload_sha256": row.payload_sha256,
                    "status": row.status,
                    "attempts": row.attempts,
                    "remote_size_bytes": row.remote_size_bytes or "",
                    "timestamp_utc": row.timestamp_utc or "",
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    return path


def render_hq_publication_plan(plan: HqChunkPublicationPlan) -> str:
    """Render a deterministic JSON publication plan."""
    return report_to_json(plan)


def write_hq_publication_reports(plan: HqChunkPublicationPlan, output_dir: Path) -> tuple[Path, Path, Path]:
    """Write plan JSON/text and planned upload manifest under *output_dir*."""
    json_path = write_json_report(plan, output_dir / "hq_chunks_publication_plan.json")
    text_path = write_text_summary(plan, output_dir / "hq_chunks_publication_plan.txt")
    manifest_path = write_hq_upload_manifest(output_dir / UPLOAD_MANIFEST_NAME, plan.upload_manifest_rows())
    return json_path, text_path, manifest_path


def render_hq_publication_execution_report(report: HqChunkPublicationExecutionReport) -> str:
    """Render a deterministic JSON publication execution report."""
    return report_to_json(report)


def write_hq_publication_execution_report(
    report: HqChunkPublicationExecutionReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write JSON/text reports for live publication execution."""
    json_path = write_json_report(report, output_dir / "hq_chunks_publication_execution.json")
    text_path = write_text_summary(report, output_dir / "hq_chunks_publication_execution.txt")
    return json_path, text_path


def select_hq_publication_sample_uris(destinations: tuple[str, ...], sample_count: int) -> tuple[str, ...]:
    """Select deterministic sample URIs for remote download hash evidence."""
    sorted_destinations = tuple(sorted(destinations))
    if sample_count <= 0 or not sorted_destinations:
        return ()
    if sample_count >= len(sorted_destinations):
        return sorted_destinations
    if sample_count == 1:
        return (sorted_destinations[0],)
    indexes = {round(index * (len(sorted_destinations) - 1) / (sample_count - 1)) for index in range(sample_count)}
    return tuple(sorted_destinations[index] for index in sorted(indexes))


def list_hq_publication_remote_prefix(
    prefix: str,
    *,
    credentials: S3Credentials,
    numworkers: int | None,
) -> tuple[S3InventoryObject, ...]:
    """List remote HQ publication objects, treating a fresh empty prefix as empty inventory."""
    s5cmd = require_tool("s5cmd", hint="Install s5cmd or use the container image.")
    argv = build_list_prefix_argv(prefix, credentials=credentials, s5cmd_path=s5cmd, numworkers=numworkers)
    result = _run_s5cmd(argv, credentials=credentials, allow_no_object_found=True)
    return _parse_hq_publication_ls_output(result.stdout_tail, prefix=prefix)


def _parse_hq_publication_ls_output(output: str, *, prefix: str) -> tuple[S3InventoryObject, ...]:
    try:
        return parse_s5cmd_ls_output(output)
    except ValueError:
        objects: list[S3InventoryObject] = []
        for line_number, line in enumerate(output.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if " DIR " in f" {stripped} ":
                continue
            parts = stripped.split()
            if len(parts) < 4:
                msg = f"unparseable s5cmd ls line {line_number}: {line!r}"
                raise ValueError(msg) from None
            try:
                size_bytes = int(parts[2])
            except ValueError:
                msg = f"unparseable s5cmd ls line {line_number}: {line!r}"
                raise ValueError(msg) from None
            name = parts[-1]
            uri = name if name.startswith("s3://") else f"{prefix.rstrip('/')}/{name.lstrip('/')}"
            objects.append(S3InventoryObject(uri=uri, size_bytes=size_bytes))
        return tuple(objects)


def _execute_publication_uploads(
    plan: HqChunkPublicationPlan,
    *,
    upload_object: UploadObjectFn,
) -> tuple[HqChunkUploadRow, ...]:
    rows: list[HqChunkUploadRow] = []
    for item in plan.items:
        if item.action in {"upload", "overwrite"}:
            upload_object(item.source_path, item.destination_uri)
            rows.append(
                HqChunkUploadRow(
                    chunk_index=item.chunk_index,
                    source_path=item.source_path,
                    destination_uri=item.destination_uri,
                    size_bytes=item.size_bytes,
                    sha256=item.sha256,
                    payload_sha256=item.payload_sha256,
                    status="upload_attempted",
                    attempts=1,
                    timestamp_utc=_timestamp(),
                )
            )
        elif item.action == "skip_existing":
            rows.append(
                HqChunkUploadRow(
                    chunk_index=item.chunk_index,
                    source_path=item.source_path,
                    destination_uri=item.destination_uri,
                    size_bytes=item.size_bytes,
                    sha256=item.sha256,
                    payload_sha256=item.payload_sha256,
                    status="skip_existing_attempted",
                    attempts=0,
                    remote_size_bytes=item.size_bytes,
                    timestamp_utc=_timestamp(),
                )
            )
        else:
            rows.append(item.to_upload_row())
    return tuple(rows)


def _collect_post_upload_inventory(
    plan: HqChunkPublicationPlan,
    *,
    list_prefix: ListPrefixFn,
    attempts: int,
    poll_seconds: float,
) -> tuple[S3InventoryObject, ...]:
    required = {item.destination_uri for item in plan.items}
    remaining_attempts = max(1, attempts)
    inventory: tuple[S3InventoryObject, ...] = ()
    for attempt_index in range(remaining_attempts):
        inventory = list_prefix(plan.target_prefix)
        present = {obj.uri for obj in inventory}
        if required <= present:
            return inventory
        if attempt_index + 1 < remaining_attempts and poll_seconds > 0:
            time.sleep(poll_seconds)
    return inventory


def _prove_upload_rows(
    plan: HqChunkPublicationPlan,
    *,
    attempted_rows: tuple[HqChunkUploadRow, ...],
    post_inventory: tuple[S3InventoryObject, ...],
) -> tuple[HqChunkUploadRow, ...]:
    remote_by_uri = {obj.uri: obj for obj in post_inventory if obj.uri.startswith(plan.target_prefix)}
    proven: list[HqChunkUploadRow] = []
    item_by_destination = {item.destination_uri: item for item in plan.items}
    for row in attempted_rows:
        item = item_by_destination[row.destination_uri]
        remote = remote_by_uri.get(row.destination_uri)
        remote_size = remote.size_bytes if remote is not None else None
        matches_remote = remote_size == row.size_bytes
        status = row.status
        if item.action in {"upload", "overwrite"}:
            status = "uploaded" if matches_remote else "upload_unverified"
        elif item.action == "skip_existing":
            status = "skipped_existing" if matches_remote else "skip_existing_unverified"
        proven.append(
            HqChunkUploadRow(
                chunk_index=row.chunk_index,
                source_path=row.source_path,
                destination_uri=row.destination_uri,
                size_bytes=row.size_bytes,
                sha256=row.sha256,
                payload_sha256=row.payload_sha256,
                status=status,
                attempts=row.attempts,
                remote_size_bytes=remote_size,
                timestamp_utc=row.timestamp_utc or _timestamp(),
            )
        )
    return tuple(proven)


def _collect_sample_hashes(
    plan: HqChunkPublicationPlan,
    *,
    sample_count: int,
    sample_download_dir: Path,
    download_object: DownloadObjectFn,
) -> tuple[SampleHashEvidence, ...]:
    destinations = tuple(item.destination_uri for item in plan.items)
    sample_uris = select_hq_publication_sample_uris(destinations, sample_count)
    hashes: list[SampleHashEvidence] = []
    for uri in sample_uris:
        destination = sample_download_dir / Path(uri).name
        download_object(uri, destination)
        hashes.append(
            SampleHashEvidence(
                uri=uri,
                sha256=_file_sha256(destination),
                size_bytes=destination.stat().st_size,
            )
        )
    return tuple(hashes)


def _upload_s3_object(
    source: Path,
    destination: str,
    *,
    credentials: S3Credentials,
    numworkers: int | None,
) -> None:
    result = s3_transfer.cp(source, destination, credentials=credentials, numworkers=numworkers)
    _require_transfer_ok(result, operation=f"upload {source} to {destination}")


def _download_s3_object(
    uri: str,
    destination: Path,
    *,
    credentials: S3Credentials,
    numworkers: int | None,
) -> None:
    result = s3_transfer.cp(uri, destination, credentials=credentials, numworkers=numworkers)
    _require_transfer_ok(result, operation=f"download {uri} to {destination}")


def _run_s5cmd(
    argv: tuple[str, ...],
    *,
    credentials: S3Credentials,
    allow_no_object_found: bool = False,
) -> TransferResult:
    started = time.monotonic()
    env = dict(os.environ)
    env.update(credentials.as_env())
    completed = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    elapsed = time.monotonic() - started
    returncode = completed.returncode
    if allow_no_object_found and completed.returncode != 0 and _is_s5cmd_no_object_found(completed.stderr):
        returncode = 0
    result = TransferResult(
        tool="s5cmd",
        argv=argv,
        returncode=returncode,
        elapsed_s=elapsed,
        stdout_tail=completed.stdout,
        stderr_tail=completed.stderr[-4096:],
    )
    _require_transfer_ok(result, operation="list remote HQ chunk publication prefix")
    return result


def _is_s5cmd_no_object_found(stderr: str) -> bool:
    return "no object found" in stderr.lower()


def _require_transfer_ok(result: object, *, operation: str) -> None:
    if isinstance(result, TransferResult) and result.ok:
        return
    if isinstance(result, TransferResult):
        raise_transport_failure_if_audited(result)
        msg = f"HQ chunk publication {operation} failed with rc={result.returncode}: {result.stderr_tail.strip()}"
        raise HqChunkPublicationExecutionError(msg)
    msg = f"HQ chunk publication {operation} did not return a transfer result"
    raise HqChunkPublicationExecutionError(msg)


def _verify_chunk(chunk: HqChunkBuildChunk) -> HqChunkBuildChunk:
    if not chunk.tar_path.is_file():
        raise FileNotFoundError(chunk.tar_path)
    actual_size = chunk.tar_path.stat().st_size
    if actual_size != chunk.size_bytes:
        msg = f"{chunk.tar_path} size mismatch: manifest={chunk.size_bytes} actual={actual_size}"
        raise ValueError(msg)
    actual_sha = _file_sha256(chunk.tar_path)
    if actual_sha != chunk.sha256:
        msg = f"{chunk.tar_path} sha256 mismatch: manifest={chunk.sha256} actual={actual_sha}"
        raise ValueError(msg)
    return chunk


def _matches_prior_upload(
    previous: HqChunkUploadRow | None,
    chunk: HqChunkBuildChunk,
    remote: S3InventoryObject,
) -> bool:
    return (
        previous is not None
        and previous.destination_uri == remote.uri
        and previous.size_bytes == chunk.size_bytes
        and previous.sha256 == chunk.sha256
        and previous.payload_sha256 == chunk.payload_sha256
        and previous.remote_size_bytes == remote.size_bytes
        and remote.size_bytes == chunk.size_bytes
        and previous.status in {"uploaded", "skipped_existing"}
    )


def _upload_row_from_csv(row: dict[str, str], *, path: Path, row_number: int) -> HqChunkUploadRow:
    remote_size = (row.get("remote_size_bytes") or "").strip()
    return HqChunkUploadRow(
        chunk_index=_read_int(row, "chunk_index", path=path, row_number=row_number),
        source_path=Path(_required(row, "source_path", path=path, row_number=row_number)),
        destination_uri=_required(row, "destination_uri", path=path, row_number=row_number),
        size_bytes=_read_int(row, "size_bytes", path=path, row_number=row_number),
        sha256=_required(row, "sha256", path=path, row_number=row_number),
        payload_sha256=_required(row, "payload_sha256", path=path, row_number=row_number),
        status=_required(row, "status", path=path, row_number=row_number),
        attempts=_read_int(row, "attempts", path=path, row_number=row_number),
        remote_size_bytes=int(remote_size) if remote_size else None,
        timestamp_utc=(row.get("timestamp_utc") or "").strip() or None,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _require_columns(path: Path, fieldnames: Sequence[str] | None, required: set[str]) -> None:
    if fieldnames is None:
        msg = f"CSV has no header: {path}"
        raise ValueError(msg)
    missing = required - set(fieldnames)
    if missing:
        msg = f"{path} is missing required columns: {', '.join(sorted(missing))}"
        raise ValueError(msg)


def _required(row: dict[str, str], column: str, *, path: Path, row_number: int) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        msg = f"{path} row {row_number} has empty {column}"
        raise ValueError(msg)
    return value


def _read_int(row: dict[str, str], column: str, *, path: Path, row_number: int) -> int:
    value = _required(row, column, path=path, row_number=row_number)
    try:
        return int(value)
    except ValueError as exc:
        msg = f"{path} row {row_number} has invalid integer {column}: {value!r}"
        raise ValueError(msg) from exc


_UPLOAD_MANIFEST_FIELDS = (
    "chunk_index",
    "source_path",
    "destination_uri",
    "size_bytes",
    "sha256",
    "payload_sha256",
    "status",
    "attempts",
    "remote_size_bytes",
    "timestamp_utc",
)


__all__ = [
    "POST_UPLOAD_INVENTORY_NAME",
    "PRE_UPLOAD_INVENTORY_NAME",
    "SAMPLED_DOWNLOAD_DIR_NAME",
    "SAMPLED_HASHES_NAME",
    "UPLOAD_MANIFEST_NAME",
    "HqChunkPublicationExecutionError",
    "HqChunkPublicationExecutionReport",
    "HqChunkPublicationItem",
    "HqChunkPublicationPlan",
    "HqChunkUploadRow",
    "execute_hq_chunk_publication",
    "list_hq_publication_remote_prefix",
    "plan_hq_chunk_publication",
    "read_hq_upload_manifest",
    "render_hq_publication_execution_report",
    "render_hq_publication_plan",
    "select_hq_publication_sample_uris",
    "write_hq_publication_execution_report",
    "write_hq_publication_reports",
    "write_hq_upload_manifest",
]
