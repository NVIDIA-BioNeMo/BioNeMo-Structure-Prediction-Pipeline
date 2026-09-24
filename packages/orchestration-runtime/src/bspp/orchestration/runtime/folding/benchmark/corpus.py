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

"""Cluster-side benchmark corpus fetch and fail-closed integrity verification.

The workstation-side ``bsppctl validate-run`` command (e05s09) submits the
``bspp-orchestration-runtime benchmark validate-run`` worker into a Slurm job.
This module owns the worker-side corpus boundary: it fetches the pinned
benchmark corpus from S3 through the existing s5cmd transfer adapter
and then re-verifies both the per-file SHA256SUMS and the dataset
fingerprint before any validation code reads the bytes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

from bspp.orchestration.runtime.data_movement.common import PlannedTransfer
from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer
from bspp.orchestration.runtime.data_movement.s3.client import S3Credentials

_FINGERPRINT_PROFILES = {
    "source_gzip_sha256": (
        b"afdb-pdb-benchmark-v1\0",
        (
            "target_id",
            "sequence_sha256",
            "stratum",
            "reference_sha256",
            "source_gzip_sha256",
        ),
    ),
    "source_mmcif_sha256": (
        b"afdb-pdb-benchmark-mmcif-v1\0",
        (
            "target_id",
            "sequence_sha256",
            "stratum",
            "reference_sha256",
            "source_mmcif_sha256",
        ),
    ),
}


def fetch_pinned_corpus(
    s3_location: str,
    destination: Path,
    *,
    credentials: S3Credentials | None = None,
    expected_fingerprint: str,
) -> Path:
    """Fetch the pinned benchmark corpus and verify its integrity, fail-closed.

    ``s3_location`` is an S3 directory prefix (with or without a
    trailing slash) whose contents are copied recursively into ``destination``
    by the existing s5cmd ``cp`` adapter. The prefix is normalized to the
    recursive ``<prefix>/*`` source form before the transfer, so callers never
    need to supply the wildcard themselves. Any transfer failure, SHA256SUMS
    mismatch, or dataset fingerprint mismatch raises :class:`ValueError`; a
    clean pass returns ``destination``.
    """
    source = _recursive_s3_source(s3_location)
    result = s3_transfer.cp(source, destination, credentials=credentials)
    if isinstance(result, PlannedTransfer):
        raise ValueError("benchmark corpus transfer unexpectedly returned a dry-run plan")
    if not result.ok:
        stderr = result.stderr_tail.strip()
        detail = f": {stderr}" if stderr else ""
        raise ValueError(f"benchmark corpus transfer failed (rc={result.returncode}){detail}")
    _verify_sha256sums(destination)
    _verify_dataset_fingerprint(destination, expected_fingerprint)
    return destination


def _recursive_s3_source(s3_location: str) -> str:
    """Normalize a corpus prefix to the recursive ``<prefix>/*`` s5cmd source."""
    if s3_location.endswith("*"):
        return s3_location
    return s3_location.rstrip("/") + "/*"


def _verify_sha256sums(destination: Path) -> None:
    """Recompute every SHA256SUMS-listed file hash, fail-closed on any mismatch."""
    sums_path = destination / "SHA256SUMS"
    try:
        text = sums_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read SHA256SUMS: {exc}") from exc
    root = destination.resolve()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Malformed SHA256SUMS line {line_number}")
        digest, relpath = parts
        path = destination / relpath
        if not path.is_file():
            raise ValueError(f"SHA256SUMS entry missing file: {relpath}")
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"SHA256SUMS entry escapes the corpus root: {relpath}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"SHA256SUMS digest mismatch for {relpath}")


def _verify_dataset_fingerprint(destination: Path, expected_fingerprint: str) -> None:
    """Cross-check the stored, recomputed, and pinned dataset fingerprints."""
    dataset_path = destination / "dataset.json"
    try:
        raw = dataset_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read dataset.json: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in dataset.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("dataset.json must be a JSON object")
    dataset = cast("Mapping[str, object]", payload)
    stored_fingerprint = dataset.get("dataset_fingerprint")
    if not isinstance(stored_fingerprint, str) or not stored_fingerprint:
        raise ValueError("dataset.json is missing dataset_fingerprint")
    records = _read_target_records(destination)
    recomputed = _recompute_dataset_fingerprint(dataset, records)
    if stored_fingerprint != recomputed or recomputed != expected_fingerprint:
        raise ValueError("dataset fingerprint mismatch: stored, recomputed, and pinned values must agree")


def _read_target_records(destination: Path) -> list[Mapping[str, object]]:
    """Read targets.jsonl records, sorted by target_id for order stability."""
    targets_path = destination / "targets.jsonl"
    try:
        text = targets_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read targets.jsonl: {exc}") from exc
    records: list[Mapping[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed targets.jsonl line {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"targets.jsonl line {line_number} must be a JSON object")
        records.append(cast("Mapping[str, object]", payload))
    records.sort(key=_record_target_id)
    return records


def _record_target_id(record: Mapping[str, object]) -> str:
    value = record.get("target_id")
    if not isinstance(value, str) or not value:
        raise ValueError("targets.jsonl record is missing target_id")
    return value


def _recompute_dataset_fingerprint(
    dataset: Mapping[str, object],
    records: Iterable[Mapping[str, object]],
) -> str:
    """Recompute the dataset fingerprint byte-for-byte as the curator does.

    The byte format must match the curator's ``_dataset_fingerprint`` exactly:
    the fingerprint prefix, then the canonical specification mapping, then one
    JSON object per record (five pinned keys, sorted keys, compact separators).

    The fingerprint profile (prefix + record keys) is selected dynamically from
    the first record: records with ``source_mmcif_sha256`` use the mmCIF profile;
    records with ``source_gzip_sha256`` use the original gzip profile.
    """
    specification = dataset.get("specification")
    if not isinstance(specification, dict):
        raise ValueError("dataset.json is missing a specification object")
    records = list(records)
    first = records[0] if records else {}
    if "source_mmcif_sha256" in first:
        prefix, keys = _FINGERPRINT_PROFILES["source_mmcif_sha256"]
    else:
        prefix, keys = _FINGERPRINT_PROFILES["source_gzip_sha256"]
    digest = hashlib.sha256()
    digest.update(prefix)
    digest.update(json.dumps(specification, sort_keys=True, separators=(",", ":")).encode())
    for record in records:
        try:
            value = {key: record[key] for key in keys}
        except KeyError as exc:
            raise ValueError(f"targets.jsonl record missing required key {exc.args[0]!r}") from exc
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


__all__ = ["fetch_pinned_corpus"]
