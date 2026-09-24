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

"""Verify GCS uploads by listing objects.

Uses the ``google-cloud-storage`` Python client (not ``gcloud`` CLI)
because listing is a metadata query, not a transfer — the Python client
is fine for bulk ``list_blobs`` calls and returns structured objects
we can filter/count without parsing CLI output.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from google.cloud import storage


@dataclass(frozen=True)
class BlobRecord:
    """Minimal projection of a GCS blob for verification purposes."""

    name: str
    size: int
    updated: datetime | None


def list_blobs(
    credentials_path: Path,
    bucket: str,
    *,
    prefix: str = "",
    max_results: int | None = None,
) -> Iterator[BlobRecord]:
    """Iterate over blobs under *prefix* in *bucket*.

    Pulls blobs via the client's pagination; callers that only need a
    count can consume the iterator without materialising a full list.
    """
    client = storage.Client.from_service_account_json(str(credentials_path))
    bucket_ref = client.bucket(bucket)
    iterator = bucket_ref.list_blobs(
        prefix=prefix or None,
        max_results=max_results,
    )
    for blob in iterator:
        yield BlobRecord(
            name=blob.name,
            size=int(blob.size or 0),
            updated=blob.updated,
        )


def count_with_prefix(
    credentials_path: Path,
    bucket: str,
    *,
    prefix: str = "",
    target_date: date | None = None,
) -> tuple[int, int]:
    """Return ``(total_matched, total_bytes)`` for blobs under *prefix*.

    If *target_date* is provided, only blobs whose ``updated`` timestamp
    falls on that UTC date are counted.  This is the primary check done
    after a pipeline run to confirm the expected objects landed today.
    """
    total_count = 0
    total_bytes = 0
    for blob in list_blobs(credentials_path, bucket, prefix=prefix):
        if target_date is not None:
            if blob.updated is None:
                continue
            updated_utc = blob.updated.astimezone(UTC).date()
            if updated_utc != target_date:
                continue
        total_count += 1
        total_bytes += blob.size
    return total_count, total_bytes


__all__ = ["BlobRecord", "count_with_prefix", "list_blobs"]
