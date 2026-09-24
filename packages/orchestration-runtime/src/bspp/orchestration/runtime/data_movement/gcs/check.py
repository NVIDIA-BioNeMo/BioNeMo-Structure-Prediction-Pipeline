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

"""GCS connectivity check."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from google.cloud import storage


@dataclass
class GCSStatus:
    """Result of a GCS connectivity check."""

    service_account: str
    project: str
    bucket: str
    bucket_accessible: bool


def check_connection(
    credentials_path: Path,
    bucket_name: str,
    prefix: str = "",
) -> GCSStatus:
    """Check GCS connectivity using a service account JSON key.

    Bucket reachability is tested with a bounded list_blobs call
    (works even without storage.buckets.get).

    Args:
        credentials_path: Path to the service account JSON key file.
        bucket_name: GCS bucket to check access to.
        prefix: Prefix to scope the list_blobs probe to.

    Returns:
        GCSStatus with connection details.
    """
    cred_info = json.loads(credentials_path.read_text())
    client = storage.Client.from_service_account_json(str(credentials_path))

    bucket = client.bucket(bucket_name)

    # Probe bucket access with a bounded list — works even when the
    # service account lacks storage.buckets.get.
    bucket_accessible = False
    try:
        next(iter(bucket.list_blobs(prefix=prefix or None, max_results=1)))
        bucket_accessible = True
    except StopIteration:
        bucket_accessible = True
    except Exception:
        bucket_accessible = False

    return GCSStatus(
        service_account=cred_info.get("client_email", "unknown"),
        project=cred_info.get("project_id", "unknown"),
        bucket=bucket_name,
        bucket_accessible=bucket_accessible,
    )
