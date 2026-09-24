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

"""Upload the pipeline-verification-data archive to SwiftStack using the existing
orchestration data-movement machinery (``s3_transfer.cp`` + s5cmd), and verify the
uploaded object by exact key + exact size.

This is the canonical way to publish the toolkit verification/example archive (the
oversized example fixtures that are deliberately excluded from git per the 5 MiB
GitLab limit) so any cluster can fetch it without a shared filesystem. It runs
inside the postprocessing image (which carries s5cmd + the
orchestration-runtime package), so it also serves as a live gate for the image's
SwiftStack machinery.

Inputs (environment):
  ARCHIVE_PATH   local path to the archive file (e.g. …/….tar.zst)
  MIRROR_URI     destination s3:// URI for the archive object
  EVIDENCE_JSON  optional path to write a small JSON receipt

Never run on a login node or bare-metal; run inside the postprocessing image.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from bspp.orchestration.runtime.data_movement.common import TransferResult, require_tool, run_transfer
from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer
from bspp.orchestration.runtime.data_movement.s3.client import load_credentials_from_env


def exact_size_from_s5cmd_ls(stdout: str, uri: str) -> int | None:
    """Parse the exact object size for `uri` from `s5cmd ls` output (same approach as
    the baseline preservation verify step)."""
    expected_key = uri[len("s3://") :].split("/", 1)[1]
    fallback: int | None = None
    for line in stdout.splitlines():
        parts = re.split(r"\s+", line.strip(), maxsplit=3)
        if len(parts) < 4:
            continue
        try:
            size = int(parts[2])
        except ValueError:
            continue
        if parts[3] == expected_key:
            return size
        fallback = size
    return fallback


def main() -> int:
    archive = Path(os.environ["ARCHIVE_PATH"])
    uri = os.environ["MIRROR_URI"].strip()
    evidence_json = os.environ.get("EVIDENCE_JSON")
    if not archive.is_file():
        print(f"ERROR: archive not found: {archive}", file=sys.stderr)
        return 1

    creds = load_credentials_from_env()
    s5cmd = require_tool("s5cmd")
    expected_size = archive.stat().st_size
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Upload via the existing data-movement machinery.
    up = s3_transfer.cp(archive, uri, numworkers=16)
    if not isinstance(up, TransferResult):  # dry_run path returns PlannedTransfer; we never dry-run here
        print(f"ERROR: unexpected planned (non-executed) transfer for {uri}", file=sys.stderr)
        return 1
    if not up.ok:
        print(f"ERROR: upload failed rc={up.returncode}: {up.stderr_tail}", file=sys.stderr)
        return 1

    # Verify by exact key + exact size (eventual-consistency backoff).
    verified = False
    actual_size: int | None = None
    for delay in [0, 2, 5, 10, 20, 30, 60]:
        if delay:
            time.sleep(delay)
        ls = run_transfer(
            [s5cmd, "--endpoint-url", creds.endpoint_url, "ls", uri],
            tool="s5cmd",
            env={**os.environ, **creds.as_env()},
        )
        actual_size = exact_size_from_s5cmd_ls(ls.stdout_tail, uri)
        verified = ls.ok and actual_size == expected_size
        if verified:
            break

    receipt = {
        "created_utc": started,
        "mirror_uri": uri,
        "expected_size_bytes": expected_size,
        "actual_size_bytes": actual_size,
        "verified": verified,
        "upload_returncode": up.returncode,
        "note": "pipeline-verification-data archive, uploaded + verified via the image data_movement machinery",
    }
    if evidence_json:
        Path(evidence_json).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"mirror_uri={uri}")
    print(f"expected_size_bytes={expected_size}")
    print(f"actual_size_bytes={actual_size}")
    print(f"verified={verified}")
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
