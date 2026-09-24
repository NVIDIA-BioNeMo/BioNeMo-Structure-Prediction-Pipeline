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

"""GCS transfers via the ``gcloud storage`` CLI.

Thin subprocess wrapper around ``gcloud storage cp`` / ``gcloud storage
rsync``.  Used for all GCS file-level uploads/downloads across the
orchestration pipeline.  For listing / verification queries (no data
transfer) use :mod:`bspp.orchestration.runtime.data_movement.gcs.verify` which
uses the ``google-cloud-storage`` Python client directly.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path

from bspp.orchestration.runtime.data_movement.common import (
    PlannedTransfer,
    TransferResult,
    ensure_uri_parent_local,
    require_tool,
    run_transfer,
)

_TOOL_HINT = "Install Google Cloud SDK or use the container image (gcloud is baked in)."


def cp(
    src: str | Path,
    dst: str | Path,
    *,
    recursive: bool = False,
    extra_args: Iterable[str] = (),
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> TransferResult | PlannedTransfer:
    """Copy *src* to *dst* via ``gcloud storage cp``.

    Either endpoint may be a ``gs://`` URI or a local path. For directory
    transfers set ``recursive=True`` (equivalent to ``gcloud storage cp -r``).
    """
    if dry_run:
        gcloud_path = shutil.which("gcloud") or "gcloud"
        argv: list[str] = [gcloud_path, "storage", "cp"]
        if recursive:
            argv.append("--recursive")
        argv.extend(extra_args)
        argv.extend([str(src), str(dst)])
        return PlannedTransfer(tool="gcloud", argv=tuple(argv), note="cp")

    gcloud = require_tool("gcloud", hint=_TOOL_HINT)
    argv = [gcloud, "storage", "cp"]
    if recursive:
        argv.append("--recursive")
    argv.extend(extra_args)
    argv.extend([str(src), str(dst)])

    if not str(dst).startswith("gs://"):
        ensure_uri_parent_local(Path(str(dst)))

    return run_transfer(argv, tool="gcloud", env=env)


def rsync(
    src: str | Path,
    dst: str | Path,
    *,
    recursive: bool = True,
    delete_unmatched: bool = False,
    extra_args: Iterable[str] = (),
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> TransferResult | PlannedTransfer:
    """Rsync *src* to *dst* via ``gcloud storage rsync``.

    ``recursive=True`` (the default) matches the most common use case.
    Set ``delete_unmatched=True`` to also delete objects at *dst* that
    are absent at *src* (the ``--delete-unmatched-destination-objects``
    flag).
    """
    if dry_run:
        gcloud_path = shutil.which("gcloud") or "gcloud"
        argv: list[str] = [gcloud_path, "storage", "rsync"]
        if recursive:
            argv.append("--recursive")
        if delete_unmatched:
            argv.append("--delete-unmatched-destination-objects")
        argv.extend(extra_args)
        argv.extend([str(src), str(dst)])
        return PlannedTransfer(tool="gcloud", argv=tuple(argv), note="rsync")

    gcloud = require_tool("gcloud", hint=_TOOL_HINT)
    argv = [gcloud, "storage", "rsync"]
    if recursive:
        argv.append("--recursive")
    if delete_unmatched:
        argv.append("--delete-unmatched-destination-objects")
    argv.extend(extra_args)
    argv.extend([str(src), str(dst)])

    return run_transfer(argv, tool="gcloud", env=env)


__all__ = ["cp", "rsync"]
