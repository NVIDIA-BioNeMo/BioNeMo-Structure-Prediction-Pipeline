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

"""Unit tests for the gcloud-storage subprocess wrapper."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from bspp.orchestration.runtime.data_movement.common import PlannedTransfer, ToolMissingError, TransferResult
from bspp.orchestration.runtime.data_movement.gcs import transfer as gcs_transfer


@pytest.fixture(autouse=True)
def _fake_gcloud_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend gcloud is installed without actually invoking it."""
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/gcloud" if name == "gcloud" else None)


def test_cp_dry_run_builds_recursive_argv() -> None:
    planned = gcs_transfer.cp("src/dir", "gs://bucket/prefix/", recursive=True, dry_run=True)

    assert isinstance(planned, PlannedTransfer)
    assert planned.tool == "gcloud"
    assert planned.argv == (
        "/usr/local/bin/gcloud",
        "storage",
        "cp",
        "--recursive",
        "src/dir",
        "gs://bucket/prefix/",
    )


def test_cp_invokes_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []

    def fake_run(argv, capture_output, text, env, check):  # type: ignore[no-untyped-def]
        captured.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    local_src = tmp_path / "input.txt"
    local_src.write_text("hello")
    result = gcs_transfer.cp(local_src, "gs://bucket/out.txt")

    assert isinstance(result, TransferResult)
    assert result.ok
    assert result.tool == "gcloud"
    assert captured == [
        [
            "/usr/local/bin/gcloud",
            "storage",
            "cp",
            str(local_src),
            "gs://bucket/out.txt",
        ]
    ]


def test_rsync_adds_delete_flag() -> None:
    planned = gcs_transfer.rsync(
        "gs://src/",
        "gs://dst/",
        recursive=True,
        delete_unmatched=True,
        dry_run=True,
    )

    assert isinstance(planned, PlannedTransfer)
    assert "--recursive" in planned.argv
    assert "--delete-unmatched-destination-objects" in planned.argv


def test_cp_dry_run_without_gcloud_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run must render the plan even when gcloud is not installed."""
    monkeypatch.setattr(shutil, "which", lambda name: None)

    planned = gcs_transfer.cp("src", "gs://dst/", dry_run=True)

    assert isinstance(planned, PlannedTransfer)
    assert planned.tool == "gcloud"
    assert planned.argv[0] == "gcloud"  # bare tool name, not a resolved path
    assert planned.argv[1:4] == ("storage", "cp", "src")  # src appended after flags


def test_rsync_dry_run_without_gcloud_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run rsync must also render without gcloud on PATH."""
    monkeypatch.setattr(shutil, "which", lambda name: None)

    planned = gcs_transfer.rsync("gs://src/", "gs://dst/", dry_run=True)

    assert isinstance(planned, PlannedTransfer)
    assert planned.argv[0] == "gcloud"


def test_cp_non_dry_run_raises_without_gcloud(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-dry-run must still raise ToolMissingError when gcloud is absent."""
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(ToolMissingError):
        gcs_transfer.cp("src", "gs://dst/")
