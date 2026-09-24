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

from __future__ import annotations

from pathlib import Path

import pytest

from bspp.orchestration.runtime.worker import (
    TaskContext,
    cleanup_scratch_workspace,
    create_scratch_workspace,
    extract_archive_to_workspace,
    plan_scratch_workspace,
    relink_renamed_inputs,
)


def test_plan_and_create_scratch_workspace_uses_legacy_task_layout(tmp_path: Path) -> None:
    workspace = plan_scratch_workspace(
        tmp_path / "scratch",
        TaskContext(job_id="wp8d-job", array_task_id=278),
    )

    assert workspace.root == tmp_path / "scratch" / "bspp_wp8d-job_278"
    assert workspace.input_dir == workspace.root / "input"
    assert workspace.work_dir == workspace.root / "work"

    create_scratch_workspace(workspace)

    assert workspace.input_dir.is_dir()
    assert workspace.work_dir.is_dir()


def test_extract_archive_uses_injected_extractor(tmp_path: Path) -> None:
    archive = tmp_path / "archive.tar.lz4"
    archive.write_text("compressed")
    workspace = create_scratch_workspace(
        plan_scratch_workspace(tmp_path / "scratch", TaskContext(job_id="job", array_task_id=0)),
    )
    calls: list[tuple[Path, Path]] = []

    def fake_extractor(archive_path: Path, destination_dir: Path) -> None:
        calls.append((archive_path, destination_dir))
        (destination_dir / "AF-0000000000000001").mkdir()

    extract_archive_to_workspace(archive, workspace, extractor=fake_extractor)

    assert calls == [(archive, workspace.input_dir)]
    assert (workspace.input_dir / "AF-0000000000000001").is_dir()


def test_extract_archive_missing_file_raises_before_extractor(tmp_path: Path) -> None:
    workspace = create_scratch_workspace(
        plan_scratch_workspace(tmp_path / "scratch", TaskContext(job_id="job", array_task_id=0)),
    )

    with pytest.raises(FileNotFoundError):
        extract_archive_to_workspace(tmp_path / "missing.tar.lz4", workspace, extractor=lambda *_: None)


def test_cleanup_policy_controls_workspace_removal(tmp_path: Path) -> None:
    workspace = create_scratch_workspace(
        plan_scratch_workspace(
            tmp_path / "scratch",
            TaskContext(job_id="job", array_task_id=0),
            cleanup_policy="on_success",
        ),
    )

    assert cleanup_scratch_workspace(workspace, success=False) is False
    assert workspace.root.exists()

    assert cleanup_scratch_workspace(workspace, success=True) is True
    assert not workspace.root.exists()


def test_relink_renamed_inputs_creates_unified_id_symlinks(tmp_path: Path) -> None:
    compound_id = "AF_1001_AF_1002"
    unified_id = "AF-9999999999999999"
    source = tmp_path / f"{compound_id}.merged_scores_rank_001.json"
    source.write_text("{}")
    existing = tmp_path / f"{unified_id}.already_there"
    existing.write_text("keep")

    created = relink_renamed_inputs(tmp_path, {compound_id: unified_id})

    expected_link = tmp_path / f"{unified_id}.merged_scores_rank_001.json"
    assert created == (expected_link,)
    assert expected_link.is_symlink()
    assert expected_link.resolve() == source
    assert existing.read_text() == "keep"
