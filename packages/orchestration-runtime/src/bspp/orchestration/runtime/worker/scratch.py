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

"""Scratch workspace lifecycle, archive extraction, and input relinking."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from bspp.orchestration.runtime.worker.types import CleanupPolicy, ScratchWorkspace, TaskContext


class ArchiveExtractor(Protocol):
    """Fakeable archive extraction boundary."""

    def __call__(self, archive_path: Path, destination_dir: Path) -> None:
        """Extract *archive_path* into *destination_dir*."""


def plan_scratch_workspace(
    scratch_root: Path,
    task: TaskContext,
    *,
    cleanup_policy: CleanupPolicy = "always",
) -> ScratchWorkspace:
    """Return the legacy-compatible scratch workspace path for one task."""

    root = scratch_root / f"bspp_{task.job_id}_{task.array_task_id}"
    return ScratchWorkspace(
        root=root,
        input_dir=root / "input",
        work_dir=root / "work",
        cleanup_policy=cleanup_policy,
    )


def create_scratch_workspace(workspace: ScratchWorkspace) -> ScratchWorkspace:
    """Create the input and work directories for a planned workspace."""

    workspace.input_dir.mkdir(parents=True, exist_ok=True)
    workspace.work_dir.mkdir(parents=True, exist_ok=True)
    return workspace


def extract_archive_to_workspace(
    archive_path: Path,
    workspace: ScratchWorkspace,
    *,
    extractor: ArchiveExtractor | None = None,
) -> None:
    """Extract one ``.tar.lz4`` archive into the workspace input directory."""

    if not archive_path.exists():
        msg = f"Archive not found: {archive_path}"
        raise FileNotFoundError(msg)
    workspace.input_dir.mkdir(parents=True, exist_ok=True)
    (extractor or extract_tar_lz4)(archive_path, workspace.input_dir)


def extract_tar_lz4(archive_path: Path, destination_dir: Path) -> None:
    """Extract a legacy ``tar.lz4`` archive using ``lz4 -dc | tar -xf -``."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    lz4_proc = subprocess.Popen(
        ["lz4", "-dc", str(archive_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if lz4_proc.stdout is None:
        lz4_proc.wait()
        msg = "lz4 did not expose stdout for tar extraction"
        raise RuntimeError(msg)
    try:
        tar_proc = subprocess.Popen(
            ["tar", "-xf", "-", "-C", str(destination_dir)],
            stdin=lz4_proc.stdout,
        )
    finally:
        lz4_proc.stdout.close()

    tar_returncode = tar_proc.wait()
    lz4_returncode = lz4_proc.wait()
    if tar_returncode != 0 or lz4_returncode != 0:
        msg = f"Extraction failed for {archive_path.name}: lz4={lz4_returncode}, tar={tar_returncode}"
        raise RuntimeError(msg)


def cleanup_scratch_workspace(workspace: ScratchWorkspace, *, success: bool) -> bool:
    """Remove a workspace according to its cleanup policy.

    Returns ``True`` when the workspace root was removed.
    """

    should_remove = workspace.cleanup_policy == "always" or (success and workspace.cleanup_policy == "on_success")
    if not should_remove:
        return False
    shutil.rmtree(workspace.root, ignore_errors=True)
    return True


def relink_renamed_inputs(input_dir: Path, rename_map: Mapping[str, str]) -> tuple[Path, ...]:
    """Create unified-ID symlinks for heterodimer renamed inputs.

    For each ``compound_id -> unified_id`` mapping, every regular file whose
    name starts with ``compound_id`` receives a sibling symlink with the same
    suffix and the ``unified_id`` prefix. Existing files or links are left
    untouched, matching the legacy side effect.
    """

    if not rename_map or not input_dir.is_dir():
        return ()

    created: list[Path] = []
    for entry in input_dir.iterdir():
        if not entry.is_file():
            continue
        source_name = entry.name
        for compound_id, unified_id in rename_map.items():
            if not source_name.startswith(compound_id):
                continue
            link_path = input_dir / f"{unified_id}{source_name[len(compound_id) :]}"
            if link_path.exists() or link_path.is_symlink():
                continue
            link_path.symlink_to(entry)
            created.append(link_path)
            break
    return tuple(created)


__all__ = [
    "ArchiveExtractor",
    "cleanup_scratch_workspace",
    "create_scratch_workspace",
    "extract_archive_to_workspace",
    "extract_tar_lz4",
    "plan_scratch_workspace",
    "relink_renamed_inputs",
]
