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

"""Temporary recovery tripwire for the restored Python scheduler architecture.

Removing or relaxing this guard requires an explicit, reviewed architecture
decision after the recovery is complete. Documentation is deliberately outside
the literal scan so historical negative evidence remains available.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

SCHEDULER_SURFACES = (
    "packages/orchestration-control/src/bspp/orchestration/control/workflow_rendering.py",
    "packages/orchestration-runtime/src/bspp/orchestration/runtime/slurm",
    "tests/test_slurm",
)

PATH_SCAN_ROOTS = ("packages", "containers", "tests")
LITERAL_SCAN_ROOTS = ("packages", "containers", "pyproject.toml", "uv.lock")
BANNED_PRODUCTION_LITERALS = (
    "nextflow",
    "nxf_",
    "nf-amazon",
    "nextflow_session_id",
    "nextflow_task_id",
)


def _tracked_paths(*roots: str) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *roots],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(REPOSITORY_ROOT / os.fsdecode(path) for path in result.stdout.split(b"\0") if path)


@pytest.mark.parametrize("relative_path", SCHEDULER_SURFACES)
def test_python_scheduler_surface_is_present(relative_path: str) -> None:
    assert (REPOSITORY_ROOT / relative_path).exists(), relative_path


def test_tracked_production_paths_exclude_workflow_engine_assets() -> None:
    violations: list[str] = []

    for path in _tracked_paths(*PATH_SCAN_ROOTS):
        relative_path = path.relative_to(REPOSITORY_ROOT)
        names = tuple(part.casefold() for part in relative_path.parts)
        if (
            any(suffix.casefold() == ".nf" for suffix in relative_path.suffixes)
            or relative_path.name.casefold() == "nextflow.config"
            or any(name.startswith("nextflow") for name in names)
        ):
            violations.append(relative_path.as_posix())

    assert violations == []


def test_tracked_production_text_excludes_workflow_engine_runtime_literals() -> None:
    violations: list[str] = []

    for path in _tracked_paths(*LITERAL_SCAN_ROOTS):
        try:
            text = path.read_text(encoding="utf-8").casefold()
        except UnicodeDecodeError:
            continue
        matched = [literal for literal in BANNED_PRODUCTION_LITERALS if literal.casefold() in text]
        if matched:
            relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
            violations.append(f"{relative_path}: {', '.join(matched)}")

    assert violations == []
