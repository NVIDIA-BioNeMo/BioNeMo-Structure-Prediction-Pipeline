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

"""Proof-based safety gate for V2 first submission and explicit retry."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from bspp.orchestration.contract.runplan import RunPlan, reject_environment_interpolation
from bspp.orchestration.control.postprocessing_phase_types import PostprocessingAuthority

V2ProjectionSafetyStatus = Literal["safe", "broken", "unverifiable"]


@dataclass(frozen=True)
class V2ProjectionSafety:
    status: V2ProjectionSafetyStatus
    reason: str


def classify_v2_projection(authority: PostprocessingAuthority) -> V2ProjectionSafety:
    """Compare the projection selector with the pinned authored Run Plan bytes."""
    reference = authority.phase_plan.legacy_run_plan
    path = Path(reference.path)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or path.is_symlink():
            return V2ProjectionSafety("unverifiable", "pinned legacy Run Plan is not a regular non-symlink file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                return V2ProjectionSafety("unverifiable", "pinned legacy Run Plan changed during open")
            payload = handle.read(reference.size_bytes + 1)
    except OSError as exc:
        return V2ProjectionSafety("unverifiable", f"pinned legacy Run Plan is unavailable: {exc}")
    if before.st_size != reference.size_bytes or len(payload) != reference.size_bytes:
        return V2ProjectionSafety("unverifiable", "pinned legacy Run Plan size differs from Phase Plan")
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        return V2ProjectionSafety("unverifiable", "pinned legacy Run Plan SHA-256 differs from Phase Plan")
    try:
        mapping = yaml.safe_load(payload)
        if not isinstance(mapping, dict):
            raise TypeError("Run Plan must be a mapping")
        reject_environment_interpolation(mapping, context="Run Plan")
        authored = RunPlan.model_validate(mapping)
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        return V2ProjectionSafety("unverifiable", f"pinned legacy Run Plan cannot be parsed: {exc}")
    projected = authority.legacy_runspec.dataset.name
    if projected != authored.dataset.name:
        return V2ProjectionSafety(
            "broken",
            "stored execution projection dataset.name differs from the pinned authored tracking selector",
        )
    return V2ProjectionSafety("safe", "stored execution projection preserves the pinned authored tracking selector")


def require_safe_v2_projection(authority: PostprocessingAuthority) -> None:
    result = classify_v2_projection(authority)
    if result.status != "safe":
        raise ValueError(
            f"postprocessing V2 projection is {result.status}: {result.reason}; "
            "rematerialize the Phase Plan as an explicit V3 authority"
        )


__all__ = [
    "V2ProjectionSafety",
    "V2ProjectionSafetyStatus",
    "classify_v2_projection",
    "require_safe_v2_projection",
]
