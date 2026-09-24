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

"""Shared parser for SLURM array specifications."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bspp.orchestration.contract.runspec import RunSpec

_RANGE_RE = re.compile(r"(?P<start>\d+)(?:-(?P<end>\d+))?")


@dataclass(frozen=True)
class SlurmArraySpec:
    """Parsed SLURM array ranges plus optional ``%N`` throttle."""

    ranges: tuple[tuple[int, int], ...]
    throttle: int | None = None

    @property
    def task_count(self) -> int:
        """Return the number of array task IDs covered by the spec."""
        return sum(end - start + 1 for start, end in self.ranges)

    @property
    def parallelism(self) -> int:
        """Return submitit-compatible parallelism, using ``0`` when unthrottled."""
        return self.throttle or 0

    def render(self) -> str:
        """Render a normalized SLURM array spec."""
        chunks = [str(start) if start == end else f"{start}-{end}" for start, end in self.ranges]
        rendered = ",".join(chunks)
        if self.throttle is not None:
            rendered = f"{rendered}%{self.throttle}"
        return rendered


def parse_array_spec(value: str) -> SlurmArraySpec:
    """Parse a SLURM array spec with comma ranges and an optional throttle."""
    text = value.strip()
    if not text:
        msg = "SLURM array spec is empty"
        raise ValueError(msg)
    if text.count("%") > 1:
        msg = f"Invalid SLURM array throttle syntax: {value}"
        raise ValueError(msg)

    range_text, throttle_text = text, None
    if "%" in text:
        range_text, throttle_text = text.split("%", 1)
        throttle_text = throttle_text.strip()
        if not throttle_text or not throttle_text.isdigit():
            msg = f"Invalid SLURM array throttle: {value}"
            raise ValueError(msg)

    throttle = int(throttle_text) if throttle_text is not None else None
    if throttle is not None and throttle <= 0:
        msg = f"Invalid SLURM array throttle: {value}"
        raise ValueError(msg)

    ranges: list[tuple[int, int]] = []
    for raw_chunk in range_text.split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            msg = f"Invalid SLURM array range: {value}"
            raise ValueError(msg)
        match = _RANGE_RE.fullmatch(chunk)
        if match is None:
            msg = f"Invalid SLURM array range: {value}"
            raise ValueError(msg)
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end < start:
            msg = f"Invalid SLURM array range: {value}"
            raise ValueError(msg)
        ranges.append((start, end))

    return SlurmArraySpec(tuple(ranges), throttle=throttle)


def array_task_count(value: str) -> int:
    """Return how many task IDs a SLURM array spec covers."""
    return parse_array_spec(value).task_count


def array_task_ids(value: str) -> tuple[int, ...]:
    """Return every numeric task ID covered by a SLURM array spec."""
    spec = parse_array_spec(value)
    task_ids: list[int] = []
    for start, end in spec.ranges:
        task_ids.extend(range(start, end + 1))
    return tuple(task_ids)


def array_parallelism(value: str | None, *, default: int = 0) -> int:
    """Return ``%N`` throttle from a SLURM array spec, or *default*."""
    if not value:
        return default
    try:
        return parse_array_spec(value).parallelism
    except ValueError:
        return default


def render_zero_based_array(count: int, *, throttle: int | None = None) -> str:
    """Render ``0-(count-1)`` with an optional throttle."""
    if count <= 0:
        msg = f"SLURM array count must be positive, got {count}"
        raise ValueError(msg)
    if throttle is not None and throttle <= 0:
        msg = f"SLURM array throttle must be positive, got {throttle}"
        raise ValueError(msg)
    spec = SlurmArraySpec(((0, count - 1),), throttle=throttle)
    return spec.render()


def effective_archive_array(spec: RunSpec, archive_count: int) -> str:
    """Return the validated archive array selected by resource, dataset, or count."""
    gpu = spec.resources.get("gpu_worker")
    if gpu is not None and gpu.array:
        parse_array_spec(gpu.array)
        return gpu.array.strip()
    if spec.dataset.array:
        parse_array_spec(spec.dataset.array)
        return spec.dataset.array.strip()
    if spec.validation.expected_archives is not None and spec.validation.expected_archives > 0:
        return render_zero_based_array(spec.validation.expected_archives)
    return render_zero_based_array(archive_count)


__all__ = [
    "SlurmArraySpec",
    "array_parallelism",
    "array_task_count",
    "array_task_ids",
    "effective_archive_array",
    "parse_array_spec",
    "render_zero_based_array",
]
