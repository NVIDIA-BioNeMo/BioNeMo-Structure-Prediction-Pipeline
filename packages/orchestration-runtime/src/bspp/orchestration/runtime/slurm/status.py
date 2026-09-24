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

"""Query submitit job states.

submitit persists per-job state to ``<log_folder>/<job_id>_<task_id>.pkl``
and exposes it via :class:`submitit.Job`. This wrapper returns a
compact, typed view so callers (e.g. the CLI) don't depend directly on
submitit's types.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import submitit


@dataclass(frozen=True)
class JobStatus:
    """Compact status view for one submitit job (possibly an array task)."""

    job_id: str
    state: str
    done: bool
    stderr_tail: str = ""


def query_jobs(job_ids: Iterable[str], *, log_folder: Path) -> list[JobStatus]:
    """Return :class:`JobStatus` for each *job_id*.

    ``job_id`` may be either a base id (``12345``) or an array-task id
    (``12345_7``) — both are accepted by submitit.
    """
    import submitit

    results: list[JobStatus] = []
    for job_id in job_ids:
        job: submitit.Job[object] = submitit.Job(folder=log_folder, job_id=job_id)
        try:
            state = job.state
        except Exception as exc:
            state = f"UNKNOWN ({exc})"
        stderr_tail = ""
        if job.done():
            try:
                stderr_tail = (job.stderr() or "")[-500:]
            except Exception:
                stderr_tail = ""
        results.append(
            JobStatus(
                job_id=job_id,
                state=state,
                done=job.done(),
                stderr_tail=stderr_tail,
            )
        )
    return results


def render_job_statuses(statuses: Iterable[JobStatus]) -> str:
    """Format a list of job statuses as a plain-text table."""
    lines = [f"{'Job ID':<20} {'State':<12} {'Done':>5}"]
    lines.append("-" * 38)
    for s in statuses:
        lines.append(f"{s.job_id:<20} {s.state:<12} {'yes' if s.done else 'no':>5}")
    return "\n".join(lines)


__all__ = ["JobStatus", "query_jobs", "render_job_statuses"]
