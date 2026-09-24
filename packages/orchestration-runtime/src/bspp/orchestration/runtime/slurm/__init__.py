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

"""SLURM submission via submitit (facebookincubator/submitit).

Three entry points:

- :mod:`bspp.orchestration.runtime.slurm.submit` — submit an array job across
  a list of shard IDs; each array task subprocesses the orchestration
  CLI's ``process`` subcommand for one shard.
- :mod:`bspp.orchestration.runtime.slurm.resubmit` — derive the failed-shard
  list via :mod:`bspp.orchestration.runtime.validation.count_outputs` and
  submit only those.
- :mod:`bspp.orchestration.runtime.slurm.status` — query job states for a
  previously-submitted job folder.

All three go through :func:`bspp.orchestration.runtime.slurm.executor.make_executor`
which builds a :class:`submitit.AutoExecutor` from the recipe's
``config.yaml`` ``slurm.*`` block.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "JobStatus",
    "SubmitSummary",
    "make_executor",
    "query_jobs",
    "resubmit_failed_shards",
    "run_shard",
    "submit_array",
]


def __getattr__(name: str) -> Any:
    """Import public SLURM helpers lazily to avoid validation import cycles."""
    if name == "make_executor":
        from bspp.orchestration.runtime.slurm.executor import make_executor

        return make_executor
    if name == "resubmit_failed_shards":
        from bspp.orchestration.runtime.slurm.resubmit import resubmit_failed_shards

        return resubmit_failed_shards
    if name in {"JobStatus", "query_jobs"}:
        from bspp.orchestration.runtime.slurm.status import JobStatus, query_jobs

        return {"JobStatus": JobStatus, "query_jobs": query_jobs}[name]
    if name in {"SubmitSummary", "run_shard", "submit_array"}:
        from bspp.orchestration.runtime.slurm.submit import SubmitSummary, run_shard, submit_array

        return {"SubmitSummary": SubmitSummary, "run_shard": run_shard, "submit_array": submit_array}[name]
    raise AttributeError(name)
