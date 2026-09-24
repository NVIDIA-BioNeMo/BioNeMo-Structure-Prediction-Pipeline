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

"""Build a submitit executor from a recipe's ``config.yaml``.

The recipe's ``slurm.*`` block names follow SLURM's ``#SBATCH`` flags;
submitit uses its own canonical parameter names. This module maps the
common fields and forwards everything else via
``slurm_additional_parameters``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from bspp.orchestration.runtime.postprocessing.recipe import load_recipe
from bspp.orchestration.runtime.slurm.arrays import array_parallelism

if TYPE_CHECKING:
    import submitit


def _time_to_minutes(value: str | int | None) -> int:
    """Parse a SLURM time string (``HH:MM:SS`` / ``D-HH:MM:SS``) to minutes.

    Also accepts a bare int (treated as minutes) for convenience.
    """
    if value is None:
        return 60
    if isinstance(value, int):
        return value
    text = value.strip()
    days = 0
    if "-" in text:
        days_part, _, text = text.partition("-")
        days = int(days_part)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.append(0)
    hours, minutes, seconds = parts[0], parts[1], parts[2]
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return max(1, total_seconds // 60)


def make_executor(
    recipe_dir: Path,
    *,
    log_folder: Path | None = None,
    cluster: str = "slurm",
) -> submitit.AutoExecutor:
    """Build a :class:`submitit.AutoExecutor` from ``recipe_dir/config.yaml``.

    Recipe mapping:
      ``slurm.partition``       -> ``slurm_partition``
      ``slurm.account``         -> ``slurm_additional_parameters['account']``
      ``slurm.cpus_per_task``   -> ``cpus_per_task``
      ``slurm.memory``          -> ``slurm_mem``
      ``slurm.gres``            -> ``slurm_gres``
      ``slurm.time``            -> ``timeout_min`` (parsed)
      ``slurm.array_range``     -> ``slurm_array_parallelism`` (if ``%N`` present)
      ``job_name``              -> ``name``
      ``paths.log_dir``         -> ``log_folder`` (if *log_folder* arg omitted)

    The default ``cluster="slurm"`` forces submitit's SlurmExecutor. We
    do **not** let ``submitit.AutoExecutor`` autodetect, because when
    ``sbatch`` is absent it silently falls back to ``LocalExecutor`` and
    drops every ``slurm_*`` resource hint — a command named
    ``slurm submit`` then quietly runs the array on the current host
    with whatever CPUs/RAM it happens to have. SlurmExecutor instead
    raises a clear ``RuntimeError("Could not detect srun, are you
    indeed on a slurm cluster?")`` when SLURM is missing, which is the
    failure mode we want. Callers that genuinely want local execution
    (e.g. ``debug`` or ``local`` for smoke-testing) pass ``cluster``
    explicitly.
    """
    import submitit  # lazy import so the rest of the package works without it

    config = load_recipe(recipe_dir)
    slurm_cfg: dict[str, Any] = dict(config.get("slurm") or {})
    paths_cfg: dict[str, Any] = dict(config.get("paths") or {})

    folder = log_folder if log_folder is not None else Path(paths_cfg.get("log_dir") or recipe_dir / "slurm_logs")
    folder.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=folder, cluster=cluster)
    extra: dict[str, Any] = {}

    if account := slurm_cfg.get("account"):
        extra["account"] = account
    if export := slurm_cfg.get("export"):
        extra["export"] = export

    parameters: dict[str, Any] = {
        "name": config.get("job_name") or f"bspp_{config.get('run_name', 'run')}",
        "timeout_min": _time_to_minutes(slurm_cfg.get("time")),
    }
    if partition := slurm_cfg.get("partition"):
        parameters["slurm_partition"] = partition
    if gres := slurm_cfg.get("gres"):
        parameters["slurm_gres"] = gres
    if mem := slurm_cfg.get("memory") or slurm_cfg.get("mem"):
        parameters["slurm_mem"] = mem
    if cpus := slurm_cfg.get("cpus_per_task"):
        parameters["cpus_per_task"] = int(cpus)
    parallelism = array_parallelism(slurm_cfg.get("array_range"))
    if parallelism > 0:
        parameters["slurm_array_parallelism"] = parallelism
    if extra:
        parameters["slurm_additional_parameters"] = extra

    executor.update_parameters(**parameters)
    return executor


__all__ = ["make_executor"]
