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

"""Submit an array of shards via submitit.

Each array task is a Python function that shells out to the
orchestration CLI's ``process`` subcommand for one shard. We keep the
work unit as a subprocess so the pipeline logic stays in one place
(the existing ``process`` command) and submitit is only responsible
for SLURM submission + checkpoint-on-preempt.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from bspp.orchestration.runtime.slurm.executor import make_executor

if TYPE_CHECKING:
    import submitit


def run_shard(
    shard_id: int,
    recipe_dir: str,
    input_dir: str,
    output_dir: str,
    manifest_csv: str | None,
    uniprot_db: str | None,
    stages: str,
    workers: int,
    cli_path: str,
) -> int:
    """Run a single shard by invoking ``bspp-orchestration-runtime process``.

    Note: **every parameter is positional** (no keyword-only marker).
    ``submit_array`` hands this function to :meth:`submitit.AutoExecutor.map_array`,
    which internally wraps each tuple of inputs in a
    :class:`~submitit.core.utils.DelayedSubmission` and calls
    ``fn(*args)`` at shard-run time. A keyword-only signature would make
    every real (non-dry-run) shard crash with ``TypeError: run_shard()
    takes 1 positional argument but 9 were given``. Keep the parameter
    list aligned with the call site in :func:`submit_array`.

    Returns the subprocess return code (0 on success). Raises
    :class:`subprocess.CalledProcessError` only if the CLI itself fails
    to launch; a non-zero pipeline exit is returned as-is for the
    caller (submitit) to record.
    """
    argv: list[str] = [
        cli_path,
        "process",
        "--shard-id",
        str(shard_id),
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--stages",
        stages,
        "--workers",
        str(workers),
    ]
    if manifest_csv:
        argv.extend(["--manifest-csv", manifest_csv])
    if uniprot_db:
        argv.extend(["--uniprot-db", uniprot_db])

    completed = subprocess.run(argv, check=False)
    return completed.returncode


@dataclass(frozen=True)
class SubmitSummary:
    """Summary of a submitit array submission."""

    job_ids: tuple[str, ...]
    shard_ids: tuple[int, ...]
    log_folder: Path
    dry_run: bool = False
    argv_preview: tuple[str, ...] = field(default_factory=tuple)


def submit_array(
    recipe_dir: Path,
    shard_ids: Sequence[int],
    *,
    input_dir: Path,
    output_dir: Path,
    manifest_csv: Path | None = None,
    uniprot_db: Path | None = None,
    stages: str = "ipsae dssp validation metadata_export modelcif_export",
    workers: int = 24,
    cli_path: str = "bspp-orchestration-runtime",
    dry_run: bool = False,
) -> SubmitSummary:
    """Submit *shard_ids* as one SLURM array via submitit's ``map_array``.

    When ``dry_run=True`` the argv for a representative shard is returned
    without invoking SLURM.
    """
    if not shard_ids:
        msg = "shard_ids must be non-empty"
        raise ValueError(msg)

    executor = make_executor(recipe_dir)

    if dry_run:
        preview = (
            cli_path,
            "process",
            "--shard-id",
            str(shard_ids[0]),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--stages",
            stages,
            "--workers",
            str(workers),
        )
        return SubmitSummary(
            job_ids=(),
            shard_ids=tuple(shard_ids),
            log_folder=Path(executor.folder),
            dry_run=True,
            argv_preview=preview,
        )

    recipe_arg = [str(recipe_dir)] * len(shard_ids)
    input_arg = [str(input_dir)] * len(shard_ids)
    output_arg = [str(output_dir)] * len(shard_ids)
    manifest_arg = [str(manifest_csv) if manifest_csv else None] * len(shard_ids)
    uniprot_arg = [str(uniprot_db) if uniprot_db else None] * len(shard_ids)
    stages_arg = [stages] * len(shard_ids)
    workers_arg = [workers] * len(shard_ids)
    cli_arg = [cli_path] * len(shard_ids)

    jobs: list[submitit.Job[int]] = executor.map_array(
        run_shard,
        list(shard_ids),
        recipe_arg,
        input_arg,
        output_arg,
        manifest_arg,
        uniprot_arg,
        stages_arg,
        workers_arg,
        cli_arg,
    )

    return SubmitSummary(
        job_ids=tuple(job.job_id for job in jobs),
        shard_ids=tuple(shard_ids),
        log_folder=Path(executor.folder),
    )


__all__ = ["SubmitSummary", "run_shard", "submit_array"]
