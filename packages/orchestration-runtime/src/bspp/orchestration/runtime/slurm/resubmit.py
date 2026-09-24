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

"""Resubmit only failed shards for a dataset.

Drives :mod:`bspp.orchestration.runtime.validation.count_outputs` to identify
shards that didn't produce the expected number of outputs, then hands
the trimmed shard-id list to :func:`bspp.orchestration.runtime.slurm.submit.submit_array`.
"""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.runtime.slurm.submit import SubmitSummary, submit_array
from bspp.orchestration.runtime.validation.count_outputs import count_shard_outputs


def resubmit_failed_shards(
    recipe_dir: Path,
    *,
    dataset: str,
    output_base: Path,
    input_dir: Path,
    manifest_csv: Path | None = None,
    uniprot_db: Path | None = None,
    stages: str = "ipsae dssp validation metadata_export modelcif_export",
    workers: int = 24,
    cli_path: str = "bspp-orchestration-runtime",
    dry_run: bool = False,
) -> tuple[list[int], SubmitSummary | None]:
    """Scan *output_base*/*dataset* for failed shards and resubmit them.

    Returns ``(failed_ids, submit_summary)``. ``submit_summary`` is
    :data:`None` when no shards needed resubmitting.
    """
    dataset_dir = output_base / dataset
    report = count_shard_outputs(dataset_dir)
    failed = report.failed_ids

    if not failed:
        return [], None

    summary = submit_array(
        recipe_dir,
        failed,
        input_dir=input_dir,
        output_dir=dataset_dir,
        manifest_csv=manifest_csv,
        uniprot_db=uniprot_db,
        stages=stages,
        workers=workers,
        cli_path=cli_path,
        dry_run=dry_run,
    )
    return failed, summary


__all__ = ["resubmit_failed_shards"]
