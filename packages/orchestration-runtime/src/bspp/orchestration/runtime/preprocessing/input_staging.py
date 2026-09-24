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

"""Stage the declared preprocessing input before science dispatch.

Downloads a ``VerifiedRemoteInputLocation`` FASTA (or reads a local one),
normalizes headers to identity-only, and materializes per-chunk ``.fa`` files
at the exact ``input_root``/``split_input_root`` paths declared in the Phase
RunSpec so that ``execute-chunk`` preflight passes on a fresh materialize
→ submit → run without manual staging.
"""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.contract.phase import (
    PhaseRunSpec,
    VerifiedLocalInputLocation,
    VerifiedRemoteInputLocation,
)
from bspp.orchestration.runtime.preprocessing.fasta import parse_preprocessing_fasta
from bspp.orchestration.runtime.preprocessing.input_intake import (
    download_remote_fasta,
    normalize_fasta_to_identity_headers,
)


class InputStagingError(Exception):
    """Raised when preprocessing input staging fails closed."""


def stage_preprocessing_input(
    runspec: PhaseRunSpec,
    *,
    workspace_root: Path,
) -> tuple[Path, ...]:
    """Download (if remote) and split the FASTA into per-chunk files.

    Returns the tuple of materialized chunk file paths (search-input paths).
    """
    location = runspec.input_location
    source_path = (workspace_root / location.path).resolve()
    workspace_resolved = workspace_root.resolve()
    if source_path != workspace_resolved and workspace_resolved not in source_path.parents:
        raise InputStagingError("input staging destination escapes the workspace root")

    if isinstance(location, VerifiedRemoteInputLocation):
        verified = download_remote_fasta(location, source_path)
        normalize_fasta_to_identity_headers(verified)
    elif isinstance(location, VerifiedLocalInputLocation):
        if not source_path.is_file():
            raise InputStagingError(f"local input FASTA not found: {source_path}")
        normalize_fasta_to_identity_headers(source_path)
        verified = source_path
    else:
        raise InputStagingError(f"unsupported input location kind: {location.kind!r}")

    records = parse_preprocessing_fasta(
        verified,
        normalization_mode=runspec.payload.work_plan.input.normalization_mode,
    )
    records_by_ordinal = {record.source_ordinal: record for record in records}
    work_plan = runspec.payload.work_plan
    actions = runspec.payload.actions
    by_action_id = {action.action_id: action for action in actions}

    materialized: list[Path] = []
    for chunk in work_plan.chunks:
        action_id = f"preprocessing-chunk-{chunk.ordinal:06d}"
        action = by_action_id.get(action_id)
        if action is None:
            raise InputStagingError(f"no Runtime Action for chunk {chunk.name} ({action_id})")
        site = action.payload.site
        runtime = action.payload.runtime
        runtime_folder = f"n{runtime.slurm_node_id}g{runtime.gpu_id}"

        chunk_records = tuple(records_by_ordinal[ordinal] for ordinal in chunk.record_ordinals)
        chunk_bytes = "".join(f">{record.identity}\n{record.sequence}\n" for record in chunk_records).encode()

        search_input = Path(site.input_root) / runtime_folder / chunk.name
        split_input = Path(site.split_input_root) / chunk.name

        for destination in (search_input, split_input):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise InputStagingError(f"refusing to overwrite existing chunk file: {destination}")
            destination.write_bytes(chunk_bytes)

        materialized.append(search_input)

    return tuple(materialized)


__all__ = ["InputStagingError", "stage_preprocessing_input"]
