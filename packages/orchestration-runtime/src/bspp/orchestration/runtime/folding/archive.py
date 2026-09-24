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

"""Pure folding archive membership, command, and manifest planning.

Port Baseline:
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/scripts/archive_openfold_results.py:463-600,728-744,945-982``.
This seam validates source evidence and renders the pinned tar/lz4 pipeline but
does not copy files, create directories, execute commands, or claim success.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import numpy as np

from bspp.orchestration.contract.folding_archive import (
    ArchiveBatchPlan,
    ArchiveManifestRecord,
    ArchiveMember,
    ArchivePlanOptions,
    FoldingArchivePlan,
    FoldingResultAssociation,
    FoldingResultInventory,
    ResultKind,
    archive_manifest_record_from_mapping,
)


def plan_folding_archives(
    inventory: FoldingResultInventory,
    options: ArchivePlanOptions,
    *,
    selected_protein_ids: tuple[str, ...] | None = None,
    prior_manifest: Iterable[ArchiveManifestRecord] = (),
    force: bool = False,
) -> FoldingArchivePlan:
    """Plan complete-pair archive batches without performing any mutation."""
    if not isinstance(force, bool):
        msg = "force must be a boolean"
        raise ValueError(msg)
    prior_records = tuple(prior_manifest)
    prior_by_protein = _validate_prior_manifest(prior_records)
    used_indices = {record.archive_index for record in prior_records}

    association_by_id = {association.protein_id: association for association in inventory.associations}
    selected = None if selected_protein_ids is None else _validate_selection(selected_protein_ids, association_by_id)
    complete = [
        association
        for association in inventory.associations
        if association.status == "complete" and (selected is None or association.protein_id in selected)
    ]

    skipped_prior: tuple[str, ...] = ()
    if not force:
        skipped_prior = tuple(
            association.protein_id for association in complete if association.protein_id in prior_by_protein
        )
        complete = [association for association in complete if association.protein_id not in prior_by_protein]

    if options.shuffle and complete:
        permutation = np.random.RandomState(options.shuffle_seed).permutation(len(complete))
        complete = [complete[int(index)] for index in permutation]

    batch_count = (len(complete) + options.proteins_per_archive - 1) // options.proteins_per_archive
    if options.max_archives is not None:
        batch_count = min(batch_count, options.max_archives)

    batches: list[ArchiveBatchPlan] = []
    next_index = options.start_index
    for offset in range(batch_count):
        start = offset * options.proteins_per_archive
        batch_associations = complete[start : start + options.proteins_per_archive]
        while next_index in used_indices:
            next_index += 1
        batches.append(_plan_batch(batch_associations, options=options, archive_index=next_index))
        used_indices.add(next_index)
        next_index += 1
    return FoldingArchivePlan(batches=tuple(batches), skipped_prior_protein_ids=skipped_prior)


def render_archive_manifest_jsonl(plan: FoldingArchivePlan) -> str:
    """Render stable append-ready JSONL for newly planned membership only."""
    if not plan.batches:
        return ""
    lines = [
        json.dumps(batch.manifest_record.to_mapping(), sort_keys=True, separators=(",", ":")) for batch in plan.batches
    ]
    return "\n".join(lines) + "\n"


def parse_archive_manifest_jsonl(contents: str) -> tuple[ArchiveManifestRecord, ...]:
    """Load the port's planned-membership JSONL, failing closed on drift."""
    records: list[ArchiveManifestRecord] = []
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"archive manifest line {line_number} is not valid JSON: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(value, Mapping):
            msg = f"archive manifest line {line_number} must contain an object"
            raise ValueError(msg)
        records.append(archive_manifest_record_from_mapping(cast(Mapping[str, object], value)))
    _validate_prior_manifest(tuple(records))
    return tuple(records)


def _plan_batch(
    associations: list[FoldingResultAssociation],
    *,
    options: ArchivePlanOptions,
    archive_index: int,
) -> ArchiveBatchPlan:
    batch_id = f"{options.run_tag}_{archive_index:05d}"
    archive_name = f"{batch_id}.tar.lz4"
    stage_dir = Path(options.stage_root) / batch_id
    archive_path = Path(options.archive_root) / archive_name
    used_names: set[str] = set()
    members: list[ArchiveMember] = []
    for association in associations:
        if association.status != "complete" or len(association.pdb_paths) != 1 or len(association.json_paths) != 1:
            msg = f"archive batch contains incomplete result association: {association.protein_id}"
            raise ValueError(msg)
        for kind, source in (("pdb", association.pdb_paths[0]), ("json", association.json_paths[0])):
            source_path = Path(source)
            try:
                is_file = source_path.is_file()
            except OSError as exc:
                msg = f"Cannot inspect archive member {source_path}: {exc}"
                raise ValueError(msg) from exc
            if not is_file:
                msg = f"Archive member is not a readable file: {source_path}"
                raise ValueError(msg)
            stage_name = _unique_stage_name(source_path.name, used_names, association.protein_id)
            members.append(
                ArchiveMember(
                    protein_id=association.protein_id,
                    kind=cast(ResultKind, kind),
                    source_path=str(source_path),
                    stage_name=stage_name,
                )
            )
    protein_ids = tuple(association.protein_id for association in associations)
    member_names = tuple(member.stage_name for member in members)
    manifest = ArchiveManifestRecord(
        archive_status="planned",
        archive_batch_id=batch_id,
        archive_file=archive_name,
        archive_index=archive_index,
        run_tag=options.run_tag,
        protein_ids=protein_ids,
        member_names=member_names,
    )
    return ArchiveBatchPlan(
        archive_index=archive_index,
        archive_name=archive_name,
        archive_path=str(archive_path),
        stage_dir=str(stage_dir),
        protein_ids=protein_ids,
        members=tuple(members),
        tar_argv=("tar", "-cf", "-", "-C", str(stage_dir), "."),
        lz4_argv=(options.lz4_executable, "-1", "-"),
        stdout_path=str(archive_path),
        manifest_record=manifest,
    )


def _unique_stage_name(filename: str, used: set[str], protein_id: str) -> str:
    if filename not in used:
        used.add(filename)
        return filename
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", protein_id)
    candidate = f"{safe_id}__{filename}"
    counter = 1
    while candidate in used:
        candidate = f"{safe_id}__{counter}__{filename}"
        counter += 1
    used.add(candidate)
    return candidate


def _validate_prior_manifest(records: tuple[ArchiveManifestRecord, ...]) -> dict[str, ArchiveManifestRecord]:
    by_protein: dict[str, ArchiveManifestRecord] = {}
    by_index: dict[int, ArchiveManifestRecord] = {}
    for record in records:
        prior_index = by_index.get(record.archive_index)
        if prior_index is not None and prior_index != record:
            msg = f"conflicting prior manifest records use archive index {record.archive_index}"
            raise ValueError(msg)
        by_index[record.archive_index] = record
        for protein_id in record.protein_ids:
            # Explicit force replans prior proteins into fresh archive indices.
            # Preserve append-only replay by letting the latest membership win,
            # matching the baseline manifest application behavior.
            by_protein[protein_id] = record
    return by_protein


def _validate_selection(
    selected: tuple[str, ...],
    association_by_id: dict[str, FoldingResultAssociation],
) -> set[str]:
    if not isinstance(selected, tuple) or any(not isinstance(item, str) or not item.strip() for item in selected):
        msg = "selected_protein_ids must be an immutable tuple of non-empty strings"
        raise ValueError(msg)
    if len(selected) != len(set(selected)):
        msg = "selected_protein_ids must be unique"
        raise ValueError(msg)
    unknown = sorted(set(selected) - set(association_by_id))
    if unknown:
        msg = f"selected archive identities are not present in the result inventory: {', '.join(unknown)}"
        raise ValueError(msg)
    return set(selected)


__all__ = ["parse_archive_manifest_jsonl", "plan_folding_archives", "render_archive_manifest_jsonl"]
