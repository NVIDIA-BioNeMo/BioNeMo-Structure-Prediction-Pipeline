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

"""Deterministic local folding-result association without table mutation.

Port Baseline:
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/scripts/archive_openfold_results.py:201-228,313-398``.
The port makes duplicate, collision, malformed, and unplanned evidence explicit
instead of overwriting candidates in a dictionary or mutating a run table.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from bspp.orchestration.contract.folding_archive import (
    FoldingResultAssociation,
    FoldingResultInventory,
    ResultKind,
    ResultStatus,
    UnmatchedFoldingResult,
)
from bspp.orchestration.contract.folding_index import FoldingIndex

_PDB_SUFFIX_RE = re.compile(r"(_unrelaxed|_relaxed)?_rank_\d+_.*$")
_JSON_SUFFIX_RE = re.compile(r"(_scores)?_rank_\d+_.*$")
_ALT_JSON_SUFFIX_RE = re.compile(r"_model_\d+_multimer_v\d+_scores$")
_AF_UNDERSCORE_RE = re.compile(r"AF_(\d)")


def normalize_result_protein_id(path: Path) -> str:
    """Normalize one supported PDB or scores-JSON filename to model identity."""
    suffix = path.suffix.lower()
    stem = path.stem
    if suffix == ".pdb":
        stem = _PDB_SUFFIX_RE.sub("", stem)
        stem = re.sub(r"_unrelaxed$", "", stem)
        stem = re.sub(r"_relaxed$", "", stem)
    elif suffix == ".json":
        stem = _JSON_SUFFIX_RE.sub("", stem)
        stem = _ALT_JSON_SUFFIX_RE.sub("", stem)
        stem = re.sub(r"_scores$", "", stem)
    else:
        msg = f"unsupported folding result suffix: {path}"
        raise ValueError(msg)
    return _normalize_protein_id(stem)


def scan_folding_results(index: FoldingIndex, predictions_root: Path) -> FoldingResultInventory:
    """Scan the baseline root/predictions/gpu layouts and classify every plan."""
    try:
        is_directory = predictions_root.is_dir()
    except OSError as exc:
        msg = f"Cannot inspect folding result root {predictions_root}: {exc}"
        raise ValueError(msg) from exc
    if not is_directory:
        msg = f"Folding result root is not a readable directory: {predictions_root}"
        raise ValueError(msg)

    candidates = _scan_candidates(predictions_root)
    planned_records = tuple(sorted(index.records, key=lambda record: record.source_ordinal))
    planned_by_normalized: dict[str, list[str]] = {}
    for record in planned_records:
        normalized = _normalize_protein_id(record.protein_id)
        if not normalized:
            msg = f"planned protein identity normalizes to blank: {record.protein_id!r}"
            raise ValueError(msg)
        planned_by_normalized.setdefault(normalized, []).append(record.protein_id)

    paths_by_identity: dict[str, dict[ResultKind, list[str]]] = {}
    unmatched: list[UnmatchedFoldingResult] = []
    ordered_valid_candidates: list[tuple[str, ResultKind, str]] = []
    for path, kind in candidates:
        normalized = normalize_result_protein_id(path)
        if not normalized:
            unmatched.append(
                UnmatchedFoldingResult(
                    path=str(path),
                    kind=kind,
                    normalized_protein_id="",
                    reason="malformed-name",
                )
            )
            continue
        ordered_valid_candidates.append((normalized, kind, str(path)))
        paths_by_identity.setdefault(normalized, {"pdb": [], "json": []})[kind].append(str(path))

    for normalized, kind, candidate_path in ordered_valid_candidates:
        if normalized not in planned_by_normalized:
            unmatched.append(
                UnmatchedFoldingResult(
                    path=candidate_path,
                    kind=kind,
                    normalized_protein_id=normalized,
                    reason="unplanned-identity",
                )
            )

    associations: list[FoldingResultAssociation] = []
    for record in planned_records:
        normalized = _normalize_protein_id(record.protein_id)
        grouped = paths_by_identity.get(normalized, {"pdb": [], "json": []})
        pdb_paths = tuple(grouped["pdb"])
        json_paths = tuple(grouped["json"])
        status: ResultStatus
        if len(planned_by_normalized[normalized]) > 1:
            status = "identity-collision"
        elif len(pdb_paths) > 1 or len(json_paths) > 1:
            status = "duplicate"
        elif pdb_paths and json_paths:
            status = "complete"
        elif pdb_paths:
            status = "pdb-only"
        elif json_paths:
            status = "json-only"
        else:
            status = "missing"
        associations.append(
            FoldingResultAssociation(
                source_ordinal=record.source_ordinal,
                protein_id=record.protein_id,
                normalized_protein_id=normalized,
                status=status,
                pdb_paths=pdb_paths,
                json_paths=json_paths,
            )
        )
    return FoldingResultInventory(associations=tuple(associations), unmatched=tuple(unmatched))


def _normalize_protein_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return _AF_UNDERSCORE_RE.sub(r"AF-\1", text)


def _scan_candidates(root: Path) -> tuple[tuple[Path, ResultKind], ...]:
    directories = _candidate_directories(root)
    candidates: list[tuple[Path, ResultKind]] = []
    for directory in directories:
        try:
            with os.scandir(directory) as iterator:
                entries = tuple(sorted(iterator, key=lambda entry: entry.name))
        except OSError as exc:
            msg = f"Cannot scan folding result directory {directory}: {exc}"
            raise ValueError(msg) from exc
        for entry in entries:
            try:
                is_file = entry.is_file(follow_symlinks=False)
            except OSError as exc:
                msg = f"Cannot inspect folding result candidate {entry.path}: {exc}"
                raise ValueError(msg) from exc
            if not is_file:
                continue
            suffix = Path(entry.name).suffix.lower()
            if suffix not in {".pdb", ".json"}:
                continue
            path = Path(entry.path)
            candidates.append((path, "pdb" if suffix == ".pdb" else "json"))
    return tuple(candidates)


def _candidate_directories(root: Path) -> tuple[Path, ...]:
    directories: list[Path] = []
    physical_directories: set[Path] = set()

    def append_once(directory: Path) -> None:
        try:
            physical = directory.resolve(strict=True)
        except OSError as exc:
            msg = f"Cannot resolve folding result directory {directory}: {exc}"
            raise ValueError(msg) from exc
        if physical not in physical_directories:
            directories.append(directory)
            physical_directories.add(physical)

    append_once(root)
    predictions = root / "predictions"
    try:
        predictions_is_directory = predictions.is_dir()
    except OSError as exc:
        msg = f"Cannot inspect folding result predictions directory {predictions}: {exc}"
        raise ValueError(msg) from exc
    if predictions_is_directory:
        append_once(predictions)
    try:
        with os.scandir(root) as iterator:
            entries = tuple(sorted(iterator, key=lambda entry: entry.name))
    except OSError as exc:
        msg = f"Cannot scan folding result root {root}: {exc}"
        raise ValueError(msg) from exc
    for entry in entries:
        try:
            if entry.name.startswith("gpu") and entry.is_dir(follow_symlinks=False):
                append_once(Path(entry.path))
        except OSError as exc:
            msg = f"Cannot inspect folding result layout entry {entry.path}: {exc}"
            raise ValueError(msg) from exc
    return tuple(directories)


__all__ = ["normalize_result_protein_id", "scan_folding_results"]
