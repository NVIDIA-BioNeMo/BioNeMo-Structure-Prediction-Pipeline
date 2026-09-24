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

"""Folding→postprocessing seam derivation: master-parquet projection + tracking parquet.

This is a pure, local control-side seam step. Given a completed folding run's
canonical-pair index and canonical-pair action evidence, it derives the
complete master-parquet rows and the postprocessing tracking
parquet, so a postprocessing Phase Plan can pin ``references.master_parquet``
and ``references.tracking_parquet`` to folding output.

It reuses, unchanged:

* ``runtime/folding/execution/master_parquet_writer.py`` — ``build_master_parquet_row``
  and ``write_master_parquet`` (the master-parquet projection producer).
* ``runtime/postprocessing/tracking.py`` — ``create_tracking_parquet`` (the
  master → tracking derivation).

The only new logic here is the deterministic mapping from a canonical-pair
entry to ``build_master_parquet_row`` kwargs. Postprocessing behavior is not
changed. The derivation is a pure local transform: it reads already-fetched
evidence files and writes local parquets, and it prevalidates every input
(including the output prefix) before writing any file, so a failure can never
leave a partial master/tracking pair.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from bspp.orchestration.contract.folding_evidence import CanonicalPairActionEvidence
from bspp.orchestration.runtime.folding.benchmark.index import load_canonical_pair_index
from bspp.orchestration.runtime.folding.execution.master_parquet_writer import (
    build_master_parquet_row,
    write_master_parquet,
)
from bspp.orchestration.runtime.postprocessing.tracking import create_tracking_parquet

__all__ = [
    "SeamDerivationError",
    "SeamDerivationResult",
    "derive_seam_parquets",
]


class SeamDerivationError(RuntimeError):
    """Fail-closed error surface for the folding→postprocessing seam derivation."""


@dataclass(frozen=True)
class SeamDerivationResult:
    """The two reference inputs a postprocessing Phase Plan pins."""

    master_path: Path
    tracking_path: Path
    row_count: int


@contextmanager
def _destination_lock(*destinations: Path) -> Iterator[None]:
    """Serialize derivations targeting the same master/tracking pair.

    Locks every distinct destination (master and tracking, which callers choose
    independently) in sorted canonical order, so concurrent derivations that
    share either output are serialized and cannot both pass the
    existing-output check and overwrite each other's files. Each destination's
    parent directory is created first so the lock file has a home even when the
    caller points at a not-yet-existing directory.
    """
    lock_fds: list[int] = []
    try:
        for destination in sorted({os.fspath(d.resolve()) for d in destinations}):
            lock_path = Path(destination).with_name(Path(destination).name + ".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            lock_fds.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        for fd in reversed(lock_fds):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _load_json_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SeamDerivationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SeamDerivationError(f"expected a JSON object at {path}")
    return payload


def _mean(values: tuple[float | int, ...]) -> float:
    return sum(values) / len(values)


def _plddt_above_70(values: tuple[float | int, ...]) -> float:
    return sum(1 for value in values if value >= 70.0) / len(values)


def _require_s3_output_prefix(value: str) -> None:
    if not value or not value.startswith("s3://"):
        raise SeamDerivationError(f"s3_output_prefix must be a non-empty s3:// URI, got {value!r}")


def _require_gcs_destination_prefix(value: str) -> None:
    if not value.startswith("gs://"):
        raise SeamDerivationError(f"gcs_destination_prefix must be a non-empty gs:// URI, got {value!r}")


def _build_row(
    *,
    target_id: str,
    structure_path: str,
    scores_path: str,
    plddt: tuple[float | int, ...],
    max_pae: float | int,
    ptm: float | int | None,
    iptm: float | int | None,
    source_run: str,
    archive_name: str,
) -> dict[str, object]:
    row = build_master_parquet_row(
        msa_path=target_id,
        source_run=source_run,
        pdb_path=structure_path,
        json_path=scores_path,
        pdb_residue_count=len(plddt),
        mean_plddt=_mean(plddt),
        plddt_above_70=_plddt_above_70(plddt),
        ptm=None if ptm is None else float(ptm),
        iptm=None if iptm is None else float(iptm),
        max_pae=float(max_pae),
        output_has_nan="no",
        pdb_json_match="not_checked",
        swiftstack_archive=archive_name,
        uploaded_to_gcp="no",
        seq_length=len(plddt),
    )
    if row is None:
        raise SeamDerivationError(f"master-parquet row for target {target_id!r} has a pending required field")
    return row


def derive_seam_parquets(
    *,
    index_path: Path,
    evidence_path: Path,
    master_output: Path,
    tracking_output: Path,
    s3_output_prefix: str,
    source_run: str,
    archive_name: str,
    phase_run_id: str | None = None,
    gcs_destination_prefix: str | None = None,
    force: bool = False,
) -> SeamDerivationResult:
    """Derive the master + tracking parquets from a completed folding run.

    Loads the canonical-pair index and the canonical-pair action
    evidence, matches them by ``target_id`` (exact same set), derives one
    complete master-parquet row per target, writes the master parquet, then
    derives the tracking parquet from it via ``create_tracking_parquet``.

    ``phase_run_id``, when supplied, must equal the index's ``run_id`` — the
    derivation is provenance-bound to the validated Phase Run and fails closed
    on mismatch. Every index entry is reconciled against its evidence entry on
    ``sequence_sha256``, ``model_entity_id``, ``structure_path``,
    ``scores_path``, and ``tool_used``.

    ``source_run`` is the authored postprocessing ``dataset.name`` selector:
    ``create_tracking_parquet`` hardcodes ``dataset_name = source_run``, and the
    frozen postprocessing ``archive_source: tracking`` path filters by that
    value, so it must not be left at the folding ``phase-run-…`` id.

    ``archive_name`` is the prediction bundle name (``PredictionArchiveBundle.
    bundle_name``) that already exists at this point in the
    publication order (pair validation → archive evidence → transfer state).
    It populates the required ``swiftstack_archive`` column.

    Fails closed on any missing/malformed input, index/evidence mismatch, an
    incomplete row, or a parquet write failure. Every input (including the
    output prefix and the existing-output check) is prevalidated before any
    file is written, so a failure can never leave a partial master/tracking
    pair. ``ptm``/``iptm`` may be ``None`` (engine-optional resolved null —
    BioIR).
    """
    if not source_run:
        raise SeamDerivationError("source_run must be a non-empty dataset selector")
    if not archive_name:
        raise SeamDerivationError("archive_name must be a non-empty prediction bundle name")
    _require_s3_output_prefix(s3_output_prefix)
    if gcs_destination_prefix is not None:
        _require_gcs_destination_prefix(gcs_destination_prefix)

    try:
        index = load_canonical_pair_index(index_path)
    except (OSError, ValueError) as exc:
        raise SeamDerivationError(f"cannot load canonical-pair index {index_path}: {exc}") from exc

    if phase_run_id is not None and index.run_id != phase_run_id:
        raise SeamDerivationError(
            f"canonical-pair index run_id {index.run_id!r} does not match the validated phase_run_id {phase_run_id!r}"
        )

    try:
        evidence = CanonicalPairActionEvidence.from_mapping(_load_json_mapping(evidence_path))
    except ValueError as exc:
        raise SeamDerivationError(f"cannot load canonical-pair evidence {evidence_path}: {exc}") from exc

    if not index.entries:
        raise SeamDerivationError("canonical-pair index has no entries")

    evidence_by_target = {entry.target_id: entry for entry in evidence.entries}
    if len(evidence_by_target) != len(evidence.entries):
        raise SeamDerivationError("canonical-pair evidence has duplicate target_id entries")
    index_targets = {entry.target_id for entry in index.entries}
    if index_targets != set(evidence_by_target):
        raise SeamDerivationError("canonical-pair index and evidence target_id sets do not match")

    rows: list[dict[str, object]] = []
    for index_entry in index.entries:
        evidence_entry = evidence_by_target[index_entry.target_id]
        if evidence_entry.sequence_sha256 != index_entry.sequence_sha256:
            raise SeamDerivationError(
                f"target {index_entry.target_id!r} sequence_sha256 differs between index and evidence"
            )
        pair = evidence_entry.pair
        if pair.model_entity_id != index_entry.model_entity_id:
            raise SeamDerivationError(
                f"target {index_entry.target_id!r} model_entity_id differs between index and evidence"
            )
        if pair.tool_used != index_entry.tool_used:
            raise SeamDerivationError(f"target {index_entry.target_id!r} tool_used differs between index and evidence")
        if pair.structure_path != index_entry.structure_path:
            raise SeamDerivationError(
                f"target {index_entry.target_id!r} structure_path differs between index and evidence"
            )
        if pair.scores_path != index_entry.scores_path:
            raise SeamDerivationError(
                f"target {index_entry.target_id!r} scores_path differs between index and evidence"
            )
        rows.append(
            _build_row(
                target_id=index_entry.target_id,
                structure_path=pair.structure_path,
                scores_path=pair.scores_path,
                plddt=pair.scores.plddt,
                max_pae=pair.scores.max_pae,
                ptm=pair.scores.ptm,
                iptm=pair.scores.iptm,
                source_run=source_run,
                archive_name=archive_name,
            )
        )

    with _destination_lock(master_output, tracking_output):
        if not force and master_output.exists():
            raise SeamDerivationError(f"master output already exists: {master_output} (use force=True)")
        if not force and tracking_output.exists():
            raise SeamDerivationError(f"tracking output already exists: {tracking_output} (use force=True)")

        master_tmp = master_output.with_name(master_output.name + f".tmp-{uuid4().hex}")
        tracking_tmp = tracking_output.with_name(tracking_output.name + f".tmp-{uuid4().hex}")
        master_backup = master_output.with_name(master_output.name + f".bak-{uuid4().hex}")
        tracking_backup = tracking_output.with_name(tracking_output.name + f".bak-{uuid4().hex}")
        published: list[Path] = []
        backed_up: list[tuple[Path, Path]] = []
        try:
            # Stage both outputs to temporary paths.
            write_master_parquet(rows, master_tmp)
            create_tracking_parquet(
                master_tmp,
                tracking_tmp,
                s3_output_prefix=s3_output_prefix,
                gcs_destination_prefix=gcs_destination_prefix,
            )

            # Preserve any existing outputs under uniquely named backups so a
            # failed forced replacement can restore them instead of leaving a
            # master without its matching tracking parquet. Unique names never
            # collide with or overwrite a legitimate sibling file.
            for original, backup in ((master_output, master_backup), (tracking_output, tracking_backup)):
                if original.exists():
                    os.replace(original, backup)
                    backed_up.append((backup, original))

            # Publish the staged outputs together. The master rename is last and
            # acts as the commit marker.
            os.replace(tracking_tmp, tracking_output)
            published.append(tracking_output)
            os.replace(master_tmp, master_output)
            published.append(master_output)
        except (OSError, ValueError) as exc:
            # Roll back: drop any newly-published file, then restore the backups.
            for path in published:
                path.unlink(missing_ok=True)
            for backup, original in backed_up:
                os.replace(backup, original)
            master_tmp.unlink(missing_ok=True)
            tracking_tmp.unlink(missing_ok=True)
            raise SeamDerivationError(f"seam parquet derivation failed: {exc}") from exc

        # Commit: drop the backups now that both replacements succeeded.
        for backup, _ in backed_up:
            backup.unlink(missing_ok=True)

    return SeamDerivationResult(
        master_path=master_output,
        tracking_path=tracking_output,
        row_count=len(rows),
    )
