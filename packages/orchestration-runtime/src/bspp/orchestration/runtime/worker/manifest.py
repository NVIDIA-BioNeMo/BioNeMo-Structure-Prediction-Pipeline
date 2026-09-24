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

"""Manifest filtering and shard-manifest persistence for the native worker.

Heterodimer ID rewrite helpers only transform manifest rows. The side-effect
layer must also re-point the corresponding ``work_input_dir`` symlinks before
pipeline execution can consume unified AF IDs.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

ManifestMode = Literal["homodimer", "heterodimer"]
ManifestRow = Mapping[str, object]

_MANIFEST_MODEL_ID = "model_entity_id"
_ENTITY_ID = "entity_id"
_CHAIN_ID = "chain_id"
_UNIPROT_AC = "uniprot_ac"
_LEGACY_COMPOUND_MODEL_ID_RE = re.compile(r"^(AF_\d+)_AF_(\d+)$")


@dataclass(frozen=True, slots=True)
class FilteredManifest:
    """A filtered shard manifest held in memory until an explicit write step."""

    fieldnames: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    matched_model_ids: tuple[str, ...]

    @property
    def row_count(self) -> int:
        """Number of manifest rows selected for this shard."""

        return len(self.rows)

    @property
    def matched_model_count(self) -> int:
        """Number of distinct requested model IDs represented by selected rows."""

        return len(self.matched_model_ids)


@dataclass(frozen=True, slots=True)
class HeterodimerIdRewritePlan:
    """Pure plan for replacing compound IDs with unified heterodimer AF-IDs."""

    rename_pairs: tuple[tuple[str, str], ...]
    swapped_model_ids: tuple[str, ...]

    @property
    def rename_map(self) -> dict[str, str]:
        """Return the planned compound-to-unified ID map."""

        return dict(self.rename_pairs)


@dataclass(frozen=True, slots=True)
class ShardManifestPersistencePlan:
    """Explicit parquet write plan for a shard manifest."""

    manifest: FilteredManifest
    output_path: Path
    shard_id: int
    dataset_tag: str


@dataclass(frozen=True, slots=True)
class ShardManifestWriteResult:
    """Accounting for an executed shard manifest parquet write."""

    output_path: Path
    row_count: int


@dataclass(frozen=True, slots=True)
class _ComponentRef:
    compound_id: str
    entity_id: int
    chain_id: str


def filter_manifest_csv(
    manifest_csv: Path,
    model_ids: Iterable[str],
    *,
    mode: ManifestMode = "homodimer",
) -> FilteredManifest:
    """Read and filter a manifest CSV without writing intermediate files."""

    with manifest_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            msg = f"Manifest CSV has no header: {manifest_csv}"
            raise ValueError(msg)

        fieldnames = tuple(reader.fieldnames)
        if mode == "homodimer":
            return filter_homodimer_manifest_rows(reader, model_ids, fieldnames=fieldnames)
        if mode == "heterodimer":
            return filter_heterodimer_manifest_rows(reader, model_ids, fieldnames=fieldnames)

    msg = f"Unsupported manifest mode: {mode!r}"
    raise ValueError(msg)


def filter_homodimer_manifest_rows(
    rows: Iterable[ManifestRow],
    model_ids: Iterable[str],
    *,
    fieldnames: Iterable[str],
) -> FilteredManifest:
    """Filter manifest rows by exact homodimer model ID membership."""

    fieldnames_tuple = _validate_fieldnames(fieldnames, required=(_MANIFEST_MODEL_ID,))
    model_id_set = {model_id for model_id in model_ids if model_id}
    filtered_rows: list[dict[str, str]] = []
    matched_model_ids: list[str] = []
    seen_matched_model_ids: set[str] = set()

    for row in rows:
        manifest_model_id = _value(row, _MANIFEST_MODEL_ID)
        if manifest_model_id not in model_id_set:
            continue

        filtered_rows.append(_project_row(row, fieldnames_tuple))
        if manifest_model_id not in seen_matched_model_ids:
            seen_matched_model_ids.add(manifest_model_id)
            matched_model_ids.append(manifest_model_id)

    return FilteredManifest(
        fieldnames=fieldnames_tuple,
        rows=tuple(filtered_rows),
        matched_model_ids=tuple(matched_model_ids),
    )


def filter_heterodimer_manifest_rows(
    rows: Iterable[ManifestRow],
    model_ids: Iterable[str],
    *,
    fieldnames: Iterable[str],
) -> FilteredManifest:
    """Build a compound heterodimer shard manifest from component rows.

    Compound model IDs such as ``AF_1001_AF_1002`` are expanded to component
    IDs ``AF-1001`` and ``AF-1002`` for lookup in the source manifest. Matching
    rows are emitted with the compound ID restored, ``entity_id`` set to
    ``1``/``2``, and ``chain_id`` set to ``A``/``B``. Only the first matching
    row per compound chain is kept, matching legacy WP8a behavior.
    """

    fieldnames_tuple = _validate_fieldnames(
        fieldnames,
        required=(_MANIFEST_MODEL_ID, _ENTITY_ID, _CHAIN_ID, _UNIPROT_AC),
    )
    component_lookup = _build_component_lookup(model_ids)
    filtered_rows: list[dict[str, str]] = []
    matched_model_ids: list[str] = []
    seen_matched_model_ids: set[str] = set()
    written: set[tuple[str, int]] = set()

    for row in rows:
        component_id = _value(row, _MANIFEST_MODEL_ID)
        if component_id not in component_lookup or not _value(row, _UNIPROT_AC):
            continue

        for ref in component_lookup[component_id]:
            key = (ref.compound_id, ref.entity_id)
            if key in written:
                continue

            out_row = _project_row(row, fieldnames_tuple)
            out_row[_MANIFEST_MODEL_ID] = ref.compound_id
            out_row[_ENTITY_ID] = str(ref.entity_id)
            out_row[_CHAIN_ID] = ref.chain_id
            filtered_rows.append(out_row)
            written.add(key)

            if ref.compound_id not in seen_matched_model_ids:
                seen_matched_model_ids.add(ref.compound_id)
                matched_model_ids.append(ref.compound_id)

    return FilteredManifest(
        fieldnames=fieldnames_tuple,
        rows=tuple(filtered_rows),
        matched_model_ids=tuple(matched_model_ids),
    )


def plan_heterodimer_id_rewrites(
    shard_manifest: FilteredManifest,
    heterodimer_id_rows: Iterable[ManifestRow],
) -> HeterodimerIdRewritePlan:
    """Plan compound-to-unified heterodimer ID rewrites by UniProt AC pairs."""

    compound_to_acs = _pivot_entity_uniprot_pairs(shard_manifest.rows)
    unified_id_by_acs = _canonical_unified_id_by_uniprot_pair(heterodimer_id_rows)

    rename_pairs: list[tuple[str, str]] = []
    swapped_model_ids: list[str] = []
    for compound_id, pair in compound_to_acs.items():
        if not pair[0] or not pair[1]:
            continue
        if unified_id := unified_id_by_acs.get(pair):
            rename_pairs.append((compound_id, unified_id))
            continue
        reversed_pair = (pair[1], pair[0])
        if unified_id := unified_id_by_acs.get(reversed_pair):
            rename_pairs.append((compound_id, unified_id))
            swapped_model_ids.append(compound_id)

    return HeterodimerIdRewritePlan(
        rename_pairs=tuple(rename_pairs),
        swapped_model_ids=tuple(swapped_model_ids),
    )


def _canonical_unified_id_by_uniprot_pair(rows: Iterable[ManifestRow]) -> dict[tuple[str, str], str]:
    """Map UniProt AC pairs to canonical unified IDs with a stable tie-breaker.

    The rescue manifest can contain multiple unified AF IDs for the same ordered
    UniProt pair. The legacy path now orders by ``model_entity_id`` before dict
    assignment, so duplicates resolve to the last ordered ID. Mirror that rule
    here: the same pair always resolves to the lexicographically greatest
    unified ID, independent of CSV or DB row order.
    """

    unified_id_by_acs: dict[tuple[str, str], str] = {}
    for model_id, pair in sorted(_pivot_entity_uniprot_pairs(rows).items()):
        if pair[0] and pair[1]:
            unified_id_by_acs[pair] = model_id
    return unified_id_by_acs


def apply_heterodimer_id_rewrites(
    manifest: FilteredManifest,
    rewrite_plan: HeterodimerIdRewritePlan,
) -> FilteredManifest:
    """Return a new manifest with planned heterodimer ID and chain swaps applied."""

    rename_map = rewrite_plan.rename_map
    swapped = set(rewrite_plan.swapped_model_ids)
    rewritten_rows: list[dict[str, str]] = []
    matched_model_ids: list[str] = []
    seen_matched_model_ids: set[str] = set()
    seen_chain_keys: set[tuple[str, str, str]] = set()

    for row in manifest.rows:
        rewritten = dict(row)
        old_model_id = rewritten.get(_MANIFEST_MODEL_ID, "")
        rewritten[_MANIFEST_MODEL_ID] = rename_map.get(old_model_id, old_model_id)
        if old_model_id in swapped:
            rewritten[_ENTITY_ID] = _swap_entity_id(rewritten.get(_ENTITY_ID, ""))
            rewritten[_CHAIN_ID] = _swap_chain_id(rewritten.get(_CHAIN_ID, ""))

        # Match legacy _rewrite_manifest_ids: two compound IDs (forward + swapped)
        # can rename to the same unified ID; after the swap-flip their rows
        # coincide, so dedup by (model_entity_id, entity_id, chain_id) to keep
        # one row per physical chain. Without this the converter sees each chain
        # twice and can bind the wrong chain order.
        chain_key = (
            rewritten[_MANIFEST_MODEL_ID],
            rewritten.get(_ENTITY_ID, ""),
            rewritten.get(_CHAIN_ID, ""),
        )
        if chain_key in seen_chain_keys:
            continue
        seen_chain_keys.add(chain_key)
        rewritten_rows.append(rewritten)

        new_model_id = rewritten[_MANIFEST_MODEL_ID]
        if new_model_id not in seen_matched_model_ids:
            seen_matched_model_ids.add(new_model_id)
            matched_model_ids.append(new_model_id)

    return FilteredManifest(
        fieldnames=manifest.fieldnames,
        rows=tuple(rewritten_rows),
        matched_model_ids=tuple(matched_model_ids),
    )


def read_heterodimer_id_rewrite_plan(
    shard_manifest: FilteredManifest,
    heterodimer_id_manifest: Path,
) -> HeterodimerIdRewritePlan:
    """Read a heterodimer ID manifest and plan in-memory shard manifest rewrites."""

    with heterodimer_id_manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            msg = f"Manifest CSV has no header: {heterodimer_id_manifest}"
            raise ValueError(msg)
        return plan_heterodimer_id_rewrites(shard_manifest, reader)


def plan_shard_manifest_persistence(
    manifest: FilteredManifest,
    *,
    shard_id: int,
    dataset_tag: str,
    output_path: Path,
) -> ShardManifestPersistencePlan:
    """Create a parquet persistence plan without touching the filesystem."""

    return ShardManifestPersistencePlan(
        manifest=manifest,
        output_path=output_path,
        shard_id=shard_id,
        dataset_tag=dataset_tag,
    )


def write_shard_manifest_parquet(plan: ShardManifestPersistencePlan) -> ShardManifestWriteResult:
    """Write a planned shard manifest parquet with legacy metadata columns."""

    import pyarrow as pa
    import pyarrow.csv as pa_csv
    import pyarrow.parquet as pq

    if plan.manifest.rows:
        read_csv = cast(Any, vars(pa_csv)["read_csv"])
        table = read_csv(pa.BufferReader(_manifest_to_csv_bytes(plan.manifest)))
    else:
        table = pa.table({fieldname: pa.array([], pa.string()) for fieldname in plan.manifest.fieldnames})

    if table.num_rows:
        table = table.group_by(table.column_names).aggregate([])

    n = table.num_rows
    table = table.append_column("shard_id", pa.array([plan.shard_id] * n, pa.int32()))
    table = table.append_column("dataset_tag", pa.array([plan.dataset_tag] * n, pa.string()))

    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_table = cast(Any, vars(pq)["write_table"])
    write_table(table, plan.output_path, compression="snappy")
    return ShardManifestWriteResult(output_path=plan.output_path, row_count=n)


def write_shard_manifest_csv(manifest: FilteredManifest, output_path: Path) -> ShardManifestWriteResult:
    """Write the legacy intermediate ``shard_manifest.csv`` compatibility file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_manifest_to_csv_bytes(manifest))
    return ShardManifestWriteResult(output_path=output_path, row_count=manifest.row_count)


def persist_shard_manifest_parquet(
    manifest: FilteredManifest,
    *,
    shard_id: int,
    dataset_tag: str,
    output_path: Path,
) -> ShardManifestWriteResult:
    """Plan and write a shard manifest parquet in one explicit persistence step."""

    return write_shard_manifest_parquet(
        plan_shard_manifest_persistence(
            manifest,
            shard_id=shard_id,
            dataset_tag=dataset_tag,
            output_path=output_path,
        ),
    )


def _build_component_lookup(model_ids: Iterable[str]) -> dict[str, tuple[_ComponentRef, ...]]:
    component_lookup: dict[str, list[_ComponentRef]] = {}
    seen: set[tuple[str, int]] = set()
    for model_id in model_ids:
        if not model_id:
            continue
        components = _parse_legacy_compound_model_id(model_id)
        if components is None:
            continue

        for component_id, entity_id, chain_id in (
            (components[0], 1, "A"),
            (components[1], 2, "B"),
        ):
            key = (model_id, entity_id)
            if key in seen:
                continue
            seen.add(key)
            component_lookup.setdefault(component_id, []).append(
                _ComponentRef(compound_id=model_id, entity_id=entity_id, chain_id=chain_id),
            )

    return {component_id: tuple(refs) for component_id, refs in component_lookup.items()}


def _parse_legacy_compound_model_id(model_id: str) -> tuple[str, str] | None:
    match = _LEGACY_COMPOUND_MODEL_ID_RE.fullmatch(model_id)
    if match is None:
        return None
    return match.group(1).replace("_", "-", 1), f"AF-{match.group(2)}"


def _pivot_entity_uniprot_pairs(rows: Iterable[ManifestRow]) -> dict[str, tuple[str, str]]:
    pivot: dict[str, dict[int, str]] = {}
    for row in rows:
        model_id = _value(row, _MANIFEST_MODEL_ID)
        entity_id = _entity_id(row)
        if not model_id or entity_id not in {1, 2}:
            continue
        pivot.setdefault(model_id, {})[entity_id] = _value(row, _UNIPROT_AC)

    return {model_id: (entities.get(1, ""), entities.get(2, "")) for model_id, entities in pivot.items()}


def _entity_id(row: ManifestRow) -> int | None:
    try:
        return int(_value(row, _ENTITY_ID))
    except ValueError:
        return None


def _manifest_to_csv_bytes(manifest: FilteredManifest) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=manifest.fieldnames)
    writer.writeheader()
    writer.writerows(manifest.rows)
    return output.getvalue().encode()


def _validate_fieldnames(fieldnames: Iterable[str], *, required: Iterable[str]) -> tuple[str, ...]:
    fieldnames_tuple = tuple(fieldnames)
    missing = tuple(fieldname for fieldname in required if fieldname not in fieldnames_tuple)
    if missing:
        msg = f"Manifest is missing required columns: {', '.join(missing)}"
        raise ValueError(msg)
    return fieldnames_tuple


def _project_row(row: ManifestRow, fieldnames: Iterable[str]) -> dict[str, str]:
    return {fieldname: _value(row, fieldname) for fieldname in fieldnames}


def _value(row: ManifestRow, fieldname: str) -> str:
    value = row.get(fieldname, "")
    if value is None:
        return ""
    return str(value)


def _swap_entity_id(entity_id: str) -> str:
    return {"1": "2", "2": "1"}.get(entity_id, entity_id)


def _swap_chain_id(chain_id: str) -> str:
    return {"A": "B", "B": "A"}.get(chain_id, chain_id)


__all__ = [
    "FilteredManifest",
    "HeterodimerIdRewritePlan",
    "ManifestMode",
    "ShardManifestPersistencePlan",
    "ShardManifestWriteResult",
    "apply_heterodimer_id_rewrites",
    "filter_heterodimer_manifest_rows",
    "filter_homodimer_manifest_rows",
    "filter_manifest_csv",
    "persist_shard_manifest_parquet",
    "plan_heterodimer_id_rewrites",
    "plan_shard_manifest_persistence",
    "read_heterodimer_id_rewrite_plan",
    "write_shard_manifest_csv",
    "write_shard_manifest_parquet",
]
