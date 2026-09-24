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

"""Strict external chain-manifest reader and per-target leaked-homodimer classifier.

The external chain-manifest CSV is the single source of truth for
leaked-homodimer classification: a compound target is
"leaked" when two or more distinct physical chains all resolve to one UniProt
accession. This module reads that CSV strictly and classifies one target at a
time so that an ambiguous target fails closed without discarding unaffected
targets in the same batch.

The CSV schema is exactly four named columns:

    model_entity_id,entity_id,chain_id,uniprot_ac

``csv.DictReader`` alone would silently fill short rows, collect surplus
fields, and reorder nothing, so this module enforces the domain schema itself:

- the header must be exactly the four named columns in order (no missing,
  unknown, reordered, duplicate, or blank header names);
- every data row must carry exactly the four named fields: surplus values
  beyond the last column are rejected (``DictReader`` would otherwise collect
  them under the ``None`` key and silently discard them);
- every field in every row must be non-blank (short rows are rejected).

Duplicate or conflicting ``(model_entity_id, chain_id)`` evidence is NOT a
parse-time error: every nonblank, structurally valid row is retained
in input order, and ``classify_target`` marks only the affected
``model_entity_id`` as ``ambiguous``. Other targets in the same manifest remain
usable.

``parse_chain_manifest`` is the only production entry point and owns all
blank/short-row rejection. ``classify_target`` is deliberately self-contained
on already-parsed chain evidence: it never re-validates blank fields, so a
hand-built ``ChainManifest`` with an empty ``uniprot_ac`` can be classified
``not_leaked`` under the distinct-accessions branch rather than ``ambiguous``.
Callers that hand-build manifests are responsible for feeding them
already-validated rows.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .emitter_support import select_tool_used

__all__ = [
    "REQUIRED_COLUMNS",
    "ChainClassification",
    "ChainManifest",
    "ChainManifestRow",
    "ambiguous_chain_manifest_metadata",
    "classify_target",
    "parse_chain_manifest",
    "resolve_tool_used",
]

REQUIRED_COLUMNS: tuple[str, ...] = ("model_entity_id", "entity_id", "chain_id", "uniprot_ac")

ChainClassification = Literal["leaked", "not_leaked", "ambiguous"]


@dataclass(frozen=True)
class ChainManifestRow:
    """One strict chain-manifest row: a physical chain of a model entity."""

    model_entity_id: str
    entity_id: str
    chain_id: str
    uniprot_ac: str


@dataclass(frozen=True)
class ChainManifest:
    """The parsed, order-preserving contents of one chain-manifest CSV."""

    rows: tuple[ChainManifestRow, ...]


def ambiguous_chain_manifest_metadata() -> dict[str, object]:
    """Return a fresh fail-closed result marker for an ambiguous target.

    Both Track B emitters use this exact marker when
    ``resolve_tool_used(...)`` returns ``None``, so the two backends share one
    result vocabulary instead of duplicating literals. A function (rather than
    a module-level dict) guarantees every caller receives its own mutable copy.
    """

    return {"failed_closed": True, "failure": "ambiguous_chain_manifest"}


def parse_chain_manifest(path: Path) -> ChainManifest:
    """Read ``path`` as a strict chain-manifest CSV.

    Raises ``ValueError`` on a missing header row, a header that is not exactly
    ``REQUIRED_COLUMNS`` in order, a surplus field, or a blank field. Duplicate
    or conflicting ``(model_entity_id, chain_id)`` rows are retained in input
    order and become ``ambiguous`` for that target only when classified.
    """

    rows: list[ChainManifestRow] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("chain manifest is missing a header row")
        if tuple(fieldnames) != REQUIRED_COLUMNS:
            raise ValueError(f"chain manifest header must be exactly {REQUIRED_COLUMNS!r}; got {tuple(fieldnames)!r}")

        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"chain manifest line {line_number} has surplus fields")
            values: dict[str, str] = {}
            for column in REQUIRED_COLUMNS:
                raw = row.get(column)
                if raw is None or not raw.strip():
                    raise ValueError(f"chain manifest line {line_number} has a blank {column!r} field")
                values[column] = raw.strip()

            rows.append(
                ChainManifestRow(
                    model_entity_id=values["model_entity_id"],
                    entity_id=values["entity_id"],
                    chain_id=values["chain_id"],
                    uniprot_ac=values["uniprot_ac"],
                )
            )

    return ChainManifest(tuple(rows))


def classify_target(manifest: ChainManifest, model_entity_id: str) -> ChainClassification:
    """Classify one target from already-parsed chain evidence.

    Only rows whose ``model_entity_id`` matches the request are considered. The
    result is ``"leaked"`` when the target has two or more distinct chains that
    all share one ``uniprot_ac``, ``"not_leaked"`` for a single-chain target or
    distinct accessions, and ``"ambiguous"`` when there are no matching rows or
    any repeated ``chain_id``. This function never raises for a single target
    and never re-validates blank fields (``parse_chain_manifest`` owns that).
    """

    chains: dict[str, tuple[str, str]] = {}
    for row in manifest.rows:
        if row.model_entity_id != model_entity_id:
            continue
        if row.chain_id in chains:
            return "ambiguous"
        chains[row.chain_id] = (row.entity_id, row.uniprot_ac)

    if not chains:
        return "ambiguous"

    accessions = {uniprot_ac for _, uniprot_ac in chains.values()}
    if len(chains) >= 2 and len(accessions) == 1:
        return "leaked"
    return "not_leaked"


def resolve_tool_used(manifest: ChainManifest, model_entity_id: str, *, tool_used: str) -> str | None:
    """Resolve the ``tool_used`` provenance string for one target.

    Classification is resolved first: an ``"ambiguous"`` target returns ``None``
    immediately (fail the target closed), before any identity or tool-string
    work. Otherwise the single ``tool_used`` is passed to ``select_tool_used``
    for both the homodimer and heterodimer slots, because each Track B emitter
    folds with one backend. Malformed ``tool_used`` values and malformed model
    IDs propagate as ``ValueError`` from ``select_tool_used`` on the
    non-ambiguous path.
    """

    classification = classify_target(manifest, model_entity_id)
    if classification == "ambiguous":
        return None
    return select_tool_used(
        model_entity_id,
        leaked_homodimer=(classification == "leaked"),
        homodimer_tool_used=tool_used,
        heterodimer_tool_used=tool_used,
    )
