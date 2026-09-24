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

"""Canonical fold-to-postprocessing model identity surface.

This module is the single source of truth for the harvested model naming and
identity policy shared by the folding and postprocessing phases.

Harvest sources:

- ``KNOWN_SUFFIXES`` and ``MAX_PROTEINS_PER_SHARD`` previously lived as
  duplicate definitions in
  ``bspp/orchestration/runtime/constants.py`` and are re-exported from there
  as direct name bindings.
- The model-ID grammar below deliberately mirrors the frozen postprocessing
  grammar in
  ``bspp/orchestration/runtime/worker/model_inventory.py``. That runtime
  module remains authoritative for existing worker behavior. The AF-grammar
  regexes (homodimer, compound, archive-compound) remain byte-identical
  between this contract copy and the runtime mirror. The pdb-assembly
  grammar (``pdb_[a-z0-9]+_assembly_\\d+``) was extended in this contract
  and brought into lockstep in the runtime mirror; the two now agree on that
  grammar. The contract and runtime
  intentionally differ in their compound-id regex surfaces: the runtime has
  both an underscore-only ``_COMPOUND_MODEL_ID_RE`` (for flat-member
  validation) and a hyphen-or-underscore ``_ARCHIVE_COMPOUND_MODEL_ID_RE``
  (for archive-member validation), while the contract has only the
  hyphen-or-underscore ``_ARCHIVE_COMPOUND_MODEL_ID_RE``. That compound-id
  surface difference is pre-existing and out of scope here. This
  contract copy exists so downstream consumers can validate identities
  without importing the runtime distribution.

HumanSTRING roots are an additional folding-only family. The frozen
postprocessing mirror rejects them explicitly at discovery; their component
and entity mapping is not supported by that pipeline.

Rationale:

- The identity policy is engine-neutral: the contract mirrors the
  frozen grammar without moving the frozen runtime behavior.
- Naming/identity constants are assigned to the contract package so the
  runtime and control planes share one canonical surface.
"""

from __future__ import annotations

import re

__all__ = [
    "CANONICAL_META_SUFFIX",
    "CANONICAL_MODEL_SUFFIX",
    "KNOWN_SUFFIXES",
    "MAX_PROTEINS_PER_SHARD",
    "is_compound_model_entity_id",
    "is_homodimer_model_entity_id",
    "is_human_string_model_entity_id",
    "is_pdb_assembly_model_entity_id",
    "normalize_model_entity_id",
    "parse_compound_model_entity_id",
    "parse_human_string_model_entity_id",
]

# The order is significant: the frozen runtime parser iterates this list in
# order when deriving a model root from a flat filename.
#
# READ-ONLY INVARIANT: this list is re-exported by object identity from the
# runtime constants module.  Mutating either binding
# (append/pop/slice-assign/reassign) silently corrupts both packages' view of
# the same object.  Treat KNOWN_SUFFIXES as frozen; never mutate or rebind it.
KNOWN_SUFFIXES: list[str] = [
    "-model_v1.pdb",
    "-meta_v1.json",
    ".merged_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb",
    ".merged_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
    "_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb",
    "_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
]

MAX_PROTEINS_PER_SHARD: int = 5000

CANONICAL_MODEL_SUFFIX = "-model_v1.pdb"
CANONICAL_META_SUFFIX = "-meta_v1.json"

_HOMODIMER_MODEL_ID_RE = re.compile(r"AF-\d{16}")
_ARCHIVE_COMPOUND_MODEL_ID_RE = re.compile(r"AF[_-][A-Za-z0-9]+[_-]AF[_-][A-Za-z0-9]+")
_COMPOUND_COMPONENTS_RE = re.compile(r"AF[_-]([A-Za-z0-9]+)[_-]AF[_-]([A-Za-z0-9]+)")
_PDB_ASSEMBLY_MODEL_ID_RE = re.compile(r"pdb_[a-z0-9]+_assembly_\d+")

# Corpus-compatible ASCII accession syntax; no lookup, isoforms or case folding.
_HUMAN_STRING_ACCESSION = r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
_HUMAN_STRING_HOMO_RE = re.compile(rf"homo_({_HUMAN_STRING_ACCESSION})")
_HUMAN_STRING_HETERO_RE = re.compile(rf"hetero_({_HUMAN_STRING_ACCESSION})_({_HUMAN_STRING_ACCESSION})")

_AFDB_PREFIX = "AFDB_"
_AF_UNDERSCORE_PREFIX = "AF_"
_AF_HYPHEN_PREFIX = "AF-"
_HARVESTED_MODEL_TOKEN = "_model"


def normalize_model_entity_id(value: str) -> str:
    """Return the canonical model entity ID for a model ID or flat filename.

    The function is pure string normalization: it strips a known flat-file
    suffix, strips the harvested ``_model`` token, strips one leading
    ``AFDB_`` batch prefix, and normalizes one leading ``AF_`` to ``AF-``.
    HumanSTRING roots preserve their exact identity and accept only known
    flat-file suffix stripping, never the AF-specific transforms above.
    The result is then validated against the accepted identity grammar; malformed
    values are rejected instead of being returned as arbitrary normalized text.
    """

    _raise_for_non_string_or_empty(value)
    if _PDB_ASSEMBLY_MODEL_ID_RE.fullmatch(value) is not None:
        return value
    model_id = _model_id_root_from_flat_filename(value) or value
    if _human_string_accessions(model_id) is not None:
        return model_id
    model_id = _strip_harvested_model_token(model_id)
    model_id = _strip_afdb_prefix(model_id)
    model_id = _normalize_leading_af_underscore(model_id)
    if not _is_model_identity(model_id):
        msg = f"invalid model entity ID: {value!r}"
        raise ValueError(msg)
    return model_id


def is_homodimer_model_entity_id(value: str) -> bool:
    """Return whether ``value`` normalizes to the exact 16-digit homodimer form."""

    _raise_for_non_string_or_empty(value)
    try:
        model_id = normalize_model_entity_id(value)
    except ValueError:
        return False
    return _HOMODIMER_MODEL_ID_RE.fullmatch(model_id) is not None


def is_compound_model_entity_id(value: str) -> bool:
    """Return whether ``value`` normalizes to an accepted compound form."""

    _raise_for_non_string_or_empty(value)
    try:
        model_id = normalize_model_entity_id(value)
    except ValueError:
        return False
    return _is_compound_identity(model_id)


def parse_compound_model_entity_id(value: str) -> tuple[str, str] | None:
    """Return hyphenated component IDs for an accepted compound identity.

    Recognized non-AF-compound identities return ``None``.
    Malformed values are rejected rather than silently treated as
    non-compounds.
    """

    _raise_for_non_string_or_empty(value)
    model_id = normalize_model_entity_id(value)
    if _PDB_ASSEMBLY_MODEL_ID_RE.fullmatch(model_id) is not None or _human_string_accessions(model_id) is not None:
        return None
    if _HOMODIMER_MODEL_ID_RE.fullmatch(model_id) is not None:
        return None

    match = _COMPOUND_COMPONENTS_RE.fullmatch(model_id)
    if match is None:
        msg = f"invalid compound model entity ID: {value!r}"
        raise ValueError(msg)
    return f"AF-{match.group(1)}", f"AF-{match.group(2)}"


def is_human_string_model_entity_id(value: str) -> bool:
    """Return whether a model ID or known flat filename has a HumanSTRING root."""
    _raise_for_non_string_or_empty(value)
    try:
        model_id = normalize_model_entity_id(value)
    except ValueError:
        return False
    return _human_string_accessions(model_id) is not None


def parse_human_string_model_entity_id(value: str) -> tuple[str, ...] | None:
    """Return one homo or two ordered hetero accessions; legacy IDs return None.

    Distinct hetero accessions need not have distinct sequences. This parser
    neither rewrites accession identity nor infers leaked-homodimer status.
    """
    return _human_string_accessions(normalize_model_entity_id(value))


def _human_string_accessions(model_id: str) -> tuple[str, ...] | None:
    homo = _HUMAN_STRING_HOMO_RE.fullmatch(model_id)
    if homo is not None:
        return (homo.group(1),)
    hetero = _HUMAN_STRING_HETERO_RE.fullmatch(model_id)
    if hetero is not None and hetero.group(1) != hetero.group(2):
        return (hetero.group(1), hetero.group(2))
    return None


def _is_model_identity(model_id: str) -> bool:
    return (
        _HOMODIMER_MODEL_ID_RE.fullmatch(model_id) is not None
        or _ARCHIVE_COMPOUND_MODEL_ID_RE.fullmatch(model_id) is not None
        or _PDB_ASSEMBLY_MODEL_ID_RE.fullmatch(model_id) is not None
    )


def _is_compound_identity(model_id: str) -> bool:
    return _COMPOUND_COMPONENTS_RE.fullmatch(model_id) is not None


def is_pdb_assembly_model_entity_id(value: str) -> bool:
    """Return whether ``value`` normalizes to a PDB assembly identity."""

    _raise_for_non_string_or_empty(value)
    try:
        model_id = normalize_model_entity_id(value)
    except ValueError:
        return False
    return _PDB_ASSEMBLY_MODEL_ID_RE.fullmatch(model_id) is not None


def _model_id_root_from_flat_filename(filename: str) -> str | None:
    for suffix in KNOWN_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return None


def _strip_harvested_model_token(model_id: str) -> str:
    if model_id.endswith(_HARVESTED_MODEL_TOKEN):
        return model_id[: -len(_HARVESTED_MODEL_TOKEN)]
    return model_id


def _strip_afdb_prefix(model_id: str) -> str:
    if model_id.startswith(_AFDB_PREFIX):
        return model_id[len(_AFDB_PREFIX) :]
    return model_id


def _normalize_leading_af_underscore(model_id: str) -> str:
    if model_id.startswith(_AF_UNDERSCORE_PREFIX):
        return f"{_AF_HYPHEN_PREFIX}{model_id[len(_AF_UNDERSCORE_PREFIX) :]}"
    return model_id


def _raise_for_non_string_or_empty(value: str) -> None:
    if not isinstance(value, str) or not value:
        msg = "model entity ID must be a non-empty string"
        raise ValueError(msg)
