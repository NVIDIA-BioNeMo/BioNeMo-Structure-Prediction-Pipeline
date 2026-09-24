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

"""Pure model discovery and archive allowlist filtering."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from os import PathLike, fspath

from bspp.orchestration.contract.model_identity import is_human_string_model_entity_id, normalize_model_entity_id
from bspp.orchestration.runtime.constants import KNOWN_SUFFIXES

type ArchiveMemberPath = str | PathLike[str]
type ReassignedModelIdRow = tuple[str, str] | str

_HOMODIMER_MODEL_ID_RE = re.compile(r"AF-\d{16}")
_COMPOUND_MODEL_ID_RE = re.compile(r"AF_[A-Za-z0-9]+_AF_[A-Za-z0-9]+")
_ARCHIVE_COMPOUND_MODEL_ID_RE = re.compile(r"AF[_-][A-Za-z0-9]+[_-]AF[_-][A-Za-z0-9]+")
_PDB_ASSEMBLY_MODEL_ID_RE = re.compile(r"pdb_[a-z0-9]+_assembly_\d+")
_AF_ID_GROUP = re.compile(r"AF[_-](\d+)")
_AFDB_PREFIX = "AFDB_"
_AF_UNDERSCORE_PREFIX = "AF_"
_AF_HYPHEN_PREFIX = "AF-"


@dataclass(frozen=True, slots=True)
class ModelAllowlistSelection:
    """Archive-ordered allowlist selection with explicit informational accounting.

    ``missing_allowlist_model_ids`` reports allowlist IDs that were not present
    in the archive inventory. Missing allowlist IDs do not change the kept IDs.
    """

    kept_model_ids: tuple[str, ...]
    dropped_model_ids: tuple[str, ...]
    missing_allowlist_model_ids: tuple[str, ...]


def parse_compound_model_id(model_id: str) -> tuple[str, str] | None:
    """Return manifest-compatible component IDs for a heterodimer compound ID."""

    _raise_for_empty_model_id(model_id)
    if _COMPOUND_MODEL_ID_RE.fullmatch(model_id) is None:
        return None

    first, second_suffix = model_id.split("_AF_", maxsplit=1)
    return _hyphenated_component_id(first), _hyphenated_component_id(f"AF_{second_suffix}")


def is_compound_model_id(model_id: str) -> bool:
    """Return whether ``model_id`` is a legacy heterodimer compound ID."""

    return parse_compound_model_id(model_id) is not None


def is_model_id(model_id: str) -> bool:
    """Return whether ``model_id`` is a model root recognized by WP8b."""

    _raise_for_empty_model_id(model_id)
    return _is_model_id_candidate(model_id)


def canonical_link_model_id(model_id_or_filename: str) -> str:
    """Return a legacy link-compatible ID from a model ID or flat filename.

    The function is pure string normalization: it strips a known flat-file
    suffix, strips one leading ``AFDB_`` batch prefix, and normalizes one leading
    ``AF_`` to ``AF-``. The returned link ID is not guaranteed to be an
    inventory model ID; for example, compound link IDs intentionally differ from
    compound archive model IDs.
    """

    _raise_for_empty_model_id(model_id_or_filename)
    model_id = _model_id_root_from_flat_filename(model_id_or_filename) or model_id_or_filename
    model_id = _strip_afdb_prefix(model_id)
    return _normalize_leading_af_underscore(model_id)


def discover_model_ids(archive_members: Iterable[ArchiveMemberPath]) -> tuple[str, ...]:
    """Discover unique top-level model IDs from archive member names.

    The function only inspects member strings. Non-model archive members are
    ignored, and model IDs are returned exactly as they appear in the archive.
    HumanSTRING roots fail explicitly: their postprocessing mapping is unsupported.
    """

    seen: set[str] = set()
    model_ids: list[str] = []
    for member in archive_members:
        _reject_human_string_member(member)
        model_id = _model_id_from_archive_member(member)
        if model_id is None or model_id in seen:
            continue

        seen.add(model_id)
        model_ids.append(model_id)

    return tuple(model_ids)


def filter_model_ids_by_allowlist(
    model_ids: Iterable[str],
    allowlist_model_ids: Iterable[str],
) -> ModelAllowlistSelection:
    """Filter archive-ordered model IDs by an exact allowlist.

    Missing allowlist IDs are reported for diagnostics only. They do not alter
    the archive-ordered kept IDs.
    """

    archive_ordered_model_ids = _unique_valid_model_ids(model_ids, field_name="model_ids")
    allowlist_ordered_model_ids = _unique_valid_model_ids(allowlist_model_ids, field_name="allowlist_model_ids")
    allowed = set(allowlist_ordered_model_ids)
    archive_ids = set(archive_ordered_model_ids)

    kept_model_ids = tuple(model_id for model_id in archive_ordered_model_ids if model_id in allowed)
    dropped_model_ids = tuple(model_id for model_id in archive_ordered_model_ids if model_id not in allowed)
    missing_allowlist_model_ids = tuple(
        model_id for model_id in allowlist_ordered_model_ids if model_id not in archive_ids
    )

    return ModelAllowlistSelection(
        kept_model_ids=kept_model_ids,
        dropped_model_ids=dropped_model_ids,
        missing_allowlist_model_ids=missing_allowlist_model_ids,
    )


def filter_archive_members_by_allowlist(
    archive_members: Iterable[ArchiveMemberPath],
    allowlist_model_ids: Iterable[str],
) -> ModelAllowlistSelection:
    """Discover archive model IDs and filter them by an exact allowlist.

    Missing allowlist IDs are informational only and never add to or remove from
    the kept IDs beyond the exact allowlist membership test.
    """

    return filter_model_ids_by_allowlist(discover_model_ids(archive_members), allowlist_model_ids)


def plan_reassigned_model_id_mapping(rows: Iterable[ReassignedModelIdRow]) -> dict[str, str]:
    """Build a pure old-to-new model ID mapping from parsed reassignment rows.

    Rows may be ``(old_id, new_id)`` tuples or strings containing two fields
    separated by a comma or ASCII whitespace. Empty string rows are ignored.
    File/path reads and file renames are intentionally outside this WP8b helper.
    """

    mapping: dict[str, str] = {}
    assigned_new_ids: dict[str, str] = {}
    for row in rows:
        parsed = _parse_reassigned_model_id_row(row)
        if parsed is None:
            continue

        old_id, new_id = parsed
        _raise_for_empty_model_id(old_id, field_name="old_id")
        _raise_for_empty_model_id(new_id, field_name="new_id")
        if old_id == new_id:
            continue

        existing = mapping.get(old_id)
        if existing is not None and existing != new_id:
            msg = f"conflicting reassigned model ID for {old_id}: {existing} != {new_id}"
            raise ValueError(msg)
        existing_old_id = assigned_new_ids.get(new_id)
        if existing_old_id is not None and existing_old_id != old_id:
            msg = f"duplicate reassigned new model ID {new_id}: {existing_old_id} and {old_id}"
            raise ValueError(msg)
        mapping[old_id] = new_id
        assigned_new_ids[new_id] = old_id

    return mapping


def rewrite_reassigned_model_ids(
    model_ids: Iterable[str],
    reassigned_model_ids: Mapping[str, str],
) -> tuple[str, ...]:
    """Rewrite model IDs with a reassignment mapping, preserving input order."""

    rewritten_model_ids: list[str] = []
    for model_id in model_ids:
        _raise_for_empty_model_id(model_id, field_name="model_ids")
        rewritten_model_ids.append(reassigned_model_ids.get(model_id, model_id))
    return tuple(rewritten_model_ids)


def _reject_human_string_member(member: ArchiveMemberPath) -> None:
    parts = [part for part in fspath(member).replace("\\", "/").split("/") if part and part != "."]
    if not parts:
        return
    candidate = parts[0]
    if len(parts) == 1:
        candidate = _model_id_root_from_flat_filename(candidate) or candidate
    # Bare top-level model directories and recognized flat files only. Do not
    # scan nested unrelated paths or broaden the frozen postprocessing grammar.
    if is_human_string_model_entity_id(candidate) and normalize_model_entity_id(candidate) == candidate:
        raise ValueError(f"HumanSTRING postprocessing is unsupported: {candidate}")


def _model_id_from_archive_member(member: ArchiveMemberPath) -> str | None:
    parts = [part for part in fspath(member).replace("\\", "/").split("/") if part not in {"", "."}]
    if not parts or parts[0] == "..":
        return None

    top_level = parts[0]
    if _is_archive_model_id_candidate(top_level):
        return top_level
    if len(parts) == 1:
        return _model_id_from_flat_filename(top_level)
    return None


def _model_id_from_flat_filename(filename: str) -> str | None:
    model_id = _model_id_root_from_flat_filename(filename)
    if model_id is None:
        return None

    model_id = _strip_afdb_prefix(model_id)
    af_groups = _AF_ID_GROUP.findall(model_id)
    if len(af_groups) == 1:
        return f"AF-{af_groups[0]}"
    if len(af_groups) > 1:
        return model_id if _is_archive_model_id_candidate(model_id) else None

    if _is_model_id_candidate(model_id):
        return model_id

    canonical_model_id = _normalize_leading_af_underscore(model_id)
    if _is_model_id_candidate(canonical_model_id):
        return canonical_model_id

    return None


def _model_id_root_from_flat_filename(filename: str) -> str | None:
    for suffix in KNOWN_SUFFIXES:
        if not filename.endswith(suffix):
            continue

        return filename[: -len(suffix)]
    return None


def _strip_afdb_prefix(model_id: str) -> str:
    if model_id.startswith(_AFDB_PREFIX):
        return model_id[len(_AFDB_PREFIX) :]
    return model_id


def _normalize_leading_af_underscore(model_id: str) -> str:
    if model_id.startswith(_AF_UNDERSCORE_PREFIX):
        return f"{_AF_HYPHEN_PREFIX}{model_id[len(_AF_UNDERSCORE_PREFIX) :]}"
    return model_id


def _hyphenated_component_id(component_id: str) -> str:
    return component_id.replace("_", "-", 1)


def _is_model_id_candidate(model_id: str) -> bool:
    return (
        _HOMODIMER_MODEL_ID_RE.fullmatch(model_id) is not None
        or _COMPOUND_MODEL_ID_RE.fullmatch(model_id) is not None
        or _PDB_ASSEMBLY_MODEL_ID_RE.fullmatch(model_id) is not None
    )


def _is_archive_model_id_candidate(model_id: str) -> bool:
    return (
        _HOMODIMER_MODEL_ID_RE.fullmatch(model_id) is not None
        or _ARCHIVE_COMPOUND_MODEL_ID_RE.fullmatch(model_id) is not None
        or _PDB_ASSEMBLY_MODEL_ID_RE.fullmatch(model_id) is not None
    )


def _unique_valid_model_ids(model_ids: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    seen: set[str] = set()
    unique_model_ids: list[str] = []
    for model_id in model_ids:
        _raise_for_empty_model_id(model_id, field_name=field_name)
        if model_id in seen:
            continue
        seen.add(model_id)
        unique_model_ids.append(model_id)
    return tuple(unique_model_ids)


def _parse_reassigned_model_id_row(row: ReassignedModelIdRow) -> tuple[str, str] | None:
    if isinstance(row, str):
        stripped_row = row.strip()
        if not stripped_row:
            return None
        parts = [part.strip() for part in stripped_row.split(",")] if "," in stripped_row else stripped_row.split()
        if len(parts) != 2:
            msg = f"reassigned model ID rows must contain exactly two fields: {row!r}"
            raise ValueError(msg)
        return parts[0], parts[1]

    if len(row) != 2:
        msg = f"reassigned model ID tuple rows must contain exactly two fields: {row!r}"
        raise ValueError(msg)
    return row


def _raise_for_empty_model_id(model_id: str, *, field_name: str = "model_id") -> None:
    if not model_id:
        msg = f"{field_name} cannot contain empty model IDs"
        raise ValueError(msg)


__all__ = [
    "ArchiveMemberPath",
    "ModelAllowlistSelection",
    "ReassignedModelIdRow",
    "canonical_link_model_id",
    "discover_model_ids",
    "filter_archive_members_by_allowlist",
    "filter_model_ids_by_allowlist",
    "is_compound_model_id",
    "is_model_id",
    "parse_compound_model_id",
    "plan_reassigned_model_id_mapping",
    "rewrite_reassigned_model_ids",
]
