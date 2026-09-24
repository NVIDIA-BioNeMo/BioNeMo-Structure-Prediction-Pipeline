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

"""Focused control-side value objects for the folding Phase adapter.

These records are deliberately control-pure: they import only from
``bspp.orchestration.contract.*`` and never from ``bspp.orchestration.runtime.*``.
The canonical-pair index surface is a schema-compatible reimplementation of the
delivered Track C module (``runtime/folding/benchmark/index.py``) so the adapter
can build and write the exact ``schema_version=1`` index JSON without importing
the runtime distribution.

The five folding action-evidence records plus ``CanonicalPairEvidenceEntry`` now
live in ``bspp.orchestration.contract.folding_evidence`` and are
re-exported here with identical object identity so existing import sites keep
working. The canonical-pair index surface remains Control-side.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.folding_evidence import (
    CanonicalPairActionEvidence,
    CanonicalPairEvidenceEntry,
    FoldActionEvidence,
    MsaFlattenActionEvidence,
    PreprocessActionEvidence,
    SplitActionEvidence,
)
from bspp.orchestration.contract.folding_execution import FoldingBackendAssetsSnapshot
from bspp.orchestration.contract.folding_release import FoldingReleasePreset
from bspp.orchestration.contract.phase import (
    FOLDING_BACKENDS,
    PhaseMountSnapshot,
    PhaseSlurmResources,
    TransportKind,
    canonical_mapping_digest,
)
from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SEQUENCE_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FoldingBackendImageSelection:
    """Per-backend kernel image selection (MFI-008)."""

    backend_images: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.backend_images, Mapping) or not self.backend_images:
            raise ValueError("folding backend image selection must be a non-empty mapping")

    def image_for_backend(self, backend: str) -> str:
        """Return the exact kernel image for one backend, failing closed."""
        try:
            image = self.backend_images[backend]
        except KeyError as exc:
            raise ValueError(f"no kernel image selected for folding backend {backend!r}") from exc
        if not isinstance(image, str) or not image:
            raise ValueError(f"kernel image for folding backend {backend!r} must be non-empty")
        return image


@dataclass(frozen=True)
class FoldingPhaseAttemptOperationalSelection:
    """Curated operational selections for one folding Phase Attempt.

    This is the folding analogue of the preprocessing
    ``PhaseAttemptOperationalSelection``; it carries the resolved Cluster
    Profile selections directly (the folding Cluster Profile has no
    preprocessing runtime qualification) plus the backend kernel images and the
    per-action/fold Slurm resources.
    """

    profile_name: str
    owner: str
    transport: TransportKind
    ssh_target: str | None
    account: str
    project_root: str
    staging_root: str
    orchestration_repo: str
    runtime_image: str
    backend_images: FoldingBackendImageSelection
    release_preset: FoldingReleasePreset
    resources: PhaseSlurmResources
    fold_resources: PhaseSlurmResources | None = None
    extra_mounts: tuple[PhaseMountSnapshot, ...] = ()
    assets: FoldingBackendAssetsSnapshot | None = None
    mount_orchestration_source: bool = False

    def __post_init__(self) -> None:
        required = (
            self.profile_name,
            self.owner,
            self.account,
            self.project_root,
            self.staging_root,
            self.orchestration_repo,
            self.runtime_image,
        )
        if any(not value for value in required):
            raise ValueError("folding operational selection required fields must be non-empty")
        if self.transport not in {"ssh", "local-slurm"}:
            raise ValueError(f"unsupported folding transport: {self.transport!r}")
        if (self.transport == "ssh") != (self.ssh_target is not None):
            raise ValueError("folding operational selection ssh_target must match its transport")
        if self.ssh_target == "":
            raise ValueError("folding operational selection ssh_target must be non-empty when present")
        if not isinstance(self.backend_images, FoldingBackendImageSelection):
            raise ValueError("folding operational selection backend_images must be a FoldingBackendImageSelection")
        if not isinstance(self.release_preset, FoldingReleasePreset):
            raise ValueError("folding operational selection release_preset must be a FoldingReleasePreset")
        if not isinstance(self.resources, PhaseSlurmResources):
            raise ValueError("folding operational selection resources must be a PhaseSlurmResources")
        if self.fold_resources is not None and not isinstance(self.fold_resources, PhaseSlurmResources):
            raise ValueError("folding operational selection fold_resources must be a PhaseSlurmResources")
        if not isinstance(self.extra_mounts, tuple) or any(
            not isinstance(mount, PhaseMountSnapshot) for mount in self.extra_mounts
        ):
            raise ValueError("folding operational selection extra_mounts must be a tuple of PhaseMountSnapshot")
        if self.assets is not None and not isinstance(self.assets, FoldingBackendAssetsSnapshot):
            raise ValueError("folding operational selection assets must be a FoldingBackendAssetsSnapshot")


def _validate_nonempty_str(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_sequence_sha256(value: object) -> None:
    if not isinstance(value, str) or _SEQUENCE_SHA256_RE.fullmatch(value) is None:
        raise ValueError("sequence_sha256 must be 64 lowercase hex characters")


def _reject_duplicate_target_ids(entries: tuple[CanonicalPairIndexEntry, ...]) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.target_id in seen:
            raise ValueError(f"Duplicate target_id {entry.target_id!r} in CanonicalPairIndex")
        seen.add(entry.target_id)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


@dataclass(frozen=True)
class CanonicalPairIndexEntry:
    """One canonical structure/scores pair keyed by target identity."""

    target_id: str
    sequence_sha256: str
    model_entity_id: str
    tool_used: str
    structure_path: str
    scores_path: str

    def __post_init__(self) -> None:
        _validate_nonempty_str(self.target_id, "target_id")
        _validate_sequence_sha256(self.sequence_sha256)
        _validate_nonempty_str(self.model_entity_id, "model_entity_id")
        if self.tool_used not in VALID_TOOL_USED:
            raise ValueError(f"tool_used must be one of {VALID_TOOL_USED!r}; got {self.tool_used!r}")
        _validate_nonempty_str(self.structure_path, "structure_path")
        _validate_nonempty_str(self.scores_path, "scores_path")

    def to_mapping(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "sequence_sha256": self.sequence_sha256,
            "model_entity_id": self.model_entity_id,
            "tool_used": self.tool_used,
            "structure_path": self.structure_path,
            "scores_path": self.scores_path,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class CanonicalPairIndex:
    """An immutable, schema-versioned mapping of canonical prediction pairs."""

    schema_version: int
    run_id: str
    entries: tuple[CanonicalPairIndexEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported canonical-pair index schema_version {self.schema_version!r}; "
                f"supported version: {_INDEX_SCHEMA_VERSION}"
            )
        _validate_nonempty_str(self.run_id, "run_id")
        if not isinstance(self.entries, tuple) or not all(
            isinstance(entry, CanonicalPairIndexEntry) for entry in self.entries
        ):
            raise ValueError("entries must be an immutable tuple of CanonicalPairIndexEntry values")
        _reject_duplicate_target_ids(self.entries)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "entries": [entry.to_mapping() for entry in self.entries],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


def _entry_from_mapping(payload: Mapping[str, object]) -> CanonicalPairIndexEntry:
    return CanonicalPairIndexEntry(
        target_id=_required_str(payload, "target_id"),
        sequence_sha256=_required_str(payload, "sequence_sha256"),
        model_entity_id=_required_str(payload, "model_entity_id"),
        tool_used=_required_str(payload, "tool_used"),
        structure_path=_required_str(payload, "structure_path"),
        scores_path=_required_str(payload, "scores_path"),
    )


def load_canonical_pair_index(path: Path) -> CanonicalPairIndex:
    """Load and strictly validate a canonical-pair index JSON file.

    Fails closed (``ValueError``) on an absent/unreadable file, malformed JSON,
    a non-object top level, an unknown/missing schema version, duplicate
    target_ids, or any invalid entry field. Never opens the pair files.
    """
    if not isinstance(path, Path):
        raise ValueError("path must be a pathlib.Path")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read canonical-pair index {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in canonical-pair index {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("canonical-pair index must be a JSON object")
    schema_version = _required_int(payload, "schema_version")
    run_id = _required_str(payload, "run_id")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("entries must be a list")
    entries: list[CanonicalPairIndexEntry] = []
    for position, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            raise ValueError(f"entries[{position}] must be an object")
        entries.append(_entry_from_mapping(item))
    return CanonicalPairIndex(schema_version=schema_version, run_id=run_id, entries=tuple(entries))


def build_canonical_pair_index(
    *,
    run_id: str,
    entries: Iterable[tuple[str, str, str, str, str, str]],
) -> CanonicalPairIndex:
    """Build a canonical-pair index, ordering entries by target_id.

    Duplicate target_ids are rejected exactly like the loader.
    """
    records = tuple(CanonicalPairIndexEntry(*entry) for entry in entries)
    ordered = tuple(sorted(records, key=lambda record: record.target_id))
    return CanonicalPairIndex(schema_version=_INDEX_SCHEMA_VERSION, run_id=run_id, entries=ordered)


def write_canonical_pair_index(index: CanonicalPairIndex, path: Path) -> None:
    """Write a canonical-pair index as deterministic JSON via atomic replace."""
    if not isinstance(index, CanonicalPairIndex):
        raise ValueError("index must be a CanonicalPairIndex")
    if not isinstance(path, Path):
        raise ValueError("path must be a pathlib.Path")
    text = index.to_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def folding_qualification_tuple_id(*, backend: str, kernel_image: str, cluster_snapshot_digest: str) -> str:
    """Return the deterministic folding operational-selection tuple id.

    Folding has no preprocessing runtime qualification; the tuple
    binds the backend, its selected kernel image, and the resolved cluster
    snapshot digest.
    """
    if backend not in FOLDING_BACKENDS:
        raise ValueError(f"unsupported folding backend: {backend!r}")
    if not kernel_image:
        raise ValueError("folding kernel image must be non-empty")
    if _SHA256.fullmatch(cluster_snapshot_digest) is None:
        raise ValueError("folding cluster snapshot digest must be a lowercase SHA-256")
    return canonical_mapping_digest(
        {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "backend": backend,
            "kernel_image": kernel_image,
            "cluster_snapshot_digest": cluster_snapshot_digest,
        }
    )


__all__ = [
    "CanonicalPairActionEvidence",
    "CanonicalPairEvidenceEntry",
    "CanonicalPairIndex",
    "CanonicalPairIndexEntry",
    "FoldActionEvidence",
    "FoldingBackendImageSelection",
    "FoldingPhaseAttemptOperationalSelection",
    "MsaFlattenActionEvidence",
    "PreprocessActionEvidence",
    "SplitActionEvidence",
    "build_canonical_pair_index",
    "folding_qualification_tuple_id",
    "load_canonical_pair_index",
    "write_canonical_pair_index",
]
