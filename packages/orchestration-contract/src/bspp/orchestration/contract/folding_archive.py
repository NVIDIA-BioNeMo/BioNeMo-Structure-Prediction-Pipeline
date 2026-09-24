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

"""Versioned folding-result and archive-plan records.

Port Baseline:
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/scripts/archive_openfold_results.py:201-228,313-398,463-600,728-744,945-982``.

These records describe local evidence and immutable plans only. They do not
copy, archive, compress, upload, or claim a hash for bytes not created here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

ResultStatus = Literal["missing", "complete", "pdb-only", "json-only", "duplicate", "identity-collision"]
ResultKind = Literal["pdb", "json"]
UnmatchedReason = Literal["malformed-name", "unplanned-identity"]


@dataclass(frozen=True)
class FoldingResultAssociation:
    """Observed result candidates associated with one planned model identity."""

    source_ordinal: int
    protein_id: str
    normalized_protein_id: str
    status: ResultStatus
    pdb_paths: tuple[str, ...]
    json_paths: tuple[str, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingResultAssociation")
        _nonnegative_int(self.source_ordinal, "source_ordinal")
        _nonempty_str(self.protein_id, "protein_id")
        _nonempty_str(self.normalized_protein_id, "normalized_protein_id")
        _string_tuple(self.pdb_paths, "pdb_paths")
        _string_tuple(self.json_paths, "json_paths")
        if self.status not in {"missing", "complete", "pdb-only", "json-only", "duplicate", "identity-collision"}:
            msg = f"unsupported result association status: {self.status!r}"
            raise ValueError(msg)
        counts = (len(self.pdb_paths), len(self.json_paths))
        expected = {
            "missing": (0, 0),
            "complete": (1, 1),
            "pdb-only": (1, 0),
            "json-only": (0, 1),
        }
        if self.status in expected and counts != expected[self.status]:
            msg = f"{self.status} association has inconsistent PDB/JSON cardinality"
            raise ValueError(msg)
        if self.status == "duplicate" and counts[0] <= 1 and counts[1] <= 1:
            msg = "duplicate association requires more than one PDB or JSON candidate"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_ordinal": self.source_ordinal,
            "protein_id": self.protein_id,
            "normalized_protein_id": self.normalized_protein_id,
            "status": self.status,
            "pdb_paths": list(self.pdb_paths),
            "json_paths": list(self.json_paths),
        }


@dataclass(frozen=True)
class UnmatchedFoldingResult:
    """One observed result file that cannot be associated safely."""

    path: str
    kind: ResultKind
    normalized_protein_id: str
    reason: UnmatchedReason
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "UnmatchedFoldingResult")
        _nonempty_str(self.path, "path")
        if self.kind not in {"pdb", "json"}:
            msg = f"unsupported result kind: {self.kind!r}"
            raise ValueError(msg)
        if self.reason == "malformed-name":
            if self.normalized_protein_id:
                msg = "malformed-name evidence cannot declare a normalized identity"
                raise ValueError(msg)
        elif self.reason == "unplanned-identity":
            _nonempty_str(self.normalized_protein_id, "normalized_protein_id")
        else:
            msg = f"unsupported unmatched reason: {self.reason!r}"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "kind": self.kind,
            "normalized_protein_id": self.normalized_protein_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FoldingResultInventory:
    """Deterministic association view for a declared folding index."""

    associations: tuple[FoldingResultAssociation, ...]
    unmatched: tuple[UnmatchedFoldingResult, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingResultInventory")
        _tuple(self.associations, "associations")
        _tuple(self.unmatched, "unmatched")
        if tuple(item.source_ordinal for item in self.associations) != tuple(range(len(self.associations))):
            msg = "result associations must retain contiguous planned source order"
            raise ValueError(msg)
        protein_ids = tuple(item.protein_id for item in self.associations)
        if len(protein_ids) != len(set(protein_ids)):
            msg = "result inventory contains duplicate planned protein identities"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "associations": [item.to_mapping() for item in self.associations],
            "unmatched": [item.to_mapping() for item in self.unmatched],
        }


@dataclass(frozen=True)
class ArchivePlanOptions:
    """Explicit deterministic batching and local path choices."""

    run_tag: str
    stage_root: str
    archive_root: str
    proteins_per_archive: int = 5_000
    lz4_executable: str = "lz4"
    start_index: int = 0
    max_archives: int | None = None
    shuffle: bool = True
    shuffle_seed: int = 42
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "ArchivePlanOptions")
        _nonempty_str(self.run_tag, "run_tag")
        if "/" in self.run_tag or self.run_tag in {".", ".."}:
            msg = "run_tag must be one safe archive filename component"
            raise ValueError(msg)
        _nonempty_str(self.stage_root, "stage_root")
        _nonempty_str(self.archive_root, "archive_root")
        _positive_int(self.proteins_per_archive, "proteins_per_archive")
        _nonempty_str(self.lz4_executable, "lz4_executable")
        _nonnegative_int(self.start_index, "start_index")
        if self.max_archives is not None:
            _nonnegative_int(self.max_archives, "max_archives")
        if not isinstance(self.shuffle, bool):
            msg = "shuffle must be a boolean"
            raise ValueError(msg)
        _nonnegative_int(self.shuffle_seed, "shuffle_seed")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_tag": self.run_tag,
            "stage_root": self.stage_root,
            "archive_root": self.archive_root,
            "proteins_per_archive": self.proteins_per_archive,
            "lz4_executable": self.lz4_executable,
            "start_index": self.start_index,
            "max_archives": self.max_archives,
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
        }


@dataclass(frozen=True)
class ArchiveMember:
    """One planned collision-safe copy into an isolated archive stage."""

    protein_id: str
    kind: ResultKind
    source_path: str
    stage_name: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "ArchiveMember")
        _nonempty_str(self.protein_id, "protein_id")
        if self.kind not in {"pdb", "json"}:
            msg = f"unsupported archive member kind: {self.kind!r}"
            raise ValueError(msg)
        _nonempty_str(self.source_path, "source_path")
        _nonempty_str(self.stage_name, "stage_name")
        if "/" in self.stage_name or self.stage_name in {".", ".."}:
            msg = "stage_name must be one safe relative filename"
            raise ValueError(msg)
        expected_suffix = f".{self.kind}"
        if not self.source_path.lower().endswith(expected_suffix) or not self.stage_name.lower().endswith(
            expected_suffix
        ):
            msg = f"{self.kind} archive members must retain the {expected_suffix} suffix"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protein_id": self.protein_id,
            "kind": self.kind,
            "source_path": self.source_path,
            "stage_name": self.stage_name,
        }


@dataclass(frozen=True)
class ArchiveManifestRecord:
    """Replayable planned membership; never evidence of created bytes."""

    archive_status: Literal["planned"]
    archive_batch_id: str
    archive_file: str
    archive_index: int
    run_tag: str
    protein_ids: tuple[str, ...]
    member_names: tuple[str, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "ArchiveManifestRecord")
        if self.archive_status != "planned":
            msg = "archive_status must be 'planned'; this seam cannot claim archive success"
            raise ValueError(msg)
        _nonempty_str(self.run_tag, "run_tag")
        _nonnegative_int(self.archive_index, "archive_index")
        expected_batch = f"{self.run_tag}_{self.archive_index:05d}"
        if self.archive_batch_id != expected_batch or self.archive_file != f"{expected_batch}.tar.lz4":
            msg = "manifest archive identity must match run_tag and archive_index"
            raise ValueError(msg)
        _string_tuple(self.protein_ids, "protein_ids", allow_empty=False)
        _string_tuple(self.member_names, "member_names", allow_empty=False)
        if len(self.protein_ids) != len(set(self.protein_ids)):
            msg = "manifest protein_ids must be unique"
            raise ValueError(msg)
        if len(self.member_names) != 2 * len(self.protein_ids) or len(self.member_names) != len(set(self.member_names)):
            msg = "manifest must declare two unique member names per protein"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "archive_status": self.archive_status,
            "archive_batch_id": self.archive_batch_id,
            "archive_file": self.archive_file,
            "archive_index": self.archive_index,
            "run_tag": self.run_tag,
            "protein_ids": list(self.protein_ids),
            "member_names": list(self.member_names),
        }


@dataclass(frozen=True)
class ArchiveBatchPlan:
    """One non-executing stage/tar/lz4 pipeline plan."""

    archive_index: int
    archive_name: str
    archive_path: str
    stage_dir: str
    protein_ids: tuple[str, ...]
    members: tuple[ArchiveMember, ...]
    tar_argv: tuple[str, ...]
    lz4_argv: tuple[str, ...]
    stdout_path: str
    manifest_record: ArchiveManifestRecord
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "ArchiveBatchPlan")
        _nonnegative_int(self.archive_index, "archive_index")
        _nonempty_str(self.archive_name, "archive_name")
        _nonempty_str(self.archive_path, "archive_path")
        _nonempty_str(self.stage_dir, "stage_dir")
        _string_tuple(self.protein_ids, "protein_ids", allow_empty=False)
        _tuple(self.members, "members")
        _string_tuple(self.tar_argv, "tar_argv", allow_empty=False)
        _string_tuple(self.lz4_argv, "lz4_argv", allow_empty=False)
        _nonempty_str(self.stdout_path, "stdout_path")
        if self.stdout_path != self.archive_path:
            msg = "stdout_path must identify the planned archive output"
            raise ValueError(msg)
        if (
            self.archive_name != self.manifest_record.archive_file
            or self.archive_index != self.manifest_record.archive_index
        ):
            msg = "batch and manifest archive identities must agree"
            raise ValueError(msg)
        if self.archive_path.rstrip("/").rsplit("/", maxsplit=1)[-1] != self.archive_name:
            msg = "archive_path must end with archive_name"
            raise ValueError(msg)
        if self.stage_dir.rstrip("/").rsplit("/", maxsplit=1)[-1] != self.manifest_record.archive_batch_id:
            msg = "stage_dir must end with archive_batch_id"
            raise ValueError(msg)
        if self.tar_argv != ("tar", "-cf", "-", "-C", self.stage_dir, "."):
            msg = "tar_argv must preserve the pinned stage-directory pipeline"
            raise ValueError(msg)
        if len(self.lz4_argv) != 3 or self.lz4_argv[1:] != ("-1", "-"):
            msg = "lz4_argv must preserve the pinned stdin compression vector"
            raise ValueError(msg)
        if self.protein_ids != self.manifest_record.protein_ids:
            msg = "batch and manifest protein identities must agree"
            raise ValueError(msg)
        if tuple(member.stage_name for member in self.members) != self.manifest_record.member_names:
            msg = "batch and manifest member names must agree"
            raise ValueError(msg)
        if len(self.members) != 2 * len(self.protein_ids):
            msg = "archive batch must contain one PDB and one JSON member per protein"
            raise ValueError(msg)
        by_protein: dict[str, set[ResultKind]] = {protein_id: set() for protein_id in self.protein_ids}
        for member in self.members:
            if member.protein_id not in by_protein:
                msg = "archive member references an unknown batch protein"
                raise ValueError(msg)
            by_protein[member.protein_id].add(member.kind)
        if any(kinds != {"pdb", "json"} for kinds in by_protein.values()):
            msg = "archive batch must contain exactly one PDB and JSON per protein"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "archive_index": self.archive_index,
            "archive_name": self.archive_name,
            "archive_path": self.archive_path,
            "stage_dir": self.stage_dir,
            "protein_ids": list(self.protein_ids),
            "members": [member.to_mapping() for member in self.members],
            "tar_argv": list(self.tar_argv),
            "lz4_argv": list(self.lz4_argv),
            "stdout_path": self.stdout_path,
            "manifest_record": self.manifest_record.to_mapping(),
        }


@dataclass(frozen=True)
class FoldingArchivePlan:
    """Deterministic collection of non-executing archive batches."""

    batches: tuple[ArchiveBatchPlan, ...]
    skipped_prior_protein_ids: tuple[str, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_version(self.schema_version, "FoldingArchivePlan")
        _tuple(self.batches, "batches")
        _string_tuple(self.skipped_prior_protein_ids, "skipped_prior_protein_ids")
        indices = tuple(batch.archive_index for batch in self.batches)
        if indices != tuple(sorted(set(indices))):
            msg = "archive batch indices must be unique and increasing"
            raise ValueError(msg)
        planned_ids = tuple(protein_id for batch in self.batches for protein_id in batch.protein_ids)
        if len(planned_ids) != len(set(planned_ids)):
            msg = "a protein identity cannot appear in more than one archive batch"
            raise ValueError(msg)
        if set(planned_ids) & set(self.skipped_prior_protein_ids):
            msg = "prior-manifest identities cannot also appear in new archive batches"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "batches": [batch.to_mapping() for batch in self.batches],
            "skipped_prior_protein_ids": list(self.skipped_prior_protein_ids),
        }


def archive_plan_from_mapping(payload: Mapping[str, object]) -> FoldingArchivePlan:
    _reject_unknown(payload, {"schema_version", "batches", "skipped_prior_protein_ids"}, "FoldingArchivePlan")
    _require_version(payload, "FoldingArchivePlan")
    batches = _mapping_list(payload, "batches", archive_batch_plan_from_mapping)
    return FoldingArchivePlan(
        batches=batches,
        skipped_prior_protein_ids=_str_tuple(payload, "skipped_prior_protein_ids"),
    )


def archive_plan_options_from_mapping(payload: Mapping[str, object]) -> ArchivePlanOptions:
    fields = {
        "schema_version",
        "run_tag",
        "stage_root",
        "archive_root",
        "proteins_per_archive",
        "lz4_executable",
        "start_index",
        "max_archives",
        "shuffle",
        "shuffle_seed",
    }
    _reject_unknown(payload, fields, "ArchivePlanOptions")
    _require_version(payload, "ArchivePlanOptions")
    max_archives = payload.get("max_archives")
    if max_archives is not None and (not isinstance(max_archives, int) or isinstance(max_archives, bool)):
        msg = "max_archives must be null or a non-negative integer"
        raise ValueError(msg)
    shuffle = payload.get("shuffle")
    if not isinstance(shuffle, bool):
        msg = "shuffle must be a boolean"
        raise ValueError(msg)
    return ArchivePlanOptions(
        run_tag=_str(payload, "run_tag"),
        stage_root=_str(payload, "stage_root"),
        archive_root=_str(payload, "archive_root"),
        proteins_per_archive=_int(payload, "proteins_per_archive"),
        lz4_executable=_str(payload, "lz4_executable"),
        start_index=_int(payload, "start_index"),
        max_archives=max_archives,
        shuffle=shuffle,
        shuffle_seed=_int(payload, "shuffle_seed"),
    )


def archive_batch_plan_from_mapping(payload: Mapping[str, object]) -> ArchiveBatchPlan:
    fields = {
        "schema_version",
        "archive_index",
        "archive_name",
        "archive_path",
        "stage_dir",
        "protein_ids",
        "members",
        "tar_argv",
        "lz4_argv",
        "stdout_path",
        "manifest_record",
    }
    _reject_unknown(payload, fields, "ArchiveBatchPlan")
    _require_version(payload, "ArchiveBatchPlan")
    return ArchiveBatchPlan(
        archive_index=_int(payload, "archive_index"),
        archive_name=_str(payload, "archive_name"),
        archive_path=_str(payload, "archive_path"),
        stage_dir=_str(payload, "stage_dir"),
        protein_ids=_str_tuple(payload, "protein_ids"),
        members=_mapping_list(payload, "members", archive_member_from_mapping),
        tar_argv=_str_tuple(payload, "tar_argv"),
        lz4_argv=_str_tuple(payload, "lz4_argv"),
        stdout_path=_str(payload, "stdout_path"),
        manifest_record=archive_manifest_record_from_mapping(_mapping(payload, "manifest_record")),
    )


def archive_member_from_mapping(payload: Mapping[str, object]) -> ArchiveMember:
    _reject_unknown(payload, {"schema_version", "protein_id", "kind", "source_path", "stage_name"}, "ArchiveMember")
    _require_version(payload, "ArchiveMember")
    return ArchiveMember(
        protein_id=_str(payload, "protein_id"),
        kind=cast(ResultKind, _str(payload, "kind")),
        source_path=_str(payload, "source_path"),
        stage_name=_str(payload, "stage_name"),
    )


def archive_manifest_record_from_mapping(payload: Mapping[str, object]) -> ArchiveManifestRecord:
    fields = {
        "schema_version",
        "archive_status",
        "archive_batch_id",
        "archive_file",
        "archive_index",
        "run_tag",
        "protein_ids",
        "member_names",
    }
    _reject_unknown(payload, fields, "ArchiveManifestRecord")
    _require_version(payload, "ArchiveManifestRecord")
    return ArchiveManifestRecord(
        archive_status=cast(Literal["planned"], _str(payload, "archive_status")),
        archive_batch_id=_str(payload, "archive_batch_id"),
        archive_file=_str(payload, "archive_file"),
        archive_index=_int(payload, "archive_index"),
        run_tag=_str(payload, "run_tag"),
        protein_ids=_str_tuple(payload, "protein_ids"),
        member_names=_str_tuple(payload, "member_names"),
    )


def folding_result_inventory_from_mapping(payload: Mapping[str, object]) -> FoldingResultInventory:
    _reject_unknown(payload, {"schema_version", "associations", "unmatched"}, "FoldingResultInventory")
    _require_version(payload, "FoldingResultInventory")
    return FoldingResultInventory(
        associations=_mapping_list(payload, "associations", folding_result_association_from_mapping),
        unmatched=_mapping_list(payload, "unmatched", unmatched_folding_result_from_mapping),
    )


def folding_result_association_from_mapping(payload: Mapping[str, object]) -> FoldingResultAssociation:
    fields = {
        "schema_version",
        "source_ordinal",
        "protein_id",
        "normalized_protein_id",
        "status",
        "pdb_paths",
        "json_paths",
    }
    _reject_unknown(payload, fields, "FoldingResultAssociation")
    _require_version(payload, "FoldingResultAssociation")
    return FoldingResultAssociation(
        source_ordinal=_int(payload, "source_ordinal"),
        protein_id=_str(payload, "protein_id"),
        normalized_protein_id=_str(payload, "normalized_protein_id"),
        status=cast(ResultStatus, _str(payload, "status")),
        pdb_paths=_str_tuple(payload, "pdb_paths"),
        json_paths=_str_tuple(payload, "json_paths"),
    )


def unmatched_folding_result_from_mapping(payload: Mapping[str, object]) -> UnmatchedFoldingResult:
    fields = {"schema_version", "path", "kind", "normalized_protein_id", "reason"}
    _reject_unknown(payload, fields, "UnmatchedFoldingResult")
    _require_version(payload, "UnmatchedFoldingResult")
    return UnmatchedFoldingResult(
        path=_str(payload, "path"),
        kind=cast(ResultKind, _str(payload, "kind")),
        normalized_protein_id=_str(payload, "normalized_protein_id", allow_empty=True),
        reason=cast(UnmatchedReason, _str(payload, "reason")),
    )


def _validate_version(value: int, name: str) -> None:
    validated = validate_schema_version(value, record_name=name)
    if validated != value:
        msg = f"{name} schema_version must be declared explicitly"
        raise ValueError(msg)


def _nonempty_str(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{name} must be a non-empty string"
        raise ValueError(msg)


def _nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{name} must be a non-negative integer"
        raise ValueError(msg)


def _positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)


def _tuple(value: object, name: str) -> None:
    if not isinstance(value, tuple):
        msg = f"{name} must be an immutable tuple"
        raise ValueError(msg)


def _string_tuple(value: tuple[str, ...], name: str, *, allow_empty: bool = True) -> None:
    _tuple(value, name)
    if not allow_empty and not value:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    if any(not isinstance(item, str) or not item.strip() for item in value):
        msg = f"{name} must contain only non-empty strings"
        raise ValueError(msg)


def _reject_unknown(payload: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(str(key) for key in set(payload) - allowed)
    if unknown:
        msg = f"{name} has unknown fields: {', '.join(unknown)}"
        raise ValueError(msg)


def _require_version(payload: Mapping[str, object], name: str) -> None:
    value = payload.get("schema_version")
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{name} requires integer schema_version"
        raise ValueError(msg)
    validate_schema_version(value, record_name=name)


def _str(payload: Mapping[str, object], key: str, *, allow_empty: bool = False) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        msg = f"{key} must be a {'string' if allow_empty else 'non-empty string'}"
        raise ValueError(msg)
    return value


def _int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{key} must be a non-negative integer"
        raise ValueError(msg)
    return value


def _str_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        msg = f"{key} must be a list of strings"
        raise ValueError(msg)
    return tuple(value)


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be an object"
        raise ValueError(msg)
    return cast(Mapping[str, object], value)


def _mapping_list[T](
    payload: Mapping[str, object],
    key: str,
    loader: Callable[[Mapping[str, object]], T],
) -> tuple[T, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        msg = f"{key} must be a list of objects"
        raise ValueError(msg)
    return tuple(loader(cast(Mapping[str, object], item)) for item in value)


__all__ = [
    "ArchiveBatchPlan",
    "ArchiveManifestRecord",
    "ArchiveMember",
    "ArchivePlanOptions",
    "FoldingArchivePlan",
    "FoldingResultAssociation",
    "FoldingResultInventory",
    "ResultKind",
    "ResultStatus",
    "UnmatchedFoldingResult",
    "UnmatchedReason",
    "archive_batch_plan_from_mapping",
    "archive_manifest_record_from_mapping",
    "archive_member_from_mapping",
    "archive_plan_from_mapping",
    "archive_plan_options_from_mapping",
    "folding_result_association_from_mapping",
    "folding_result_inventory_from_mapping",
    "unmatched_folding_result_from_mapping",
]
