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

"""Fail-closed isolation, validation, and publication of raw search artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from pathlib import Path

from bspp.orchestration.contract.phase import PhaseRunSpec, PreprocessingRuntimeAction
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardAdoptionEvidence,
    AttemptCarryForwardRecord,
)
from bspp.orchestration.contract.preprocessing import PreprocessingFastaRecord
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingOutputHash,
    PreprocessingRawSearchArtifact,
    PreprocessingRawSearchArtifactRole,
    PreprocessingRawSearchEvidence,
)
from bspp.orchestration.runtime.preprocessing.content_validation import validate_preprocessing_a3m_header_bytes


class PreprocessingRawSearchError(ValueError):
    """The raw paired-query closure did not match the pinned adapter model."""


def searched_preprocessing_records(
    runspec: PhaseRunSpec,
    carry_forward_record: AttemptCarryForwardRecord | None,
) -> tuple[PreprocessingFastaRecord, ...]:
    """Select the exact source-ordered records supplied to the search kernel."""
    action = runspec.payload.actions[0]
    records = {item.source_ordinal: item for item in runspec.payload.work_plan.input.records}
    ordinals = (
        carry_forward_record.remaining_record_ordinals
        if carry_forward_record is not None
        else tuple(item.source_ordinal for item in action.payload.expected_a3ms)
    )
    try:
        selected = tuple(records[ordinal] for ordinal in ordinals)
    except KeyError as exc:
        raise PreprocessingRawSearchError("searched source ordinal is absent from the work plan") from exc
    _unique_chain_lengths_and_cardinalities(selected)
    return selected


def validate_raw_search_closure(
    runspec: PhaseRunSpec,
    action: PreprocessingRuntimeAction,
    *,
    carry_forward_record: AttemptCarryForwardRecord | None,
) -> PreprocessingRawSearchEvidence:
    """Validate and attest the exact raw inventory for the sealed search mode."""
    records = searched_preprocessing_records(runspec, carry_forward_record)
    raw_directory = Path(action.payload.evidence.raw_search_output_directory)
    _require_real_directory(raw_directory)
    expected_by_ordinal = {item.source_ordinal: item for item in action.payload.expected_a3ms}
    expected_named = tuple(expected_by_ordinal[item.source_ordinal] for item in records)
    count = len(records)
    unique_lengths, unique_cardinalities = _unique_chain_lengths_and_cardinalities(records)
    expected_numeric = _expected_numeric_ids(action.payload.search_argv, count, len(unique_lengths))
    expected_members = tuple(item.member_name for item in expected_named) + tuple(
        f"{raw_id}.a3m" for raw_id in expected_numeric
    )
    entries = tuple(raw_directory.iterdir())
    observed_members = tuple(item.name for item in entries)
    if len(observed_members) != len(expected_members) or set(observed_members) != set(expected_members):
        missing = sorted(set(expected_members) - set(observed_members))
        extra = sorted(set(observed_members) - set(expected_members))
        raise PreprocessingRawSearchError(f"raw-search inventory mismatch; missing={missing}, extra={extra}")

    artifacts: list[PreprocessingRawSearchArtifact] = []
    for record, expected in zip(records, expected_named, strict=True):
        path = raw_directory / expected.member_name
        data = _read_nonempty_regular_no_follow(path)
        lengths = tuple(len(chain) for chain in record.sequence.split(":"))
        try:
            validate_preprocessing_a3m_header_bytes(data, chain_lengths=lengths, label=str(path))
        except ValueError as exc:
            raise PreprocessingRawSearchError(str(exc)) from exc
        artifacts.append(
            _artifact(
                role="named-a3m",
                path=path,
                data=data,
                source_ordinal=record.source_ordinal,
                declared_member=expected.member_name,
            )
        )
    for raw_id in expected_numeric:
        path = raw_directory / f"{raw_id}.a3m"
        data = _read_nonempty_regular_no_follow(path)
        modeled_length = unique_lengths[raw_id]
        modeled_cardinality = unique_cardinalities[raw_id]
        expected_data = f"#{modeled_length}\t{modeled_cardinality}\n".encode()
        if data != expected_data:
            raise PreprocessingRawSearchError(f"numeric raw placeholder bytes do not match model: {path}")
        artifacts.append(
            _artifact(
                role="numeric-placeholder",
                path=path,
                data=data,
                raw_query_id=raw_id,
                modeled_chain_length=modeled_length,
                modeled_cardinality=modeled_cardinality,
            )
        )
    return PreprocessingRawSearchEvidence(
        raw_search_output_directory=str(raw_directory),
        searched_source_ordinals=tuple(item.source_ordinal for item in records),
        artifacts=tuple(artifacts),
    )


def publish_validated_named_a3ms(
    evidence: PreprocessingRawSearchEvidence,
    *,
    staging_directory: Path,
) -> None:
    """Exclusively publish raw named A3Ms and prove the copy is byte-identical."""
    _require_real_directory(staging_directory)
    named = tuple(item for item in evidence.artifacts if item.role == "named-a3m")
    for item in named:
        data = _read_nonempty_regular_no_follow(Path(item.path))
        if len(data) != item.size_bytes or hashlib.sha256(data).hexdigest() != item.sha256:
            raise PreprocessingRawSearchError(f"raw named A3M changed before publication: {item.member_name}")
        destination = staging_directory / item.member_name
        _write_exclusive(destination, data)
        published = _read_nonempty_regular_no_follow(destination)
        if published != data:
            raise PreprocessingRawSearchError(f"published named A3M changed during publication: {item.member_name}")


def reconcile_raw_search_evidence(
    runspec: PhaseRunSpec,
    evidence: PreprocessingRawSearchEvidence,
    output_hashes: tuple[PreprocessingOutputHash, ...],
    *,
    carry_forward_adoption: AttemptCarryForwardAdoptionEvidence | None,
    verify_raw_files: bool,
) -> None:
    """Reproduce raw semantics from authority, optionally reopening ephemeral raw files."""
    action = runspec.payload.actions[0]
    plan = action.payload.evidence
    if evidence.raw_search_output_directory != plan.raw_search_output_directory:
        raise PreprocessingRawSearchError("raw-search evidence directory does not match the execution plan")
    all_expected = tuple(action.payload.expected_a3ms)
    carried_members = (
        frozenset(item.member_name for item in carry_forward_adoption.content)
        if carry_forward_adoption is not None
        else frozenset()
    )
    searched_expected = tuple(item for item in all_expected if item.member_name not in carried_members)
    searched_ordinals = tuple(item.source_ordinal for item in searched_expected)
    if evidence.searched_source_ordinals != searched_ordinals:
        raise PreprocessingRawSearchError("raw-search ordinals do not match the exact searched complement")
    records_by_ordinal = {item.source_ordinal: item for item in runspec.payload.work_plan.input.records}
    records = tuple(records_by_ordinal[ordinal] for ordinal in searched_ordinals)
    unique_lengths, unique_cardinalities = _unique_chain_lengths_and_cardinalities(records)
    count = len(records)
    expected_numeric = _expected_numeric_ids(action.payload.search_argv, count, len(unique_lengths))
    if len(evidence.artifacts) != count + len(expected_numeric):
        raise PreprocessingRawSearchError("raw artifact count does not match the sealed search mode")
    named = evidence.artifacts[:count]
    numeric = evidence.artifacts[count:]
    for artifact, expected in zip(named, searched_expected, strict=True):
        if (
            artifact.role != "named-a3m"
            or artifact.source_ordinal != expected.source_ordinal
            or artifact.member_name != expected.member_name
            or artifact.declared_member != expected.member_name
        ):
            raise PreprocessingRawSearchError("raw named evidence does not match declared searched outputs")
    for artifact, raw_id in zip(numeric, expected_numeric, strict=True):
        expected_data = f"#{unique_lengths[raw_id]}\t{unique_cardinalities[raw_id]}\n".encode()
        if (
            artifact.role != "numeric-placeholder"
            or artifact.raw_query_id != raw_id
            or artifact.modeled_chain_length != unique_lengths[raw_id]
            or artifact.modeled_cardinality != unique_cardinalities[raw_id]
            or artifact.member_name != f"{raw_id}.a3m"
            or artifact.size_bytes != len(expected_data)
            or artifact.sha256 != hashlib.sha256(expected_data).hexdigest()
        ):
            raise PreprocessingRawSearchError("raw numeric placeholder evidence does not match the pinned model")
    published = {item.member_name: item for item in output_hashes if item.role == "a3m"}
    for artifact in named:
        output = published.get(artifact.member_name)
        if output is not None and (output.size_bytes != artifact.size_bytes or output.sha256 != artifact.sha256):
            raise PreprocessingRawSearchError("raw named evidence does not match published A3M hashes")
    if verify_raw_files:
        records_by_named_member = {
            expected.member_name: record for expected, record in zip(searched_expected, records, strict=True)
        }
        for artifact in evidence.artifacts:
            data = _read_nonempty_regular_no_follow(Path(artifact.path))
            if len(data) != artifact.size_bytes or hashlib.sha256(data).hexdigest() != artifact.sha256:
                raise PreprocessingRawSearchError(f"raw-search artifact changed after validation: {artifact.path}")
            if artifact.role == "named-a3m":
                record = records_by_named_member[artifact.member_name]
                lengths = tuple(len(chain) for chain in record.sequence.split(":"))
                try:
                    validate_preprocessing_a3m_header_bytes(
                        data,
                        chain_lengths=lengths,
                        label=artifact.path,
                    )
                except ValueError as exc:
                    raise PreprocessingRawSearchError(str(exc)) from exc


def _expected_numeric_ids(search_argv: tuple[str, ...], record_count: int, unique_chain_count: int) -> tuple[int, ...]:
    """Reproduce pinned ColabFold cleanup from authority, never from observed files."""
    if search_argv.count("--pair-mode") != 1:
        raise PreprocessingRawSearchError("raw search requires exactly one sealed pair mode")
    index = search_argv.index("--pair-mode") + 1
    mode = search_argv[index] if index < len(search_argv) else None
    if mode == "unpaired_paired":
        # ColabFold consumes and unlinks every per-chain unpaired A3M before
        # assembling and renaming the per-record named A3Ms.
        return ()
    if mode == "paired":
        # Historical sealed plans retain header-only per-chain placeholders;
        # the first M are overwritten by per-record assembly and then renamed.
        return tuple(range(record_count, unique_chain_count))
    raise PreprocessingRawSearchError(f"unsupported sealed raw-search pair mode: {mode!r}")


def _unique_chain_lengths_and_cardinalities(
    records: tuple[PreprocessingFastaRecord, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return per-record unique chain lengths and cardinalities in record-major order.

    Extends the mmsa chain-length model from fixed-2-distinct to variable-1-N-unique
    with cardinalities.  For each record: split sequence by ``:``, check all chains
    non-empty, deduplicate in first-occurrence order (dict insertion order),
    collect ``len(unique_chain)`` and ``cardinality``.
    """
    unique_lengths: list[int] = []
    unique_cardinalities: list[int] = []
    for record in records:
        chains = record.sequence.split(":")
        if any(not chain for chain in chains):
            raise PreprocessingRawSearchError("preprocessing search requires non-empty chains per searched record")
        seen: dict[str, int] = {}
        for chain in chains:
            seen[chain] = seen.get(chain, 0) + 1
        unique_lengths.extend(len(chain) for chain in seen)
        unique_cardinalities.extend(count for count in seen.values())
    return tuple(unique_lengths), tuple(unique_cardinalities)


def _artifact(
    *,
    role: PreprocessingRawSearchArtifactRole,
    path: Path,
    data: bytes,
    source_ordinal: int | None = None,
    declared_member: str | None = None,
    raw_query_id: int | None = None,
    modeled_chain_length: int | None = None,
    modeled_cardinality: int | None = None,
) -> PreprocessingRawSearchArtifact:
    return PreprocessingRawSearchArtifact(
        role=role,
        member_name=path.name,
        path=str(path),
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        source_ordinal=source_ordinal,
        declared_member=declared_member,
        raw_query_id=raw_query_id,
        modeled_chain_length=modeled_chain_length,
        modeled_cardinality=modeled_cardinality,
    )


def _require_real_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise PreprocessingRawSearchError(f"raw-search path must be a real directory: {path}")


def _read_nonempty_regular_no_follow(path: Path) -> bytes:
    try:
        expected = path.lstat()
    except OSError as exc:
        raise PreprocessingRawSearchError(f"raw-search artifact cannot be inspected safely: {path}") from exc
    if not stat.S_ISREG(expected.st_mode):
        raise PreprocessingRawSearchError(f"raw-search artifact must be a nonempty regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PreprocessingRawSearchError(f"raw-search artifact cannot be opened safely: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or (before.st_dev, before.st_ino) != (expected.st_dev, expected.st_ino)
        ):
            raise PreprocessingRawSearchError(f"raw-search artifact must be a nonempty regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PreprocessingRawSearchError(f"raw-search artifact changed during validation: {path}")
    data = b"".join(chunks)
    if len(data) != before.st_size:
        raise PreprocessingRawSearchError(f"raw-search artifact size changed during validation: {path}")
    return data


def _write_exclusive(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.raw-publish-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "PreprocessingRawSearchError",
    "publish_validated_named_a3ms",
    "reconcile_raw_search_evidence",
    "searched_preprocessing_records",
    "validate_raw_search_closure",
]
