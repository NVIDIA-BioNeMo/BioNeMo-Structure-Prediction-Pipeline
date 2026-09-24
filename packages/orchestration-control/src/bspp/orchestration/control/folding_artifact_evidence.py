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

"""Bounded folding finalization metadata; never follows prediction locators."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from bspp.orchestration.contract.folding_artifact_evidence import (
    ARTIFACT_EVIDENCE_PROFILE,
    MAX_FINALIZATION_INDEX_BYTES,
    ArtifactFoldEvidence,
    FoldingFinalizationIndex,
)
from bspp.orchestration.contract.folding_execution import folding_canonical_pair_handoff_from_mapping
from bspp.orchestration.contract.phase import FoldingPhaseRunSpec, canonical_mapping_digest

from .phase_authority import PhaseAuthorityStore, PhaseAuthorityValidation, require_complete_current_runspec
from .postprocessing_evidence_transfer import _fsync_tree, _rename_directory_no_replace
from .transport import CommandRunner, RemoteSlurmTransport, default_command_runner


class FoldingMetadataSource(Protocol):
    def fetch_stable_artifact(self, remote_path: Path, *, remote_root: Path, maximum_bytes: int) -> bytes: ...


def folding_actions_root(runspec: FoldingPhaseRunSpec) -> PurePosixPath:
    return (
        PurePosixPath(runspec.cluster.project_root)
        / "bspp-phase-runs"
        / runspec.phase_run_id
        / runspec.attempt_id
        / "actions"
    )


def _canonical_id(runspec: FoldingPhaseRunSpec) -> str:
    return next(action.action_id for action in runspec.payload.actions if action.action_kind == "canonical-pair")


def _mapping(data: bytes) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key in folding metadata")
            result[key] = value
        return result

    def invalid(value: str) -> object:
        raise ValueError(f"nonfinite folding metadata constant: {value}")

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, Mapping):
        raise ValueError("folding metadata must contain an object")
    return cast("Mapping[str, object]", value)


def _open(path: Path, *, directory: bool = False) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("folding bundle path must be absolute and normalized")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, component in enumerate(path.parts[1:]):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if directory or index < len(path.parts) - 2:
                flags |= os.O_DIRECTORY
            following = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _snapshot(path: Path, limit: int) -> bytes:
    with os.fdopen(_open(path), "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= limit:
            raise ValueError("folding bundle member is not one bounded regular file")
        data = handle.read(limit + 1)
        after = os.fstat(handle.fileno())
        current = _open(path)
        try:
            present = os.fstat(current)
        finally:
            os.close(current)
        if (
            len(data) != before.st_size
            or _signature(before) != _signature(after)
            or _signature(after) != _signature(present)
        ):
            raise ValueError("folding bundle member changed during its snapshot")
        return data


def _index(data: bytes, runspec: FoldingPhaseRunSpec) -> FoldingFinalizationIndex:
    if not 0 < len(data) <= MAX_FINALIZATION_INDEX_BYTES:
        raise ValueError("folding finalization index exceeds its fixed bound")
    result = FoldingFinalizationIndex.from_mapping(_mapping(data))
    result.validate_binding(runspec)
    return result


def _tree(root: Path, expected: set[str]) -> None:
    descriptor = _open(root, directory=True)
    os.close(descriptor)
    actual: set[str] = set()
    expected_directories = {str(PurePosixPath(path).parent) for path in expected}
    directories: set[str] = set()
    for parent, names, files in os.walk(root, followlinks=False):
        for name in names:
            path = Path(parent) / name
            if path.is_symlink() or not path.is_dir():
                raise ValueError("unsafe folding bundle directory")
            directories.add(path.relative_to(root).as_posix())
        for name in files:
            actual.add((Path(parent) / name).relative_to(root).as_posix())
        if len(actual) > 5 or len(directories) > 2:
            raise ValueError("folding bundle has unexpected descendants")
    if actual != expected or directories != expected_directories:
        raise ValueError("folding bundle does not have exactly its indexed metadata")


def validate_local_folding_bundle(
    handoff: Path,
    *,
    runspec: FoldingPhaseRunSpec,
    action_evidence_path: Path | None = None,
) -> Mapping[str, Mapping[str, object]]:
    """Authenticate the four metadata snapshots, without opening score/PDB paths."""
    canonical_id = _canonical_id(runspec)
    index_path = handoff / canonical_id / "finalization-index.json"
    index_bytes = _snapshot(index_path, MAX_FINALIZATION_INDEX_BYTES)
    index = _index(index_bytes, runspec)
    expected = {f"{canonical_id}/finalization-index.json", *(member.path for member in index.members)}
    _tree(handoff, expected)
    aggregate_path = handoff / canonical_id / "action-evidence.json"
    if action_evidence_path is not None and action_evidence_path != aggregate_path:
        raise ValueError("folding action evidence must be the exact indexed bundle member")
    documents: dict[str, Mapping[str, object]] = {}
    for member in index.members:
        data = _snapshot(handoff / member.path, member.size_bytes)
        if len(data) != member.size_bytes or hashlib.sha256(data).hexdigest() != member.sha256:
            raise ValueError("folding metadata differs from its published index")
        documents[member.path] = _mapping(data)
    if _snapshot(index_path, MAX_FINALIZATION_INDEX_BYTES) != index_bytes:
        raise ValueError("folding finalization index changed while reading its bundle")
    aggregate = documents[f"{canonical_id}/action-evidence.json"]
    if any(not isinstance(value, Mapping) for value in aggregate.values()):
        raise ValueError("folding aggregate requires per-action objects")
    evidence = cast("Mapping[str, Mapping[str, object]]", aggregate)
    from .folding_phase_adapter import validate_folding_action_evidence

    canonical_index = validate_folding_action_evidence(phase_runspec=runspec, evidence=evidence)
    assert canonical_index is not None
    fold = ArtifactFoldEvidence.from_mapping(documents[f"{index.fold_action_id}/handoff.json"])
    if fold.to_mapping() != evidence[index.fold_action_id]:
        raise ValueError("fold handoff differs from aggregate fold evidence")
    canonical = ArtifactFoldEvidence.from_mapping(evidence[canonical_id])
    handoff_record = folding_canonical_pair_handoff_from_mapping(documents[f"{canonical_id}/handoff.json"])
    if (handoff_record.phase_run_id, handoff_record.attempt_id, handoff_record.action_id) != (
        runspec.phase_run_id,
        runspec.attempt_id,
        canonical_id,
    ):
        raise ValueError("canonical handoff authority mismatch")
    if handoff_record.predecessor_digest != canonical_mapping_digest(fold.to_mapping()):
        raise ValueError("canonical handoff predecessor differs from fold handoff")
    if handoff_record.index_path != str(folding_actions_root(runspec) / canonical_id / "canonical-pair-index.json"):
        raise ValueError("canonical index locator differs from this Attempt")
    index_member = next(
        member for member in index.members if member.path == f"{canonical_id}/canonical-pair-index.json"
    )
    if (
        handoff_record.index_digest != index_member.sha256
        or handoff_record.index_digest != hashlib.sha256(canonical_index.to_json().encode()).hexdigest()
    ):
        raise ValueError("canonical handoff index digest mismatch")
    if documents[f"{canonical_id}/canonical-pair-index.json"] != canonical_index.to_mapping():
        raise ValueError("indexed canonical index differs from accepted references")
    expected_entries = [
        {
            "schema_version": 1,
            "target_id": entry.target.target_id,
            "sequence_sha256": entry.target.sequence_sha256,
            "model_entity_id": entry.model_entity_id,
            "tool_used": entry.tool_used,
            "structure_path": entry.structure.path,
            "scores_path": entry.scores.path,
        }
        for entry in canonical.entries
    ]
    if [entry.to_mapping() for entry in handoff_record.entries] != expected_entries:
        raise ValueError("canonical handoff entries differ from authenticated references")
    for source in (
        fold.orchestration_source_commit,
        canonical.orchestration_source_commit,
        handoff_record.orchestration_source_commit,
    ):
        if source != index.orchestration_source_commit:
            raise ValueError("folding metadata source attestations disagree")
    if handoff_record.install_mode != canonical.install_mode:
        raise ValueError("folding metadata install-mode attestations disagree")
    # This source is Runtime-attested and cross-bound, not compared to a source
    # SHA absent from the historical Cluster Snapshot. Campaign deployment pins
    # and the independent artifact audit provide that separate verification.
    return evidence


def _require_terminal(authority: PhaseAuthorityValidation) -> FoldingPhaseRunSpec:
    require_complete_current_runspec(authority, operation="folding evidence fetch")
    runspec = authority.phase_runspec
    if not isinstance(runspec, FoldingPhaseRunSpec) or runspec.payload.evidence_profile != ARTIFACT_EVIDENCE_PROFILE:
        raise ValueError("folding evidence fetch requires artifact-backed-v2 authority")
    canonical_id = _canonical_id(runspec)
    submission = authority.submission
    if submission is None or submission.status != "submitted":
        raise ValueError("folding evidence fetch requires complete durable submission")
    assignments = [item for item in submission.actions if item.action_id == canonical_id]
    observations = [item for item in authority.terminal_observations if item.action_id == canonical_id]
    if len(assignments) != 1 or len(observations) != 1:
        raise ValueError("folding evidence fetch requires exact canonical terminal authority")
    assigned, observed = assignments[0], observations[0]
    if (
        assigned.status != "submitted"
        or assigned.job_id is None
        or (observed.job_id, observed.state, observed.exit_code, observed.source, observed.outcome)
        != (assigned.job_id, "COMPLETED", "0:0", "sacct", "succeeded")
    ):
        raise ValueError("folding evidence fetch requires the assigned canonical job COMPLETED 0:0")
    return runspec


def fetch_folding_finalization_evidence(
    phase_run_id: str,
    *,
    authority_root: Path,
    destination: Path,
    runner: CommandRunner = default_command_runner,
    source: FoldingMetadataSource | None = None,
) -> dict[str, object]:
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    runspec = _require_terminal(authority)
    if not destination.is_absolute():
        destination = Path(os.path.abspath(destination))
    canonical_id = _canonical_id(runspec)
    remote_root = Path(folding_actions_root(runspec))
    remote_index = remote_root / canonical_id / "finalization-index.json"
    if source is None:
        source = RemoteSlurmTransport(
            kind=runspec.cluster.transport, ssh_target=runspec.cluster.ssh_target, runner=runner
        )
    index_bytes = source.fetch_stable_artifact(
        remote_index, remote_root=remote_root, maximum_bytes=MAX_FINALIZATION_INDEX_BYTES
    )
    index = _index(index_bytes, runspec)
    if destination.exists() or destination.is_symlink():
        validate_local_folding_bundle(destination, runspec=runspec)
        if (
            _snapshot(destination / canonical_id / "finalization-index.json", MAX_FINALIZATION_INDEX_BYTES)
            != index_bytes
        ):
            raise ValueError("existing folding bundle differs from published evidence")
    else:
        descriptor = _open(destination.parent, directory=True)
        os.close(descriptor)
        temporary = Path(tempfile.mkdtemp(prefix=".folding-evidence-", dir=destination.parent))
        try:
            for relative, data in [(f"{canonical_id}/finalization-index.json", index_bytes)]:
                path = temporary / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            for member in index.members:
                data = source.fetch_stable_artifact(
                    remote_root / member.path, remote_root=remote_root, maximum_bytes=member.size_bytes
                )
                if len(data) != member.size_bytes or hashlib.sha256(data).hexdigest() != member.sha256:
                    raise ValueError("fetched folding metadata differs from published index")
                path = temporary / member.path
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            if (
                source.fetch_stable_artifact(
                    remote_index, remote_root=remote_root, maximum_bytes=MAX_FINALIZATION_INDEX_BYTES
                )
                != index_bytes
            ):
                raise ValueError("published folding index changed during transfer")
            validate_local_folding_bundle(temporary, runspec=runspec)
            _fsync_tree(temporary)
            _rename_directory_no_replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return {
        "schema_version": 1,
        "phase_kind": "folding",
        "operation": "evidence-fetch",
        "phase_run_id": phase_run_id,
        "attempt_id": runspec.attempt_id,
        "destination": str(destination),
        "indexed_file_count": 4,
        "aggregate_bytes": sum(member.size_bytes for member in index.members) + len(index_bytes),
        "status": "fetched",
    }
