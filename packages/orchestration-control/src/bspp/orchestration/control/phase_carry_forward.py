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

"""Control-owned derivation of immutable preprocessing Attempt carry-forward."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from bspp.orchestration.contract.phase import PhasePlan, PhaseRunSpec, canonical_mapping_digest
from bspp.orchestration.contract.phase_action_evidence_attestation import (
    PhaseActionEvidenceAttestedEvent,
)
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardContent,
    AttemptCarryForwardRecord,
    AttemptCarryForwardReference,
    AttemptCarryForwardRequest,
    AttemptCarryForwardVerificationReference,
    AttemptWorkspaceBinding,
    AttemptWorkspaceRootBinding,
    attempt_carry_forward_id,
    attempt_carry_forward_request_from_mapping,
)
from bspp.orchestration.contract.phase_retry import (
    phase_input_set_identity_digest,
    phase_scientific_identity_digest,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION

if TYPE_CHECKING:
    from bspp.orchestration.control.phase_authority import PhaseAuthorityValidation


def load_attempt_carry_forward_request(path: Path) -> AttemptCarryForwardRequest:
    """Strict-load one caller selection without accepting paths or target authority."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("carry-forward request must be a regular file")
    try:
        payload = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("carry-forward request must be UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("carry-forward request must be one JSON mapping")
    return attempt_carry_forward_request_from_mapping(payload)


def derive_attempt_carry_forward(
    *,
    authority: PhaseAuthorityValidation,
    successor_runspec: PhaseRunSpec,
    target_attempt_ordinal: int,
    request: AttemptCarryForwardRequest,
    declared_at: str,
) -> tuple[AttemptCarryForwardRecord, PhaseRunSpec]:
    """Derive the exact carry record and referenced successor RunSpec."""
    phase_plan = authority.phase_plan
    if not isinstance(phase_plan, PhasePlan):
        raise ValueError("carry-forward derivation requires a preprocessing Phase Plan")
    phase_runspec = authority.phase_runspec
    if not isinstance(phase_runspec, PhaseRunSpec):
        raise ValueError("carry-forward derivation requires a preprocessing Phase RunSpec")
    attestation = authority.current_action_evidence_attestation
    if attestation is None:
        raise ValueError("carry-forward derivation requires durable evidence attestation")
    if request.source_attempt_id != authority.current_attempt.attempt_id:
        raise ValueError("carry-forward request source must be the current predecessor Attempt")
    source_action = phase_runspec.payload.actions[0]
    target_action = successor_runspec.payload.actions[0]
    if source_action.action_id != target_action.action_id:
        raise ValueError("carry-forward requires one invariant preprocessing action")
    expected = target_action.payload.expected_a3ms
    if len(request.content) >= len(expected):
        raise ValueError("carry-forward selection must be a proper ExpectedA3M subset")
    selected_names = tuple(item.member_name for item in request.content)
    declared_names = tuple(item.member_name for item in expected)
    ordered_selection = tuple(name for name in declared_names if name in set(selected_names))
    if selected_names != ordered_selection:
        raise ValueError("carry-forward selection must follow ExpectedA3M declaration order")
    evidence_hashes = {
        item.member_name: item
        for item in attestation.payload.evidence.output_hashes
        if item.role == "a3m" and item.member_name is not None
    }
    records = {item.source_ordinal: item for item in phase_plan.payload.work_plan.input.records}
    workspace = _target_workspace(successor_runspec)
    content: list[AttemptCarryForwardContent] = []
    for selection in request.content:
        declared = next((item for item in expected if item.member_name == selection.member_name), None)
        observed = evidence_hashes.get(selection.member_name)
        if declared is None or observed is None:
            raise ValueError("carry-forward selection lacks exact expected attested A3M evidence")
        work_record = records.get(declared.source_ordinal)
        if (
            work_record is None
            or work_record.identity != declared.record_identity
            or work_record.header != declared.source_header
        ):
            raise ValueError("carry-forward ExpectedA3M identity does not match immutable work record")
        if observed.size_bytes != selection.size_bytes or observed.sha256 != selection.sha256:
            raise ValueError("carry-forward request hash/size does not match attested evidence")
        source_declared = str(
            PurePosixPath(source_action.payload.evidence.scratch_output_directory) / declared.member_name
        )
        target_declared = str(
            PurePosixPath(target_action.payload.evidence.scratch_output_directory) / declared.member_name
        )
        if observed.path != source_declared:
            raise ValueError("attested A3M path does not match predecessor RunSpec")
        source_physical, source_mode = _source_physical_path(authority, source_declared)
        if source_mode != workspace.source_path_mode:
            workspace = AttemptWorkspaceBinding(
                **{**workspace.__dict__, "source_path_mode": source_mode},
            )
        target_physical = _physical_path(workspace.roots, target_declared)
        private_source = str(
            PurePosixPath("/run/bspp-carry/sources")
            / authority.current_attempt.attempt_id
            / successor_runspec.attempt_id
            / target_action.action_id
            / f"{declared.source_ordinal:06d}"
            / declared.member_name
        )
        content.append(
            AttemptCarryForwardContent(
                source_action_id=source_action.action_id,
                target_action_id=target_action.action_id,
                member_name=declared.member_name,
                source_ordinal=declared.source_ordinal,
                record_identity=declared.record_identity,
                source_header=declared.source_header,
                source_declared_path=source_declared,
                target_declared_path=target_declared,
                source_physical_path=source_physical,
                target_physical_path=target_physical,
                source_private_mount_path=private_source,
                size_bytes=selection.size_bytes,
                sha256=selection.sha256,
            )
        )
    remaining_ordinals = tuple(item.source_ordinal for item in expected if item.member_name not in selected_names)
    remaining_bytes = fasta_bytes(tuple(records[ordinal] for ordinal in remaining_ordinals))
    content_tuple = tuple(content)
    content_digest = canonical_mapping_digest(
        {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "content": [item.to_mapping() for item in content_tuple],
        }
    )
    verification = _verification_reference(attestation)
    identity: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "phase_run_id": authority.phase_run.phase_run_id,
        "phase_kind": "preprocessing",
        "phase_plan_digest": phase_plan.digest,
        "input_set_identity_digest": phase_input_set_identity_digest(phase_plan),
        "scientific_identity_digest": phase_scientific_identity_digest(phase_plan),
        "source_attempt_id": authority.current_attempt.attempt_id,
        "source_attempt_ordinal": authority.current_attempt.ordinal,
        "source_runspec_digest": phase_runspec.digest,
        "target_attempt_id": successor_runspec.attempt_id,
        "target_attempt_ordinal": target_attempt_ordinal,
        "verification": verification.to_mapping(),
        "workspace": workspace.to_mapping(),
        "content": [item.to_mapping() for item in content_tuple],
        "content_digest": content_digest,
        "remaining_record_ordinals": list(remaining_ordinals),
        "remaining_search_input_sha256": hashlib.sha256(remaining_bytes).hexdigest(),
        "declared_at": declared_at,
    }
    record = AttemptCarryForwardRecord(
        attempt_carry_forward_id=attempt_carry_forward_id(identity),
        phase_run_id=authority.phase_run.phase_run_id,
        phase_kind="preprocessing",
        phase_plan_digest=phase_plan.digest,
        input_set_identity_digest=phase_input_set_identity_digest(phase_plan),
        scientific_identity_digest=phase_scientific_identity_digest(phase_plan),
        source_attempt_id=authority.current_attempt.attempt_id,
        source_attempt_ordinal=authority.current_attempt.ordinal,
        source_runspec_digest=phase_runspec.digest,
        target_attempt_id=successor_runspec.attempt_id,
        target_attempt_ordinal=target_attempt_ordinal,
        verification=verification,
        workspace=workspace,
        content=content_tuple,
        content_digest=content_digest,
        remaining_record_ordinals=remaining_ordinals,
        remaining_search_input_sha256=hashlib.sha256(remaining_bytes).hexdigest(),
        declared_at=declared_at,
    )
    reference = AttemptCarryForwardReference(
        attempt_carry_forward_id=record.attempt_carry_forward_id,
        digest=record.digest,
        location=f"attempts/{successor_runspec.attempt_id}/attempt-carry-forward.json",
    )
    carried_runspec = PhaseRunSpec(**{**successor_runspec.__dict__, "carry_forward": reference})
    return record, carried_runspec


def fasta_bytes(records: Sequence[object]) -> bytes:
    """Return canonical strict-two-line FASTA bytes for ordered work records."""
    lines: list[str] = []
    for record in records:
        identity = getattr(record, "identity", None)
        sequence = getattr(record, "sequence", None)
        if not isinstance(identity, str) or not isinstance(sequence, str):
            raise TypeError("FASTA serialization requires strict work-plan records")
        lines.extend((f">{identity}", sequence))
    return ("\n".join(lines) + "\n").encode()


def _verification_reference(
    event: PhaseActionEvidenceAttestedEvent,
) -> AttemptCarryForwardVerificationReference:
    return AttemptCarryForwardVerificationReference(
        attestation_id=event.payload.attestation_id,
        attestation_digest=event.payload.digest,
        attestation_event_sequence=event.sequence,
        evidence_path=event.payload.action_evidence_path,
        evidence_document_sha256=event.payload.evidence_document_sha256,
        evidence_mapping_digest=event.payload.evidence_mapping_digest,
        terminal_event_digest=event.payload.terminal_event_digest,
        evidence_finished_at=event.payload.evidence.finished_at,
    )


def _target_workspace(runspec: PhaseRunSpec) -> AttemptWorkspaceBinding:
    action = runspec.payload.actions[0]
    site = action.payload.site
    action_root = (
        PurePosixPath(runspec.cluster.output_root)
        / "bspp-phase-runs"
        / runspec.phase_run_id
        / runspec.attempt_id
        / action.action_id
    )
    workspace_root = action_root / "work"
    logical = (
        ("input", site.input_root),
        ("scratch-output", site.scratch_output_root),
        ("project-logs", site.project_logs_root),
        ("finished-msa", site.finished_msa_root),
        ("split-input", site.split_input_root),
        ("finished-input", site.finished_input_root),
    )
    roots = tuple(
        AttemptWorkspaceRootBinding(
            role=role,
            logical_root=value,
            physical_root=str(workspace_root / role),
        )
        for role, value in logical
    )
    return AttemptWorkspaceBinding(
        phase_run_id=runspec.phase_run_id,
        target_attempt_id=runspec.attempt_id,
        target_action_id=action.action_id,
        workspace_root=str(workspace_root),
        roots=roots,
        search_input_physical_path=_physical_path(roots, action.payload.search_argv[3]),
        split_input_physical_path=_physical_path(roots, action.payload.package.completed_input_source_path),
        identity_sentinel_path=str(workspace_root / "carry-forward-identity.json"),
        private_workspace_mount_path=str(
            PurePosixPath("/run/bspp-carry/workspace") / runspec.phase_run_id / runspec.attempt_id / action.action_id
        ),
        source_path_mode="predecessor-workspace" if runspec.carry_forward is not None else "frozen-mount-inverse",
    )


def _physical_path(roots: tuple[AttemptWorkspaceRootBinding, ...], logical: str) -> str:
    logical_path = PurePosixPath(logical)
    matches = tuple(
        item
        for item in roots
        if logical_path == PurePosixPath(item.logical_root) or PurePosixPath(item.logical_root) in logical_path.parents
    )
    if not matches:
        raise ValueError(f"logical work path has no workspace binding: {logical}")
    selected = max(matches, key=lambda item: len(PurePosixPath(item.logical_root).parts))
    relative = logical_path.relative_to(PurePosixPath(selected.logical_root))
    return str(PurePosixPath(selected.physical_root) / relative)


def _source_physical_path(
    authority: PhaseAuthorityValidation,
    logical: str,
) -> tuple[str, str]:
    if authority.current_carry_forward is not None:
        record = authority.current_carry_forward
        if not isinstance(record, AttemptCarryForwardRecord):
            raise ValueError("predecessor source resolution requires a preprocessing carry record")
        return _physical_path(record.workspace.roots, logical), "predecessor-workspace"
    phase_runspec = authority.phase_runspec
    if not isinstance(phase_runspec, PhaseRunSpec):
        raise ValueError("predecessor source resolution requires a preprocessing Phase RunSpec")
    action = phase_runspec.payload.actions[0]
    raw_mounts: list[tuple[str, str]] = []
    for value in action.payload.site.container_mounts:
        if value.count(":") > 1:
            raise ValueError("predecessor container mount must use source[:target] grammar")
        source, separator, target = value.partition(":")
        raw_mounts.append((source, target if separator else source))
    raw_mounts.extend((mount.source, mount.target) for mount in phase_runspec.cluster.extra_mounts)
    logical_path = _normalized(logical, "source logical path")
    candidates: set[str] = set()
    for source, target in raw_mounts:
        source_path = _normalized(source, "predecessor mount source")
        target_path = _normalized(target, "predecessor mount target")
        if logical_path == target_path or target_path in logical_path.parents:
            candidates.add(str(source_path / logical_path.relative_to(target_path)))
    if len(candidates) != 1:
        raise ValueError("predecessor logical A3M must inverse-resolve through exactly one frozen mount")
    return next(iter(candidates)), "frozen-mount-inverse"


def _normalized(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{label} must be a normalized absolute POSIX path")
    return path


__all__ = [
    "derive_attempt_carry_forward",
    "fasta_bytes",
    "load_attempt_carry_forward_request",
]
