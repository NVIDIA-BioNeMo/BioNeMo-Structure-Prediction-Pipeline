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

"""Seal one postprocessing Phase Attempt from bounded local evidence only."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_acceptance_adjudication import (
    PostprocessingAcceptanceAdjudication,
    postprocessing_acceptance_adjudication_from_mapping,
)
from bspp.orchestration.contract.postprocessing_acceptance_capture import (
    PostprocessingAcceptanceCapture,
    postprocessing_acceptance_capture_from_mapping,
)
from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    AcceptanceStepName,
)
from bspp.orchestration.contract.postprocessing_action09_bundle import (
    PostprocessingAction09AssemblyWitness,
    PostprocessingFinalizationHandoffIndex,
    postprocessing_action09_assembly_witness_from_mapping,
)
from bspp.orchestration.contract.postprocessing_artifact_locations import (
    postprocessing_artifact_location_set_from_mapping,
)
from bspp.orchestration.contract.postprocessing_artifacts import (
    PostprocessingEvidenceArtifact,
    PostprocessingScientificOutputInventory,
)
from bspp.orchestration.contract.postprocessing_attestations import (
    PostprocessingRuntimeInputAttestationSet,
    postprocessing_runtime_input_attestation_set_from_mapping,
)
from bspp.orchestration.contract.postprocessing_bundle_manifest import (
    PostprocessingScientificOutputRoot,
    PostprocessingTarManifest,
    postprocessing_scientific_output_root_from_mapping,
    postprocessing_tar_manifest_from_mapping,
)
from bspp.orchestration.contract.postprocessing_handoff import (
    PostprocessingActionReceiptEvidence,
    PostprocessingOutputHandoff,
    postprocessing_output_handoff_from_mapping,
)
from bspp.orchestration.contract.postprocessing_phase_receipt import (
    PostprocessingFinalizedPayload,
    PostprocessingPhaseReceipt,
    postprocessing_receipt_id,
)
from bspp.orchestration.contract.postprocessing_runspec_v3 import PostprocessingPhaseRunSpecV3
from bspp.orchestration.contract.postprocessing_runtime_evidence import (
    PostprocessingRuntimeActionEvidenceAggregate,
    postprocessing_runtime_action_evidence_aggregate_from_mapping,
)
from bspp.orchestration.contract.postprocessing_scheduler_evidence import (
    PostprocessingSchedulerEvidence,
    postprocessing_scheduler_evidence_from_mapping,
)
from bspp.orchestration.contract.postprocessing_transfer_limits import (
    POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1,
)
from bspp.orchestration.control.postprocessing_attempt_projection import (
    validate_v3_attempt_execution_projection,
)
from bspp.orchestration.control.postprocessing_authority_reader import (
    reject_historical_postprocessing_mutation,
    require_postprocessing_v2_authority,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    append_event as _append_event,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    postprocessing_operation_lock as _postprocessing_operation_lock,
)
from bspp.orchestration.control.postprocessing_evidence_transfer import (
    validate_local_postprocessing_handoff,
)
from bspp.orchestration.control.postprocessing_phase_lifecycle import Clock
from bspp.orchestration.control.postprocessing_phase_rendering import (
    PostprocessingRenderInput,
    postprocessing_action_command_digest,
)
from bspp.orchestration.control.postprocessing_phase_types import PostprocessingAuthority
from bspp.orchestration.control.postprocessing_scheduler_evidence import (
    postprocessing_scheduler_evidence_from_authority,
)

_CAPTURE_LAYOUT = (
    (
        "acceptance-tar-payload-parity",
        "acceptance/captures/acceptance-tar-payload-parity.json",
        "acceptance/reports/tar-payload-parity-report.json",
    ),
    (
        "acceptance-semantic",
        "acceptance/captures/acceptance-semantic.json",
        "acceptance/reports/semantic-acceptance-summary.json",
    ),
    (
        "acceptance-verify-evidence",
        "acceptance/captures/acceptance-verify-evidence.json",
        "acceptance/reports/acceptance-evidence-report.json",
    ),
)


@dataclass(frozen=True)
class PostprocessingPhaseFinalizationResult:
    phase_run_id: str
    attempt_id: str
    phase_receipt_id: str
    output_handoff_id: str
    artifact_set_id: str
    physical_location_ids: tuple[str, ...]
    status: Literal["accepted"] = "accepted"

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase_kind": "postprocessing",
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_receipt_id": self.phase_receipt_id,
            "output_handoff_id": self.output_handoff_id,
            "artifact_set_id": self.artifact_set_id,
            "physical_location_ids": list(self.physical_location_ids),
            "status": self.status,
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


def finalize_postprocessing_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    scheduler_evidence_path: Path,
    aggregate_action_evidence_path: Path,
    handoff_path: Path,
    acceptance_adjudication_path: Path,
    clock: Clock | None = None,
) -> PostprocessingPhaseFinalizationResult:
    """Verify a fetched handoff plus Control scheduler authority and seal it."""
    reject_historical_postprocessing_mutation(authority_root, phase_run_id)
    with _postprocessing_operation_lock(authority_root, phase_run_id):
        authority = require_postprocessing_v2_authority(authority_root, phase_run_id)
        if isinstance(authority.runspec, PostprocessingPhaseRunSpecV3):
            validate_v3_attempt_execution_projection(
                authority.runspec,
                authority.legacy_runspec,
                phase_plan_output_namespace=authority.phase_plan.output_namespace,
            )
        stored = _stored_finalized_payload(authority)
        if not authority.current_attempt_projection_complete:
            raise ValueError("postprocessing Finalization requires complete current Attempt projections")
        if stored is None and (authority.sealed or authority.status != "submitted"):
            raise ValueError("postprocessing Finalization requires a non-failed submitted Attempt")

        index = validate_local_postprocessing_handoff(handoff_path, authority=authority)
        _require_exact_member_argument(
            aggregate_action_evidence_path,
            handoff_path / "aggregate-action-evidence.json",
            label="aggregate action evidence",
        )
        _require_exact_member_argument(
            acceptance_adjudication_path,
            handoff_path / "acceptance/adjudication.json",
            label="acceptance adjudication",
        )
        finalized_at = stored.receipt.finalized_at if stored is not None else _timestamp((clock or _now)())
        candidate = _build_finalized_payload(
            authority,
            scheduler=_load_scheduler_evidence(scheduler_evidence_path, authority=authority),
            aggregate=_load_aggregate(aggregate_action_evidence_path),
            handoff_root=handoff_path,
            index=index,
            finalized_at=finalized_at,
        )
        if stored is not None:
            if candidate.to_mapping() != stored.to_mapping():
                raise ValueError("sealed postprocessing Phase evidence differs from the accepted receipt")
            return _result(stored)
        accepted = _append_event(
            authority,
            event_type="postprocessing-phase-finalized",
            occurred_at=finalized_at,
            payload=candidate,
        )
        persisted = _stored_finalized_payload(accepted)
        if persisted is None:
            raise AssertionError("accepted postprocessing authority omitted its final event")
        return _result(persisted)


def _build_finalized_payload(
    authority: PostprocessingAuthority,
    *,
    scheduler: PostprocessingSchedulerEvidence,
    aggregate: PostprocessingRuntimeActionEvidenceAggregate,
    handoff_root: Path,
    index: PostprocessingFinalizationHandoffIndex,
    finalized_at: str,
) -> PostprocessingFinalizedPayload:
    render_input = PostprocessingRenderInput(runspec=authority.runspec, legacy_runspec=authority.legacy_runspec)
    command_digests = {
        action.action_id: postprocessing_action_command_digest(render_input, action)
        for action in authority.runspec.payload.actions
    }
    _reconcile_runtime_actions(authority, scheduler, aggregate, command_digests=command_digests)
    _reconcile_action09_witness(aggregate.action09_prepublication_witness, index=index)

    attestations = postprocessing_runtime_input_attestation_set_from_mapping(
        _load_member(handoff_root, "inputs/runtime-input-attestations.json")
    )
    _reconcile_input_attestations(authority, attestations)
    captures = _load_captures(authority, handoff_root)
    adjudication = postprocessing_acceptance_adjudication_from_mapping(
        _load_member(handoff_root, "acceptance/adjudication.json")
    )
    _reconcile_adjudication(authority, captures, adjudication)
    witness_document = postprocessing_action09_assembly_witness_from_mapping(
        _load_member(handoff_root, "acceptance/bundle.json")
    )
    if witness_document != aggregate.action09_prepublication_witness:
        raise ValueError("Action 09 witness differs between the acceptance bundle and action aggregate")

    scientific_root = postprocessing_scientific_output_root_from_mapping(
        _load_member(handoff_root, "outputs/scientific-output-root.json")
    )
    manifests = tuple(
        postprocessing_tar_manifest_from_mapping(_load_member(handoff_root, reference.path))
        for reference in scientific_root.tar_manifests
    )
    inventory = _scientific_inventory(authority, scientific_root, manifests)
    handoff = postprocessing_output_handoff_from_mapping(_load_member(handoff_root, "outputs/output-handoff.json"))
    _reconcile_handoff(authority, handoff_root, handoff, inventory, witness_document)

    terminal_actions = tuple(
        PostprocessingActionReceiptEvidence(
            action_id=action.action_id,
            runtime_action_digest=action.runtime_action_digest,
            parent_job_id=action.parent_job_id,
            expected_task_indexes=action.expected_task_indexes,
            tasks=action.tasks,
        )
        for action in scheduler.actions
    )
    inventory_document = _canonical_bytes(inventory.to_mapping())
    action_digest = canonical_mapping_digest(
        {"schema_version": 1, "terminal_actions": [item.to_mapping() for item in terminal_actions]}
    )
    handoff_digest = canonical_mapping_digest(handoff.to_mapping())
    policy = authority.runspec.payload.acceptance_policy
    receipt_identity: dict[str, object] = {
        "schema_version": 1,
        "phase_run_id": authority.phase_run_id,
        "attempt_id": authority.attempt_id,
        "phase_plan_digest": authority.runspec.phase_plan_digest,
        "phase_runspec_digest": authority.runspec.digest,
        "logical_input_manifest_digest": authority.runspec.payload.logical_inputs.digest,
        "scientific_identity_digest": authority.runspec.payload.scientific_identity.digest,
        "action_semantics_digest": authority.runspec.payload.action_semantics_digest,
        "execution_projection_sha256": authority.runspec.payload.execution_projection.document_sha256,
        "qualified_runtime_digest": authority.runspec.payload.qualified_runtime.digest,
        "runtime_input_attestations_digest": attestations.digest,
        "scientific_output_inventory_digest": inventory.digest,
        "scientific_output_inventory_sha256": hashlib.sha256(inventory_document).hexdigest(),
        "scientific_output_inventory_size_bytes": len(inventory_document),
        "acceptance_policy_id": policy.policy_id,
        "acceptance_policy_sha256": policy.sha256,
        "acceptance_policy_size_bytes": policy.size_bytes,
        "acceptance_policy_semantic_digest": policy.semantic_digest,
        "baseline_id": authority.acceptance_policy.baseline_id,
        "baseline_version": authority.acceptance_policy.baseline_version,
        "action_graph_digest": authority.runspec.payload.action_graph_digest,
        "terminal_action_evidence_digest": action_digest,
        "acceptance_capture_digests": [item.digest for item in captures],
        "acceptance_adjudication_digest": adjudication.digest,
        "output_handoff_id": handoff.handoff_id,
        "output_handoff_digest": handoff_digest,
        "output_artifact_set_id": handoff.artifact_set.artifact_set_id,
        "physical_artifact_location_ids": [item.artifact_location_id for item in handoff.physical_locations],
        "finalized_at": finalized_at,
        "result": "passed",
    }
    receipt = PostprocessingPhaseReceipt(
        phase_receipt_id=postprocessing_receipt_id(receipt_identity),
        phase_run_id=authority.phase_run_id,
        attempt_id=authority.attempt_id,
        phase_plan_digest=authority.runspec.phase_plan_digest,
        phase_runspec_digest=authority.runspec.digest,
        logical_input_manifest_digest=authority.runspec.payload.logical_inputs.digest,
        scientific_identity_digest=authority.runspec.payload.scientific_identity.digest,
        action_semantics_digest=authority.runspec.payload.action_semantics_digest,
        execution_projection_sha256=authority.runspec.payload.execution_projection.document_sha256,
        qualified_runtime_digest=authority.runspec.payload.qualified_runtime.digest,
        runtime_input_attestations_digest=attestations.digest,
        scientific_output_inventory_digest=inventory.digest,
        scientific_output_inventory_sha256=hashlib.sha256(inventory_document).hexdigest(),
        scientific_output_inventory_size_bytes=len(inventory_document),
        acceptance_policy_id=policy.policy_id,
        acceptance_policy_sha256=policy.sha256,
        acceptance_policy_size_bytes=policy.size_bytes,
        acceptance_policy_semantic_digest=policy.semantic_digest,
        baseline_id=authority.acceptance_policy.baseline_id,
        baseline_version=authority.acceptance_policy.baseline_version,
        action_graph_digest=authority.runspec.payload.action_graph_digest,
        terminal_action_evidence_digest=action_digest,
        acceptance_capture_digests=tuple(item.digest for item in captures),
        acceptance_adjudication_digest=adjudication.digest,
        output_handoff_id=handoff.handoff_id,
        output_handoff_digest=handoff_digest,
        output_artifact_set_id=handoff.artifact_set.artifact_set_id,
        physical_artifact_location_ids=tuple(item.artifact_location_id for item in handoff.physical_locations),
        finalized_at=finalized_at,
    )
    return PostprocessingFinalizedPayload(
        terminal_actions=terminal_actions,
        runtime_input_attestations=attestations,
        scientific_output_inventory=inventory,
        acceptance_captures=captures,
        acceptance_adjudication=adjudication,
        output_handoff=handoff,
        receipt=receipt,
    )


def _load_scheduler_evidence(
    path: Path,
    *,
    authority: PostprocessingAuthority,
) -> PostprocessingSchedulerEvidence:
    scheduler = postprocessing_scheduler_evidence_from_mapping(
        _canonical_mapping_from_path(path, label="postprocessing scheduler evidence")
    )
    expected = postprocessing_scheduler_evidence_from_authority(authority)
    if scheduler != expected:
        raise ValueError("provided scheduler evidence differs from durable Control authority")
    return scheduler


def _load_aggregate(path: Path) -> PostprocessingRuntimeActionEvidenceAggregate:
    return postprocessing_runtime_action_evidence_aggregate_from_mapping(
        _canonical_mapping_from_path(path, label="postprocessing Runtime action aggregate")
    )


def _reconcile_runtime_actions(
    authority: PostprocessingAuthority,
    scheduler: PostprocessingSchedulerEvidence,
    aggregate: PostprocessingRuntimeActionEvidenceAggregate,
    *,
    command_digests: Mapping[str, str],
) -> None:
    runspec = authority.runspec
    if (
        aggregate.phase_run_id,
        aggregate.attempt_id,
        aggregate.phase_runspec_digest,
        aggregate.action_graph_digest,
    ) != (runspec.phase_run_id, runspec.attempt_id, runspec.digest, runspec.payload.action_graph_digest):
        raise ValueError("Runtime action aggregate differs from current Attempt authority")
    expected_prior = runspec.payload.actions[:-1]
    if tuple(item.action_id for item in aggregate.completed_actions) != tuple(
        item.action_id for item in expected_prior
    ):
        raise ValueError("Runtime action aggregate differs from the exact RunSpec Action 01--08 subset")
    scheduler_by_id = {item.action_id: item for item in scheduler.actions}
    for action, runtime_action in zip(expected_prior, aggregate.completed_actions, strict=True):
        scheduled = scheduler_by_id[action.action_id]
        runtime_tasks = tuple((item.task_index, item.scheduler_job_id) for item in runtime_action.tasks)
        scheduler_tasks = tuple((item.task_index, item.scheduler_job_id) for item in scheduled.tasks)
        if (
            runtime_action.runtime_action_digest != action.digest
            or runtime_action.expected_task_indexes != action.expected_task_indexes
            or runtime_tasks != scheduler_tasks
            or any(item.command_digest != command_digests[action.action_id] for item in runtime_action.tasks)
        ):
            raise ValueError(f"Runtime and scheduler evidence differ for {action.action_id!r}")
    action09 = runspec.payload.actions[-1]
    witness = aggregate.action09_prepublication_witness
    scheduled09 = scheduler_by_id.get(action09.action_id)
    if (
        action09.action_id != witness.action_id
        or witness.runtime_action_digest != action09.digest
        or witness.command_digest != command_digests[action09.action_id]
        or scheduled09 is None
        or scheduled09.runtime_action_digest != action09.digest
    ):
        raise ValueError("Action 09 witness/scheduler evidence differs from its frozen command")


def _reconcile_action09_witness(
    witness: PostprocessingAction09AssemblyWitness,
    *,
    index: PostprocessingFinalizationHandoffIndex,
) -> None:
    intended = tuple(
        item for item in index.members if item.path not in {"aggregate-action-evidence.json", "acceptance/bundle.json"}
    )
    if witness.intended_members != intended:
        raise ValueError("Action 09 prepublication witness differs from the indexed assembly")


def _reconcile_input_attestations(
    authority: PostprocessingAuthority,
    attestations: PostprocessingRuntimeInputAttestationSet,
) -> None:
    runspec = authority.runspec
    if (
        attestations.phase_run_id,
        attestations.attempt_id,
        attestations.phase_runspec_digest,
    ) != (runspec.phase_run_id, runspec.attempt_id, runspec.digest):
        raise ValueError("runtime input attestations differ from current Attempt identity")
    logical = {item.name: item for item in runspec.payload.logical_inputs.entries}
    physical = {item.name: item for item in runspec.payload.physical_inputs}
    observed = {item.logical_input_name: item for item in attestations.attestations}
    if set(observed) != set(logical):
        raise ValueError("runtime input attestations omit or add logical inputs")
    for name, expected in logical.items():
        actual = observed[name]
        locator = physical.get(name)
        if (
            locator is None
            or actual.verification_kind != "authority-declared-content-v1"
            or actual.authority != expected.member_identity
            or actual.physical_input_name != name
            or actual.physical_locator != locator.locator
            or actual.member_identity != expected.member_identity
            or (actual.content_sha256, actual.size_bytes)
            != (expected.expected_content_sha256, expected.expected_size_bytes)
            or not actual.accessible
        ):
            raise ValueError(f"runtime declared input identity differs for {name!r}")
    qualified = runspec.payload.qualified_runtime
    qualification = attestations.runtime_qualification
    if (
        qualification.qualified_runtime_digest != qualified.digest
        or qualification.record_location != qualified.qualification_location
        or qualification.record_sha256 != qualified.qualification_sha256
        or qualification.record_size_bytes != qualified.qualification_size_bytes
        or qualification.tuple_id != qualified.tuple_id
        or qualification.source_identity_digest != qualified.source_identity_digest
        or qualification.source_package_identity_digest != qualified.source_package_identity_digest
        or qualification.toolkit_identity_digest != qualified.toolkit_identity_digest
        or qualification.runtime_component_identity_digest != qualified.runtime_component_identity_digest
    ):
        raise ValueError("runtime qualification attestation differs from the current Attempt")


def _load_captures(
    authority: PostprocessingAuthority,
    handoff_root: Path,
) -> tuple[PostprocessingAcceptanceCapture, ...]:
    result: list[PostprocessingAcceptanceCapture] = []
    policy = authority.runspec.payload.acceptance_policy
    action_by_step = {item.step_name: item for item in authority.runspec.payload.actions}
    contract_by_step = {item.step_name: item for item in authority.acceptance_policy.completion_exit_contracts}
    for step, capture_path, report_bundle_path in _CAPTURE_LAYOUT:
        capture_mapping, capture_document = _load_member_document(handoff_root, capture_path)
        capture = postprocessing_acceptance_capture_from_mapping(capture_mapping)
        action = action_by_step[step]
        contract = contract_by_step[cast("AcceptanceStepName", step)]
        if (
            capture.phase_run_id != authority.phase_run_id
            or capture.attempt_id != authority.attempt_id
            or capture.action_id != action.action_id
            or capture.step_name != step
            or capture.policy_sha256 != policy.sha256
            or capture.raw_exit_code not in contract.allowed_raw_exit_codes
            or tuple(item.path for item in capture.reports) != contract.report_paths
        ):
            raise ValueError(f"acceptance capture differs from current Attempt/policy: {step}")
        report_binding = capture.reports[0]
        report_mapping, report_document = _load_member_document(handoff_root, report_bundle_path)
        del report_mapping
        if (
            _bytes_identity(report_document)
            != (
                report_binding.sha256,
                report_binding.size_bytes,
            )
            or report_binding.path != contract.report_paths[0]
        ):
            raise ValueError(f"bundled acceptance report differs from its capture: {step}")
        if _canonical_bytes(capture.to_mapping()) != capture_document:
            raise ValueError(f"bundled acceptance capture is not canonical: {step}")
        result.append(capture)
    return tuple(result)


def _reconcile_adjudication(
    authority: PostprocessingAuthority,
    captures: tuple[PostprocessingAcceptanceCapture, ...],
    adjudication: PostprocessingAcceptanceAdjudication,
) -> None:
    policy = authority.runspec.payload.acceptance_policy
    if (
        adjudication.phase_run_id != authority.phase_run_id
        or adjudication.attempt_id != authority.attempt_id
        or adjudication.policy_id != policy.policy_id
        or adjudication.policy_sha256 != policy.sha256
        or adjudication.capture_digests != tuple(item.digest for item in captures)
        or adjudication.result != "passed"
    ):
        raise ValueError("acceptance adjudication is not the exact passed current-Attempt verdict")


def _scientific_inventory(
    authority: PostprocessingAuthority,
    root: PostprocessingScientificOutputRoot,
    manifests: tuple[PostprocessingTarManifest, ...],
) -> PostprocessingScientificOutputInventory:
    runspec = authority.runspec
    if (
        root.phase_run_id,
        root.attempt_id,
        root.phase_runspec_digest,
        root.output_root,
    ) != (
        runspec.phase_run_id,
        runspec.attempt_id,
        runspec.digest,
        runspec.payload.attempt_paths.output_dir,
    ):
        raise ValueError("scientific output root differs from current Attempt authority")
    manifest_by_id = {item.manifest_id: item for item in manifests}
    if len(manifest_by_id) != len(manifests):
        raise ValueError("scientific output root contains duplicate tar manifests")
    members: list[PostprocessingEvidenceArtifact] = [
        PostprocessingEvidenceArtifact(path=item.path, sha256=item.sha256, size_bytes=item.size_bytes)
        for item in root.small_outputs
    ]
    tar_paths: set[str] = set()
    for reference in root.tar_manifests:
        manifest = manifest_by_id.get(reference.manifest_id)
        if (
            manifest is None
            or reference.path != f"outputs/tar-manifests/{manifest.manifest_id}.json"
            or reference.member_count != len(manifest.members)
            or manifest.tar_path in tar_paths
        ):
            raise ValueError("scientific tar manifest differs from its root reference")
        tar_paths.add(manifest.tar_path)
        members.append(
            PostprocessingEvidenceArtifact(
                path=manifest.tar_path,
                sha256=manifest.manifest_id,
                size_bytes=sum(item.size_bytes for item in manifest.members),
                verification_kind="inventory-metadata-v1",
                verification_source_path=reference.path,
            )
        )
    return PostprocessingScientificOutputInventory(
        phase_run_id=authority.phase_run_id,
        attempt_id=authority.attempt_id,
        phase_runspec_digest=authority.runspec.digest,
        output_root=root.output_root,
        members=tuple(sorted(members, key=lambda item: item.path)),
    )


def _reconcile_handoff(
    authority: PostprocessingAuthority,
    handoff_root: Path,
    handoff: PostprocessingOutputHandoff,
    inventory: PostprocessingScientificOutputInventory,
    witness: PostprocessingAction09AssemblyWitness,
) -> None:
    paths = authority.runspec.payload.attempt_paths
    if (
        handoff.phase_run_id,
        handoff.attempt_id,
        handoff.phase_runspec_digest,
        handoff.legacy_run_id,
        handoff.output_namespace,
        handoff.output_dir,
        handoff.object_prefix,
    ) != (
        authority.phase_run_id,
        authority.attempt_id,
        authority.runspec.digest,
        paths.legacy_run_id,
        authority.phase_plan.output_namespace,
        paths.output_dir,
        paths.object_prefix,
    ):
        raise ValueError("output handoff differs from current Attempt authority")
    artifact_set_mapping = _load_member(handoff_root, "outputs/artifact-set-root.json")
    if artifact_set_mapping != handoff.artifact_set.to_mapping():
        raise ValueError("published Artifact Set root differs from the output handoff")
    locations = postprocessing_artifact_location_set_from_mapping(
        _load_member(handoff_root, "outputs/artifact-locations.json")
    )
    if (
        locations.phase_run_id != authority.phase_run_id
        or locations.attempt_id != authority.attempt_id
        or locations.phase_runspec_digest != authority.runspec.digest
        or locations.artifact_set_id != handoff.artifact_set.artifact_set_id
        or locations.locations != handoff.physical_locations
        or any(item.verified_at != witness.assembled_at for item in locations.locations)
    ):
        raise ValueError("published Artifact Locations differ from the output handoff/Action 09 witness")
    children = {item.root_name: item for item in handoff.artifact_set.children}
    scientific = children.get("scientific-output")
    if scientific is None or scientific.members != inventory.members:
        raise ValueError("published scientific Artifact Set differs from tar/small-output manifests")


def _load_member(root: Path, relative: str) -> Mapping[str, object]:
    return _load_member_document(root, relative)[0]


def _load_member_document(root: Path, relative: str) -> tuple[Mapping[str, object], bytes]:
    path = root / Path(*relative.split("/"))
    document = path.read_bytes()
    return _canonical_mapping(document, label=f"postprocessing handoff member {relative!r}"), document


def _canonical_mapping_from_path(path: Path, *, label: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    document = path.read_bytes()
    if len(document) > POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1.max_file_bytes:
        raise ValueError(f"{label} exceeds the immutable evidence file limit")
    return _canonical_mapping(document, label=label)


def _canonical_mapping(document: bytes, *, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(document)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping) or _canonical_bytes(payload) != document:
        raise ValueError(f"{label} must be a canonical JSON mapping")
    return cast("Mapping[str, object]", payload)


def _require_exact_member_argument(path: Path, expected: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError(f"{label} must be the exact indexed member beneath --handoff")


def _stored_finalized_payload(authority: PostprocessingAuthority) -> PostprocessingFinalizedPayload | None:
    matches = tuple(
        event.payload for event in authority.events if isinstance(event.payload, PostprocessingFinalizedPayload)
    )
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("postprocessing authority has duplicate finalization events")
    return matches[0]


def _result(payload: PostprocessingFinalizedPayload) -> PostprocessingPhaseFinalizationResult:
    return PostprocessingPhaseFinalizationResult(
        phase_run_id=payload.receipt.phase_run_id,
        attempt_id=payload.receipt.attempt_id,
        phase_receipt_id=payload.receipt.phase_receipt_id,
        output_handoff_id=payload.output_handoff.handoff_id,
        artifact_set_id=payload.output_handoff.artifact_set.artifact_set_id,
        physical_location_ids=tuple(item.artifact_location_id for item in payload.output_handoff.physical_locations),
    )


def _bytes_identity(document: bytes) -> tuple[str, int]:
    return hashlib.sha256(document).hexdigest(), len(document)


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def _timestamp(moment: datetime) -> str:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("postprocessing Finalization clock must be timezone-aware")
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _now() -> datetime:
    return datetime.now(UTC)


__all__ = ["PostprocessingPhaseFinalizationResult", "finalize_postprocessing_phase"]
