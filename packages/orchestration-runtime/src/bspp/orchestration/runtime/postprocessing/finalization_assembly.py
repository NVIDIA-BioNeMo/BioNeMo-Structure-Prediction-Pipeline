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

"""Runtime-only scientific roots and atomic postprocessing finalization bundles.

Large scientific tar payloads never enter the bounded handoff.  Runtime streams
their regular-file members into content manifests, hashes ordinary small output
files, verifies all prior Runtime and acceptance evidence, and atomically
publishes the JSON-only Action 09 handoff.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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
    PostprocessingAcceptancePolicySnapshot,
    PostprocessingCompletionExitContract,
    postprocessing_acceptance_policy_from_mapping,
)
from bspp.orchestration.contract.postprocessing_action09_bundle import (
    PostprocessingAction09AssemblyWitness,
    PostprocessingFinalizationHandoffIndex,
)
from bspp.orchestration.contract.postprocessing_artifact_locations import (
    PostprocessingArtifactLocationSet,
)
from bspp.orchestration.contract.postprocessing_artifacts import (
    PostprocessingEvidenceArtifact,
    PostprocessingLogicalArtifactSet,
    postprocessing_artifact_set_id,
)
from bspp.orchestration.contract.postprocessing_attestations import (
    PostprocessingRuntimeInputAttestationSet,
    postprocessing_runtime_input_attestation_set_from_mapping,
)
from bspp.orchestration.contract.postprocessing_bundle_manifest import (
    PostprocessingTarManifest,
)
from bspp.orchestration.contract.postprocessing_handoff import (
    PostprocessingArtifactPlacement,
    PostprocessingOutputHandoff,
    PostprocessingVerifiedArtifactLocation,
    postprocessing_handoff_id,
)
from bspp.orchestration.contract.postprocessing_phase_ids import POSTPROCESSING_ACTION_IDS
from bspp.orchestration.contract.postprocessing_runspec import (
    ExecutablePostprocessingPhaseRunSpec,
)
from bspp.orchestration.contract.postprocessing_runtime_evidence import (
    PostprocessingRuntimeActionEvidenceAggregate,
)
from bspp.orchestration.contract.postprocessing_transfer_limits import (
    POSTPROCESSING_FINALIZATION_FIXED_PATHS,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION
from bspp.orchestration.runtime.postprocessing.finalization_io import (
    FinalizationPublicationStore,
    _absolute_directory,
    _authority_relative_document,
    _bundle_identity,
    _canonical_bytes,
    _canonical_mapping,
    _deterministic_assembled_at,
    _existing_assembled_at,
    _load_runspec,
    _publish_exact_directory,
    _sha,
    _stable_file_bytes,
    _strict_tree,
    _verify_bound_artifact,
)
from bspp.orchestration.runtime.postprocessing.runtime_action_evidence import (
    _action,
    _load_prior_runtime_actions,
    validate_postprocessing_runtime_action_evidence_aggregate,
)
from bspp.orchestration.runtime.postprocessing.scientific_output_snapshot import (
    LOCAL_SCIENTIFIC_SNAPSHOT_IO,
    ScientificSnapshotIO,
    _load_staged_scientific_documents,
    _ScientificDocuments,
    _verify_source_files_descriptor,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TAR_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".tar.zst",
    ".tzst",
)
_ACCEPTANCE_STEPS = (
    "acceptance-tar-payload-parity",
    "acceptance-semantic",
    "acceptance-verify-evidence",
)
_CAPTURE_DESTINATIONS = {step: f"acceptance/captures/{step}.json" for step in _ACCEPTANCE_STEPS}
_REPORT_DESTINATIONS = {
    "acceptance-tar-payload-parity": "acceptance/reports/tar-payload-parity-report.json",
    "acceptance-semantic": "acceptance/reports/semantic-acceptance-summary.json",
    "acceptance-verify-evidence": "acceptance/reports/acceptance-evidence-report.json",
}
_ACTION09_ID = POSTPROCESSING_ACTION_IDS["acceptance-adjudication"]


@dataclass(frozen=True)
class PostprocessingFinalizationBundleResult:
    phase_run_id: str
    attempt_id: str
    destination: Path
    indexed_file_count: int
    aggregate_bytes: int


def publish_action09_finalization_bundle(
    *,
    phase_runspec_path: Path,
    execution_projection_path: Path,
    acceptance_policy_path: Path,
    command_digest: str,
    workers: int = 1,
    assembled_at: str | None = None,
    publication_store: FinalizationPublicationStore | None = None,
    snapshot_io: ScientificSnapshotIO = LOCAL_SCIENTIFIC_SNAPSHOT_IO,
) -> PostprocessingFinalizationBundleResult:
    """Verify Action 09 inputs and atomically publish the exact JSON handoff."""
    runspec = _load_runspec(phase_runspec_path)
    _sha(command_digest, "postprocessing Action 09 command digest")
    evidence_root = _absolute_directory(Path(runspec.payload.attempt_paths.evidence_dir), "evidence root", create=False)
    output_root = _absolute_directory(Path(runspec.payload.attempt_paths.output_dir), "output root", create=False)
    if output_root not in evidence_root.parents:
        raise ValueError("postprocessing evidence root must remain inside the frozen output root")
    _verify_projection(runspec, execution_projection_path)
    policy = _verify_policy(runspec, acceptance_policy_path)
    input_document = _verify_runtime_inputs(runspec, evidence_root)
    acceptance_documents = _verify_acceptance(runspec, policy, evidence_root)
    completed_actions = _load_prior_runtime_actions(runspec, evidence_root)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1 or workers > 256:
        raise ValueError("postprocessing tar manifest workers must be between 1 and 256")
    scientific = _load_staged_scientific_documents(runspec, evidence_root)
    _verify_staged_scientific_documents(evidence_root, scientific.documents)

    destination = evidence_root / "phase-finalization"
    moment = assembled_at or _existing_assembled_at(destination) or _deterministic_assembled_at(completed_actions)
    documents: dict[str, bytes] = {
        "inputs/runtime-input-attestations.json": input_document,
        **acceptance_documents,
        **scientific.documents,
    }
    documents.update(_output_handoff_documents(runspec, scientific, documents, assembled_at=moment))

    intended = tuple(
        _bundle_identity(path, document)
        for path, document in sorted(documents.items())
        if path not in {"acceptance/bundle.json", "aggregate-action-evidence.json"}
    )
    action09 = _action(runspec, _ACTION09_ID)
    witness = PostprocessingAction09AssemblyWitness(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        action_graph_digest=runspec.payload.action_graph_digest,
        action_id=action09.action_id,
        runtime_action_digest=action09.digest,
        command_digest=command_digest,
        assembled_at=moment,
        intended_members=intended,
    )
    documents["acceptance/bundle.json"] = _canonical_bytes(witness.to_mapping())
    aggregate = PostprocessingRuntimeActionEvidenceAggregate(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        action_graph_digest=runspec.payload.action_graph_digest,
        completed_actions=completed_actions,
        action09_prepublication_witness=witness,
    )
    expected_command_digests = {
        action.action_id: action.tasks[0].command_digest for action in aggregate.completed_actions
    }
    expected_command_digests[action09.action_id] = command_digest
    validate_postprocessing_runtime_action_evidence_aggregate(
        aggregate,
        runspec,
        expected_command_digests=expected_command_digests,
    )
    documents["aggregate-action-evidence.json"] = _canonical_bytes(aggregate.to_mapping())
    index = _handoff_index(runspec, documents, scientific.manifests)
    documents["handoff-index.json"] = _canonical_bytes(index.to_mapping())
    _verify_source_files_descriptor(output_root, scientific.source_files, snapshot_io=snapshot_io)
    _publish_exact_directory(destination, documents, publication_store=publication_store)
    return PostprocessingFinalizationBundleResult(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        destination=destination,
        indexed_file_count=index.declared_file_count,
        aggregate_bytes=index.declared_aggregate_bytes,
    )


def _verify_projection(runspec: ExecutablePostprocessingPhaseRunSpec, path: Path) -> None:
    document = _stable_file_bytes(path)
    projection = runspec.payload.execution_projection
    if (
        len(document) != projection.document_size_bytes
        or hashlib.sha256(document).hexdigest() != projection.document_sha256
    ):
        raise ValueError("staged legacy execution projection differs from the Phase RunSpec")


def _verify_policy(
    runspec: ExecutablePostprocessingPhaseRunSpec,
    path: Path,
) -> PostprocessingAcceptancePolicySnapshot:
    document = _stable_file_bytes(path)
    reference = runspec.payload.acceptance_policy
    if len(document) != reference.size_bytes or hashlib.sha256(document).hexdigest() != reference.sha256:
        raise ValueError("staged acceptance policy differs from the Phase RunSpec")
    payload = _canonical_mapping(document, label="staged acceptance policy")
    policy = postprocessing_acceptance_policy_from_mapping(payload)
    if policy.policy_id != reference.policy_id or policy.semantic_digest != reference.semantic_digest:
        raise ValueError("staged acceptance policy identity differs from the Phase RunSpec")
    return policy


def _verify_runtime_inputs(runspec: ExecutablePostprocessingPhaseRunSpec, evidence_root: Path) -> bytes:
    path = evidence_root / "phase-inputs/runtime-input-attestations.json"
    document = _stable_file_bytes(path)
    attestations = postprocessing_runtime_input_attestation_set_from_mapping(
        _canonical_mapping(document, label="runtime input attestations")
    )
    if (
        attestations.phase_run_id,
        attestations.attempt_id,
        attestations.phase_runspec_digest,
    ) != (runspec.phase_run_id, runspec.attempt_id, runspec.digest):
        raise ValueError("runtime input attestations differ from the current RunSpec")
    _reconcile_input_attestations(runspec, attestations, evidence_root)
    return document


def _reconcile_input_attestations(
    runspec: ExecutablePostprocessingPhaseRunSpec,
    attestations: PostprocessingRuntimeInputAttestationSet,
    evidence_root: Path,
) -> None:
    logical = {item.name: item for item in runspec.payload.logical_inputs.entries}
    observed = {item.logical_input_name: item for item in attestations.attestations}
    physical = {item.name: item for item in runspec.payload.physical_inputs}
    qualified = runspec.payload.qualified_runtime
    runtime_attestation = attestations.runtime_qualification
    if (
        runtime_attestation.qualified_runtime_digest,
        runtime_attestation.record_location,
        runtime_attestation.record_sha256,
        runtime_attestation.record_size_bytes,
        runtime_attestation.tuple_id,
        runtime_attestation.source_identity_digest,
        runtime_attestation.source_package_identity_digest,
        runtime_attestation.toolkit_identity_digest,
        runtime_attestation.runtime_component_identity_digest,
    ) != (
        qualified.digest,
        qualified.qualification_location,
        qualified.qualification_sha256,
        qualified.qualification_size_bytes,
        qualified.tuple_id,
        qualified.source_identity_digest,
        qualified.source_package_identity_digest,
        qualified.toolkit_identity_digest,
        qualified.runtime_component_identity_digest,
    ):
        raise ValueError("runtime qualification attestation differs from the current Attempt")
    if set(observed) != set(logical):
        raise ValueError("runtime input attestations omit or add logical inputs")
    for name, expected in logical.items():
        item = observed[name]
        if item.verification_kind != "authority-declared-content-v1" or item.authority != expected.member_identity:
            raise ValueError(f"runtime input attestation differs for {name!r}")
        physical_name = name
        locator = physical.get(physical_name)
        if (
            locator is None
            or item.physical_input_name != physical_name
            or item.physical_locator != locator.locator
            or item.member_identity != expected.member_identity
            or (item.content_sha256, item.size_bytes)
            != (expected.expected_content_sha256, expected.expected_size_bytes)
        ):
            raise ValueError(f"runtime declared input attestation differs for {name!r}")
        proof_path = item.accessibility_evidence_path
        proof_sha = item.accessibility_evidence_sha256
        proof_size = item.accessibility_evidence_size_bytes
        if proof_path is None or proof_sha is None or proof_size is None:
            raise ValueError(f"runtime input accessibility proof is incomplete for {name!r}")
        proof = _authority_relative_document(evidence_root, proof_path, label=f"runtime input proof {name!r}")
        if len(proof) != proof_size or hashlib.sha256(proof).hexdigest() != proof_sha:
            raise ValueError(f"runtime input accessibility proof changed for {name!r}")


def _verify_acceptance(
    runspec: ExecutablePostprocessingPhaseRunSpec,
    policy: PostprocessingAcceptancePolicySnapshot,
    evidence_root: Path,
) -> dict[str, bytes]:
    documents: dict[str, bytes] = {}
    captures: list[PostprocessingAcceptanceCapture] = []
    contract_by_step: dict[str, PostprocessingCompletionExitContract] = {
        item.step_name: item for item in policy.completion_exit_contracts
    }
    action_by_step = {item.step_name: item for item in runspec.payload.actions}
    for step in _ACCEPTANCE_STEPS:
        source = evidence_root / "phase-acceptance" / f"{step}-capture.json"
        document = _stable_file_bytes(source)
        capture = postprocessing_acceptance_capture_from_mapping(
            _canonical_mapping(document, label=f"acceptance capture {step}")
        )
        contract = contract_by_step.get(step)
        action = action_by_step.get(step)
        if (
            contract is None
            or action is None
            or capture.phase_run_id != runspec.phase_run_id
            or capture.attempt_id != runspec.attempt_id
            or capture.action_id != action.action_id
            or capture.step_name != step
            or capture.policy_sha256 != runspec.payload.acceptance_policy.sha256
            or capture.raw_exit_code not in contract.allowed_raw_exit_codes
            or len(capture.reports) != 1
            or tuple(item.path for item in capture.reports) != contract.report_paths
        ):
            raise ValueError(f"acceptance capture differs from RunSpec/policy: {step}")
        _verify_bound_artifact(evidence_root, capture.raw_stdout, label=f"{step} stdout")
        _verify_bound_artifact(evidence_root, capture.raw_stderr, label=f"{step} stderr")
        report_binding = capture.reports[0]
        report = _authority_relative_document(evidence_root, report_binding.path, label=f"{step} report")
        _canonical_mapping(report, label=f"{step} report")
        if len(report) != report_binding.size_bytes or hashlib.sha256(report).hexdigest() != report_binding.sha256:
            raise ValueError(f"captured acceptance report changed: {step}")
        documents[_CAPTURE_DESTINATIONS[step]] = document
        documents[_REPORT_DESTINATIONS[step]] = report
        captures.append(capture)

    adjudication_path = evidence_root / "phase-acceptance/adjudication.json"
    adjudication_document = _stable_file_bytes(adjudication_path)
    adjudication = postprocessing_acceptance_adjudication_from_mapping(
        _canonical_mapping(adjudication_document, label="acceptance adjudication")
    )
    _reconcile_adjudication(runspec, policy, tuple(captures), adjudication)
    documents["acceptance/adjudication.json"] = adjudication_document
    return documents


def _reconcile_adjudication(
    runspec: ExecutablePostprocessingPhaseRunSpec,
    policy: PostprocessingAcceptancePolicySnapshot,
    captures: tuple[PostprocessingAcceptanceCapture, ...],
    adjudication: PostprocessingAcceptanceAdjudication,
) -> None:
    if (
        adjudication.phase_run_id != runspec.phase_run_id
        or adjudication.attempt_id != runspec.attempt_id
        or adjudication.policy_id != policy.policy_id
        or adjudication.policy_sha256 != runspec.payload.acceptance_policy.sha256
        or adjudication.capture_digests != tuple(item.digest for item in captures)
        or adjudication.result != "passed"
    ):
        raise ValueError("acceptance adjudication is not the exact passed Action 09 input")


def _output_handoff_documents(
    runspec: ExecutablePostprocessingPhaseRunSpec,
    scientific: _ScientificDocuments,
    documents: Mapping[str, bytes],
    *,
    assembled_at: str,
) -> dict[str, bytes]:
    evidence_root = Path(runspec.payload.attempt_paths.evidence_dir)
    output_root = Path(runspec.payload.attempt_paths.output_dir)
    evidence_relative = evidence_root.relative_to(output_root).as_posix()
    acceptance_by_path: dict[str, PostprocessingEvidenceArtifact] = {}
    for step in _ACCEPTANCE_STEPS:
        capture_document = documents[_CAPTURE_DESTINATIONS[step]]
        capture = postprocessing_acceptance_capture_from_mapping(
            _canonical_mapping(capture_document, label=f"acceptance capture {step}")
        )
        capture_path = f"phase-acceptance/{step}-capture.json"
        acceptance_by_path[capture_path] = PostprocessingEvidenceArtifact(
            path=capture_path,
            sha256=hashlib.sha256(capture_document).hexdigest(),
            size_bytes=len(capture_document),
        )
        for binding in (capture.raw_stdout, capture.raw_stderr):
            acceptance_by_path[binding.path] = PostprocessingEvidenceArtifact(
                path=binding.path,
                sha256=binding.sha256,
                size_bytes=binding.size_bytes,
            )
        report_binding = capture.reports[0]
        report_document = documents[_REPORT_DESTINATIONS[step]]
        if (
            hashlib.sha256(report_document).hexdigest() != report_binding.sha256
            or len(report_document) != report_binding.size_bytes
        ):
            raise ValueError(f"bundled acceptance report identity changed: {step}")
        acceptance_by_path[report_binding.path] = PostprocessingEvidenceArtifact(
            path=report_binding.path,
            sha256=report_binding.sha256,
            size_bytes=report_binding.size_bytes,
        )
    adjudication_document = documents["acceptance/adjudication.json"]
    adjudication_path = "phase-acceptance/adjudication.json"
    acceptance_by_path[adjudication_path] = PostprocessingEvidenceArtifact(
        path=adjudication_path,
        sha256=hashlib.sha256(adjudication_document).hexdigest(),
        size_bytes=len(adjudication_document),
    )
    acceptance_members = tuple(acceptance_by_path[path] for path in sorted(acceptance_by_path))
    manifest_by_id = {item.manifest_id: item for item in scientific.manifests}
    scientific_members: list[PostprocessingEvidenceArtifact] = [
        PostprocessingEvidenceArtifact(path=item.path, sha256=item.sha256, size_bytes=item.size_bytes)
        for item in scientific.root.small_outputs
    ]
    for reference in scientific.root.tar_manifests:
        manifest = manifest_by_id[reference.manifest_id]
        scientific_members.append(
            PostprocessingEvidenceArtifact(
                path=manifest.tar_path,
                sha256=manifest.manifest_id,
                size_bytes=sum(item.size_bytes for item in manifest.members),
                verification_kind="inventory-metadata-v1",
                verification_source_path=reference.path,
            )
        )
    acceptance_set = _artifact_set("acceptance-evidence", members=acceptance_members)
    scientific_set = _artifact_set(
        "scientific-output",
        members=tuple(sorted(scientific_members, key=lambda item: item.path)),
    )
    root_set = _artifact_set("postprocessing-output", children=(acceptance_set, scientific_set))
    physical_by_logical: dict[str, str] = {}
    for item in acceptance_set.members:
        physical_by_logical[f"postprocessing-output/acceptance-evidence/{item.path}"] = (
            f"{evidence_relative}/{item.path}"
        )
    for item in scientific_set.members:
        physical_by_logical[f"postprocessing-output/scientific-output/{item.path}"] = item.path
    placements = tuple(
        PostprocessingArtifactPlacement(
            logical_path=item.path,
            physical_path=physical_by_logical[item.path],
            sha256=item.sha256,
            size_bytes=item.size_bytes,
            verification_kind=item.verification_kind,
            verification_source_path=item.verification_source_path,
        )
        for item in root_set.flattened_members
    )
    placements_digest = canonical_mapping_digest(
        {"schema_version": CURRENT_CONTRACT_SCHEMA_VERSION, "placements": [item.to_mapping() for item in placements]}
    )
    location_identity: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "location_kind": "verified-local-output-tree-v1",
        "artifact_set_id": root_set.artifact_set_id,
        "root": str(output_root),
        "verified_members_digest": placements_digest,
        "verified_at": assembled_at,
        "placements": [item.to_mapping() for item in placements],
    }
    location = PostprocessingVerifiedArtifactLocation(
        artifact_location_id="postprocessing-artifact-location-" + canonical_mapping_digest(location_identity),
        artifact_set_id=root_set.artifact_set_id,
        root=str(output_root),
        verified_members_digest=placements_digest,
        verified_at=assembled_at,
        placements=placements,
        location_kind="verified-local-output-tree-v1",
    )
    identity = runspec.payload.execution_projection.phase_identity
    handoff_identity: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "handoff_kind": "postprocessing-output-handoff-v1",
        "phase_run_id": runspec.phase_run_id,
        "attempt_id": runspec.attempt_id,
        "phase_runspec_digest": runspec.digest,
        "legacy_run_id": runspec.payload.attempt_paths.legacy_run_id,
        "output_namespace": identity.output_namespace,
        "output_dir": str(output_root),
        "object_prefix": runspec.payload.attempt_paths.object_prefix,
        "artifact_set": root_set.to_mapping(),
        "physical_locations": [location.to_mapping()],
    }
    handoff = PostprocessingOutputHandoff(
        handoff_id=postprocessing_handoff_id(handoff_identity),
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        legacy_run_id=runspec.payload.attempt_paths.legacy_run_id,
        output_namespace=identity.output_namespace,
        output_dir=str(output_root),
        object_prefix=runspec.payload.attempt_paths.object_prefix,
        artifact_set=root_set,
        physical_locations=(location,),
    )
    locations = PostprocessingArtifactLocationSet(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        artifact_set_id=root_set.artifact_set_id,
        locations=(location,),
    )
    return {
        "outputs/artifact-set-root.json": _canonical_bytes(root_set.to_mapping()),
        "outputs/artifact-locations.json": _canonical_bytes(locations.to_mapping()),
        "outputs/output-handoff.json": _canonical_bytes(handoff.to_mapping()),
    }


def _artifact_set(
    root_name: str,
    *,
    members: tuple[PostprocessingEvidenceArtifact, ...] = (),
    children: tuple[PostprocessingLogicalArtifactSet, ...] = (),
) -> PostprocessingLogicalArtifactSet:
    identity = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "artifact_set_kind": "postprocessing-logical-artifact-set-v1",
        "root_name": root_name,
        "members": [item.to_mapping() for item in members],
        "children": [item.to_mapping() for item in children],
    }
    return PostprocessingLogicalArtifactSet(
        artifact_set_id=postprocessing_artifact_set_id(identity),
        root_name=root_name,
        members=members,
        children=children,
    )


def _handoff_index(
    runspec: ExecutablePostprocessingPhaseRunSpec,
    documents: Mapping[str, bytes],
    manifests: tuple[PostprocessingTarManifest, ...],
) -> PostprocessingFinalizationHandoffIndex:
    expected = set(POSTPROCESSING_FINALIZATION_FIXED_PATHS) | {
        f"outputs/tar-manifests/{item.manifest_id}.json" for item in manifests
    }
    if set(documents) != expected:
        raise ValueError("postprocessing Runtime finalization documents differ from the fixed layout")
    members = tuple(_bundle_identity(path, document) for path, document in sorted(documents.items()))
    counts = tuple(sorted((f"outputs/tar-manifests/{item.manifest_id}.json", len(item.members)) for item in manifests))
    return PostprocessingFinalizationHandoffIndex(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        action_graph_digest=runspec.payload.action_graph_digest,
        execution_projection_sha256=runspec.payload.execution_projection.document_sha256,
        acceptance_policy_sha256=runspec.payload.acceptance_policy.sha256,
        members=members,
        tar_manifest_member_counts=counts,
        declared_file_count=len(members),
        declared_aggregate_bytes=sum(item.size_bytes for item in members),
        declared_tar_manifest_count=len(manifests),
        declared_total_tar_members=sum(len(item.members) for item in manifests),
    )


def _verify_staged_scientific_documents(evidence_root: Path, expected: Mapping[str, bytes]) -> None:
    root = evidence_root / "phase-output"
    source_expected = {path.removeprefix("outputs/"): document for path, document in expected.items()}
    observed_files, observed_directories = _strict_tree(root)
    expected_directories = {"tar-manifests"}
    if observed_files != {*source_expected, "source-fingerprints.json"} or observed_directories != expected_directories:
        raise ValueError("staged scientific output root/manifests are missing, extra, or unsafe")
    for relative, document in source_expected.items():
        observed = _stable_file_bytes(root / relative)
        _canonical_mapping(observed, label=f"staged scientific document {relative}")
        if observed != document:
            raise ValueError("current scientific output fingerprints differ from the staged scientific root")
