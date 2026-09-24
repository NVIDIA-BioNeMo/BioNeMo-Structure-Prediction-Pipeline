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

"""Folding Phase adapter: materialization, rendering, and evidence validation.

This facade turns a ``FoldingPhasePlan`` into an immutable
``FoldingPhaseRunSpec`` carrying the exact 5-action graph
``msa-flatten -> split -> preprocess -> fold -> canonical-pair``, renders the
per-action Slurm submission intent with the kernel image selected by backend
(MFI-008), and strictly validates folding action evidence — including building
and publishing the completed run's canonical-pair index on the canonical-pair
action.

The adapter is control-pure: it imports no
``bspp.orchestration.runtime.*`` module.  Reuse of the delivered Track A/B/C
modules is expressed by naming the exact action kinds/params in the
materialized RunSpec.  Each per-action script invokes the job-local Runtime
folding executor through one deterministic, safely quoted argv string
(``python -m bspp.orchestration.runtime.folding.executor ...``); the executor
is named as a string literal and never imported, which keeps the AST/import
boundary scan green because strings are not ``import`` nodes.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from bspp.orchestration.contract.folding_carry_forward import FoldingCarryForwardRecord
from bspp.orchestration.contract.folding_release import FoldingReleasePreset
from bspp.orchestration.contract.folding_shard import FOLD_SHARD_PROJECTION_FILENAME
from bspp.orchestration.contract.phase import (
    FOLDING_BACKENDS,
    FoldingActionPayload,
    FoldingPhasePlan,
    FoldingPhaseRunSpec,
    FoldingPhaseRunSpecPayload,
    FoldingResolvedClusterSnapshot,
    FoldingRuntimeAction,
    FoldingRuntimeActionKind,
    PhaseSlurmResources,
    canonical_mapping_digest,
    validate_phase_attempt_id,
    validate_phase_run_id,
)
from bspp.orchestration.contract.phase_submission import (
    PhaseActionSubmissionPlan,
    PhaseContainerMountDescriptor,
    PhaseSubmissionIntendedPayload,
    phase_action_scheduler_correlation_token,
    phase_action_submission_identity_mapping,
    phase_submission_id,
)
from bspp.orchestration.contract.prediction_pair import PredictionPair
from bspp.orchestration.contract.preprocessing_handoff import VerifiedLocalBundledArtifactLocation
from bspp.orchestration.control.folding_phase_types import (
    CanonicalPairActionEvidence,
    CanonicalPairIndex,
    FoldActionEvidence,
    FoldingBackendImageSelection,
    FoldingPhaseAttemptOperationalSelection,
    MsaFlattenActionEvidence,
    PreprocessActionEvidence,
    SplitActionEvidence,
    build_canonical_pair_index,
    folding_qualification_tuple_id,
    write_canonical_pair_index,
)
from bspp.orchestration.control.folding_shard import derive_fold_shard_projection

_SHA256 = re.compile(r"[0-9a-f]{64}")

_MISSING_MSA_SET_MANIFEST_ERROR = (
    "folding executable authority requires the root MSA set manifest; "
    "regenerate a complete Phase Plan rather than repairing immutable authority"
)
_LEGACY_MSA_SET_MANIFEST_ERROR = (
    "legacy MSA manifest requires Runtime length enrichment before folding; "
    "run the legacy-MSA import operation to publish an enriched manifest"
)
_MISSING_BACKEND_ASSETS_ERROR = (
    "folding submission requires resolved backend assets; "
    "regenerate a complete Phase Plan rather than repairing immutable authority"
)

_ACTION_SPECS: tuple[tuple[str, FoldingRuntimeActionKind, tuple[str, ...], tuple[tuple[str, str], ...]], ...] = (
    (
        "msa-flatten-000001",
        "msa-flatten",
        (),
        (("runtime_module", "msa_projection"), ("msa_preparation", "prepare_projected_msa")),
    ),
    ("split-000001", "split", ("msa-flatten-000001",), (("runtime_module", "a3m_split"),)),
    ("preprocess-000001", "preprocess", ("split-000001",), (("runtime_module", "__PREPROCESS_MODULE__"),)),
    (
        "fold-000001",
        "fold",
        ("preprocess-000001",),
        (("backend", "__BACKEND__"), ("kernel_image", "__KERNEL_IMAGE__")),
    ),
    ("canonical-pair-000001", "canonical-pair", ("fold-000001",), (("runtime_module", "emitter_support"),)),
)


def _executor_argv(
    *,
    phase_runspec_path: str,
    action_id: str,
    action_evidence_path: str,
    handoff_path: str,
    carry_record_path: str | None = None,
) -> str:
    """Return the deterministic job-local Runtime executor argv for one action.

    The argv names the delivered Runtime folding executor as a string literal —
    control-pure: the executor is never imported. Flags are
    literal; only their values are ``shlex.quote``-quoted so the rendered Slurm
    script runs as a single ``srun`` command without shell interpolation. The
    optional ``--carry-record`` flag is emitted only for carried fold and
    canonical-pair actions.
    """
    argv = (
        "python -m bspp.orchestration.runtime.folding.executor"
        f" --phase-runspec {shlex.quote(phase_runspec_path)}"
        f" --action-id {shlex.quote(action_id)}"
        f" --action-evidence {shlex.quote(action_evidence_path)}"
        f" --handoff {shlex.quote(handoff_path)}"
    )
    if carry_record_path is not None:
        argv += f" --carry-record {shlex.quote(carry_record_path)}"
    return argv


def materialize_folding_attempt_runspec(
    *,
    phase_run_id: str,
    attempt_id: str,
    phase_plan: FoldingPhasePlan,
    materialized_at: str,
    operational: FoldingPhaseAttemptOperationalSelection,
) -> FoldingPhaseRunSpec:
    """Construct one complete candidate folding RunSpec from Plan plus named authority."""
    validate_phase_run_id(phase_run_id)
    validate_phase_attempt_id(attempt_id)
    backend = phase_plan.payload.backend
    if backend not in FOLDING_BACKENDS:
        raise ValueError(f"unsupported folding backend: {backend!r}")
    if phase_plan.payload.msa_set_manifest is None:
        raise ValueError(_MISSING_MSA_SET_MANIFEST_ERROR)
    if not phase_plan.payload.msa_set_manifest.has_member_lengths():
        raise ValueError(_LEGACY_MSA_SET_MANIFEST_ERROR)
    if operational.assets is None:
        raise ValueError(f"folding backend {backend!r} requires resolved backend assets")
    if operational.assets.backend != backend:
        msg = (
            f"folding backend assets backend {operational.assets.backend!r} "
            f"does not match Phase Plan backend {backend!r}"
        )
        raise ValueError(msg)
    kernel_image = operational.backend_images.image_for_backend(backend)
    actions = _build_folding_actions(phase_plan, operational, kernel_image)
    # The fold action's typed topology (worker_count = nodes * tasks_per_node)
    # drives the canonical LPT projection bound on the RunSpec payload.
    fold_resources = operational.fold_resources or operational.resources
    _projection, binding = derive_fold_shard_projection(phase_plan, fold_resources, attempt_id)
    cluster = _folding_cluster_snapshot(operational)
    return FoldingPhaseRunSpec(
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        phase_plan_digest=phase_plan.digest,
        materialized_at=materialized_at,
        input_location=phase_plan.input_location,
        cluster=cluster,
        payload=FoldingPhaseRunSpecPayload(
            msa_set=phase_plan.payload.msa_set,
            backend=backend,
            actions=tuple(actions),
            msa_set_manifest=phase_plan.payload.msa_set_manifest,
            fold_shard_projection=binding,
            transport=phase_plan.payload.transport,
            s3_prediction_prefix=phase_plan.payload.s3_prediction_prefix,
            bioir_model_policy=phase_plan.payload.bioir_model_policy,
            evidence_profile=phase_plan.payload.evidence_profile,
        ),
    )


def _build_folding_actions(
    phase_plan: FoldingPhasePlan,
    operational: FoldingPhaseAttemptOperationalSelection,
    kernel_image: str,
) -> list[FoldingRuntimeAction]:
    backend = phase_plan.payload.backend
    preprocess_module = _preprocess_runtime_module(backend)
    fold_resources = operational.fold_resources or operational.resources
    actions: list[FoldingRuntimeAction] = []
    for action_id, kind, dependencies, params_template in _ACTION_SPECS:
        params = tuple(
            (
                key,
                _substitute_param(
                    value,
                    backend=backend,
                    kernel_image=kernel_image,
                    preprocess_module=preprocess_module,
                ),
            )
            for key, value in params_template
        )
        if phase_plan.payload.bioir_model_policy is not None and kind in {"fold", "canonical-pair"}:
            params += (("bioir_model_policy_digest", phase_plan.payload.bioir_model_policy.digest),)
        if phase_plan.payload.evidence_profile is not None and kind in {"fold", "canonical-pair"}:
            params += (("evidence_profile", phase_plan.payload.evidence_profile),)
        actions.append(
            FoldingRuntimeAction(
                action_id=action_id,
                dependencies=dependencies,
                resources=fold_resources if kind == "fold" else operational.resources,
                payload=FoldingActionPayload(action_kind=kind, params=params),
                action_kind=kind,
            )
        )
    return actions


def _substitute_param(
    value: str,
    *,
    backend: str,
    kernel_image: str,
    preprocess_module: str,
) -> str:
    if value == "__BACKEND__":
        return backend
    if value == "__KERNEL_IMAGE__":
        return kernel_image
    if value == "__PREPROCESS_MODULE__":
        return preprocess_module
    return value


def _preprocess_runtime_module(backend: str) -> str:
    if backend == "colabfold":
        return "colabfold_inputs"
    if backend == "openfold-cli":
        return "openfold_inputs"
    if backend in {"bioir", "openfold-trt"}:
        return "bioir_inputs"
    raise ValueError(f"unsupported folding backend: {backend!r}")


def _folding_cluster_snapshot(
    operational: FoldingPhaseAttemptOperationalSelection,
) -> FoldingResolvedClusterSnapshot:
    return FoldingResolvedClusterSnapshot(
        profile_name=operational.profile_name,
        owner=operational.owner,
        transport=operational.transport,
        ssh_target=operational.ssh_target,
        account=operational.account,
        project_root=operational.project_root,
        staging_root=operational.staging_root,
        orchestration_repo=operational.orchestration_repo,
        runtime_image=operational.runtime_image,
        extra_mounts=operational.extra_mounts,
        backend_assets=operational.assets,
        release_preset=operational.release_preset.value,
        mount_orchestration_source=operational.mount_orchestration_source,
    )


def validate_folding_plan_runspec_binding(
    phase_plan: FoldingPhasePlan,
    phase_runspec: FoldingPhaseRunSpec,
) -> None:
    """Reject a folding RunSpec that is not the adapter's canonical Plan derivation.

    The generic authority path already proves the stored RunSpec is a valid DAG
    and that its digests bind the Plan. This guard additionally proves the
    stored five-action graph is exactly the adapter-generated graph for the
    Plan's backend and MSA set — not merely an arbitrary valid DAG. It rebuilds
    the canonical actions from the RunSpec's own independently resolved
    operational selections (cluster snapshot, the canonical control action's
    resources, the canonical fold action's resources, and the fold action's
    selected kernel image) and compares that tuple to the stored one, so
    legitimate attempt-specific operational selections survive while backend,
    MSA-set, action-kind, dependency, static-parameter, preprocessing-module,
    and fold-backend drift is rejected. A snapshotted backend-assets record
    tagged for a different backend than the payload's is likewise rejected.
    """
    if phase_plan.payload.backend != phase_runspec.payload.backend:
        msg = (
            "folding Phase RunSpec backend does not bind its Phase Plan backend: "
            f"{phase_runspec.payload.backend!r} != {phase_plan.payload.backend!r}"
        )
        raise ValueError(msg)
    if phase_plan.payload.evidence_profile != phase_runspec.payload.evidence_profile:
        raise ValueError("folding Phase RunSpec evidence profile does not bind its Phase Plan")
    if phase_plan.payload.bioir_model_policy != phase_runspec.payload.bioir_model_policy:
        raise ValueError("folding Phase RunSpec BioIR model policy does not bind its Phase Plan policy")
    if phase_plan.payload.msa_set != phase_runspec.payload.msa_set:
        msg = "folding Phase RunSpec msa_set does not bind its Phase Plan msa_set"
        raise ValueError(msg)
    if phase_plan.payload.msa_set_manifest != phase_runspec.payload.msa_set_manifest:
        msg = "folding Phase RunSpec msa_set_manifest does not bind its Phase Plan msa_set_manifest"
        raise ValueError(msg)
    if phase_plan.payload.transport != phase_runspec.payload.transport:
        msg = "folding Phase RunSpec transport does not bind its Phase Plan transport"
        raise ValueError(msg)
    if phase_plan.payload.s3_prediction_prefix != phase_runspec.payload.s3_prediction_prefix:
        msg = "folding Phase RunSpec s3_prediction_prefix does not bind its Phase Plan prefix"
        raise ValueError(msg)
    cluster_assets = phase_runspec.cluster.backend_assets
    if cluster_assets is not None and cluster_assets.backend != phase_runspec.payload.backend:
        msg = (
            "folding Phase RunSpec backend assets do not bind the selected backend: "
            f"{cluster_assets.backend!r} != {phase_runspec.payload.backend!r}"
        )
        raise ValueError(msg)

    control_action = _single_folding_action(phase_runspec, "msa-flatten")
    fold_action = _single_folding_action(phase_runspec, "fold")
    kernel_image = _kernel_image_for_action(phase_runspec, fold_action)
    operational = FoldingPhaseAttemptOperationalSelection(
        profile_name=phase_runspec.cluster.profile_name,
        owner=phase_runspec.cluster.owner,
        transport=phase_runspec.cluster.transport,
        ssh_target=phase_runspec.cluster.ssh_target,
        account=phase_runspec.cluster.account,
        project_root=phase_runspec.cluster.project_root,
        staging_root=phase_runspec.cluster.staging_root,
        orchestration_repo=phase_runspec.cluster.orchestration_repo,
        runtime_image=phase_runspec.cluster.runtime_image,
        backend_images=FoldingBackendImageSelection(backend_images={phase_runspec.payload.backend: kernel_image}),
        release_preset=FoldingReleasePreset(phase_runspec.cluster.release_preset),
        resources=control_action.resources,
        fold_resources=fold_action.resources,
        extra_mounts=phase_runspec.cluster.extra_mounts,
        assets=phase_runspec.cluster.backend_assets,
        mount_orchestration_source=phase_runspec.cluster.mount_orchestration_source,
    )
    expected = tuple(_build_folding_actions(phase_plan, operational, kernel_image))
    if expected != phase_runspec.payload.actions:
        msg = "folding Phase RunSpec action graph is not the canonical Plan-derived graph"
        raise ValueError(msg)

    manifest = phase_plan.payload.msa_set_manifest
    if manifest is not None and manifest.has_member_lengths():
        _expected_projection, expected_binding = derive_fold_shard_projection(
            phase_plan, fold_action.resources, phase_runspec.attempt_id
        )
        if phase_runspec.payload.fold_shard_projection != expected_binding:
            msg = "folding Phase RunSpec fold shard projection binding is not the canonical Plan-derived projection"
            raise ValueError(msg)
    elif phase_runspec.payload.fold_shard_projection is not None:
        raise ValueError("folding Phase RunSpec fold shard projection binding requires an enriched MSA manifest")


def _single_folding_action(
    phase_runspec: FoldingPhaseRunSpec,
    action_kind: FoldingRuntimeActionKind,
) -> FoldingRuntimeAction:
    matches = [action for action in phase_runspec.payload.actions if action.action_kind == action_kind]
    if len(matches) != 1:
        msg = f"folding Phase RunSpec must contain exactly one {action_kind!r} Runtime Action"
        raise ValueError(msg)
    return matches[0]


def render_folding_submission_intent(
    *,
    phase_runspec: FoldingPhaseRunSpec,
    phase_runspec_location: str,
    phase_runspec_document_sha256: str,
    carry_forward_record: FoldingCarryForwardRecord | None = None,
    carry_forward_document_sha256: str | None = None,
) -> PhaseSubmissionIntendedPayload:
    """Render every declared folding action without filesystem, clock, or scheduler access."""
    if _SHA256.fullmatch(phase_runspec_document_sha256) is None:
        raise ValueError("Phase RunSpec document SHA-256 must be lowercase SHA-256")
    if not phase_runspec_location:
        raise ValueError("Phase RunSpec location must be non-empty")
    # Executable-authority guards: historical RunSpecs remain
    # loadable for inspection, but they cannot be submitted without a root MSA
    # set manifest and a resolved backend-assets snapshot.
    if phase_runspec.payload.msa_set_manifest is None:
        raise ValueError(_MISSING_MSA_SET_MANIFEST_ERROR)
    if phase_runspec.cluster.backend_assets is None:
        raise ValueError(_MISSING_BACKEND_ASSETS_ERROR)
    if (carry_forward_record is None) != (carry_forward_document_sha256 is None):
        raise ValueError("folding carry record and document SHA-256 must be present together")
    if carry_forward_record is not None:
        reference = phase_runspec.carry_forward
        if (
            reference is None
            or reference.folding_carry_forward_id != carry_forward_record.folding_carry_forward_id
            or reference.digest != carry_forward_record.digest
            or carry_forward_record.phase_run_id != phase_runspec.phase_run_id
            or carry_forward_record.target_attempt_id != phase_runspec.attempt_id
        ):
            raise ValueError("carried folding rendering requires the exact RunSpec-referenced carry record")
        if carry_forward_document_sha256 is None or _SHA256.fullmatch(carry_forward_document_sha256) is None:
            raise ValueError("carry record document SHA-256 must be lowercase SHA-256")
    elif phase_runspec.carry_forward is not None:
        raise ValueError("carried folding RunSpec rendering requires its complete carry record")
    cluster_runspec_path = str(
        PurePosixPath(phase_runspec.cluster.staging_root)
        / "bspp-phase-runs"
        / phase_runspec.phase_run_id
        / phase_runspec.attempt_id
        / "phase-runspec.json"
    )
    cluster_projection_path = str(PurePosixPath(cluster_runspec_path).parent / FOLD_SHARD_PROJECTION_FILENAME)
    fold_action = next(action for action in phase_runspec.payload.actions if action.action_kind == "fold")
    packed = fold_action.resources.is_packed
    cluster_carry_record_path: str | None = None
    carry_mounts: tuple[PhaseContainerMountDescriptor, ...] = ()
    if carry_forward_record is not None:
        cluster_carry_record_path = str(
            PurePosixPath(phase_runspec.cluster.staging_root)
            / "bspp-phase-runs"
            / phase_runspec.phase_run_id
            / phase_runspec.attempt_id
            / "folding-carry-forward.json"
        )
        carry_mounts = _folding_carry_mount_descriptors(carry_forward_record, cluster_carry_record_path)
    identities: list[dict[str, object]] = []
    attempt_root = (
        PurePosixPath(phase_runspec.cluster.project_root)
        / "bspp-phase-runs"
        / phase_runspec.phase_run_id
        / phase_runspec.attempt_id
    )
    for action in phase_runspec.payload.actions:
        # Action outputs, evidence, and the handoff all live beneath
        # <attempt-root>/actions/<action-id>/ — the same action root the job-local
        # executor derives from --handoff and confines --action-evidence to.
        action_root = attempt_root / "actions" / action.action_id
        cluster_script_path = str(
            PurePosixPath(phase_runspec.cluster.staging_root)
            / "bspp-phase-runs"
            / phase_runspec.phase_run_id
            / phase_runspec.attempt_id
            / "actions"
            / f"{action.action_id}.sbatch"
        )
        carries = _action_carries(action) and carry_forward_record is not None
        identities.append(
            phase_action_submission_identity_mapping(
                action_id=action.action_id,
                runtime_action_digest=canonical_mapping_digest(action.to_mapping()),
                dependency_action_ids=action.dependencies,
                cluster_script_path=cluster_script_path,
                action_evidence_path=str(action_root / "action-evidence.json"),
                handoff_path=str(attempt_root / "actions" / action.action_id / "handoff.json"),
                carry_forward_record_path=cluster_carry_record_path if carries else None,
                carry_forward_record_sha256=carry_forward_document_sha256 if carries else None,
                carry_forward_mounts=carry_mounts if carries else (),
            )
        )
    fold_action = next(action for action in phase_runspec.payload.actions if action.action_kind == "fold")
    fold_kernel_image = _kernel_image_for_action(phase_runspec, fold_action)
    qualification_tuple_id = folding_qualification_tuple_id(
        backend=phase_runspec.payload.backend,
        kernel_image=fold_kernel_image,
        cluster_snapshot_digest=canonical_mapping_digest(phase_runspec.cluster.to_mapping()),
    )
    submission_id = phase_submission_id(
        phase_run_id=phase_runspec.phase_run_id,
        attempt_id=phase_runspec.attempt_id,
        phase_runspec_digest=phase_runspec.digest,
        phase_runspec_document_sha256=phase_runspec_document_sha256,
        qualification_tuple_id=qualification_tuple_id,
        actions=identities,
    )
    plans: list[PhaseActionSubmissionPlan] = []
    for action, identity in zip(phase_runspec.payload.actions, identities, strict=True):
        carries = _action_carries(action) and carry_forward_record is not None
        kernel_image = _kernel_image_for_action(phase_runspec, action)
        correlation = phase_action_scheduler_correlation_token(submission_id, action.action_id)
        job_name = correlation
        script = _render_action_script(
            phase_runspec,
            action=action,
            cluster_runspec_path=cluster_runspec_path,
            runspec_document_sha256=phase_runspec_document_sha256,
            attempt_root=attempt_root,
            action_evidence_path=str(identity["action_evidence_path"]),
            handoff_path=str(identity["handoff_path"]),
            job_name=job_name,
            correlation=correlation,
            kernel_image=kernel_image,
            carry_record_path=cluster_carry_record_path if carries else None,
            carry_mounts=carry_mounts if carries else (),
            projection_mounts=_fold_projection_mount_descriptors(
                action,
                packed=packed,
                cluster_projection_path=cluster_projection_path,
            ),
        )
        plans.append(
            PhaseActionSubmissionPlan(
                action_id=action.action_id,
                runtime_action_digest=str(identity["runtime_action_digest"]),
                dependency_action_ids=action.dependencies,
                cluster_script_path=str(identity["cluster_script_path"]),
                script_sha256=hashlib.sha256(script.encode()).hexdigest(),
                script_body=script,
                job_name=job_name,
                scheduler_correlation_token=correlation,
                action_evidence_path=str(identity["action_evidence_path"]),
                handoff_path=str(identity["handoff_path"]),
                carry_forward_record_path=cluster_carry_record_path if carries else None,
                carry_forward_record_sha256=carry_forward_document_sha256 if carries else None,
                carry_forward_mounts=carry_mounts if carries else (),
                carry_forward_submission_id=submission_id if carries else None,
                expected_task_indexes=(
                    _fold_expected_task_indexes(action.resources) if action.action_kind == "fold" else ()
                ),
            )
        )
    return PhaseSubmissionIntendedPayload(
        submission_id=submission_id,
        phase_runspec_location=phase_runspec_location,
        phase_runspec_digest=phase_runspec.digest,
        phase_runspec_document_sha256=phase_runspec_document_sha256,
        qualification_tuple_id=qualification_tuple_id,
        actions=tuple(plans),
    )


def _kernel_image_for_action(phase_runspec: FoldingPhaseRunSpec, action: FoldingRuntimeAction) -> str:
    if action.action_kind != "fold":
        return phase_runspec.cluster.runtime_image
    for key, value in action.payload.params:
        if key == "kernel_image":
            return value
    raise ValueError("fold Runtime Action is missing its selected kernel image")


def _action_carries(action: FoldingRuntimeAction) -> bool:
    """Only the fold and canonical-pair plans stage the sealed carry record."""
    return action.action_kind in {"fold", "canonical-pair"}


def _folding_carry_mount_descriptors(
    record: FoldingCarryForwardRecord,
    cluster_carry_path: str,
) -> tuple[PhaseContainerMountDescriptor, ...]:
    """Narrow read-only predecessor source mounts plus the staged carry record.

    Each carried output file is mounted read-only at its exact predecessor path
    (the source the Runtime adoption module reads), and the staged carry-record
    document is mounted read-only at its cluster staging path. Duplicate
    (source, target, read_only) entries collapse; a conflicting target is
    rejected.
    """
    mounts: list[PhaseContainerMountDescriptor] = [
        PhaseContainerMountDescriptor(
            source=cluster_carry_path,
            target=cluster_carry_path,
            source_kind="file",
            read_only=True,
            origin="carry-record",
        )
    ]
    for item in record.content:
        for output in item.outputs:
            mounts.append(
                PhaseContainerMountDescriptor(
                    source=output.output_path,
                    target=output.output_path,
                    source_kind="file",
                    read_only=True,
                    origin="carry-source",
                )
            )
    return _dedupe_mount_descriptors(tuple(mounts))


def _fold_projection_mount_descriptors(
    action: FoldingRuntimeAction,
    *,
    packed: bool,
    cluster_projection_path: str,
) -> tuple[PhaseContainerMountDescriptor, ...]:
    """Read-only exact-file projection mount for packed fold/canonical-pair only.

    The projection document is mounted at its exact staged sibling path, never
    the containing staging directory. Scalar actions (and every non-consumer
    action) get no projection mount, preserving the committed scalar script
    bytes and their historical behavior.
    """
    if not packed or action.action_kind not in {"fold", "canonical-pair"}:
        return ()
    return (
        PhaseContainerMountDescriptor(
            source=cluster_projection_path,
            target=cluster_projection_path,
            source_kind="file",
            read_only=True,
            origin="runspec",
        ),
    )


def _dedupe_mount_descriptors(
    descriptors: tuple[PhaseContainerMountDescriptor, ...],
) -> tuple[PhaseContainerMountDescriptor, ...]:
    by_target: dict[str, PhaseContainerMountDescriptor] = {}
    ordered: list[PhaseContainerMountDescriptor] = []
    for descriptor in descriptors:
        existing = by_target.get(descriptor.target)
        if existing is None:
            by_target[descriptor.target] = descriptor
            ordered.append(descriptor)
        elif existing != descriptor:
            raise ValueError(f"conflicting container mount target: {descriptor.target}")
    return tuple(ordered)


def _fold_topology_is_packed(resources: PhaseSlurmResources) -> bool:
    """Typed topology is the sole packed authority; workers > 1 => packed."""
    return resources.is_packed


def _fold_expected_task_indexes(resources: PhaseSlurmResources) -> tuple[int, ...]:
    """Return the exact packed fold array task indexes, or ``()`` for scalar.

    The discriminator is the same contract-owned packed predicate used by the
    renderer and Runtime (``resources.is_packed``): a scalar fold action has one
    worker and therefore no expected array task indexes.
    """
    if not resources.is_packed:
        return ()
    assert resources.nodes is not None
    return tuple(range(resources.nodes))


def _fold_array_directive(resources: PhaseSlurmResources) -> str:
    """--array=0-(nodes-1)[%max_parallel]; max_parallel omitted means no concurrency cap."""
    assert resources.nodes is not None
    directive = f"0-{resources.nodes - 1}"
    if resources.max_parallel is not None:
        directive += f"%{resources.max_parallel}"
    return directive


def _render_action_script(
    phase_runspec: FoldingPhaseRunSpec,
    *,
    action: FoldingRuntimeAction,
    cluster_runspec_path: str,
    runspec_document_sha256: str,
    attempt_root: PurePosixPath,
    action_evidence_path: str,
    handoff_path: str,
    job_name: str,
    correlation: str,
    kernel_image: str,
    carry_record_path: str | None = None,
    carry_mounts: tuple[PhaseContainerMountDescriptor, ...] = (),
    projection_mounts: tuple[PhaseContainerMountDescriptor, ...] = (),
) -> str:
    resources = action.resources
    packed = _fold_topology_is_packed(resources)
    if not packed and resources.array is not None:
        raise ValueError("folding Phase Submission requires typed packed topology (nodes) for Slurm arrays")
    if packed:
        if resources.tasks_per_node < 1:
            raise ValueError("packed fold topology requires tasks_per_node >= 1")
        if resources.gpus_per_task != 1:
            raise ValueError("packed fold topology requires gpus-per-task=1")
    action_root = PurePosixPath(action_evidence_path).parent
    handoff_parent = PurePosixPath(handoff_path).parent
    log_token = "%A_%a" if packed else "%j"
    lines = [
        "#!/usr/bin/env bash",
        "# BSPP folding Phase Runtime Action",
        f"# Phase Run: {phase_runspec.phase_run_id}",
        f"# Attempt: {phase_runspec.attempt_id}",
        f"# RunSpec digest: {phase_runspec.digest}",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --comment={correlation}",
        f"#SBATCH --partition={resources.partition}",
        f"#SBATCH --account={phase_runspec.cluster.account}",
    ]
    if packed:
        lines.extend(
            (
                f"#SBATCH --array={_fold_array_directive(resources)}",
                "#SBATCH --nodes=1",
                f"#SBATCH --ntasks-per-node={resources.tasks_per_node}",
                f"#SBATCH --gpus-per-task={resources.gpus_per_task}",
            )
        )
    else:
        lines.extend(("#SBATCH --nodes=1", "#SBATCH --ntasks=1"))
    lines.extend(
        (
            f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
            f"#SBATCH --mem={resources.memory}",
            f"#SBATCH --time={resources.time}",
            f"#SBATCH --output={action_root / f'slurm-{log_token}.out'}",
            f"#SBATCH --error={action_root / f'slurm-{log_token}.err'}",
        )
    )
    if not packed and resources.gres is not None:
        lines.append(f"#SBATCH --gres={resources.gres}")
    if resources.nodelist is not None:
        lines.append(f"#SBATCH --nodelist={resources.nodelist}")
    mounts = _render_mounts(
        phase_runspec,
        cluster_runspec_path,
        attempt_root,
        carry_mounts,
        projection_mounts,
    )
    executor_argv = _executor_argv(
        phase_runspec_path=cluster_runspec_path,
        action_id=action.action_id,
        action_evidence_path=action_evidence_path,
        handoff_path=handoff_path,
        carry_record_path=carry_record_path,
    )
    # Packed fold actions carry one executor invocation per rank: the bash -c
    # wrapper computes the global rank per srun-spawned task (SLURM_PROCID is
    # only set inside a task, never in the batch script body). The executor
    # argv values are shlex.quote-d absolute POSIX paths derived from validated
    # authority (staging/attempt roots, action ids) and contain no single
    # quotes, so embedding them inside the single-quoted bash -c script is safe.
    if packed:
        executor_argv += ' --rank "${OPENFOLDCTL_GLOBAL_RANK}"'
    writable = "--container-writable " if phase_runspec.cluster.mount_orchestration_source else ""
    dev_env = "env BSPP_ORCHESTRATION_DEV_MOUNT=1 " if phase_runspec.cluster.mount_orchestration_source else ""
    if packed:
        tpn = resources.tasks_per_node
        srun = (
            "srun "
            f"--container-image={shlex.quote(kernel_image)} "
            f"--container-mounts={shlex.quote(mounts)} "
            "--no-container-mount-home "
            f"{writable}"
            f"{dev_env}"
            f"bash -c 'OPENFOLDCTL_GLOBAL_RANK=$((SLURM_ARRAY_TASK_ID * {tpn} + SLURM_PROCID)); "
            f"/usr/local/bin/entrypoint.sh {executor_argv}'"
        )
    else:
        srun = (
            "srun "
            f"--container-image={shlex.quote(kernel_image)} "
            f"--container-mounts={shlex.quote(mounts)} "
            "--no-container-mount-home "
            f"{writable}"
            f"{dev_env}"
            f"/usr/local/bin/entrypoint.sh {executor_argv}"
        )
    lines.extend(
        (
            "",
            "set -euo pipefail",
            f"PHASE_RUNSPEC={shlex.quote(cluster_runspec_path)}",
            f"EXPECTED_RUNSPEC_DOCUMENT_SHA256={runspec_document_sha256}",
            f"ACTION_EVIDENCE={shlex.quote(action_evidence_path)}",
            f"HANDOFF={shlex.quote(handoff_path)}",
            f"mkdir -p -- {shlex.quote(str(action_root))}",
            *((f"mkdir -p -- {shlex.quote(str(handoff_parent))}",) if handoff_parent != action_root else ()),
            '[[ -f "$PHASE_RUNSPEC" ]] || { echo "missing staged Phase RunSpec: $PHASE_RUNSPEC" >&2; exit 127; }',
            '_runspec_sha256="$(sha256sum "$PHASE_RUNSPEC" | awk \'{print $1}\')"',
            '[[ "$_runspec_sha256" == "$EXPECTED_RUNSPEC_DOCUMENT_SHA256" ]] || '
            '{ echo "Phase RunSpec document SHA-256 mismatch" >&2; exit 127; }',
            srun,
            "",
        )
    )
    return "\n".join(lines)


# A rendered mount value must be an absolute path and must not carry the
# Pyxis delimiters (a comma separates the mount list; a colon separates
# source:target[:ro]), control characters, or whitespace (the same script's
# #SBATCH directives are whitespace-separated). Everything else — including
# `+`, `@`, parentheses, and Unicode — stays valid; the whole mount argument
# is additionally shell-quoted at render time, so shell metacharacters in an
# otherwise-valid path cannot split the srun command.
_MOUNT_DELIMITERS = (",", ":")


def _validate_mount_value(value: str, name: str) -> None:
    if (
        not value.startswith("/")
        or any(delimiter in value for delimiter in _MOUNT_DELIMITERS)
        or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(
            f"folding container mount {name} {value!r} must be an absolute path free of "
            "Pyxis delimiters (comma/colon), whitespace, and control characters"
        )


def _render_mounts(
    phase_runspec: FoldingPhaseRunSpec,
    cluster_runspec_path: str,
    attempt_root: PurePosixPath,
    carry_mounts: tuple[PhaseContainerMountDescriptor, ...] = (),
    projection_mounts: tuple[PhaseContainerMountDescriptor, ...] = (),
) -> str:
    """Render deduplicated, conflict-rejecting Pyxis container mounts.

    Order is deterministic: when ``mount_orchestration_source`` is true, the
    read-only orchestration-source mount at ``/workspace/bspp-orchestration``
    comes first; then the immutable staged RunSpec authority (read-only),
    then the writable shared Attempt workspace, then each authenticated local
    input file (read-only, mounted at its exact path — parent directories are
    never exposed), then every declared profile ``extra_mounts`` entry with its
    ``read_only`` honored as a literal Pyxis ``:ro`` suffix, then the narrow
    read-only carry-record and predecessor carry-source mounts (fold and
    canonical-pair only), then the read-only exact-file staged shard projection
    mount (packed fold and packed canonical-pair only). A remote input location
    contributes no local input mount. Identical ``(source, target, read_only)``
    entries collapse to one; the same target with a different source or a
    different read-only mode is rejected as a conflict.
    """
    _orch_target = "/workspace/bspp-orchestration"
    ordered: list[tuple[str, str, bool]] = []
    if phase_runspec.cluster.mount_orchestration_source:
        ordered.append((phase_runspec.cluster.orchestration_repo, _orch_target, True))
    ordered.extend(
        [
            (cluster_runspec_path, cluster_runspec_path, True),
            (str(attempt_root), str(attempt_root), False),
        ]
    )
    input_location = phase_runspec.input_location
    if isinstance(input_location, VerifiedLocalBundledArtifactLocation):
        # Mount each authenticated input FILE at its exact path — never its whole
        # parent directory, which could expose unrelated co-located files.
        for path in (input_location.tar_path, input_location.bundle_path):
            ordered.append((path, path, True))
    for mount in phase_runspec.cluster.extra_mounts:
        ordered.append((mount.source, mount.target, mount.read_only))
    for carry_mount in carry_mounts:
        ordered.append((carry_mount.source, carry_mount.target, carry_mount.read_only))
    for projection_mount in projection_mounts:
        ordered.append((projection_mount.source, projection_mount.target, projection_mount.read_only))
    seen: dict[str, tuple[str, bool]] = {}
    rendered: list[str] = []
    for source, target, read_only in ordered:
        _validate_mount_value(source, "source")
        _validate_mount_value(target, "target")
        existing = seen.get(target)
        if existing is None:
            seen[target] = (source, read_only)
            rendered.append(f"{source}:{target}" + (":ro" if read_only else ""))
            continue
        if existing != (source, read_only):
            raise ValueError(f"conflicting container mount target {target!r}")
    return ",".join(rendered)


def validate_folding_action_evidence(
    *,
    phase_runspec: FoldingPhaseRunSpec,
    evidence: Mapping[str, Mapping[str, object]],
    index_path: Path | None = None,
) -> CanonicalPairIndex | None:
    """Strictly validate folding action evidence and build the canonical-pair index.

    Scheduler-free and control-side: it never opens the pair files and never
    imports the runtime distribution.  The canonical-pair index is built by the
    control-side schema-compatible builder and, when ``index_path`` is given,
    written via atomic replace; otherwise the built index is returned.
    """
    expected = {action.action_id for action in phase_runspec.payload.actions}
    actual = set(evidence)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        detail: list[str] = []
        if missing:
            detail.append(f"missing: {', '.join(missing)}")
        if unknown:
            detail.append(f"unknown: {', '.join(unknown)}")
        raise ValueError("folding action evidence must cover exactly the RunSpec actions (" + "; ".join(detail) + ")")

    if phase_runspec.payload.evidence_profile is not None:
        index = _validate_artifact_folding_evidence(phase_runspec, evidence)
        if index_path is not None:
            write_canonical_pair_index(index, index_path)
            return None
        return index

    index_entries: list[tuple[str, str, str, str, str, str]] = []
    fold_pairs: tuple[PredictionPair, ...] = ()
    for action in phase_runspec.payload.actions:
        payload = evidence[action.action_id]
        if action.action_kind == "msa-flatten":
            _validate_msa_flatten_evidence(phase_runspec, payload)
        elif action.action_kind == "split":
            _validate_split_evidence(payload)
        elif action.action_kind == "preprocess":
            _validate_preprocess_evidence(phase_runspec, payload)
        elif action.action_kind == "fold":
            fold_pairs = _validate_fold_evidence(payload)
        elif action.action_kind == "canonical-pair":
            index_entries.extend(_validate_canonical_pair_evidence(payload, fold_pairs))
        else:
            raise ValueError(f"unsupported folding action kind: {action.action_kind!r}")

    index = build_canonical_pair_index(run_id=phase_runspec.phase_run_id, entries=index_entries)
    if index_path is not None:
        write_canonical_pair_index(index, index_path)
        return None
    return index


def _validate_artifact_folding_evidence(
    runspec: FoldingPhaseRunSpec,
    evidence: Mapping[str, Mapping[str, object]],
) -> CanonicalPairIndex:
    from bspp.orchestration.contract.folding_artifact_evidence import ArtifactFoldEvidence

    actions_root = (
        PurePosixPath(runspec.cluster.project_root)
        / "bspp-phase-runs"
        / runspec.phase_run_id
        / runspec.attempt_id
        / "actions"
    )
    fold: ArtifactFoldEvidence | None = None
    canonical: ArtifactFoldEvidence | None = None
    for action in runspec.payload.actions:
        payload = evidence[action.action_id]
        if action.action_kind == "msa-flatten":
            _validate_msa_flatten_evidence(runspec, payload)
        elif action.action_kind == "split":
            _validate_split_evidence(payload)
        elif action.action_kind == "preprocess":
            _validate_preprocess_evidence(runspec, payload)
        else:
            record = ArtifactFoldEvidence.from_mapping(payload)
            if record.action_id != action.action_id:
                raise ValueError("artifact evidence is assigned to the wrong action")
            record.validate_binding(runspec, actions_root=actions_root)
            if action.action_kind == "fold":
                fold = record
            elif action.action_kind == "canonical-pair":
                canonical = record
            else:
                raise ValueError("unsupported artifact evidence action")
    if fold is None or canonical is None or fold.entries != canonical.entries:
        raise ValueError("artifact fold and canonical pair references differ")
    if canonical.predecessor_digest != canonical_mapping_digest(fold.to_mapping()):
        raise ValueError("artifact canonical predecessor does not bind the complete fold handoff")
    if (fold.install_mode, fold.orchestration_source_commit) != (
        canonical.install_mode,
        canonical.orchestration_source_commit,
    ):
        raise ValueError("artifact fold and canonical source provenance differs")
    return build_canonical_pair_index(
        run_id=runspec.phase_run_id,
        entries=[
            (
                entry.target.target_id,
                entry.target.sequence_sha256,
                entry.model_entity_id,
                entry.tool_used,
                entry.structure.path,
                entry.scores.path,
            )
            for entry in canonical.entries
        ],
    )


def _validate_msa_flatten_evidence(
    phase_runspec: FoldingPhaseRunSpec,
    payload: Mapping[str, object],
) -> None:
    """Reject missing declared MSA members; extra members are intentionally tolerated.

    The check is subset-only (declared members must be present) rather than
    exact-set, matching split/preprocess evidence strictness: an undeclared
    extra ``a3m_paths`` entry is not itself a fold-safety defect and is left
    for the downstream consumer to ignore.
    """
    evidence = MsaFlattenActionEvidence.from_mapping(payload)
    declared = set(phase_runspec.payload.msa_set.member_a3m_paths)
    present = set(evidence.a3m_paths)
    missing = sorted(declared - present)
    if missing:
        raise ValueError(f"msa-flatten evidence is missing declared MSA members: {', '.join(missing)}")


def _validate_split_evidence(payload: Mapping[str, object]) -> None:
    SplitActionEvidence.from_mapping(payload)


def _validate_preprocess_evidence(
    phase_runspec: FoldingPhaseRunSpec,
    payload: Mapping[str, object],
) -> None:
    evidence = PreprocessActionEvidence.from_mapping(payload)
    expected_layout = _expected_layout(phase_runspec.payload.backend)
    if evidence.layout != expected_layout:
        raise ValueError(
            f"preprocess evidence layout {evidence.layout!r} does not match backend {phase_runspec.payload.backend!r} "
            f"(expected {expected_layout!r})"
        )
    if evidence.fasta_dir != "fasta" or evidence.alignment_dir != "alignments":
        raise ValueError("preprocess evidence must declare fasta and alignments directories")


def _expected_layout(backend: str) -> str:
    if backend == "colabfold":
        return "colabfold"
    if backend == "openfold-cli":
        return "openfold"
    if backend in {"bioir", "openfold-trt"}:
        return "bioir"
    raise ValueError(f"unsupported folding backend: {backend!r}")


def _validate_fold_evidence(payload: Mapping[str, object]) -> tuple[PredictionPair, ...]:
    return FoldActionEvidence.from_mapping(payload).pairs


def _validate_canonical_pair_evidence(
    payload: Mapping[str, object],
    fold_pairs: tuple[PredictionPair, ...],
) -> tuple[tuple[str, str, str, str, str, str], ...]:
    evidence = CanonicalPairActionEvidence.from_mapping(payload)
    fold_by_id = {pair.model_entity_id: pair for pair in fold_pairs}
    if len(fold_by_id) != len(fold_pairs):
        raise ValueError("fold evidence contains duplicate model_entity_id prediction pairs")
    referenced = {entry.pair.model_entity_id for entry in evidence.entries}
    if referenced != set(fold_by_id):
        raise ValueError("canonical-pair evidence must reference exactly the fold action prediction pairs")
    for entry in evidence.entries:
        fold_pair = fold_by_id[entry.pair.model_entity_id]
        if fold_pair != entry.pair:
            raise ValueError(
                f"canonical-pair entry {entry.target_id!r} prediction pair does not match the fold action pair "
                f"for {entry.pair.model_entity_id!r}"
            )
    return tuple(
        (
            entry.target_id,
            entry.sequence_sha256,
            entry.pair.model_entity_id,
            entry.pair.tool_used,
            entry.pair.structure_path,
            entry.pair.scores_path,
        )
        for entry in evidence.entries
    )


__all__ = [
    "materialize_folding_attempt_runspec",
    "render_folding_submission_intent",
    "validate_folding_action_evidence",
    "validate_folding_plan_runspec_binding",
]
