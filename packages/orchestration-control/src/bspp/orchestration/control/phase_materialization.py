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

"""Scheduler-free preprocessing Phase Materialization service."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from bspp.orchestration.control.postprocessing_phase_adapter import (
        PostprocessingPhaseMaterializationResult,
    )

from bspp.orchestration.contract.folding_execution import OPENFOLD_TRT_DEFERRED_MODEL_FN_ERROR
from bspp.orchestration.contract.phase import (
    FoldingPhasePlan,
    PhaseMountSnapshot,
    PhasePlan,
    PhaseSlurmResources,
    phase_plan_family_from_mapping,
    phase_plan_from_mapping,
)
from bspp.orchestration.contract.phase_state import (
    PhaseAttempt,
    PhaseMaterializedEvent,
    PhaseMaterializedPayload,
    PhaseRun,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    VerifiedLocalBundledArtifactLocation,
    VerifiedRemoteBundledArtifactLocation,
)
from bspp.orchestration.control.folding_phase_adapter import (
    _LEGACY_MSA_SET_MANIFEST_ERROR,
    materialize_folding_attempt_runspec,
)
from bspp.orchestration.control.folding_phase_types import (
    FoldingBackendImageSelection,
    FoldingPhaseAttemptOperationalSelection,
)
from bspp.orchestration.control.phase_attempt_materialization import (
    materialize_preprocessing_attempt_runspec,
    qualify_phase_attempt_operational_selection,
)
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore
from bspp.orchestration.control.profiles import (
    ResolvedClusterProfile,
    resolve_cluster_profile,
    validate_folding_backend_asset_mount_coverage,
)

Clock = Callable[[], datetime]
PhaseRunIdFactory = Callable[[], str]


@dataclass(frozen=True)
class PhaseMaterializationResult:
    """Stable operator-facing result of scheduler-free materialization."""

    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    authority_root: Path

    def to_mapping(self) -> dict[str, object]:
        return {
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "authority_root": str(self.authority_root),
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


def materialize_phase(
    phase_plan_path: Path,
    *,
    authority_root: Path,
    config_path: Path,
    source_repo: Path | None = None,
    clock: Clock | None = None,
    phase_run_id_factory: PhaseRunIdFactory | None = None,
    authority_store: PhaseAuthorityStore | None = None,
) -> PhaseMaterializationResult | PostprocessingPhaseMaterializationResult:
    """Create and validate one initial Phase Attempt without scheduler access."""
    from bspp.orchestration.control.phase_adapters import phase_plan_family

    family = phase_plan_family(phase_plan_path)
    if family == "postprocessing":
        from bspp.orchestration.control.postprocessing_phase_adapter import materialize_postprocessing_phase

        if authority_store is not None:
            raise ValueError("postprocessing Phase Materialization rejects a preprocessing authority store")
        return materialize_postprocessing_phase(
            phase_plan_path,
            authority_root=authority_root,
            config_path=config_path,
            source_repo=source_repo,
            clock=clock,
            phase_run_id_factory=phase_run_id_factory,
        )
    if family == "folding":
        return _materialize_folding_phase(
            phase_plan_path,
            authority_root=authority_root,
            config_path=config_path,
            clock=clock,
            phase_run_id_factory=phase_run_id_factory,
            authority_store=authority_store,
        )
    phase_plan = load_phase_plan(phase_plan_path)
    _verify_input_bytes(phase_plan, phase_plan_path=phase_plan_path)
    profile = resolve_cluster_profile(phase_plan.target_cluster, config_path=config_path)
    now = (clock or _utc_now)()
    operational = qualify_phase_attempt_operational_selection(
        profile,
        source_repo=source_repo or Path.cwd(),
        now=now,
    )

    phase_run_id = (phase_run_id_factory or _new_phase_run_id)()
    attempt_id = "attempt-0001"
    materialized_at = _format_timestamp(now)
    phase_runspec = materialize_preprocessing_attempt_runspec(
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        phase_plan=phase_plan,
        materialized_at=materialized_at,
        operational=operational,
        rematerialize_runtime_image=False,
    )
    attempt = PhaseAttempt(
        attempt_id=attempt_id,
        ordinal=1,
        phase_runspec_location=f"attempts/{attempt_id}/phase-runspec.json",
        phase_runspec_digest=phase_runspec.digest,
        created_at=materialized_at,
    )
    phase_run = PhaseRun(
        phase_run_id=phase_run_id,
        phase_plan_location="phase-plan.json",
        phase_plan_digest=phase_plan.digest,
        created_at=materialized_at,
        current_attempt_id=attempt_id,
        attempts=(attempt,),
    )
    event = PhaseMaterializedEvent(
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        occurred_at=materialized_at,
        payload=PhaseMaterializedPayload(phase_run=phase_run, phase_runspec=phase_runspec),
    )
    if authority_store is not None and authority_store.authority_root != authority_root:
        msg = "injected PhaseAuthorityStore root must match authority_root"
        raise ValueError(msg)
    store = authority_store or PhaseAuthorityStore(authority_root)
    validation = store.publish(
        phase_plan=phase_plan,
        phase_run=phase_run,
        phase_runspec=phase_runspec,
        materialized_event=event,
    )
    return PhaseMaterializationResult(
        phase_run_id=validation.phase_run.phase_run_id,
        attempt_id=validation.phase_run.current_attempt_id,
        phase_runspec_digest=validation.phase_runspec.digest,
        authority_root=authority_root,
    )


def load_phase_plan(path: Path) -> PhasePlan:
    """Load a JSON/YAML Phase Plan through the strict contract loader."""
    payload = yaml.safe_load(path.read_bytes())
    if not isinstance(payload, Mapping):
        msg = f"Expected Phase Plan mapping in {path}"
        raise TypeError(msg)
    return phase_plan_from_mapping(payload)


def _materialize_folding_phase(
    phase_plan_path: Path,
    *,
    authority_root: Path,
    config_path: Path,
    clock: Clock | None,
    phase_run_id_factory: PhaseRunIdFactory | None,
    authority_store: PhaseAuthorityStore | None,
) -> PhaseMaterializationResult:
    """Materialize one folding Phase Attempt through the generalized authority path."""
    phase_plan = _load_folding_phase_plan(phase_plan_path)
    _verify_folding_msa_set_input_bytes(phase_plan, phase_plan_path=phase_plan_path)
    profile = resolve_cluster_profile(phase_plan.target_cluster, config_path=config_path)
    now = (clock or _utc_now)()
    operational = _resolve_folding_operational_selection(profile, phase_plan.payload.backend)
    _reject_legacy_msa_set_manifest(phase_plan)

    phase_run_id = (phase_run_id_factory or _new_phase_run_id)()
    attempt_id = "attempt-0001"
    materialized_at = _format_timestamp(now)
    phase_runspec = materialize_folding_attempt_runspec(
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        phase_plan=phase_plan,
        materialized_at=materialized_at,
        operational=operational,
    )
    attempt = PhaseAttempt(
        attempt_id=attempt_id,
        ordinal=1,
        phase_runspec_location=f"attempts/{attempt_id}/phase-runspec.json",
        phase_runspec_digest=phase_runspec.digest,
        created_at=materialized_at,
    )
    phase_run = PhaseRun(
        phase_run_id=phase_run_id,
        phase_plan_location="phase-plan.json",
        phase_plan_digest=phase_plan.digest,
        created_at=materialized_at,
        current_attempt_id=attempt_id,
        attempts=(attempt,),
        phase_kind="folding",
    )
    event = PhaseMaterializedEvent(
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        occurred_at=materialized_at,
        payload=PhaseMaterializedPayload(phase_run=phase_run, phase_runspec=phase_runspec),
    )
    if authority_store is not None and authority_store.authority_root != authority_root:
        msg = "injected PhaseAuthorityStore root must match authority_root"
        raise ValueError(msg)
    store = authority_store or PhaseAuthorityStore(authority_root)
    validation = store.publish(
        phase_plan=phase_plan,
        phase_run=phase_run,
        phase_runspec=phase_runspec,
        materialized_event=event,
    )
    return PhaseMaterializationResult(
        phase_run_id=validation.phase_run.phase_run_id,
        attempt_id=validation.phase_run.current_attempt_id,
        phase_runspec_digest=validation.phase_runspec.digest,
        authority_root=authority_root,
    )


def _load_folding_phase_plan(path: Path) -> FoldingPhasePlan:
    """Load a folding Phase Plan through the family-dispatched contract loader."""
    payload = yaml.safe_load(path.read_bytes())
    if not isinstance(payload, Mapping):
        msg = f"Expected folding Phase Plan mapping in {path}"
        raise TypeError(msg)
    plan = phase_plan_family_from_mapping(payload)
    if not isinstance(plan, FoldingPhasePlan):
        msg = f"Phase Plan is not a folding Phase Plan: {path}"
        raise ValueError(msg)
    return plan


def _reject_legacy_msa_set_manifest(phase_plan: FoldingPhasePlan) -> None:
    """Reject direct folding of a legacy MSA set manifest lacking member lengths.

    A pre-existing folding Phase Plan/RunSpec with a legacy manifest must still
    deserialize unchanged (the epic's backward-compat constraint), so this check
    lives at materialization rather than in Plan/RunSpec deserialization. It runs
    after operational selection (so the openfold-trt deferred-model_fn fail-closed
    keeps precedence) but before the adapter materializes the attempt, and covers
    both local and remote locations.
    """
    manifest = phase_plan.payload.msa_set_manifest
    if manifest is not None and not manifest.has_member_lengths():
        raise ValueError(_LEGACY_MSA_SET_MANIFEST_ERROR)


def _verify_folding_msa_set_input_bytes(phase_plan: FoldingPhasePlan, *, phase_plan_path: Path) -> None:
    """Verify local bundled MSA-set input bytes during materialization.

    Local locations are byte-verified here before authority publication. Remote
    locations are accepted from the already verified, immutable
    ``VerifiedRemoteBundledArtifactLocation``; the generic Phase lifecycle does
    not automatically fetch or re-verify remote bytes in this epic.
    """
    location = phase_plan.input_location
    if isinstance(location, VerifiedRemoteBundledArtifactLocation):
        return
    if not isinstance(location, VerifiedLocalBundledArtifactLocation):
        raise ValueError("folding Phase Plan input location must be a verified bundled Artifact Location")
    bundle_path = Path(location.bundle_path)
    tar_path = Path(location.tar_path)
    source_bundle = bundle_path if bundle_path.is_absolute() else phase_plan_path.parent / bundle_path
    source_tar = tar_path if tar_path.is_absolute() else phase_plan_path.parent / tar_path
    _verify_bundled_bytes(source_bundle, location.lz4_size_bytes, location.lz4_sha256, "bundle")
    _verify_bundled_bytes(source_tar, location.tar_size_bytes, location.tar_sha256, "tar")


def _verify_bundled_bytes(path: Path, expected_size: int, expected_sha256: str, label: str) -> None:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    if size_bytes != expected_size:
        msg = f"declared folding {label} size mismatch for {path}: expected {expected_size}, observed {size_bytes}"
        raise ValueError(msg)
    observed_digest = digest.hexdigest()
    if observed_digest != expected_sha256:
        msg = (
            f"declared folding {label} SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, observed {observed_digest}"
        )
        raise ValueError(msg)


def _resolve_folding_operational_selection(
    profile: ResolvedClusterProfile,
    backend: str,
) -> FoldingPhaseAttemptOperationalSelection:
    """Build a folding operational selection from one resolved Cluster Profile.

    The per-backend kernel image override table (``folding_backend_images``)
    selects the fold action's kernel image; the fold action never falls back
    to the non-fold runtime image (``profile.image``), and a backend without
    an explicit override fails closed. The selected backend's external assets
    are resolved fail-closed (absent assets fail before submission) and every
    selected asset path must be covered by an exact read_only extra mount.
    The release preset is passed through to the operational
    selection for qualification identity.
    """
    resources = _folding_slurm_resources(profile, "control_cpu")
    fold_resources = _folding_slurm_resources(profile, "gpu_worker", fold=True)
    # openfold-trt is deliberately fail-closed before the ordinary
    # backend-asset lookup, so a profile with no TRT asset stanza still reaches
    # the stable deferred-model_fn error rather than a generic missing-assets one.
    if backend == "openfold-trt":
        raise ValueError(OPENFOLD_TRT_DEFERRED_MODEL_FN_ERROR)
    # The fold action's kernel image must be an explicit per-backend override;
    # it never falls back to the non-fold runtime image (profile.image).
    # image_for_backend fails closed when the selected backend has no override.
    backend_images = FoldingBackendImageSelection(backend_images=dict(profile.folding_backend_images))
    try:
        assets = profile.folding_backend_assets[backend]
    except KeyError as exc:
        raise ValueError(
            f"Cluster Profile {profile.name!r} does not resolve folding backend assets for {backend!r}"
        ) from exc
    validate_folding_backend_asset_mount_coverage(assets, profile.extra_mounts)
    return FoldingPhaseAttemptOperationalSelection(
        profile_name=profile.name,
        owner=profile.owner,
        transport=profile.transport,
        ssh_target=profile.ssh_target,
        account=profile.account,
        project_root=profile.project_root,
        staging_root=profile.staging_root,
        orchestration_repo=profile.orchestration_repo,
        runtime_image=profile.image,
        backend_images=backend_images,
        release_preset=profile.folding_release_preset,
        resources=resources,
        fold_resources=fold_resources,
        extra_mounts=tuple(
            PhaseMountSnapshot(source=mount.source, target=mount.target, read_only=mount.read_only)
            for mount in profile.extra_mounts
        ),
        assets=assets,
        mount_orchestration_source=profile.mount_orchestration_source,
    )


def _folding_slurm_resources(profile: ResolvedClusterProfile, key: str, *, fold: bool = False) -> PhaseSlurmResources:
    try:
        selected = profile.resources[key]
    except KeyError as exc:
        msg = f"Cluster Profile {profile.name!r} does not define required {key!r} resources"
        raise ValueError(msg) from exc
    nodes = selected.nodes
    array = selected.array
    if selected.is_packed:
        assert nodes is not None
        if fold and selected.gpus_per_task != 1:
            raise ValueError(
                f"Cluster Profile {profile.name!r} fold resource {key!r} requires gpus_per_task=1 when packed"
            )
        array = f"0-{nodes - 1}"
        if selected.max_parallel is not None:
            array += f"%{selected.max_parallel}"
    return PhaseSlurmResources(
        partition=selected.partition,
        cpus_per_task=selected.cpus_per_task,
        memory=selected.memory,
        time=selected.time,
        gres=selected.gres,
        array=array,
        nodelist=selected.nodelist,
        nodes=selected.nodes,
        tasks_per_node=selected.tasks_per_node,
        gpus_per_task=selected.gpus_per_task,
        max_parallel=selected.max_parallel,
    )


def _verify_input_bytes(phase_plan: PhasePlan, *, phase_plan_path: Path) -> None:
    from bspp.orchestration.contract.phase import VerifiedRemoteInputLocation

    if isinstance(phase_plan.input_location, VerifiedRemoteInputLocation):
        # Remote input bytes are verified after download at runtime, not at materialization.
        return
    declared = Path(phase_plan.input_location.path)
    source_path = declared if declared.is_absolute() else phase_plan_path.parent / declared
    digest = hashlib.sha256()
    size_bytes = 0
    with source_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    if size_bytes != phase_plan.input_location.size_bytes:
        msg = (
            f"declared preprocessing input size mismatch for {phase_plan.input_location.path}: "
            f"expected {phase_plan.input_location.size_bytes}, observed {size_bytes}"
        )
        raise ValueError(msg)
    observed_digest = digest.hexdigest()
    if observed_digest != phase_plan.input_location.sha256:
        msg = (
            f"declared preprocessing input SHA-256 mismatch for {phase_plan.input_location.path}: "
            f"expected {phase_plan.input_location.sha256}, observed {observed_digest}"
        )
        raise ValueError(msg)


def _new_phase_run_id() -> str:
    return f"phase-run-{uuid.uuid4().hex}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        msg = "Phase Materialization clock must return a timezone-aware datetime"
        raise ValueError(msg)
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = [
    "PhaseMaterializationResult",
    "load_phase_plan",
    "materialize_phase",
]
