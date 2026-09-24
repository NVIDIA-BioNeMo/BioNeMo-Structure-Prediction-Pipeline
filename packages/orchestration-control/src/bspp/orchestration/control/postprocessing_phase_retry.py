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

"""Clean, crash-recoverable Retry for one postprocessing Phase Attempt."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, cast

import yaml

from bspp.orchestration.contract.phase import PhaseMountSnapshot
from bspp.orchestration.contract.postprocessing_action_contract import (
    postprocessing_action_graph_digest,
)
from bspp.orchestration.contract.postprocessing_event import (
    PostprocessingPhaseEvent,
    postprocessing_phase_event_from_mapping,
)
from bspp.orchestration.contract.postprocessing_execution import (
    PostprocessingAttemptPaths,
    PostprocessingClusterSnapshot,
    PostprocessingExecutionProjection,
    PostprocessingPhaseExecutionIdentity,
    QualifiedPostprocessingRuntimeSelection,
)
from bspp.orchestration.contract.postprocessing_retry_events import (
    PostprocessingAttemptRetriedPayload,
    postprocessing_retry_id,
)
from bspp.orchestration.contract.postprocessing_runspec import ExecutablePostprocessingPhaseRunSpec
from bspp.orchestration.contract.postprocessing_runspec_v2 import (
    PostprocessingPhaseRunSpec,
    PostprocessingPhaseRunSpecPayload,
)
from bspp.orchestration.contract.postprocessing_runspec_v3 import (
    PostprocessingPhaseRunSpecPayloadV3,
    PostprocessingPhaseRunSpecV3,
)
from bspp.orchestration.contract.runspec import runspec_from_mapping
from bspp.orchestration.contract.runspec_validation import validate_active_workflow_static
from bspp.orchestration.control.plan import (
    AFDB_TOOLKIT_CONTAINER_TARGET,
    ORCHESTRATION_CONTAINER_TARGET,
    render_runspec_yaml,
)
from bspp.orchestration.control.postprocessing_attempt_projection import (
    derive_v3_attempt_paths,
    postprocessing_physical_input_locators,
    retarget_v3_attempt_runspec_mapping,
    validate_v3_attempt_execution_projection,
)
from bspp.orchestration.control.postprocessing_authority_reader import (
    reject_historical_postprocessing_mutation,
    require_postprocessing_v2_authority,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    canonical_json_bytes as _canonical_json_bytes,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    format_timestamp as _format_timestamp,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    postprocessing_authority_lock as _postprocessing_authority_lock,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    postprocessing_operation_lock as _postprocessing_operation_lock,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    utc_now as _utc_now,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    write_no_replace as _write_no_replace,
)
from bspp.orchestration.control.postprocessing_credential_mounts import (
    snapshot_postprocessing_credential_mounts,
)
from bspp.orchestration.control.postprocessing_phase_lifecycle import Clock
from bspp.orchestration.control.postprocessing_phase_materialization import (
    postprocessing_runtime_actions,
    require_autorequeue_cap_for_policy,
)
from bspp.orchestration.control.postprocessing_phase_types import PostprocessingAuthority
from bspp.orchestration.control.postprocessing_runtime_qualification import (
    resolve_current_postprocessing_runtime,
)
from bspp.orchestration.control.postprocessing_v2_projection_safety import require_safe_v2_projection
from bspp.orchestration.control.profiles import ResolvedClusterProfile
from bspp.orchestration.control.transport import CommandRunner, default_command_runner


@dataclass(frozen=True)
class PostprocessingRetryResult:
    phase_run_id: str
    predecessor_attempt_id: str
    successor_attempt_id: str
    retry_id: str
    phase_runspec_digest: str
    phase_runspec_location: str
    phase_plan_digest: str
    logical_input_manifest_digest: str
    selected_cluster_profile: str
    status: str = "materialized"

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase_kind": "postprocessing",
            "phase_run_id": self.phase_run_id,
            "predecessor_attempt_id": self.predecessor_attempt_id,
            "successor_attempt_id": self.successor_attempt_id,
            "retry_id": self.retry_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "phase_runspec_location": self.phase_runspec_location,
            "phase_plan_digest": self.phase_plan_digest,
            "logical_input_manifest_digest": self.logical_input_manifest_digest,
            "selected_cluster_profile": self.selected_cluster_profile,
            "carry_forward": None,
            "status": self.status,
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


class PostprocessingRuntimeResolver(Protocol):
    """Typed boundary for resolving the current qualified Retry runtime."""

    def resolve(
        self,
        *,
        attempt_id: str,
        profile_name: str,
        config_path: Path,
        source_repo: Path,
        observed_at: datetime,
        runner: CommandRunner,
    ) -> tuple[ResolvedClusterProfile, QualifiedPostprocessingRuntimeSelection, bytes]: ...


@dataclass(frozen=True)
class CurrentPostprocessingRuntimeResolver:
    def resolve(
        self,
        *,
        attempt_id: str,
        profile_name: str,
        config_path: Path,
        source_repo: Path,
        observed_at: datetime,
        runner: CommandRunner,
    ) -> tuple[ResolvedClusterProfile, QualifiedPostprocessingRuntimeSelection, bytes]:
        return resolve_current_postprocessing_runtime(
            attempt_id=attempt_id,
            profile_name=profile_name,
            config_path=config_path,
            source_repo=source_repo,
            observed_at=observed_at,
            runner=runner,
        )


CURRENT_POSTPROCESSING_RUNTIME_RESOLVER = CurrentPostprocessingRuntimeResolver()


@dataclass(frozen=True)
class PostprocessingRetryProjectionStore:
    """Typed create-once projection seam for crash-recovery tests."""

    publish_exact: Callable[[Path, bytes], None]


@dataclass(frozen=True)
class PreparedPostprocessingRetry:
    """One exact predecessor-to-successor Retry transition prepared before mutation."""

    event: PostprocessingPhaseEvent

    def __post_init__(self) -> None:
        if self.event.event_type != "phase-attempt-retried" or not isinstance(
            self.event.payload, PostprocessingAttemptRetriedPayload
        ):
            raise ValueError("prepared postprocessing Retry must contain one Retry event")

    @property
    def payload(self) -> PostprocessingAttemptRetriedPayload:
        return cast("PostprocessingAttemptRetriedPayload", self.event.payload)

    @property
    def phase_run_id(self) -> str:
        return self.event.phase_run_id

    @property
    def predecessor_attempt_id(self) -> str:
        return self.payload.predecessor_attempt_id

    @property
    def successor_attempt_id(self) -> str:
        return self.event.attempt_id

    @property
    def retry_id(self) -> str:
        return self.payload.retry_id

    def to_mapping(self) -> dict[str, object]:
        return self.event.to_mapping()


def prepared_postprocessing_retry_from_mapping(payload: Mapping[str, object]) -> PreparedPostprocessingRetry:
    """Replay a prepared transition using only its exact authenticated event bytes."""
    return PreparedPostprocessingRetry(event=postprocessing_phase_event_from_mapping(payload))


def retry_postprocessing_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    config_path: Path,
    profile_name: str | None = None,
    source_repo: Path | None = None,
    clock: Clock | None = None,
    runner: CommandRunner = default_command_runner,
    runtime_resolver: PostprocessingRuntimeResolver = CURRENT_POSTPROCESSING_RUNTIME_RESOLVER,
    projection_store: PostprocessingRetryProjectionStore | None = None,
) -> PostprocessingRetryResult:
    """Create one immutable successor Attempt without predecessor output adoption."""
    reject_historical_postprocessing_mutation(authority_root, phase_run_id)
    with _postprocessing_operation_lock(authority_root, phase_run_id):
        authority = require_postprocessing_v2_authority(authority_root, phase_run_id)
        if not authority.current_attempt_projection_complete:
            completed = _publish_retry_projection(authority, projection_store=projection_store)
            return _retry_result(completed)
        prepared = _prepare_postprocessing_retry_locked(
            authority,
            config_path=config_path,
            profile_name=profile_name,
            source_repo=source_repo or Path.cwd(),
            clock=clock,
            runner=runner,
            runtime_resolver=runtime_resolver,
        )
        return _apply_prepared_postprocessing_retry_locked(authority, prepared, projection_store=projection_store)


def prepare_postprocessing_retry(
    phase_run_id: str,
    *,
    authority_root: Path,
    config_path: Path,
    source_repo: Path,
    clock: Clock | None = None,
    runner: CommandRunner = default_command_runner,
    runtime_resolver: PostprocessingRuntimeResolver = CURRENT_POSTPROCESSING_RUNTIME_RESOLVER,
) -> PreparedPostprocessingRetry:
    """Prepare one exact Retry without changing durable Phase authority."""
    reject_historical_postprocessing_mutation(authority_root, phase_run_id)
    with _postprocessing_operation_lock(authority_root, phase_run_id):
        authority = require_postprocessing_v2_authority(authority_root, phase_run_id)
        if not authority.current_attempt_projection_complete:
            raise ValueError("postprocessing Retry preparation requires complete current Attempt projections")
        return _prepare_postprocessing_retry_locked(
            authority,
            config_path=config_path,
            profile_name=None,
            source_repo=source_repo,
            clock=clock,
            runner=runner,
            runtime_resolver=runtime_resolver,
        )


def apply_prepared_postprocessing_retry(
    prepared: PreparedPostprocessingRetry,
    *,
    authority_root: Path,
    projection_store: PostprocessingRetryProjectionStore | None = None,
) -> PostprocessingRetryResult:
    """Apply or recover exactly the prepared Retry, rejecting every other transition."""
    reject_historical_postprocessing_mutation(authority_root, prepared.phase_run_id)
    with _postprocessing_operation_lock(authority_root, prepared.phase_run_id):
        authority = require_postprocessing_v2_authority(authority_root, prepared.phase_run_id)
        return _apply_prepared_postprocessing_retry_locked(authority, prepared, projection_store=projection_store)


def _prepare_postprocessing_retry_locked(
    authority: PostprocessingAuthority,
    *,
    config_path: Path,
    profile_name: str | None,
    source_repo: Path,
    clock: Clock | None,
    runner: CommandRunner,
    runtime_resolver: PostprocessingRuntimeResolver,
) -> PreparedPostprocessingRetry:
    if isinstance(authority.runspec, PostprocessingPhaseRunSpecV3):
        validate_v3_attempt_execution_projection(
            authority.runspec,
            authority.legacy_runspec,
            phase_plan_output_namespace=authority.phase_plan.output_namespace,
        )
    else:
        require_safe_v2_projection(authority)
    if authority.sealed or authority.status not in {"failed", "cancelled"}:
        raise ValueError("postprocessing Phase Retry requires a failed or cancelled unsealed current Attempt")
    selected_profile = profile_name or authority.phase_plan.target_cluster
    if selected_profile != authority.phase_plan.target_cluster:
        raise ValueError("postprocessing Phase Retry must use the Phase Plan's pinned target Cluster Profile")
    predecessor_ordinal = int(authority.attempt_id.removeprefix("attempt-"))
    if predecessor_ordinal >= 9999:
        raise ValueError("postprocessing Phase Retry cannot exceed Attempt ordinal 9999")
    successor_attempt_id = f"attempt-{predecessor_ordinal + 1:04d}"
    observed_at = (clock or _utc_now)()
    materialized_at = _format_timestamp(observed_at)
    profile, qualified_runtime, qualification_document = runtime_resolver.resolve(
        attempt_id=successor_attempt_id,
        profile_name=selected_profile,
        config_path=config_path,
        source_repo=source_repo,
        observed_at=observed_at,
        runner=runner,
    )
    require_autorequeue_cap_for_policy(
        policy=authority.phase_plan.autorequeue_policy,
        qualified_runtime=qualified_runtime,
    )
    successor_runspec, successor_legacy = _successor_runspec(
        authority,
        successor_attempt_id=successor_attempt_id,
        materialized_at=materialized_at,
        profile=profile,
        qualified_runtime=qualified_runtime,
    )
    predecessor_outcome = cast("Literal['failed', 'cancelled']", authority.status)
    retry_id = postprocessing_retry_id(
        predecessor_attempt_id=authority.attempt_id,
        predecessor_phase_runspec_digest=authority.runspec.digest,
        predecessor_outcome=predecessor_outcome,
        successor_phase_runspec=successor_runspec,
    )
    payload = PostprocessingAttemptRetriedPayload(
        retry_id=retry_id,
        predecessor_attempt_id=authority.attempt_id,
        predecessor_phase_runspec_digest=authority.runspec.digest,
        predecessor_outcome=predecessor_outcome,
        phase_plan_digest=authority.phase_plan.digest,
        logical_input_manifest_digest=authority.runspec.payload.logical_inputs.digest,
        scientific_identity_digest=authority.runspec.payload.scientific_identity.digest,
        acceptance_policy_semantic_digest=authority.acceptance_policy.semantic_digest,
        action_graph_digest=authority.runspec.payload.action_graph_digest,
        action_semantics_digest=authority.runspec.payload.action_semantics_digest,
        successor_phase_runspec=successor_runspec,
        successor_legacy_runspec_yaml=successor_legacy.decode(),
        successor_runtime_qualification_json=qualification_document.decode(),
        carry_forward=None,
    )
    return PreparedPostprocessingRetry(
        event=PostprocessingPhaseEvent(
            sequence=len(authority.events) + 1,
            event_type="phase-attempt-retried",
            phase_run_id=authority.phase_run_id,
            attempt_id=successor_attempt_id,
            occurred_at=materialized_at,
            payload=payload,
        )
    )


def _apply_prepared_postprocessing_retry_locked(
    authority: PostprocessingAuthority,
    prepared: PreparedPostprocessingRetry,
    *,
    projection_store: PostprocessingRetryProjectionStore | None,
) -> PostprocessingRetryResult:
    expected_event_bytes = _canonical_json_bytes(prepared.to_mapping())
    if authority.attempt_id == prepared.predecessor_attempt_id:
        if not authority.current_attempt_projection_complete:
            raise ValueError("prepared postprocessing Retry predecessor projection is incomplete")
        if authority.runspec.digest != prepared.payload.predecessor_phase_runspec_digest:
            raise ValueError("prepared postprocessing Retry predecessor RunSpec differs")
        if authority.status != prepared.payload.predecessor_outcome or authority.sealed:
            raise ValueError("prepared postprocessing Retry predecessor outcome differs")
        if len(authority.events) + 1 != prepared.event.sequence:
            raise ValueError("prepared postprocessing Retry event sequence is no longer current")
        event_path = authority.authority_path / "events" / f"{prepared.event.sequence:06d}-phase-attempt-retried.json"
        with _postprocessing_authority_lock(authority.authority_path.parent):
            _write_no_replace(event_path, expected_event_bytes)
        authority = require_postprocessing_v2_authority(authority.authority_path.parent, authority.phase_run_id)
    elif authority.attempt_id == prepared.successor_attempt_id:
        matches = tuple(
            event
            for event in authority.events
            if isinstance(event.payload, PostprocessingAttemptRetriedPayload)
            and event.payload.retry_id == prepared.retry_id
        )
        if len(matches) != 1 or _canonical_json_bytes(matches[0].to_mapping()) != expected_event_bytes:
            raise ValueError("durable postprocessing Retry differs from the prepared transition")
    else:
        raise ValueError(
            "prepared postprocessing Retry expected current Attempt "
            f"{prepared.predecessor_attempt_id} or {prepared.successor_attempt_id}, "
            f"not {authority.attempt_id}"
        )
    if authority.attempt_id != prepared.successor_attempt_id:
        raise AssertionError("durable postprocessing Retry did not select its expected successor")
    completed = _publish_retry_projection(authority, projection_store=projection_store)
    if completed.runspec.digest != prepared.payload.successor_phase_runspec.digest:
        raise ValueError("durable postprocessing Retry successor RunSpec differs")
    return _retry_result(completed)


def _successor_runspec(
    authority: PostprocessingAuthority,
    *,
    successor_attempt_id: str,
    materialized_at: str,
    profile: ResolvedClusterProfile,
    qualified_runtime: QualifiedPostprocessingRuntimeSelection,
) -> tuple[ExecutablePostprocessingPhaseRunSpec, bytes]:
    old_paths = authority.runspec.payload.attempt_paths
    mapping = yaml.safe_load(authority.legacy_runspec_bytes)
    if not isinstance(mapping, Mapping):
        raise TypeError("stored legacy RunSpec projection must be a mapping")
    if isinstance(authority.runspec, PostprocessingPhaseRunSpecV3):
        object_output_base_prefix = authority.runspec.object_output_base_prefix
        if object_output_base_prefix is None:
            raise ValueError("postprocessing V3 Retry requires envelope-bound object output authority")
        paths = derive_v3_attempt_paths(
            output_namespace=authority.runspec.payload.execution_projection.phase_identity.output_namespace,
            phase_run_id=authority.phase_run_id,
            attempt_id=successor_attempt_id,
            cluster_output_root=profile.output_root,
            cluster_staging_root=profile.staging_root,
            object_output_base_prefix=object_output_base_prefix,
        )
        successor_mapping = retarget_v3_attempt_runspec_mapping(
            mapping,
            old_paths=old_paths,
            new_paths=paths,
        )
        successor_mapping = _replace_retry_operational_fields(
            successor_mapping,
            profile=profile,
            qualified_runtime=qualified_runtime,
        )
    else:
        paths = _successor_paths(authority, successor_attempt_id=successor_attempt_id, profile=profile)
        successor_mapping = _replace_projection_fields(
            mapping,
            old_paths=old_paths,
            new_paths=paths,
            profile=profile,
            qualified_runtime=qualified_runtime,
        )
    successor_legacy = render_runspec_yaml(successor_mapping).encode()
    exact_mapping = yaml.safe_load(successor_legacy)
    if not isinstance(exact_mapping, Mapping):
        raise TypeError("successor legacy RunSpec projection must be a mapping")
    legacy_runspec = runspec_from_mapping(exact_mapping)
    validate_active_workflow_static(legacy_runspec).raise_if_invalid()
    actions = postprocessing_runtime_actions(legacy_runspec)
    if postprocessing_action_graph_digest(actions) != authority.runspec.payload.action_graph_digest:
        raise ValueError("postprocessing Retry changed the frozen Runtime Action graph")
    stored_action_semantics = authority.runspec.payload.action_semantics
    observed_action_semantics = tuple(
        (item.action_id, item.dependencies, item.expected_task_indexes) for item in actions
    )
    expected_action_semantics = tuple(
        (item.action_id, item.dependencies, item.normalized_task_scope) for item in stored_action_semantics.actions
    )
    if observed_action_semantics != expected_action_semantics:
        raise ValueError("postprocessing Retry changed frozen action semantics")
    physical_inputs = postprocessing_physical_input_locators(legacy_runspec)
    phase_identity = PostprocessingPhaseExecutionIdentity(
        phase_plan_digest=authority.phase_plan.digest,
        logical_input_manifest_digest=authority.runspec.payload.logical_inputs.digest,
        scientific_identity_digest=authority.runspec.payload.scientific_identity.digest,
        action_semantics_digest=stored_action_semantics.digest,
        acceptance_semantic_digest=authority.runspec.payload.acceptance_policy.semantic_digest,
        phase_run_id=authority.phase_run_id,
        attempt_id=successor_attempt_id,
        qualified_runtime_digest=qualified_runtime.digest,
        output_namespace=authority.phase_plan.output_namespace,
        substitutions=paths,
    )
    projection = PostprocessingExecutionProjection(
        document_location=f"attempts/{successor_attempt_id}/legacy-runspec.yaml",
        document_sha256=hashlib.sha256(successor_legacy).hexdigest(),
        document_size_bytes=len(successor_legacy),
        legacy_schema_version=1,
        phase_identity=phase_identity,
        phase_identity_digest=phase_identity.digest,
    )
    policy = replace(
        authority.runspec.payload.acceptance_policy,
        location=f"attempts/{successor_attempt_id}/acceptance-policy.json",
    )
    if isinstance(authority.runspec, PostprocessingPhaseRunSpecV3):
        payload_v3 = PostprocessingPhaseRunSpecPayloadV3(
            actions=actions,
            action_graph_digest=authority.runspec.payload.action_graph_digest,
            action_semantics_digest=authority.runspec.payload.action_semantics.digest,
            action_semantics=authority.runspec.payload.action_semantics,
            scientific_identity=authority.runspec.payload.scientific_identity,
            logical_inputs=authority.runspec.payload.logical_inputs,
            physical_inputs=physical_inputs,
            execution_projection=projection,
            acceptance_policy=policy,
            qualified_runtime=qualified_runtime,
            attempt_paths=paths,
            autorequeue_policy=authority.phase_plan.autorequeue_policy,
        )
        successor_v3 = PostprocessingPhaseRunSpecV3(
            phase_run_id=authority.phase_run_id,
            attempt_id=successor_attempt_id,
            phase_plan_digest=authority.phase_plan.digest,
            materialized_at=materialized_at,
            cluster=_cluster_snapshot(profile, qualified_runtime=qualified_runtime),
            credential_mounts=snapshot_postprocessing_credential_mounts(legacy_runspec, profile),
            cluster_output_root=profile.output_root,
            object_output_base_prefix=object_output_base_prefix,
            payload=payload_v3,
        )
        validate_v3_attempt_execution_projection(
            successor_v3,
            legacy_runspec,
            phase_plan_output_namespace=authority.phase_plan.output_namespace,
        )
        successor: ExecutablePostprocessingPhaseRunSpec = successor_v3
    else:
        payload_v2 = PostprocessingPhaseRunSpecPayload(
            actions=actions,
            action_graph_digest=authority.runspec.payload.action_graph_digest,
            action_semantics_digest=authority.runspec.payload.action_semantics.digest,
            action_semantics=authority.runspec.payload.action_semantics,
            scientific_identity=authority.runspec.payload.scientific_identity,
            logical_inputs=authority.runspec.payload.logical_inputs,
            physical_inputs=physical_inputs,
            execution_projection=projection,
            acceptance_policy=policy,
            qualified_runtime=qualified_runtime,
            attempt_paths=paths,
        )
        successor = PostprocessingPhaseRunSpec(
            phase_run_id=authority.phase_run_id,
            attempt_id=successor_attempt_id,
            phase_plan_digest=authority.phase_plan.digest,
            materialized_at=materialized_at,
            cluster=_cluster_snapshot(profile, qualified_runtime=qualified_runtime),
            payload=payload_v2,
        )
    return successor, successor_legacy


def _successor_paths(
    authority: PostprocessingAuthority,
    *,
    successor_attempt_id: str,
    profile: ResolvedClusterProfile,
) -> PostprocessingAttemptPaths:
    old = authority.runspec.payload.attempt_paths
    new_suffix = f"{authority.phase_run_id}-{successor_attempt_id}"
    old_suffix = f"{authority.phase_run_id}-{authority.attempt_id}"
    expected_old_run_id = f"{authority.phase_plan.output_namespace}-{old_suffix}"
    if old.legacy_run_id != expected_old_run_id:
        raise ValueError("postprocessing predecessor paths do not match their deterministic identity")

    legacy_run_id = f"{authority.phase_plan.output_namespace}-{new_suffix}"
    output_dir = str(Path(profile.output_root) / legacy_run_id)
    prefix_suffix = f"{old_suffix}/"
    if not old.object_prefix.endswith(prefix_suffix):
        raise ValueError("postprocessing predecessor object prefix does not end in its Attempt identity")
    prefix_base = old.object_prefix[: -len(prefix_suffix)].rstrip("/")
    return PostprocessingAttemptPaths(
        legacy_run_id=legacy_run_id,
        output_dir=output_dir,
        evidence_dir=str(Path(output_dir) / "evidence"),
        staging_dir=str(Path(profile.staging_root) / legacy_run_id / "staging"),
        object_prefix=f"{prefix_base}/{new_suffix}/",
    )


def _replace_projection_fields(
    value: Mapping[str, object],
    *,
    old_paths: PostprocessingAttemptPaths,
    new_paths: PostprocessingAttemptPaths,
    profile: ResolvedClusterProfile,
    qualified_runtime: QualifiedPostprocessingRuntimeSelection,
) -> dict[str, object]:
    """Update only fields generated from explicit phase-owned Attempt substitutions."""
    successor = copy.deepcopy(dict(value))
    replacements = {
        ("dataset", "run_id"): (old_paths.legacy_run_id, new_paths.legacy_run_id),
        ("paths", "staging_dir"): (old_paths.staging_dir, new_paths.staging_dir),
        ("paths", "output_dir"): (old_paths.output_dir, new_paths.output_dir),
        ("paths", "log_dir"): (
            str(Path(old_paths.output_dir) / "logs"),
            str(Path(new_paths.output_dir) / "logs"),
        ),
        ("paths", "recipe_dir"): (
            str(Path(old_paths.output_dir) / "rendered_recipe"),
            str(Path(new_paths.output_dir) / "rendered_recipe"),
        ),
        ("storage", "s3_output_prefix"): (old_paths.object_prefix, new_paths.object_prefix),
        ("submission", "evidence_dir"): (old_paths.evidence_dir, new_paths.evidence_dir),
        ("submission", "report_path"): (
            str(Path(old_paths.output_dir) / "RUN_REPORT.md"),
            str(Path(new_paths.output_dir) / "RUN_REPORT.md"),
        ),
        ("acceptance", "candidate_run_name"): (old_paths.legacy_run_id, new_paths.legacy_run_id),
    }
    for field_path, (old, new) in replacements.items():
        parent = successor.get(field_path[0])
        if not isinstance(parent, dict) or parent.get(field_path[1]) != old:
            raise ValueError(f"stored legacy RunSpec has unexpected Retry-owned field: {'.'.join(field_path)}")
        parent[field_path[1]] = new
    for section, field, selected in (
        ("cluster", "name", profile.name),
        ("cluster", "account", profile.account),
        ("cluster", "owner", profile.owner),
        ("paths", "project_root", profile.project_root),
        ("paths", "orchestration_repo", profile.orchestration_repo),
        ("container", "image", qualified_runtime.image_path),
    ):
        parent = successor.get(section)
        if not isinstance(parent, dict):
            raise ValueError(f"stored legacy RunSpec lacks Retry operational section: {section}")
        parent[field] = selected
    path_mapping = cast("dict[str, object]", successor["paths"])
    if profile.afdb_toolkit_repo is None:
        path_mapping.pop("afdb_toolkit_repo", None)
    else:
        path_mapping["afdb_toolkit_repo"] = profile.afdb_toolkit_repo
    container = cast("dict[str, object]", successor["container"])
    mounts: list[dict[str, str]] = [
        {"source": profile.orchestration_repo, "target": ORCHESTRATION_CONTAINER_TARGET},
        {"source": profile.project_root, "target": profile.project_root},
    ]
    if profile.afdb_toolkit_repo is not None:
        mounts.insert(1, {"source": profile.afdb_toolkit_repo, "target": AFDB_TOOLKIT_CONTAINER_TARGET})
    mounts.extend({"source": str(item.source), "target": str(item.target)} for item in profile.extra_mounts)
    container["mounts"] = mounts
    old_resources = successor.get("resources")
    if not isinstance(old_resources, Mapping):
        raise ValueError("stored legacy RunSpec lacks Retry operational resources")
    old_gpu = old_resources.get("gpu_worker")
    array = old_gpu.get("array") if isinstance(old_gpu, Mapping) else None
    resources = {
        name: resource.model_dump(mode="json", exclude_none=True) for name, resource in profile.resources.items()
    }
    if "gpu_worker" in resources and array is not None:
        resources["gpu_worker"]["array"] = array
    successor["resources"] = resources
    return successor


def _replace_retry_operational_fields(
    successor: dict[str, object],
    *,
    profile: ResolvedClusterProfile,
    qualified_runtime: QualifiedPostprocessingRuntimeSelection,
) -> dict[str, object]:
    """Refresh non-Attempt operational fields after V3 output retargeting."""
    for section, field, selected in (
        ("cluster", "name", profile.name),
        ("cluster", "account", profile.account),
        ("cluster", "owner", profile.owner),
        ("paths", "project_root", profile.project_root),
        ("paths", "orchestration_repo", profile.orchestration_repo),
        ("container", "image", qualified_runtime.image_path),
    ):
        parent = successor.get(section)
        if not isinstance(parent, dict):
            raise ValueError(f"stored legacy RunSpec lacks Retry operational section: {section}")
        parent[field] = selected
    path_mapping = cast("dict[str, object]", successor["paths"])
    if profile.afdb_toolkit_repo is None:
        path_mapping.pop("afdb_toolkit_repo", None)
    else:
        path_mapping["afdb_toolkit_repo"] = profile.afdb_toolkit_repo
    container = cast("dict[str, object]", successor["container"])
    mounts: list[dict[str, str]] = [
        {"source": profile.orchestration_repo, "target": ORCHESTRATION_CONTAINER_TARGET},
        {"source": profile.project_root, "target": profile.project_root},
    ]
    if profile.afdb_toolkit_repo is not None:
        mounts.insert(1, {"source": profile.afdb_toolkit_repo, "target": AFDB_TOOLKIT_CONTAINER_TARGET})
    mounts.extend({"source": str(item.source), "target": str(item.target)} for item in profile.extra_mounts)
    container["mounts"] = mounts
    old_resources = successor.get("resources")
    if not isinstance(old_resources, Mapping):
        raise ValueError("stored legacy RunSpec lacks Retry operational resources")
    old_gpu = old_resources.get("gpu_worker")
    array = old_gpu.get("array") if isinstance(old_gpu, Mapping) else None
    resources = {
        name: resource.model_dump(mode="json", exclude_none=True) for name, resource in profile.resources.items()
    }
    if "gpu_worker" in resources and array is not None:
        resources["gpu_worker"]["array"] = array
    successor["resources"] = resources
    return successor


def _cluster_snapshot(
    profile: ResolvedClusterProfile,
    *,
    qualified_runtime: QualifiedPostprocessingRuntimeSelection,
) -> PostprocessingClusterSnapshot:
    return PostprocessingClusterSnapshot(
        profile_name=profile.name,
        owner=profile.owner,
        transport=profile.transport,
        ssh_target=profile.ssh_target,
        account=profile.account,
        project_root=profile.project_root,
        staging_root=profile.staging_root,
        orchestration_repo=profile.orchestration_repo,
        runtime_image=qualified_runtime.image_path,
        extra_mounts=tuple(PhaseMountSnapshot(source=item.source, target=item.target) for item in profile.extra_mounts),
    )


def _publish_retry_projection(
    authority: PostprocessingAuthority,
    *,
    projection_store: PostprocessingRetryProjectionStore | None = None,
) -> PostprocessingAuthority:
    if authority.current_attempt_projection_complete:
        return authority
    with _postprocessing_authority_lock(authority.authority_path.parent):
        attempt_root = authority.authority_path / "attempts" / authority.attempt_id
        if os.path.lexists(attempt_root):
            if attempt_root.is_symlink() or not attempt_root.is_dir():
                raise ValueError("postprocessing Retry Attempt projection parent is unsafe")
        else:
            attempt_root.mkdir()
        projections = (
            (attempt_root / "phase-runspec.json", _canonical_json_bytes(authority.runspec.to_mapping())),
            (attempt_root / "legacy-runspec.yaml", authority.legacy_runspec_bytes),
            (attempt_root / "acceptance-policy.json", authority.acceptance_policy_bytes),
            (attempt_root / "runtime-qualification.json", authority.runtime_qualification_bytes),
        )
        for path, payload in projections:
            (projection_store or LOCAL_POSTPROCESSING_RETRY_PROJECTION_STORE).publish_exact(path, payload)
    completed = require_postprocessing_v2_authority(authority.authority_path.parent, authority.phase_run_id)
    if not completed.current_attempt_projection_complete:
        raise ValueError("postprocessing Retry projection did not become complete")
    return completed


def _publish_exact(path: Path, payload: bytes) -> None:
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing postprocessing Retry projection differs: {path}")
        return
    _write_no_replace(path, payload)


LOCAL_POSTPROCESSING_RETRY_PROJECTION_STORE = PostprocessingRetryProjectionStore(publish_exact=_publish_exact)


def _retry_result(authority: PostprocessingAuthority) -> PostprocessingRetryResult:
    matches = tuple(
        event.payload
        for event in authority.events
        if event.attempt_id == authority.attempt_id and isinstance(event.payload, PostprocessingAttemptRetriedPayload)
    )
    if len(matches) != 1:
        raise ValueError("current postprocessing Retry requires exactly one durable authority event")
    payload = matches[0]
    return PostprocessingRetryResult(
        phase_run_id=authority.phase_run_id,
        predecessor_attempt_id=payload.predecessor_attempt_id,
        successor_attempt_id=authority.attempt_id,
        retry_id=payload.retry_id,
        phase_runspec_digest=authority.runspec.digest,
        phase_runspec_location=f"attempts/{authority.attempt_id}/phase-runspec.json",
        phase_plan_digest=authority.phase_plan.digest,
        logical_input_manifest_digest=authority.runspec.payload.logical_inputs.digest,
        selected_cluster_profile=authority.runspec.cluster.profile_name,
    )


__all__ = [
    "PostprocessingRetryResult",
    "PreparedPostprocessingRetry",
    "apply_prepared_postprocessing_retry",
    "prepare_postprocessing_retry",
    "prepared_postprocessing_retry_from_mapping",
    "retry_postprocessing_phase",
]
