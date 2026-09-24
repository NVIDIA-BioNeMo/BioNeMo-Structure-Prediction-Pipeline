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

"""Strict contracts for preprocessing and folding Phase Plans and Phase RunSpecs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast, get_args, overload

if TYPE_CHECKING:
    from bspp.orchestration.contract.folding_execution import FoldingBackendAssetsSnapshot
    from bspp.orchestration.contract.folding_input import MsaSetConsumption
    from bspp.orchestration.contract.postprocessing_plan import (
        PostprocessingPhasePlan,
    )
    from bspp.orchestration.contract.postprocessing_runspec import (
        ExecutablePostprocessingPhaseRunSpec,
    )
    from bspp.orchestration.contract.preprocessing_handoff import (
        MsaArtifactSetManifest,
        VerifiedLocalBundledArtifactLocation,
        VerifiedRemoteBundledArtifactLocation,
    )

from bspp.orchestration.contract.database_placement import (
    DatabaseSetSelection,
    PreprocessingDatabaseBinding,
    database_set_selection_from_mapping,
    preprocessing_database_binding_from_mapping,
)
from bspp.orchestration.contract.folding_bioir import BioIRModelPolicy, bioir_model_policy_from_mapping
from bspp.orchestration.contract.folding_carry_forward import (
    FoldingCarryForwardReference,
    folding_carry_forward_reference_from_mapping,
)
from bspp.orchestration.contract.folding_shard import (
    FoldShardProjectionBinding,
    fold_shard_projection_binding_from_mapping,
)
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardReference,
    attempt_carry_forward_reference_from_mapping,
)
from bspp.orchestration.contract.preprocessing import PreprocessingWorkPlan, preprocessing_work_plan_from_mapping
from bspp.orchestration.contract.preprocessing_execution import (
    PreprocessingChunkExecutionIntent,
    PreprocessingChunkExecutionPlan,
    preprocessing_chunk_execution_intent_from_mapping,
    preprocessing_chunk_execution_plan_from_mapping,
    validate_scientific_schema_version,
)
from bspp.orchestration.contract.preprocessing_runtime import (
    QualifiedPreprocessingRuntimeSelection,
    qualified_preprocessing_runtime_selection_from_mapping,
)
from bspp.orchestration.contract.runplan import reject_environment_interpolation
from bspp.orchestration.contract.runspec_policies import parse_slurm_time_seconds
from bspp.orchestration.contract.slurm import validate_exact_slurm_node
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

PhaseKind = Literal["preprocessing", "folding"]
TransportKind = Literal["ssh", "local-slurm"]
InputLocationKind = Literal["verified-local-file", "verified-remote-file"]
RuntimeActionKind = Literal["preprocessing-chunk"]
FoldingRuntimeActionKind = Literal["msa-flatten", "split", "preprocess", "fold", "canonical-pair"]
PhaseSeamTransportPolicy = Literal["publish-to-s3", "local"]

FOLDING_BACKENDS = ("openfold-cli", "bioir", "colabfold", "openfold-trt")

_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FOLDING_ACTION_ID = re.compile(r"(msa-flatten|split|preprocess|fold|canonical-pair)-[0-9]{6}")


def canonical_mapping_digest(payload: Mapping[str, object]) -> str:
    """Return the repository's stable compact-JSON SHA-256 digest."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class VerifiedLocalInputLocation:
    """A local FASTA location whose declared bytes are verified at materialization."""

    path: str
    sha256: str
    size_bytes: int
    kind: InputLocationKind = "verified-local-file"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "VerifiedLocalInputLocation")
        if self.kind != "verified-local-file":
            msg = f"unsupported input location kind: {self.kind!r}"
            raise ValueError(msg)
        if not self.path or not self.path.endswith(".fa"):
            msg = "verified local input path must be a non-empty .fa path"
            raise ValueError(msg)
        if _SHA256.fullmatch(self.sha256) is None:
            msg = "verified local input sha256 must be 64 lowercase hexadecimal characters"
            raise ValueError(msg)
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            msg = "verified local input size_bytes must be a non-negative integer"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class VerifiedRemoteInputLocation:
    """A remote FASTA location whose declared bytes are verified after download.

    The ``source_uri`` must be an ``s3://`` object key (not bare ``s3://``).
    The ``path`` is a relative ``.fa`` path under the attempt workspace where
    the downloaded bytes are materialized; it is not a local source path that
    exists at materialization time.
    """

    source_uri: str
    sha256: str
    size_bytes: int
    path: str
    kind: InputLocationKind = "verified-remote-file"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "VerifiedRemoteInputLocation")
        if self.kind != "verified-remote-file":
            msg = f"unsupported input location kind: {self.kind!r}"
            raise ValueError(msg)
        if not self.source_uri.startswith("s3://") or self.source_uri[5:].strip("/") == "":
            msg = "verified remote input source_uri must be a non-empty s3:// object key"
            raise ValueError(msg)
        if _SHA256.fullmatch(self.sha256) is None:
            msg = "verified remote input sha256 must be 64 lowercase hexadecimal characters"
            raise ValueError(msg)
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes <= 0:
            msg = "verified remote input size_bytes must be a positive integer"
            raise ValueError(msg)
        if not self.path or not self.path.endswith(".fa") or self.path.startswith("/"):
            msg = "verified remote input path must be a relative .fa path under the attempt workspace"
            raise ValueError(msg)
        if any(component == ".." for component in self.path.split("/")):
            msg = "verified remote input path must not contain '..' path components"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "source_uri": self.source_uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "path": self.path,
        }


@dataclass(frozen=True)
class PhaseMountSnapshot:
    """One explicit mount captured from the selected Cluster Profile."""

    source: str
    target: str
    read_only: bool = False
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PhaseMountSnapshot")
        if not self.source or not self.target:
            msg = "phase mount source and target must be non-empty"
            raise ValueError(msg)
        if not isinstance(self.read_only, bool):
            msg = "phase mount read_only must be a boolean"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "source": self.source,
            "target": self.target,
        }
        if self.read_only:
            result["read_only"] = self.read_only
        return result


@dataclass(frozen=True)
class PhaseSlurmResources:
    """Concrete Slurm resource selection for one Runtime Action."""

    partition: str
    cpus_per_task: int
    memory: str
    time: str
    gres: str | None = None
    array: str | None = None
    nodelist: str | None = None
    nodes: int | None = None
    tasks_per_node: int = 1
    gpus_per_task: int | None = None
    max_parallel: int | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PhaseSlurmResources")
        if not self.partition or not self.memory or not self.time:
            msg = "phase Slurm partition, memory, and time must be non-empty"
            raise ValueError(msg)
        if not isinstance(self.cpus_per_task, int) or isinstance(self.cpus_per_task, bool) or self.cpus_per_task <= 0:
            msg = "phase Slurm cpus_per_task must be positive"
            raise ValueError(msg)
        if self.gres == "" or self.array == "":
            msg = "optional phase Slurm resource strings must be non-empty when present"
            raise ValueError(msg)
        validate_exact_slurm_node(self.nodelist, field_name="phase Slurm nodelist")
        if self.nodes is not None and (
            not isinstance(self.nodes, int) or isinstance(self.nodes, bool) or self.nodes < 1
        ):
            msg = "phase Slurm nodes must be a positive integer when present"
            raise ValueError(msg)
        if not isinstance(self.tasks_per_node, int) or isinstance(self.tasks_per_node, bool) or self.tasks_per_node < 1:
            msg = "phase Slurm tasks_per_node must be a positive integer"
            raise ValueError(msg)
        if self.gpus_per_task is not None and (
            not isinstance(self.gpus_per_task, int) or isinstance(self.gpus_per_task, bool) or self.gpus_per_task < 0
        ):
            msg = "phase Slurm gpus_per_task must be a non-negative integer when present"
            raise ValueError(msg)
        if self.max_parallel is not None and (
            not isinstance(self.max_parallel, int) or isinstance(self.max_parallel, bool) or self.max_parallel < 1
        ):
            msg = "phase Slurm max_parallel must be a positive integer when present"
            raise ValueError(msg)
        if self.max_parallel is not None and self.nodes is not None and self.max_parallel > self.nodes:
            msg = "phase Slurm max_parallel must not exceed nodes"
            raise ValueError(msg)
        if self.nodes is None:
            if self.tasks_per_node != 1:
                msg = "phase Slurm tasks_per_node requires typed topology (nodes)"
                raise ValueError(msg)
            if self.gpus_per_task is not None:
                msg = "phase Slurm gpus_per_task requires typed topology (nodes)"
                raise ValueError(msg)
            if self.max_parallel is not None:
                msg = "phase Slurm max_parallel requires typed topology (nodes)"
                raise ValueError(msg)
        else:
            if self.gpus_per_task is None:
                msg = "phase Slurm typed topology (nodes) requires gpus_per_task"
                raise ValueError(msg)
            if self.nodes == 1 and self.tasks_per_node == 1:
                msg = "phase Slurm typed topology cannot be the degenerate one-worker shape (nodes=1, tasks_per_node=1)"
                raise ValueError(msg)

    @property
    def workers(self) -> int:
        return (self.nodes if self.nodes is not None else 1) * self.tasks_per_node

    @property
    def is_packed(self) -> bool:
        """Return True when this topology packs more than one fold worker."""
        return self.workers > 1

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "partition": self.partition,
            "cpus_per_task": self.cpus_per_task,
            "memory": self.memory,
            "time": self.time,
            "gres": self.gres,
            "array": self.array,
        }
        if self.nodelist is not None:
            result["nodelist"] = self.nodelist
        if self.nodes is not None:
            result["nodes"] = self.nodes
        if self.tasks_per_node != 1:
            result["tasks_per_node"] = self.tasks_per_node
        if self.gpus_per_task is not None:
            result["gpus_per_task"] = self.gpus_per_task
        if self.max_parallel is not None:
            result["max_parallel"] = self.max_parallel
        return result


@dataclass(frozen=True)
class ResolvedClusterSnapshot:
    """Deliberately curated operational selections from one resolved Cluster Profile."""

    profile_name: str
    owner: str
    transport: TransportKind
    ssh_target: str | None
    account: str
    project_root: str
    output_root: str
    staging_root: str
    orchestration_repo: str
    runtime_image: str
    preprocessing_runtime: QualifiedPreprocessingRuntimeSelection
    source_bundle_root: str | None
    runtime_image_cache_root: str | None
    runtime_qualification_root: str | None
    runtime_qualification_expires_hours: int
    extra_mounts: tuple[PhaseMountSnapshot, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "ResolvedClusterSnapshot")
        values = (
            self.profile_name,
            self.owner,
            self.account,
            self.project_root,
            self.output_root,
            self.staging_root,
            self.orchestration_repo,
            self.runtime_image,
        )
        if any(not value for value in values):
            msg = "resolved cluster snapshot selections must be non-empty"
            raise ValueError(msg)
        if self.runtime_image != self.preprocessing_runtime.qualification_tuple.cluster_image_path:
            msg = "resolved cluster runtime_image must equal the qualified preprocessing image path"
            raise ValueError(msg)
        if self.transport not in {"ssh", "local-slurm"}:
            msg = f"unsupported phase transport: {self.transport!r}"
            raise ValueError(msg)
        if (self.transport == "ssh") != (self.ssh_target is not None):
            msg = "resolved cluster snapshot ssh_target must match its transport"
            raise ValueError(msg)
        if self.ssh_target == "":
            msg = "resolved cluster snapshot ssh_target must be non-empty when present"
            raise ValueError(msg)
        optional_paths = (
            self.source_bundle_root,
            self.runtime_image_cache_root,
            self.runtime_qualification_root,
        )
        if any(value == "" for value in optional_paths):
            msg = "optional resolved cluster paths must be non-empty when present"
            raise ValueError(msg)
        if (
            not isinstance(self.runtime_qualification_expires_hours, int)
            or isinstance(self.runtime_qualification_expires_hours, bool)
            or self.runtime_qualification_expires_hours <= 0
        ):
            msg = "runtime_qualification_expires_hours must be positive"
            raise ValueError(msg)
        if not isinstance(self.extra_mounts, tuple) or any(
            not isinstance(mount, PhaseMountSnapshot) for mount in self.extra_mounts
        ):
            msg = "resolved cluster extra_mounts must be an immutable tuple of PhaseMountSnapshot records"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_name": self.profile_name,
            "owner": self.owner,
            "transport": self.transport,
            "ssh_target": self.ssh_target,
            "account": self.account,
            "project_root": self.project_root,
            "output_root": self.output_root,
            "staging_root": self.staging_root,
            "orchestration_repo": self.orchestration_repo,
            "runtime_image": self.runtime_image,
            "preprocessing_runtime": self.preprocessing_runtime.to_mapping(),
            "source_bundle_root": self.source_bundle_root,
            "runtime_image_cache_root": self.runtime_image_cache_root,
            "runtime_qualification_root": self.runtime_qualification_root,
            "runtime_qualification_expires_hours": self.runtime_qualification_expires_hours,
            "extra_mounts": [mount.to_mapping() for mount in self.extra_mounts],
        }


@dataclass(frozen=True)
class PreprocessingPhasePlanPayload:
    """Compatibility Port intent and Database Set selection authored into a Phase Plan."""

    work_plan: PreprocessingWorkPlan
    chunk_execution_intent: PreprocessingChunkExecutionIntent
    database: DatabaseSetSelection
    transport: PhaseSeamTransportPolicy = "local"
    s3_publish_prefix: str | None = None
    phase_kind: PhaseKind = "preprocessing"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingPhasePlanPayload")
        if self.phase_kind != "preprocessing":
            msg = "preprocessing Phase Plan payload must declare phase_kind 'preprocessing'"
            raise ValueError(msg)
        _validate_compatibility_port_records(self.work_plan, self.chunk_execution_intent)
        _validate_preprocessing_seam_transport(self.transport, self.s3_publish_prefix)

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "work_plan": self.work_plan.to_mapping(),
            "chunk_execution_intent": self.chunk_execution_intent.to_mapping(),
            "database": self.database.to_mapping(),
        }
        if self.transport != "local":
            result["transport"] = self.transport
        if self.s3_publish_prefix is not None:
            result["s3_publish_prefix"] = self.s3_publish_prefix
        return result


@dataclass(frozen=True)
class PhasePlan:
    """Versioned user-authored intent for one preprocessing phase."""

    target_cluster: str
    input_location: VerifiedLocalInputLocation | VerifiedRemoteInputLocation
    payload: PreprocessingPhasePlanPayload
    phase_kind: PhaseKind = "preprocessing"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PhasePlan")
        if self.phase_kind != "preprocessing" or self.payload.phase_kind != self.phase_kind:
            msg = "Phase Plan envelope and payload must both declare phase_kind 'preprocessing'"
            raise ValueError(msg)
        if not self.target_cluster:
            msg = "Phase Plan target_cluster must be non-empty"
            raise ValueError(msg)
        source_path = self.payload.work_plan.input.source_path
        if isinstance(self.input_location, VerifiedLocalInputLocation):
            if self.input_location.path != source_path:
                msg = "Phase Plan input location path must match the embedded work-plan source path"
                raise ValueError(msg)
        else:
            if self.input_location.path != source_path:
                msg = "Phase Plan remote input location path must match the embedded work-plan source path"
                raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "target_cluster": self.target_cluster,
            "input_location": self.input_location.to_mapping(),
            "payload": self.payload.to_mapping(),
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class PreprocessingRuntimeAction:
    """The sole bounded preprocessing chunk action in the first phase slice."""

    action_id: str
    dependencies: tuple[str, ...]
    resources: PhaseSlurmResources
    payload: PreprocessingChunkExecutionPlan
    action_kind: RuntimeActionKind = "preprocessing-chunk"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingRuntimeAction")
        if self.action_kind != "preprocessing-chunk":
            msg = "preprocessing Runtime Action must declare action_kind 'preprocessing-chunk'"
            raise ValueError(msg)
        if re.fullmatch(r"preprocessing-chunk-[0-9]{6}", self.action_id) is None:
            msg = "preprocessing Runtime Action id must use a six-digit chunk ordinal"
            raise ValueError(msg)
        if not isinstance(self.dependencies, tuple) or any(not item for item in self.dependencies):
            msg = "Runtime Action dependencies must be an immutable tuple of non-empty ids"
            raise ValueError(msg)
        if len(set(self.dependencies)) != len(self.dependencies):
            msg = "Runtime Action dependencies must be unique"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_kind": self.action_kind,
            "action_id": self.action_id,
            "dependencies": list(self.dependencies),
            "resources": self.resources.to_mapping(),
            "payload": self.payload.to_mapping(),
        }


@dataclass(frozen=True)
class PreprocessingPhaseRunSpecPayload:
    """Concrete preprocessing work and action graph for one Phase Attempt."""

    work_plan: PreprocessingWorkPlan
    actions: tuple[PreprocessingRuntimeAction, ...]
    database: PreprocessingDatabaseBinding
    transport: PhaseSeamTransportPolicy = "local"
    s3_publish_prefix: str | None = None
    phase_kind: PhaseKind = "preprocessing"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingPhaseRunSpecPayload")
        if self.phase_kind != "preprocessing":
            msg = "preprocessing Phase RunSpec payload must declare phase_kind 'preprocessing'"
            raise ValueError(msg)
        if not isinstance(self.actions, tuple):
            msg = "Phase RunSpec actions must be an immutable tuple"
            raise ValueError(msg)
        _validate_action_graph(self.actions)
        if len(self.actions) != 1:
            msg = "the first preprocessing slice requires exactly one Runtime Action"
            raise ValueError(msg)
        action = self.actions[0]
        if action.dependencies:
            msg = "the first preprocessing Runtime Action must have no dependencies"
            raise ValueError(msg)
        if self.database.staging is not None:
            action_wall_seconds = parse_slurm_time_seconds(action.resources.time)
            if self.database.staging.lock_wait_seconds > action_wall_seconds:
                msg = "database staging lock_wait_seconds cannot exceed the Runtime Action wall time"
                raise ValueError(msg)
        _validate_compatibility_port_records(self.work_plan, action.payload)
        _validate_database_action_binding(self.database, action.payload)
        expected_action_id = f"preprocessing-chunk-{self.work_plan.chunks[0].ordinal:06d}"
        if action.action_id != expected_action_id:
            msg = "preprocessing Runtime Action id must match the embedded chunk ordinal"
            raise ValueError(msg)
        _validate_preprocessing_seam_transport(self.transport, self.s3_publish_prefix)

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "work_plan": self.work_plan.to_mapping(),
            "actions": [action.to_mapping() for action in self.actions],
            "database": self.database.to_mapping(),
        }
        if self.transport != "local":
            result["transport"] = self.transport
        if self.s3_publish_prefix is not None:
            result["s3_publish_prefix"] = self.s3_publish_prefix
        return result


@dataclass(frozen=True)
class PhaseRunSpec:
    """Immutable concrete execution contract for one preprocessing Phase Attempt."""

    phase_run_id: str
    attempt_id: str
    phase_plan_digest: str
    materialized_at: str
    input_location: VerifiedLocalInputLocation | VerifiedRemoteInputLocation
    cluster: ResolvedClusterSnapshot
    payload: PreprocessingPhaseRunSpecPayload
    carry_forward: AttemptCarryForwardReference | None = None
    phase_kind: PhaseKind = "preprocessing"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PhaseRunSpec")
        _validate_phase_run_id(self.phase_run_id)
        _validate_attempt_id(self.attempt_id)
        if _SHA256.fullmatch(self.phase_plan_digest) is None:
            msg = "Phase RunSpec phase_plan_digest must be a lowercase SHA-256"
            raise ValueError(msg)
        _validate_timestamp(self.materialized_at, "Phase RunSpec materialized_at")
        if self.phase_kind != "preprocessing" or self.payload.phase_kind != self.phase_kind:
            msg = "Phase RunSpec envelope and payload must both declare phase_kind 'preprocessing'"
            raise ValueError(msg)
        if self.input_location.path != self.payload.work_plan.input.source_path:
            msg = "Phase RunSpec input location path must match the embedded work-plan source path"
            raise ValueError(msg)
        if self.carry_forward is not None:
            expected_location = f"attempts/{self.attempt_id}/attempt-carry-forward.json"
            if self.carry_forward.location != expected_location:
                raise ValueError("Phase RunSpec carry-forward reference must match its Attempt")
        expected_manifest_location = f"attempts/{self.attempt_id}/database-source-manifest.json"
        if self.payload.database.source_manifest_projection != expected_manifest_location:
            raise ValueError("Phase RunSpec database source-manifest projection must match its Attempt")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_plan_digest": self.phase_plan_digest,
            "materialized_at": self.materialized_at,
            "input_location": self.input_location.to_mapping(),
            "cluster": self.cluster.to_mapping(),
            "payload": self.payload.to_mapping(),
        }
        if self.carry_forward is not None:
            result["carry_forward"] = self.carry_forward.to_mapping()
        return result

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class FoldingPhasePlanPayload:
    """Folding intent authored into a Phase Plan: one consumed MSA set plus backend."""

    msa_set: MsaSetConsumption
    backend: str
    msa_set_manifest: MsaArtifactSetManifest | None = None
    transport: PhaseSeamTransportPolicy = "publish-to-s3"
    s3_prediction_prefix: str | None = None
    phase_kind: PhaseKind = "folding"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION
    bioir_model_policy: BioIRModelPolicy | None = None
    evidence_profile: str | None = None

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingPhasePlanPayload")
        if self.phase_kind != "folding":
            msg = "folding Phase Plan payload must declare phase_kind 'folding'"
            raise ValueError(msg)
        if self.backend not in FOLDING_BACKENDS:
            msg = f"unsupported folding backend: {self.backend!r}"
            raise ValueError(msg)
        _validate_bioir_model_policy(self.backend, self.bioir_model_policy)
        _validate_folding_evidence_profile(self.evidence_profile, self.backend, self.bioir_model_policy)
        if self.transport not in get_args(PhaseSeamTransportPolicy):
            msg = f"unsupported phase seam transport policy: {self.transport!r}"
            raise ValueError(msg)
        _validate_folding_prediction_prefix(self.transport, self.s3_prediction_prefix)
        from bspp.orchestration.contract.folding_input import MsaSetConsumption

        if not isinstance(self.msa_set, MsaSetConsumption):
            msg = "folding Phase Plan payload must contain an exact MsaSetConsumption record"
            raise ValueError(msg)
        if self.evidence_profile is not None:
            from .folding_artifact_evidence import MAX_ARTIFACT_TARGETS

            if len(self.msa_set.member_a3m_paths) > MAX_ARTIFACT_TARGETS:
                raise ValueError("artifact-backed folding exceeds its fixed target bound")
        if self.msa_set_manifest is not None:
            from bspp.orchestration.contract.preprocessing_handoff import MsaArtifactSetManifest

            if not isinstance(self.msa_set_manifest, MsaArtifactSetManifest):
                raise ValueError("folding Phase Plan msa_set_manifest must be an MsaArtifactSetManifest")
            self.msa_set.validate_against_manifest(self.msa_set_manifest)

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "msa_set": self.msa_set.to_mapping(),
            "backend": self.backend,
            "transport": self.transport,
        }
        if self.s3_prediction_prefix is not None:
            result["s3_prediction_prefix"] = self.s3_prediction_prefix
        if self.msa_set_manifest is not None:
            result["msa_set_manifest"] = self.msa_set_manifest.to_mapping()
        if self.bioir_model_policy is not None:
            result["bioir_model_policy"] = self.bioir_model_policy.to_mapping()
        if self.evidence_profile is not None:
            result["evidence_profile"] = self.evidence_profile
        return result


@dataclass(frozen=True)
class PhaseSeamTransport:
    """Per-seam transport policy binding one phase seam to a transport mode."""

    seam: str
    policy: PhaseSeamTransportPolicy
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PhaseSeamTransport")
        if not self.seam:
            msg = "phase seam transport seam must be non-empty"
            raise ValueError(msg)
        if self.policy not in get_args(PhaseSeamTransportPolicy):
            msg = f"unsupported phase seam transport policy: {self.policy!r}"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seam": self.seam,
            "policy": self.policy,
        }


@dataclass(frozen=True)
class FoldingActionPayload:
    """One folding Runtime Action payload: a kind plus ordered string params."""

    action_kind: FoldingRuntimeActionKind
    params: tuple[tuple[str, str], ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingActionPayload")
        if self.action_kind not in {"msa-flatten", "split", "preprocess", "fold", "canonical-pair"}:
            msg = f"unsupported folding Runtime Action kind: {self.action_kind!r}"
            raise ValueError(msg)
        if not isinstance(self.params, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not isinstance(item[1], str)
            or not item[1]
            for item in self.params
        ):
            msg = "folding Runtime Action params must be an immutable tuple of non-empty (key, value) pairs"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_kind": self.action_kind,
            "params": [[key, value] for key, value in self.params],
        }


@dataclass(frozen=True)
class FoldingRuntimeAction:
    """One folding Runtime Action in the run-spec action graph."""

    action_id: str
    dependencies: tuple[str, ...]
    resources: PhaseSlurmResources
    payload: FoldingActionPayload
    action_kind: FoldingRuntimeActionKind
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingRuntimeAction")
        if self.action_kind not in {"msa-flatten", "split", "preprocess", "fold", "canonical-pair"}:
            msg = f"unsupported folding Runtime Action kind: {self.action_kind!r}"
            raise ValueError(msg)
        if re.fullmatch(rf"{self.action_kind}-[0-9]{{6}}", self.action_id) is None:
            msg = f"folding Runtime Action id must match its {self.action_kind!r} kind with a six-digit ordinal"
            raise ValueError(msg)
        if not isinstance(self.dependencies, tuple) or any(not item for item in self.dependencies):
            msg = "folding Runtime Action dependencies must be an immutable tuple of non-empty ids"
            raise ValueError(msg)
        if len(set(self.dependencies)) != len(self.dependencies):
            msg = "folding Runtime Action dependencies must be unique"
            raise ValueError(msg)
        if not isinstance(self.resources, PhaseSlurmResources):
            msg = "folding Runtime Action resources must be an exact PhaseSlurmResources record"
            raise ValueError(msg)
        if not isinstance(self.payload, FoldingActionPayload):
            msg = "folding Runtime Action payload must be an exact FoldingActionPayload record"
            raise ValueError(msg)
        if self.payload.action_kind != self.action_kind:
            msg = "folding Runtime Action payload kind must match its action_kind"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_kind": self.action_kind,
            "action_id": self.action_id,
            "dependencies": list(self.dependencies),
            "resources": self.resources.to_mapping(),
            "payload": self.payload.to_mapping(),
        }


@dataclass(frozen=True)
class FoldingPhaseRunSpecPayload:
    """Concrete folding action graph for one Phase Attempt."""

    msa_set: MsaSetConsumption
    backend: str
    actions: tuple[FoldingRuntimeAction, ...]
    msa_set_manifest: MsaArtifactSetManifest | None = None
    fold_shard_projection: FoldShardProjectionBinding | None = None
    transport: PhaseSeamTransportPolicy = "publish-to-s3"
    s3_prediction_prefix: str | None = None
    phase_kind: PhaseKind = "folding"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION
    bioir_model_policy: BioIRModelPolicy | None = None
    evidence_profile: str | None = None

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingPhaseRunSpecPayload")
        if self.phase_kind != "folding":
            msg = "folding Phase RunSpec payload must declare phase_kind 'folding'"
            raise ValueError(msg)
        if self.backend not in FOLDING_BACKENDS:
            msg = f"unsupported folding backend: {self.backend!r}"
            raise ValueError(msg)
        _validate_bioir_model_policy(self.backend, self.bioir_model_policy)
        _validate_folding_evidence_profile(self.evidence_profile, self.backend, self.bioir_model_policy)
        if self.transport not in get_args(PhaseSeamTransportPolicy):
            msg = f"unsupported phase seam transport policy: {self.transport!r}"
            raise ValueError(msg)
        _validate_folding_prediction_prefix(self.transport, self.s3_prediction_prefix)
        from bspp.orchestration.contract.folding_input import MsaSetConsumption

        if not isinstance(self.msa_set, MsaSetConsumption):
            msg = "folding Phase RunSpec payload must contain an exact MsaSetConsumption record"
            raise ValueError(msg)
        if self.evidence_profile is not None:
            from .folding_artifact_evidence import MAX_ARTIFACT_TARGETS

            if len(self.msa_set.member_a3m_paths) > MAX_ARTIFACT_TARGETS:
                raise ValueError("artifact-backed folding exceeds its fixed target bound")
        if self.msa_set_manifest is not None:
            from bspp.orchestration.contract.preprocessing_handoff import MsaArtifactSetManifest

            if not isinstance(self.msa_set_manifest, MsaArtifactSetManifest):
                raise ValueError("folding Phase RunSpec msa_set_manifest must be an MsaArtifactSetManifest")
            self.msa_set.validate_against_manifest(self.msa_set_manifest)
        if self.fold_shard_projection is not None and not isinstance(
            self.fold_shard_projection, FoldShardProjectionBinding
        ):
            raise ValueError("folding Phase RunSpec fold_shard_projection must be a FoldShardProjectionBinding")
        if not isinstance(self.actions, tuple) or not self.actions:
            msg = "folding Phase RunSpec actions must be a non-empty immutable tuple"
            raise ValueError(msg)
        if any(not isinstance(action, FoldingRuntimeAction) for action in self.actions):
            msg = "folding Phase RunSpec actions must be FoldingRuntimeAction records"
            raise ValueError(msg)
        _validate_folding_action_graph(self.actions)
        if self.evidence_profile is not None:
            for action in self.actions:
                if action.action_kind == "fold" and not action.resources.is_packed:
                    raise ValueError("artifact-backed-v2 requires packed folding")
                if (
                    action.action_kind in {"fold", "canonical-pair"}
                    and dict(action.payload.params).get("evidence_profile") != self.evidence_profile
                ):
                    raise ValueError("folding action does not bind its evidence profile")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "msa_set": self.msa_set.to_mapping(),
            "backend": self.backend,
            "actions": [action.to_mapping() for action in self.actions],
        }
        if self.transport != "publish-to-s3":
            result["transport"] = self.transport
        if self.s3_prediction_prefix is not None:
            result["s3_prediction_prefix"] = self.s3_prediction_prefix
        if self.msa_set_manifest is not None:
            result["msa_set_manifest"] = self.msa_set_manifest.to_mapping()
        if self.fold_shard_projection is not None:
            result["fold_shard_projection"] = self.fold_shard_projection.to_mapping()
        if self.bioir_model_policy is not None:
            result["bioir_model_policy"] = self.bioir_model_policy.to_mapping()
        if self.evidence_profile is not None:
            result["evidence_profile"] = self.evidence_profile
        return result


@dataclass(frozen=True)
class FoldingResolvedClusterSnapshot:
    """Curated operational selections for one folding Cluster Profile."""

    profile_name: str
    owner: str
    transport: TransportKind
    ssh_target: str | None
    account: str
    project_root: str
    staging_root: str
    orchestration_repo: str
    runtime_image: str
    extra_mounts: tuple[PhaseMountSnapshot, ...]
    backend_assets: FoldingBackendAssetsSnapshot | None = None
    release_preset: str = "public"
    mount_orchestration_source: bool = False
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingResolvedClusterSnapshot")
        if any(
            not value
            for value in (
                self.profile_name,
                self.owner,
                self.account,
                self.project_root,
                self.staging_root,
                self.orchestration_repo,
                self.runtime_image,
            )
        ):
            msg = "folding resolved cluster snapshot required selections must be non-empty"
            raise ValueError(msg)
        if self.transport not in {"ssh", "local-slurm"}:
            msg = f"unsupported folding transport: {self.transport!r}"
            raise ValueError(msg)
        if (self.transport == "ssh") != (self.ssh_target is not None):
            msg = "folding resolved cluster snapshot ssh_target must match its transport"
            raise ValueError(msg)
        if self.ssh_target == "":
            msg = "folding resolved cluster snapshot ssh_target must be non-empty when present"
            raise ValueError(msg)
        if not isinstance(self.extra_mounts, tuple) or any(
            not isinstance(mount, PhaseMountSnapshot) for mount in self.extra_mounts
        ):
            msg = "folding resolved cluster extra_mounts must be an immutable tuple of PhaseMountSnapshot records"
            raise ValueError(msg)
        if self.backend_assets is not None:
            from bspp.orchestration.contract.folding_execution import FoldingBackendAssetsSnapshot

            if not isinstance(self.backend_assets, FoldingBackendAssetsSnapshot):
                raise ValueError("folding resolved cluster backend_assets must be a FoldingBackendAssetsSnapshot")
        if self.release_preset not in {"public"}:
            msg = f"unsupported folding release_preset: {self.release_preset!r}"
            raise ValueError(msg)
        if not isinstance(self.mount_orchestration_source, bool):
            raise TypeError("FoldingResolvedClusterSnapshot.mount_orchestration_source must be a bool")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "profile_name": self.profile_name,
            "owner": self.owner,
            "transport": self.transport,
            "ssh_target": self.ssh_target,
            "account": self.account,
            "project_root": self.project_root,
            "staging_root": self.staging_root,
            "orchestration_repo": self.orchestration_repo,
            "runtime_image": self.runtime_image,
            "extra_mounts": [mount.to_mapping() for mount in self.extra_mounts],
        }
        # The default (public) preset is omitted from the serialized form so
        # that pre-field folding RunSpecs re-serialize to their original bytes
        # and keep their canonical digests. Only a non-default preset is
        # emitted, which is what makes it observable in qualification identity.
        if self.release_preset != "public":
            result["release_preset"] = self.release_preset
        if self.backend_assets is not None:
            result["backend_assets"] = self.backend_assets.to_mapping()
        if self.mount_orchestration_source:
            result["mount_orchestration_source"] = True
        return result


@dataclass(frozen=True)
class FoldingPhasePlan:
    """Versioned user-authored intent for one folding phase."""

    target_cluster: str
    input_location: VerifiedLocalBundledArtifactLocation | VerifiedRemoteBundledArtifactLocation
    payload: FoldingPhasePlanPayload
    phase_kind: PhaseKind = "folding"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingPhasePlan")
        if self.phase_kind != "folding" or self.payload.phase_kind != self.phase_kind:
            msg = "folding Phase Plan envelope and payload must both declare phase_kind 'folding'"
            raise ValueError(msg)
        if not self.target_cluster:
            msg = "folding Phase Plan target_cluster must be non-empty"
            raise ValueError(msg)
        from bspp.orchestration.contract.preprocessing_handoff import (
            VerifiedLocalBundledArtifactLocation,
            VerifiedRemoteBundledArtifactLocation,
        )

        if not isinstance(
            self.input_location,
            VerifiedLocalBundledArtifactLocation | VerifiedRemoteBundledArtifactLocation,
        ):
            msg = "folding Phase Plan input location must be a verified bundled Artifact Location"
            raise ValueError(msg)
        if self.input_location.artifact_set_id != self.payload.msa_set.artifact_set_id:
            msg = "folding Phase Plan input location must reference the consumed MSA Artifact Set"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "target_cluster": self.target_cluster,
            "input_location": self.input_location.to_mapping(),
            "payload": self.payload.to_mapping(),
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class FoldingPhaseRunSpec:
    """Immutable concrete execution contract for one folding Phase Attempt."""

    phase_run_id: str
    attempt_id: str
    phase_plan_digest: str
    materialized_at: str
    input_location: VerifiedLocalBundledArtifactLocation | VerifiedRemoteBundledArtifactLocation
    cluster: FoldingResolvedClusterSnapshot
    payload: FoldingPhaseRunSpecPayload
    carry_forward: FoldingCarryForwardReference | None = None
    phase_kind: PhaseKind = "folding"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "FoldingPhaseRunSpec")
        _validate_phase_run_id(self.phase_run_id)
        _validate_attempt_id(self.attempt_id)
        if _SHA256.fullmatch(self.phase_plan_digest) is None:
            msg = "folding Phase RunSpec phase_plan_digest must be a lowercase SHA-256"
            raise ValueError(msg)
        _validate_timestamp(self.materialized_at, "folding Phase RunSpec materialized_at")
        if self.phase_kind != "folding" or self.payload.phase_kind != self.phase_kind:
            msg = "folding Phase RunSpec envelope and payload must both declare phase_kind 'folding'"
            raise ValueError(msg)
        if not isinstance(self.cluster, FoldingResolvedClusterSnapshot):
            msg = "folding Phase RunSpec cluster must be a FoldingResolvedClusterSnapshot"
            raise ValueError(msg)
        assets = self.cluster.backend_assets
        if self.payload.bioir_model_policy is not None:
            if assets is None or assets.backend != "bioir" or assets.bioir_monomer_checkpoint is None:
                raise ValueError("BioIR model policy requires bioir_monomer_checkpoint assets")
        elif assets is not None and assets.bioir_monomer_checkpoint is not None:
            raise ValueError("bioir_monomer_checkpoint requires an explicit BioIR model policy")
        from bspp.orchestration.contract.preprocessing_handoff import (
            VerifiedLocalBundledArtifactLocation,
            VerifiedRemoteBundledArtifactLocation,
        )

        if not isinstance(
            self.input_location,
            VerifiedLocalBundledArtifactLocation | VerifiedRemoteBundledArtifactLocation,
        ):
            msg = "folding Phase RunSpec input location must be a verified bundled Artifact Location"
            raise ValueError(msg)
        if self.input_location.artifact_set_id != self.payload.msa_set.artifact_set_id:
            msg = "folding Phase RunSpec input location must reference the consumed MSA Artifact Set"
            raise ValueError(msg)
        if self.carry_forward is not None:
            expected_location = f"attempts/{self.attempt_id}/folding-carry-forward.json"
            if self.carry_forward.location != expected_location:
                raise ValueError("folding Phase RunSpec carry-forward reference must match its Attempt")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_plan_digest": self.phase_plan_digest,
            "materialized_at": self.materialized_at,
            "input_location": self.input_location.to_mapping(),
            "cluster": self.cluster.to_mapping(),
            "payload": self.payload.to_mapping(),
        }
        if self.carry_forward is not None:
            result["carry_forward"] = self.carry_forward.to_mapping()
        return result

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


def phase_plan_from_mapping(payload: Mapping[str, object]) -> PhasePlan:
    """Strict-load one preprocessing Phase Plan, including every nested record version."""
    reject_environment_interpolation(payload, context="Phase Plan")
    _require_explicit_schema_versions(payload, path=("phase_plan",))
    _reject_unknown_fields(
        payload,
        {"schema_version", "phase_kind", "target_cluster", "input_location", "payload"},
        "PhasePlan",
    )
    kind = _preprocessing_phase_kind(payload, "phase_kind", record_name="PhasePlan")
    plan = PhasePlan(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PhasePlan"),
        phase_kind=kind,
        target_cluster=_required_str(payload, "target_cluster"),
        input_location=preprocessing_input_location_from_mapping(_required_mapping(payload, "input_location")),
        payload=preprocessing_phase_plan_payload_from_mapping(_required_mapping(payload, "payload")),
    )
    return plan


def phase_runspec_from_mapping(payload: Mapping[str, object]) -> PhaseRunSpec:
    """Strict-load one concrete preprocessing Phase RunSpec."""
    reject_environment_interpolation(payload, context="Phase RunSpec")
    _require_explicit_schema_versions(payload, path=("phase_runspec",))
    _require_explicit_runspec_database_source_manifest_version(payload)
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "phase_kind",
            "phase_run_id",
            "attempt_id",
            "phase_plan_digest",
            "materialized_at",
            "input_location",
            "cluster",
            "payload",
            "carry_forward",
        },
        "PhaseRunSpec",
    )
    carry_mapping = payload.get("carry_forward")
    if "carry_forward" in payload and not isinstance(carry_mapping, Mapping):
        raise ValueError("carry_forward must be a mapping when present")
    return PhaseRunSpec(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PhaseRunSpec"),
        phase_kind=_preprocessing_phase_kind(payload, "phase_kind", record_name="PhaseRunSpec"),
        phase_run_id=_required_str(payload, "phase_run_id"),
        attempt_id=_required_str(payload, "attempt_id"),
        phase_plan_digest=_required_str(payload, "phase_plan_digest"),
        materialized_at=_required_str(payload, "materialized_at"),
        input_location=preprocessing_input_location_from_mapping(_required_mapping(payload, "input_location")),
        cluster=resolved_cluster_snapshot_from_mapping(_required_mapping(payload, "cluster")),
        payload=preprocessing_phase_runspec_payload_from_mapping(_required_mapping(payload, "payload")),
        carry_forward=(
            attempt_carry_forward_reference_from_mapping(cast("Mapping[str, object]", carry_mapping))
            if carry_mapping is not None
            else None
        ),
    )


def phase_plan_family_from_mapping(
    payload: Mapping[str, object],
) -> PhasePlan | PostprocessingPhasePlan | FoldingPhasePlan:
    """Dispatch a Phase Plan only through its exact top-level family discriminator."""
    kind = payload.get("phase_kind")
    if kind == "preprocessing":
        return phase_plan_from_mapping(payload)
    if kind == "folding":
        return folding_phase_plan_from_mapping(payload)
    if kind == "postprocessing":
        from bspp.orchestration.contract.postprocessing_plan import (
            postprocessing_phase_plan_from_mapping,
        )

        return postprocessing_phase_plan_from_mapping(payload)
    raise ValueError(f"unsupported Phase Plan phase_kind: {kind!r}")


def phase_runspec_family_from_mapping(
    payload: Mapping[str, object],
) -> PhaseRunSpec | ExecutablePostprocessingPhaseRunSpec | FoldingPhaseRunSpec:
    """Dispatch a Phase RunSpec only through its exact top-level family discriminator."""
    kind = payload.get("phase_kind")
    if kind == "preprocessing":
        return phase_runspec_from_mapping(payload)
    if kind == "folding":
        return folding_phase_runspec_from_mapping(payload)
    if kind == "postprocessing":
        from bspp.orchestration.contract.postprocessing_runspec import (
            postprocessing_phase_runspec_from_mapping,
        )

        return postprocessing_phase_runspec_from_mapping(payload)
    raise ValueError(f"unsupported Phase RunSpec phase_kind: {kind!r}")


def verified_local_input_location_from_mapping(payload: Mapping[str, object]) -> VerifiedLocalInputLocation:
    _reject_unknown_fields(
        payload,
        {"schema_version", "kind", "path", "sha256", "size_bytes"},
        "VerifiedLocalInputLocation",
    )
    kind = _required_str(payload, "kind")
    if kind != "verified-local-file":
        msg = f"unsupported input location kind: {kind!r}"
        raise ValueError(msg)
    return VerifiedLocalInputLocation(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="VerifiedLocalInputLocation"),
        kind=cast("InputLocationKind", kind),
        path=_required_str(payload, "path"),
        sha256=_required_str(payload, "sha256"),
        size_bytes=_required_int(payload, "size_bytes"),
    )


def verified_remote_input_location_from_mapping(payload: Mapping[str, object]) -> VerifiedRemoteInputLocation:
    _reject_unknown_fields(
        payload,
        {"schema_version", "kind", "source_uri", "sha256", "size_bytes", "path"},
        "VerifiedRemoteInputLocation",
    )
    kind = _required_str(payload, "kind")
    if kind != "verified-remote-file":
        msg = f"unsupported input location kind: {kind!r}"
        raise ValueError(msg)
    return VerifiedRemoteInputLocation(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="VerifiedRemoteInputLocation"
        ),
        kind=cast("InputLocationKind", kind),
        source_uri=_required_str(payload, "source_uri"),
        sha256=_required_str(payload, "sha256"),
        size_bytes=_required_int(payload, "size_bytes"),
        path=_required_str(payload, "path"),
    )


def preprocessing_input_location_from_mapping(
    payload: Mapping[str, object],
) -> VerifiedLocalInputLocation | VerifiedRemoteInputLocation:
    """Dispatch a preprocessing input location mapping through its exact kind discriminator."""
    kind = _required_str(payload, "kind")
    if kind == "verified-local-file":
        return verified_local_input_location_from_mapping(payload)
    if kind == "verified-remote-file":
        return verified_remote_input_location_from_mapping(payload)
    msg = f"unsupported input location kind: {kind!r}"
    raise ValueError(msg)


def preprocessing_phase_plan_payload_from_mapping(payload: Mapping[str, object]) -> PreprocessingPhasePlanPayload:
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "phase_kind",
            "work_plan",
            "chunk_execution_intent",
            "database",
            "transport",
            "s3_publish_prefix",
        },
        "PreprocessingPhasePlanPayload",
    )
    return PreprocessingPhasePlanPayload(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingPhasePlanPayload"
        ),
        phase_kind=_preprocessing_phase_kind(payload, "phase_kind", record_name="PreprocessingPhasePlanPayload"),
        work_plan=preprocessing_work_plan_from_mapping(_required_mapping(payload, "work_plan")),
        chunk_execution_intent=preprocessing_chunk_execution_intent_from_mapping(
            _required_mapping(payload, "chunk_execution_intent")
        ),
        database=database_set_selection_from_mapping(_required_mapping(payload, "database")),
        transport=_preprocessing_seam_transport_policy(payload, "transport"),
        s3_publish_prefix=_optional_str(payload, "s3_publish_prefix"),
    )


def phase_mount_snapshot_from_mapping(payload: Mapping[str, object]) -> PhaseMountSnapshot:
    _reject_unknown_fields(payload, {"schema_version", "source", "target", "read_only"}, "PhaseMountSnapshot")
    read_only = payload.get("read_only", False)
    if not isinstance(read_only, bool):
        raise ValueError("PhaseMountSnapshot read_only must be a boolean")
    return PhaseMountSnapshot(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PhaseMountSnapshot"),
        source=_required_str(payload, "source"),
        target=_required_str(payload, "target"),
        read_only=read_only,
    )


def phase_slurm_resources_from_mapping(payload: Mapping[str, object]) -> PhaseSlurmResources:
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "partition",
            "cpus_per_task",
            "memory",
            "time",
            "gres",
            "array",
            "nodelist",
            "nodes",
            "tasks_per_node",
            "gpus_per_task",
            "max_parallel",
        },
        "PhaseSlurmResources",
    )
    return PhaseSlurmResources(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PhaseSlurmResources"),
        partition=_required_str(payload, "partition"),
        cpus_per_task=_required_int(payload, "cpus_per_task"),
        memory=_required_str(payload, "memory"),
        time=_required_str(payload, "time"),
        gres=_optional_str(payload, "gres"),
        array=_optional_str(payload, "array"),
        nodelist=_optional_str(payload, "nodelist"),
        nodes=_optional_int(payload, "nodes"),
        tasks_per_node=_optional_int(payload, "tasks_per_node", default=1),
        gpus_per_task=_optional_int(payload, "gpus_per_task"),
        max_parallel=_optional_int(payload, "max_parallel"),
    )


def resolved_cluster_snapshot_from_mapping(payload: Mapping[str, object]) -> ResolvedClusterSnapshot:
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "profile_name",
            "owner",
            "transport",
            "ssh_target",
            "account",
            "project_root",
            "output_root",
            "staging_root",
            "orchestration_repo",
            "runtime_image",
            "preprocessing_runtime",
            "source_bundle_root",
            "runtime_image_cache_root",
            "runtime_qualification_root",
            "runtime_qualification_expires_hours",
            "extra_mounts",
        },
        "ResolvedClusterSnapshot",
    )
    transport = _required_str(payload, "transport")
    if transport not in {"ssh", "local-slurm"}:
        msg = f"unsupported phase transport: {transport!r}"
        raise ValueError(msg)
    return ResolvedClusterSnapshot(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="ResolvedClusterSnapshot"),
        profile_name=_required_str(payload, "profile_name"),
        owner=_required_str(payload, "owner"),
        transport=cast("TransportKind", transport),
        ssh_target=_optional_str(payload, "ssh_target"),
        account=_required_str(payload, "account"),
        project_root=_required_str(payload, "project_root"),
        output_root=_required_str(payload, "output_root"),
        staging_root=_required_str(payload, "staging_root"),
        orchestration_repo=_required_str(payload, "orchestration_repo"),
        runtime_image=_required_str(payload, "runtime_image"),
        preprocessing_runtime=qualified_preprocessing_runtime_selection_from_mapping(
            _required_mapping(payload, "preprocessing_runtime")
        ),
        source_bundle_root=_optional_str(payload, "source_bundle_root"),
        runtime_image_cache_root=_optional_str(payload, "runtime_image_cache_root"),
        runtime_qualification_root=_optional_str(payload, "runtime_qualification_root"),
        runtime_qualification_expires_hours=_required_int(payload, "runtime_qualification_expires_hours"),
        extra_mounts=tuple(
            phase_mount_snapshot_from_mapping(item) for item in _required_mapping_sequence(payload, "extra_mounts")
        ),
    )


def preprocessing_runtime_action_from_mapping(payload: Mapping[str, object]) -> PreprocessingRuntimeAction:
    _reject_unknown_fields(
        payload,
        {"schema_version", "action_kind", "action_id", "dependencies", "resources", "payload"},
        "PreprocessingRuntimeAction",
    )
    action_kind = _required_str(payload, "action_kind")
    if action_kind != "preprocessing-chunk":
        msg = f"unsupported Runtime Action kind: {action_kind!r}"
        raise ValueError(msg)
    return PreprocessingRuntimeAction(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PreprocessingRuntimeAction"),
        action_kind=cast("RuntimeActionKind", action_kind),
        action_id=_required_str(payload, "action_id"),
        dependencies=_required_str_tuple(payload, "dependencies"),
        resources=phase_slurm_resources_from_mapping(_required_mapping(payload, "resources")),
        payload=preprocessing_chunk_execution_plan_from_mapping(_required_mapping(payload, "payload")),
    )


def preprocessing_phase_runspec_payload_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingPhaseRunSpecPayload:
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "phase_kind",
            "work_plan",
            "actions",
            "database",
            "transport",
            "s3_publish_prefix",
        },
        "PreprocessingPhaseRunSpecPayload",
    )
    return PreprocessingPhaseRunSpecPayload(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingPhaseRunSpecPayload"
        ),
        phase_kind=_preprocessing_phase_kind(payload, "phase_kind", record_name="PreprocessingPhaseRunSpecPayload"),
        work_plan=preprocessing_work_plan_from_mapping(_required_mapping(payload, "work_plan")),
        actions=tuple(
            preprocessing_runtime_action_from_mapping(item) for item in _required_mapping_sequence(payload, "actions")
        ),
        database=preprocessing_database_binding_from_mapping(_required_mapping(payload, "database")),
        transport=_preprocessing_seam_transport_policy(payload, "transport"),
        s3_publish_prefix=_optional_str(payload, "s3_publish_prefix"),
    )


def folding_action_payload_from_mapping(payload: Mapping[str, object]) -> FoldingActionPayload:
    _reject_unknown_fields(
        payload,
        {"schema_version", "action_kind", "params"},
        "FoldingActionPayload",
    )
    action_kind = _required_str(payload, "action_kind")
    if action_kind not in {"msa-flatten", "split", "preprocess", "fold", "canonical-pair"}:
        msg = f"unsupported folding Runtime Action kind: {action_kind!r}"
        raise ValueError(msg)
    return FoldingActionPayload(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="FoldingActionPayload"),
        action_kind=cast("FoldingRuntimeActionKind", action_kind),
        params=_required_str_pair_tuple(payload, "params"),
    )


def folding_runtime_action_from_mapping(payload: Mapping[str, object]) -> FoldingRuntimeAction:
    _reject_unknown_fields(
        payload,
        {"schema_version", "action_kind", "action_id", "dependencies", "resources", "payload"},
        "FoldingRuntimeAction",
    )
    action_kind = _required_str(payload, "action_kind")
    if action_kind not in {"msa-flatten", "split", "preprocess", "fold", "canonical-pair"}:
        msg = f"unsupported folding Runtime Action kind: {action_kind!r}"
        raise ValueError(msg)
    return FoldingRuntimeAction(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="FoldingRuntimeAction"),
        action_kind=cast("FoldingRuntimeActionKind", action_kind),
        action_id=_required_str(payload, "action_id"),
        dependencies=_required_str_tuple(payload, "dependencies"),
        resources=phase_slurm_resources_from_mapping(_required_mapping(payload, "resources")),
        payload=folding_action_payload_from_mapping(_required_mapping(payload, "payload")),
    )


def folding_phase_plan_payload_from_mapping(payload: Mapping[str, object]) -> FoldingPhasePlanPayload:
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "phase_kind",
            "msa_set",
            "backend",
            "transport",
            "s3_prediction_prefix",
            "msa_set_manifest",
            "bioir_model_policy",
            "evidence_profile",
        },
        "FoldingPhasePlanPayload",
    )
    from bspp.orchestration.contract.folding_input import msa_set_consumption_from_mapping

    msa_set_manifest = payload.get("msa_set_manifest")
    if msa_set_manifest is not None:
        from bspp.orchestration.contract.preprocessing_handoff import msa_artifact_set_manifest_from_mapping

        msa_set_manifest = msa_artifact_set_manifest_from_mapping(_required_mapping(payload, "msa_set_manifest"))

    return FoldingPhasePlanPayload(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="FoldingPhasePlanPayload"),
        phase_kind=_phase_kind(payload, "phase_kind", record_name="FoldingPhasePlanPayload"),
        msa_set=msa_set_consumption_from_mapping(_required_mapping(payload, "msa_set")),
        backend=_required_str(payload, "backend"),
        msa_set_manifest=msa_set_manifest,
        transport=_seam_transport_policy(payload, "transport"),
        s3_prediction_prefix=_optional_str(payload, "s3_prediction_prefix"),
        bioir_model_policy=_optional_bioir_model_policy(payload),
        evidence_profile=_required_str(payload, "evidence_profile") if "evidence_profile" in payload else None,
    )


def folding_phase_runspec_payload_from_mapping(
    payload: Mapping[str, object],
) -> FoldingPhaseRunSpecPayload:
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "phase_kind",
            "msa_set",
            "backend",
            "actions",
            "transport",
            "s3_prediction_prefix",
            "msa_set_manifest",
            "fold_shard_projection",
            "bioir_model_policy",
            "evidence_profile",
        },
        "FoldingPhaseRunSpecPayload",
    )
    from bspp.orchestration.contract.folding_input import msa_set_consumption_from_mapping

    msa_set_manifest = payload.get("msa_set_manifest")
    if msa_set_manifest is not None:
        from bspp.orchestration.contract.preprocessing_handoff import msa_artifact_set_manifest_from_mapping

        msa_set_manifest = msa_artifact_set_manifest_from_mapping(_required_mapping(payload, "msa_set_manifest"))

    fold_shard_projection = payload.get("fold_shard_projection")
    if fold_shard_projection is not None:
        fold_shard_projection = fold_shard_projection_binding_from_mapping(
            _required_mapping(payload, "fold_shard_projection")
        )

    return FoldingPhaseRunSpecPayload(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="FoldingPhaseRunSpecPayload"),
        phase_kind=_phase_kind(payload, "phase_kind", record_name="FoldingPhaseRunSpecPayload"),
        msa_set=msa_set_consumption_from_mapping(_required_mapping(payload, "msa_set")),
        backend=_required_str(payload, "backend"),
        actions=tuple(
            folding_runtime_action_from_mapping(item) for item in _required_mapping_sequence(payload, "actions")
        ),
        msa_set_manifest=msa_set_manifest,
        fold_shard_projection=fold_shard_projection,
        transport=_seam_transport_policy(payload, "transport"),
        s3_prediction_prefix=_optional_str(payload, "s3_prediction_prefix"),
        bioir_model_policy=_optional_bioir_model_policy(payload),
        evidence_profile=_required_str(payload, "evidence_profile") if "evidence_profile" in payload else None,
    )


def _validate_folding_evidence_profile(profile: str | None, backend: str, policy: BioIRModelPolicy | None) -> None:
    if profile is None:
        return
    if profile != "artifact-backed-v2":
        raise ValueError("unsupported folding evidence profile")
    if backend != "bioir" or policy is None:
        raise ValueError("artifact-backed-v2 requires BioIR with an explicit model policy")


def _validate_bioir_model_policy(backend: str, policy: BioIRModelPolicy | None) -> None:
    if policy is not None:
        if not isinstance(policy, BioIRModelPolicy):
            raise ValueError("bioir_model_policy must be a BioIRModelPolicy")
        if backend != "bioir":
            raise ValueError("bioir_model_policy requires the bioir backend")


def _optional_bioir_model_policy(payload: Mapping[str, object]) -> BioIRModelPolicy | None:
    if "bioir_model_policy" not in payload:
        return None
    return bioir_model_policy_from_mapping(_required_mapping(payload, "bioir_model_policy"))


def folding_resolved_cluster_snapshot_from_mapping(
    payload: Mapping[str, object],
) -> FoldingResolvedClusterSnapshot:
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "profile_name",
            "owner",
            "transport",
            "ssh_target",
            "account",
            "project_root",
            "staging_root",
            "orchestration_repo",
            "runtime_image",
            "extra_mounts",
            "backend_assets",
            "release_preset",
            "mount_orchestration_source",
        },
        "FoldingResolvedClusterSnapshot",
    )
    transport = _required_str(payload, "transport")
    if transport not in {"ssh", "local-slurm"}:
        msg = f"unsupported folding transport: {transport!r}"
        raise ValueError(msg)
    release_preset = payload.get("release_preset", "public")
    if release_preset is None:
        release_preset = "public"
    if not isinstance(release_preset, str) or release_preset not in {"public"}:
        msg = f"unsupported folding release_preset: {release_preset!r}"
        raise ValueError(msg)
    backend_assets = payload.get("backend_assets")
    if backend_assets is not None:
        from bspp.orchestration.contract.folding_execution import folding_backend_assets_snapshot_from_mapping

        backend_assets = folding_backend_assets_snapshot_from_mapping(_required_mapping(payload, "backend_assets"))
    mount = payload.get("mount_orchestration_source", False)
    if not isinstance(mount, bool):
        raise ValueError("FoldingResolvedClusterSnapshot mount_orchestration_source must be a boolean")
    return FoldingResolvedClusterSnapshot(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="FoldingResolvedClusterSnapshot"
        ),
        profile_name=_required_str(payload, "profile_name"),
        owner=_required_str(payload, "owner"),
        transport=cast("TransportKind", transport),
        ssh_target=_optional_str(payload, "ssh_target"),
        account=_required_str(payload, "account"),
        project_root=_required_str(payload, "project_root"),
        staging_root=_required_str(payload, "staging_root"),
        orchestration_repo=_required_str(payload, "orchestration_repo"),
        runtime_image=_required_str(payload, "runtime_image"),
        extra_mounts=tuple(
            phase_mount_snapshot_from_mapping(item) for item in _required_mapping_sequence(payload, "extra_mounts")
        ),
        backend_assets=backend_assets,
        release_preset=release_preset,
        mount_orchestration_source=mount,
    )


def folding_input_location_from_mapping(
    payload: Mapping[str, object],
) -> VerifiedLocalBundledArtifactLocation | VerifiedRemoteBundledArtifactLocation:
    kind = _required_str(payload, "kind")
    if kind == "verified-local-bundled":
        from bspp.orchestration.contract.preprocessing_handoff import (
            verified_local_bundled_artifact_location_from_mapping,
        )

        return verified_local_bundled_artifact_location_from_mapping(payload)
    if kind == "verified-remote-bundled":
        from bspp.orchestration.contract.preprocessing_handoff import (
            verified_remote_bundled_artifact_location_from_mapping,
        )

        return verified_remote_bundled_artifact_location_from_mapping(payload)
    msg = f"unsupported folding input location kind: {kind!r}"
    raise ValueError(msg)


def folding_phase_plan_from_mapping(payload: Mapping[str, object]) -> FoldingPhasePlan:
    reject_environment_interpolation(payload, context="folding Phase Plan")
    _require_explicit_schema_versions(payload, path=("folding_phase_plan",))
    _reject_unknown_fields(
        payload,
        {"schema_version", "phase_kind", "target_cluster", "input_location", "payload"},
        "FoldingPhasePlan",
    )
    kind = _phase_kind(payload, "phase_kind", record_name="FoldingPhasePlan")
    return FoldingPhasePlan(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="FoldingPhasePlan"),
        phase_kind=kind,
        target_cluster=_required_str(payload, "target_cluster"),
        input_location=folding_input_location_from_mapping(_required_mapping(payload, "input_location")),
        payload=folding_phase_plan_payload_from_mapping(_required_mapping(payload, "payload")),
    )


def folding_phase_runspec_from_mapping(payload: Mapping[str, object]) -> FoldingPhaseRunSpec:
    reject_environment_interpolation(payload, context="folding Phase RunSpec")
    _require_explicit_schema_versions(payload, path=("folding_phase_runspec",))
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "phase_kind",
            "phase_run_id",
            "attempt_id",
            "phase_plan_digest",
            "materialized_at",
            "input_location",
            "cluster",
            "payload",
            "carry_forward",
        },
        "FoldingPhaseRunSpec",
    )
    carry_mapping = payload.get("carry_forward")
    if "carry_forward" in payload and not isinstance(carry_mapping, Mapping):
        raise ValueError("carry_forward must be a mapping when present")
    return FoldingPhaseRunSpec(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="FoldingPhaseRunSpec"),
        phase_kind=_phase_kind(payload, "phase_kind", record_name="FoldingPhaseRunSpec"),
        phase_run_id=_required_str(payload, "phase_run_id"),
        attempt_id=_required_str(payload, "attempt_id"),
        phase_plan_digest=_required_str(payload, "phase_plan_digest"),
        materialized_at=_required_str(payload, "materialized_at"),
        input_location=folding_input_location_from_mapping(_required_mapping(payload, "input_location")),
        cluster=folding_resolved_cluster_snapshot_from_mapping(_required_mapping(payload, "cluster")),
        payload=folding_phase_runspec_payload_from_mapping(_required_mapping(payload, "payload")),
        carry_forward=(
            folding_carry_forward_reference_from_mapping(cast("Mapping[str, object]", carry_mapping))
            if carry_mapping is not None
            else None
        ),
    )


def _validate_compatibility_port_records(
    work_plan: PreprocessingWorkPlan,
    execution_plan: PreprocessingChunkExecutionIntent | PreprocessingChunkExecutionPlan,
) -> None:
    if not isinstance(work_plan, PreprocessingWorkPlan) or not isinstance(
        execution_plan, PreprocessingChunkExecutionIntent | PreprocessingChunkExecutionPlan
    ):
        msg = "preprocessing payloads must contain exact Compatibility Port contract records"
        raise ValueError(msg)
    if len(work_plan.tranches) != 1 or len(work_plan.chunks) != 1 or len(work_plan.assignments) != 1:
        msg = "the first preprocessing slice requires exactly one non-empty tranche, chunk, and assignment"
        raise ValueError(msg)
    if not work_plan.input.records:
        msg = "the first preprocessing slice rejects an empty work plan"
        raise ValueError(msg)
    chunk = work_plan.chunks[0]
    assignment = work_plan.assignments[0]
    if execution_plan.chunk_name != chunk.name or assignment.chunk_name != chunk.name:
        msg = "execution plan and assignment must reference the sole work-plan chunk"
        raise ValueError(msg)
    records = tuple(record for record in work_plan.input.records if record.source_ordinal in chunk.record_ordinals)
    for record in records:
        chains = record.sequence.split(":")
        if any(not chain for chain in chains):
            msg = "each searched record must have at least one non-empty chain"
            raise ValueError(msg)
    observed = tuple(
        (expected.source_ordinal, expected.record_identity, expected.source_header)
        for expected in execution_plan.expected_a3ms
    )
    expected = tuple((record.source_ordinal, record.identity, record.header) for record in records)
    if observed != expected or tuple(record.source_ordinal for record in records) != chunk.record_ordinals:
        msg = "execution-plan expected A3Ms must exactly preserve the chunk's source record identities"
        raise ValueError(msg)


def _validate_database_action_binding(
    database: PreprocessingDatabaseBinding,
    execution_plan: PreprocessingChunkExecutionPlan,
) -> None:
    if execution_plan.site.database_root != database.selected_container_root:
        raise ValueError("Runtime Action database root must equal the RunSpec selected database target")
    if (
        execution_plan.scientific.primary_database_name != database.primary_database_name
        or execution_plan.scientific.metagenomic_database_name != database.metagenomic_database_name
    ):
        raise ValueError("Runtime Action database names must equal the RunSpec manifest authority")
    expected_commands = (execution_plan.gpuserver_argv, execution_plan.search_argv)
    if any((branch.gpuserver_argv, branch.search_argv) != expected_commands for branch in database.branches):
        raise ValueError("Runtime Action commands must equal every RunSpec database branch binding")


def _validate_action_graph(actions: tuple[PreprocessingRuntimeAction, ...]) -> None:
    ids = tuple(action.action_id for action in actions)
    if len(set(ids)) != len(ids):
        msg = "Phase RunSpec Runtime Action ids must be unique"
        raise ValueError(msg)
    known = set(ids)
    dependencies = {action.action_id: action.dependencies for action in actions}
    for action in actions:
        if action.action_id in action.dependencies:
            msg = f"Runtime Action {action.action_id!r} cannot depend on itself"
            raise ValueError(msg)
        dangling = sorted(set(action.dependencies) - known)
        if dangling:
            msg = f"Runtime Action {action.action_id!r} has dangling dependencies: {', '.join(dangling)}"
            raise ValueError(msg)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(action_id: str) -> None:
        if action_id in visiting:
            msg = "Phase RunSpec Runtime Action graph contains a cycle"
            raise ValueError(msg)
        if action_id in visited:
            return
        visiting.add(action_id)
        for dependency in dependencies[action_id]:
            visit(dependency)
        visiting.remove(action_id)
        visited.add(action_id)

    for action_id in ids:
        visit(action_id)


def _validate_folding_action_graph(actions: tuple[FoldingRuntimeAction, ...]) -> None:
    ids = tuple(action.action_id for action in actions)
    if len(set(ids)) != len(ids):
        msg = "folding Phase RunSpec Runtime Action ids must be unique"
        raise ValueError(msg)
    known = set(ids)
    dependencies = {action.action_id: action.dependencies for action in actions}
    for action in actions:
        if action.action_id in action.dependencies:
            msg = f"folding Runtime Action {action.action_id!r} cannot depend on itself"
            raise ValueError(msg)
        dangling = sorted(set(action.dependencies) - known)
        if dangling:
            msg = f"folding Runtime Action {action.action_id!r} has dangling dependencies: {', '.join(dangling)}"
            raise ValueError(msg)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(action_id: str) -> None:
        if action_id in visiting:
            msg = "folding Phase RunSpec Runtime Action graph contains a cycle"
            raise ValueError(msg)
        if action_id in visited:
            return
        visiting.add(action_id)
        for dependency in dependencies[action_id]:
            visit(dependency)
        visiting.remove(action_id)
        visited.add(action_id)

    for action_id in ids:
        visit(action_id)


def _require_explicit_schema_versions(value: object, *, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        if path[-1] in {"database_set", "source_manifest"} or _is_postprocessing_semantic_value(path):
            return
        location = ".".join(path)
        if "schema_version" not in value:
            msg = f"missing explicit schema_version at {location}"
            raise ValueError(msg)
        if path[-1] == "scientific":
            validate_scientific_schema_version(value["schema_version"], record_name=location)
        else:
            validate_schema_version(value["schema_version"], record_name=location)
        for key, nested in value.items():
            if key != "schema_version":
                _require_explicit_schema_versions(nested, path=(*path, str(key)))
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            _require_explicit_schema_versions(nested, path=(*path, str(index)))


def _is_postprocessing_semantic_value(path: tuple[str, ...]) -> bool:
    """Treat typed semantic-field values as domain JSON, not nested contracts."""
    return path[-1] == "value" and any(component in {"semantic_fields", "normalized_arguments"} for component in path)


def _require_explicit_runspec_database_source_manifest_version(payload: Mapping[str, object]) -> None:
    """Require the manifest record version only at the strict current RunSpec boundary."""
    phase_payload = payload.get("payload")
    if not isinstance(phase_payload, Mapping):
        return
    database = phase_payload.get("database")
    if not isinstance(database, Mapping):
        return
    source_manifest = database.get("source_manifest")
    if not isinstance(source_manifest, Mapping):
        return
    if "database_source_manifest" in source_manifest:
        wrapped = source_manifest.get("database_source_manifest")
        if not isinstance(wrapped, Mapping):
            return
        manifest = wrapped
        path = "phase_runspec.payload.database.source_manifest.database_source_manifest"
    else:
        manifest = source_manifest
        path = "phase_runspec.payload.database.source_manifest"
    if "schema_version" not in manifest:
        raise ValueError(f"missing explicit schema_version at {path}")
    validate_schema_version(manifest["schema_version"], record_name=path)


def _validate_phase_run_id(value: str) -> None:
    if _PHASE_RUN_ID.fullmatch(value) is None:
        msg = "phase_run_id must be an opaque phase-run id"
        raise ValueError(msg)


def validate_phase_run_id(value: str) -> str:
    """Validate and return the shared opaque Phase Run identifier."""
    _validate_phase_run_id(value)
    return value


def _validate_attempt_id(value: str) -> None:
    if _ATTEMPT_ID.fullmatch(value) is None:
        msg = "attempt_id must use the attempt-NNNN form"
        raise ValueError(msg)


def validate_phase_attempt_id(value: str) -> str:
    """Validate and return the shared Phase Attempt identifier."""
    _validate_attempt_id(value)
    return value


def _validate_timestamp(value: str, record_name: str) -> None:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", value) is None:
        msg = f"{record_name} must be an explicit UTC timestamp"
        raise ValueError(msg)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        msg = f"{record_name} must be a valid UTC timestamp"
        raise ValueError(msg) from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        msg = f"{record_name} must be an explicit UTC timestamp"
        raise ValueError(msg)


def _validate_direct_schema_version(schema_version: int, record_name: str) -> None:
    validated = validate_schema_version(schema_version, record_name=record_name)
    if validated != schema_version:
        msg = f"{record_name} schema_version must be declared explicitly"
        raise ValueError(msg)


def _reject_unknown_fields(payload: Mapping[str, object], allowed: set[str], record_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        msg = f"Unknown {record_name} field(s): {', '.join(unknown)}"
        raise ValueError(msg)


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be a mapping"
        raise ValueError(msg)
    return value


def _required_mapping_sequence(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"{key}[{index}] must be a mapping"
            raise ValueError(msg)
        result.append(item)
    return tuple(result)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{key} must be a non-empty string"
        raise ValueError(msg)
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        msg = f"{key} must be null or a non-empty string"
        raise ValueError(msg)
    return value


@overload
def _optional_int(payload: Mapping[str, object], key: str) -> int | None: ...


@overload
def _optional_int(payload: Mapping[str, object], key: str, *, default: int) -> int: ...


def _optional_int(payload: Mapping[str, object], key: str, *, default: int | None = None) -> int | None:
    value = payload.get(key)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be null or an integer"
        raise ValueError(msg)
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


def _required_str_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple) or not all(isinstance(item, str) and item for item in value):
        msg = f"{key} must be a list of non-empty strings"
        raise ValueError(msg)
    return tuple(value)


def _required_str_pair_tuple(payload: Mapping[str, object], key: str) -> tuple[tuple[str, str], ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    result: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 2:
            msg = f"{key} must be a list of two-item string pairs"
            raise ValueError(msg)
        first, second = item[0], item[1]
        if not isinstance(first, str) or not first or not isinstance(second, str) or not second:
            msg = f"{key} must be a list of non-empty two-item string pairs"
            raise ValueError(msg)
        result.append((first, second))
    return tuple(result)


def _phase_kind(payload: Mapping[str, object], key: str, *, record_name: str) -> PhaseKind:
    value = _required_str(payload, key)
    if value not in {"preprocessing", "folding"}:
        msg = f"unsupported {record_name} phase_kind: {value!r}"
        raise ValueError(msg)
    return cast("PhaseKind", value)


def _preprocessing_phase_kind(payload: Mapping[str, object], key: str, *, record_name: str) -> PhaseKind:
    value = _phase_kind(payload, key, record_name=record_name)
    if value != "preprocessing":
        msg = f"unsupported {record_name} phase_kind: {value!r}"
        raise ValueError(msg)
    return value


def _seam_transport_policy(payload: Mapping[str, object], key: str) -> PhaseSeamTransportPolicy:
    value = payload.get(key, "publish-to-s3")
    if value not in get_args(PhaseSeamTransportPolicy):
        msg = f"unsupported phase seam transport policy: {value!r}"
        raise ValueError(msg)
    return cast("PhaseSeamTransportPolicy", value)


def _preprocessing_seam_transport_policy(payload: Mapping[str, object], key: str) -> PhaseSeamTransportPolicy:
    """Like ``_seam_transport_policy`` but defaults to ``"local"`` for preprocessing."""
    value = payload.get(key, "local")
    if value not in get_args(PhaseSeamTransportPolicy):
        msg = f"unsupported phase seam transport policy: {value!r}"
        raise ValueError(msg)
    return cast("PhaseSeamTransportPolicy", value)


def _validate_preprocessing_seam_transport(transport: PhaseSeamTransportPolicy, prefix: str | None) -> None:
    """Validate the preprocessing seam transport policy + S3 prefix pairing."""
    if transport not in get_args(PhaseSeamTransportPolicy):
        msg = f"unsupported phase seam transport policy: {transport!r}"
        raise ValueError(msg)
    if transport == "publish-to-s3":
        if prefix is None:
            msg = "transport 'publish-to-s3' requires a non-None s3_publish_prefix"
            raise ValueError(msg)
        stripped = prefix.rstrip("/")
        if not prefix.startswith("s3://") or stripped == "s3://":
            msg = "s3_publish_prefix must be a non-empty s3:// object key prefix"
            raise ValueError(msg)
    elif prefix is not None:
        msg = "s3_publish_prefix must be None when transport is not 'publish-to-s3'"
        raise ValueError(msg)


def _validate_folding_prediction_prefix(transport: PhaseSeamTransportPolicy, prefix: str | None) -> None:
    """Deferred enforcement for the folding prediction-bundle output prefix.

    If ``prefix`` is not None, it must be a non-empty ``s3://`` object key prefix.
    If ``transport == "local"`` and ``prefix`` is not None, raise (local forbids a prefix).
    If ``prefix`` is None, no validation is performed (legacy-compatible).
    """
    if transport not in get_args(PhaseSeamTransportPolicy):
        msg = f"unsupported phase seam transport policy: {transport!r}"
        raise ValueError(msg)
    if prefix is None:
        return
    stripped = prefix.rstrip("/")
    if not prefix.startswith("s3://") or stripped == "s3://":
        msg = "s3_prediction_prefix must be a non-empty s3:// object key prefix when present"
        raise ValueError(msg)
    if transport == "local":
        msg = "s3_prediction_prefix must be None when transport is 'local'"
        raise ValueError(msg)


__all__ = [
    "FOLDING_BACKENDS",
    "FoldShardProjectionBinding",
    "FoldingActionPayload",
    "FoldingPhasePlan",
    "FoldingPhasePlanPayload",
    "FoldingPhaseRunSpec",
    "FoldingPhaseRunSpecPayload",
    "FoldingResolvedClusterSnapshot",
    "FoldingRuntimeAction",
    "FoldingRuntimeActionKind",
    "InputLocationKind",
    "PhaseKind",
    "PhaseMountSnapshot",
    "PhasePlan",
    "PhaseRunSpec",
    "PhaseSeamTransport",
    "PhaseSeamTransportPolicy",
    "PhaseSlurmResources",
    "PreprocessingPhasePlanPayload",
    "PreprocessingPhaseRunSpecPayload",
    "PreprocessingRuntimeAction",
    "ResolvedClusterSnapshot",
    "RuntimeActionKind",
    "TransportKind",
    "VerifiedLocalInputLocation",
    "VerifiedRemoteInputLocation",
    "canonical_mapping_digest",
    "phase_plan_family_from_mapping",
    "phase_plan_from_mapping",
    "phase_runspec_family_from_mapping",
    "phase_runspec_from_mapping",
    "preprocessing_input_location_from_mapping",
    "validate_phase_attempt_id",
    "validate_phase_run_id",
    "verified_local_input_location_from_mapping",
    "verified_remote_input_location_from_mapping",
]
