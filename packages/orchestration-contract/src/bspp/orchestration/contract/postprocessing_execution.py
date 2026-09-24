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

"""Focused postprocessing contracts extracted from phase_postprocessing.py."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from bspp.orchestration.contract._postprocessing_validation import (
    _schema,
    _sha,
)
from bspp.orchestration.contract.phase import (
    PhaseMountSnapshot,
    TransportKind,
    canonical_mapping_digest,
    validate_phase_attempt_id,
    validate_phase_run_id,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION

ProjectionKind = Literal["legacy-runspec-yaml-v1"]


PostprocessingRuntimeSourceKind = Literal["baked", "override"]


PostprocessingRuntimeImagePolicy = Literal["digest-checked", "trusted-cache"]


_OUTPUT_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _credential_mount_source(value: str, label: str) -> None:
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path.anchor != "/"
        or ".." in path.parts
        or str(path) != value
        or any(character in value for character in ("\x00", "\n", "\r", ",", ":"))
    ):
        raise ValueError(f"{label} must be a canonical absolute POSIX mount source")


@dataclass(frozen=True)
class PostprocessingCredentialMountSnapshot:
    """Script-affecting credential source locators, never credential values."""

    aws_shared_credentials_file: str | None
    aws_config_file: str | None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if (self.aws_shared_credentials_file is None) != (self.aws_config_file is None):
            raise ValueError("postprocessing AWS credential mount locators must be both present or both null")
        if self.aws_shared_credentials_file is None:
            return
        assert self.aws_config_file is not None
        _credential_mount_source(
            self.aws_shared_credentials_file,
            "postprocessing AWS shared credentials file",
        )
        _credential_mount_source(self.aws_config_file, "postprocessing AWS config file")
        if self.aws_shared_credentials_file == self.aws_config_file:
            raise ValueError("postprocessing AWS credential mount locators must be distinct")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "aws_shared_credentials_file": self.aws_shared_credentials_file,
            "aws_config_file": self.aws_config_file,
        }


@dataclass(frozen=True)
class PostprocessingClusterSnapshot:
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
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
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
            raise ValueError("postprocessing cluster snapshot required selections must be non-empty")
        if self.transport not in {"ssh", "local-slurm"}:
            raise ValueError(f"unsupported postprocessing transport: {self.transport!r}")
        if (self.transport == "ssh") != (self.ssh_target is not None):
            raise ValueError("postprocessing ssh_target must match its transport")

    def to_mapping(self) -> dict[str, object]:
        return {
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
            "extra_mounts": [item.to_mapping() for item in self.extra_mounts],
        }


@dataclass(frozen=True)
class QualifiedPostprocessingRuntimeSelection:
    """Exact qualified runtime identities frozen into one postprocessing Attempt."""

    tuple_id: str
    qualification_location: str
    qualification_sha256: str
    qualification_size_bytes: int
    qualified_at: str
    expires_at: str
    image_path: str
    image_sha256: str
    image_size_bytes: int
    image_policy: PostprocessingRuntimeImagePolicy
    source_kind: PostprocessingRuntimeSourceKind
    source_revision: str
    source_package_path: str
    toolkit_package_path: str | None
    runtime_ipsae_binary_path: str
    runtime_ipsae_binary_sha256: str
    runtime_ipsae_binary_size_bytes: int
    source_identity_digest: str
    source_package_identity_digest: str
    toolkit_identity_digest: str
    bootstrap_sha256: str
    runtime_component_identity_digest: str
    requeue_exit: int | None = None
    max_batch_requeue: int | None = None
    selection_kind: str = "qualified-postprocessing-runtime-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.selection_kind != "qualified-postprocessing-runtime-v1":
            raise ValueError("unsupported qualified postprocessing runtime selection kind")
        for value in (
            self.tuple_id,
            self.qualification_sha256,
            self.image_sha256,
            self.runtime_ipsae_binary_sha256,
            self.source_identity_digest,
            self.source_package_identity_digest,
            self.toolkit_identity_digest,
            self.bootstrap_sha256,
            self.runtime_component_identity_digest,
        ):
            _sha(value, "qualified postprocessing runtime identity")
        if (
            not self.qualification_location
            or not self.qualified_at
            or not self.expires_at
            or not self.image_path.startswith("/")
            or not self.source_package_path.startswith("/")
            or not self.runtime_ipsae_binary_path.startswith("/")
        ):
            raise ValueError("qualified postprocessing runtime paths and timestamps must be non-empty")
        if self.source_kind not in {"baked", "override"} or self.image_policy not in {
            "digest-checked",
            "trusted-cache",
        }:
            raise ValueError("qualified postprocessing Runtime source or image policy is invalid")
        if (self.source_kind == "override") != (self.toolkit_package_path is not None):
            raise ValueError("qualified postprocessing Runtime toolkit package must match source kind")
        if self.toolkit_package_path is not None and not self.toolkit_package_path.startswith("/"):
            raise ValueError("qualified postprocessing Runtime toolkit package path must be absolute")
        if re.fullmatch(r"[0-9a-f]{40}", self.source_revision) is None:
            raise ValueError("qualified postprocessing Runtime source revision must be a full commit")
        if (
            self.qualification_size_bytes <= 0
            or self.image_size_bytes <= 0
            or self.runtime_ipsae_binary_size_bytes <= 0
        ):
            raise ValueError("qualified postprocessing Runtime record and artifact sizes must be positive")
        if (self.requeue_exit is None) != (self.max_batch_requeue is None):
            raise ValueError("qualified postprocessing Runtime requeue cap must be both present or both null")
        if self.requeue_exit is not None:
            assert self.max_batch_requeue is not None
            if isinstance(self.requeue_exit, bool) or not isinstance(self.requeue_exit, int) or self.requeue_exit != 85:
                raise ValueError("qualified postprocessing Runtime requeue_exit must equal 85")
            if (
                isinstance(self.max_batch_requeue, bool)
                or not isinstance(self.max_batch_requeue, int)
                or self.max_batch_requeue < 0
            ):
                raise ValueError("qualified postprocessing Runtime max_batch_requeue must be a non-negative integer")

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schema_version": self.schema_version,
            "selection_kind": self.selection_kind,
            "tuple_id": self.tuple_id,
            "qualification_location": self.qualification_location,
            "qualification_sha256": self.qualification_sha256,
            "qualification_size_bytes": self.qualification_size_bytes,
            "qualified_at": self.qualified_at,
            "expires_at": self.expires_at,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "image_size_bytes": self.image_size_bytes,
            "image_policy": self.image_policy,
            "source_kind": self.source_kind,
            "source_revision": self.source_revision,
            "source_package_path": self.source_package_path,
            "toolkit_package_path": self.toolkit_package_path,
            "runtime_ipsae_binary_path": self.runtime_ipsae_binary_path,
            "runtime_ipsae_binary_sha256": self.runtime_ipsae_binary_sha256,
            "runtime_ipsae_binary_size_bytes": self.runtime_ipsae_binary_size_bytes,
            "source_identity_digest": self.source_identity_digest,
            "source_package_identity_digest": self.source_package_identity_digest,
            "toolkit_identity_digest": self.toolkit_identity_digest,
            "bootstrap_sha256": self.bootstrap_sha256,
            "runtime_component_identity_digest": self.runtime_component_identity_digest,
        }
        if self.requeue_exit is not None:
            mapping["requeue_exit"] = self.requeue_exit
            mapping["max_batch_requeue"] = self.max_batch_requeue
        return mapping

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class PostprocessingPhaseExecutionIdentity:
    """Fully versioned phase-owned identity preimage for a legacy projection."""

    phase_plan_digest: str
    logical_input_manifest_digest: str
    scientific_identity_digest: str
    action_semantics_digest: str
    acceptance_semantic_digest: str
    phase_run_id: str
    attempt_id: str
    qualified_runtime_digest: str
    output_namespace: str
    substitutions: PostprocessingAttemptPaths
    identity_kind: str = "postprocessing-execution-projection-v2"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.identity_kind != "postprocessing-execution-projection-v2":
            raise ValueError("unsupported postprocessing projection identity kind")
        for value in (
            self.phase_plan_digest,
            self.logical_input_manifest_digest,
            self.scientific_identity_digest,
            self.action_semantics_digest,
            self.acceptance_semantic_digest,
            self.qualified_runtime_digest,
        ):
            _sha(value, "postprocessing projection identity digest")
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        if _OUTPUT_NAMESPACE.fullmatch(self.output_namespace) is None:
            raise ValueError("postprocessing projection output namespace is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity_kind": self.identity_kind,
            "phase_plan_digest": self.phase_plan_digest,
            "logical_input_manifest_digest": self.logical_input_manifest_digest,
            "scientific_identity_digest": self.scientific_identity_digest,
            "action_semantics_digest": self.action_semantics_digest,
            "acceptance_semantic_digest": self.acceptance_semantic_digest,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "qualified_runtime_digest": self.qualified_runtime_digest,
            "output_namespace": self.output_namespace,
            "substitutions": self.substitutions.to_mapping(),
        }

    @property
    def digest(self) -> str:
        return canonical_mapping_digest(self.to_mapping())


@dataclass(frozen=True)
class PostprocessingExecutionProjection:
    """Opaque boundary binding the exact once-rendered legacy RunSpec YAML."""

    document_location: str
    document_sha256: str
    document_size_bytes: int
    legacy_schema_version: int
    phase_identity: PostprocessingPhaseExecutionIdentity
    phase_identity_digest: str
    projection_kind: ProjectionKind = "legacy-runspec-yaml-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.projection_kind != "legacy-runspec-yaml-v1":
            raise ValueError("unsupported postprocessing execution projection kind")
        if not self.document_location.startswith("attempts/") or not self.document_location.endswith(
            "/legacy-runspec.yaml"
        ):
            raise ValueError("legacy execution projection must be attempt-relative")
        _sha(self.document_sha256, "legacy RunSpec document SHA-256")
        _sha(self.phase_identity_digest, "postprocessing phase identity digest")
        if self.phase_identity_digest != self.phase_identity.digest:
            raise ValueError("postprocessing projection identity digest does not match its mapping")
        if self.legacy_schema_version != CURRENT_CONTRACT_SCHEMA_VERSION:
            raise ValueError("legacy RunSpec projection must bind schema version 1")
        if self.document_size_bytes <= 0:
            raise ValueError("legacy RunSpec projection size must be positive")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "projection_kind": self.projection_kind,
            "legacy_schema_version": self.legacy_schema_version,
            "document_location": self.document_location,
            "document_sha256": self.document_sha256,
            "document_size_bytes": self.document_size_bytes,
            "phase_identity_digest": self.phase_identity_digest,
            "phase_identity": self.phase_identity.to_mapping(),
        }


@dataclass(frozen=True)
class PostprocessingAttemptPaths:
    legacy_run_id: str
    output_dir: str
    evidence_dir: str
    staging_dir: str
    object_prefix: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if any(
            not value
            for value in (self.legacy_run_id, self.output_dir, self.evidence_dir, self.staging_dir, self.object_prefix)
        ):
            raise ValueError("postprocessing attempt paths must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "legacy_run_id": self.legacy_run_id,
            "output_dir": self.output_dir,
            "evidence_dir": self.evidence_dir,
            "staging_dir": self.staging_dir,
            "object_prefix": self.object_prefix,
        }


__all__ = [
    "PostprocessingAttemptPaths",
    "PostprocessingClusterSnapshot",
    "PostprocessingCredentialMountSnapshot",
    "PostprocessingExecutionProjection",
    "PostprocessingPhaseExecutionIdentity",
    "PostprocessingRuntimeImagePolicy",
    "PostprocessingRuntimeSourceKind",
    "ProjectionKind",
    "QualifiedPostprocessingRuntimeSelection",
]
