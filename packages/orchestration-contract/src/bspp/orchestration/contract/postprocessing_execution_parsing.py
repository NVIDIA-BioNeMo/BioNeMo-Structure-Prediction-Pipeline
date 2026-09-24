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

"""Version-neutral parsers for postprocessing execution records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from bspp.orchestration.contract._postprocessing_validation import (
    _fields,
    _int,
    _mapping,
    _mapping_list,
    _str,
)
from bspp.orchestration.contract.phase import TransportKind, phase_mount_snapshot_from_mapping
from bspp.orchestration.contract.postprocessing_acceptance_reference import (
    PostprocessingAcceptanceSnapshotReference,
)
from bspp.orchestration.contract.postprocessing_execution import (
    PostprocessingAttemptPaths,
    PostprocessingClusterSnapshot,
    PostprocessingCredentialMountSnapshot,
    PostprocessingExecutionProjection,
    PostprocessingPhaseExecutionIdentity,
    PostprocessingRuntimeImagePolicy,
    PostprocessingRuntimeSourceKind,
    QualifiedPostprocessingRuntimeSelection,
)
from bspp.orchestration.contract.versioning import validate_schema_version


def postprocessing_credential_mount_snapshot_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingCredentialMountSnapshot:
    _fields(
        payload,
        {"schema_version", "aws_shared_credentials_file", "aws_config_file"},
        "PostprocessingCredentialMountSnapshot",
    )
    credentials = payload.get("aws_shared_credentials_file")
    config = payload.get("aws_config_file")
    if credentials is not None and not isinstance(credentials, str):
        raise ValueError("postprocessing aws_shared_credentials_file must be a string or null")
    if config is not None and not isinstance(config, str):
        raise ValueError("postprocessing aws_config_file must be a string or null")
    return PostprocessingCredentialMountSnapshot(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingCredentialMountSnapshot"
        ),
        aws_shared_credentials_file=credentials,
        aws_config_file=config,
    )


def postprocessing_cluster_snapshot_from_mapping(payload: Mapping[str, object]) -> PostprocessingClusterSnapshot:
    _fields(
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
        },
        "PostprocessingClusterSnapshot",
    )
    transport = _str(payload, "transport")
    if transport not in {"ssh", "local-slurm"}:
        raise ValueError(f"unsupported postprocessing transport: {transport!r}")
    ssh_target = payload.get("ssh_target")
    if ssh_target is not None and (not isinstance(ssh_target, str) or not ssh_target):
        raise ValueError("postprocessing ssh_target must be null or non-empty string")
    mounts = _mapping_list(payload, "extra_mounts")
    return PostprocessingClusterSnapshot(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingClusterSnapshot"
        ),
        profile_name=_str(payload, "profile_name"),
        owner=_str(payload, "owner"),
        transport=cast("TransportKind", transport),
        ssh_target=ssh_target,
        account=_str(payload, "account"),
        project_root=_str(payload, "project_root"),
        staging_root=_str(payload, "staging_root"),
        orchestration_repo=_str(payload, "orchestration_repo"),
        runtime_image=_str(payload, "runtime_image"),
        extra_mounts=tuple(phase_mount_snapshot_from_mapping(item) for item in mounts),
    )


def postprocessing_execution_projection_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingExecutionProjection:
    _fields(
        payload,
        {
            "schema_version",
            "projection_kind",
            "legacy_schema_version",
            "document_location",
            "document_sha256",
            "document_size_bytes",
            "phase_identity_digest",
            "phase_identity",
        },
        "PostprocessingExecutionProjection",
    )
    if _str(payload, "projection_kind") != "legacy-runspec-yaml-v1":
        raise ValueError("unsupported postprocessing execution projection kind")
    return PostprocessingExecutionProjection(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingExecutionProjection"
        ),
        legacy_schema_version=_int(payload, "legacy_schema_version"),
        document_location=_str(payload, "document_location"),
        document_sha256=_str(payload, "document_sha256"),
        document_size_bytes=_int(payload, "document_size_bytes"),
        phase_identity_digest=_str(payload, "phase_identity_digest"),
        phase_identity=postprocessing_phase_execution_identity_from_mapping(_mapping(payload, "phase_identity")),
    )


def postprocessing_phase_execution_identity_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingPhaseExecutionIdentity:
    _fields(
        payload,
        {
            "schema_version",
            "identity_kind",
            "phase_plan_digest",
            "logical_input_manifest_digest",
            "scientific_identity_digest",
            "action_semantics_digest",
            "acceptance_semantic_digest",
            "phase_run_id",
            "attempt_id",
            "qualified_runtime_digest",
            "output_namespace",
            "substitutions",
        },
        "PostprocessingPhaseExecutionIdentity",
    )
    if _str(payload, "identity_kind") != "postprocessing-execution-projection-v2":
        raise ValueError("unsupported postprocessing projection identity kind")
    return PostprocessingPhaseExecutionIdentity(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingPhaseExecutionIdentity"
        ),
        phase_plan_digest=_str(payload, "phase_plan_digest"),
        logical_input_manifest_digest=_str(payload, "logical_input_manifest_digest"),
        scientific_identity_digest=_str(payload, "scientific_identity_digest"),
        action_semantics_digest=_str(payload, "action_semantics_digest"),
        acceptance_semantic_digest=_str(payload, "acceptance_semantic_digest"),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        qualified_runtime_digest=_str(payload, "qualified_runtime_digest"),
        output_namespace=_str(payload, "output_namespace"),
        substitutions=postprocessing_attempt_paths_from_mapping(_mapping(payload, "substitutions")),
    )


def qualified_postprocessing_runtime_from_mapping(
    payload: Mapping[str, object],
) -> QualifiedPostprocessingRuntimeSelection:
    fields = {
        "schema_version",
        "selection_kind",
        "tuple_id",
        "qualification_location",
        "qualification_sha256",
        "qualification_size_bytes",
        "qualified_at",
        "expires_at",
        "image_path",
        "image_sha256",
        "image_size_bytes",
        "image_policy",
        "source_kind",
        "source_revision",
        "source_package_path",
        "toolkit_package_path",
        "runtime_ipsae_binary_path",
        "runtime_ipsae_binary_sha256",
        "runtime_ipsae_binary_size_bytes",
        "source_identity_digest",
        "source_package_identity_digest",
        "toolkit_identity_digest",
        "bootstrap_sha256",
        "runtime_component_identity_digest",
        "requeue_exit",
        "max_batch_requeue",
    }
    _fields(payload, fields, "QualifiedPostprocessingRuntimeSelection")
    if _str(payload, "selection_kind") != "qualified-postprocessing-runtime-v1":
        raise ValueError("unsupported qualified postprocessing runtime selection kind")
    source_kind = _str(payload, "source_kind")
    image_policy = _str(payload, "image_policy")
    toolkit_package_path = payload.get("toolkit_package_path")
    requeue_exit = payload.get("requeue_exit")
    max_batch_requeue = payload.get("max_batch_requeue")
    if requeue_exit is not None and (not isinstance(requeue_exit, int) or isinstance(requeue_exit, bool)):
        raise ValueError("qualified postprocessing Runtime requeue_exit must be an integer or null")
    if max_batch_requeue is not None and (
        not isinstance(max_batch_requeue, int) or isinstance(max_batch_requeue, bool)
    ):
        raise ValueError("qualified postprocessing Runtime max_batch_requeue must be an integer or null")
    if source_kind not in {"baked", "override"} or image_policy not in {"digest-checked", "trusted-cache"}:
        raise ValueError("qualified postprocessing Runtime source or image policy is invalid")
    if toolkit_package_path is not None and (not isinstance(toolkit_package_path, str) or not toolkit_package_path):
        raise ValueError("qualified postprocessing Runtime toolkit package path must be string or null")
    return QualifiedPostprocessingRuntimeSelection(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="QualifiedPostprocessingRuntimeSelection"
        ),
        tuple_id=_str(payload, "tuple_id"),
        qualification_location=_str(payload, "qualification_location"),
        qualification_sha256=_str(payload, "qualification_sha256"),
        qualification_size_bytes=_int(payload, "qualification_size_bytes"),
        qualified_at=_str(payload, "qualified_at"),
        expires_at=_str(payload, "expires_at"),
        image_path=_str(payload, "image_path"),
        image_sha256=_str(payload, "image_sha256"),
        image_size_bytes=_int(payload, "image_size_bytes"),
        image_policy=cast("PostprocessingRuntimeImagePolicy", image_policy),
        source_kind=cast("PostprocessingRuntimeSourceKind", source_kind),
        source_revision=_str(payload, "source_revision"),
        source_package_path=_str(payload, "source_package_path"),
        toolkit_package_path=toolkit_package_path,
        runtime_ipsae_binary_path=_str(payload, "runtime_ipsae_binary_path"),
        runtime_ipsae_binary_sha256=_str(payload, "runtime_ipsae_binary_sha256"),
        runtime_ipsae_binary_size_bytes=_int(payload, "runtime_ipsae_binary_size_bytes"),
        source_identity_digest=_str(payload, "source_identity_digest"),
        source_package_identity_digest=_str(payload, "source_package_identity_digest"),
        toolkit_identity_digest=_str(payload, "toolkit_identity_digest"),
        bootstrap_sha256=_str(payload, "bootstrap_sha256"),
        runtime_component_identity_digest=_str(payload, "runtime_component_identity_digest"),
        requeue_exit=requeue_exit,
        max_batch_requeue=max_batch_requeue,
    )


def postprocessing_acceptance_reference_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingAcceptanceSnapshotReference:
    _fields(
        payload,
        {"schema_version", "location", "sha256", "size_bytes", "semantic_digest", "policy_id"},
        "PostprocessingAcceptanceSnapshotReference",
    )
    return PostprocessingAcceptanceSnapshotReference(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingAcceptanceSnapshotReference"
        ),
        location=_str(payload, "location"),
        sha256=_str(payload, "sha256"),
        size_bytes=_int(payload, "size_bytes"),
        semantic_digest=_str(payload, "semantic_digest"),
        policy_id=_str(payload, "policy_id"),
    )


def postprocessing_attempt_paths_from_mapping(payload: Mapping[str, object]) -> PostprocessingAttemptPaths:
    _fields(
        payload,
        {"schema_version", "legacy_run_id", "output_dir", "evidence_dir", "staging_dir", "object_prefix"},
        "PostprocessingAttemptPaths",
    )
    return PostprocessingAttemptPaths(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PostprocessingAttemptPaths"),
        legacy_run_id=_str(payload, "legacy_run_id"),
        output_dir=_str(payload, "output_dir"),
        evidence_dir=_str(payload, "evidence_dir"),
        staging_dir=_str(payload, "staging_dir"),
        object_prefix=_str(payload, "object_prefix"),
    )


__all__ = [
    "postprocessing_acceptance_reference_from_mapping",
    "postprocessing_attempt_paths_from_mapping",
    "postprocessing_cluster_snapshot_from_mapping",
    "postprocessing_credential_mount_snapshot_from_mapping",
    "postprocessing_execution_projection_from_mapping",
    "postprocessing_phase_execution_identity_from_mapping",
    "qualified_postprocessing_runtime_from_mapping",
]
