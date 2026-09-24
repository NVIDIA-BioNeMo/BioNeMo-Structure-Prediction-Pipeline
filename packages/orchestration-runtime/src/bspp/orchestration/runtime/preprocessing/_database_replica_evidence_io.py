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

"""Strict Database Replica evidence loading and publication."""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaColdFailureEvidence,
    DatabaseReplicaColdResult,
    DatabaseReplicaManifest,
    DatabaseReplicaResult,
    DatabaseReplicaWarmResult,
    canonical_database_replica_cold_failure_evidence_bytes,
    canonical_database_replica_cold_result_bytes,
    canonical_database_replica_manifest_bytes,
    canonical_database_replica_result_bytes,
    canonical_database_replica_warm_result_bytes,
    database_replica_cold_failure_evidence_from_mapping,
    database_replica_cold_result_from_mapping,
    database_replica_manifest_from_mapping,
    database_replica_result_from_mapping,
    database_replica_warm_result_from_mapping,
)

from ._database_placement_evidence_io import (
    _load_canonical_json_at,
    _load_from_path,
    _publish_immutable_exclusive,
)


def load_database_replica_cold_result(path: Path) -> DatabaseReplicaColdResult:
    """Load exact canonical immutable cold Result bytes without widening direct I/O."""
    return _load_from_path(
        path,
        description="Database Replica cold Result",
        loader=_load_cold_result_at,
    )


def publish_database_replica_cold_result(result: DatabaseReplicaColdResult, destination: Path) -> None:
    """Exclusively publish and descriptor-verify one immutable cold Result."""
    _publish_immutable_exclusive(
        result,
        destination,
        serializer=canonical_database_replica_cold_result_bytes,
        loader=_load_cold_result_at,
        description="Database Replica cold Result",
    )


def load_database_replica_warm_result(path: Path) -> DatabaseReplicaWarmResult:
    """Load exact canonical immutable warm Result bytes."""
    return _load_from_path(
        path,
        description="Database Replica warm Result",
        loader=_load_warm_result_at,
    )


def publish_database_replica_warm_result(result: DatabaseReplicaWarmResult, destination: Path) -> None:
    """Exclusively publish and descriptor-verify one immutable warm Result."""
    _publish_immutable_exclusive(
        result,
        destination,
        serializer=canonical_database_replica_warm_result_bytes,
        loader=_load_warm_result_at,
        description="Database Replica warm Result",
    )


def load_database_replica_result(path: Path) -> DatabaseReplicaResult:
    """Load one exact canonical cold-or-warm Result envelope."""
    return _load_from_path(
        path,
        description="Database Replica Result",
        loader=_load_replica_result_at,
    )


def load_database_replica_cold_failure(path: Path) -> DatabaseReplicaColdFailureEvidence:
    """Load exact canonical immutable cold failure bytes."""
    return _load_from_path(
        path,
        description="Database Replica cold failure",
        loader=_load_cold_failure_at,
    )


def publish_database_replica_cold_failure(
    failure: DatabaseReplicaColdFailureEvidence,
    destination: Path,
) -> None:
    """Exclusively publish and descriptor-verify one immutable cold failure."""
    _publish_immutable_exclusive(
        failure,
        destination,
        serializer=canonical_database_replica_cold_failure_evidence_bytes,
        loader=_load_cold_failure_at,
        description="Database Replica cold failure",
    )


def load_database_replica_manifest(path: Path) -> DatabaseReplicaManifest:
    """Load exact canonical immutable Database Replica Manifest bytes."""
    return _load_from_path(
        path,
        description="Database Replica Manifest",
        loader=_load_replica_manifest_at,
    )


def load_database_replica_manifest_at(
    root_descriptor: int,
    display_root: Path,
) -> DatabaseReplicaManifest:
    """Strict-load the immutable manifest beneath an already-open replica root."""
    return _load_replica_manifest_at(
        root_descriptor,
        "replica-manifest.json",
        display_root / "replica-manifest.json",
    )


def _load_cold_result_at(parent_descriptor: int, name: str, display_path: Path) -> DatabaseReplicaColdResult:
    return _load_canonical_json_at(
        parent_descriptor,
        name,
        display_path,
        parser=database_replica_cold_result_from_mapping,
        serializer=canonical_database_replica_cold_result_bytes,
        description="Database Replica cold Result",
    )


def _load_warm_result_at(parent_descriptor: int, name: str, display_path: Path) -> DatabaseReplicaWarmResult:
    return _load_canonical_json_at(
        parent_descriptor,
        name,
        display_path,
        parser=database_replica_warm_result_from_mapping,
        serializer=canonical_database_replica_warm_result_bytes,
        description="Database Replica warm Result",
    )


def _load_replica_result_at(parent_descriptor: int, name: str, display_path: Path) -> DatabaseReplicaResult:
    return _load_canonical_json_at(
        parent_descriptor,
        name,
        display_path,
        parser=database_replica_result_from_mapping,
        serializer=canonical_database_replica_result_bytes,
        description="Database Replica Result",
    )


def _load_cold_failure_at(
    parent_descriptor: int,
    name: str,
    display_path: Path,
) -> DatabaseReplicaColdFailureEvidence:
    return _load_canonical_json_at(
        parent_descriptor,
        name,
        display_path,
        parser=database_replica_cold_failure_evidence_from_mapping,
        serializer=canonical_database_replica_cold_failure_evidence_bytes,
        description="Database Replica cold failure",
    )


def _load_replica_manifest_at(parent_descriptor: int, name: str, display_path: Path) -> DatabaseReplicaManifest:
    return _load_canonical_json_at(
        parent_descriptor,
        name,
        display_path,
        parser=database_replica_manifest_from_mapping,
        serializer=canonical_database_replica_manifest_bytes,
        description="Database Replica Manifest",
    )


__all__ = [
    "load_database_replica_cold_failure",
    "load_database_replica_cold_result",
    "load_database_replica_manifest",
    "load_database_replica_manifest_at",
    "load_database_replica_result",
    "load_database_replica_warm_result",
    "publish_database_replica_cold_failure",
    "publish_database_replica_cold_result",
    "publish_database_replica_warm_result",
]
