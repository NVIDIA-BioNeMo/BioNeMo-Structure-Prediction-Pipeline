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

"""Shared RunSpec authority for stage-required Database Replicas."""

from __future__ import annotations

import os
import pwd
from pathlib import PurePosixPath

from bspp.orchestration.contract.database_placement import (
    DATABASE_CACHE_ROOT,
    DATABASE_REPLICA_LEASE_TARGET,
    DATABASE_SOURCE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
    DatabaseProfileStagingSnapshot,
)
from bspp.orchestration.contract.phase import PhaseRunSpec

from ._database_placement_errors import DatabasePlacementError


def require_staged_authority(runspec: PhaseRunSpec) -> DatabaseProfileStagingSnapshot:
    """Prove exact staged or staged-with-fallback Runtime authority."""
    binding = runspec.payload.database
    if binding.requested_policy not in {
        DatabaseAccessPolicy.STAGE_REQUIRED,
        DatabaseAccessPolicy.STAGE_PREFERRED,
    }:
        raise DatabasePlacementError("Database Replica placement requires a staging-capable policy")
    if binding.staging is None:
        raise DatabasePlacementError("Database Replica placement requires a staging snapshot")
    if pwd.getpwuid(os.geteuid()).pw_name != binding.staging.unix_user:
        raise DatabasePlacementError("executing Unix user does not match Database Replica cache authority")
    expected_kinds = (
        ("staged",)
        if binding.requested_policy == DatabaseAccessPolicy.STAGE_REQUIRED
        else ("staged", "direct-capacity-fallback")
    )
    if tuple(branch.branch_kind for branch in binding.branches) != expected_kinds:
        raise DatabasePlacementError("staged Database Placement branch closure is invalid")
    branch = binding.branches[0]
    if branch.branch_kind != "staged" or branch.authorized_outcomes != (
        DatabasePlacementOutcomeKind.REPLICA_COLD,
        DatabasePlacementOutcomeKind.REPLICA_WARM,
    ):
        raise DatabasePlacementError("staged Database Placement branch authority is invalid")
    if len(branch.placement_mounts) != 2:
        raise DatabasePlacementError("staged Database Placement requires source and cache mounts")
    source_mount, cache_mount = branch.placement_mounts
    if (
        source_mount.source != binding.source_manifest.source_root
        or source_mount.target != DATABASE_SOURCE_ROOT
        or source_mount.purpose != "source"
        or not source_mount.read_only
        or cache_mount.source != binding.staging.user_cache_root
        or cache_mount.target != DATABASE_CACHE_ROOT
        or cache_mount.purpose != "cache"
        or cache_mount.read_only
    ):
        raise DatabasePlacementError("staged Database Placement mount authority is invalid")
    expected_replica = f"{binding.staging.user_cache_root}/replicas/{binding.source_manifest_sha256}"
    if len(branch.scientific_mounts) != 2:
        raise DatabasePlacementError("staged Database Placement requires selected replica and lease mounts")
    scientific_mount, lease_mount = branch.scientific_mounts
    expected_lease = f"{binding.staging.user_cache_root}/.locks/{binding.source_manifest_sha256}.lock"
    if (
        scientific_mount.source != expected_replica
        or scientific_mount.target != SELECTED_DATABASE_ROOT
        or scientific_mount.purpose != "selected-replica"
        or not scientific_mount.read_only
        or lease_mount.source != expected_lease
        or lease_mount.target != DATABASE_REPLICA_LEASE_TARGET
        or lease_mount.purpose != "replica-lease"
        or not lease_mount.read_only
        or branch.finalization_mounts
    ):
        raise DatabasePlacementError("staged selected replica authority is invalid")
    if binding.requested_policy == DatabaseAccessPolicy.STAGE_PREFERRED:
        fallback = binding.branches[1]
        if fallback.authorized_outcomes != (DatabasePlacementOutcomeKind.DIRECT_CAPACITY_FALLBACK,):
            raise DatabasePlacementError("stage-preferred fallback outcome authority is invalid")
        if len(fallback.placement_mounts) != 1 or not fallback.scientific_mounts:
            raise DatabasePlacementError("stage-preferred fallback requires source-only mount closures")
        fallback_placement = fallback.placement_mounts[0]
        if (
            fallback_placement != source_mount
            or any(
                mount.purpose != "selected-source"
                or mount.source_kind != "file"
                or not mount.read_only
                or PurePosixPath(mount.target).parent != PurePosixPath(SELECTED_DATABASE_ROOT)
                for mount in fallback.scientific_mounts
            )
            or fallback.finalization_mounts
            or fallback.gpuserver_argv != branch.gpuserver_argv
            or fallback.search_argv != branch.search_argv
        ):
            raise DatabasePlacementError("stage-preferred fallback mount or command authority is invalid")
    return binding.staging


def require_stage_required_authority(runspec: PhaseRunSpec) -> DatabaseProfileStagingSnapshot:
    """Retain the strict stage-required compatibility boundary."""
    if runspec.payload.database.requested_policy != DatabaseAccessPolicy.STAGE_REQUIRED:
        raise DatabasePlacementError("Database Replica placement requires stage-required policy")
    return require_staged_authority(runspec)


__all__ = ["require_stage_required_authority", "require_staged_authority"]
