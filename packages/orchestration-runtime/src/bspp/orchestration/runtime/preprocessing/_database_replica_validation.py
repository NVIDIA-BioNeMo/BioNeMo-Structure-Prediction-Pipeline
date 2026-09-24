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

"""Complete validation of one immutable Database Replica."""

from __future__ import annotations

import os
import stat
from collections import Counter
from pathlib import Path

from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaCopyEvidence,
    DatabaseReplicaManifest,
    database_replica_manifest_digest,
)
from bspp.orchestration.contract.database_set_provisioning import DatabaseSetIdentity

from ._database_placement_errors import DatabasePlacementError
from ._database_replica_copy import validate_replica_payload
from ._database_replica_evidence_io import load_database_replica_manifest_at
from ._filesystem_authority import FilesystemObjectIdentity, filesystem_object_identity


def validate_immutable_database_replica(
    root_descriptor: int,
    display_root: Path,
    *,
    database_set: DatabaseSetIdentity,
    source_manifest_sha256: str,
    expected_replica_manifest_sha256: str | None = None,
    expected_copy_evidence: DatabaseReplicaCopyEvidence | None = None,
    expected_device: int | None = None,
    expected_identity: FilesystemObjectIdentity | None = None,
) -> tuple[DatabaseReplicaManifest, str]:
    """Validate root authority, canonical manifest, and complete immutable payload."""
    root_info = os.fstat(root_descriptor)
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != os.geteuid()
        or stat.S_IMODE(root_info.st_mode) != 0o555
    ):
        raise DatabasePlacementError("Database Replica root must be effective-user-owned immutable 0555")
    if expected_device is not None and root_info.st_dev != expected_device:
        raise DatabasePlacementError("Database Replica root escaped the protected cache filesystem")
    if expected_identity is not None and filesystem_object_identity(root_info) != expected_identity:
        raise DatabasePlacementError("Database Replica root identity changed during validation")
    manifest = load_database_replica_manifest_at(root_descriptor, display_root)
    digest = database_replica_manifest_digest(manifest)
    if manifest.database_set != database_set or manifest.source_manifest_sha256 != source_manifest_sha256:
        raise DatabasePlacementError("Database Replica Manifest does not match Database Set authority")
    if expected_replica_manifest_sha256 is not None and digest != expected_replica_manifest_sha256:
        raise DatabasePlacementError("Database Replica Manifest digest changed across handoff")
    if expected_copy_evidence is not None and manifest.copy_evidence != expected_copy_evidence:
        raise DatabasePlacementError("Database Replica copy evidence changed across handoff")
    validate_replica_payload(root_descriptor, manifest.members, immutable=True)
    group_cardinality = Counter(member.hardlink_group for member in manifest.members)
    for member in manifest.members:
        try:
            member_info = os.stat(member.replica_path, dir_fd=root_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise DatabasePlacementError(f"Database Replica member is unavailable: {member.replica_path}") from exc
        if member_info.st_dev != root_info.st_dev:
            raise DatabasePlacementError(
                f"Database Replica member escaped the replica filesystem: {member.replica_path}"
            )
        if member_info.st_nlink != group_cardinality[member.hardlink_group]:
            raise DatabasePlacementError(
                f"Database Replica member has links outside its declared hard-link group: {member.replica_path}"
            )
    if filesystem_object_identity(os.fstat(root_descriptor)) != filesystem_object_identity(root_info):
        raise DatabasePlacementError("Database Replica root identity changed during validation")
    return manifest, digest


__all__ = ["validate_immutable_database_replica"]
